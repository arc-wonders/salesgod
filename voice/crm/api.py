"""CRM routes. One self-contained block: `register(api, require, DATA, CALLS_DIR, dialer=..., active=..., llm_factory=...)`.

`dialer` is `async (to, product_id, language, customer_name, link: dict) -> {"call_id", "status", "error"}` and must go through the
console's dial gate (allowed_numbers, DNC, call hours, max parallel). It is never called by anything except an explicit operator
call endpoint, or `tick` on a campaign that is running, in `auto` mode, confirmed by the operator AND with CRM_AUTO_DIAL=1.
"""

from __future__ import annotations

import hmac
import os
import secrets
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from voice.plivo import e164

from . import ai_assist, followups, importer
from . import queue as Q
from acie.calltypes.schema import validate_call_type
from .models import (BUILTIN_CALLTYPES, CT_ID_RE, Campaign, CallType, Customer, FieldDef, ID_RE, NextAction, Project, STAGES, ct_slug, slug, new_id)
from .scoring import score
from .store import JsonStore

Dialer = Callable[..., Awaitable[dict]]


def _422(e: ValidationError) -> HTTPException:
    return HTTPException(422, "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:6]))


async def _body(request: Request) -> dict:
    try:
        b = await request.json()
    except Exception:
        raise HTTPException(400, "send a JSON body")
    if not isinstance(b, dict):
        raise HTTPException(400, "send a JSON object")
    return b


def _pub_customer(c: Customer, full: bool = False) -> dict:
    d = c.model_dump()
    if not full:
        d.pop("events", None)
    return d


def _pub_project(p: Project) -> dict:
    return p.model_dump()


def register(api: FastAPI, require: Callable[[Request], None], data: Path, calls_dir: Path | None = None, *,
             dialer: Dialer | None = None, active: Callable[[], tuple[int, set[str]]] | None = None,
             llm_factory: Callable[[], Any] | None = None, clock: Callable[[], float] = time.time) -> dict:
    data = Path(data)
    calls_dir = Path(calls_dir) if calls_dir else data / "calls"
    store = JsonStore(data)

    def project_or_404(pid: str) -> Project:
        p = store.get_project(pid)
        if p is None:
            raise HTTPException(404, "unknown project")
        return p

    def customer_or_404(cid: str) -> Customer:
        c = store.get_customer(cid)
        if c is None:
            raise HTTPException(404, "unknown customer")
        return c

    def campaign_or_404(cid: str) -> Campaign:
        c = store.get_campaign(cid)
        if c is None:
            raise HTTPException(404, "unknown campaign")
        return c

    # ------------------------------------------------------------------ projects
    @api.get("/api/projects")
    async def list_projects(request: Request) -> dict:
        require(request)
        return {"projects": [_pub_project(p) for p in store.projects()]}

    @api.post("/api/projects")
    async def create_project(request: Request) -> dict:
        require(request)
        b = await _body(request)
        b.setdefault("id", slug(str(b.get("name", ""))))
        b.pop("webhook_token", None)
        try:
            p = Project(**b, webhook_token=secrets.token_urlsafe(24))
        except ValidationError as e:
            raise _422(e)
        if store.get_project(p.id):
            raise HTTPException(409, "a project with this id exists")
        store.put_project(p)
        return {"project": _pub_project(p)}

    @api.get("/api/projects/{pid}")
    async def get_project(pid: str, request: Request) -> dict:
        require(request)
        return {"project": _pub_project(project_or_404(pid))}

    @api.put("/api/projects/{pid}")
    async def put_project(pid: str, request: Request) -> dict:
        require(request)
        cur = store.get_project(pid)
        b = await _body(request)
        if cur is None:  # PUT creates too (the console's /api/tenants/{id} works the same way)
            b = {**b, "id": pid}
            b.pop("webhook_token", None)
            try:
                p = Project(**b, webhook_token=secrets.token_urlsafe(24))
            except ValidationError as e:
                raise _422(e)
        else:
            merged = {**cur.model_dump(), **{k: v for k, v in b.items() if k not in ("id", "webhook_token", "created_at", "fields")}}
            try:
                p = Project(**merged)
            except ValidationError as e:
                raise _422(e)
        store.put_project(p)
        return {"project": _pub_project(p)}

    @api.delete("/api/projects/{pid}")
    async def delete_project(pid: str, request: Request) -> dict:
        require(request)
        project_or_404(pid)
        if store.customers(pid) or store.campaigns(pid):
            raise HTTPException(409, "this project still has customers or campaigns; remove them first")
        store.delete_project(pid)
        return {"ok": True}

    @api.post("/api/projects/{pid}/webhook-token/rotate")
    async def rotate_token(pid: str, request: Request) -> dict:
        require(request)
        p = project_or_404(pid)
        p.webhook_token = secrets.token_urlsafe(24)
        store.put_project(p)
        return {"webhook_token": p.webhook_token}

    # custom field definitions
    @api.get("/api/projects/{pid}/fields")
    async def list_fields(pid: str, request: Request) -> dict:
        require(request)
        return {"fields": [f.model_dump() for f in project_or_404(pid).fields]}

    @api.post("/api/projects/{pid}/fields")
    async def add_field(pid: str, request: Request) -> dict:
        require(request)
        p = project_or_404(pid)
        try:
            f = FieldDef(**await _body(request))
        except ValidationError as e:
            raise _422(e)
        if any(x.key == f.key for x in p.fields):
            raise HTTPException(409, "a field with this key exists")
        p.fields.append(f)
        store.put_project(p)
        return {"field": f.model_dump()}

    @api.put("/api/projects/{pid}/fields/{key}")
    async def put_field(pid: str, key: str, request: Request) -> dict:
        require(request)
        p = project_or_404(pid)
        old = next((x for x in p.fields if x.key == key), None)
        if old is None:
            raise HTTPException(404, "unknown field")
        try:
            f = FieldDef(**{**old.model_dump(), **(await _body(request)), "key": key})
        except ValidationError as e:
            raise _422(e)
        p.fields = [f if x.key == key else x for x in p.fields]
        store.put_project(p)
        return {"field": f.model_dump()}

    @api.delete("/api/projects/{pid}/fields/{key}")
    async def del_field(pid: str, key: str, request: Request) -> dict:
        require(request)
        p = project_or_404(pid)
        p.fields = [x for x in p.fields if x.key != key]  # values already on customers are kept, just no longer defined
        store.put_project(p)
        return {"ok": True}

    # ------------------------------------------------------------------ call types (built-ins + custom per project)
    def _all_calltypes(project: str | None) -> list[CallType]:
        return list(BUILTIN_CALLTYPES) + store.calltypes(project) if project else list(BUILTIN_CALLTYPES) + store.calltypes()

    def _check_ct(ct: CallType) -> None:
        problems = validate_call_type(ct)
        if problems:
            raise HTTPException(422, "; ".join(problems[:6]))

    @api.get("/api/calltypes")
    async def list_calltypes(request: Request, project: str | None = None) -> dict:
        require(request)
        return {"calltypes": [c.model_dump() for c in _all_calltypes(project)]}

    @api.post("/api/calltypes")
    async def create_calltype(request: Request, project: str | None = None) -> dict:
        require(request)
        b = await _body(request)
        project = project or b.get("project_id")
        if not project:
            raise HTTPException(400, "project is required")
        project_or_404(project)
        b = {**b, "project_id": project, "builtin": False}
        b.setdefault("id", ct_slug(str(b.get("name", ""))))
        try:
            ct = CallType(**b)
        except ValidationError as e:
            raise _422(e)
        _check_ct(ct)
        if not CT_ID_RE.match(ct.id) or any(x.id == ct.id for x in BUILTIN_CALLTYPES) or any(x.id == ct.id for x in store.calltypes(project)):
            raise HTTPException(409, "call type id is taken or invalid")
        store.put_calltype(ct)
        return {"calltype": ct.model_dump()}

    @api.put("/api/calltypes/{ctid}")
    async def put_calltype(ctid: str, request: Request, project: str | None = None) -> dict:
        require(request)
        if any(x.id == ctid for x in BUILTIN_CALLTYPES):
            raise HTTPException(400, "built-in call types cannot be edited: duplicate it first")
        cur = next((x for x in store.calltypes(project) if x.id == ctid), None)
        if cur is None:
            raise HTTPException(404, "unknown call type")
        try:
            ct = CallType(**{**cur.model_dump(), **(await _body(request)), "id": ctid, "project_id": cur.project_id, "builtin": False})
        except ValidationError as e:
            raise _422(e)
        _check_ct(ct)
        store.put_calltype(ct)
        return {"calltype": ct.model_dump()}

    @api.delete("/api/calltypes/{ctid}")
    async def del_calltype(ctid: str, request: Request, project: str | None = None) -> dict:
        require(request)
        if any(x.id == ctid for x in BUILTIN_CALLTYPES):
            raise HTTPException(400, "built-in call types cannot be deleted")
        if any(c.call_type_id == ctid for c in store.campaigns()):
            raise HTTPException(409, "a campaign uses this call type")
        store.delete_calltype(ctid)
        return {"ok": True}

    @api.post("/api/calltypes/{ctid}/duplicate")
    async def dup_calltype(ctid: str, request: Request, project: str | None = None) -> dict:
        require(request)
        try:
            b = await request.json()
        except Exception:
            b = {}
        b = b if isinstance(b, dict) else {}
        project = project or b.get("project_id")
        src = next((x for x in _all_calltypes(project) if x.id == ctid), None)
        if src is None:
            raise HTTPException(404, "unknown call type")
        if not project:
            raise HTTPException(400, "project is required")
        project_or_404(project)
        name = str(b.get("name") or f"{src.name} (copy)")[:60]
        nid, i = ct_slug(name), 2
        taken = {x.id for x in _all_calltypes(project)}
        base = nid
        while nid in taken:
            nid, i = f"{base}_{i}", i + 1
        ct = CallType(**{**src.model_dump(), "id": nid, "name": name, "project_id": project, "builtin": False})
        store.put_calltype(ct)
        return {"calltype": ct.model_dump()}

    # ------------------------------------------------------------------ customers / leads
    def _check_custom(p: Project, custom: dict, creating: bool = False) -> dict:
        defs = {f.key: f for f in p.fields}
        out = {}
        for k, v in custom.items():
            f = defs.get(k)
            if f is None:
                raise HTTPException(400, f"unknown custom field {k!r}; define it under the project's fields first")
            try:
                if f.type == "number":
                    v = float(v)
                elif f.type == "bool":
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "y")
                elif f.type == "choice" and f.choices and v not in f.choices:
                    raise ValueError(f"must be one of {f.choices}")
                else:
                    v = str(v)[:500]
            except ValueError as e:
                raise HTTPException(400, f"custom field {k}: {e}")
            out[k] = v
        if creating:
            miss = [f.key for f in p.fields if f.required and f.key not in out]
            if miss:
                raise HTTPException(400, f"required custom fields missing: {miss}")
        return out

    def _log_stage(c: Customer, new: str, by: str = "operator") -> None:
        if new != c.stage:
            c.events.append({"t": clock(), "type": "stage", "from": c.stage, "to": new, "by": by})
            c.stage = new

    def _apply_dnc(c: Customer, why: str) -> None:
        store.add_dnc(c.phone)
        c.consent.dnc, c.consent.reason, c.consent.at = True, why, clock()
        c.next_action = None
        _log_stage(c, "do_not_call")

    @api.get("/api/customers")
    async def list_customers(request: Request, project: str, q: str = "", stage: str = "", tag: str = "", page: int = 1,
                             page_size: int = 50) -> dict:
        require(request)
        project_or_404(project)
        rows = store.customers(project)
        if stage:
            rows = [c for c in rows if c.stage == stage]
        if tag:
            rows = [c for c in rows if tag.lower() in c.tags]
        if q.strip():
            n = q.strip().lower()
            rows = [c for c in rows if n in f"{c.name} {c.phone} {c.email} {c.city}".lower()]
        rows.sort(key=lambda c: (-c.score, c.created_at, c.id))
        page_size = max(1, min(200, page_size))
        page = max(1, page)
        return {"customers": [_pub_customer(c) for c in rows[(page - 1) * page_size: page * page_size]], "total": len(rows),
                "page": page, "page_size": page_size, "stages": STAGES}

    @api.post("/api/customers")
    async def create_customer(request: Request) -> dict:
        require(request)
        b = await _body(request)
        p = project_or_404(str(b.get("project_id") or b.get("project") or ""))
        phone = e164(str(b.get("phone", "")))
        if not phone:
            raise HTTPException(400, "phone: use +91XXXXXXXXXX or a 10-digit Indian mobile")
        if store.find_by_phone(p.id, phone):
            raise HTTPException(409, "a customer with this phone exists in the project")
        try:
            c = Customer(project_id=p.id, phone=phone, name=str(b.get("name", ""))[:80], email=str(b.get("email", ""))[:120],
                         language=b.get("language") or (p.languages[0] if p.languages else "hi"), city=str(b.get("city", ""))[:60],
                         tags=sorted({str(t).strip().lower() for t in b.get("tags", []) if str(t).strip()}),
                         source="manual", notes=str(b.get("notes", ""))[:2000], created_at=clock(), updated_at=clock(),
                         custom=_check_custom(p, b.get("custom") or {}, creating=True))
        except ValidationError as e:
            raise _422(e)
        if phone in store.dnc():
            _apply_dnc(c, "on the do-not-call list")
        elif b.get("stage") in STAGES and b["stage"] != "do_not_call":
            c.stage = b["stage"]
        c.events.append({"t": clock(), "type": "created", "source": "manual"})
        c.score = score(c, clock())
        store.put_customer(c)
        return {"customer": _pub_customer(c)}

    @api.post("/api/customers/import")
    async def import_customers(request: Request) -> dict:
        require(request)
        b = await _body(request)
        p = project_or_404(str(b.get("project") or b.get("project_id") or ""))
        try:
            return importer.run(store, p, str(b.get("csv", "")), b.get("mapping") or None, commit=bool(b.get("commit")),
                                on_duplicate=str(b.get("on_duplicate", "skip")),
                                default_tags=[str(t).lower() for t in b.get("tags", [])], now=clock())
        except importer.ImportError_ as e:
            raise HTTPException(400, str(e))

    @api.get("/api/customers/{cid}")
    async def get_customer(cid: str, request: Request) -> dict:
        require(request)
        return {"customer": _pub_customer(customer_or_404(cid))}

    @api.put("/api/customers/{cid}")
    async def put_customer(cid: str, request: Request) -> dict:
        require(request)
        c = customer_or_404(cid)
        p = project_or_404(c.project_id)
        b = await _body(request)
        for f, n in (("name", 80), ("email", 120), ("city", 60), ("notes", 2000)):
            if f in b:
                setattr(c, f, str(b[f])[:n])
        if "language" in b:
            c.language = str(b["language"])
        if "tags" in b:
            c.tags = sorted({str(t).strip().lower() for t in b["tags"] if str(t).strip()})
        if "custom" in b:
            c.custom = {**c.custom, **_check_custom(p, b["custom"] or {})}
        if "phone" in b:
            ph = e164(str(b["phone"]))
            if not ph:
                raise HTTPException(400, "phone: not a valid number")
            other = store.find_by_phone(p.id, ph)
            if other and other.id != c.id:
                raise HTTPException(409, "another customer has this phone")
            c.phone = ph
        if "next_action" in b:
            try:
                c.next_action = NextAction(**b["next_action"]) if b["next_action"] else None
            except ValidationError as e:
                raise _422(e)
        cons = b.get("consent") or {}
        if cons.get("dnc") is True or b.get("stage") == "do_not_call":
            _apply_dnc(c, str(cons.get("reason") or "marked do-not-call by the operator")[:200])
        elif cons.get("dnc") is False and (c.consent.dnc or c.phone in store.dnc()):
            raise HTTPException(409, "do-not-call can only be lifted by the person asking us again, not from the CRM")
        elif "stage" in b:
            if b["stage"] not in STAGES:
                raise HTTPException(422, f"stage must be one of {STAGES}")
            if c.consent.dnc or c.phone in store.dnc():
                raise HTTPException(409, "this person is on the do-not-call list")
            _log_stage(c, b["stage"])
        c.updated_at = clock()
        c.score = score(c, clock())
        store.put_customer(c)
        return {"customer": _pub_customer(c)}

    @api.delete("/api/customers/{cid}")
    async def del_customer(cid: str, request: Request) -> dict:
        require(request)
        c = customer_or_404(cid)
        if c.consent.dnc or c.stage == "do_not_call":
            raise HTTPException(409, "do-not-call customers are kept on file so the number is never called again")
        store.delete_customer(cid)
        return {"ok": True}

    @api.post("/api/customers/{cid}/notes")
    async def add_note(cid: str, request: Request) -> dict:
        require(request)
        c = customer_or_404(cid)
        text = str((await _body(request)).get("text", "")).strip()[:2000]
        if not text:
            raise HTTPException(400, "write a note")
        c.events.append({"t": clock(), "type": "note", "text": text})
        c.updated_at = clock()
        store.put_customer(c)
        return {"ok": True}

    @api.get("/api/customers/{cid}/timeline")
    async def timeline(cid: str, request: Request) -> dict:
        require(request)
        c = customer_or_404(cid)
        out = []
        for i, e in enumerate(c.events):
            e = {**e, "_i": i}
            if e.get("type") in ("call", "call_started") and e.get("call_id"):
                e["map_url"] = f"/api/calls/{e['call_id']}/map"
            out.append(e)
        out.sort(key=lambda e: (e.get("t", 0), e["_i"]), reverse=True)  # newest first; same-second events keep their order
        for e in out:
            e.pop("_i")
        return {"customer_id": c.id, "timeline": out}

    # ------------------------------------------------------------------ lead webhook (token, no cookie)
    @api.post("/api/leads/webhook/{pid}")
    async def lead_webhook(pid: str, request: Request) -> dict:
        p = store.get_project(pid)
        tok = request.headers.get("x-webhook-token") or request.query_params.get("token") or ""
        if p is None or not p.webhook_token or not hmac.compare_digest(tok, p.webhook_token):
            raise HTTPException(401, "bad token")
        if int(request.headers.get("content-length") or 0) > 20_000:
            raise HTTPException(413, "payload too large")
        b = await _body(request)
        phone = e164(str(b.get("phone", "")))
        if not phone:
            raise HTTPException(400, "phone: not a valid number")
        existing = store.find_by_phone(p.id, phone)
        tags = sorted({str(t).strip().lower() for t in (b.get("tags") or []) if str(t).strip()} | {"webhook"})
        custom = {k: v for k, v in (b.get("custom") or {}).items() if k in {f.key for f in p.fields}}
        now = clock()
        if existing:
            c = existing
            for f in ("name", "email", "city"):
                if b.get(f) and not getattr(c, f):
                    setattr(c, f, str(b[f])[:120])
            c.tags = sorted(set(c.tags) | set(tags))
            c.custom = {**custom, **c.custom}
            c.events.append({"t": now, "type": "lead_again", "source": "webhook"})
            created = False
        else:
            lang = b.get("language") if b.get("language") in ("hi", "en", "hinglish") else (p.languages[0] if p.languages else "hi")
            c = Customer(project_id=p.id, phone=phone, name=str(b.get("name", ""))[:80], email=str(b.get("email", ""))[:120],
                         city=str(b.get("city", ""))[:60], language=lang, tags=tags, source="webhook", custom=custom,
                         notes=str(b.get("notes", ""))[:1000], created_at=now, updated_at=now)
            c.events.append({"t": now, "type": "created", "source": "webhook"})
            if phone in store.dnc():
                _apply_dnc(c, "on the do-not-call list")
            created = True
        c.updated_at = now
        c.score = score(c, now)
        store.put_customer(c)
        return {"ok": True, "customer_id": c.id, "created": created, "dnc": c.consent.dnc}

    # ------------------------------------------------------------------ campaigns
    def _campaign_checks(c: Campaign) -> None:
        project_or_404(c.project_id)
        if c.call_type_id not in {x.id for x in _all_calltypes(c.project_id)}:
            raise HTTPException(400, f"unknown call type {c.call_type_id!r}")

    def _auto_gate(c: Campaign, b: dict) -> None:
        if c.dial_mode == "auto":
            if b.get("confirm_auto") is True:
                c.auto_confirmed = True
            if not c.auto_confirmed:
                raise HTTPException(400, "auto dialing needs confirm_auto: true from the operator")
        else:
            c.auto_confirmed = False

    @api.get("/api/campaigns")
    async def list_campaigns(request: Request, project: str | None = None) -> dict:
        require(request)
        return {"campaigns": [c.model_dump() for c in store.campaigns(project)]}

    @api.post("/api/campaigns")
    async def create_campaign(request: Request) -> dict:
        require(request)
        b = await _body(request)
        b.pop("status", None)
        b.pop("stats", None)
        b.pop("auto_confirmed", None)
        confirm = b.pop("confirm_auto", None)
        b["project_id"] = b.get("project_id") or b.pop("project", "")
        try:
            c = Campaign(**b)
        except ValidationError as e:
            raise _422(e)
        _campaign_checks(c)
        _auto_gate(c, {"confirm_auto": confirm})
        store.put_campaign(c)
        return {"campaign": c.model_dump()}

    @api.get("/api/campaigns/{cid}")
    async def get_campaign(cid: str, request: Request) -> dict:
        require(request)
        return {"campaign": campaign_or_404(cid).model_dump()}

    @api.put("/api/campaigns/{cid}")
    async def put_campaign(cid: str, request: Request) -> dict:
        require(request)
        cur = campaign_or_404(cid)
        b = await _body(request)
        confirm = b.pop("confirm_auto", None)
        keep = {k: b[k] for k in b if k not in ("id", "project_id", "status", "stats", "auto_confirmed", "created_at")}
        try:
            c = Campaign(**{**cur.model_dump(), **keep})
        except ValidationError as e:
            raise _422(e)
        _campaign_checks(c)
        if c.dial_mode != cur.dial_mode or c.dial_mode == "auto":
            _auto_gate(c, {"confirm_auto": confirm} if c.dial_mode != cur.dial_mode else {})
        store.put_campaign(c)
        return {"campaign": c.model_dump()}

    @api.delete("/api/campaigns/{cid}")
    async def del_campaign(cid: str, request: Request) -> dict:
        require(request)
        c = campaign_or_404(cid)
        if c.status == "running":
            raise HTTPException(409, "pause the campaign first")
        store.delete_campaign(cid)
        return {"ok": True}

    @api.post("/api/campaigns/{cid}/start")
    async def start_campaign(cid: str, request: Request) -> dict:
        require(request)
        c = campaign_or_404(cid)
        if not c.product_id and c.call_type_id in ("sales_pitch", "follow_up", "lead_qualification", "payment_reminder"):
            raise HTTPException(400, "pick a product for this campaign before starting it")
        c.status = "running"
        store.put_campaign(c)
        return {"campaign": c.model_dump(), "note": "manual mode: nothing is dialed until you press Call on a customer"
                if c.dial_mode == "manual" else "auto mode: dials only through tick, inside the calling window"}

    @api.post("/api/campaigns/{cid}/pause")
    async def pause_campaign(cid: str, request: Request) -> dict:
        require(request)
        c = campaign_or_404(cid)
        c.status = "paused"
        store.put_campaign(c)
        return {"campaign": c.model_dump()}

    def _queue(c: Campaign) -> dict:
        n, ids = active() if active else (0, set())
        return Q.compute(store, c, clock(), ids, n)

    @api.get("/api/campaigns/{cid}/queue")
    async def campaign_queue(cid: str, request: Request) -> dict:
        require(request)
        return _queue(campaign_or_404(cid))

    async def _dial(c: Campaign, cust: Customer, call_type: str, language: str | None) -> dict:
        if dialer is None:
            raise HTTPException(503, "dialing is not available here")
        if not c.product_id:
            raise HTTPException(400, "this campaign has no product")
        if cust.consent.dnc or cust.stage == "do_not_call" or cust.phone in store.dnc():
            raise HTTPException(409, "on the do-not-call list: not calling")
        link = {"project_id": c.project_id, "customer_id": cust.id, "campaign_id": c.id, "call_type": call_type}
        res = await dialer(cust.phone, c.product_id, language or cust.language or c.language, cust.name, link)
        now = clock()
        if res.get("call_id"):
            store.link_call(res["call_id"], {**link, "started_at": now})
            cust.events.append({"t": now, "type": "call_started", "call_id": res["call_id"], "call_type": call_type, "campaign_id": c.id})
            cust.updated_at = now
            store.put_customer(cust)
            if c.id:  # a single call (no campaign) leaves no campaign stats behind
                c.stats["last_dial_at"] = int(now)
                c.stats["dialed"] = c.stats.get("dialed", 0) + 1
                store.put_campaign(c)
        return res

    @api.post("/api/campaigns/{cid}/call/{customer_id}")
    async def call_one(cid: str, customer_id: str, request: Request) -> dict:
        require(request)
        c = campaign_or_404(cid)
        cust = customer_or_404(customer_id)
        if cust.project_id != c.project_id:
            raise HTTPException(400, "this customer is in another project")
        try:
            b = await request.json()
        except Exception:
            b = {}
        b = b if isinstance(b, dict) else {}
        ct = b.get("call_type") or (cust.next_action.call_type if cust.next_action and cust.next_action.call_type else None) or c.call_type_id
        if ct not in {x.id for x in _all_calltypes(c.project_id)}:
            raise HTTPException(400, f"unknown call type {ct!r}")
        res = await _dial(c, cust, ct, b.get("language"))
        if res.get("error") and not res.get("call_id"):
            raise HTTPException(409, res["error"])
        return res

    @api.post("/api/customers/{cid}/call")
    async def call_customer(cid: str, request: Request) -> dict:
        """Operator-triggered single call without a campaign (the customer page's Call now). Same dial path and gates as a campaign
        call; the call is linked to the customer so its outcome updates the stage, score and next action."""
        require(request)
        cust = customer_or_404(cid)
        try:
            b = await request.json()
        except Exception:
            b = {}
        b = b if isinstance(b, dict) else {}
        pid = str(b.get("product_id") or "")
        if not pid:
            raise HTTPException(400, "product_id is required")
        ct = b.get("call_type") or (cust.next_action.call_type if cust.next_action and cust.next_action.call_type else None) or "sales_pitch"
        if ct not in {x.id for x in _all_calltypes(cust.project_id)}:
            raise HTTPException(400, f"unknown call type {ct!r}")
        stub = Campaign(id="", project_id=cust.project_id, product_id=pid, call_type_id=ct, name="single call", language=cust.language)
        res = await _dial(stub, cust, ct, b.get("language"))
        if res.get("error") and not res.get("call_id"):
            raise HTTPException(409, res["error"])
        return res

    @api.post("/api/campaigns/{cid}/tick")
    async def tick(cid: str, request: Request) -> dict:
        """Auto mode only: dial the next customers the queue allows. Needs status running, dial_mode auto, the operator's
        confirm_auto, and CRM_AUTO_DIAL=1 in the server environment. Otherwise it only reports the queue."""
        require(request)
        c = campaign_or_404(cid)
        q = _queue(c)
        armed = c.dial_mode == "auto" and c.auto_confirmed and c.status == "running" and os.getenv("CRM_AUTO_DIAL") == "1"
        dialed = []
        if armed:
            for row in q["next"]:
                cust = store.get_customer(row["customer_id"])
                if cust is None:
                    continue
                try:
                    r = await _dial(c, cust, row["call_type"], None)
                except HTTPException as e:
                    r = {"error": e.detail}
                dialed.append({"customer_id": cust.id, **{k: r.get(k) for k in ("call_id", "status", "error")}})
        return {"armed": armed, "dialed": dialed, "queue": q}

    # ------------------------------------------------------------------ follow-ups + outcomes
    @api.get("/api/followups")
    async def get_followups(request: Request, project: str, due: str = "now") -> dict:
        require(request)
        project_or_404(project)
        followups.sweep(store, calls_dir, clock())
        return {"followups": followups.due_list(store, project, due, clock()), "due": due}

    @api.post("/api/crm/calls/{call_id}/apply")
    async def apply_call(call_id: str, request: Request) -> dict:
        require(request)
        import json
        try:
            rec = json.loads((calls_dir / f"{Path(call_id).name}.json").read_text())
        except (OSError, ValueError):
            raise HTTPException(404, "unknown call")
        r = followups.apply_call(store, rec, clock())
        if r.get("applied"):
            store.link_call(call_id, {"applied": True})
        return r

    def on_call_end(rec: dict) -> dict:
        """Called by the server when a call finishes (never raises)."""
        try:
            r = followups.apply_call(store, rec, clock())
            if r.get("applied") and rec.get("call_id"):
                store.link_call(rec["call_id"], {"applied": True})
            return r
        except Exception as e:  # noqa: BLE001
            return {"applied": False, "reason": f"{type(e).__name__}: {e}"[:200]}

    # ------------------------------------------------------------------ AI assist (draft only)
    @api.post("/api/ai/assist")
    async def ai_assist_route(request: Request) -> dict:
        require(request)
        b = await _body(request)
        try:
            llm = (llm_factory or ai_assist.AssistLLM)()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"the AI service is not configured ({type(e).__name__})")
        try:
            return await ai_assist.assist(llm, str(b.get("kind", "")), str(b.get("prompt", "")), b.get("context") if isinstance(b.get("context"), dict) else {})
        except ai_assist.AssistError as e:
            raise HTTPException(e.status, e.msg)

    return {"store": store, "on_call_end": on_call_end}
