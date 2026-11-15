"""/api/studio/* routes (see team/studio/API.md). Drafts only: nothing here changes a live call until POST .../publish.
`register(api, require, data, runner=None)`; runner = the in-process sim runner from voice/server.py (needed for tests only)."""

from __future__ import annotations

import asyncio
import copy
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request

from . import recommend as R
from . import simtest as T
from . import store as S
from . import studio as D

_SAFE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,39}$")
SECTIONS = ("product", "tree", "kb", "strategy", "personas", "facts", "name")


def register(api: FastAPI, require: Callable[[Request], None], data: Path, runner=None, kb_root: Path | None = None,
             ask_factory: Callable[[], T.Ask] | None = None, llm_factory=None) -> dict:
    data = Path(data)
    jobs: dict[str, dict] = {}
    make_ask = ask_factory or T.default_ask

    def ids(t: str, p: str) -> None:
        if not _SAFE.match(t or "") or not _SAFE.match(p or ""):
            raise HTTPException(400, "tenant and product id: lower-case letters, digits, - and _")

    def get_draft(t: str, p: str, auto: bool = True) -> dict:
        ids(t, p)
        d = D.load_draft(data, t, p)
        if d is None and auto:
            try:
                d = D.import_live(data, t, p, kb_root)
            except KeyError:
                d = None
        if d is None:
            raise HTTPException(404, "no draft for this product (generate one, or import a live product)")
        return d

    def view(d: dict) -> dict:
        return {"draft": d, "readiness": D.readiness(d), "versions": S.list_versions(data, d["tenant"], d["product_id"]),
                "published": S.state(data, d["tenant"], d["product_id"]).get("published"), "recommendations": len(R.recommendations(d))}

    def job(kind: str, t: str, p: str, coro_fn) -> dict:
        j = {"id": uuid.uuid4().hex[:10], "kind": kind, "tenant": t, "product_id": p, "status": "running", "step": "starting", "error": "",
             "result": None, "created": time.time()}
        jobs[j["id"]] = j

        async def go():
            try:
                j["result"] = await coro_fn(j)
                j["status"] = "done"
            except Exception as e:  # noqa: BLE001
                j["status"], j["error"] = "failed", f"{type(e).__name__}: {str(e)[:300]}"
            j["finished"] = time.time()
        j["_task"] = asyncio.get_running_loop().create_task(go())
        return pub(j)

    def pub(j: dict) -> dict:
        return {k: v for k, v in j.items() if not k.startswith("_")}

    # ---------------------------------------------------------------- list / read / write
    @api.get("/api/studio")
    async def st_list(request: Request) -> dict:
        require(request)
        live = S._read(data / "products.json", {}) or {}
        out = []
        for pid, p in live.items():
            t = p.get("tenant") or "default"
            d = D.load_draft(data, t, pid) if _SAFE.match(t) and _SAFE.match(pid) else None
            r = D.readiness(d) if d else None
            out.append({"tenant": t, "product_id": pid, "name": p.get("product_name", pid), "live": True, "has_draft": bool(d),
                        "published": S.state(data, t, pid).get("published"), "ready": r["ready"] if r else None,
                        "open_facts": len(r["items"][0].get("items") or []) if r else None})
        root = S.root_dir(data)
        if root.exists():
            for tdir in root.iterdir():
                for pdir in (x for x in tdir.iterdir() if x.is_dir()) if tdir.is_dir() else []:
                    if pdir.name not in live and (pdir / "draft.json").exists():
                        d = D.load_draft(data, tdir.name, pdir.name)
                        r = D.readiness(d)
                        out.append({"tenant": tdir.name, "product_id": pdir.name, "name": d.get("name", pdir.name), "live": False, "has_draft": True,
                                    "published": None, "ready": r["ready"], "open_facts": len(r["items"][0].get("items") or [])})
        return {"products": out}

    @api.get("/api/studio/{tenant}/{pid}")
    async def st_get(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        return view(get_draft(tenant, pid))

    @api.post("/api/studio/{tenant}/{pid}/import")
    async def st_import(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        ids(tenant, pid)
        body = await _json(request)
        try:
            d = D.import_live(data, tenant, pid, kb_root, force=bool(body.get("force")))
        except KeyError as e:
            raise HTTPException(404, str(e))
        return view(d)

    @api.put("/api/studio/{tenant}/{pid}")
    async def st_put(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        d = get_draft(tenant, pid)
        body = await _json(request)
        for k in SECTIONS:
            if k in body:
                d[k] = body[k]
        if "tree" in body:
            from acie.playbook import Node
            try:
                for n in (d["tree"] or {}).get("nodes") or []:
                    Node.model_validate(n)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(400, f"tree: {str(e)[:300]}")
        D.save_draft(data, d)
        return view(d)

    # ---------------------------------------------------------------- facts
    def _fact(d: dict, fid: str) -> dict:
        f = next((x for x in d.get("facts") or [] if x["id"] == fid), None)
        if not f:
            raise HTTPException(404, "unknown fact")
        return f

    @api.post("/api/studio/{tenant}/{pid}/facts")
    async def st_fact_add(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        d = get_draft(tenant, pid)
        b = await _json(request)
        kind = str(b.get("kind") or "other")[:20]
        label = str(b.get("label") or kind.title())[:80]
        fid = "f-" + kind + "-" + uuid.uuid4().hex[:6]
        f = D.fact(fid, kind, label, "", b.get("value", ""), {"type": "user", "ref": str(b.get("source") or "typed in the studio")[:200]},
                   status="confirmed", critical=None)
        f["status"] = "confirmed"  # a person typed it
        d.setdefault("facts", []).append(f)
        D.save_draft(data, d)
        return {"fact": f, "readiness": D.readiness(d)}

    @api.put("/api/studio/{tenant}/{pid}/facts/{fid}")
    async def st_fact_put(tenant: str, pid: str, fid: str, request: Request) -> dict:
        require(request)
        d = get_draft(tenant, pid)
        f = _fact(d, fid)
        b = await _json(request)
        if "value" in b:
            f["value"] = b["value"]
            f["source"] = {"type": "user", "ref": "edited in the studio", "was": f.get("source")}
            if f.get("critical"):
                f["status"] = "needs_confirmation" if b.get("status") != "confirmed" else "confirmed"
        if b.get("status") in D.STATUSES:
            f["status"] = b["status"]
        for k in ("label", "kind"):
            if k in b:
                f[k] = str(b[k])[:80]
        f["updated_at"] = D.now()
        D.save_draft(data, d)
        return {"fact": f, "readiness": D.readiness(d)}

    @api.post("/api/studio/{tenant}/{pid}/facts/{fid}/confirm")
    async def st_fact_confirm(tenant: str, pid: str, fid: str, request: Request) -> dict:
        require(request)
        d = get_draft(tenant, pid)
        b = await _json(request)
        fids = [fid] if fid != "bulk" else [str(x) for x in b.get("ids") or []]
        out = []
        for x in fids:
            f = _fact(d, x)
            if fid != "bulk" and "value" in b:
                f["value"] = b["value"]
            f["status"] = "confirmed"
            f["confirmed_at"] = D.now()
            out.append(f)
        D.save_draft(data, d)
        return {"facts": out, "readiness": D.readiness(d)}

    # ---------------------------------------------------------------- versions / publish
    @api.get("/api/studio/{tenant}/{pid}/readiness")
    async def st_ready(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        return D.readiness(get_draft(tenant, pid))

    @api.get("/api/studio/{tenant}/{pid}/versions")
    async def st_versions(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        ids(tenant, pid)
        return {"versions": S.list_versions(data, tenant, pid), "published": S.state(data, tenant, pid).get("published")}

    @api.post("/api/studio/{tenant}/{pid}/versions")
    async def st_snapshot(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        d = get_draft(tenant, pid)
        v = D.save_version_from_draft(data, d, str((await _json(request)).get("note") or "")[:200])
        return {"version": v["version"], "versions": S.list_versions(data, tenant, pid)}

    @api.post("/api/studio/{tenant}/{pid}/publish")
    async def st_publish(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        get_draft(tenant, pid)
        try:
            out = D.publish(data, tenant, pid, kb_root)
        except D.NotReady as e:
            raise HTTPException(409, {"error": "not ready", "blocking": e.items})
        except S.PublishError as e:
            raise HTTPException(409, {"error": str(e)})
        return out

    # ---------------------------------------------------------------- generate
    @api.post("/api/studio/generate")
    async def st_generate(request: Request) -> dict:
        require(request)
        b = await _json(request)
        from .generate import generate
        from .llm import default_llm
        from .spec import Criteria, Economics, Price, Spec, slugify
        name = str(b.get("name") or "").strip()
        text = str(b.get("text") or "").strip()[:6000]
        tenant = str(b.get("tenant") or "default")
        if len(name) < 2:
            raise HTTPException(400, "give the product a name")
        pid = str(b.get("product_id") or slugify(name))
        ids(tenant, pid)
        num = lambda k: (float(b[k]) if b.get(k) not in (None, "") else None)  # noqa: E731
        try:
            spec = Spec(name=name, tenant=tenant, product_id=pid, seller=str(b.get("seller") or ""), notes=text, audience=str(b.get("audience") or "")[:300],
                        authorised=bool(b.get("authorised", True)),
                        price=Price(list_price=num("list_price"), offer_price=num("offer_price"), unit=str(b.get("unit") or ""), validity=str(b.get("validity") or "")),
                        economics=Economics(floor=num("floor")),
                        criteria=Criteria(min_topics=3, max_topics=12, min_total_topics=5, max_total_topics=24, min_entries=40, min_offerings=0,
                                          max_offerings=6, min_situations=10, max_situations=16, min_ladder_depth=3, max_repair_rounds=1))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)[:300])
        given = set()
        if spec.price.list_price is not None:
            given |= {"original_price", "offer_price"}
        if spec.price.offer_price is not None:
            given |= {"offer_price", "discount"}
        if spec.price.validity:
            given.add("offer_validity")
        if spec.economics.floor is not None:
            given.add("economics.cost")
        llm = (llm_factory or default_llm)()

        async def go(j):
            def prog(stage, status, detail=""):
                j["step"] = f"{stage}: {status}" + (f" ({detail})" if detail else "")
            v = await generate(spec, llm, data, prog)
            d = D.draft_from_version(data, tenant, pid, v, given, text)
            return {"tenant": tenant, "product_id": pid, "version": v.get("version"), "readiness": D.readiness(d)}
        return job("generate", tenant, pid, go)

    @api.get("/api/studio-jobs/{job_id}")
    async def st_job(job_id: str, request: Request) -> dict:
        require(request)
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        return pub(j)

    # ---------------------------------------------------------------- recommendations
    @api.get("/api/studio/{tenant}/{pid}/recommendations")
    async def st_recs(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        return {"recommendations": R.recommendations(get_draft(tenant, pid))}

    @api.post("/api/studio/{tenant}/{pid}/recommendations/add")
    async def st_rec_add(tenant: str, pid: str, request: Request) -> dict:
        """{id: "add:spouse"} or {description: "..."} -> a new situation node, added to the draft tree (returned so the UI can show it)."""
        require(request)
        d = get_draft(tenant, pid)
        b = await _json(request)
        rec = R.find(d, str(b.get("id") or "")) if b.get("id") else None
        desc = str(b.get("description") or (rec or {}).get("description") or "").strip()[:600]
        if not desc:
            raise HTTPException(400, "unknown recommendation; give a description")
        node = await _new_node(d, desc)
        nodes = d.setdefault("tree", {}).setdefault("nodes", [])
        taken = {n["id"] for n in D.walk(nodes)}
        while node["id"] in taken:
            node["id"] += "_" + uuid.uuid4().hex[:4]
        node["origin"] = "recommended" if rec else "ai"
        D.auto_cites([node], d.get("facts") or [])
        replace = (rec or {}).get("node") if (rec or {}).get("type") == "missing_hindi" else None
        if replace and (old := D.find_node(nodes, replace)):
            node["id"] = old["id"]
            node["children"] = old.get("children", [])
            old.clear()
            old.update(node)
        else:
            nodes.append(node)
        D.save_draft(data, d)
        return {"node": node, "readiness": D.readiness(d), "recommendations": R.recommendations(d)}

    async def _new_node(d: dict, desc: str) -> dict:
        from acie import playbook as pbm
        from acie.decision.domains.product_sales import Product
        try:
            prod = Product.model_validate(D.compile_product(d))
            pb = pbm.Playbook(tenant=d["tenant"], product_id=d["product_id"], nodes=(d.get("tree") or {}).get("nodes") or [])
            return (await pbm.generate_node(prod, pb, desc)).model_dump()
        except RuntimeError:  # planner endpoint not configured: use the drafting model
            fake = {"id": "new_case", "title": desc[:60], "when": {}, "points": [], "ask": "", "goal": "", "points_hi": [], "ask_hi": ""}
            tmp = copy.deepcopy(d)
            tmp.setdefault("tree", {}).setdefault("nodes", []).append(fake)
            r = await T.fix_node(make_ask(), tmp, "new_case", {"task": "write this new situation from scratch: " + desc})
            n = r["after"]
            n["id"] = re.sub(r"[^a-z0-9_]+", "_", n.get("title", "case").lower()).strip("_")[:30] or "case"
            return n

    @api.post("/api/studio/{tenant}/{pid}/nodes/{nid}/apply")
    async def st_apply(tenant: str, pid: str, nid: str, request: Request) -> dict:
        """Accept an AI proposal (fix / rewrite): {node: {...}} replaces that node's fields (children kept)."""
        require(request)
        d = get_draft(tenant, pid)
        n = D.find_node((d.get("tree") or {}).get("nodes") or [], nid)
        if not n:
            raise HTTPException(404, "unknown node")
        new = dict((await _json(request)).get("node") or {})
        new.pop("children", None)
        new["id"] = nid
        from acie.playbook import Node
        try:
            Node.model_validate({**n, **new})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)[:300])
        n.update(new)
        D.save_draft(data, d)
        return {"node": n, "readiness": D.readiness(d)}

    # ---------------------------------------------------------------- tests
    @api.post("/api/studio/{tenant}/{pid}/tests")
    async def st_tests(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        d = get_draft(tenant, pid)
        if runner is None:
            raise HTTPException(503, "the test runner is not available on this server")
        b = await _json(request)
        n = max(1, min(4, int(b.get("personas") or 4)))
        turns = max(2, min(6, int(b.get("max_turns") or 6)))
        if b.get("new_personas"):
            d["personas"] = []

        async def go(j):
            ask = make_ask()
            prod = dict(D.compile_product(d), id=pid, tenant=tenant)
            res = await T.run_tests(runner, ask, d, prod, n, turns, progress=lambda s: j.__setitem__("step", s))
            cur = D.load_draft(data, tenant, pid) or d
            cur["personas"] = res["personas"]
            cur["last_test"] = {"run_id": res["run_id"], "pass_rate": res["pass_rate"], "created_at": res["created_at"]}
            D.save_draft(data, cur)
            S._write(S.pdir(data, tenant, pid) / "tests" / f"run-{res['run_id']}.json", res)
            return res
        return job("tests", tenant, pid, go)

    @api.get("/api/studio/{tenant}/{pid}/tests")
    async def st_last_test(tenant: str, pid: str, request: Request) -> dict:
        require(request)
        d = get_draft(tenant, pid)
        lt = d.get("last_test") or {}
        run = S._read(S.pdir(data, tenant, pid) / "tests" / f"run-{lt.get('run_id')}.json") if lt else None
        return {"run": run}

    @api.post("/api/studio/{tenant}/{pid}/fix")
    async def st_fix(tenant: str, pid: str, request: Request) -> dict:
        """{node_id, persona_id?} -> {node_id, before, after, changes}. Nothing saved: accept with .../nodes/{id}/apply."""
        require(request)
        d = get_draft(tenant, pid)
        b = await _json(request)
        nid = str(b.get("node_id") or "")
        failure: dict[str, Any] = {}
        lt = d.get("last_test") or {}
        run = S._read(S.pdir(data, tenant, pid) / "tests" / f"run-{lt.get('run_id')}.json") if lt else None
        for r in (run or {}).get("results", []):
            if b.get("persona_id") and r["persona"].get("id") != b["persona_id"]:
                continue
            if nid in r.get("nodes", {}) and r["nodes"][nid]["verdict"] != "pass":
                failure = {"persona": r["persona"].get("brief"), "reasons": r["nodes"][nid]["reasons"],
                           "transcript": [{"who": t["role"], "text": t["text"]} for t in r["turns"]][-12:]}
                break
        if b.get("instruction"):
            failure["instruction"] = str(b["instruction"])[:400]
        try:
            return await T.fix_node(make_ask(), d, nid, failure)
        except KeyError as e:
            raise HTTPException(404, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"fix failed: {type(e).__name__}: {str(e)[:200]}")

    return jobs


async def _json(request: Request) -> dict:
    try:
        b = await request.json()
        return b if isinstance(b, dict) else {}
    except Exception:  # noqa: BLE001
        return {}
