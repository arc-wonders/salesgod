"""Store protocol + JSON-file implementation (atomic writes). Postgres can implement `Store` later."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Protocol

from .models import Campaign, CallType, Customer, Project


class Store(Protocol):
    def projects(self) -> list[Project]: ...
    def get_project(self, pid: str) -> Project | None: ...
    def put_project(self, p: Project) -> None: ...
    def delete_project(self, pid: str) -> None: ...
    def customers(self, project_id: str) -> list[Customer]: ...
    def get_customer(self, cid: str) -> Customer | None: ...
    def put_customer(self, c: Customer) -> None: ...
    def delete_customer(self, cid: str) -> None: ...
    def find_by_phone(self, project_id: str, phone: str) -> Customer | None: ...
    def campaigns(self, project_id: str | None = None) -> list[Campaign]: ...
    def get_campaign(self, cid: str) -> Campaign | None: ...
    def put_campaign(self, c: Campaign) -> None: ...
    def delete_campaign(self, cid: str) -> None: ...
    def calltypes(self, project_id: str | None = None) -> list[CallType]: ...
    def put_calltype(self, c: CallType) -> None: ...
    def delete_calltype(self, cid: str) -> None: ...
    def link_call(self, call_id: str, link: dict) -> None: ...
    def call_link(self, call_id: str) -> dict | None: ...
    def dnc(self) -> set[str]: ...
    def add_dnc(self, phone: str) -> None: ...


def atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:  # a corrupt file must not take the console down: keep it aside, start empty
        try:
            os.replace(path, path.with_name(path.name + ".corrupt"))
        except OSError:
            pass
        return default


class JsonStore:
    def __init__(self, data: Path):
        self.data = Path(data)
        self.lock = threading.RLock()
        self._cust_index: dict[str, str] = {}  # customer id -> project id (rebuilt lazily)

    # -- projects (projects.json; tenants.json is kept in sync so /api/tenants keeps working)
    def _tenants(self) -> list[dict]:
        return read_json(self.data / "tenants.json", [])

    def projects(self) -> list[Project]:
        with self.lock:
            raw = read_json(self.data / "projects.json", None)
            have = {p["id"]: p for p in (raw or [])}
            changed = raw is None
            for t in self._tenants():  # derive projects from tenants not yet migrated
                if t.get("id") and t["id"] not in have:
                    try:
                        p = Project(id=t["id"], name=t.get("name", t["id"]), color=t.get("color", "#888888"))
                    except ValueError:
                        continue
                    have[p.id] = p.model_dump()
                    changed = True
            if changed and have:
                atomic_write(self.data / "projects.json", list(have.values()))
            return [Project(**v) for v in have.values()]

    def get_project(self, pid: str) -> Project | None:
        return next((p for p in self.projects() if p.id == pid), None)

    def put_project(self, p: Project) -> None:
        with self.lock:
            ps = [x for x in self.projects() if x.id != p.id] + [p]
            atomic_write(self.data / "projects.json", [x.model_dump() for x in ps])
            ts = [t for t in self._tenants() if t.get("id") != p.id] + [{"id": p.id, "name": p.name, "color": p.color}]
            atomic_write(self.data / "tenants.json", ts)

    def delete_project(self, pid: str) -> None:
        with self.lock:
            atomic_write(self.data / "projects.json", [x.model_dump() for x in self.projects() if x.id != pid])
            atomic_write(self.data / "tenants.json", [t for t in self._tenants() if t.get("id") != pid])

    # -- customers: customers/<project>.json = {id: customer}
    def _cpath(self, project_id: str) -> Path:
        return self.data / "customers" / f"{project_id}.json"

    def customers(self, project_id: str) -> list[Customer]:
        return [Customer(**v) for v in read_json(self._cpath(project_id), {}).values()]

    def get_customer(self, cid: str) -> Customer | None:
        with self.lock:
            pid = self._cust_index.get(cid)
            for p in ([pid] if pid else []) + [x.id for x in self.projects()]:
                d = read_json(self._cpath(p), {})
                if cid in d:
                    self._cust_index[cid] = p
                    return Customer(**d[cid])
        return None

    def put_customer(self, c: Customer) -> None:
        with self.lock:
            d = read_json(self._cpath(c.project_id), {})
            d[c.id] = c.model_dump()
            atomic_write(self._cpath(c.project_id), d)
            self._cust_index[c.id] = c.project_id

    def put_customers(self, cs: list[Customer]) -> None:  # bulk (import commit): one write per project
        with self.lock:
            by: dict[str, dict] = {}
            for c in cs:
                by.setdefault(c.project_id, read_json(self._cpath(c.project_id), {}))[c.id] = c.model_dump()
                self._cust_index[c.id] = c.project_id
            for p, d in by.items():
                atomic_write(self._cpath(p), d)

    def delete_customer(self, cid: str) -> None:
        with self.lock:
            c = self.get_customer(cid)
            if c:
                d = read_json(self._cpath(c.project_id), {})
                d.pop(cid, None)
                atomic_write(self._cpath(c.project_id), d)
                self._cust_index.pop(cid, None)

    def find_by_phone(self, project_id: str, phone: str) -> Customer | None:
        return next((c for c in self.customers(project_id) if c.phone == phone), None)

    # -- campaigns / call types
    def campaigns(self, project_id: str | None = None) -> list[Campaign]:
        cs = [Campaign(**v) for v in read_json(self.data / "campaigns.json", [])]
        return [c for c in cs if project_id is None or c.project_id == project_id]

    def get_campaign(self, cid: str) -> Campaign | None:
        return next((c for c in self.campaigns() if c.id == cid), None)

    def put_campaign(self, c: Campaign) -> None:
        with self.lock:
            cs = [x for x in self.campaigns() if x.id != c.id] + [c]
            atomic_write(self.data / "campaigns.json", [x.model_dump() for x in cs])

    def delete_campaign(self, cid: str) -> None:
        with self.lock:
            atomic_write(self.data / "campaigns.json", [x.model_dump() for x in self.campaigns() if x.id != cid])

    def calltypes(self, project_id: str | None = None) -> list[CallType]:  # custom ones only; built-ins come from models
        cs = [CallType(**v) for v in read_json(self.data / "calltypes.json", [])]
        return [c for c in cs if project_id is None or c.project_id == project_id]

    def put_calltype(self, c: CallType) -> None:
        with self.lock:
            cs = [x for x in self.calltypes() if not (x.id == c.id and x.project_id == c.project_id)] + [c]
            atomic_write(self.data / "calltypes.json", [x.model_dump() for x in cs])

    def delete_calltype(self, cid: str) -> None:
        with self.lock:
            atomic_write(self.data / "calltypes.json", [x.model_dump() for x in self.calltypes() if x.id != cid])

    # -- call links + do-not-call (dnc.json has the same format the console writes: {number: epoch})
    def link_call(self, call_id: str, link: dict) -> None:
        with self.lock:
            d = read_json(self.data / "crm_calls.json", {})
            d[call_id] = {**d.get(call_id, {}), **link}
            atomic_write(self.data / "crm_calls.json", d)

    def call_link(self, call_id: str) -> dict | None:
        return read_json(self.data / "crm_calls.json", {}).get(call_id)

    def all_call_links(self) -> dict:
        return read_json(self.data / "crm_calls.json", {})

    def dnc(self) -> set[str]:
        return set(read_json(self.data / "dnc.json", {}))

    def add_dnc(self, phone: str) -> None:
        import time
        with self.lock:
            d = read_json(self.data / "dnc.json", {})
            d.setdefault(phone, time.time())
            atomic_write(self.data / "dnc.json", d)
