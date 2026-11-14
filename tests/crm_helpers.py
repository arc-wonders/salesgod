import json
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from voice.crm.api import register

H = {"x-admin": "yes"}


class FakeDialer:
    """Records dials, never touches a network. `block` simulates the console's dial gate refusing."""

    def __init__(self, block: str | None = None):
        self.calls, self.block, self.n = [], block, 0

    async def __call__(self, to, product_id, language, customer_name, link):
        self.calls.append({"to": to, "product": product_id, "language": language, "link": link})
        if self.block:
            return {"error": self.block}
        self.n += 1
        return {"call_id": f"call{self.n:04d}", "status": "ringing", "error": None}


class FakeLLM:
    model = "fake"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def json(self, stage, payload):
        self.calls.append((stage, payload))
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r


class Clock:
    def __init__(self, t=1_700_000_000.0):  # a Tue 14 Nov 2023 22:13 UTC = Wed 03:43 IST
        self.t = t

    def __call__(self):
        return self.t


def make(tmp_path, dialer=None, llm=None, clock=None, active=None):
    app = FastAPI()

    def require(request: Request):
        if request.headers.get("x-admin") != "yes":
            raise HTTPException(401, "sign in")
    clock = clock or Clock()
    out = register(app, require, tmp_path, tmp_path / "calls", dialer=dialer, active=active,
                   llm_factory=(lambda: llm) if llm else None, clock=clock)
    (tmp_path / "calls").mkdir(exist_ok=True)
    return TestClient(app), out, clock


def project(c, pid="acme", **kw):
    r = c.post("/api/projects", json={"name": "Acme", "id": pid, **kw}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["project"]


def customer(c, pid="acme", phone="9876543210", **kw):
    r = c.post("/api/customers", json={"project_id": pid, "phone": phone, **kw}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["customer"]


def write_call(tmp_path, call_id, **kw):
    rec = {"call_id": call_id, "status": "ended", "answered_at": 1.0, "ended_at": 1_700_000_100.0, "turns": [], "context": {}, **kw}
    (tmp_path / "calls" / f"{call_id}.json").write_text(json.dumps(rec))
    return rec
