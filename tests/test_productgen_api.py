import asyncio
import json
import time

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from acie.productgen import llm as L
from acie.productgen.api import register
from test_productgen_fake import FakeLLM

PID = "zenbrew-coffee-subscription"
SPEC = {"name": "ZenBrew Coffee Subscription", "price": {"list_price": 1200, "offer_price": 999, "unit": "per month"}, "economics": {"floor": 600}}


def make(tmp_path, fake=None):
    app = FastAPI()

    def require(request: Request):
        if request.headers.get("x-admin") != "yes":
            raise HTTPException(401, "sign in")
    jobs = register(app, require, tmp_path, tmp_path / "calls", tmp_path / "kb", llm_factory=lambda: fake or FakeLLM())
    return app, jobs


H = {"x-admin": "yes"}


def wait(c, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/productgen/jobs/{job_id}", headers=H).json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_routes_need_admin(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        for m, u in [("get", "/api/productgen/defaults"), ("post", "/api/productgen/start"), ("get", f"/api/productgen/versions?product_id={PID}"),
                     ("post", "/api/productgen/publish"), ("post", "/api/productgen/expand"), ("get", f"/api/productgen/gaps?product_id={PID}"),
                     ("post", "/api/productgen/approve-gaps"), ("get", "/api/productgen/jobs")]:
            assert getattr(c, m)(u).status_code == 401, u


def test_full_flow_start_poll_versions_publish_expand_gaps(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        d = c.get("/api/productgen/defaults", headers=H).json()
        assert d["criteria"]["min_entries"] == 150 and d["llm_configured"]
        assert c.post("/api/productgen/start", json={"spec": {"name": "x"}}, headers=H).status_code == 422
        assert c.post("/api/productgen/start", json={"spec": {**SPEC, "price": {"list_price": 5, "offer_price": 9}}}, headers=H).status_code == 422

        jid = c.post("/api/productgen/start", json={"spec": SPEC}, headers=H).json()["job_id"]
        j = wait(c, jid)
        assert j["status"] == "done" and j["version"] == 1 and [s["status"] for s in j["stages"]][:3] == ["done", "done", "done"]
        assert j["stages"][3]["status"] == "done" and j["result"]["publishable"]
        assert c.get("/api/productgen/jobs", headers=H).json()["jobs"][0]["id"] == jid
        assert c.get("/api/productgen/jobs/nope", headers=H).status_code == 404

        prods = c.get("/api/productgen/products", headers=H).json()["products"]
        assert prods[0]["product_id"] == PID and prods[0]["published"] is None
        v = c.get(f"/api/productgen/versions?product_id={PID}", headers=H).json()
        assert [x["version"] for x in v["versions"]] == [1] and v["published"] is None
        one = c.get(f"/api/productgen/version?product_id={PID}&n=1", headers=H).json()
        assert one["version"]["bundle"]["kb"]["entries"] and one["version"]["report"]["checks"]
        assert c.get(f"/api/productgen/version?product_id={PID}&n=9", headers=H).status_code == 404

        pub = c.post("/api/productgen/publish", json={"product_id": PID, "version": 1}, headers=H)
        assert pub.status_code == 200 and pub.json()["ok"]
        assert (tmp_path / "products.json").exists() and (tmp_path / "kb/default" / f"{PID}.json").exists()
        assert c.get(f"/api/productgen/versions?product_id={PID}", headers=H).json()["published"] == 1

        # expand a node
        r = c.post("/api/productgen/expand", json={"product_id": PID, "node": {"kind": "playbook", "id": "sit-1"}}, headers=H)
        j2 = wait(c, r.json()["job_id"])
        assert j2["status"] == "done" and j2["version"] == 2
        assert c.post("/api/productgen/expand", json={"product_id": PID, "node": {"kind": "bad", "id": "x"}}, headers=H).status_code == 400
        df = c.get(f"/api/productgen/diff?product_id={PID}&a=1&b=2", headers=H).json()
        assert df["diff"]["nodes"]["added"] == ["sit-1-kid-0", "sit-1-kid-1", "sit-1-kid-2"]

        # gaps: nothing yet, then calls arrive
        g = c.get(f"/api/productgen/gaps?product_id={PID}", headers=H).json()
        assert g == {"unanswerable": [], "pending": []} or (g["unanswered"] == [] and g["pending"] == [])
        calls = tmp_path / "calls"
        calls.mkdir()
        for i, q in enumerate(["do you deliver to goa pincode", "can you deliver to goa"]):
            (calls / f"x{i}.json").write_text(json.dumps({"call_id": f"x{i}", "campaign": PID, "turns": [
                {"role": "customer", "text": q}, {"role": "agent", "text": "I don't have that detail", "intent": "price"}]}))
        g = c.get(f"/api/productgen/gaps?product_id={PID}", headers=H).json()
        assert len(g["unanswered"]) == 1 and g["unanswered"][0]["count"] == 2
        sj = wait(c, c.post("/api/productgen/gaps/scan", json={"product_id": PID}, headers=H).json()["job_id"])
        assert sj["status"] == "done" and sj["result"]["proposals"] == 1
        g = c.get(f"/api/productgen/gaps?product_id={PID}", headers=H).json()
        assert g["unanswered"] == [] and g["pending"][0]["proposals"][0]["status"] == "pending"
        pend_v, pid_ = g["pending"][0]["version"], g["pending"][0]["proposals"][0]["id"]
        ap = c.post("/api/productgen/approve-gaps", json={"product_id": PID, "version": pend_v, "approve": [pid_], "publish": True}, headers=H)
        assert ap.status_code == 200 and ap.json()["approved"] == [pid_] and ap.json()["published"]["ok"]
        assert c.post("/api/productgen/approve-gaps", json={"product_id": PID, "version": 1, "approve": []}, headers=H).status_code == 404


def test_publish_blocked_returns_409_and_bad_ids_400(tmp_path):
    app, _ = make(tmp_path, FakeLLM(flaws={"missing_hi"}, fixable=False))
    with TestClient(app) as c:
        j = wait(c, c.post("/api/productgen/start", json={"spec": SPEC}, headers=H).json()["job_id"])
        assert j["result"]["publishable"] is False
        r = c.post("/api/productgen/publish", json={"product_id": PID, "version": 1}, headers=H)
        assert r.status_code == 409 and "every entry" in r.json()["detail"].lower()
        assert c.get("/api/productgen/versions?product_id=../etc", headers=H).status_code == 400
        assert not (tmp_path / "products.json").exists()


def test_brand_check_and_missing_llm(tmp_path, monkeypatch):
    for k in ("PRODUCTGEN_BASE_URL", "PRODUCTGEN_MODEL", "ACIE_EXPR_BASE_URL", "ACIE_EXPR_MODEL"):
        monkeypatch.delenv(k, raising=False)
    app = FastAPI()
    register(app, lambda r: None, tmp_path)  # default LLM factory, nothing configured
    with TestClient(app) as c:
        b = c.post("/api/productgen/brand-check", json={"name": "Netflix Premium"}).json()
        assert b["real_brand"] and b["provenance"] == "public"
        assert c.post("/api/productgen/brand-check", json={"name": "Netflix Premium", "authorised": True}).json()["real_brand"] is False
        assert c.get("/api/productgen/defaults").json()["llm_configured"] is False
        r = c.post("/api/productgen/start", json={"spec": SPEC})
        assert r.status_code == 503 and "PRODUCTGEN_BASE_URL" in r.json()["detail"]
        assert c.get("/productgen").status_code == 200


def test_cancel_a_running_job(tmp_path):
    class Slow(FakeLLM):
        async def json(self, stage, payload):
            await asyncio.sleep(30)

    app, jobs = make(tmp_path, Slow())
    with TestClient(app) as c:
        jid = c.post("/api/productgen/start", json={"spec": SPEC}, headers=H).json()["job_id"]
        time.sleep(0.2)
        assert c.get(f"/api/productgen/jobs/{jid}", headers=H).json()["status"] == "running"
        c.post(f"/api/productgen/jobs/{jid}/cancel", headers=H)
        j = wait(c, jid)
        assert j["status"] == "cancelled" and all(s["status"] != "running" for s in j["stages"])
    assert not (tmp_path / "productgen").exists()  # nothing saved


# ---------------------------------------------------------------- the real client (mocked transport)
def test_llm_env_resolution_and_key_never_shown(monkeypatch):
    monkeypatch.delenv("PRODUCTGEN_BASE_URL", raising=False)
    monkeypatch.setenv("ACIE_EXPR_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("ACIE_EXPR_MODEL", "gemini-x")
    monkeypatch.setenv("ACIE_EXPR_API_KEY", "SECRET-KEY-123")
    monkeypatch.setenv("ACIE_EXPR_EXTRA_BODY", '{"reasoning_effort": "none"}')
    assert L.configured()
    c = L.OpenAICompatLLM()
    assert c.base == "https://example.test/v1" and c.model == "gemini-x" and c.extra == {"reasoning_effort": "none"}
    assert "SECRET" not in repr(c) and "SECRET" not in str(vars(c).get("extra"))
    monkeypatch.setenv("PRODUCTGEN_MODEL", "override-model")
    assert L.OpenAICompatLLM().model == "override-model"  # PRODUCTGEN_* wins


async def test_llm_client_request_shape_and_retries(monkeypatch):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append((req.headers["authorization"], body))
        if len(seen) == 1:
            return httpx.Response(503)
        if len(seen) == 2:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json at all"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "```json\n{\"ok\": 1}\n```"}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(L.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    real_sleep = asyncio.sleep
    monkeypatch.setattr(L.asyncio, "sleep", lambda s: real_sleep(0))
    c = L.OpenAICompatLLM("https://x.test/v1", "m", "KEY-1", {"foo": "bar"}, retries=3)
    out = await c.json("profile", {"spec": {"name": "Z"}})
    assert out == {"ok": 1} and len(seen) == 3
    auth, body = seen[0]
    assert auth == "Bearer KEY-1" and body["temperature"] == 0 and body["response_format"] == {"type": "json_object"} and body["foo"] == "bar"
    assert body["messages"][0]["role"] == "system" and '"stage": "profile"' in body["messages"][1]["content"]

    monkeypatch.setattr(L.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(401)), **kw))
    with pytest.raises(L.LLMError) as e:
        await c.json("profile", {})
    assert "KEY-1" not in str(e.value)
    monkeypatch.setattr(L.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(500)), **kw))
    c2 = L.OpenAICompatLLM("https://x.test/v1", "m", "KEY-1", {}, retries=1)
    with pytest.raises(L.LLMError):
        await c2.json("profile", {})


def test_parse_json():
    assert L.parse_json('noise {"a": [1]} tail') == {"a": [1]}
    with pytest.raises(ValueError):
        L.parse_json("[1]")
