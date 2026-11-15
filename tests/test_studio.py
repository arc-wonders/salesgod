"""Product studio: import -> facts gate -> confirm -> publish; recommendations; generate (fake LLM); tests with a fake runner; fix."""

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from acie.productgen import studio as D
from acie.productgen.recommend import recommendations
from acie.productgen.studio_api import register
from test_productgen_fake import FakeLLM

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "voice"
H = {"x-admin": "yes"}


class FakeRunner:
    def __init__(self, say_price="₹2,999"):
        self.say_price = say_price

    async def start(self, product, nodes, language):
        return ("h", 0), "Hi, I'm Arjun from FitZone."

    async def turn(self, handle, text):
        if "expensive" in text or "mehenga" in text:
            return {"agent": f"Abhi offer {self.say_price} ka hai.", "intent": "answer", "pb_node": "price_high", "end_call": False, "deal_numbers": []}
        if "send" in text:
            return {"agent": "Sending the link now.", "intent": "link_sent", "pb_node": "how_to_join", "end_call": True, "deal_numbers": []}
        return {"agent": "Trainers are included.", "intent": "answer", "pb_node": "why_join", "end_call": False, "deal_numbers": []}


async def fake_ask(system, payload):
    if "test customers" in system:
        return {"personas": [{"id": "haggler", "name": "Ravi", "language": "hinglish", "brief": "haggles", "hidden_objection": "price",
                              "target_nodes": ["price_high", "trust"]},
                             {"id": "eager", "name": "Asha", "language": "en", "brief": "wants to join", "hidden_objection": "", "target_nodes": ["how_to_join"]}]}
    if "play a customer" in system:
        n = sum(1 for t in payload["transcript"] if t["who"] == "customer")
        lines = ["why should I join", "too expensive yaar", "ok send it"] if payload["persona"]["id"] == "haggler" else ["ok send it"]
        return {"say": lines[min(n, len(lines) - 1)], "done": n >= len(lines) - 1}
    if "grade" in system:
        return {"overall": "pass", "reason": "fine", "nodes": {}}
    if "fix ONE" in system:
        node = dict(payload["node"])
        node["points"] = ["We have a 3-month plan.", "Trainers included.", "Ask about your goal."]
        return node
    return {}


@pytest.fixture
def app(tmp_path):
    if not (LIVE / "products.json").exists():
        pytest.skip("needs data/voice/products.json")
    shutil.copy(LIVE / "products.json", tmp_path / "products.json")
    kb = tmp_path / "kb"
    if (LIVE / "knowledge").exists():
        shutil.copytree(LIVE / "knowledge", kb)
    a = FastAPI()

    def require(request: Request):
        if request.headers.get("x-admin") != "yes":
            raise HTTPException(401, "sign in")
    a.state.runner = FakeRunner()
    register(a, require, tmp_path, runner=a.state.runner, kb_root=kb, ask_factory=lambda: fake_ask, llm_factory=lambda: FakeLLM())
    a.state.tmp = tmp_path
    return a


def wait(c, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/studio-jobs/{jid}", headers=H).json()
        if j["status"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_needs_admin(app):
    with TestClient(app) as c:
        assert c.get("/api/studio", headers={}).status_code == 401
        assert c.get("/api/studio/fitzone/fitzone-gym").status_code == 401


def test_import_gate_confirm_publish(app):
    tmp = app.state.tmp
    with TestClient(app) as c:
        lst = c.get("/api/studio", headers=H).json()["products"]
        assert {"fitzone-gym", "claude-pro"} <= {p["product_id"] for p in lst}
        v = c.get("/api/studio/fitzone/fitzone-gym", headers=H).json()
        d = v["draft"]
        prices = {f["field"]: f for f in d["facts"] if f["field"] in ("original_price", "offer_price", "offer_validity")}
        assert prices["offer_price"]["status"] == "needs_confirmation" and prices["offer_price"]["critical"]
        assert prices["offer_price"]["source"]["type"] == "import"
        assert v["published"] == 1 and not v["readiness"]["ready"]
        assert any(n.get("cites") for n in d["tree"]["nodes"])  # auto-linked
        r = c.post("/api/studio/fitzone/fitzone-gym/publish", headers=H)
        assert r.status_code == 409 and r.json()["detail"]["blocking"][0]["id"] == "facts"
        live_before = json.loads((tmp / "products.json").read_text())["fitzone-gym"]["offer_price"]
        # edit the offer, then confirm every open critical fact in one go
        r = c.put("/api/studio/fitzone/fitzone-gym/facts/f-offer-price", json={"value": "₹2,799 for 3 months"}, headers=H).json()
        assert r["fact"]["status"] == "needs_confirmation"
        assert json.loads((tmp / "products.json").read_text())["fitzone-gym"]["offer_price"] == live_before  # drafts never touch live
        open_ids = r["readiness"]["items"][0]["items"]
        r = c.post("/api/studio/fitzone/fitzone-gym/facts/bulk/confirm", json={"ids": open_ids}, headers=H).json()
        assert r["readiness"]["ready"]
        out = c.post("/api/studio/fitzone/fitzone-gym/publish", headers=H)
        assert out.status_code == 200, out.text
        assert out.json()["version"] == 2
        assert json.loads((tmp / "products.json").read_text())["fitzone-gym"]["offer_price"] == "₹2,799 for 3 months"
        assert (tmp / "playbooks" / "fitzone" / "fitzone-gym.json").exists()


def test_rejected_benefit_drops(app):
    with TestClient(app) as c:
        c.get("/api/studio/fitzone/fitzone-gym", headers=H)
        c.put("/api/studio/fitzone/fitzone-gym/facts/f-usp-0", json={"status": "rejected"}, headers=H)
        d = D.load_draft(app.state.tmp, "fitzone", "fitzone-gym")
        p = D.compile_product(d)
        assert len(p["benefits"]) == 2 and len(p["benefits_hi"]) == 2


def test_grounding_and_recommendations(app):
    with TestClient(app) as c:
        d = c.get("/api/studio/fitzone/fitzone-gym", headers=H).json()["draft"]
        d["tree"]["nodes"][0]["points"].append("Only ₹1,499 today!")
        c.put("/api/studio/fitzone/fitzone-gym", json={"tree": d["tree"]}, headers=H)
        recs = c.get("/api/studio/fitzone/fitzone-gym/recommendations", headers=H).json()["recommendations"]
        types = {r["type"] for r in recs}
        assert "ungrounded_number" in types and "missing_situation" in types
        assert any(r["id"] == "add:spouse" for r in recs)
        assert any(r["id"] == "add:already_have" for r in recs)


def test_recommend_add_uses_generator(app, monkeypatch):
    from acie import playbook as pbm

    async def gen(product, pb, desc, timeout_s=120.0):
        return pbm.Node(id="spouse_decides", title="Needs to ask spouse", points=["FitZone has trainers"], ask="Shall I call when you've talked?",
                        points_hi=["x"], ask_hi="y")
    monkeypatch.setattr(pbm, "generate_node", gen)
    with TestClient(app) as c:
        c.get("/api/studio/fitzone/fitzone-gym", headers=H)
        r = c.post("/api/studio/fitzone/fitzone-gym/recommendations/add", json={"id": "add:spouse"}, headers=H).json()
        assert r["node"]["id"] == "spouse_decides" and r["node"]["origin"] == "recommended"
        assert not any(x["id"] == "add:spouse" for x in r["recommendations"])


def test_tests_grid_and_fix(app):
    with TestClient(app) as c:
        c.get("/api/studio/fitzone/fitzone-gym", headers=H)
        c.post("/api/studio/fitzone/fitzone-gym/facts/bulk/confirm", json={"ids": ["f-offer-price"]}, headers=H)
        j = wait(c, c.post("/api/studio/fitzone/fitzone-gym/tests", json={"personas": 2, "max_turns": 4}, headers=H).json()["id"])
        assert j["status"] == "done", j
        res = j["result"]
        assert res["grid"]["price_high"]["haggler"] == "pass"  # 2999 is a confirmed fact
        assert res["grid"]["trust"]["haggler"] == "not_reached"
        eager = next(r for r in res["results"] if r["persona"]["id"] == "eager")
        assert eager["nodes"]["how_to_join"]["verdict"] == "fail"  # link with no consent turn
        app.state.runner.say_price = "₹1,999"
        j = wait(c, c.post("/api/studio/fitzone/fitzone-gym/tests", json={"personas": 2, "max_turns": 4}, headers=H).json()["id"])
        assert j["result"]["grid"]["price_high"]["haggler"] == "fail"
        assert c.get("/api/studio/fitzone/fitzone-gym", headers=H).json()["readiness"]["items"][-1]["ok"] is False  # warns only
        fx = c.post("/api/studio/fitzone/fitzone-gym/fix", json={"node_id": "price_high"}, headers=H).json()
        assert "points" in fx["changes"]
        r = c.post("/api/studio/fitzone/fitzone-gym/nodes/price_high/apply", json={"node": fx["after"]}, headers=H).json()
        assert r["node"]["points"][0].startswith("We have")


def test_generate_small_draft(app):
    with TestClient(app) as c:
        r = c.post("/api/studio/generate", json={"tenant": "zen", "name": "ZenBrew Coffee Subscription", "text": "Monthly coffee for offices in Pune.",
                                                 "list_price": 1200, "offer_price": 999, "unit": "per month"}, headers=H).json()
        j = wait(c, r["id"])
        assert j["status"] == "done", j
        d = c.get("/api/studio/zen/zenbrew-coffee-subscription", headers=H).json()["draft"]
        offer = next(f for f in d["facts"] if f["field"] == "offer_price")
        assert offer["status"] == "confirmed" and offer["source"]["type"] == "user"
        assert any(f["status"] == "ai_draft" for f in d["facts"])
        assert any(p["product_id"] == "zenbrew-coffee-subscription" for p in c.get("/api/studio", headers=H).json()["products"])
