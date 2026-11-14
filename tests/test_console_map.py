"""Live information map (voice/live_map.py): snapshot with and without the KB, the map endpoint, the economics checks."""
import json
import os
import time
from pathlib import Path

import pytest

from acie.decision.domains.product_sales import Economics, Product, Step
from acie.schemas import SessionState, Utterance
from voice import live_map

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "data/voice/products.json").read_text(encoding="utf-8"))
FZ = Product.model_validate(P["fitzone-gym"])
CP = Product.model_validate(P["claude-pro"])


def session(*pairs, dialog=None):
    s = SessionState(session_id="s", customer_id="c")
    for role, text in pairs:
        s.history.append(Utterance(role=role, text=text))
    s.dialog.update(dialog or {})
    return s


@pytest.fixture
def kb_on(monkeypatch):
    monkeypatch.setenv("ACIE_KB", "1")


def test_snapshot_without_kb(monkeypatch):
    monkeypatch.delenv("ACIE_KB", raising=False)
    s = session(("agent", "Hello"), ("customer", "what do you have for powerlifting?"), ("agent", "We have a program."),
                dialog={"goal": "get stronger", "_signals": json.dumps({"intent": "benefits", "objection": "price", "buying_interest": 0.2})})
    s.context.objections.append("price")
    m = live_map.snapshot(s, FZ, "c1")
    assert m["kb"] is False and m["tree"] == {"topics": [], "offerings": []}
    assert m["stage"] in live_map.STAGES and m["path"][-1]["n"] == 2
    assert m["facts"]["goal"] == "get stronger" and m["facts"]["objections"] == ["price"]
    json.dumps(m)  # plain JSON


def test_snapshot_with_kb_colours_nodes(kb_on):
    s = session(("agent", "Hello"), ("customer", "do you have powerlifting and bench press coaching?"), ("agent", "Yes."))
    from acie.tree import state as S
    from acie.knowledge.store import load_kb
    kb = load_kb(FZ)
    e1, e2, e3 = [e.id for e in kb.usable_entries()][:3]
    st = S.load(s)
    st["asked"] = [e1, e2]
    st["given"] = [e1]
    st["offered"] = {kb.offerings[0].id: 1}
    st["total"] = len(kb.usable_entries())
    S.save(s, st)
    m = live_map.snapshot(s, FZ, "c1")
    assert m["kb"] is True and m["coverage"]["given"] == 1 and m["coverage"]["asked"] == 2
    states = {e["id"]: e["state"] for t in m["tree"]["topics"] for e in t["entries"]}
    assert states[e1] == "answered" and states[e2] == "asked" and states[e3] == "untouched"
    assert any(o["state"] == "recommended" for o in m["tree"]["offerings"])
    assert any(t["state"] == "answered" for t in m["tree"]["topics"])
    assert m["clusters"] and m["path"][-1]["clusters"]
    assert S.load(s)["asked"] == [e1, e2]  # read-only: the call's own coverage is untouched
    s.context.decision = "accepted"
    assert any(o["state"] == "closed" for o in live_map.snapshot(s, FZ, "c1", m)["tree"]["offerings"])


def test_path_grows_and_negotiation_accumulates(monkeypatch):
    monkeypatch.delenv("ACIE_KB", raising=False)
    s = session(("agent", "hi"), ("customer", "too expensive, any discount?"), ("agent", "Let me add a perk."),
                dialog={"_neg": json.dumps({"pushes": 2, "ladder_i": 1, "price": 2999.0}), "neg_offer": json.dumps({"price": 2999, "perk": "extra_week"})})
    m1 = live_map.snapshot(s, FZ, "c1")
    n = m1["negotiation"]
    assert n["list_price"] == 2999 and n["floor"] == 2400 and n["perks_given"] == ["extra_week"] and n["active"] and not n["floor_reached"]
    s.history += [Utterance(role="customer", text="lower please"), Utterance(role="agent", text="ok")]
    s.dialog.update({"_neg": json.dumps({"pushes": 3, "price": 2400.0, "final_said": True}), "neg_offer": json.dumps({"price": 2400}),
                     "closed_price": "2400.00", "margin_pct": "16.7"})
    m2 = live_map.snapshot(s, FZ, "c1", m1)
    assert [p["n"] for p in m2["path"]] == [2, 3]
    assert m2["negotiation"]["perks_given"] == ["extra_week"] and m2["negotiation"]["floor_reached"]
    assert m2["negotiation"]["agreed_price"] == 2400 and len(m2["negotiation"]["offers"]) == 2
    assert live_map.snapshot(s, CP, "c1")["negotiation"] is None  # no economics: no negotiation block


def test_capture_never_raises():
    class App:
        maps = {}
        live_sessions = {}
        def products(self): raise RuntimeError("boom")
    class Call: call_id = "x"; campaign = "none"
    class Broken:
        @property
        def history(self): raise RuntimeError("boom")
    assert live_map.capture(App(), Call(), Broken()) is None
    live_map.on_turn(App(), {"call_id": "x"})  # no live session: silently nothing


def test_from_record_fallback():
    rec = {"call_id": "a", "turns": [{"role": "customer", "text": "hello there"}, {"role": "agent", "text": "hi", "intent": "pitch"}],
           "context": {"decision": "accepted", "objections": ["price"], "promised": ["payment_link"]}}
    m = live_map.from_record(rec)
    assert m["stage"] == "after_close" and m["path"][0]["intent"] == "pitch" and m["facts"]["promised"] == ["payment_link"]


# ---------------------------------------------------------------- economics validation

def eco(**kw):
    # raw operator input (what the console PUTs), validated before the Economics model trims it
    base = dict(currency="₹", unit="3 months", list_price=3000, cost=2000, min_margin_pct=20,
                ladder=[Step(kind="price", price=2800), Step(kind="price", price=2500)])
    base.update(kw)
    base["ladder"] = [st.model_dump() if hasattr(st, "model_dump") else st for st in base["ladder"]]
    return base


def bad(e, product=FZ, match=""):
    with pytest.raises(ValueError, match=match):
        live_map.validate_economics(e, product)


def test_economics_rules():
    live_map.validate_economics(eco(), FZ)
    live_map.validate_economics(Economics(), FZ)  # empty = no concessions
    live_map.validate_economics(FZ.economics, FZ)  # the shipped config passes
    bad(eco(ladder=[Step(kind="price", price=2300)]), match="below the floor 2400")
    bad(eco(ladder=[Step(kind="price", price=3100)]), match="above the list price")
    bad(eco(ladder=[Step(kind="price", price=2800), Step(kind="price", price=2800)]), match="lower than the step before")
    bad(eco(cost=2600), match="floor .* above the list price")
    bad(eco(list_price=0), match="List price")
    bad(eco(ladder=[Step(kind="perk", id="a", text={"en": "x"})]), match="needs wording in hi, hinglish")
    ok = {"en": "a", "hi": "b", "hinglish": "c"}
    bad(eco(ladder=[Step(kind="perk", id="a", text=ok), Step(kind="perk", id="a", text=ok)]), match="used twice")
    bad(eco(), product=CP, match="real company")
    assert live_map.is_real_company(CP) and not live_map.is_real_company(FZ)


def test_max_perks():
    ok = {"en": "a", "hi": "b", "hinglish": "c"}
    e = eco(ladder=[Step(kind="perk", id="a", text=ok), Step(kind="perk", id="b", text=ok)])
    assert live_map.validate_max_perks(e, 2) == 2
    assert live_map.validate_max_perks(e, 1) == 1  # a ladder may list more perks than one sale may get
    with pytest.raises(ValueError):
        live_map.validate_max_perks(e, 9)
    with pytest.raises(ValueError):
        live_map.validate_max_perks(e, "x")


# ---------------------------------------------------------------- endpoints

@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")  # the base test venv has no web stack; CI for the voice server has
    pytest.importorskip("httpx")
    for k, v in {"PLIVO_AUTH_ID": "x", "PLIVO_AUTH_TOKEN": "x", "PLIVO_NUMBER": "+910000000000", "ADMIN_PASSWORD": "pw", "SESSION_SECRET": "s"}.items():
        os.environ.setdefault(k, v)
    from fastapi.testclient import TestClient
    try:
        from voice import server
    except ImportError as e:  # onnxruntime etc. missing in the base venv
        pytest.skip(f"voice server deps missing: {e}")
    data = tmp_path / "voice"
    (data / "calls").mkdir(parents=True)
    (data / "products.json").write_text((ROOT / "data/voice/products.json").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(server, "DATA", data)
    monkeypatch.setattr(server, "CALLS_DIR", data / "calls")
    monkeypatch.setattr(server.STATE.cfg, "admin_password", "pw")
    c = TestClient(server.app, base_url="https://t")
    assert c.post("/api/login", json={"password": "pw"}).status_code == 200
    c.server, c.data = server, data
    yield c
    server.STATE.maps.clear()


def test_map_endpoint(client):
    assert client.get("/api/calls/nope/map").status_code == 404
    snap = {"v": 1, "call_id": "done1", "stage": "close", "path": [], "kb": False}
    (client.data / "calls" / "done1.json").write_text(json.dumps({"call_id": "done1", "turns": [], "map": snap, "created_at": 1}))
    assert client.get("/api/calls/done1/map").json() == {"map": snap, "live": False}
    (client.data / "calls" / "old1.json").write_text(json.dumps({"call_id": "old1", "created_at": 2, "campaign": "fitzone-gym", "context": {"decision": "accepted"},
                                                                  "turns": [{"role": "customer", "text": "yes"}, {"role": "agent", "text": "ok", "intent": "link_sent"}]}))
    m = client.get("/api/calls/old1/map").json()["map"]
    assert m["rebuilt"] and m["stage"] == "after_close"
    client.server.STATE.maps["live1"] = {"stage": "open"}
    assert client.get("/api/calls/live1/map").json()["map"] == {"stage": "open"}
    listed = client.get("/api/calls").json()["calls"]
    assert all("map" not in c for c in listed)


def test_save_stores_map_and_hook_broadcasts(client):
    import asyncio
    st = client.server.STATE
    from voice.call import CallRecord
    call = CallRecord(to="+911", campaign="fitzone-gym", call_id="hk1")
    st.calls["hk1"] = call
    class Pipe: sessions = {"call-hk1": session(("agent", "hi"), ("customer", "hello"), ("agent", "hey"))}
    class CS: pipeline = Pipe(); session_id = "call-hk1"; product = FZ
    st.live_sessions["hk1"] = CS()
    q = asyncio.Queue()
    st.listeners.add(q)
    try:
        st.broadcast({"type": "turn", "call_id": "hk1", "role": "agent", "intent": "pitch", "text": "hey"})
        evs = []
        while not q.empty():
            evs.append(q.get_nowait())
        assert [e["type"] for e in evs] == ["turn", "map"] and evs[1]["map"]["call_id"] == "hk1"
        st.save(call)
        assert json.loads((client.data / "calls" / "hk1.json").read_text())["map"]["path"][0]["intent"] == "pitch"
    finally:
        st.listeners.discard(q)
        st.live_sessions.pop("hk1", None)
        st.calls.pop("hk1", None)


def test_put_product_economics(client):
    prods = {p["id"]: p for p in client.get("/api/products").json()["products"]}
    fz, cp = prods["fitzone-gym"], prods["claude-pro"]
    assert fz["real_company"] is False and cp["real_company"] is True and fz["economics"]["max_perks"] >= 1
    r = client.put("/api/products/fitzone-gym", json=fz)
    assert r.status_code == 200, r.text
    fz["economics"]["ladder"][2]["price"] = 2100  # below the 2400 floor
    r = client.put("/api/products/fitzone-gym", json=fz)
    assert r.status_code == 400 and "below the floor" in r.json()["detail"]
    fz["economics"]["ladder"][2]["price"] = 2800
    fz["economics"]["ladder"][3]["price"] = 2700
    fz["economics"]["ladder"][4]["price"] = 2500
    fz["economics"]["max_perks"] = 9
    r = client.put("/api/products/fitzone-gym", json=fz)
    assert r.status_code == 400 and "max_perks" in r.json()["detail"].lower().replace(" ", "_")
    fz["economics"]["max_perks"] = 2
    r = client.put("/api/products/fitzone-gym", json=fz)
    assert r.status_code == 200 and r.json()["product"]["economics"]["max_perks"] == 2
    saved = json.loads((client.data / "products.json").read_text(encoding="utf-8"))["fitzone-gym"]["economics"]
    assert saved["ladder"][4]["price"] == 2500 and saved["max_perks"] == 2
    cp["economics"] = {"list_price": 8, "cost": 1, "min_margin_pct": 10, "ladder": []}
    r = client.put("/api/products/claude-pro", json=cp)
    assert r.status_code == 400 and "real company" in r.json()["detail"]
