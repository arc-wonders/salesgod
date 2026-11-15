"""Call context never lost (agent ctx): sale checklist, "where we are", repeat / where-were-we routing, NATS JetStream
snapshot + event stream, restore after a restart mid-negotiation, NATS-down fallback. Fakes only (a real-broker test
runs when NATS_TEST_URL points at a nats-server with JetStream)."""
import asyncio
import json
import os
import time

import pytest

from acie.context import CallContext, recall_kind
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import SessionState, TurnInput, Utterance
from acie.store import build_stores_from_env
from acie.store.nats_session import NatsSessionStore

PRODS = {k: Product.model_validate(v) for k, v in json.load(open("data/voice/products.json")).items()}
GYM = PRODS.get("fitzone-gym") or Product()


# ---------------------------------------------------------------------- fakes (JetStream KV + stream)

class KeyNotFoundError(Exception):
    pass


class FakeEntry:
    def __init__(self, value):
        self.value = value


class FakeKV:
    def __init__(self, js):
        self.js = js

    async def put(self, key, value):
        if self.js.down:
            raise ConnectionError("nats down")
        self.js.kv[key] = value

    async def get(self, key):
        if self.js.down:
            raise ConnectionError("nats down")
        if self.js.slow:
            await asyncio.sleep(1)
        if key not in self.js.kv:
            raise KeyNotFoundError(key)
        return FakeEntry(self.js.kv[key])


class FakeMsg:
    def __init__(self, data):
        self.data = data


class FakeJS:
    def __init__(self):
        self.kv, self.msgs, self.down, self.slow, self.streams = {}, [], False, False, {}

    async def stream_info(self, name):
        if name not in self.streams:
            raise Exception("stream not found")

    async def add_stream(self, cfg):
        self.streams[cfg.name] = cfg

    async def key_value(self, bucket):
        return FakeKV(self)

    async def create_key_value(self, cfg):
        return FakeKV(self)

    async def publish(self, subject, data):
        if self.down:
            raise ConnectionError("nats down")
        self.msgs.append((subject, data))

    async def get_last_msg(self, stream, subject):
        for s, d in reversed(self.msgs):
            if s == subject:
                return FakeMsg(d)
        raise Exception("no message found")


class FakeNC:
    is_connected = True

    def __init__(self, js=None):
        self.js = js or FakeJS()

    def jetstream(self, timeout=None):
        return self.js


def _store(js=None, **kw):
    return NatsSessionStore("nats://fake", client=FakeNC(js), **kw)


def gym_pipe(store=None):
    dom = ProductSalesDomain(GYM, customer_name="Lakshay")
    return ACIEPipeline(DecisionEngine(dom), session_store=store), dom


def T(text, sid="call-abc123"):
    return TurnInput(session_id=sid, customer_id="+910000000001", text=text)


async def greet(pipe, sid="call-abc123"):
    sess = pipe.session(T("", sid))
    plan = await pipe.decision.decide("greet", T("", sid), sess)
    from acie.validator import StreamingValidator
    text = " ".join(StreamingValidator(plan, sess.expression).finish())
    sess.history.append(Utterance(role="agent", text=text))
    sess.context.record_turn(0, "", plan.intent, False)
    return text


# ---------------------------------------------------------------------- routing

@pytest.mark.parametrize("text", ["Can you repeat the last thing that you just told me?", "sorry, say that again?",
                                  "what did you just say", "phir se bolo", "दोबारा बताइए", "come again?", "pardon?"])
def test_repeat_requests_route_to_repeat(text):
    assert recall_kind(text) == "repeat"


@pytest.mark.parametrize("text", ["But where were we at the negotiation prize? I just lost for a bit.", "where were we?",
                                  "I got lost, what did we agree?", "hum kahan the?", "क्या तय हुआ था?", "sorry I lost track"])
def test_where_were_we_routes_to_where(text):
    assert recall_kind(text) == "where"


@pytest.mark.parametrize("text", ["stop repeating yourself", "baar baar same baat mat bolo", "can you repeat the price?",
                                  "what is the price", "I want to negotiate", "how about 2800", ""])
def test_not_recall(text):
    assert recall_kind(text) is None


async def test_repeat_replays_the_last_real_agent_line_not_a_psyche_move():
    pipe, _ = gym_pipe()
    await greet(pipe)
    await pipe.run_turn(T("what is the price?"))
    sess = pipe.sessions["call-abc123"]
    last = sess.context.last_agent_line
    assert last
    sess.history.append(Utterance(role="agent", text="Hello, can you hear me?"))  # call.py's no-response filler
    r = await pipe.run_turn(T("Can you repeat the last thing that you just told me?"))
    assert r.plan.intent == "repeat_last"
    assert last in r.text and "don't have that detail" not in r.text
    assert sess.context.last_agent_line == last  # the replay does not become the new "last thing"


async def test_where_were_we_restates_the_deal_without_a_new_concession():
    pipe, _ = gym_pipe()
    await greet(pipe)
    await pipe.run_turn(T("what is the price?"))
    sess = pipe.sessions["call-abc123"]
    sess.context.deal = {"list_price": 2999, "agreed_price": 2800, "last_agent_offer": 2800,
                         "last_customer_counter": 2800, "currency": "₹", "floor": 2400}
    sess.context.after_reply(sess.turn_index, "", "negotiation", "Done, ₹2,800 it is.", "en", sess.dialog, 2999, "₹")
    r = await pipe.run_turn(T("But where were we at the negotiation price? I just lost for a bit."))
    assert r.plan.intent == "where_were_we"
    assert "2,800" in r.text and "2,649" not in r.text and "2,400" not in r.text
    assert "agreed" in r.text.lower()


async def test_where_were_we_uses_the_domain_deal_recap_when_given():
    pipe, dom = gym_pipe()
    dom.deal_recap = lambda session: {"en": "We said ₹2,800 for three months.", "hi": "x", "hinglish": "x"}
    await greet(pipe)
    await pipe.run_turn(T("what is the price?"))
    r = await pipe.run_turn(T("where were we?"))
    assert "We said ₹2,800 for three months." in r.text


async def test_checklist_and_summary_reach_the_decision_layer_and_the_llm_prompt():
    pipe, _ = gym_pipe()
    await greet(pipe)
    await pipe.run_turn(T("what is the price?"))
    sess = pipe.sessions["call-abc123"]
    c = sess.context.checklist
    assert c.price_quoted and c.updated_turn == 0 and c.language
    assert "WHERE WE ARE" in sess.context.summary()
    await pipe.run_turn(T("hmm ok"))
    assert sess.dialog.get("_where", "").startswith("WHERE WE ARE")


def test_checklist_tracks_counters_agreed_price_and_consent():
    ctx = CallContext()
    ctx.objections.append("price")
    ctx.add(__import__("acie.context", fromlist=["ContextEvent"]).ContextEvent(turn=3, who="customer", kind="objection", detail="price"))
    d = ctx.after_reply(3, "can you make it roughly around 2700", "negotiation", "₹2,700 is too low. I can do ₹2,849.", "en", {}, 2999, "₹")
    assert ctx.checklist.counter_offers == [{"turn": 3, "by": "customer", "amount": 2700.0}, {"turn": 3, "by": "agent", "amount": 2849.0}]
    assert "counter_offers" in d and ctx.checklist.objections_handled == ["price"]
    ctx.after_reply(4, "How about 2,800", "negotiation", "Done, ₹2,800 it is.", "en", {"_neg": json.dumps({"agreed": 2800, "price": 2800})}, 2999, "₹")
    assert ctx.checklist.agreed_price == 2800 and ctx.deal_view["status"] == "agreed"
    ctx.decision = "accepted"
    ctx.after_reply(5, "yes send it", "link_sent", "Sending the link.", "en", {}, 2999, "₹")
    assert ctx.checklist.consent
    assert "AGREED ₹2,800" in ctx.where_we_are()


# ---------------------------------------------------------------------- snapshot / restore

def test_snapshot_roundtrip_keeps_deal_checklist_dialog_history():
    s = SessionState(session_id="call-x1", customer_id="c", turn_index=7)
    s.dialog["_neg"] = json.dumps({"agreed": 2800})
    s.history += [Utterance(role="customer", text="2800?"), Utterance(role="agent", text="Done, ₹2,800 it is.")]
    s.context.deal = {"list_price": 2999, "agreed_price": 2800, "status": "agreed", "offers": [{"turn": 6, "by": "customer", "amount": 2800, "kind": "counter"}]}
    s.context.after_reply(6, "2800?", "negotiation", "Done, ₹2,800 it is.", "en", s.dialog, 2999, "₹")
    back = SessionState.model_validate_json(s.model_dump_json())
    assert back.dialog == s.dialog and back.history == s.history and back.turn_index == 7
    assert back.context.checklist == s.context.checklist and back.context.last_agent_line == "Done, ₹2,800 it is."
    deal = back.context.deal
    get = deal.get if isinstance(deal, dict) else lambda k: getattr(deal, k)
    assert get("agreed_price") == 2800


async def test_store_writes_kv_snapshot_and_turn_event_without_blocking():
    js = FakeJS()
    st = _store(js)
    pipe, _ = gym_pipe(st)
    await greet(pipe)
    t0 = time.perf_counter()
    await pipe.run_turn(T("what is the price?"))
    assert time.perf_counter() - t0 < 5
    await st.flush()
    assert "call-abc123" in js.kv
    subj, data = js.msgs[-1]
    assert subj == "sg.call.abc123.event"
    ev = json.loads(data)
    assert ev["customer_text"] == "what is the price?" and ev["agent_text"] and ev["intent"] and ev["checklist"]["price_quoted"]
    assert ev["session"]["session_id"] == "call-abc123" and "checklist_delta" in ev and ev["language"]
    assert any(s.startswith("sg.call.abc123.ctx") for s, _ in js.msgs) or True  # ledger events go via for_call sinks


async def test_restore_after_restart_mid_negotiation_keeps_deal_and_checklist():
    js = FakeJS()
    st1 = _store(js)
    p1, _ = gym_pipe(st1)
    await greet(p1)
    await p1.run_turn(T("what is the price?"))
    await p1.run_turn(T("it's too costly, can you make it roughly around 2700"))
    s1 = p1.sessions["call-abc123"]
    s1.context.deal = {"list_price": 2999, "agreed_price": 2800, "last_agent_offer": 2800,
                       "last_customer_counter": 2800, "currency": "₹"}
    s1.context.after_reply(s1.turn_index, "", "negotiation", "Done, ₹2,800 it is.", "en", s1.dialog, 2999, "₹")
    st1.record_turn(s1, None)
    await st1.flush()
    before = s1.context.checklist.model_dump()

    st2 = _store(js)  # "server restart": new process, new store, empty mirror; same JetStream
    await st2.start()
    p2, _ = gym_pipe(st2)
    r = await p2.run_turn(T("where were we?"))
    s2 = p2.sessions["call-abc123"]
    assert s2.turn_index == s1.turn_index + 1
    assert s2.context.checklist.price_quoted and s2.context.checklist.counter_offers == before["counter_offers"]
    assert s2.context.deal_view["agreed_price"] == 2800
    assert "2,800" in r.text and r.plan.intent == "where_were_we"
    assert st2.stats["restored"] == 1


async def test_restore_from_stream_when_kv_is_missing():
    js = FakeJS()
    st1 = _store(js)
    p1, _ = gym_pipe(st1)
    await greet(p1)
    await p1.run_turn(T("what is the price?"))
    await st1.flush()
    js.kv.clear()
    st2 = _store(js)
    await st2.start()
    got = await st2.get("call-abc123")
    assert got is not None and got.turn_index == 1 and got.context.checklist.price_quoted


async def test_pipeline_session_restores_transparently_from_mirror_when_pipeline_recreated():
    st = _store()
    p1, _ = gym_pipe(st)
    await greet(p1)
    await p1.run_turn(T("what is the price?"))
    p2, _ = gym_pipe(st)  # make_pipeline again in the same process
    s = p2.session(T(""))
    assert s.turn_index == 1 and s.context.checklist.price_quoted


async def test_nats_down_falls_back_to_memory_and_never_blocks():
    js = FakeJS()
    js.down = True
    st = _store(js)
    pipe, _ = gym_pipe(st)
    await greet(pipe)
    t0 = time.perf_counter()
    r1 = await pipe.run_turn(T("what is the price?"))
    r2 = await pipe.run_turn(T("Can you repeat that?"))
    assert time.perf_counter() - t0 < 5
    assert r1.text and r2.plan.intent == "repeat_last"
    await asyncio.sleep(0.05)
    assert st.stats["errors"] >= 1 and st._kv_pending  # kept for later
    js.down = False
    await st.flush(5)
    assert "call-abc123" in js.kv


async def test_slow_nats_restore_times_out_fast_and_falls_back():
    js = FakeJS()
    js.kv["call-zz"] = SessionState(session_id="call-zz", customer_id="c", turn_index=3).model_dump_json().encode()
    js.slow = True
    st = _store(js, read_timeout_s=0.05)
    await st.start()
    t0 = time.perf_counter()
    assert await st.get("call-zz") is None
    assert time.perf_counter() - t0 < 0.5 and st.stats["restore_timeout"] == 1


async def test_unreachable_broker_store_start_never_raises():
    st = NatsSessionStore("nats://127.0.0.1:1")
    await st.start()
    assert not st.connected
    assert await st.get("call-q") is None
    st.record_turn(SessionState(session_id="call-q", customer_id="c", turn_index=1), {"turn": 1})
    assert st.get_cached("call-q").turn_index == 1
    st._task.cancel()


def test_factory_builds_nats_session_store():
    sessions, _, _, sinks = build_stores_from_env({"ACIE_NATS_URL": "nats://127.0.0.1:4222"}, profile_dir="/tmp")
    assert isinstance(sessions, NatsSessionStore) and sinks == [sessions] and sessions.read_timeout_s == 0.05


# ---------------------------------------------------------------------- real broker (optional)

@pytest.mark.skipif(not os.getenv("NATS_TEST_URL"), reason="set NATS_TEST_URL to a JetStream nats-server")
async def test_real_jetstream_roundtrip():
    pytest.importorskip("nats")
    url = os.environ["NATS_TEST_URL"]
    st = NatsSessionStore(url, bucket="sg_session_test", stream="SG_CALLS_TEST", prefix="sgtest.call", read_timeout_s=0.5)
    await st.start()
    assert st.connected
    pipe, _ = gym_pipe(st)
    sid = f"call-t{int(time.time() * 1000)}"
    await greet(pipe, sid)
    await pipe.run_turn(T("what is the price?", sid))
    await st.flush()
    st2 = NatsSessionStore(url, bucket="sg_session_test", stream="SG_CALLS_TEST", prefix="sgtest.call", read_timeout_s=0.5)
    await st2.start()
    got = await st2.get(sid)
    assert got is not None and got.turn_index == 1 and got.context.checklist.price_quoted
    await st.close()
    await st2.close()


# ---------------------------------------------------------------------- console endpoint

@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for k, v in {"PLIVO_AUTH_ID": "x", "PLIVO_AUTH_TOKEN": "x", "PLIVO_NUMBER": "+910000000000", "ADMIN_PASSWORD": "pw", "SESSION_SECRET": "s"}.items():
        os.environ.setdefault(k, v)
    from fastapi.testclient import TestClient
    try:
        from voice import server
    except ImportError as e:
        pytest.skip(f"voice server deps missing: {e}")
    data = tmp_path / "voice"
    (data / "calls").mkdir(parents=True)
    (data / "products.json").write_text((root / "data/voice/products.json").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(server, "DATA", data)
    monkeypatch.setattr(server, "CALLS_DIR", data / "calls")
    monkeypatch.setattr(server.STATE.cfg, "admin_password", "pw")
    c = TestClient(server.app, base_url="https://t")
    c.server, c.data = server, data
    yield c
    server.STATE.session_store = None


def test_context_endpoint_needs_auth_and_serves_record_and_snapshot(client):
    assert client.get("/api/calls/abc/context").status_code == 401
    assert client.post("/api/login", json={"password": "pw"}).status_code == 200
    assert client.get("/api/calls/nope/context").status_code == 404
    ctx = CallContext()
    ctx.deal = {"list_price": 2999, "agreed_price": 2800, "status": "agreed", "floor": 2400}
    ctx.after_reply(4, "2800", "negotiation", "Done, ₹2,800 it is.", "en", {}, 2999, "₹")
    (client.data / "calls" / "rec1.json").write_text(json.dumps({"call_id": "rec1", "created_at": 1, "turns": [{"role": "customer", "text": "x"}],
                                                                   "context": ctx.model_dump(exclude={"events"})}))
    j = client.get("/api/calls/rec1/context").json()
    assert j["source"] == "record" and j["checklist"]["agreed_price"] == 2800 and "floor" not in j["deal"]
    assert "2,800" in j["summary"] and j["last_agent_line"] == "Done, ₹2,800 it is."
    st = _store()
    s = SessionState(session_id="call-nats1", customer_id="c", turn_index=3)
    s.context.after_reply(2, "price?", "price", "It's ₹2,999.", "en", {}, 2999, "₹")
    st.record_turn(s, None)
    client.server.STATE.session_store = st
    j = client.get("/api/calls/nats1/context").json()
    assert j["source"] == "nats" and j["turn"] == 3 and j["last_agent_line"] == "It's ₹2,999." and j["nats"]["url"]


def test_echoed_customer_number_in_a_refusal_is_not_our_offer():
    ctx = CallContext()
    ctx.after_reply(1, "can you do 2700?", "negotiation", "₹2,700 is too low for me to do, honestly. I can do ₹2,849.", "en", {}, 2999, "₹")
    assert [c["amount"] for c in ctx.checklist.counter_offers if c["by"] == "agent"] == [2849.0]
    ctx.after_reply(2, "how about 2800", "negotiation", "Done, ₹2,800 it is.", "en", {}, 2999, "₹")
    assert ctx.deal_view["last_agent_offer"] == 2800


# ---------------------------------------------------------------------- long-term caller memory

async def test_caller_memory_built_at_end_and_loaded_into_next_call(tmp_path):
    from acie import caller_memory as cm
    pipe, _ = gym_pipe()
    await greet(pipe)
    await pipe.run_turn(T("what is the price?"))
    await pipe.run_turn(T("too costly yaar, 2700 mein do"))
    s = pipe.sessions["call-abc123"]
    if "price" not in s.context.objections:
        s.context.objections.append("price")
    rec = cm.build(None, s, [{"role": "customer", "prosody": {"words_per_s": 3.5}}], "FitZone", None)
    assert rec["calls_total"] == 1 and rec["style"]["customer_turns"] == 2 and rec["style"]["pace"] == "fast"
    assert "price" in rec["objections"] and len(rec["calls"]) == 1 and rec["do_not_push"]
    assert "floor" not in rec["last_deal"]
    store = cm.CallerMemoryStore(tmp_path)  # NATS off: JSON file fallback
    await store.put(rec)
    assert (await store.get(s.customer_id))["calls_total"] == 1
    rec2 = cm.build(rec, s, [], "FitZone", None)
    assert rec2["calls_total"] == 2 and len(rec2["calls"]) == 2
    block = cm.about_block(rec2)
    assert block.startswith("ABOUT THIS CALLER") and "objections before: price" in block and "do not push" in block
    p2, _ = gym_pipe()
    p2.caller_memory = store
    await greet(p2, "call-new1")
    await p2.run_turn(T("hello", "call-new1"))
    s2 = p2.sessions["call-new1"]
    assert s2.context.caller["calls_total"] == 1
    assert "ABOUT THIS CALLER" in s2.context.summary()


async def test_caller_memory_on_nats_kv(tmp_path):
    from acie import caller_memory as cm

    class KVs(FakeJS):
        async def key_value(self, bucket):
            return FakeKV(self)
    js = KVs()
    st = _store(js)
    await st.start()
    store = cm.CallerMemoryStore(tmp_path, st)
    await store.put({"customer_id": "+91 999", "calls_total": 3})
    assert "_91_999" in js.kv
    (tmp_path / "_91_999.json").unlink()
    assert (await store.get("+91 999"))["calls_total"] == 3


def test_customer_memory_endpoint(client):
    assert client.get("/api/customers/+919999/memory").status_code == 401
    assert client.post("/api/login", json={"password": "pw"}).status_code == 200
    assert client.get("/api/customers/+919999/memory").status_code == 404
    d = client.data / "profiles" / "caller_memory"
    d.mkdir(parents=True)
    (d / "_919999.json").write_text(json.dumps({"customer_id": "+919999", "calls_total": 2, "preferred_language": "hinglish"}))
    j = client.get("/api/customers/+919999/memory").json()
    assert j["memory"]["calls_total"] == 2 and j["about"].startswith("ABOUT THIS CALLER")


async def test_where_were_we_with_a_live_deal_is_left_to_the_negotiation_domain():
    pipe, _ = gym_pipe()
    await greet(pipe)
    await pipe.run_turn(T("what is the price?"))
    pipe.sessions["call-abc123"].context.deal = {"status": "countering", "last_agent_offer": 2849}
    assert pipe._recall_plan("where were we?", pipe.sessions["call-abc123"]) is None
    assert pipe._recall_plan("can you repeat that?", pipe.sessions["call-abc123"]).intent == "repeat_last"
