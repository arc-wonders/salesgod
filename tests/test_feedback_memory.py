import asyncio
import json

from acie.feedback import FeedbackEngine
from acie.memory.base import Embedder, MemoryStore, make_store
from acie.memory.hashing import HashingEmbedder
from acie.memory.inmemory import InMemoryStore
from acie.schemas import CustomerProfile, SessionState


def run(c):
    return asyncio.run(c)


def test_embedder_shape_and_protocols():
    e = HashingEmbedder()
    v = e.embed(["namaste", "नमस्ते", ""])
    assert len(v) == 3 and all(len(x) == 256 for x in v) and abs(sum(x * x for x in v[0]) - 1) < 1e-6
    assert isinstance(e, Embedder) and isinstance(InMemoryStore(), MemoryStore)


def test_ranking_english_and_hindi():
    async def go():
        m = InMemoryStore()
        for t in ("customer asked about price and found it too costly", "wants a callback tomorrow evening",
                  "worried the service is fake, asked for refund policy"):
            await m.add("c", t, {})
        en = await m.search("c", "was the price too costly", 3)
        assert en[0]["text"].startswith("customer asked about price")
        hi = InMemoryStore()
        for t in ("ग्राहक को कीमत बहुत ज्यादा लगी", "कल शाम को वापस कॉल करना है", "सेवा नकली लगती है, रिफंड पूछा"):
            await hi.add("c", t, {})
        assert (await hi.search("c", "कल शाम कॉल", 1))[0]["text"] == "कल शाम को वापस कॉल करना है"
        assert (await hi.search("c", "कीमत ज्यादा", 1))[0]["text"].startswith("ग्राहक को कीमत")
        rh = InMemoryStore()
        for t in ("price bahut zyada hai", "kal shaam ko call karna", "refund policy kya hai"):
            await rh.add("c", t, {})
        assert (await rh.search("c", "kal shaam call", 1))[0]["text"] == "kal shaam ko call karna"
        assert await m.search("other", "x") == []
        assert (await m.search("c", "", 1))[0]["text"].startswith("worried")  # no query -> most recent
    run(go())


def test_remember_and_recall_off_by_default(monkeypatch):
    monkeypatch.delenv("ACIE_MEMORY", raising=False)
    fb = FeedbackEngine()
    assert fb.memory is None
    s = SessionState(session_id="s", customer_id="c")
    run(fb.remember(s))
    run(fb.recall(s, "x"))
    assert "_memories" not in s.dialog


def test_remember_then_recall_next_call():
    fb = FeedbackEngine(memory=InMemoryStore())
    s = SessionState(session_id="s1", customer_id="c")
    s.context.decision = "declined"
    s.context.objections = ["too costly"]
    s.context.questions = [{"q": "kya refund milega?", "answer": "answer"}]
    p = CustomerProfile(customer_id="c")
    run(fb.remember(s, p))
    run(fb.remember(s, p))  # once
    s2 = SessionState(session_id="s2", customer_id="c")
    run(fb.recall(s2, "price too costly"))
    mem = json.loads(s2.dialog["_memories"])
    assert len(mem) == 1 and "too costly" in mem[0] and "declined" in mem[0]


def test_make_store_falls_back(monkeypatch):
    monkeypatch.setenv("ACIE_MEMORY_STORE", "pgvector")
    assert isinstance(make_store(), (InMemoryStore, object))
