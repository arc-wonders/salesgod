import asyncio
import json

from acie.feedback import FeedbackEngine
from acie.memory.base import HashingEmbedder
from acie.memory.inmemory import InMemoryStore
from acie.personalization.learner import learn_from_turn
from acie.pipeline import ACIEPipeline
from acie.decision.domains.bank_demo import BankDemoDomain
from acie.decision.engine import DecisionEngine
from acie.schemas import TurnInput


class Spy:
    def __init__(self):
        self.calls = []

    def decide(self, *a, **k):
        from acie.personalization.policy import AdaptationPolicy
        return self._p.decide(*a, **k)

    def observe(self, state, expr, reward, profile):
        self.calls.append(reward)


def mkpipe(**kw):
    from acie.personalization.policy import AdaptationPolicy
    spy = Spy()
    spy._p = AdaptationPolicy()
    return spy, ACIEPipeline(DecisionEngine(BankDemoDomain()), policy=spy, **kw)


def T(text, sid="s1"):
    return TurnInput(session_id=sid, customer_id="c1", text=text)


def test_reexport():
    assert HashingEmbedder().dim == 256


def test_learner_and_observe_run_in_pipeline(monkeypatch):
    seen = []
    import acie.personalization.learner as L
    monkeypatch.setattr(L, "learn_from_turn", lambda *a: seen.append(a[1]) or [])
    spy, pipe = mkpipe()

    async def go():
        await pipe.run_turn(T("mera balance batao"))
        await pipe.run_turn(T("aur kitna hai"))
    asyncio.run(go())
    assert seen == ["mera balance batao", "aur kitna hai"]
    assert len(spy.calls) == 1  # turn 1 resolved by turn 2


def test_compiler_gets_memories_only_when_on(monkeypatch):
    got = []
    spy, pipe = mkpipe()
    orig = pipe.compiler.compile

    def cap(*a, **k):
        got.append(k)
        return orig(*a, **k)
    pipe.compiler.compile = cap
    monkeypatch.delenv("ACIE_MEMORY", raising=False)
    asyncio.run(pipe.run_turn(T("balance")))
    assert got[0] == {}  # default path identical to before

    monkeypatch.setenv("ACIE_MEMORY", "1")
    mem = InMemoryStore()
    asyncio.run(mem.add("c1", "decision: later; objections: too costly", {}))
    pipe.feedback = FeedbackEngine(memory=mem)
    got.clear()
    asyncio.run(pipe.run_turn(T("balance", sid="s2")))
    assert got[0]["memories"] == ["decision: later; objections: too costly"] and got[0]["profile"].customer_id == "c1"
