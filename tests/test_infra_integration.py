"""Integration tests against `docker compose -p sg-test up -d` (ports 16379/15432/14222). Run with ACIE_IT=1."""
import asyncio
import json
import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("ACIE_IT") != "1", reason="set ACIE_IT=1 with the compose services up")

PG = os.getenv("ACIE_PG_DSN", "postgresql://acie:acie@localhost:15432/acie")
REDIS = os.getenv("ACIE_REDIS_URL", "redis://localhost:16379/0")
NATS = os.getenv("ACIE_NATS_URL", "nats://localhost:14222")


async def test_redis_sessions_and_restart_resume():
    from acie.decision.domains.bank_demo import BankDemoDomain
    from acie.decision.engine import DecisionEngine
    from acie.pipeline import ACIEPipeline
    from acie.schemas import TurnInput
    from acie.store.sessions import RedisSessionStore
    st = RedisSessionStore(REDIS)
    await st.delete("it1")
    p1 = ACIEPipeline(DecisionEngine(BankDemoDomain()), session_store=st)
    await p1.run_turn(TurnInput(session_id="it1", customer_id="c", text="what is my balance"))
    assert 0 < await st.r.ttl("acie:session:it1") <= 7200
    p2 = ACIEPipeline(DecisionEngine(BankDemoDomain()), session_store=RedisSessionStore(REDIS))
    await p2.run_turn(TurnInput(session_id="it1", customer_id="c", text="and my limit"))
    assert p2.sessions["it1"].turn_index == 2
    await p2.forget_session("it1")
    assert await st.get("it1") is None
    await st.close()


async def test_postgres_profiles_and_migration(tmp_path):
    from acie.personalization.pg_store import PostgresProfileStore
    from acie.schemas import CustomerProfile
    st = PostgresProfileStore(PG)
    p = CustomerProfile(customer_id="it-c1")
    p.outcomes.append("accepted")
    await st.save(p)
    await st.save(p)  # upsert
    assert (await st.load("it-c1")).outcomes == ["accepted"]
    assert (await st.load("it-missing")).customer_id == "it-missing"
    import importlib.util
    spec = importlib.util.spec_from_file_location("mp", "deploy/migrate_profiles.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    (tmp_path / "m.json").write_text(CustomerProfile(customer_id="it-mig").model_dump_json())
    assert await mp.migrate(tmp_path, st) == (1, 0)
    assert (await st.load("it-mig")).customer_id == "it-mig"
    pool = await st._get_pool()
    async with pool.acquire() as c:
        await c.execute("DELETE FROM customer_profiles WHERE customer_id LIKE 'it-%'")
    await st.close()


class Emb:
    """Bag of letters, 26 dims: similar words land close together."""
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 26
            for ch in t.lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1
            out.append(v)
        return out


async def test_pgvector_memory_search():
    from acie.store.pgvector_memory import PgVectorMemoryStore
    st = PgVectorMemoryStore(PG, Emb(), dim=26)
    pool = await st._get_pool()
    async with pool.acquire() as c:
        await c.execute("DELETE FROM memories WHERE customer_id IN ('it-m1','it-m2')")
    await st.add("it-m1", "prefers payment by upi", {"k": 1})
    await st.add("it-m1", "dislikes long calls", {"k": 2})
    await st.add("it-m2", "upi upi upi", {})
    hits = await st.search("it-m1", "upi payment", k=2)
    assert hits[0]["text"] == "prefers payment by upi" and hits[0]["meta"] == {"k": 1}
    assert len(hits) == 2 and all("upi upi" not in h["text"] for h in hits)  # per-customer isolation
    async with pool.acquire() as c:
        await c.execute("DELETE FROM memories WHERE customer_id IN ('it-m1','it-m2')")
    await st.close()


async def test_nats_sink_real_broker():
    import nats
    from acie.context import ContextEvent
    from acie.events import NatsSink
    nc = await nats.connect(NATS)
    got = []
    async def cb(m):
        got.append(json.loads(m.data))
    sub = await nc.subscribe("acie.calls.it-call.context", cb=cb)
    await nc.flush()
    root = NatsSink(NATS)
    root.for_call("it-call")(ContextEvent(turn=1, who="agent", kind="promised", detail="callback_later"))
    await root.flush()
    await asyncio.sleep(0.3)
    assert got and got[0]["detail"] == "callback_later"
    await root.close()
    await sub.unsubscribe()
    await nc.close()


async def test_nats_sink_broker_unreachable_never_raises():
    from acie.context import ContextEvent
    from acie.events import NatsSink
    root = NatsSink("nats://localhost:1", max_queue=3)
    s = root.for_call("x")
    for i in range(10):
        s(ContextEvent(turn=i, who="agent", kind="k"))
    assert root.dropped == 7
    await asyncio.sleep(0.05)
    await root.close()
