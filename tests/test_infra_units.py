"""Infra lane unit tests: fakes only, no services, no optional deps needed."""
import asyncio
import json

import pytest

from acie.context import ContextEvent
from acie.decision.domains.bank_demo import BankDemoDomain
from acie.decision.engine import DecisionEngine
from acie.events import NatsSink
from acie.personalization.pg_store import PostgresProfileStore
from acie.pipeline import ACIEPipeline
from acie.schemas import CustomerProfile, SessionState, TurnInput
from acie.store import build_stores_from_env
from acie.store.pgvector_memory import PgVectorMemoryStore
from acie.store.sessions import InMemorySessionStore, RedisSessionStore


class FakeRedis:
    def __init__(self):
        self.d, self.ttl = {}, {}

    async def get(self, k):
        return self.d.get(k)

    async def set(self, k, v, ex=None):
        self.d[k], self.ttl[k] = v, ex

    async def delete(self, k):
        self.d.pop(k, None)


def _sess():
    s = SessionState(session_id="s1", customer_id="c1", turn_index=3)
    s.dialog["x"] = "y"
    s.context.record_turn(1, "price?", "price", True)
    return s


async def test_inmemory_roundtrip():
    st = InMemorySessionStore()
    assert await st.get("s1") is None
    await st.put(_sess())
    assert (await st.get("s1")).turn_index == 3
    await st.delete("s1")
    assert await st.get("s1") is None


async def test_redis_store_roundtrip_and_ttl():
    r = FakeRedis()
    st = RedisSessionStore(client=r)
    await st.put(_sess())
    assert r.ttl["acie:session:s1"] == 7200
    got = await st.get("s1")
    assert got.dialog == {"x": "y"} and got.context.events and got.turn_index == 3
    await st.delete("s1")
    assert await st.get("s1") is None


async def test_pipeline_resumes_session_after_restart():
    r = FakeRedis()
    p1 = ACIEPipeline(DecisionEngine(BankDemoDomain()), session_store=RedisSessionStore(client=r))
    await p1.run_turn(TurnInput(session_id="s9", customer_id="c1", text="what is my balance"))
    assert "s9" in p1.sessions and r.d
    p2 = ACIEPipeline(DecisionEngine(BankDemoDomain()), session_store=RedisSessionStore(client=r))  # "restart"
    await p2.run_turn(TurnInput(session_id="s9", customer_id="c1", text="and my limit"))
    assert p2.sessions["s9"].turn_index == 2
    assert len(p2.sessions["s9"].history) == 4
    await p2.forget_session("s9")
    assert not r.d


async def test_pipeline_default_has_no_store():
    p = ACIEPipeline(DecisionEngine(BankDemoDomain()))
    assert p.session_store is None
    await p.run_turn(TurnInput(session_id="a", customer_id="c1", text="hello"))
    assert "a" in p.sessions


class FakePool:
    """Just enough asyncpg pool for the two stores."""
    def __init__(self):
        self.rows, self.executed = {}, []

    def acquire(self):
        pool = self

        class Ctx:
            async def __aenter__(self_):
                return pool

            async def __aexit__(self_, *a):
                return False
        return Ctx()

    async def execute(self, sql, *args):
        self.executed.append(sql)
        if sql.startswith("INSERT INTO customer_profiles"):
            self.rows[args[0]] = args[1]

    async def fetchval(self, sql, cid):
        return self.rows.get(cid)


async def test_pg_profile_store_with_fake_pool():
    pool = FakePool()
    st = PostgresProfileStore("dsn", pool=pool)
    assert (await st.load("new")).customer_id == "new"
    prof = CustomerProfile(customer_id="c7", likes=["tea"]) if "likes" in CustomerProfile.model_fields else CustomerProfile(customer_id="c7")
    await st.save(prof)
    assert (await st.load("c7")) == prof
    assert any("ON CONFLICT" in s for s in pool.executed)


class HashEmb:
    def embed(self, texts):
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]


async def test_pgvector_checks_dimension_and_schema():
    st = PgVectorMemoryStore("dsn", HashEmb(), dim=8, pool=FakePool())
    with pytest.raises(ValueError):
        await st.add("c1", "hello", {})
    sql = " ".join(PgVectorMemoryStore("d", HashEmb(), 4).schema())
    assert "vector(4)" in sql and "hnsw" in sql and "memories" in sql
    assert "ivfflat" in " ".join(PgVectorMemoryStore("d", HashEmb(), 4, index="ivfflat").schema())


class FakeNats:
    def __init__(self, fail_first=0):
        self.sent, self.fail = [], fail_first

    async def publish(self, subject, data):
        if self.fail:
            self.fail -= 1
            raise RuntimeError("down")
        self.sent.append((subject, json.loads(data)))

    async def flush(self):
        pass

    async def close(self):
        pass


def _ev(i=0):
    return ContextEvent(turn=i, who="agent", kind="promised", detail="payment_link")


async def test_nats_sink_publishes_to_call_subject():
    nc = FakeNats()
    root = NatsSink("nats://x", client=nc)
    root.for_call("call42")(_ev())
    await root.flush()
    assert nc.sent[0][0] == "acie.calls.call42.context" and nc.sent[0][1]["detail"] == "payment_link"
    await root.close()


async def test_nats_sink_drops_on_overflow_and_never_raises():
    nc = FakeNats()
    root = NatsSink("nats://x", max_queue=2, client=nc)
    sink = root.for_call("c")
    for i in range(5):
        sink(_ev(i))  # sync, no await: queue fills before the task runs
    assert root.dropped == 3
    sink(object())  # bad event: swallowed
    await root.flush()
    assert len(nc.sent) == 2
    await root.close()


async def test_nats_sink_survives_broker_outage():
    nc = FakeNats(fail_first=1)
    root = NatsSink("nats://x", client=nc)
    root.for_call("c")(_ev())
    await root.flush(timeout=3)
    assert len(nc.sent) == 1
    await root.close()


def test_nats_sink_without_loop_does_not_raise():
    NatsSink("nats://x", client=FakeNats()).for_call("c")(_ev())


def test_factory_defaults_are_todays(tmp_path):
    sessions, profiles, memory, sinks = build_stores_from_env({}, profile_dir=tmp_path)
    assert isinstance(sessions, InMemorySessionStore) and memory is None and sinks == []
    assert type(profiles).__name__ == "JsonProfileStore"


def test_factory_picks_backends():
    pytest.importorskip("redis")
    pytest.importorskip("asyncpg")
    sessions, profiles, memory, sinks = build_stores_from_env(
        {"ACIE_SESSION_STORE": "redis", "ACIE_PROFILE_STORE": "postgres", "ACIE_PG_DSN": "postgresql://x/y",
         "ACIE_MEMORY_STORE": "pgvector", "ACIE_NATS_URL": "nats://x"}, embedder=HashEmb(), embed_dim=4)
    assert type(sessions).__name__ == "RedisSessionStore"
    assert type(profiles).__name__ == "PostgresProfileStore"
    assert type(memory).__name__ == "PgVectorMemoryStore"
    assert type(sinks[0]).__name__ == "NatsSink"


async def test_migrate_profiles_script(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mp", "deploy/migrate_profiles.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    (tmp_path / "a.json").write_text(CustomerProfile(customer_id="a").model_dump_json())
    (tmp_path / "bad.json").write_text("{nope")
    from acie.personalization.profile_store import MemoryProfileStore
    st = MemoryProfileStore()
    assert await mp.migrate(tmp_path, st) == (1, 1)
    assert (await st.load("a")).customer_id == "a"
    assert await mp.migrate(tmp_path, None, dry_run=True) == (1, 1)
