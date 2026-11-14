import asyncio

import pytest

from acie import expression as ex
from acie.expression import (SESSION_KEY, ABBackend, FailoverBackend, FirstTokenTimeout, OpenAICompatibleBackend,
                             make_backend_from_env, parse_ab)
from acie.schemas import CompiledPrompt, ExpressionParams, Fact, TurnPlan

PLAN = TurnPlan(intent="x", objective="o", facts={"b": Fact(value="₹5", description="d")},
                fallback={"en": "Balance {b}."})
PROMPT = CompiledPrompt(system="s", messages=[])
EXPR = ExpressionParams()


def run(coro):
    return asyncio.run(coro)


async def collect(b):
    return [d async for d in b.stream(PROMPT, PLAN, EXPR)]


def backend(raw, **kw):
    b = OpenAICompatibleBackend(base_url="http://x/v1", model="m", **kw)
    b._stream_raw = raw
    return b


async def ok(*a):
    yield "hello "
    yield "there."


async def slow(*a):
    await asyncio.sleep(1)
    yield "late"


async def boom(*a):
    raise RuntimeError("down")
    yield ""


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ex.METRICS.reset()
    for k in ("ACIE_EXPR_AB", "ACIE_EXPR_FALLBACK_BASE_URL", "ACIE_EXPR_FIRST_TOKEN_S", "ACIE_EXPR_FALLBACK_MODEL"):
        monkeypatch.delenv(k, raising=False)


def test_default_stream_unchanged_and_metrics():
    assert run(collect(backend(ok))) == ["hello ", "there."]
    snap = ex.METRICS.snapshot()["backends"]["m"]
    assert snap["turns"] == 1 and snap["errors"] == 0 and snap["first_token_p50_ms"] is not None


def test_first_token_timeout_falls_back_to_template():
    out = run(collect(backend(slow, first_token_s=0.05)))
    assert out == ["Balance {b}."]
    assert ex.METRICS.snapshot()["backends"]["m"]["timeouts"] == 1


def test_first_token_env(monkeypatch):
    monkeypatch.setenv("ACIE_EXPR_FIRST_TOKEN_S", "0.05")
    assert run(collect(backend(slow))) == ["Balance {b}."]
    monkeypatch.delenv("ACIE_EXPR_FIRST_TOKEN_S")
    assert backend(ok).first_token_s is None  # default off


def test_error_counted_and_raised():
    with pytest.raises(RuntimeError):
        run(collect(backend(boom)))
    assert ex.METRICS.snapshot()["backends"]["m"]["errors"] == 1


def test_failover_on_error_and_timeout():
    sec = backend(ok, label="sec")
    assert run(collect(FailoverBackend(backend(boom), sec))) == ["hello ", "there."]
    assert run(collect(FailoverBackend(backend(slow, first_token_s=0.05), backend(ok, label="sec")))) == ["hello ", "there."]
    assert ex.METRICS.failovers == 2


def test_failover_not_after_text_started():
    async def half(*a):
        yield "part "
        raise RuntimeError("cut")

    async def go():
        got = []
        with pytest.raises(RuntimeError):
            async for d in FailoverBackend(backend(half), backend(ok)).stream(PROMPT, PLAN, EXPR):
                got.append(d)
        return got

    assert run(go()) == ["part "]


def test_parse_ab():
    assert parse_ab("a:0.5,b:0.5") == [("a", .5), ("b", .5)]
    assert parse_ab("solo") == [("solo", 1.0)]
    assert parse_ab("m:x") == [("m:x", 1.0)]
    assert parse_ab("a:0,b:1") == [("b", 1.0)]


def test_ab_sticky_and_split():
    ab = ABBackend([("A", backend(ok, label="A"), .5), ("B", backend(ok, label="B"), .5)])
    first = {}
    for i in range(200):
        SESSION_KEY.set(f"s{i}")
        first[i] = ab.pick()[0]
    for i in range(200):
        SESSION_KEY.set(f"s{i}")
        assert ab.pick()[0] == first[i]
    share = sum(v == "A" for v in first.values()) / 200
    assert 0.35 < share < 0.65
    SESSION_KEY.set("s0")
    run(collect(ab))
    assert ex.METRICS.turns[first[0]] == 1
    SESSION_KEY.set(None)


def test_make_backend_from_env(monkeypatch):
    assert isinstance(make_backend_from_env(), OpenAICompatibleBackend)
    monkeypatch.setenv("ACIE_EXPR_AB", "ma:0.7,mb:0.3")
    b = make_backend_from_env()
    assert isinstance(b, ABBackend) and [v[0] for v in b.variants] == ["ma", "mb"]
    monkeypatch.setenv("ACIE_EXPR_FALLBACK_BASE_URL", "http://backup/v1")
    monkeypatch.setenv("ACIE_EXPR_FALLBACK_MODEL", "small")
    f = make_backend_from_env()
    assert isinstance(f, FailoverBackend) and f.secondary.model == "small"
    assert all(v[1].fallback_to_template is False for v in f.primary.variants)
