"""Every customer line of every real call (git-ignored data/real/calls) goes through the pipeline with the build products
and a minimal 'live-like' gym product: the domain must never fall back to its crash handler (a silent line on a live call)."""
import asyncio
import json
import logging
from pathlib import Path

import pytest

from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import TurnInput

ROOT = Path(__file__).resolve().parents[1]
CALLS = sorted((ROOT / "data/real/calls").glob("*.json"))
RAW = json.loads((ROOT / "data/voice/products.json").read_text())

pytestmark = pytest.mark.skipif(not CALLS, reason="no real-call data on this machine (git-ignored)")


def _live_like() -> Product:
    """Minimal gym product as the live server had it: no economics, no negotiation levers, no FAQs."""
    d = {k: v for k, v in RAW["fitzone-gym"].items() if k not in ("economics", "negotiation", "faqs", "terms", "info_source")}
    return Product.model_validate(d)


def _build(camp: str) -> Product:
    return Product.model_validate(RAW[camp if camp in RAW else "claude-pro"])


async def _replay(prod: Product, lines: list[str], sid: str) -> None:
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(prod)))
    T = lambda t: TurnInput(session_id=sid, customer_id="+910000000000", text=t)
    sess = pipe.session(T(""))
    await pipe.decision.decide("greet", T(""), sess)
    for line in lines:
        r = await pipe.run_turn(T(line))
        assert r.text.strip(), f"empty reply for {line!r}"
        if r.plan.end_call:
            break


@pytest.mark.parametrize("variant", ["build", "live_like"])
def test_no_exception_on_any_real_call_line(variant, caplog):
    caplog.set_level(logging.ERROR, logger="acie.decision")
    for f in CALLS:
        c = json.loads(f.read_text())
        lines = [t["text"] for t in c.get("turns", []) if t.get("role") == "customer" and t.get("text", "").strip()]
        prod = _live_like() if variant == "live_like" else _build(c.get("campaign", ""))
        asyncio.run(_replay(prod, lines, f"rc-{f.stem}-{variant}"))
    crashes = [r for r in caplog.records if "plan() failed" in r.getMessage()]
    assert not crashes, [r.exc_text.splitlines()[-1] for r in crashes[:3] if r.exc_text]


def test_plan_exception_falls_back_safely(monkeypatch):
    dom = ProductSalesDomain(Product())

    async def boom(*a, **k):
        raise TypeError("boom")

    monkeypatch.setattr(dom, "_plan_unsafe", boom)
    pipe = ACIEPipeline(DecisionEngine(dom))
    T = lambda t: TurnInput(session_id="s", customer_id="+910000000000", text=t)
    r = asyncio.run(pipe.run_turn(T("kitna hai")))
    assert r.text.strip() and r.plan.intent == "clarify" and not r.plan.end_call
