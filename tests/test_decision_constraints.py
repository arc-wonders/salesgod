from datetime import datetime, timezone

from decision_helpers import run

from acie.business import Customer
from acie.decision import constraints as cons
from acie.decision.domains.product_sales import Product


def P(*rules, **kw):
    return Product(constraints=list(rules), **kw)


async def test_default_empty_constraints_change_nothing():
    out, _, _, _, opening = await run(["yes tell me"])
    assert opening.intent == "opening" and out[-1] == "pitch"


async def test_min_age_blocks_with_compliant_template():
    out, last, pipe, _, opening = await run([], P({"type": "min_age", "value": 18}), customer_age=16)
    assert opening.intent == "constraint_stop" and opening.end_call and not opening.llm_allowed
    assert set(opening.fallback) == {"en", "hi", "hinglish"}
    out, _, _, _, opening = await run([], P({"type": "min_age", "value": 18}), customer_age=30)
    assert opening.intent == "opening"
    out, _, _, _, opening = await run([], P({"type": "min_age", "value": 18}))  # unknown age: rule does not apply
    assert opening.intent == "opening"


async def test_region_and_existing_subscriber():
    r = {"type": "region_allow", "regions": ["Delhi", "Mumbai"]}
    assert (await run([], P(r), customer=Customer(region="chennai")))[4].intent == "constraint_stop"
    assert (await run([], P(r), customer=Customer(region="delhi")))[4].intent == "opening"
    e = {"type": "exclude_existing_subscriber"}
    assert (await run([], P(e), customer=Customer(existing_subscriber=True)))[4].intent == "constraint_stop"
    assert (await run([], P(e), customer=Customer(existing_subscriber=False)))[4].intent == "opening"
    assert (await run([], P(), customer=Customer(existing_subscriber=True)))[4].intent == "opening"


async def test_price_floor():
    assert (await run([], P({"type": "price_floor", "min": 10})))[4].intent == "constraint_stop"  # offer is $8
    assert (await run([], P({"type": "price_floor", "min": 5})))[4].intent == "opening"


async def test_call_hours_uses_local_clock():
    rule = {"type": "call_hours", "start": 9, "end": 21, "utc_offset_minutes": 330}
    night = lambda: datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc)  # 01:30 IST
    day = lambda: datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)  # 11:30 IST
    o = (await run([], P(rule), now_fn=night))[4]
    assert o.intent == "constraint_stop" and "time" in o.fallback["en"]
    assert (await run([], P(rule), now_fn=day))[4].intent == "opening"


async def test_dnc_list_and_live_lookup():
    rule = {"type": "dnc", "numbers": ["+91 98765 43210"]}
    o = (await run([], P(rule), customer=Customer(id="+919876543210")))[4]
    assert o.intent == "constraint_stop" and "do-not-call" in o.fallback["en"]
    assert (await run([], P({"type": "dnc"}), customer=Customer(id="+911111111111"),
                      dnc_lookup=lambda n: n.endswith("1111")))[4].intent == "constraint_stop"
    assert (await run([], P(rule), customer=Customer(id="+911111111111")))[4].intent == "opening"


async def test_max_reoffers_zero_respects_the_first_no():
    out, last, pipe, _, _ = await run(["yes tell me", "no not interested"], P({"type": "max_reoffers", "value": 0}))
    assert out[-1] == "declined" and last.plan.end_call
    out, _, _, _, _ = await run(["yes tell me", "no not interested"])
    assert out[-1] == "reason_ask"  # the first soft no is asked about, not re-offered (OWNER UPDATE 2)


def test_unknown_constraint_type_is_ignored():
    ctx = cons.ConstraintContext(product=Product())
    assert cons.evaluate([{"type": "nope"}], ctx) is None
