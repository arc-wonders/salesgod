"""Booking slot memory and the close (live call c1d9da4eca, concert, 15 Nov): the caller said VIP, five tickets and Delhi; the agent
asked again for the city, zone and count, re-listed the shows, read "Danny" (STT for Delhi) as an artist, gave the haggle line to
"the most expensive one", logged "5 pickets" as a counter-offer of 12 and never read the order back. These replay the caller's lines."""

import asyncio
import json
import re
from pathlib import Path

import pytest

from acie import playbook as pbm
from acie.context import CallContext, price_amounts
from acie.decision import negotiation
from acie.decision import order as om
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import TurnInput, Utterance
from acie.validator import StreamingValidator

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "data/voice/products.json"
pytestmark = pytest.mark.skipif(not PRODUCTS.exists(), reason="needs data/voice/products.json")
CONCERT_ID = "karan-aujla-concert-tickets"


def _prod(pid: str) -> Product:
    return Product.model_validate(json.loads(PRODUCTS.read_text())[pid])


class FakeLLM:
    async def stream(self, prompt, plan, expr):
        yield plan.fallback.get(expr.language) or plan.fallback.get("en")


def _call(pid, tenant, lines, sid="order1"):
    pb = pbm.load(ROOT / "data/voice", tenant, pid)
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(_prod(pid), customer_name="Asha", playbook=pb)), backend=FakeLLM(), buffer_reply=True)
    T = lambda t: TurnInput(session_id=sid, customer_id="+910000000000", text=t)  # noqa: E731

    async def go():
        sess = pipe.session(T(""))
        sess.dialog["_lang"] = "en"
        plan = await pipe.decision.decide("greet", T(""), sess)
        text = " ".join(StreamingValidator(plan, sess.expression).finish())
        sess.history.append(Utterance(role="agent", text=text))
        sess.context.record_turn(0, "", plan.intent, False)
        out = []
        for c in lines:
            r = await pipe.run_turn(T(c))
            out.append((c, r.text, r.plan))
        return out, sess
    return asyncio.run(go())


# ---------------------------------------------------------------- unit: the catalog and the slot readers come from product data

def test_catalog_from_concert_data():
    cat = om.catalog(_prod(CONCERT_ID))
    names = {t.name: t.price for t in cat.tiers}
    assert names["VIP Lounge"] == 14999 and names["Gold"] == 3999 and names["General Standing"] == 1999 and names["Fan Pit"] == 7999
    assert [l.name for l in cat.locations] == ["Delhi", "Mumbai", "Chandigarh"]
    assert cat.location("Delhi").weekday == "Saturday" and cat.location("Delhi").date == "12 December"
    assert cat.countable and cat.slots == ["tier", "quantity", "location"]
    assert cat.next_steps.startswith("After payment")


def test_fitzone_catalog_has_no_required_slots():
    cat = om.catalog(_prod("fitzone-gym"))
    assert cat.slots == []  # one plan, per-membership unit: the usual confirmation (with the renewal terms) is kept


@pytest.mark.parametrize("text,n", [("5 pickets, VIP zone and", 5), ("Okay, I want five tickets for VIP.", 5), ("we are 4 people", 4),
                                    ("me and my wife", 2), ("Delhi on 12 December", 0), ("at 7 pm", 0), ("₹14,999", 0), ("paanch tickets", 5)])
def test_quantity(text, n):
    assert om.parse_quantity(text) == n


def test_near_miss_city_only_in_booking_context():
    cat = om.catalog(_prod(CONCERT_ID))
    assert om.parse_location("Okay, Danny, Danny, Danny. Book the Danny show for me.", cat, booking_ctx=True) == "Delhi"
    assert om.parse_location("Yes, daily show, daily show.", cat, booking_ctx=True) == "Delhi"
    assert om.parse_location("I spoke to Danny yesterday", cat, booking_ctx=False) == ""
    assert om.parse_location("not Mumbai, Delhi", cat) == "Delhi"


def test_premium_is_the_top_tier_and_not_a_haggle():
    cat = om.catalog(_prod(CONCERT_ID))
    t = "Okay, I want to inquire about the most expensive one for my family."
    assert om.parse_tier(t, cat) == "VIP Lounge"
    assert negotiation.detect(t) not in ("haggle", "counter_offer")


def test_quantities_and_dates_are_not_counter_offers():
    assert price_amounts("5 pickets, VIP zone and 12 December") == []
    assert price_amounts("can you do 3,000?") == [3000.0]
    ctx = CallContext()
    ctx.after_reply(13, "5 pickets, VIP zone and 12", "answer", "VIP Lounge is ₹14,999 a ticket.", "en",
                    {"_neg": "{}"}, list_price=3999.0, currency="₹")
    assert ctx.checklist.counter_offers == []


# ---------------------------------------------------------------- replay of c1d9da4eca

CALL = ["I'm good.", "Sure.", "Yes, do tell me the brands.",
        "Okay, I want to inquire about the most expensive one for my family.",
        "Tell me the price for VIP.", "Okay, I want five tickets for VIP.", "Okay, now",
        "I think that Delhi would be more suitable for me.", "Yes."]


def test_replay_c1d9da4eca_books_vip_for_delhi():
    out, sess = _call(CONCERT_ID, "concerts", CALL, sid="c1d9")
    said = {c: (t, p) for c, t, p in out}
    t, p = said["Okay, I want to inquire about the most expensive one for my family."]
    assert p.intent != "negotiation" and "cutting the price" not in t and "VIP Lounge" in t
    t, p = said["Okay, I want five tickets for VIP."]
    assert p.intent == "order_slot" and "Delhi" in t and not re.search(r"how many|which zone", t, re.I)
    t, p = said["I think that Delhi would be more suitable for me."]
    assert p.intent == "confirm_close", (t, p.intent)
    assert "5 VIP Lounge tickets for Delhi" in t and "₹14,999 each" in t and "₹74,995 total" in t and "Saturday 12 December" in t
    assert "Fan Pit" not in t and "Gold" not in t
    t, p = said["Yes."]
    assert p.intent == "link_sent" and "₹74,995" in t and "to your email" in t and "you's" not in t
    o = sess.context.checklist.order
    assert o["tier"] == "VIP Lounge" and o["quantity"] == 5 and o["location"] == "Delhi" and o["status"] == "link"
    assert sess.context.checklist.counter_offers == []


def test_replay_danny_and_pickets_never_reasks():
    lines = ["I'm good.", "Sure.", "Okay, I want five tickets for VIP.", "Okay, Danny, Danny, Danny. Book the Danny show for me.", "Yes."]
    out, sess = _call(CONCERT_ID, "concerts", lines, sid="danny")
    t, p = out[3][1], out[3][2]
    assert "not Danny" not in t and p.intent == "confirm_close" and "Delhi" in t
    assert out[4][2].intent == "link_sent"


def test_booking_without_slots_asks_one_at_a_time():
    lines = ["I'm good.", "Sure.", "Yes, book the tickets for me", "Gold", "3", "Mumbai", "Yes."]
    out, sess = _call(CONCERT_ID, "concerts", lines, sid="steps")
    asks = [p for _, _, p in out[2:5]]
    assert all(p.intent == "order_slot" for p in asks)
    assert sum(t.count("?") for _, t, _ in out[2:5]) == 3  # one question per turn
    t = out[5][1]
    assert out[5][2].intent == "confirm_close" and "3 Gold tickets for Mumbai" in t and "₹3,999 each" in t and "₹11,997 total" in t
    assert out[6][2].intent == "link_sent"


def test_slot_change_after_readback_reads_back_again():
    lines = ["I'm good.", "Sure.", "I want 2 Fan Pit tickets for Chandigarh", "Actually make it 4 tickets", "Yes."]
    out, sess = _call(CONCERT_ID, "concerts", lines, sid="chg")
    assert out[2][2].intent == "confirm_close" and "₹15,998 total" in out[2][1]
    assert out[3][2].intent == "confirm_close" and "4 Fan Pit tickets" in out[3][1] and "₹31,996 total" in out[3][1]
    assert out[4][2].intent == "link_sent"


def test_fitzone_booking_still_confirms_full_terms():
    lines = ["I'm good.", "Sure.", "Sign me up, take my money", "Yes."]
    out, sess = _call("fitzone-gym", "fitzone", lines, sid="gym")
    assert out[2][2].intent == "confirm_close" and "₹2,999" in out[2][1] and "₹6,000" in out[2][1]
    assert out[3][2].intent == "link_sent"


def test_tier_price_named_is_a_choice_not_a_counter():
    lines = ["I'm good.", "Sure.", "Koi sasta option hai?", "General Standing ₹1,999 wala sahi rahega."]
    out, sess = _call(CONCERT_ID, "concerts", lines, sid="tierprice")
    assert out[3][2].intent != "negotiation" and "mushkil" not in out[3][1] and "difficult" not in out[3][1]


def test_total_named_after_readback_is_consent_not_a_counter():
    lines = ["I'm good.", "Sure.", "Mujhe Mumbai show ke liye 3 Gold tickets chahiye.", "3 Gold tickets ka final total ₹11,997 hua na? Payment link bhej dijiye."]
    out, sess = _call(CONCERT_ID, "concerts", lines, sid="total")
    assert out[2][2].intent == "confirm_close" and "₹11,997" in out[2][1]
    assert out[3][2].intent == "link_sent", (out[3][1], out[3][2].intent)


def test_item_count_beats_head_count():
    assert om.parse_quantity("membership do logon ke liye kitne ki hai? Mujhe 3 Gold tickets chahiye") == 3
    assert om.parse_quantity("₹2,999 for 3 months") == 0


def test_fitzone_two_memberships_read_back_with_total():
    lines = ["I'm good.", "Sure.", "Me and my wife both want to join, sign us up.", "Yes."]
    out, sess = _call("fitzone-gym", "fitzone", lines, sid="gym2")
    t = out[2][1]
    assert out[2][2].intent == "confirm_close" and "2 memberships" in t and "₹5,998" in t and "₹6,000" in t, t
    assert out[3][2].intent == "link_sent" and "₹5,998" in out[3][1]


def test_fitzone_head_count_is_a_total_not_rupees():
    lines = ["I'm good.", "Sure.", "Okay, so what is the total for 5 of us?"]
    out, sess = _call("fitzone-gym", "fitzone", lines, sid="gym5")
    t = out[2][1]
    assert out[2][2].intent == "confirm_close" and "₹14,995" in t and "₹5?" not in t, t
