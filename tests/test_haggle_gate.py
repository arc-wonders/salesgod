"""Gate regressions: the ledger price is the only price after the ladder moved (every tone), one address word per reply, no default Sir,
boss/bhai/yaar casual only, 'mushkil' once, link confirmed at the agreed price, a counter in range accepted consistently."""
import asyncio
import json
import re

import pytest

from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import ExpressionParams, TurnInput

RAW = json.load(open("data/voice/products.json"))
TONES = ("professional", "mid", "casual")
LIST_PRICE = re.compile(r"2,?999|50% off")
ADDR_PRO = re.compile(r"(?<![\wऀ-ॿ])(sir|ma'am|ji|सर|मैम|जी)(?![\wऀ-ॿ])", re.I)
ADDR_CAS = re.compile(r"(?<![\wऀ-ॿ])(bhai|boss|yaar|भाई|बॉस|यार)(?![\wऀ-ॿ])", re.I)


def talk(lines, tone, name="Ravi", lang="hinglish"):
    async def go():
        p = Product.model_validate({**RAW["fitzone-gym"], "tone_level": tone})
        pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(p, customer_name=name)))
        T = lambda t: TurnInput(session_id="s", customer_id="+910000000000", text=t)
        sess = pipe.session(T(""))
        sess.expression = ExpressionParams(language=lang)
        await pipe.decision.decide("greet", T(""), sess)
        out = []
        for line in lines:
            r = await pipe.run_turn(T(line))
            out.append((r.plan.intent, r.text))
        return out, sess
    return asyncio.run(go())


PROBES = ["kitna hai price?", "ye 50% off kya hai", "kya kya milega isme", "cancel kar sakte hain?", "phir se price bolo"]


@pytest.mark.parametrize("tone", TONES)
@pytest.mark.parametrize("probe", PROBES)
def test_after_the_ladder_moved_no_later_reply_says_the_list_price_or_50_off(tone, probe):
    out, _ = talk(["haan batao", "bahut mehenga hai", "aur kam karo", probe, "aur kam karo", probe], tone)
    for intent, text in out[3:]:
        assert not LIST_PRICE.search(text), (tone, probe, text)
    first_after = out[3][1]
    if "₹" in first_after:
        assert "₹2,799" in first_after


@pytest.mark.parametrize("tone", TONES)
def test_agreed_price_holds_in_confirm_and_link_and_the_link_is_confirmed_at_that_amount(tone):
    out, sess = talk(["haan batao", "bahut mehenga hai", "aur kam karo", "aur kam karo", "haan theek hai, kar do", "haan link bhej do", "kitna tha price?"], tone)
    assert out[4][0] == "confirm_close" and "₹2,649" in out[4][1]
    assert out[5][0] == "link_sent" and "₹2,649" in out[5][1] and "refund" not in out[5][1].lower() and "wapas" not in out[5][1]
    for _, t in out[4:]:
        assert not LIST_PRICE.search(t), t
    assert float(sess.dialog["closed_price"]) == 2649.0


@pytest.mark.parametrize("tone", TONES)
def test_counter_in_range_after_step_two_is_accepted_in_every_tone(tone):
    out, sess = talk(["haan batao", "bahut mehenga hai", "aur kam karo", "aur kam karo", "2600 kar do", "haan"], tone)
    assert "₹2,600" in out[4][1] and out[4][0] == "confirm_close"          # their number clears our next step: AGREED, confirmed with the terms
    assert out[5][0] == "link_sent" and "₹2,600" in out[5][1]
    assert float(sess.dialog["closed_price"]) == 2600.0


@pytest.mark.parametrize("name,expect", [("Pooja", "ma'am"), ("Rahul", "sir"), ("Zorba", "ji")])
def test_professional_never_defaults_to_sir_and_addresses_once_per_reply(name, expect):
    out, _ = talk(["haan batao", "bahut mehenga hai", "aur kam karo", "aur kam karo", "aur kam karo", "aur kam karo"], "professional", name)
    seen = set()
    for _, t in out[1:]:
        found = [m.group(0).lower() for m in ADDR_PRO.finditer(t)]
        assert len(found) <= 1, t
        seen |= set(found)
    assert "sir" not in seen or expect == "sir"
    assert "ma'am" not in seen or expect == "ma'am"
    if expect == "ji":
        assert seen <= {"ji"}


def test_boss_bhai_yaar_only_in_casual_or_after_the_caller_said_them():
    for tone in ("professional", "mid"):
        out, sess = talk(["haan batao", "bahut mehenga hai", "aur kam karo", "kam karo na", "aur kam", "rude: band karo ye sab"], tone)
        for _, t in out:
            assert not ADDR_CAS.search(t), (tone, t)
    out, sess = talk(["haan bhai batao", "bahut mehenga hai", "aur kam karo"], "mid")
    assert sess.dialog.get("_mirror") == "1" and ADDR_CAS.search(out[-1][1])
    out, sess = talk(["haan batao", "bahut mehenga hai", "aur kam karo"], "casual")
    for _, t in out[1:]:
        assert len(ADDR_CAS.findall(t)) <= 1, t


@pytest.mark.parametrize("tone", TONES)
def test_mushkil_once_and_final_once_per_call(tone):
    out, _ = talk(["haan batao", "mehenga"] + ["aur kam karo"] * 6, tone)
    txt = " ".join(t for _, t in out[1:])
    assert len(re.findall(r"mushkil|मुश्किल|kathin|कठिन|difficult", txt)) <= 2   # one "lower is difficult" (the first-no wording is separate)
    assert len(re.findall(r"final|फ़ाइनल|antim|अंतिम", txt, re.I)) <= 1
