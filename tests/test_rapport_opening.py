"""The step-by-step opening (owner, 15 Nov), generic for every product: 1) greet + who + AI disclosure + how are you; 2) reply like a
person, then purpose + recording notice + "interested?" (no price); 3) a yes gets a warm acknowledgement and the normal flow.
Busy -> callback; heard-only: an unheard step is not treated as said. ACIE_RAPPORT_OPENING=1 is pinned for this file (tests/conftest.py)."""
import json
import re

import pytest

from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.schemas import TurnInput

from test_flow_fixes import PRODUCTS, _call

pytestmark = pytest.mark.skipif(not PRODUCTS.exists(), reason="needs data/voice/products.json")


def _opening(product_id="fitzone-gym"):
    import asyncio
    raw = json.loads(PRODUCTS.read_text())[product_id]
    dom = ProductSalesDomain(Product.model_validate(raw), customer_name="Asha")
    from acie.pipeline import ACIEPipeline
    pipe = ACIEPipeline(DecisionEngine(dom))
    T = TurnInput(session_id="rap0", customer_id="+910000000000", text="")
    return asyncio.run(pipe.decision.decide("greet", T, pipe.session(T)))


@pytest.mark.parametrize("pid", ["fitzone-gym", "claude-pro", "karan-aujla-concert-tickets"])
def test_rapport_step1_greets_discloses_ai_and_asks_how_they_are(pid):
    plan = _opening(pid)
    en = plan.fallback["en"]
    assert "AI assistant" in en and "How are you" in en and "Asha" in en and "offer" not in en.lower() and "₹" not in en, en
    assert "आप कैसे हैं" in plan.fallback["hi"] and "Aap kaise hain" in plan.fallback["hinglish"]


def test_rapport_reply_then_purpose_recording_and_permission_no_price():
    out, sess = _call([("I'm good, how are you?", None, None)], sid="rap1")
    text, plan, tr = out[0]
    assert plan.intent == "purpose" and text.startswith("I'm doing well, thanks for asking!"), text
    assert "offer on FitZone membership" in text and "This call is recorded." in text and text.rstrip().endswith("?") and "₹" not in text, text
    assert tr["gate"] == "scripted" and tr["decision"].startswith("opening step 2")
    assert "call_recorded" in sess.context.told and "purpose_stated" in sess.context.told


def test_rapport_fine_gets_warm_reply_and_yes_gets_connected_ack_then_normal_flow():
    out, _ = _call([("I'm fine.", None, None), ("Yes, I'm interested.", None, {"intent": "affirm", "buying_signal": 0.6})], sid="rap2")
    assert out[0][0].startswith("Glad to hear that!"), out[0][0]
    text, plan, _ = out[1]
    assert plan.intent == "pitch" and text.startswith("Perfect") and "₹" not in text, text
    assert plan.intent not in ("confirm_close", "link_sent")


def test_rapport_busy_gets_a_callback_offer():
    out, _ = _call([("I'm busy right now.", None, None)], sid="rap3")
    text, plan, _ = out[0]
    assert plan.intent == "callback_time" and re.search(r"call back", text, re.I) and "offer" not in text.lower(), text


def test_rapport_no_after_purpose_is_respected_not_pitched():
    out, _ = _call([("Fine, thanks.", None, None), ("No, not interested.", None, {"intent": "deny", "buying_signal": 0.0})], sid="rap4")
    assert out[1][1].intent in ("reason_ask", "declined", "reoffer") and out[1][1].intent != "pitch" and "₹" not in out[1][0]


def test_rapport_cut_greeting_purpose_step_says_who_and_ai_again():
    out, _ = _call([("Hello?", None, None)], sid="rap5", cut_opening=0.15)
    text, plan, _ = out[0]
    assert plan.intent == "purpose" and "This is Arjun, an AI assistant from FitZone" in text and "recorded" in text, text
    assert text.count("Just so you know") == 0


def test_rapport_price_question_first_gets_the_recording_notice_before_the_answer():
    out, _ = _call([("How much is the membership?", None, None)], sid="rap6")
    text = out[0][0]
    assert "recorded" in text and text.index("recorded") < text.index("₹"), text


def test_rapport_bad_day_is_acknowledged():
    out, _ = _call([("Not so good, long day.", None, None)], sid="rap7")
    assert out[0][0].startswith("Sorry to hear that"), out[0][0]


def test_rapport_pain_or_sarcasm_never_gets_glad_to_hear():
    out, _ = _call([("Main thik hu, bas aajkal ghutne mein thoda dard rehta hai. Aap batao, kya kaam tha?", None, None)], sid="rap8")
    assert not re.search(r"accha laga|glad to hear", out[0][0], re.I), out[0][0]
    out, _ = _call([("Oh brilliant, an AI sales call. Just what my day was missing.", None, {"intent": "unknown", "sarcasm": 0.9})], sid="rap9")
    assert out[0][1].intent == "purpose" and not re.search(r"glad|great to hear", out[0][0], re.I), out[0][0]


def test_rapport_yes_with_a_rider_still_gets_the_pitch():
    out, _ = _call([("I'm okay.", None, None), ("Fine, give me the thirty-second pitch before I hang up.", None, None)], sid="rap10")
    assert out[1][1].intent == "pitch" and not out[1][1].end_call, out[1][0]
    out, _ = _call([("ठीक हूँ।", None, None), ("हाँ बताइए क्या ऑफ़र है? लेकिन ध्यान रखना, मुझे कोई झूठी गारंटी नहीं चाहिए।", None, None)], sid="rap11")
    assert out[1][1].intent == "pitch", out[1][0]


def test_rapport_purpose_after_other_questions_has_no_stale_how_are_you_reply():
    out, _ = _call([("Where did you get my number?", None, None), ("Okay. Why did you call?", None, None)], sid="rap12")
    assert "thanks for asking" not in out[1][0] and "recorded" in out[1][0], out[1][0]
