"""Gate run 4 deploy blockers: each test replays the exact caller lines from the transcripts / real calls."""
import asyncio
import json
from pathlib import Path

from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import TurnInput

RAW = json.loads((Path(__file__).resolve().parents[1] / "data/voice/products.json").read_text())


def converse(lines, camp="claude-pro", nlu_crisis=(), sid="gf"):
    """-> [(intent, text)] for each caller line. nlu_crisis = turn indexes where the NLU flags a crisis."""
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(Product.model_validate(RAW[camp]))))
    T = lambda t: TurnInput(session_id=sid, customer_id="+910000000000", text=t)  # noqa: E731
    sess = pipe.session(T(""))

    async def go():
        await pipe.decision.decide("greet", T(""), sess)
        out = []
        for i, line in enumerate(lines):
            if i in nlu_crisis:
                sess.dialog["_nlu_crisis"] = "model flag"
            r = await pipe.run_turn(T(line))
            out.append((r.plan.intent, r.text, r.plan.end_call))
        return out

    return asyncio.run(go())


def test_nlu_flag_alone_is_not_crisis():
    out = converse(["haan bolo", "मुझे ये तो बताओ क्या मैं ये offer अभी भी ले सकता हूँ क्योंकि मेरे पास पहले से cloud है मेरा account नया नहीं है"],
                   nlu_crisis=(1,))
    assert not any(i.startswith(("safety_distress", "distress")) for i, _, _ in out)
    assert not out[1][0].startswith("safety")  # a product question is released, not even a check-in


def test_nlu_flag_gives_at_most_one_checkin_then_reassurance_releases():
    out = converse(["haan bolo", "abhi job nahi hai yaar", "नहीं, मैंने महीने की fees पूछी थी", "मैं ठीक हूँ, बस fees पूछ रहा हूँ", "ok batao"],
                   nlu_crisis=(2, 3))
    intents = [i for i, _, _ in out]
    assert "safety_distress" not in intents and "safety_followup" not in intents
    assert not any(e for _, _, e in out)
    assert not intents[3].startswith("safety")


def test_nlu_flag_plus_concern_lexical_is_crisis_and_true_crisis_stays():
    out = converse(["haan", "मेरे मरने का मन कर रहा है"])
    assert out[1][0] == "safety_distress"


def test_optout_question_is_answered_not_confirmed():
    out = converse(["haan bolo", "Uh, who is this? How did you get my number?", "Hmm okay. But what is the offer? Why is it sixty percent off, is it fake?"])
    assert out[1][0] == "number_source"
    assert out[2][0] != "opt_out" and not out[2][2]
    assert "removed your number" not in out[2][1].lower()


def test_optout_explicit_yes_confirms():
    for yes in ("haan hata do", "yes remove it", "हाँ हटाओ", "don't call again"):
        out = converse(["haan bolo", "who is this? how did you get my number", yes])
        assert out[2][0] in ("opt_out",) and out[2][2], yes


def test_no_empathy_on_plan_questions_or_consult():
    out = converse(["haan bolo bhai", "monthly option hai kya? ek saath 3 mahine ka paisa nahi hai abhi"], camp="fitzone-gym")
    assert out[1][0] != "personal" and "बुरा" not in out[1][1] and "bura laga" not in out[1][1]
    out = converse(["हैलो", "बेटा मैं आज पैसे नहीं दे सकती, पोते से पूछना पड़ेगा। आप बाद में फोन कर लेना, ठीक है?"], camp="fitzone-gym")
    assert out[1][0] != "personal" and "बुरा" not in out[1][1]
    out = converse(["haan bolo", "abhi job nahi hai yaar"])  # real hardship still gets the warm line
    assert out[1][0] == "personal"


def test_free_plan_flow_is_product_scoped():
    out = converse(["haan bolo", "abhi job nahi hai yaar", "free wala bhi chalega kya?"], camp="fitzone-gym")
    assert "free plan" not in out[2][1].lower() and "फ्री प्लान" not in out[2][1]
    out = converse(["haan bolo", "abhi job nahi hai yaar", "free wala bhi resume mein kaam karta hai kya"])
    assert "free plan" in out[2][1].lower()


def test_trying_the_free_plan_is_not_consent_to_buy():
    for lines in (["haan bolo", "free wala bhi resume mein kaam karta hai kya", "accha, theek hai. try karta hoon"],
                  ["haan bolo", "price?", "free plan try karta hoon"]):
        out = converse(lines)
        assert out[-1][0] not in ("confirm_close", "link_sent", "confirm_buy") and "$8" not in out[-1][1]


def test_price_pace_no_urgency_and_comparison_is_answered():
    out = converse(["haan bolo", "Cult.fit में 1999 का मंथली है। आपके यहाँ क्या अलग है?", "Cult से महँगा पड़ेगा न?", "what's better than Cult?"], camp="fitzone-gym")
    for _, text, _ in out[1:]:
        assert "₹2,800" not in text and "₹2,700" not in text and "अगर अभी" not in text
    assert out[1][0] == "negotiation" and "फ्री पर्सनल ट्रेनर" in out[1][1]
    out = converse(["haan bolo", "arre 500 kam karo na bhai", "aur kam karo", "aur kam karo"], camp="fitzone-gym")
    assert not any("agar abhi" in t.lower() or "if we finalise" in t.lower() for _, t, _ in out)


def test_questions_are_not_refusal_or_counter_offer():
    out = converse(["haan bolo", "Gemini to free hai, uska kya?"])
    assert "kisi aur time" not in out[1][1] and "kam nahi kar sakta" not in out[1][1]
    out = converse(["haan bolo", "₹2,999 monthly ya yearly?"], camp="fitzone-gym")
    assert "पक्का" not in out[1][1] and out[1][0] != "negotiation"
    out = converse(["haan bolo", "price?", "link bhej do, $8 wala try karta hoon"])
    assert out[2][0] == "link_sent"
    out = converse(["हैलो", "बेटा कितने पैसे लगेंगे?"], camp="fitzone-gym")
    assert out[1][0] == "price" and "किसी और" not in out[1][1]


def test_callback_for_any_day_is_booked():
    out = converse(["haan bolo", "ठीक है, मैं पहले जिम आकर देख लेता हूँ, फिर decide करूँगा। संडे को callback कर सकते हैं?"], camp="fitzone-gym")
    assert out[1][0] == "callback" and "संडे" in out[1][1]
    out = converse(["haan bolo", "मतलब callback नहीं आएगा? मैं संडे को खाली रहूँगा, आप note कर लो।"], camp="fitzone-gym")
    assert out[1][0] == "callback" and "नहीं कर पाऊँगा" not in out[1][1]


def test_after_send_it_the_link_is_confirmed_not_pitched():
    out = converse(["I'm in a meeting, 20 seconds.", "Just tell me the price.", "Send it on WhatsApp.", "Fine, SMS. Quick.", "Yes send it.", "Did you send it? I have to go."])
    assert out[3][0] == "link_sent"
    assert "higher usage" not in out[4][1] and "higher usage" not in out[5][1]
    assert "sent" in out[5][1].lower() or "gone" in out[5][1].lower()
