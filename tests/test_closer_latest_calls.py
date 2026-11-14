"""Paraphrased from live calls 2c4ec7a4ff / a0c97d20f1 / ef6400b533: bad input must never become an invented meaning."""
import asyncio
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "qa"))
import run_qa  # noqa: E402
from acie.decision.garble import looks_garbled  # noqa: E402
from acie.rules.respectful_register import RespectfulRegisterPack  # noqa: E402
from acie.schemas import ExpressionParams, TurnPlan  # noqa: E402

PRODS = run_qa.load_products()


def play(lines, lang="hi", psyche="", camp="fitzone-gym"):
    old = os.environ.get("ACIE_PSYCHE")
    os.environ["ACIE_PSYCHE"] = psyche
    try:
        fx = {"id": "t", "campaign": camp, "opening_language": lang, "turns": [{"say": s} for s in lines], "class": "x"}
        return asyncio.run(run_qa.play(fx, PRODS))[1:]
    finally:
        os.environ.pop("ACIE_PSYCHE", None) if old is None else os.environ.__setitem__("ACIE_PSYCHE", old)


INVENTED = re.compile(r"crowd|भीड़|screen|sound|official website|ऑफिशियल वेबसाइट|दोस्तों के साथ|friends", re.I)


@pytest.mark.parametrize("psyche", ["", "1"])
@pytest.mark.parametrize("lang,lines", [("hi", ["पूर्ण लोग", "आब दोस्तों", "हवागारिया"]), ("en", ["हवागारिया", "को नियर झाल"])])
def test_garbled_turns_ask_to_repeat_never_invent(lang, lines, psyche):
    r = play(lines, lang, psyche)
    for x in r:
        assert x["intent"] == "clarify" and not INVENTED.search(x["reply"]) and not x["end_call"]
    assert len({x["reply"] for x in r}) == len(r)  # varied, never the same line twice
    if len(r) >= 2:  # from the second garbled turn in a row: a callback or another language is offered
        assert re.search(r"call|कॉल|language|भाषा|bhasha", r[1]["reply"], re.I)


def test_garble_detector_is_conservative():
    for ok in ["haan", "बोलो बोलो", "हम्म", "acha", "kitna", "refund", "trainer", "क्या मिलेगा", "what is the price", "hmm ok", "ठीक है"]:
        assert not looks_garbled(ok), ok
    for bad in ["पूर्ण लोग", "आब दोस्तों", "हवागारिया"]:
        assert looks_garbled(bad), bad
    assert looks_garbled("whatever words", signal=True)


def test_low_stt_confidence_is_garbled():
    class T:  # the call record's recogniser scores
        stt_avg_logprob = -1.8
        stt_no_speech = None
    assert looks_garbled("kitna hai", turn=T())


@pytest.mark.parametrize("lang,q", [("en", "What if I don't want to buy?"), ("hinglish", "agar mujhe nahi lena ho to?"), ("hi", "अगर मुझे नहीं लेना हो तो क्या होगा?")])
def test_a_question_about_not_buying_is_never_a_decline(lang, q):
    r = play(["yes", q], lang)[1]
    assert r["intent"] != "declined" and not r["end_call"]
    assert re.search(r"obligation|बाध्यता|baadhyata|nothing is charged|कुछ नहीं कटता|kuch nahi katta", r["reply"], re.I)


@pytest.mark.parametrize("frag", ["कर दो कि अ", "kar do ki a"])
def test_short_fragment_without_a_buying_signal_is_never_consent(frag):
    r = play(["हाँ बोलो", "आपका fitness center कैसा है", frag])
    assert r[-1]["intent"] not in ("confirm_close", "link_sent")


@pytest.mark.parametrize("q", ["am I audible?", "can you hear me?", "क्या आपको मेरी आवाज़ आ रही है", "meri awaaz aa rahi hai?"])
def test_audio_complaint_gets_a_short_honest_answer_not_a_pitch(q):
    r = play(["yes", q], "en" if q[0].isascii() and "awaaz" not in q else "hi")[1]
    assert r["intent"] == "clarify" and not re.search(r"₹|price|offer|goal|ऑफर|कीमत", r["reply"], re.I)
    assert re.search(r"hear|audible|आवाज़|awaaz", r["reply"], re.I)


def test_register_pack_rejects_casual_address():
    pk = RespectfulRegisterPack()
    plan, expr = TurnPlan(intent="answer", objective="x"), ExpressionParams(language="hi")
    for bad in ["सोरी यार, हाँ मुझे तेरी बात समझ आ रही है।", "अरे भाई, दोस्तों के साथ आना है तो मज़ा आएगा!", "अब तुम्हारी आवाज़ आ रही है।", "Sorry yaar, I can hear you."]:
        assert pk.check(bad, plan, expr), bad
    for ok in ["जी, आपकी आवाज़ आ रही है।", "Of course, I can hear you."]:
        assert not pk.check(ok, plan, expr), ok
