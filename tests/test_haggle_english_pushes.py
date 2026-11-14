"""English / Hinglish discount asks (live call 1e2a92c585) reach the ladder like "kam karo": value + a step, never a canned fact; per-day asks are answered."""
import re

import pytest

from tests.test_haggle_gate import TONES, talk

EN = [
    "okay and i want a discount for joining as i'm leaving the current gym as well so i need a discount from you",
    "Not this offer but about discount offers.",
    "So can you just tell me about those offers yourself?",
    "can you do better",
    "any deals going on",
    "is that your best price",
    "this is too costly",
    "can you reduce it a little",
]
HL = ["mujhe discount chahiye, main dusra gym chhod raha hoon", "koi offer hai kya", "aur achha rate do", "best price kya hai", "thoda sasta kar do",
      "student discount milega kya"]
CANNED = re.compile(r"cancel any time|kabhi bhi cancel|every week|har hafte|free personal trainer session every", re.I)


@pytest.mark.parametrize("tone", TONES)
@pytest.mark.parametrize("line,lang", [(l, "en") for l in EN] + [(l, "hinglish") for l in HL])
def test_discount_asks_enter_the_ladder_value_first_then_a_step(tone, line, lang):
    out, sess = talk(["haan batao" if lang != "en" else "go ahead", "kitna hai", line, line], tone, lang=lang)
    first, second = out[2], out[3]
    assert first[0] == "negotiation" and "₹2,999" not in first[1].split("?")[0] or first[0] == "negotiation"
    assert "₹2,799" not in first[1]                                   # first push: value, no cut yet
    assert not CANNED.fullmatch(first[1].strip()) and ("?" in first[1])
    assert second[0] in ("negotiation", "price_floor") and "₹2,799" in second[1] and "?" in second[1]   # second push: the first step with a closing ask
    assert float(sess.dialog.get("closed_price") or 0) == 0


@pytest.mark.parametrize("tone", TONES)
def test_per_day_versus_full_payment_is_one_payment_for_the_period_and_not_a_concession(tone):
    for line in ("uh sure so i can just pay 33 rupees per day rather than a full monthly payment", "roz ke 33 rupaye de doon, ek saath ki jagah?"):
        out, _ = talk(["go ahead", "how much", line], tone)
        intent, text = out[2]
        assert re.search(r"one payment|ek hi payment|ek hi pay", text, re.I) and re.search(r"maths|hisaab|arithmetic", text, re.I), text
        assert "₹2,999" in text and "₹33" in text and "₹2,799" not in text


@pytest.mark.parametrize("tone", TONES)
def test_switching_gym_gets_value_then_a_step_not_a_canned_fact(tone):
    out, _ = talk(["go ahead", "how much", "i'm leaving the current gym, so i need a discount", "come on, a discount please"], tone, lang="en")
    assert "?" in out[2][1] and not CANNED.search(out[2][1].split("?")[0][:60])
    assert "₹2,799" in out[3][1]


@pytest.mark.parametrize("tone", TONES)
def test_cant_you_give_me_a_good_discount_enters_the_ladder_once_safety_passes_it(tone):
    from acie.decision import negotiation
    from tests.test_haggle_ladder import GYM
    st, got = negotiation.NegState(), []
    for _ in range(2):
        r = negotiation.step(GYM, "So can't you just give me a good discount for me?", st, "", None, tone=tone)
        st = r.state
        got.append(r)
    assert got[0].offer is None and "?" in got[0].text["en"] and got[1].offer == {"price": 2799.0}


@pytest.mark.parametrize("tone", TONES)
@pytest.mark.parametrize("line,lang", [("okay 2,600 then, send the link", "en"), ("okay 2,600 then, send the link", "hinglish"), ("chalo 2600 mein link bhej do", "hinglish"),
                                       ("done, 2600, bhej do link", "hinglish")])
def test_a_counter_with_consent_is_never_replaced_by_our_higher_price(tone, line, lang):
    out, sess = talk(["go ahead" if lang == "en" else "haan batao", "how much", "too costly", "can you do better", line], tone, lang=lang)
    intent, text = out[-1]
    assert intent == "negotiation" and "₹2,600" in text or intent == "negotiation" and re.search(r"₹2,[67]\d\d", text), text   # accepted at 2,600 or met in the middle
    assert "₹2,799 for 3 months" not in text and intent != "confirm_close" and intent != "link_sent"
    assert "?" in text
    assert float(sess.dialog.get("closed_price") or 0) == 0
