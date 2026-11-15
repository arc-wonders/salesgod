"""Owner-call order (gate b8d5d84): per-day answer never a confirm, a named number is always a counter (even after terms_confirmed), no 'where we agreed'
without an agreement, abuse gets an apology in every tone, template glitches."""
import re

import pytest

from tests.test_haggle_gate import TONES, talk

SEQ = ["go ahead", "how much", "too costly", "can you do better", "can you do better",
       "uh sure so i can just pay 33 rupees per day rather than a full monthly payment", "okay 2,600 then, send the link",
       "So can't you just give me a good discount for me?", "What the fuck are you talking about?"]


@pytest.mark.parametrize("tone", TONES)
def test_owner_call_sequence(tone):
    out, sess = talk(SEQ, tone, lang="en")
    assert out[4][0] == "negotiation" and "₹2,649" in out[4][1]
    pd = out[5]                                                           # (a) the plain per-day answer, never confirm_close
    assert pd[0] == "negotiation" and re.search(r"one payment of ₹2,649 for 3 months", pd[1]) and "just the maths" in pd[1] or "arithmetic" in pd[1]
    ct = out[6]                                                           # (b) 2,600 >= floor and >= the next step 2,549: accepted at 2,600, link not sent
    assert ct[0] == "confirm_close" and "₹2,600" in ct[1] and "₹2,649" not in ct[1] and "?" in ct[1]   # agreed at their number, confirmed
    assert not any(i == "link_sent" for i, _ in out[:7])
    assert "₹2,600" in out[7][1] and not re.search(r"₹2,[0-5]\d\d", out[7][1])   # a push after agreement: the agreed price stays
    assert out[8][0] == "apology" and re.search(r"sorry", out[8][1], re.I) and "detail" not in out[8][1]


@pytest.mark.parametrize("tone", TONES)
def test_counter_after_terms_confirmed_is_still_a_counter_never_a_link_at_our_price(tone):
    out, sess = talk(["go ahead", "how much", "too costly", "can you do better", "can you do better", "ok fine, 2649 is okay", "okay 2,600 then, send the link"], tone, lang="en")
    assert out[5][0] == "confirm_close"
    last = out[6]   # 2,649 was already accepted: a lower number never reopens it
    assert last[0] == "negotiation" and "₹2,649" in last[1] and "₹2,600" not in last[1]


@pytest.mark.parametrize("tone", TONES)
def test_counter_never_meets_above_our_current_offer(tone):
    out, _ = talk(["go ahead", "how much", "too costly", "can you do better", "can you do better", "ok fine, 2649 is okay", "okay 2,450 then, send the link"], tone, lang="en")
    prices = [int(x.replace(",", "")) for x in re.findall(r"₹([\d,]+)", out[-1][1])]
    assert out[-1][0] != "link_sent" and prices and all(2400 <= x <= 2649 for x in prices)


@pytest.mark.parametrize("tone", TONES)
def test_abuse_gets_an_apology_in_every_tone_and_language(tone):
    for lang, line in (("en", "What the fuck are you talking about?"), ("hinglish", "ye kya bakwas hai bewakoof")):
        out, _ = talk(["go ahead", "how much", line], tone, lang=lang)
        assert out[-1][0] == "apology" and re.search(r"sorry|maaf", out[-1][1], re.I)


def test_template_glitches_are_forbidden_in_the_style_lines_and_the_address_lead_lowercases():
    from acie.compiler import PromptCompiler
    from acie.rules import tone as t
    from acie.schemas import ExpressionParams, SessionState, TurnPlan
    sys_ = PromptCompiler().compile(TurnPlan(intent="x", objective="o", tone_level="casual"), ExpressionParams(), SessionState(session_id="s", customer_id="c"), "hi").system
    assert "never in words" in sys_ and "membership membership" in sys_
    assert t.restyle_text("Just to confirm: it's ₹2,649.", "en", "professional") == "Sir, just to confirm: it's ₹2,649."
    assert t.address({"en": "Sir, just to confirm"}, "professional", "ji")["en"] == "Ji, just to confirm"
