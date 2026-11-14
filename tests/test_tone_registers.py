"""Three registers (professional / mid / casual) for every reply type: negotiation banks and template replies carry deterministic markers,
an elderly caller always gets 'aap' and no bhai, a mid tone mirrors a caller who says bhai, the respectful pack allows bhai/boss/yaar only in casual."""
import asyncio
import json
import re

import pytest

from acie.compiler import PromptCompiler
from acie.decision import negotiation
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.rules import tone as tone_rules
from acie.rules.respectful_register import RespectfulRegisterPack
from acie.schemas import ExpressionParams, SessionState, TurnInput, TurnPlan

RAW = json.load(open("data/voice/products.json"))
LANGS = ("en", "hi", "hinglish")
TONES = ("professional", "mid", "casual")
EDGE = r"(?<![\wऀ-ॿ])"
EDGE_END = r"(?![\wऀ-ॿ])"
PRO = re.compile(EDGE + r"(sir|ma'am|सर|मैम)" + EDGE_END, re.I)
CASUAL = re.compile(EDGE + r"(bhai|boss|yaar|भाई|बॉस|यार)" + EDGE_END, re.I)
DISRESPECT = re.compile(EDGE + r"(तू|तेरी|तेरा|तुझे|तुम|tu|teri|tera|tujhe|tum|abe|oye|बे|अबे|ओए)" + EDGE_END, re.I)


def gym(tone="mid"):
    return Product.model_validate({**RAW["fitzone-gym"], "tone_level": tone})


def check_tone(text, tone, ctx=""):
    assert not DISRESPECT.search(text), (ctx, text)
    if tone == "professional":
        assert PRO.search(text) and not CASUAL.search(text), (ctx, text)
    elif tone == "casual":
        assert CASUAL.search(text) and not PRO.search(text), (ctx, text)
    else:
        assert not PRO.search(text) and not CASUAL.search(text), (ctx, text)


SCENARIOS = {
    "first_no": ["bahut mehenga hai"],
    "first_no_number": ["2000 mein karo"],
    "concession": ["bahut mehenga hai", "aur kam karo"],
    "midpoint": ["bahut mehenga", "2500 mein karo"],
    "accept_counter": ["bahut mehenga", "3000 mein do"],
    "second_step": ["mehenga", "kam karo", "aur kam karo"],
    "final": ["kam karo"] * 5,
    "sweetener": ["kam karo"] * 6,
    "walk_away": ["kam karo"] * 7,
    "after_walk_away": ["kam karo"] * 8,
    "competitor": ["Cult 2,500 mein deta hai, aapka kyun lu"],
}


@pytest.mark.parametrize("tone", TONES)
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_negotiation_reply_carries_its_tone_in_all_three_languages(tone, name):
    p, st = gym(tone), negotiation.NegState()
    r = None
    for line in SCENARIOS[name]:
        r = negotiation.step(p, line, st, "", None, tone=tone)
        st = r.state
    assert r is not None and r.toned
    for l in LANGS:
        check_tone(r.text[l], tone, f"{name}/{l}")


def test_the_same_scenario_says_different_things_in_each_tone_and_it_is_deterministic():
    out = {}
    for tone in TONES:
        st, r = negotiation.NegState(), None
        for line in ["bahut mehenga", "aur kam karo"]:
            r = negotiation.step(gym(tone), line, st, "", None, tone=tone)
            st = r.state
        out[tone] = r.text["hinglish"]
    assert len(set(out.values())) == 3
    assert out["professional"].startswith("Sir") and "₹2,799" in out["mid"] and out["casual"].startswith(("Bhai", "Boss"))


def plan_intents(res):
    return [p.intent for p, _ in res]


def _talk(lines, tone, age=None, lang="hinglish"):
    async def go():
        dom = ProductSalesDomain(gym(tone), customer_name="Ravi", customer_age=age)
        pipe = ACIEPipeline(DecisionEngine(dom))
        T = lambda t: TurnInput(session_id="s", customer_id="+910000000000", text=t)
        sess = pipe.session(T(""))
        sess.expression = ExpressionParams(language=lang)
        await pipe.decision.decide("greet", T(""), sess)
        res = []
        for line in lines:
            r = await pipe.run_turn(T(line))
            res.append((r.plan, r.text))
        return res, sess
    return asyncio.run(go())


@pytest.mark.parametrize("tone", TONES)
def test_non_negotiation_replies_pitch_price_answer_goodbye_carry_the_tone(tone):
    res, _ = _talk(["haan batao", "kitna hai", "timings kya hain?", "nahi chahiye, bye"], tone)
    for plan, text in res:
        assert plan.tone_level == tone
        if plan.llm_allowed:   # an LLM-phrased reply gets the register as a style line (below); its template fallback carries it too
            for l in LANGS:
                check_tone(plan.fallback[l], tone, plan.intent + "/" + l)
        else:
            check_tone(text, tone, plan.intent)


@pytest.mark.parametrize("tone", TONES)
def test_the_opening_and_legal_lines_keep_their_words_whatever_the_tone(tone):
    res, _ = _talk(["tum AI ho kya?", "mera number kahan se mila"], tone)
    for plan, text in res:
        assert not DISRESPECT.search(text)
    assert plan_intents(res)[0] in ("are_you_ai", "identify") and re.search(r"\bAI\b", res[0][1])   # the disclosure keeps its words, no address lead added
    assert not PRO.search(res[0][1]) and not CASUAL.search(res[0][1])


def test_an_elderly_caller_always_gets_aap_and_no_bhai_even_in_casual():
    for kw in ({"age": 62}, {"age": None}):
        lines = ["haan batao", "kitna hai", "bahut mehenga hai", "aur kam karo"]
        if kw["age"] is None:
            lines[0] = "haan beta batao"
        res, sess = _talk(lines, "casual", **kw)
        assert sess.dialog.get("_elder") == "1"
        for plan, text in res[1:]:
            assert not CASUAL.search(text) and not DISRESPECT.search(text), text
            assert plan.tone_level == "mid"
    res, _ = _talk(["haan batao", "kitna hai", "bahut mehenga hai"], "professional", age=70)
    assert PRO.search(res[-1][1])   # formal stays formal for elders


def test_mid_drifts_casual_when_the_caller_says_bhai_and_stays_mid_otherwise():
    res, sess = _talk(["haan bhai batao", "kitna hai", "bahut mehenga hai"], "mid")
    assert sess.dialog.get("_mirror") == "1" and CASUAL.search(res[-1][1])
    res, sess = _talk(["haan batao", "kitna hai", "bahut mehenga hai"], "mid")
    assert not CASUAL.search(res[-1][1]) and not PRO.search(res[-1][1])
    # an elder who says bhai does not pull the register down
    d = {}
    assert tone_rules.effective_tone("mid", d, "beta bhai batao", 70) == "mid"


def test_effective_tone_is_pure_and_defaults_to_mid():
    assert tone_rules.effective_tone(None, {}, "") == "mid" and tone_rules.effective_tone("nonsense", {}, "") == "mid"
    assert tone_rules.effective_tone("casual", {}, "kitna", 54) == "casual" and tone_rules.effective_tone("casual", {}, "kitna", 55) == "mid"


def test_restyle_never_changes_the_content_only_adds_the_address():
    fb = {"en": "The offer is {offer}. Shall I send the link?", "hi": "ऑफर {offer} है।", "hinglish": "Offer {offer} hai. Link bhej doon?"}
    for tone in TONES:
        out = tone_rules.restyle(fb, tone)
        for l in LANGS:
            assert "{offer}" in out[l] and fb[l].split()[-1] in out[l]
    assert tone_rules.restyle(fb, "mid") == fb
    assert tone_rules.restyle(tone_rules.restyle(fb, "casual"), "casual") == tone_rules.restyle(fb, "casual")   # idempotent


PACK = RespectfulRegisterPack()


def _chk(sentence, tone):
    return PACK.check(sentence, TurnPlan(intent="x", objective="", tone_level=tone), ExpressionParams())


def test_respectful_pack_allows_bhai_boss_yaar_only_in_casual_and_always_blocks_tu_tum():
    for w in ("Bhai, kar deta hoon", "Boss, deal?", "theek hai yaar", "भाई, ठीक है", "यार, चलेगा"):
        assert _chk(w, "casual") is None
        assert _chk(w, "mid") and _chk(w, "professional") and _chk(w, "")
    for w in ("तू आ जा", "tum aa jao", "teri marzi", "Bhai, tu bata", "बॉस, तुम बताओ", "oye bhai"):
        assert _chk(w, "casual"), w
    assert PACK.check("Bhai, ok", TurnPlan(intent="x", objective=""), ExpressionParams(tone="casual")) is None   # the ExpressionParams knob works too


def test_compiler_adds_a_register_line_per_tone_and_nothing_by_default():
    sess = SessionState(session_id="s", customer_id="c")
    base = PromptCompiler().compile(TurnPlan(intent="x", objective="o"), ExpressionParams(), sess, "hi").system
    assert "Register:" not in base
    for tone, word in (("professional", "PROFESSIONAL"), ("mid", "MID"), ("casual", "CASUAL")):
        sys_ = PromptCompiler().compile(TurnPlan(intent="x", objective="o", tone_level=tone), ExpressionParams(), sess, "hi").system
        assert f"Register: {word}" in sys_
    sys_ = PromptCompiler().compile(TurnPlan(intent="x", objective="o"), ExpressionParams(tone="casual"), sess, "hi").system
    assert "Register: CASUAL" in sys_ and "never 'तू" in sys_
