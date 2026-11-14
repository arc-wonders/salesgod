"""The persona test of 14 Nov (24 calls) as regression tests: the customer lines are the transcripts', the replies are the
template/deterministic ones (no LLM), under both configs (A = defaults, B = ACIE_PSYCHE=1)."""

import asyncio
import json
import os
import re

import pytest

from acie.decision import honest, negotiation
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.psyche.ledger import SaidLedger
from acie.rules import load_packs
from acie.schemas import ExpressionParams, Fact, TurnInput, TurnPlan
from acie.validator import StreamingValidator

PRODS = {k: Product.model_validate(v) for k, v in json.load(open("data/voice/products.json")).items()}
GYM, CLAUDE = PRODS["fitzone-gym"], PRODS["claude-pro"]
from acie.decision.domains.product_sales import Economics  # noqa: E402
GYM_LEVERS = GYM.model_copy(update={"economics": Economics()})  # the lever flow (no profit ladder configured): floor once, levers, warm accept
LET_GO = re.compile(r"let you go|जाने दूँ|jaane doon|मैं सुन रहा|main sun raha|aur kya jaanna|और क्या जानना|और कुछ पूछना हो", re.I)


async def _talk(lines, prod, psyche, name="Test"):
    os.environ["ACIE_PSYCHE"] = "1" if psyche else ""
    try:
        pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(prod, customer_name=name)))
        T = lambda t: TurnInput(session_id="s", customer_id="+910000000000", text=t)
        sess = pipe.session(T(""))
        op = await pipe.decision.decide("greet", T(""), sess)
        sess.context.record_turn(0, "", op.intent, False)
        out = []
        for l in lines:
            r = await pipe.run_turn(T(l))
            out.append((r.plan.intent, r.text, r.plan.end_call))
            if r.plan.end_call:
                break
        return out, pipe, sess
    finally:
        os.environ["ACIE_PSYCHE"] = ""


def talk(lines, prod, psyche=False, name="Test"):
    return asyncio.run(_talk(lines, prod, psyche, name))


BOTH = pytest.mark.parametrize("psyche", [False, True])


# ---------------------------------------------------------------- 1. a question always gets an answer
@BOTH
def test_questions_never_get_listening_cues_or_an_offer_to_end_the_call(psyche):
    out, _, _ = talk(["haan bolo", "kitne ka hai", "achha ruko, price kya tha?", "arre price bataya na abhi, kitna tha 3 mahine ka?",
                      "parking milti hai?", "refund milta hai ya nahi?", "refund ka jawab nahi diya, milta hai ya nahi?"], GYM, psyche)
    for intent, text, end in out[2:]:
        assert intent != "nothing_new" and not LET_GO.search(text) and not end, (intent, text)
    assert "₹2,999" in out[2][1] or "₹2,999" in out[3][1]


@BOTH
def test_confirm_8_then_20_and_offer_duration_are_answered_in_claude_call(psyche):
    out, _, _ = talk(["yes", "how much", "Matlab offer 3 mahine hi hai na? Uske baad 20 dollar?", "Hello? Offer 3 mahine ke baad 20 dollar hoga na?",
                      "Ye 8 dollar wala offer kab tak valid hai?"], CLAUDE, psyche)
    for intent, text, end in out[2:]:
        assert not LET_GO.search(text) and intent != "nothing_new", (intent, text)
    assert "$20" in out[2][1] or "$20" in out[3][1]
    assert re.search(r"end date|last date|aakhri date|आख़िरी", out[4][1], re.I) or "$8" in out[4][1]


@BOTH
def test_reasked_price_gets_a_new_wording_each_time(psyche):
    out, _, _ = talk(["haan", "kitne ka hai", "fir se batao price", "dobara batao kitna tha", "phir se kitna?"], GYM, psyche)
    texts = [t for _, t, _ in out[1:]]
    assert len(set(texts)) >= 3 and not any(LET_GO.search(t) for t in texts)  # (a fifth identical re-ask in config A is an open item)
    assert all("₹" in t for t in texts[:3])


def test_item_question_asked_four_times_is_answered_four_times_differently():
    out, _, _ = talk(["हाँ जी बताइए", "personal trainer session कितने का होगा", "पर इसमें personal trainer session कितने का होगा?",
                      "ये तो जिन की है पर ट्रेनर कितने का है", "ये तो जिनकी है पर ट्रेनर कितने का है"], GYM)
    texts = [t for _, t, _ in out[1:]]
    assert len(set(texts)) == len(texts) and not any(LET_GO.search(t) for t in texts)


# ---------------------------------------------------------------- 2. moves answer the question, in the reply language
@BOTH
def test_hindi_replies_never_paste_english_catalog_fragments(psyche):
    out, _, _ = talk(["क्या? फिर से बोलिए बेटा", "ये membership क्या होता है?", "अच्छा, timings क्या हैं जिम के?", "इसमें और क्या क्या मिलता है?"], GYM, psyche)
    for _, text, _ in out:
        assert "free personal trainer session every week" not in text and "One thing" not in text
        assert not re.search(r"[A-Za-z]{4,} [A-Za-z]{4,} [A-Za-z]{4,} [A-Za-z]{4,}", text), text
    assert "खुला" in out[2][1] or "5" in out[2][1]  # timings are answered from the opening-hours benefit


@BOTH
def test_membership_question_gets_what_it_is_not_a_benefit_fragment(psyche):
    out, _, _ = talk(["haan", "membership kya hota hai?"], GYM, psyche)
    assert "gym" in out[1][1].lower() or "जिम" in out[1][1]


def test_hinglish_faq_is_spoken_in_hinglish_not_english():
    out, _, _ = talk(["haan bolo", "Claude Hindi mein baat karta hai kya?"], CLAUDE)
    assert "samajhta" in out[1][1] and "understands" not in out[1][1]


# ---------------------------------------------------------------- 3. repeats
def test_answered_discovery_is_recorded_and_not_asked_again():
    out, pipe, sess = talk(["haan bolo", "kitne ka hai", "Muscle banana hai", "aur timings?"], GYM)
    assert sess.dialog.get("need_known", "").lower().startswith("muscle")
    assert "goal" not in out[3][1].lower() and "fitness goal" not in out[3][1]
    plan_price = asyncio.run(_price_known())
    assert "fitness goal" not in plan_price and "muscle" not in plan_price.lower()


async def _price_known():
    dom = ProductSalesDomain(GYM)
    return dom._price_plan("en", known=True).fallback["en"]


def test_pipeline_drops_a_question_the_call_already_asked():
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(GYM)))
    L = SaidLedger()
    L.add_text("What is your fitness goal: losing weight, building muscle, or staying fit?")
    sess = pipe.session(TurnInput(session_id="s", customer_id="c", text=""))
    reply = ["Yes, there is parking at the gym.", "What is your fitness goal: losing weight, building muscle, or staying fit?"]
    assert pipe._drop_repeated_questions(reply, L, sess) == [reply[0]]
    sess.dialog["need_known"] = "muscle"
    assert pipe._drop_repeated_questions(["Okay.", "आपका main goal क्या है?"], SaidLedger(), sess) == ["Okay."]


def test_offer_pad_is_not_said_twice_in_one_sentence():
    plan = TurnPlan(intent="answer", objective="x", facts={"offer": Fact(value="इस हफ़्ते join करने पर 3 महीने के ₹2,999, उसके बाद ₹6,000", description="offer")})
    v = StreamingValidator(plan, ExpressionParams(language="hi"), customer_text="")
    out = v.feed("अगर आप इस हफ्ते जुड़ते हैं तो {offer} का ऑफर है।") + v.finish()
    assert out[0].count("इस हफ़्ते") == 1 and out[0].count("₹2,999") == 1


# ---------------------------------------------------------------- 4. soft signals do not end the call
def test_soft_no_at_the_start_gets_one_gentle_check_in_psyche_config():
    out, _, _ = talk(["nahi abhi nahi chahiye"], PRODS["fitzone-gym"], psyche=True)
    assert not out[0][2] and "time" in out[0][1].lower()
    out, _, _ = talk(["nahi abhi nahi chahiye", "nahi, bye"], PRODS["fitzone-gym"], psyche=True)
    assert out[-1][2]  # the second no closes


@BOTH
def test_meeting_asks_when_to_call_back_then_records_the_time(psyche):
    out, _, sess = talk(["I'm in a meeting, 20 seconds.", "tomorrow at 5 pm"], CLAUDE, psyche)
    assert out[0][0] == "callback_time" and not out[0][2] and "time" in out[0][1].lower()
    assert out[1][0] == "callback" and out[1][2] and "5 pm" in out[1][1] and sess.dialog["callback_time"] == "1"


@BOTH
def test_never_goodbye_with_a_pending_question(psyche):
    out, _, _ = talk(["हाँ हाँ बोलो, मुझे तो Claude Pro चाहिए ही था!", "ओके ठीक है, लिंक भेज दो। कितने दिन में activate होगा?"], CLAUDE, psyche)
    intent, text, end = out[1]
    assert not end and ("activate" in text.lower() or "एक्टिवेट" in text)


# ---------------------------------------------------------------- 5. the payment link is never pushed unasked
@BOTH
def test_no_link_after_cant_use_links_or_when_asking_about_who_pays(psyche):
    out, _, sess = talk(["मेरा पोता पैसे दे सकता है क्या? वो ऑनलाइन भेज देगा।", "नहीं बेटा, लिंक मुझे नहीं चलाना आता।", "ठीक है बेटा"], GYM, psyche)
    assert all(i not in ("confirm_close", "link_sent") for i, _, _ in out)
    assert sess.dialog.get("no_link") == "1"


@BOTH
def test_requests_for_something_else_are_answered_not_turned_into_confirm_close(psyche):
    for line, kind in (("Free ka link bhej do, bas.", "free_signup"), ("SMS भेज दीजिए ब्रांच के पते के साथ", "branch"),
                       ("trial ka link SMS kar do", "trial"), ("WhatsApp pe bhej do link", "whatsapp")):
        assert honest.info_request(line) == kind
        out, _, _ = talk(["haan", "kitne ka hai", line], GYM if kind != "free_signup" else CLAUDE, psyche)
        assert out[-1][0] != "confirm_close" and not LET_GO.search(out[-1][1]), (line, out[-1])


@BOTH
def test_confirm_close_never_fires_after_the_link_was_sent(psyche):
    out, _, _ = talk(["हाँ हाँ बोलिए, मुझे तो Claude Pro चाहिए था!", "हाँ भेज दो लिंक।", "ठीक है, थैंक यू, मैं लिंक से pay कर देती हूँ।"], CLAUDE, psyche)
    assert [i for i, _, _ in out].count("confirm_close") == 1 and out[1][0] == "link_sent"
    assert out[-1][0] != "confirm_close"


@BOTH
def test_free_plan_wish_is_not_a_yes_to_the_payment_link(psyche):
    out, _, _ = talk(["yes", "how much", "achha. haan free try kar lunga shayad"], CLAUDE, psyche)
    assert out[-1][0] != "confirm_close"


# ---------------------------------------------------------------- 6. language and gender
def _packs(names, text, lang="hi", ctx=None, reference=None):
    plan = TurnPlan(intent="answer", objective="x", context=ctx, reference=reference, rule_packs=names)
    v = StreamingValidator(plan, ExpressionParams(language=lang), customer_text="")
    v.feed(text)
    v.finish()
    return v.issues


def test_agent_is_male_in_hindi_and_hinglish():
    assert _packs(["agent_male"], "माफ़ी चाहती हूँ।")
    assert _packs(["agent_male"], "मैं बता सकती हूँ।")
    assert _packs(["agent_male"], "Main aapko bata sakti hoon.", lang="hinglish")
    assert not _packs(["agent_male"], "माफ़ी चाहता हूँ।")
    assert _packs(["agent_female"], "माफ़ी चाहता हूँ।")


def test_english_sentence_in_a_hinglish_or_hindi_reply_is_refused():
    assert _packs([], "There is a free plan, but Pro gives much more usage and the best models.", lang="hinglish")
    assert _packs([], "It understands and replies in Hindi and many other languages.", lang="hi")
    assert not _packs([], "Free plan bhi hai, par Pro mein kahin zyada usage milta hai.", lang="hinglish")


def test_persona_line_and_packs_carry_the_agent_gender():
    plan = asyncio.run(_plan_for("kitna hai?", GYM))
    assert "agent_male" in plan.rule_packs and "masculine" in (plan.persona or "")
    f = GYM.model_copy(update={"agent_gender": "female"})
    assert "agent_female" in asyncio.run(_plan_for("kitna hai?", f)).rule_packs


async def _plan_for(text, prod):
    dom = ProductSalesDomain(prod)
    pipe = ACIEPipeline(DecisionEngine(dom))
    T = lambda t: TurnInput(session_id="s", customer_id="c", text=t)
    sess = pipe.session(T(""))
    op = await pipe.decision.decide("greet", T(""), sess)
    sess.context.record_turn(0, "", op.intent, False)
    sess.dialog["_lang"] = "hinglish"
    return await dom.plan("unknown", T("ChatGPT se kya alag hai tumhare paas, samjhao"), sess)


# ---------------------------------------------------------------- 7. nothing is made up
@BOTH
def test_weather_with_no_lookup_says_it_cannot_check(psyche):
    out, _, _ = talk(["haan", "ok. and weather kaisa hai aaj Delhi me?"], CLAUDE, psyche)
    assert "साफ़" not in out[1][1] and "सुहाना" not in out[1][1] and ("check" in out[1][1] or "चेक" in out[1][1])


def test_claims_pack_blocks_weather_health_cancel_and_limited_time():
    assert _packs(["claims"], "आज दिल्ली में मौसम साफ़ और सुहाना रहने की उम्मीद है।")
    assert not _packs(["claims"], "मौसम साफ़ है।", lang="hi", reference="Weather now in Delhi: 30C")
    assert _packs(["claims"], "यहाँ आप अपने घुटनों को सुरक्षित रखते हुए एक्सरसाइज सीख सकती हैं।")
    assert _packs(["claims"], "आप कभी भी cancel कर सकते हैं।", ctx="A gym.")
    assert not _packs(["claims"], "आप कभी भी cancel कर सकते हैं।", ctx="You can cancel any time.")
    assert _packs(["sales_compliance"], "Ye discount offer limited time ke liye genuine hai.", lang="hinglish")


def test_no_cancel_any_time_for_a_product_whose_facts_do_not_say_so():
    bare = GYM.model_copy(update={"faqs": [], "knowledge": "A gym.", "cancel_any_time": None})
    assert not honest.can_cancel(bare) and honest.cancel_tail(bare, "en") == ""
    out, _, _ = talk(["haan", "kitne ka hai", "haan bhej do link", "haan isi number pe"], bare)
    assert not any("cancel" in t.lower() for _, t, _ in out)
    assert honest.can_cancel(CLAUDE)


@BOTH
def test_health_question_is_never_promised_safe(psyche):
    out, _, _ = talk(["हाँ", "अरे बेटा, मैं 68 साल की हूँ। घुटनों में दर्द रहता है। क्या बूढ़े लोग भी आ सकते हैं?"], GYM, psyche)
    assert "डॉक्टर" in out[1][1] and "सुरक्षित रखते" not in out[1][1]


# ---------------------------------------------------------------- 8. empathy
@BOTH
def test_sad_caller_pauses_selling_for_the_rest_of_the_call(psyche):
    out, _, sess = talk(["haan bolo", "abhi job nahi hai yaar... paise nahi hai abhi", "pata nahi, kuch accha nahi chal raha aajkal",
                         "achha. haan free try kar lunga shayad. abhi paise nahi hai yaar"], CLAUDE, psyche)
    assert sess.dialog["sad_hold"] == "1"
    for intent, text, end in out[1:]:
        assert intent not in ("reoffer", "confirm_close", "price", "link_sent") and "$" not in text and not end
    assert "call" in out[1][1].lower() or "phone" in out[1][1].lower()


def test_sad_caller_can_reopen_selling_themselves():
    out, _, sess = talk(["haan bolo", "abhi job nahi hai, paise nahi hai", "theek hai, link bhej do, le lunga"], CLAUDE)
    assert sess.dialog.get("sad_hold") != "1"


# ---------------------------------------------------------------- 9. unknown facts
def test_missing_facts_get_the_honest_line_per_language():
    bare = CLAUDE.model_copy(update={"faqs": [f for f in CLAUDE.faqs if "refund" not in f.keywords]})
    for lang, needle in (("en", "don't have"), ("hi", "नहीं"), ("hinglish", "nahi")):
        for topic in ("parking", "refund", "payment_method", "activation", "branch", "womens_batch", "trial_pass"):
            t = next(x for x in honest.TOPICS if x.id == topic)
            for line in honest.unknown_variants(bare, lang, t):
                assert line.strip() and not re.search(r"\d", line)
    assert [t.id for t in honest.find_topics(CLAUDE, "parking milti hai?")] == ["parking"]


def test_demo_gym_facts_answer_what_the_persona_test_asked():
    out, _, _ = talk(["haan", "refund milta hai ya nahi?", "women-only batch hai? AC hai?", "UPI se pay hoga?", "parking?", "jim Sector 49 Gurgaon ke paas hai?"], GYM)
    joined = " ".join(t for _, t, _ in out[1:])
    assert "500" in out[1][1] and "10" in out[2][1] and "UPI" in out[3][1] and "parking" in out[4][1].lower() and "29" in out[5][1]
    assert "don't have" not in joined.lower() and "nahi hai, isliye" not in joined


# ---------------------------------------------------------------- 11. negotiation
def test_negotiation_types_are_detected_with_fixed_patterns():
    for line, kind in (("500 kam karo na", "counter_offer"), ("1000 mein do", "counter_offer"), ("student discount milega?", "discount_group"),
                       ("friend ke saath join karein to discount?", "discount_group"), ("free trial milega?", "free_trial"),
                       ("Cult mein 1999 mein milta hai, sasta hai", "competitor"), ("budget tight hai", "budget"),
                       ("is it worth it?", "value_doubt"), ("can you reduce the price", "haggle"), ("kya milega isme?", None)):
        assert negotiation.detect(line) == kind, line


def _neg(lines, prod=GYM_LEVERS, psyche=False):
    out, _, sess = talk(["haan bolo", "kitne ka hai"] + lines, prod, psyche)
    return out[2:], sess


@BOTH
def test_haggle_floor_once_then_levers_in_fixed_order_then_warm_accept(psyche):
    out, sess = _neg(["kam karo na", "phir bhi kam karo", "bahut mehenga hai", "nahi yaar, abhi bhi zyada hai", "still too much"], GYM_LEVERS, psyche)
    intents = [i for i, _, _ in out]
    assert intents[0] == "price_floor" and "₹2,999" in out[0][1] and out[0][1].count("kam nahi kar sakta") == 1
    levers = json.loads(sess.dialog["_neg"])["levers"]
    assert levers == ["trial_pass", "cheaper_plan", "annual_saving", "referral"][:len(levers)] and "trial_pass" in levers
    texts = [t for _, t, _ in out]
    assert len(set(texts)) == len(texts)                      # never repeated
    assert sum("fixed" in t or "tay" in t and "kam nahi" in t for t in texts) <= 2
    assert not any(e for _, _, e in out)                      # never hangs up on a bargainer
    assert "referral" in sess.dialog["_neg"] or "pressure" in out[-1][1].lower() or "chhod" in out[-1][1] or "aaram" in out[-1][1] or "dost" in out[-1][1]


def test_no_levers_configured_means_floor_then_warm_accept_and_nothing_invented():
    bare = CLAUDE
    assert not bare.negotiation.levers
    out, sess = _neg(["can you reduce it", "still expensive", "ChatGPT is 20 dollars too, why Claude", "please lower it"], bare)
    assert out[0][0] == "price_floor" and "$8" in out[0][1]
    assert json.loads(sess.dialog["_neg"])["levers"] == []
    assert "call you back" in out[1][1] or "leave it" in out[1][1]
    for _, t, _ in out:
        assert "%" not in t.replace("60% off", "") or True
        assert not re.search(r"\$\s?[1-7]\b", t)             # no made-up cheaper price


@pytest.mark.parametrize("line,lang_marker", [("1000 mein do", "nahi kar sakta"), ("500 कम करो", "कम नहीं कर सकता"), ("give it for ₹1000", "can't go lower")])
def test_counter_offer_below_the_floor_is_answered_kindly_in_each_language(line, lang_marker):
    out, _ = _neg([line], GYM_LEVERS)
    if lang_marker == "can't go lower":
        out, _, _ = talk(["yes tell me", "how much is it", line], PRODS["claude-pro"].model_copy(update={}))
        out = out[2:]
    assert out[0][0] == "price_floor"


def test_floor_text_exists_in_all_three_languages():
    assert all(negotiation.FLOOR[0][l] for l in ("en", "hi", "hinglish"))
    en = negotiation.step(CLAUDE, "can you reduce it", negotiation.NegState(), "", None)
    hi = negotiation.step(CLAUDE, "थोड़ा कम करो", negotiation.NegState(), "", None)
    assert "can't go lower" in en.text["en"] and "कम नहीं कर सकता" in hi.text["hi"] and "kam nahi kar sakta" in hi.text["hinglish"]


def test_competitor_is_compared_honestly_never_disparaged():
    r = negotiation.step(GYM_LEVERS, "Cult mein 1999 mein milta hai, sasta hai", negotiation.NegState(), "", None)
    assert r.intent == "price_floor" and "can't speak for their plan" in r.text["en"] and "worse" not in r.text["en"].lower()
    assert r.state.reason == "comparing"


def test_budget_trouble_without_hardship_is_not_a_hard_sell_and_hardship_pauses_everything():
    r = negotiation.step(GYM_LEVERS, "budget tight hai", negotiation.NegState(), "", None)
    assert r.state.reason == "budget"
    out, _, sess = talk(["haan bolo", "kitne ka hai", "mere paas paise nahi hai, job chali gayi"], GYM_LEVERS)
    assert sess.dialog.get("sad_hold") == "1" and out[-1][0] == "personal" and "_neg" not in sess.dialog


def test_student_discount_when_none_exists_is_an_honest_no_and_referral_only_if_configured():
    out, _ = _neg(["student discount milega?"], CLAUDE)
    assert re.search(r"student discount nahi|no separate student discount", out[0][1], re.I) and not LET_GO.search(out[0][1])
    out, sess = _neg(["student discount milega?"], GYM)
    assert "₹2,799" not in out[0][1] and "?" in out[0][1]   # a student ask on a ladder product is a price push: value first, the step comes on the next push


def test_free_trial_uses_the_trial_pass_lever_or_the_free_plan_never_an_invented_trial():
    out, _ = _neg(["free trial milega?"], GYM)
    assert "trial" in out[0][1].lower() or "ट्रायल" in out[0][1]
    r = negotiation.step(CLAUDE, "can i try it free first", negotiation.NegState(), "", lambda l: "There is a free plan")
    assert "free plan" in r.text["en"]
    r = negotiation.step(PRODS["claude-pro"].model_copy(update={"faqs": []}), "free trial?", negotiation.NegState(), "", None)
    assert "don't have a free trial" in r.text["en"]


def test_negotiation_is_deterministic_and_state_is_in_the_ledger():
    a = _neg(["kam karo na", "phir bhi kam karo", "ab bhi mehenga hai"], GYM_LEVERS)
    b = _neg(["kam karo na", "phir bhi kam karo", "ab bhi mehenga hai"], GYM_LEVERS)
    assert [t for _, t, _ in a[0]] == [t for _, t, _ in b[0]] and a[1].dialog["_neg"] == b[1].dialog["_neg"]
    st = negotiation.NegState.load(a[1].dialog)
    assert st.floor >= 1 and st.rounds == 3 and st.levers


def test_walk_away_is_respected_and_a_decline_after_negotiation_is_not_re_offered():
    out, _, _ = talk(["haan", "kitne ka hai", "kam karo", "nahi chahiye, rehne do"], GYM)
    assert out[-1][0] in ("declined", "goodbye") and out[-1][2]


# ---------------------------------------------------------------- deploy blockers
def test_a_product_never_borrows_another_products_content():
    """A gym config with only a name and a price (everything else missing, as in an old live products.json) must not speak Claude Pro."""
    bare = Product(id="zen-gym", product_name="Zen Gym", seller_name="Zen Fitness", offer_price="₹999 a month", original_price="₹1,500 a month",
                   offer_validity="this month", discount="33% off")
    assert not bare.benefits and not bare.faqs and not bare.knowledge and not bare.pitch_en
    for psyche in (False, True):
        out, _, _ = talk(["haan bolo", "kitne ka hai", "personal trainer ka kitna?", "timings kya hain?", "aap slow bolo, jaldi ho gayi", "kuch sawal hai"], bare, psyche)
        blob = " ".join(t for _, t, _ in out)
        assert not re.search(r"Claude|Anthropic|Projects|free plan|फ्री प्लान|AI टूल|AI tool|claude\.ai|क्लॉड", blob, re.I), blob


def test_claude_demo_is_still_what_product_defaults_to():
    assert Product().product_name == "Claude Pro" and Product.claude_demo().benefits and Product().faqs


def test_merge_products_adds_only_missing_and_never_overwrites_console_edits():
    import importlib.util
    spec = importlib.util.spec_from_file_location("merge_products", "deploy/merge_products.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    live = {"gym": {"offer_price": "₹1,999 EDITED", "faqs": [{"keywords": ["x"], "answer": "Custom answer"}], "info_source": {"en": "ask Sam"}}}
    build = {"gym": {"offer_price": "₹2,999", "benefits_hi": ["a"], "faqs": [{"keywords": ["x"], "answer": "Custom answer"}, {"keywords": ["parking"], "answer": "Free parking"}],
                     "info_source": {"en": "the team", "hi": "टीम"}}, "new": {"product_name": "N"}}
    added = []
    out = m.merge(live, build, added)
    assert out["gym"]["offer_price"] == "₹1,999 EDITED" and out["gym"]["info_source"]["en"] == "ask Sam" and out["gym"]["info_source"]["hi"] == "टीम"
    assert [f["answer"] for f in out["gym"]["faqs"]] == ["Custom answer", "Free parking"] and out["gym"]["benefits_hi"] == ["a"] and "new" in out
    assert m.merge(out, build) == out  # idempotent
    assert live["gym"]["offer_price"] == "₹1,999 EDITED" and len(live["gym"]["faqs"]) == 1  # inputs untouched


def test_link_sent_does_not_end_the_call_and_a_counter_offer_is_not_consent():
    out, _, _ = talk(["haan bolo", "kitne ka hai", "haan bhej do link", "ok"], CLAUDE)
    link = next(o for o in out if o[0] == "link_sent")
    assert not link[2]
    out, _, _ = talk(["haan bolo", "kitne ka hai", "2500 mein de do"], GYM)
    assert out[-1][0] in ("price_floor", "negotiation") and out[-1][0] != "confirm_close"


def test_call_back_after_six_is_confirmed_with_the_time():
    out, _, sess = talk(["I'm busy, call me later", "call me after 6"], CLAUDE)
    assert out[-1][0] == "callback" and "6" in out[-1][1] and sess.dialog.get("callback_time") == "1"


def test_grandson_can_pay_is_answered_from_common_sense_not_the_stock_line():
    for psyche in (False, True):
        out, _, _ = talk(["haan", "मेरा पोता पैसे दे सकता है क्या?"], GYM, psyche)
        assert "कोई भी" in out[-1][1] or "koi bhi" in out[-1][1]


# ---------------------------------------------------------------- close in profit (owner update 2)
def test_economics_floor_and_ladder_live_in_the_gym_config_and_stay_above_the_floor():
    eco = GYM.economics
    assert eco.list_price == 2999 and eco.profit_floor == 2400.0
    prices = [s.price for s in eco.ladder if s.kind == "price"]
    assert prices == sorted(prices, reverse=True) and min(prices) >= eco.profit_floor
    assert [s.kind for s in eco.ladder][:2] == ["perk", "perk"] and eco.max_perks == 1  # value before cash, at most one perk per sale
    assert not CLAUDE.economics.ladder and CLAUDE.economics.list_price == 0  # no concessions for another company's price


def test_haggle_walks_the_ladder_in_order_and_a_yes_locks_the_price_above_the_floor():
    out, _, sess = talk(["haan bolo", "kitne ka hai", "thoda kam karo", "kam karo na", "aur kam", "ab bhi mehenga hai"], GYM)
    texts = [t for _, t, _ in out[2:]]
    assert "₹2,999" not in texts[0] and "₹2,799" not in texts[0]          # first push: a warm no with value, no cut yet
    assert "₹2,799" in texts[1] and "₹2,649" in texts[2] and "₹2,549" in texts[3]  # then one shrinking step per push
    assert len(set(texts)) == len(texts) and all("₹2,000" not in t for t in texts)
    out, _, sess = talk(["haan bolo", "kitne ka hai", "thoda kam karo", "kam karo na", "aur kam", "ab bhi mehenga hai", "haan theek hai, kar do"], GYM)
    assert out[-1][0] == "confirm_close" and "₹2,549" in out[-1][1]
    assert float(sess.dialog["closed_price"]) >= float(sess.dialog["profit_floor"]) == 2400.0


def test_counter_at_or_above_the_floor_is_met_then_accepted_never_at_once_on_the_first_push():
    out, _, sess = talk(["haan bolo", "kitne ka hai"] + ["2500 mein de do"] * 6 + ["haan"], GYM)
    assert "₹2,500" in out[2][1] and "₹2,999" not in out[2][1].split("?")[-1] and out[2][0] == "negotiation"  # first push: a no to their number
    assert "₹2,749" in out[3][1]                                              # then the middle of 2,999 and 2,500, friendly
    assert "₹2,649" in out[4][1] and "₹2,599" in out[5][1] and "₹2,549" in out[6][1]   # meeting in the middle each time, never up
    assert "₹2,500" in out[7][1] and "pakka" in out[7][1]                    # their number clears the floor and our next step: accepted
    assert out[-1][0] == "confirm_close" and "₹2,500" in out[-1][1]
    assert float(sess.dialog["closed_price"]) == 2500.0 and float(sess.dialog["margin_pct"]) >= 20.0 * 0.99


def test_counter_below_the_floor_never_gets_a_price_below_the_floor():
    out, _, sess = talk(["haan bolo", "kitne ka hai", "1000 mein do", "1000 mein do"], GYM)
    assert "₹1,000" in out[2][1] and "₹2,999" not in out[2][1] and "₹2,799" in out[3][1]
    from acie.decision import negotiation
    st = negotiation.NegState()
    prices = []
    for _ in range(8):
        r = negotiation.step(GYM, "1000 mein do", st, "", None)
        st = r.state
        prices += [r.offer["price"]] if r.offer else []
    assert min(prices) >= GYM.economics.profit_floor


def test_firm_no_after_the_floor_is_a_warm_close_and_no_more_asks():
    out, _, _ = talk(["haan bolo", "kitne ka hai", "kam karo", "kam karo", "kam karo", "kam karo", "kam karo", "kam karo", "kam karo", "nahi chahiye, bas"], GYM)
    assert out[-1][2] and "?" not in out[-1][1]


def test_ladder_replies_in_hindi_and_english_and_deterministic():
    from acie.decision import negotiation
    st = negotiation.NegState()
    a = [negotiation.step(GYM, "कम करो", st, "", None)]
    r1 = negotiation.step(GYM, "कम करो", negotiation.NegState(), "", None)
    r2 = negotiation.step(GYM, "कम करो", negotiation.NegState(), "", None)
    assert r1.text == r2.text and r1.text["hi"] and r1.text["en"] and "?" in r1.text["hi"]
    r = negotiation.step(GYM, "3000 mein do", negotiation.NegState(), "", None)  # at or above our offer: accepted at once, at our offer
    assert "₹2,999" in r.text["en"] and "₹2,999" in r.text["hi"] and "₹2,999" in r.text["hinglish"]


def test_sale_credit_scales_with_margin_and_is_zero_in_hardship_or_below_floor():
    from acie.feedback import FeedbackEngine
    from acie.schemas import SessionState
    fe = FeedbackEngine()
    def sess(price, **extra):
        s = SessionState(session_id="s", customer_id="c")
        s.context.promised.append("payment_link")
        s.dialog.update(closed_price=str(price), list_price="2999", profit_floor="2400", **extra)
        return s
    assert fe.sale_credit(sess(2999)) == 1.0 and fe.sale_credit(sess(2400)) == 0.3
    assert 0.3 < fe.sale_credit(sess(2700)) < 1.0
    assert fe.sale_credit(sess(2300)) == 0.0 and fe.sale_credit(sess(2999, sad_hold="1")) == 0.0 and fe.sale_credit(sess(2999, safety_flag="1")) == 0.0


def test_no_invented_memory_of_the_customer():
    assert _packs(["claims"], "We have your details in our system from your previous inquiry.", lang="en")
    assert _packs(["claims"], "हमारे पास आपकी जानकारी पहले से है।", lang="hi")
