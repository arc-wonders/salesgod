from collections.abc import AsyncIterator

from acie.decision.domains.bank_demo import BankDemoDomain
from acie.decision.engine import DecisionEngine
from acie.personalization.policy import AdaptationPolicy, detect_explicit_requests
from acie.personalization.profile_store import MemoryProfileStore, update_preference
from acie.pipeline import ACIEPipeline
from acie.schemas import (CustomerProfile, ExpressionParams, Fact, HumanState, SessionState,
                          TurnInput, TurnPlan, TurnTiming, Utterance)
from acie.state.estimator import HumanStateEstimator
from acie.text import detect_language, inr, parse_amount
from acie.validator import StreamingValidator


def T(text, sid="s1", cid="c1", **timing):
    return TurnInput(session_id=sid, customer_id=cid, text=text, timing=TurnTiming(**timing))


# ------------------------------------------------------------------ text utils

def test_amounts_and_inr():
    assert parse_amount("50,000 bhejna hai") == 50_000
    assert parse_amount("send 50k") == 50_000
    assert parse_amount("1.5 lakh transfer") == 150_000
    assert parse_amount("20 hazaar bhejo") == 20_000
    assert parse_amount("no number here") is None
    assert inr(1250000) == "₹12,50,000"
    assert inr(40000) == "₹40,000"


def test_language_detection():
    assert detect_language("मेरा बैलेंस बताओ") == "hi"
    assert detect_language("mera balance kitna hai") == "hinglish"
    assert detect_language("what is my balance") == "en"


# ------------------------------------------------------------------ state estimator

def test_state_rises_fast_decays_slow():
    est = HumanStateEstimator()
    sess = SessionState(session_id="s", customer_id="c")
    s1 = est.estimate(T("kitni baar bolna padega, bakwas service!!"), sess)
    assert s1.frustration >= 0.5
    sess.state = s1
    s2 = est.estimate(T("ok check my balance"), sess)
    assert 0 < s2.frustration < s1.frustration  # decays, doesn't vanish
    assert s2.frustration > 0.3


def test_repetition_raises_confusion():
    est = HumanStateEstimator()
    sess = SessionState(session_id="s", customer_id="c",
                        history=[Utterance(role="customer", text="what is my transfer limit")])
    s = est.estimate(T("what is my transfer limit"), sess)
    assert s.confusion > 0.1 and "customer repeated previous turn" in s.signals


# ------------------------------------------------------------------ personalization

def test_explicit_preference_locks():
    p = CustomerProfile(customer_id="c")
    update_preference(p, "language", "hi", explicit=True)
    for _ in range(10):
        update_preference(p, "language", "en")
    assert p.preferences["language"].value == "hi"


def test_implicit_preference_needs_evidence_to_flip():
    p = CustomerProfile(customer_id="c")
    for _ in range(5):
        update_preference(p, "language", "en")
    update_preference(p, "language", "hi")
    assert p.preferences["language"].value == "en"


def test_language_hysteresis():
    pol = AdaptationPolicy()
    sess = SessionState(session_id="s", customer_id="c", turn_index=1,
                        expression=ExpressionParams(language="en"))
    prof = CustomerProfile(customer_id="c")
    a = pol.decide("balance batao", HumanState(), prof, sess, {})
    assert a.language == "en"  # one short Hinglish turn isn't enough
    a = pol.decide("aur limit kitni", HumanState(), prof, sess, {})
    assert a.language == "hinglish"


def test_full_sentence_switches_language_immediately():
    pol = AdaptationPolicy()
    sess = SessionState(session_id="s", customer_id="c", turn_index=1, expression=ExpressionParams(language="hi"))
    a = pol.decide("what is my balance now", HumanState(), CustomerProfile(customer_id="c"), sess, {})
    assert a.language == "en"


def test_explicit_language_request_is_immediate():
    assert detect_explicit_requests("please hindi mein baat karo") == {"language": "hi"}


def test_frustration_ack_has_cooldown():
    pol = AdaptationPolicy()
    sess = SessionState(session_id="s", customer_id="c")
    prof = CustomerProfile(customer_id="c")
    angry = HumanState(frustration=0.8)
    acks = []
    for i in range(4):
        sess.turn_index = i
        acks.append(pol.decide("x", angry, prof, sess, {}).acknowledge_emotion)
    assert acks == [True, False, False, True]


# ------------------------------------------------------------------ validator

PLAN = TurnPlan(intent="t", objective="o",
                facts={"daily_limit": Fact(value="₹40,000", description="d"),
                       "remaining_limit": Fact(value="₹25,000", description="r")},
                fallback={"en": "Limit {daily_limit}, left {remaining_limit}."})


def run_validator(chunks):
    v = StreamingValidator(PLAN, ExpressionParams())
    out = []
    for c in chunks:
        out += v.feed(c)
    out += v.finish()
    return out, v.report()


def test_validator_fills_slots_streaming():
    out, rep = run_validator(["Your daily limit is {daily", "_limit}. Today ", "{remaining_limit} is left."])
    assert out == ["Your daily limit is ₹40,000.", "Today ₹25,000 is left."]
    assert rep.ok and not rep.used_fallback


def test_validator_blocks_hallucinated_number():
    out, rep = run_validator(["Your limit is ₹50,000 per day."])
    assert rep.used_fallback and out == ["Limit ₹40,000, left ₹25,000."]


def test_validator_blocks_invented_slot_and_missing_fact():
    _, rep = run_validator(["Your {interest_rate} is low."])
    assert not rep.ok and rep.used_fallback
    _, rep = run_validator(["Your limit is {daily_limit}."])
    assert any("missing facts" in i for i in rep.issues)


# ------------------------------------------------------------------ end to end

def make_pipe(backend=None):
    return ACIEPipeline(DecisionEngine(BankDemoDomain()), backend=backend, store=MemoryProfileStore())


async def test_transfer_over_limit_then_accept_remaining():
    pipe = make_pipe()
    r = await pipe.run_turn(T("I want to transfer 50,000"))
    assert r.plan.intent == "transfer_over_limit"
    assert "₹40,000" in r.text and "₹25,000" in r.text
    r = await pipe.run_turn(T("yes please"))
    assert r.plan.intent == "transfer_done" and "₹25,000" in r.text


async def test_ask_amount_followup():
    pipe = make_pipe()
    r = await pipe.run_turn(T("paise bhejna hai"))
    assert r.plan.intent == "transfer_ask_amount"
    r = await pipe.run_turn(T("10000"))
    assert r.plan.intent == "transfer_confirm"


async def test_card_decline_is_respected():
    pipe = make_pipe()
    await pipe.run_turn(T("tell me about the credit card"))
    r = await pipe.run_turn(T("no not interested"))
    assert r.plan.intent == "card_declined"
    r = await pipe.run_turn(T("what about that card"))
    assert r.plan.intent == "card_declined_again"


async def test_clarify_reexplains_last_answer():
    pipe = make_pipe()
    await pipe.run_turn(T("what is my transfer limit"))
    r = await pipe.run_turn(T("samajh nahi aaya"))
    assert r.plan.intent == "transfer_limit" and r.expression.complexity == "low"


async def test_hindi_request_switches_language_and_persists():
    store = MemoryProfileStore()
    pipe = ACIEPipeline(DecisionEngine(BankDemoDomain()), store=store)
    r = await pipe.run_turn(T("hindi mein bolo, balance kitna hai"))
    assert r.expression.language == "hi" and "बैलेंस" in r.text
    prof = await store.load("c1")
    assert prof.preferences["language"].explicit
    r = await pipe.run_turn(T("what is my balance", sid="s2"))  # new call, still Hindi
    assert r.expression.language == "hi"


class HallucinatingLLM:
    async def stream(self, prompt, plan, expr) -> AsyncIterator[str]:
        for tok in ["Sure! ", "Your balance is ", "₹1,00,000. "]:
            yield tok


class BrokenLLM:
    async def stream(self, prompt, plan, expr) -> AsyncIterator[str]:
        raise ConnectionError("vllm down")
        yield ""  # pragma: no cover


async def test_hallucinating_llm_is_caught():
    r = await make_pipe(HallucinatingLLM()).run_turn(T("balance batao"))
    assert "₹82,500" in r.text and "1,00,000" not in r.text
    assert r.validation.used_fallback


async def test_llm_outage_falls_back():
    r = await make_pipe(BrokenLLM()).run_turn(T("what is my balance"))
    assert "₹82,500" in r.text and r.validation.used_fallback


# ------------------------------------------------------------------ regressions found by the eval harness

async def test_new_amount_while_pending_requotes_not_executes():
    pipe = make_pipe()
    r = await pipe.run_turn(T("पचास हज़ार भेजना है"))
    assert r.plan.intent == "transfer_over_limit"
    r = await pipe.run_turn(T("haan das hazaar bhej do"))  # must NOT execute the pending ₹25,000
    assert r.plan.intent == "transfer_confirm" and "₹10,000" in r.text
    r = await pipe.run_turn(T("हां"))
    assert r.plan.intent == "transfer_done" and "₹10,000" in r.text


async def test_amount_after_declined_transfer_continues_flow():
    pipe = make_pipe()
    await pipe.run_turn(T("fifteen thousand transfer karna hai", cid="c2"))  # c2 balance ₹12,000
    r = await pipe.run_turn(T("hmm ok ten thousand then", cid="c2"))
    assert r.plan.intent == "transfer_confirm"


async def test_topic_change_is_not_card_consent():
    pipe = make_pipe()
    await pipe.run_turn(T("tell me about the credit card"))
    r = await pipe.run_turn(T("ok then balance please"))
    assert r.plan.intent == "check_balance"


def test_neutral_engagement_is_not_disengaged():
    pol = AdaptationPolicy()
    sess = SessionState(session_id="s", customer_id="c")
    a = pol.decide("what is my balance", HumanState(engagement=0.5), CustomerProfile(customer_id="c"), sess, {})
    assert a.verbosity == "med"


def test_clarify_intent_lowers_complexity():
    pol = AdaptationPolicy()
    sess = SessionState(session_id="s", customer_id="c")
    a = pol.decide("which limit", HumanState(), CustomerProfile(customer_id="c"), sess, {}, intent="clarify")
    assert a.complexity == "low"


def test_validator_blocks_amount_in_words_and_wrong_script():
    _, rep = run_validator(["Your limit is forty thousand rupees."])
    assert rep.used_fallback
    v = StreamingValidator(PLAN, ExpressionParams(language="en"))
    out = v.feed("आपकी लिमिट {daily_limit} है। ") + v.finish()
    assert v.report().used_fallback and out == ["Limit ₹40,000, left ₹25,000."]


def test_validator_accepts_value_copied_verbatim():
    plan = TurnPlan(intent="greet", objective="o", facts={"name": Fact(value="Rahul", description="n")},
                    fallback={"en": "Hello {name}."})
    v = StreamingValidator(plan, ExpressionParams())
    out = v.feed("Hello Rahul, how can I help? ") + v.finish()
    assert out == ["Hello Rahul, how can I help?"] and not v.report().used_fallback


async def test_greeting_plus_request_is_the_request():
    r = await make_pipe().run_turn(T("नमस्ते, मुझे पचास हजार रुपे ट्रांसफर करने है"))
    assert r.plan.intent == "transfer_over_limit"


class PartlyBadLLM:
    async def stream(self, prompt, plan, expr):
        for tok in ["Your daily limit is {daily_limit}. ", "And forty thousand is the cap."]:
            yield tok


async def test_buffered_reply_never_mixes_llm_and_template():
    pipe = ACIEPipeline(DecisionEngine(BankDemoDomain()), backend=PartlyBadLLM(), buffer_reply=True)
    spoken = []
    r = await pipe.run_turn(T("what is my transfer limit"), on_sentence=spoken.append)
    assert r.validation.used_fallback and len(spoken) == 1 and spoken[0].count("₹40,000") == 1


def test_validator_tolerates_currency_before_slot_and_emoji():
    out, rep = run_validator(["Limit ₹{daily_limit} 😊, left {remaining_limit}."])
    assert out == ["Limit ₹40,000 , left ₹25,000."] and rep.ok


async def test_distress_stops_business_and_gives_helpline():
    pipe = make_pipe()
    await pipe.run_turn(T("tell me about the credit card"))
    r = await pipe.run_turn(T("कि मेरे मरने का मन कर रहा है"))
    assert r.plan.intent == "safety_distress" and not r.plan.end_call  # stays on the line with care
    assert "14416" in r.text and "112" in r.text
    r2 = await make_pipe().run_turn(T("honestly i feel like dying"))
    assert r2.plan.intent == "safety_distress"
    r3 = await make_pipe().run_turn(T("the price is killing me, too costly"))
    assert r3.plan.intent != "safety_distress"


def test_validator_drops_cut_off_sentence():
    v = StreamingValidator(PLAN, ExpressionParams())
    out = v.feed("Your limit is {daily_limit} and {remaining_limit} is left. And also you can") + v.finish()
    assert out == ["Your limit is ₹40,000 and ₹25,000 is left."]


async def test_distress_is_sticky_for_the_rest_of_the_call():
    pipe = make_pipe()
    await pipe.run_turn(T("मेरे मरने का मन कर रहा है"))
    r = await pipe.run_turn(T("haan theek hai credit card batao"))
    assert r.plan.intent == "safety_followup" and "14416" in r.text


async def test_distress_followup_says_number_slowly_and_goodbye():
    pipe = make_pipe()
    await pipe.run_turn(T("मेरे मरने का मन कर रहा है"))
    r = await pipe.run_turn(T("नंबर धीरे धीरे बोलो, लिख लेते हैं"))
    assert "एक, चार" in r.text and not r.plan.end_call
    r = await pipe.run_turn(T("लिख लिया बेटा, अच्छा रखते हैं फ़ोन"))
    assert r.plan.intent == "safety_goodbye" and r.plan.end_call
