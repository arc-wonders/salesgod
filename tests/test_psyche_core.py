"""Psyche layer: needs table, move selection without repeats, emotion table, and the real-call bug fixes."""

import asyncio
import json
import os

from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.psyche import emotion_policy
from acie.psyche.ledger import SaidLedger
from acie.psyche.moves import MoveContext, choose_move
from acie.psyche.needs import NeedInput, estimate_need, table_need
from acie.schemas import HumanState, TurnInput

FIT = Product.model_validate(json.load(open("data/voice/products.json"))["fitzone-gym"])


def need(text, **state):
    return table_need(NeedInput(text=text, state=state)).need


def test_need_table_rows():
    assert need("नहीं चाहिए") == "autonomy"
    assert need("call me later") == "autonomy"
    assert need("bye") == "closure"
    assert need("hello", frustration=0.8) == "be_heard"
    assert need("अरे बार बार एक चीज रिपेट क्यों कर रहा है?") == "be_heard"
    assert need("is it fake? why so cheap") == "reassurance"
    assert need("I don't understand Hindi") == "clarity"
    assert need("what do you have in the gym?") == "answer"
    assert need("I'm busy") == "space"
    assert need("my son is studying engineering") == "connection"
    assert need("yes tell me") == "information"
    assert need("प्रेम प्रेम प्रेम प्रेम प्रेम प्रेम प्रेम प्रेम") == "clarity"
    # price negotiation with "नहीं" is not a no
    assert need("आज बोरा सा कम नहीं कर सकते हैं") == "answer"


def test_table_is_deterministic_and_model_never_overrides_autonomy():
    a = estimate_need(NeedInput(text="नहीं चाहिए"))
    b = estimate_need(NeedInput(text="नहीं चाहिए"))
    assert a == b and a.need == "autonomy" and a.source == "table"


def mc(text, lang="en", ledger=None, **kw):
    n = table_need(NeedInput(text=text, state=kw.pop("state", {})))
    L = ledger or SaidLedger()
    L.add_move("need:" + n.need)
    return MoveContext(need=n, lang=lang, product=FIT, ledger=L, text=text, **kw), L


def test_information_moves_never_repeat_and_run_out_honestly():
    L = SaidLedger()
    said = []
    for _ in range(14):
        ctx, L = mc("yes tell me more", ledger=L)
        m = choose_move(ctx)
        line = m.line("en")
        assert not line or not any(line == s for s in said)
        said.append(line)
        L.add_text(line)
        L.add_slot(m.slot)
        L.add_move(m.name)
        if m.end_call:
            break
    assert any("everything I have" in s for s in said) and said[-1] != said[-2]


def test_no_selling_move_for_autonomy_space_reassurance_be_heard_connection():
    for text, st in (("no thanks", {}), ("I'm busy", {}), ("is it fake?", {}), ("this is annoying", {"frustration": 0.9}),
                     ("my son is studying", {})):
        ctx, _ = mc(text, state=st)
        m = choose_move(ctx)
        assert not m.sells and "link" not in m.line("en").lower().replace("a link by sms", "") or not m.sells


def test_one_gentle_offer_after_interest_never_after_a_no():
    L = SaidLedger()
    L.add_slot("offer")
    ctx, L = mc("yes go ahead", ledger=L, engagement=0.7)
    ctx.need.flags["go_on"] = True
    m = choose_move(ctx)
    assert m.sells and "only if you want" in m.line("en")
    L2 = SaidLedger()
    L2.add_slot("offer")
    L2.refused = True
    ctx2, _ = mc("yes go ahead", ledger=L2, engagement=0.7)
    ctx2.need.flags["go_on"] = True
    assert not choose_move(ctx2).sells


def test_clarity_ladder_escalates():
    L = SaidLedger()
    names = []
    for _ in range(4):
        ctx, L = mc("हुआ है हुआ है हुआ है हुआ है हुआ है हुआ है", ledger=L)
        m = choose_move(ctx)
        names.append(m.name)
        L.add_text(m.line("en"))
    assert names[0] == "ask_repeat" and names[1] == "offer_stop" and names[2] == "keep_listening"


# ---------------------------------------------------------------- emotion policy
def test_emotion_rows_priority_and_crisis():
    assert emotion_policy.decide(HumanState(), "I want to die").row == "crisis"
    assert emotion_policy.decide(HumanState(), "hi", flagged=True).row == "crisis"
    assert emotion_policy.decide(HumanState(frustration=0.8), "x").row == "angry"
    assert emotion_policy.decide(HumanState(frustration=0.5), "x").row == "frustrated"
    assert emotion_policy.decide(HumanState(anxiety=0.7), "x").row == "anxious"
    assert emotion_policy.decide(HumanState(confusion=0.7), "x").row == "confused"
    assert emotion_policy.decide(HumanState(cognitive_load=0.8), "x").row == "overwhelmed"
    assert emotion_policy.decide(HumanState(trust=0.2), "x").row == "suspicious"
    assert emotion_policy.decide(HumanState(urgency=0.7, engagement=0.2), "x").row == "busy"
    assert emotion_policy.decide(HumanState(excitement=0.8), "x").row == "excited"
    assert emotion_policy.decide(HumanState(), "x").row == "calm"
    s = emotion_policy.decide(HumanState(), "kill myself")
    assert not s.continue_business and s.sticky and s.acknowledge


def test_voice_arousal_makes_angry():
    class V:
        arousal, confidence, calibrated = 0.8, 0.9, True
    assert emotion_policy.decide(HumanState(frustration=0.45), "x", voice=V()).row == "angry"


# ---------------------------------------------------------------- pipeline: real-call bugs
def make(flag):
    os.environ["ACIE_PSYCHE"] = "1" if flag else ""
    return ACIEPipeline(DecisionEngine(ProductSalesDomain(FIT)))


async def talk(pipe, lines):
    T = lambda t: TurnInput(session_id="s", customer_id="+910000000000", text=t)
    sess = pipe.session(T(""))
    op = await pipe.decision.decide("greet", T(""), sess)
    sess.context.record_turn(0, "", op.intent, False)
    return [await pipe.run_turn(T(l)) for l in lines]


def test_trainer_price_asked_four_times_never_gets_membership_price_or_repeats():
    for flag in (False, True):
        rs = asyncio.run(talk(make(flag), ["हाँ जी बताइए", "personal trainer session कितने का होगा", "पर इसमें personal trainer session कितने का होगा?",
                                         "ये तो जिन की है पर ट्रेनर कितने का है", "ये तो जिनकी है पर ट्रेनर कितने का है"]))
        texts = [r.text for r in rs[1:]]
        assert "₹2,999" not in texts[0] and "free personal trainer" in texts[0]
        assert len(set(texts)) == len(texts)
    os.environ["ACIE_PSYCHE"] = ""


def test_price_negotiation_with_nahi_is_not_a_decline():
    for flag in (False, True):
        rs = asyncio.run(talk(make(flag), ["हाँ जी बताइए", "कीमत बताइए", "आज बोरा सा कम नहीं कर सकते हैं"]))
        assert rs[-1].plan.intent != "declined" and not rs[-1].plan.end_call
    os.environ["ACIE_PSYCHE"] = ""


def test_repeated_garbage_never_repeats_the_same_line(monkeypatch):
    rs = asyncio.run(talk(make(False), ["हेलो", "हुआ है", "हुआ है", "हुआ है", "हुआ है"]))
    texts = [r.text for r in rs]
    from scripts.real_calls.common import count_repetitions
    assert count_repetitions(texts) == 0


def test_crisis_row_stays_on_by_default_and_is_sticky():
    os.environ["ACIE_PSYCHE"] = ""
    rs = asyncio.run(talk(make(False), ["मेरे मरने का मन कर रहा है", "ठीक है"]))
    assert rs[0].plan.intent == "safety_distress" and rs[1].plan.intent.startswith("safety")


def test_psyche_replay_is_deterministic():
    a = [r.text for r in asyncio.run(talk(make(True), ["हेलो", "नहीं चाहिए"]))]
    b = [r.text for r in asyncio.run(talk(make(True), ["हेलो", "नहीं चाहिए"]))]
    os.environ["ACIE_PSYCHE"] = ""
    assert a == b


def test_exhaustion_never_ends_the_call():
    """Live replay 9bde114312: running out of fresh wording hung up on a caller with ten questions left."""
    from acie.psyche.moves import MoveContext, _exhausted
    from acie.psyche.ledger import SaidLedger
    from acie.psyche.needs import NeedResult
    L = SaidLedger()
    seen = []
    for _ in range(12):
        mc = MoveContext(need=NeedResult(need="answer", rule="t"), ledger=L, lang="en")
        m = _exhausted(mc, "test")
        assert not m.end_call and m.text["en"]
        seen.append(m.name)
        L.add_move(m.name) if hasattr(L, "add_move") else None
        L.add_text(m.text["en"])
    assert "keep_listening" in seen and "listen" in seen


def test_same_sentence_never_spoken_twice_in_one_turn():
    """Live call 7abf60f9bd: the lead and the safe template carried the same answer."""
    import asyncio
    from acie.pipeline import ACIEPipeline
    from acie.decision.engine import DecisionEngine
    from acie.schemas import TurnInput, TurnPlan
    plan = TurnPlan(intent="faq", objective="x", llm_allowed=False,
                    lead={"en": "Fair question. It is a launch offer."},
                    fallback={"en": "Fair question. It is a launch offer. Anything else?"})

    class D:
        name = "t"
        class classifier:
            @staticmethod
            def classify(text, session):
                return "unknown"
        async def plan(self, intent, turn, session):
            return plan
    out = []
    p = ACIEPipeline(DecisionEngine(D()), buffer_reply=True)
    asyncio.run(p.run_turn(TurnInput(session_id="s", customer_id="c", text="is it real?"), on_sentence=out.append))
    spoken = " ".join(out)
    assert spoken.count("Fair question.") == 1 and "Anything else?" in spoken
