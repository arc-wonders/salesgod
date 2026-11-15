"""Conversational style (acie/psyche/style.py): the hard limits from team/human/PLAN.md, the partial-utterance backchannel
and the narrowed anti-echo guard. Template backend, no network."""

import asyncio
import json
import re
from pathlib import Path

import pytest

from acie.psyche import style as S
from acie.psyche.ledger import CustomerFacts

ROOT = Path(__file__).resolve().parents[1]

CALM = {"frustration": 0.0, "engagement": 0.9, "trust": 0.6, "confusion": 0.0, "urgency": 0.0}


def _style(text="Nice, sounds good", row="calm", turn=5, dialog=None, state=None, **kw):
    return S.decide_style(row=row, state=state or CALM, text=text, intent=kw.pop("intent", "affirm"), turn_index=turn,
                          dialog=dialog if dialog is not None else {}, **kw)


def _dialog(turns=6, **b):
    d = {}
    bb = S.budgets(d)
    bb["turns"] = turns
    bb.update(b)
    S.save_budgets(d, bb)
    return d


# ------------------------------------------------------------------ humour hard limits

@pytest.mark.parametrize("row", ["crisis", "angry", "frustrated", "anxious", "sad", "confused", "overwhelmed", "suspicious"])
def test_no_humour_in_banned_rows(row):
    st = _style(row=row, dialog=_dialog())
    assert not st.humor_ok and st.lead_key != "HUMOR"
    assert not S.humour_allowed(st, row, False, 5, "Nice, sounds good", _dialog())


def test_humour_possible_when_calm_and_positive():
    st = _style(dialog=_dialog())
    assert st.lead_key == "HUMOR" and st.humor_ok


@pytest.mark.parametrize("kw", [dict(elder=True), dict(garbled=True), dict(turn=2), dict(text="Nice, but I can't afford it, lost my job")])
def test_no_humour_elder_garbled_early_hardship(kw):
    st = _style(dialog=_dialog(), **kw)
    assert st.lead_key != "HUMOR" and not st.humor_ok


def test_humour_budget_per_call_and_spacing():
    assert _style(dialog=_dialog(turns=8, humor_turns=[2, 7])).lead_key != "HUMOR"   # 2 per call max
    assert _style(dialog=_dialog(turns=8, humor_turns=[5])).lead_key != "HUMOR"      # <= 1 per 5 turns
    assert _style(dialog=_dialog(turns=8, humor_turns=[2])).lead_key == "HUMOR"


def test_no_humour_after_caller_was_upset():
    assert _style(dialog=_dialog(turns=8, upset_turns=[3])).lead_key != "HUMOR"


def test_humour_lines_never_about_person():
    for lang, lines in S.BANKS["HUMOR"].items():
        for line in lines:
            assert not S._NEVER_IN_HUMOR.search(line), line


# ------------------------------------------------------------------ question budgets

def test_never_two_asking_turns_in_a_row():
    d = _dialog(turns=4, q_turns=[4])
    assert not S.ask_budget_ok(S.budgets(d))
    assert not _style(text="okay", dialog=d, state={**CALM, "engagement": 0.3}).ask_ok
    assert S.ask_budget_ok(S.budgets(_dialog(turns=4, q_turns=[3])))


def test_discovery_budget():
    assert not S.discovery_ok(_dialog(turns=4, disc_turns=[4]))          # never 2 in a row
    assert not S.discovery_ok(_dialog(turns=5, disc_turns=[4]))          # nor with one turn between
    assert not S.discovery_ok(_dialog(turns=9, disc_turns=[1, 3, 6]))    # <= 3 per call
    assert S.discovery_ok(_dialog(turns=6, disc_turns=[3]))


def test_question_share_over_a_call():
    d = {}
    for i in range(10):
        st = _style(text="okay", dialog=d, state={**CALM, "engagement": 0.3}, turn=i + 1)
        S.note_reply(d, "Point." + (" Want more?" if st.ask_ok else ""))
    b = S.budgets(d)
    assert len(b["q_turns"]) <= b["turns"] / 2


def test_mirror_budget_and_cap():
    d = _dialog(turns=6)
    st = _style(text="I have to do 21 kilometer marathon", dialog=d, new_detail=["marathon"])
    assert st.lead_key == "MIRROR" and st.mirror == ["marathon"]
    assert len(S.mirror_words("I like power lifting a lot", ["power lift"])) <= 4
    assert S.mirror_words("I like power lifting a lot", ["power lift"]) == ["power", "lifting"]
    st2 = _style(text="I train mornings", dialog=_dialog(turns=6, mirror_turns=[5]), new_detail=["morning"])
    assert st2.lead_key != "MIRROR"


# ------------------------------------------------------------------ rows

def test_held_back_on_repeat_complaint():
    st = _style(text="I told you it was to run 21 km marathon.", dialog=_dialog())
    assert st.style == "held-back" and st.ask_ok and st.lead_key == "LABEL_UPSET"  # one question that hands control back


def test_price_objection_first_curious_then_calm():
    d = _dialog()
    st = _style(text="This is very expensive", dialog=d)
    assert st.row == 6 and st.lead_key == "LABEL_PRICE"
    d["_style_price_obj"] = "1"
    assert _style(text="Still too expensive", dialog=d).row == 7


def test_crisis_defers_to_safety():
    st = _style(row="crisis", dialog=_dialog())
    assert st.style == "safety" and st.lead_key == ""


# ------------------------------------------------------------------ partial utterances

@pytest.mark.parametrize("t", ["So basically,", "I was thinking here about...", "Majorly I am into lifting and also", "matlab mujhe ye chahiye ki",
                               "मुझे लगता है कि"])
def test_partial(t):
    assert S.is_partial(t)


@pytest.mark.parametrize("t", ["2000 rupees.", "Can you lower the price?", "No, thank you.", "I have trained before.", "Okay, bye.",
                               "I want it for 2,500 and", "मुझे join करना है कैसे करूँ?", "Sure, I'll just..."])
def test_not_partial(t):
    assert not S.is_partial(t)


# ------------------------------------------------------------------ banks: aap forms, gender

def test_banks_never_tu():
    for key, langs in S.BANKS.items():
        for lang, lines in langs.items():
            for line in lines:
                assert not re.search(r"(^|\s)(तू|तेरी|तेरा|तुम|tu|teri|tera|tum)(\s|[,.।!?]|$)", line, re.I), (key, line)


def test_bank_gender_follows_agent():
    assert "रही हूँ" in S.pick("GO_ON", "hi", 2, "female") or "सुन रही" in S.pick("GO_ON", "hi", 2, "female")
    assert "sun rahi" in S.pick("GO_ON", "hinglish", 2, "female")
    assert "सुन रहा" in S.pick("GO_ON", "hi", 2, "male")
    assert "kar leti" in S.pick("HUMOR", "hinglish", 0, "female")


# ------------------------------------------------------------------ anti-echo guard (narrowed)

def test_hook_with_callers_fact_is_allowed_but_news_and_parrot_are_not():
    f = CustomerFacts()
    f.learn_caller("I have to do 21 kilometer marathon for my running.")
    assert f.restates("Since you run marathons, the treadmills and the leg machines will build your stamina.") == []
    assert f.restates("I have to do 21 kilometer marathon for my running.")  # whole-sentence parrot
    assert f.restates("So you want to run a marathon.")  # their fact handed back as news


# ------------------------------------------------------------------ pipeline: never re-ask a known slot, backchannel

def _pipe():
    from acie.decision.domains.product_sales import Product, ProductSalesDomain
    from acie.decision.engine import DecisionEngine
    from acie.pipeline import ACIEPipeline
    raw = json.loads((ROOT / "data/voice/products.json").read_text())
    return ACIEPipeline(DecisionEngine(ProductSalesDomain(Product.model_validate(raw["fitzone-gym"]))))


def _run(lines, sid="style1"):
    from acie.schemas import TurnInput
    pipe = _pipe()
    T = lambda t: TurnInput(session_id=sid, customer_id="+910000000000", text=t)  # noqa: E731
    sess = pipe.session(T(""))

    async def go():
        op = await pipe.decision.decide("greet", T(""), sess)
        sess.context.record_turn(0, "", op.intent, False)
        return [(await pipe.run_turn(T(x))).text for x in lines]
    return asyncio.run(go()), sess


@pytest.mark.skipif(not (ROOT / "data/voice/products.json").exists(), reason="needs data/voice/products.json")
def test_marathon_goal_is_never_reasked():
    out, _ = _run(["Yeah, I have.", "What do you have in the gym?", "I have to do 21 kilometer marathon.", "Okay.", "Tell me more.", "Hmm okay."])
    for line in out[2:]:
        assert not re.search(r"fitness goal|main goal|losing weight, building muscle", line, re.I), line


@pytest.mark.skipif(not (ROOT / "data/voice/products.json").exists(), reason="needs data/voice/products.json")
def test_partial_gets_backchannel_not_a_fact():
    out, _ = _run(["Yeah, I have.", "Why should I join?", "So basically,"], sid="style2")
    assert out[-1] in sum(S.BANKS["GO_ON"].values(), []), out[-1]


@pytest.mark.skipif(not (ROOT / "data/voice/products.json").exists(), reason="needs data/voice/products.json")
def test_style_off_switch(monkeypatch):
    monkeypatch.setenv("ACIE_STYLE", "0")
    out, sess = _run(["Yeah, I have.", "So basically,"], sid="style3")
    assert "_style" not in sess.dialog
    assert out[-1] not in sum(S.BANKS["GO_ON"].values(), [])


# ------------------------------------------------------------------ length budget (owner: phone-call length)

def _cap_pipe():
    return _pipe()


def _fit(pipe, text, plan_intent="answer", customer="okay", decision=""):
    import os
    from acie.schemas import TurnPlan
    os.environ["ACIE_LENGTH_CAP"] = "1"  # the cap is off by default (owner, 15 Nov); these tests exercise the code
    plan = TurnPlan(intent=plan_intent, objective="x", decision=decision, llm_allowed=False, fallback={"en": text})
    cap = pipe._length_cap(plan, customer)
    os.environ.pop("ACIE_LENGTH_CAP", None)
    if cap is None:
        return text
    return pipe._fit_length(text, cap, {"w": 0, "s": 0})


LONG = ("At FitZone membership, you get unlimited group classes like yoga and zumba, plus one free personal trainer session every week "
        "to help with your plan and form. The gym is open from 5 am to 11 pm. What's your main goal right now: losing weight, building muscle, or staying fit?")


def test_normal_turn_capped_at_22_words_two_sentences():
    out = _fit(_cap_pipe(), LONG)
    assert len(out.split()) <= 22 and len(re.split(r"(?<=[.!?])\s+", out)) <= 2, out


def test_question_answer_capped_at_25_words():
    out = _fit(_cap_pipe(), LONG, customer="What do I get with the membership?")
    assert len(out.split()) <= 25, out


def test_ack_capped_at_6_words():
    out = _fit(_cap_pipe(), "Mm-hmm, go on, I am listening to everything you say.", plan_intent="acknowledge", decision="style:backchannel")
    assert len(out.split()) <= 6, out


def test_terms_kept_nothing_extra_after():
    t = ("Just to confirm: it's ₹2,999 for 3 months if you join this week, then ₹6,000 for 3 months, and you can cancel any time. "
         "It also comes with yoga and zumba classes. Shall I have our team send you the payment link?")
    out = _fit(_cap_pipe(), t, plan_intent="confirm_close")
    assert "₹2,999" in out and "₹6,000" in out and out.endswith("payment link?") and "zumba" not in out, out


def test_cut_never_inside_a_number():
    p = _cap_pipe()
    t = "For you it is ₹2,999 for 3 months if you join this week and that works out to roughly ₹33 a day which is less than a coffee for most people around here."
    out = _fit(p, t)
    assert "₹2,999" in out or out == t


def test_opening_states_the_purpose():
    from acie.schemas import TurnInput
    pipe = _pipe()
    T = TurnInput(session_id="op", customer_id="+910000000000", text="")
    plan = asyncio.run(pipe.decision.decide("greet", T, pipe.session(T)))
    assert "calling about an offer" in plan.fallback["en"] and "two minutes?" in plan.fallback["en"]


def test_plain_digits_do_not_bypass_the_cap_and_glued_question_after_price_goes():
    p = _cap_pipe()
    t = ("Ji, FitZone membership has treadmills and cardio machines to support your marathon preparation. "
         "Plus, you get unlimited group classes like yoga and zumba, open from 5 am to 11 pm every day.")
    assert len(_fit(p, t, customer="Do you have treadmills?").split()) <= 25
    t2 = ("Haan, powerlifting ke liye squat rack aur barbells hain, aur har hafte ek free trainer session milega. "
          "Is week join karne par 3 mahine ke ₹2,999, uske baad 3 mahine ke ₹6,000 chal raha hai, kya aapne pehle weight lifting ki hai ya fresh start kar rahe hain?")
    out = _fit(p, t2)
    assert "₹6,000" in out and "fresh start" not in out, out


def test_llm_reply_drops_recent_price_unless_asked():
    from acie.schemas import TurnPlan
    p = _pipe()
    d = {}
    S.note_reply(d, "It's ₹2,999 for 3 months if you join this week.")
    sess = type("Sx", (), {"dialog": d})()
    plan = TurnPlan(intent="answer", objective="x", llm_allowed=True, fallback={"en": "x"})
    reply = ["Treadmills are on the cardio floor. And it's ₹2,999 for 3 months."]
    assert p._style_drop_recent_price(reply, plan, sess, "Do you have treadmills?") == ["Treadmills are on the cardio floor."]
    assert p._style_drop_recent_price(reply, plan, sess, "How much is it again?") == reply
    assert p._style_drop_recent_price(reply, plan.model_copy(update={"intent": "negotiation"}), sess, "ok") == reply


def test_offer_slot_hidden_from_llm_on_non_price_turn():
    from acie.schemas import Fact, TurnPlan
    p = _pipe()
    plan = TurnPlan(intent="answer", objective="answer about equipment", llm_allowed=True, fallback={"en": "x"},
                    facts={"offer": Fact(value="₹2,999 for 3 months", description="offer")})
    assert p._offer_off_topic(plan, "Do you have treadmills?")
    assert not p._offer_off_topic(plan, "How much does it cost?")
    assert not p._offer_off_topic(plan.model_copy(update={"intent": "price"}), "ok")


def test_length_cap_off_by_default():
    from acie.schemas import TurnPlan
    p = _pipe()
    assert p._length_cap(TurnPlan(intent="answer", objective="x", llm_allowed=False, fallback={"en": "x"}), "ok") is None


def test_cap_keeps_the_closing_ask():
    out = _fit(_cap_pipe(), "FitZone is a gym with trainers and yoga and zumba classes, open from 5 in the morning to 11 at night. "
                            "Right now it's 50% off. Shall I tell you the price?", plan_intent="pitch")
    assert out.endswith("Shall I tell you the price?"), out


# ------------------------------------------------------------------ owner call f79a90af02 (07:02 IST): "too rude"

F79 = ["Uh, sure.", "Love it.",
       "I think you are being a bit too rude by just suddenly giving me all the price petitions and all. Did you even ask me if I want to join or not?",
       "Okay. Love it."]


@pytest.mark.skipif(not (ROOT / "data/voice/products.json").exists(), reason="needs data/voice/products.json")
def test_f79_pitch_asks_and_weak_affirm_gets_permission_not_price():
    out, _ = _run(F79[:2], sid="f79a")
    assert out[0].rstrip().endswith("?"), out[0]          # the pitch keeps its permission question
    assert "₹" not in out[1], out[1]   # "Love it." -> never the price dump
    assert sum(x.rstrip().endswith("?") for x in re.split(r"(?<=[.!?])\s+", out[1])) <= 1


def test_permission_first_logic():
    from acie.schemas import Utterance
    p = _pipe()
    sess = type("Sx", (), {})()
    sess.dialog = {}
    sess.history = [Utterance(role="agent", text="FitZone is a gym with trainers. Right now it's 50% off.")]
    assert p._permission_first(sess, "Love it.") and p._permission_first(sess, "Okay.")
    sess.history = [Utterance(role="agent", text="Right now it's 50% off. Shall I tell you the price?")]
    assert not p._permission_first(sess, "Okay.")          # a plain yes to our own price question is consent
    assert p._permission_first(sess, "Love it.")           # appreciation is not consent
    assert not p._permission_first(sess, "How much is it?")
    sess.dialog["_style_perm"] = "1"
    assert not p._permission_first(sess, "Love it.")       # asked once already: the next affirm gets the price


@pytest.mark.skipif(not (ROOT / "data/voice/products.json").exists(), reason="needs data/voice/products.json")
def test_f79_rude_complaint_owns_it_and_hands_back_control():
    out, _ = _run(F79[:3], sid="f79b")
    a = out[2]
    assert re.search(r"you're right|fair point|sorry", a, re.I) and a.rstrip().endswith("?") and len(a.split()) <= 22, a
    assert "₹" not in a


@pytest.mark.skipif(not (ROOT / "data/voice/products.json").exists(), reason="needs data/voice/products.json")
def test_price_turn_has_one_ask_and_a_soft_opener():
    out, _ = _run(["Uh, sure.", "Yes, tell me the price."], sid="f79c")
    a = out[1]
    assert "₹" in a and sum(x.rstrip().endswith("?") for x in re.split(r"(?<=[.!?])\s+", a)) <= 1, a
    assert not re.search(r"fitness goal|main goal", a, re.I), a
    assert re.match(r"(Sure|Of course|Fair enough|Right|Okay|Mm-hmm)\b", a), a


def test_handback_bank_aap_and_gender():
    for line in S.BANKS["HANDBACK"]["hi"] + S.BANKS["HANDBACK"]["hinglish"]:
        assert not re.search(r"(^|\s)(तू|तुम|tu|tum)(\s|$)", line)
    assert "बढ़ गई" in S.pick("HANDBACK", "hi", 0, "female") and "badh gayi" in S.pick("HANDBACK", "hinglish", 0, "female")
