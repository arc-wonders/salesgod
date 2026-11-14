"""Follow-up call: continuity from the customer's previous call record(s), open questions first, agreed price honoured."""

import json
import time

import pytest

from acie.calltypes.memory import from_record, load_previous_calls
from tests.calltypes_helpers import Call, run

PREV = {
    "call_id": "old1", "to": "+919990001111", "customer_id": "cust-7", "customer_name": "Yash", "campaign": "claude-pro", "created_at": 1_790_000_000.0,
    "turns": [
        {"role": "agent", "text": "Namaste", "intent": "greet"},
        {"role": "customer", "text": "abhi busy hoon, kal shaam 6 baje call karna"},
        {"role": "agent", "text": "Zaroor", "intent": "callback"},
        {"role": "agent", "text": "Link bhej raha hoon, ₹2,999 pehle 3 mahine", "intent": "link_sent"},
    ],
    "context": {"promised": ["payment_link", "callback_later"], "quoted": ["offer_price"], "told": ["what_the_product_is", "who_is_calling"],
                "questions": [{"q": "kya isme data privacy hai?", "answer": "clarify"}, {"q": "kitne ka hai", "answer": "price"}],
                "objections": ["price"], "decision": "later", "outcome": "later"},
}


def seed(tmp_path, rec=None, name="old1"):
    (tmp_path / "calls").mkdir(exist_ok=True)
    (tmp_path / "calls" / f"{name}.json").write_text(json.dumps(rec or PREV, ensure_ascii=False))
    return tmp_path


def follow(tmp_path, **kw):
    return Call("follow_up", data_dir=tmp_path, customer_id="cust-7", phone="+919990001111", call_id="new1", **kw)


# --------------------------------------------------------------------------- memory

def test_memory_from_record():
    m = from_record(PREV)
    assert m.link_sent and m.agreed_price == "₹2,999" and m.objections == ["price"]
    assert m.open_questions == ["kya isme data privacy hai?"]  # the price question was answered; the clarify one was not
    assert m.callback_text.startswith("abhi busy") and m.callback_at.endswith("T18:00:00+05:30")
    assert m.told == ["what_the_product_is", "who_is_calling"]


def test_memory_reads_a_call_type_result_too():
    rec = {"call_id": "x", "created_at": 1.0, "turns": [{"role": "agent", "text": "hi", "intent": "greet"}],
           "context": {"result": {"outcome": "callback_booked", "next_action": {"type": "follow_up_call", "due_at": "2026-11-16T18:00:00+05:30"},
                                  "facts": {"agreed_price": "₹1,999", "open_questions": "is there a refund?|what about GST?"}}}}
    m = from_record(rec)
    assert m.outcome == "callback_booked" and m.callback_at == "2026-11-16T18:00:00+05:30" and m.agreed_price == "₹1,999"
    assert m.open_questions == ["is there a refund?", "what about GST?"]


def test_load_matches_customer_or_phone_newest_first_and_never_the_placeholder(tmp_path):
    seed(tmp_path)
    older = {**PREV, "call_id": "old0", "created_at": 1_780_000_000.0, "customer_id": "cust-7", "context": {"outcome": "declined"}}
    seed(tmp_path, older, "old0")
    other = {**PREV, "call_id": "oth", "customer_id": "cust-9", "to": "+910000000009"}
    seed(tmp_path, other, "oth")
    got = load_previous_calls(tmp_path, customer_id="cust-7")
    assert [g.call_id for g in got] == ["old1", "old0"]
    assert [g.call_id for g in load_previous_calls(tmp_path, phone="+919990001111")] == ["old1", "old0"]
    assert load_previous_calls(tmp_path, customer_id="c1") == []  # CallRecord's default id is not a customer
    assert [g.call_id for g in load_previous_calls(tmp_path, customer_id="cust-7", exclude_call_id="old1")] == ["old0"]
    assert load_previous_calls(tmp_path / "nope", customer_id="cust-7") == []


# --------------------------------------------------------------------------- the call

@pytest.mark.parametrize("lang,needles", [
    ("en", ["Last time you asked me to call you", "payment link"]),
    ("hi", ["पिछली बार आपने कहा था", "पेमेंट लिंक"]),
    ("hinglish", ["Pichhli baar aapne kaha tha", "payment link"]),
])
async def test_opens_with_continuity(tmp_path, lang, needles):
    seed(tmp_path)
    c = follow(tmp_path, lang=lang)
    plan = await c.open()
    t = plan.fallback[lang]
    assert all(n in t for n in needles), t
    assert "AI" in t and ("recorded" in t or "रिकॉर्ड" in t or "record" in t) and plan.llm_allowed is False


async def test_answers_the_open_question_first_then_asks_for_the_decision(tmp_path):
    seed(tmp_path)
    c = follow(tmp_path, lang="hinglish")
    await c.open()
    await c.say("haan boliye")
    t = c.last.text
    assert "privacy settings" in t.lower()  # the approved FAQ answer, word for word
    assert t.index("privacy") < t.index("Kya aapne is baare mein socha")
    assert c.last.plan.intent == "step_ask"


async def test_agreed_price_is_honoured_and_the_old_offer_is_not_quoted(tmp_path):
    seed(tmp_path)
    c = follow(tmp_path, lang="hinglish")
    await c.open()
    await c.say("haan boliye")
    await c.say("price kitna hai ab?")
    assert "₹2,999" in c.last.text and "$8" not in c.last.text and "wahi rahega" in c.last.text
    assert not any(i in ("pitch", "price", "reoffer", "confirm_close") for i in c.intents)


async def test_never_repeats_what_was_already_said(tmp_path):
    seed(tmp_path)
    c = follow(tmp_path, lang="hinglish")
    await c.open()
    await c.say("haan boliye")
    await c.say("haan sochna padega")
    await c.say("kal shaam 6 baje")
    said = " ".join(c.agent[1:])
    assert "ek AI assistant" not in said and "Claude Pro" not in said  # no second introduction / pitch
    assert len(set(c.agent)) == len(c.agent)


async def test_go_ahead_confirms_the_link_at_the_agreed_price_and_sends_it(tmp_path):
    seed(tmp_path)
    c = follow(tmp_path, lang="hinglish")
    await c.open()
    await c.say("haan boliye")
    await c.say("haan main lena chahta hoon, aage badhte hain")
    assert "₹2,999" in c.last.text and "payment link" in c.last.text and c.last.plan.end_call is False
    await c.say("haan bhej do")
    assert c.last.plan.intent == "link_sent" and c.last.plan.end_call
    assert c.sess.dialog["payment_link_url"] == "https://claude.ai/upgrade" and c.last.plan.facts["payment_link"].hidden
    r = c.result
    assert r["outcome"] == "link_sent" and r["stage_suggestion"] == "negotiating" and r["facts"]["link_sent"] == "1"
    assert r["next_action"]["type"] == "follow_up_call" and r["next_action"]["call_type"] == "follow_up" and r["next_action"]["due_at"] == "2026-11-17T11:00:00+05:30"


async def test_go_ahead_but_no_link_is_ready_to_buy(tmp_path):
    seed(tmp_path)
    c = follow(tmp_path, lang="hinglish")
    await c.open()
    await c.say("haan boliye")
    await c.say("haan lena hai")
    await c.say("nahi abhi mat bhejo")
    assert c.result["outcome"] == "ready_to_buy" and c.last.plan.end_call


async def test_need_time_reschedules(tmp_path):
    seed(tmp_path)
    c = follow(tmp_path, lang="hinglish")
    await c.open()
    await c.say("haan boliye")
    await c.say("mujhe thoda time chahiye sochna hai")
    assert "kab call karoon" in c.last.text
    await c.say("parso subah 10 baje")
    r = c.result
    assert r["outcome"] == "callback_booked" and r["next_action"] == {"type": "follow_up_call", "due_at": "2026-11-17T10:00:00+05:30", "call_type": "follow_up"}
    assert r["answers"]["decision"] == "need_time" and c.last.plan.end_call


async def test_not_interested_is_a_polite_close(tmp_path):
    seed(tmp_path)
    c = follow(tmp_path, lang="en")
    await c.open()
    await c.say("yes tell me")
    await c.say("it's not for me")
    assert c.result["outcome"] == "declined" and c.result["stage_suggestion"] == "lost" and c.last.plan.end_call
    assert c.last.plan.intent == "declined"


async def test_already_paid_ends_politely(tmp_path):
    seed(tmp_path)
    c = follow(tmp_path, lang="hinglish")
    await c.open()
    await c.say("haan boliye")
    await c.say("maine payment kar diya hai already paid")
    assert c.result["outcome"] == "paid" and c.result["stage_suggestion"] == "won" and c.last.plan.end_call


async def test_opt_out_and_crisis_still_override(tmp_path):
    seed(tmp_path)
    c = follow(tmp_path, lang="en")
    await c.open()
    await c.say("stop calling me, remove my number")
    assert c.last.plan.intent == "opt_out" and c.result["outcome"] == "opt_out" and c.result["stage_suggestion"] == "do_not_call"
    c2 = follow(tmp_path, lang="hinglish")
    await c2.open()
    await c2.say("mujhe marna hai")
    assert c2.last.plan.intent == "safety_distress"


async def test_without_a_previous_call_it_uses_the_generic_opening(tmp_path):
    c = Call("follow_up", data_dir=tmp_path, customer_id="nobody", phone="+910000000000", lang="en")
    plan = await c.open()
    assert "following up on Claude Pro" in plan.fallback["en"] and "Last time" not in plan.fallback["en"]
    await c.say("yes")
    assert c.intents[-1] == "step_ask"  # straight to the decision question: no open questions to answer


async def test_a_question_in_the_middle_is_answered_and_the_decision_comes_back(tmp_path):
    seed(tmp_path, {**PREV, "context": {**PREV["context"], "questions": []}})
    c = follow(tmp_path, lang="hinglish")
    await c.open()
    await c.say("haan boliye")
    await c.say("kya ye UPI se ho jayega?")
    assert c.last.plan.intent == "answer" and "Kya aapne is baare mein socha" in c.last.text


async def test_same_input_same_output(tmp_path):
    seed(tmp_path)
    runs = []
    for _ in range(2):
        c = follow(tmp_path, lang="hinglish")
        await c.open()
        for line in ["haan boliye", "haan sochna padega", "kal shaam 6 baje"]:
            await c.say(line)
        runs.append((c.agent, c.result))
    assert runs[0] == runs[1]
