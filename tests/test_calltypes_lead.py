"""Lead qualification call: scripted en / hi / hinglish conversations through the real pipeline."""

import pytest

from tests.calltypes_helpers import Call, asked, run, step_text

SCRIPTS = {
    "en": ["yes go ahead", "we need a CRM for our sales team", "around 50,000 rupees", "yes, I make the decision", "right away", "tomorrow at 4 pm"],
    "hi": ["हाँ बोलिए", "हमें अपनी टीम के लिए सीआरएम चाहिए", "बजट पचास हजार है", "हाँ फ़ैसला मैं ही लेता हूँ", "तुरंत शुरू करना है", "कल शाम 6 बजे"],
    "hinglish": ["haan boliye", "mujhe CRM chahiye sales team ke liye", "budget das hazaar hai", "haan main hi decide karta hoon", "isi mahine shuru karna hai", "kal shaam 6 baje"],
}


@pytest.mark.parametrize("lang", ["en", "hi", "hinglish"])
async def test_full_qualification_books_a_demo(lang):
    c = await run("lead_qualification", SCRIPTS[lang], lang=lang)
    r = c.result
    assert r["complete"] and r["outcome"] == "demo_booked" and r["stage_suggestion"] == "demo_booked"
    assert r["answers"]["budget"] in (10000, 50000) and r["answers"]["authority"] is True
    assert r["answers"]["timeline"] in ("now", "this_month")
    assert r["next_action"]["type"] == "book_demo" and r["next_action"]["due_at"].startswith("2026-11-16T1")
    assert r["score_delta"] >= 80 and r["call_type"] == "lead_qualification"
    assert c.last.plan.end_call and c.last.result == r  # surfaced on the turn result too
    # every configured step was asked exactly once, in order, in the caller's language
    order = [step_text("lead_qualification", s, lang) for s in ("need", "budget", "authority", "timeline", "book_demo")]
    pos = [next(i for i, t in enumerate(c.agent) if o in t) for o in order]
    assert pos == sorted(pos) and all(sum(o in t for t in c.agent) == 1 for o in order)
    assert c.sess.context.result == r and c.sess.context.call_type == "lead_qualification"


async def test_opening_discloses_ai_recording_and_purpose_without_selling():
    c = Call("lead_qualification", lang="en")
    plan = await c.open()
    t = plan.fallback["en"]
    assert "AI assistant" in t and "recorded" in t and "Claude Pro" in t and "no selling" in t and t.endswith("?")
    assert plan.llm_allowed is False and plan.intent == "opening"


async def test_answers_in_one_sentence_are_never_asked_again():
    c = await run("lead_qualification", ["haan boliye", "CRM chahiye, mera budget 20 hazaar hai"], lang="hinglish")
    assert c.result["answers"]["budget"] == 20000 and "need" in c.result["answers"]
    assert step_text("lead_qualification", "budget", "hinglish") not in " ".join(c.agent)
    assert step_text("lead_qualification", "authority", "hinglish") in c.agent[-1]


async def test_unanswered_step_is_reasked_once_then_skipped():
    c = await run("lead_qualification", ["haan", "CRM", "hmm", "hmm", "haan main hi"], lang="hinglish")
    assert asked(c, "lead_qualification", "budget") == 2  # asked, then once more with a hint; never a third time
    assert c.result["answers"]["budget"] is None and c.result["answers"]["authority"] is True
    assert asked(c, "lead_qualification", "timeline") == 1


async def test_question_is_answered_from_the_facts_then_the_step_is_asked_again():
    c = await run("lead_qualification", ["haan", "CRM chahiye", "does it use my data for training?"], lang="en")
    assert c.last.plan.intent == "answer"
    t = c.last.text
    assert "privacy settings" in t.lower() or "train" in t.lower()
    assert step_text("lead_qualification", "budget", "en") in t  # the open question comes back
    assert "budget" not in c.result["answers"]
    r = await c.say("around 30000")
    assert c.result["answers"]["budget"] == 30000 and c.last.plan.intent == "step_ask"


async def test_price_question_gets_the_locked_price_line_no_pitch():
    c = await run("lead_qualification", ["haan", "CRM chahiye", "price kitna hai?"], lang="hinglish")
    assert "$8" in c.last.text and step_text("lead_qualification", "budget", "hinglish") in c.last.text
    assert not any(i in ("pitch", "reoffer", "confirm_close") for i in c.intents)


async def test_low_score_gets_a_polite_close_not_a_demo():
    c = await run("lead_qualification", ["haan", "bas dekh raha tha", "pata nahi", "nahi", "just looking"], lang="hinglish")
    r = c.result
    assert r["complete"] and r["outcome"] == "not_qualified" and r["stage_suggestion"] == "contacted"
    assert r["score_delta"] < 40 and r["next_action"]["type"] == ""
    assert step_text("lead_qualification", "book_demo", "hinglish") not in " ".join(c.agent)
    assert c.last.plan.end_call


async def test_good_score_but_demo_declined_is_qualified():
    c = await run("lead_qualification", ["haan", "CRM chahiye", "5 lakh", "haan", "abhi shuru karna hai", "nahi abhi nahi"], lang="hinglish")
    assert c.result["outcome"] == "qualified" and c.result["answers"]["book_demo"] == "declined" and c.result["stage_suggestion"] == "qualified"


async def test_busy_caller_gets_a_callback_booked():
    c = await run("lead_qualification", ["abhi busy hoon, kal shaam 6 baje call karna"], lang="hinglish")
    r = c.result
    assert r["outcome"] == "callback_booked" and r["next_action"] == {"type": "follow_up_call", "due_at": "2026-11-16T18:00:00+05:30", "call_type": "follow_up"}
    assert c.last.plan.end_call and c.last.plan.intent == "callback"


async def test_busy_without_time_asks_when_then_books():
    c = await run("lead_qualification", ["abhi busy hoon", "parso subah 10 baje"], lang="hinglish")
    assert c.intents[1] == "callback_time" and c.result["outcome"] == "callback_booked"
    assert c.result["next_action"]["due_at"] == "2026-11-17T10:00:00+05:30"


async def test_no_to_the_permission_question_is_not_now_not_a_hang_up_on_the_data():
    c = await run("lead_qualification", ["nahi"], lang="hinglish")
    assert c.intents[1] == "callback_time"


async def test_opt_out_overrides_everything():
    c = await run("lead_qualification", ["haan", "CRM chahiye", "please don't call me again, remove my number"], lang="en")
    assert c.last.plan.intent == "opt_out" and c.last.plan.end_call
    r = c.result
    assert r["outcome"] == "opt_out" and r["stage_suggestion"] == "do_not_call" and r["next_action"]["type"] == "" and r["complete"]
    assert c.sess.dialog["outcome"] == "opt_out"


async def test_firm_no_is_respected():
    c = await run("lead_qualification", ["haan", "not interested, thanks"], lang="en")
    assert c.last.plan.intent == "declined" and c.last.plan.end_call
    assert c.result["outcome"] == "declined" and c.result["stage_suggestion"] == "lost"
    assert "pitch" not in " ".join(c.intents)


async def test_crisis_row_still_wins():
    c = await run("lead_qualification", ["haan", "mujhe marna hai"], lang="hinglish")
    assert c.last.plan.intent == "safety_distress" and c.last.plan.end_call is False
    assert "budget" not in c.last.text.lower()


async def test_are_you_ai_is_answered_honestly_then_the_question_resumes():
    c = await run("lead_qualification", ["haan", "are you a bot?"], lang="en")
    assert c.last.plan.intent == "are_you_ai" and "AI assistant" in c.last.text
    assert step_text("lead_qualification", "need", "en") in c.last.text


async def test_number_source_offers_removal():
    c = await run("lead_qualification", ["haan", "where did you get my number?"], lang="en")
    assert c.last.plan.intent == "number_source" and "remove" in c.last.text


async def test_a_budget_in_digits_is_not_mistaken_for_a_card_number_but_an_otp_is_blocked():
    c = await run("lead_qualification", ["haan", "CRM", "mera budget 500000 hai"], lang="hinglish")
    assert c.result["answers"]["budget"] == 500000 and c.last.plan.intent == "step_ask"
    c2 = await run("lead_qualification", ["haan", "CRM", "my card number is 4111111111111111"], lang="en")
    assert c2.last.plan.intent == "sensitive_stop"
    assert "budget" not in c2.result["answers"]


async def test_max_call_length_wraps_up():
    import tests.calltypes_helpers as h
    from datetime import timedelta
    c = Call("lead_qualification", lang="en")
    await c.open()
    await c.say("yes")
    clock = {"now": h.NOW + timedelta(seconds=1000)}
    c.domain._now_fn = lambda: clock["now"]
    await c.say("we need a CRM")
    assert c.last.plan.end_call and c.result["complete"]


async def test_a_partial_result_exists_if_the_caller_hangs_up_mid_call():
    c = await run("lead_qualification", ["haan", "CRM chahiye", "budget 10 hazaar"], lang="hinglish")
    r = c.result
    assert r["complete"] is False and r["outcome"] == "ended_early" and r["answers"]["budget"] == 10000 and r["score_delta"] == 30


async def test_same_input_same_output():
    a = await run("lead_qualification", SCRIPTS["hinglish"], lang="hinglish")
    b = await run("lead_qualification", SCRIPTS["hinglish"], lang="hinglish")
    assert a.agent == b.agent and a.result == b.result
