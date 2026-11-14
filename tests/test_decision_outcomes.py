import pytest
from decision_helpers import run


def led(pipe):
    return pipe.sessions["s"].context


async def test_accepted():
    _, _, pipe, _, _ = await run(["yes tell me", "how much", "yes send the link"])
    c = led(pipe)
    assert (c.decision, c.outcome) == ("accepted", "accepted") and c.has("payment_link")
    assert pipe.sessions["s"].dialog["outcome"] == "accepted"


async def test_declined_after_reoffer():
    _, _, pipe, _, _ = await run(["yes tell me", "no not interested", "nahi chahiye", "nahi chahiye"])
    c = led(pipe)
    assert (c.decision, c.outcome) == ("declined", "declined") and not c.has("payment_link")


async def test_declined_after_agreeing_removes_promise():
    _, _, pipe, _, _ = await run(["yes tell me", "how much", "yes send the link", "no don't send it"])
    c = led(pipe)
    assert (c.decision, c.outcome) == ("declined", "declined") and not c.has("payment_link")
    assert pipe.sessions["s"].dialog["outcome"] == "declined"


async def test_goodbye_after_sale_keeps_accepted():
    _, _, pipe, _, _ = await run(["yes tell me", "how much", "yes send the link", "ok thanks"])
    assert led(pipe).outcome == "accepted"


async def test_later_callback():
    # (was one turn: "call me later" ended the call with no time; now the agent asks when, the second turn closes)
    _, last, pipe, _, _ = await run(["abhi busy hoon baad mein call karo", "kal shaam 6 baje"])
    c = led(pipe)
    assert (c.decision, c.outcome) == ("later", "later") and c.has("callback_later") and last.plan.end_call


async def test_opt_out():
    _, _, pipe, _, _ = await run(["हाँ बोलिए", "mujhe dobara call mat karna"])
    c = led(pipe)
    assert (c.decision, c.outcome) == ("declined", "opt_out") and c.has("number_removed_no_more_calls")


async def test_opt_out_reconfirmed_stays_opt_out():
    _, _, pipe, _, _ = await run(["हाँ बोलिए", "मेरा नंबर हटा दो", "पक्का हटा दिया?"])
    assert led(pipe).outcome == "opt_out"


async def test_human_agent():
    _, last, pipe, _, _ = await run(["mujhe kisi insaan se baat karni hai"])
    c = led(pipe)
    assert c.outcome == "human_agent" and c.has("team_member_will_call") and last.plan.end_call


async def test_safety():
    out, last, pipe, _, _ = await run(["हाँ बोलिए", "I want to die"])
    c = led(pipe)
    assert out[-1] == "safety_distress" and c.outcome == "safety" and c.has("helpline_given_call_stopped")
    out, _, pipe, _, _ = await run(["हाँ बोलिए", "I want to die", "bye"])
    assert led(pipe).outcome == "safety"


async def test_link_cancelled_reopens():
    out, _, pipe, _, _ = await run(["yes tell me", "how much", "yes send the link", "I never said yes, I didn't agree"])
    c = led(pipe)
    assert out[-1] == "link_cancelled"
    assert (c.decision, c.outcome) == ("undecided", "") and not c.has("payment_link")
    assert pipe.sessions["s"].dialog["outcome"] == ""


async def test_outcome_event_recorded_once():
    _, _, pipe, _, _ = await run(["yes tell me", "how much", "yes send the link", "ok thanks"])
    assert sum(1 for e in led(pipe).events if e.kind == "outcome") == 1


def test_goodbye_after_link_is_not_a_decline():
    """Gate run 7: 'Okay fine, will pay now. Thanks, bye' after link_sent was recorded as declined (sale lost in CRM)."""
    from acie.context import CallContext
    c = CallContext()
    c.record_turn(1, "haan bhej do", "link_sent", False)
    c.record_turn(2, "Okay fine, will pay now. Thanks, bye", "declined", False)
    assert "payment_link" in c.promised and c.decision == "accepted"
    c2 = CallContext()
    c2.record_turn(1, "haan bhej do", "link_sent", False)
    c2.record_turn(2, "no don't send it, cancel", "declined", False)
    assert "payment_link" not in c2.promised and c2.decision == "declined"
