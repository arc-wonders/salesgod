import asyncio

from acie.feedback import FeedbackEngine
from acie.schemas import CustomerProfile, SessionState, Utterance


def sess(decision="undecided", promised=(), spoke=True, safety=False, sid="s"):
    s = SessionState(session_id=sid, customer_id="c")
    s.context.decision = decision
    s.context.promised = list(promised)
    if spoke:
        s.history = [Utterance(role="customer", text="haan"), Utterance(role="agent", text="ok")]
    if safety:
        s.dialog["safety_flag"] = "1"
    return s


def close(s):
    p = CustomerProfile(customer_id="c")
    FeedbackEngine().end_session(p, s)
    return p


def test_each_terminal_path():
    assert close(sess("accepted", ["payment_link"])).outcomes == ["accepted"]
    assert close(sess("declined")).outcomes == ["declined"]
    assert close(sess("later", ["callback_later"])).outcomes == ["later"]
    assert close(sess("declined", ["number_removed_no_more_calls"])).outcomes == ["opt_out"]
    assert close(sess(safety=True, decision="declined")).outcomes == ["safety"]
    assert close(sess(promised=["team_member_will_call"])).outcomes == ["human_agent"]
    assert close(sess("accepted", ["team_member_will_call"])).outcomes == ["accepted", "human_agent"]
    assert close(sess(spoke=False)).outcomes == ["no_answer"]
    assert close(sess()).outcomes == []  # spoke, undecided: nothing to record


def test_once_per_session_and_keep_last_50():
    s, p, fb = sess("accepted"), CustomerProfile(customer_id="c", outcomes=["x"] * 60), FeedbackEngine()
    fb.end_session(p, s)
    fb.end_session(p, s)  # hang-up handler after the end_call turn
    assert p.interaction_count == 1 and p.outcomes.count("accepted") == 1 and len(p.outcomes) == 50


def test_legacy_call_without_session():
    p = CustomerProfile(customer_id="c")
    FeedbackEngine().end_session(p)
    assert p.interaction_count == 1 and p.outcomes == []
