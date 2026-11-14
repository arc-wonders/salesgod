import sys
import types

import pytest

from acie.feedback import FeedbackEngine, validation_penalty
from acie.schemas import (CustomerProfile, ExpressionParams, HumanState, SessionState, TurnPlan, Utterance, ValidationReport)

PLAN = TurnPlan(intent="pitch", objective="x")


class Pol:
    def __init__(self):
        self.calls = []

    def observe(self, state, expr, reward, profile):
        self.calls.append((state, expr, reward))


def turn(fb, p, s, st, expr=None, val=None, barge=False, text="haan"):
    fb.after_turn(p, text, {}, None, st, PLAN, session=s, expr=expr or ExpressionParams(), validation=val, barge_in=barge)


def test_reward_math():
    fb = FeedbackEngine()
    a, b = HumanState(frustration=0.2, confusion=0.4), HumanState(frustration=0.0, confusion=0.0)
    assert fb.turn_reward(a, b) == pytest.approx(0.8 * 0.2 + 0.6 * 0.4)
    assert fb.turn_reward(b, a) == pytest.approx(-(0.8 * 0.2 + 0.6 * 0.4))
    assert fb.turn_reward(a, a, penalty=1.0, barge_in=True) == pytest.approx(-0.8)
    assert fb.turn_reward(HumanState(), HumanState(frustration=1, confusion=1), 1.0, True) == -1.0  # clipped
    pos = HumanState(engagement=0.8)
    assert fb.turn_reward(HumanState(), pos) == pytest.approx(0.5 * 0.3 / 4)


def test_weights_configurable(monkeypatch):
    a, b = HumanState(frustration=0.5), HumanState()
    assert FeedbackEngine(weights={"frustration": 1.0}).turn_reward(a, b) == pytest.approx(0.5)
    monkeypatch.setenv("ACIE_REWARD_WEIGHTS", "frustration=0.1")
    assert FeedbackEngine().turn_reward(a, b) == pytest.approx(0.05)


def test_penalty_codes_or_issues():
    assert validation_penalty(None) == 0
    assert validation_penalty(ValidationReport(ok=True)) == 0
    assert validation_penalty(ValidationReport(ok=False, used_fallback=True)) == 1.0
    assert validation_penalty(ValidationReport(ok=False, issues=["trimmed reply from 3 to 2 sentences"])) == 0
    assert validation_penalty(ValidationReport(ok=False, issues=["bad: 'x'"])) == 0.4
    object.__setattr__(r, "issue_codes", ["trimmed"]) if False else None
    ns = types.SimpleNamespace(used_fallback=False, issue_codes=["regenerated after: x"], issues=["bad"])
    assert validation_penalty(ns) == 0  # issue_codes win


def mk():
    pol, fb = Pol(), FeedbackEngine()
    fb.policy = pol
    return pol, fb, CustomerProfile(customer_id="c"), SessionState(session_id="s", customer_id="c")


def say(s, cust, agent):
    s.history += [Utterance(role="customer", text=cust), Utterance(role="agent", text=agent)]


def plan(intent, end=False):
    return TurnPlan(intent=intent, objective="x", end_call=end)


def step(fb, p, s, st, intent="answer", cust="haan", agent="ok", val=None, barge=False, end=False):
    say(s, cust, agent)
    fb.after_turn(p, cust, {}, None, st, plan(intent, end), session=s, expr=ExpressionParams(), validation=val,
                  barge_in=barge)


def test_pending_resolved_on_next_turn_and_calm_credit():
    pol, fb, p, s = mk()
    step(fb, p, s, HumanState(frustration=0.4), agent="first reply")
    assert pol.calls == []  # needs the next turn
    step(fb, p, s, HumanState(frustration=0.1), agent="a different answer entirely", barge=True)
    assert len(pol.calls) == 1 and pol.calls[0][2] == pytest.approx(0.8 * 0.3 - 0.3)
    fb.end_session(p, s)
    r = [c[2] for c in pol.calls]
    assert r[1] == pytest.approx(0.0)  # last turn: no next state
    assert r[2] == pytest.approx(0.5 * 0.9) and r[3] == pytest.approx(0.5)  # calm + answered credit


def test_sales_outcome_has_no_weight():
    rewards = []
    for decision in ("accepted", "undecided"):
        pol, fb, p, s = mk()
        step(fb, p, s, HumanState())
        s.context.decision = decision
        fb.end_session(p, s)
        rewards.append([c[2] for c in pol.calls])
    assert rewards[0] == rewards[1]  # accepting the offer earns nothing extra


def test_respected_refusal_and_pushed_opt_out():
    pol, fb, p, s = mk()
    step(fb, p, s, HumanState(), intent="pitch")
    s.context.decision = "later"
    step(fb, p, s, HumanState(), intent="callback", end=True, agent="sure, I will call later")
    fb.end_session(p, s)
    assert pol.calls[-1][2] == pytest.approx(1.0)  # respected 0.5 + calm 0.5
    pol, fb, p, s = mk()
    s.context.decision = "declined"
    step(fb, p, s, HumanState(), intent="recover", agent="but it is a great deal")
    s.context.promised = ["number_removed_no_more_calls"]
    step(fb, p, s, HumanState(), intent="opt_out", agent="removed your number", end=True)
    fb.end_session(p, s)
    assert pol.calls[-1][2] == pytest.approx(-1.0)


def test_repeat_hangup_and_unanswered():
    pol, fb, p, s = mk()
    step(fb, p, s, HumanState(), agent="Our plan costs a small monthly fee.")
    step(fb, p, s, HumanState(), agent="Our plan costs a small monthly fee.")  # repeat, caller hangs up
    fb.end_session(p, s)
    assert pol.calls[-1][2] == pytest.approx(-1.0)  # hangup -1 plus repeat -0.5, clipped
    pol, fb, p, s = mk()
    step(fb, p, s, HumanState(), cust="what does it cost?", agent="sorry, say again",
         val=ValidationReport(ok=False, used_fallback=True))
    fb.end_session(p, s)
    assert pol.calls[-1][2] <= -0.5 + 1e-9


def test_deterministic():
    out = []
    for _ in range(2):
        pol, fb, p, s = mk()
        for i in range(3):
            step(fb, p, s, HumanState(frustration=0.1 * i), agent=f"reply number {i}")
        fb.end_session(p, s)
        out.append([c[2] for c in pol.calls])
    assert out[0] == out[1]


def test_safety_excluded():
    pol, fb, p, s = mk()
    step(fb, p, s, HumanState())
    s.dialog["safety_flag"] = "1"
    fb.end_session(p, s)
    assert pol.calls == []


def test_learner_hook_called_and_missing_is_skipped(monkeypatch):
    got = []
    mod = types.ModuleType("acie.personalization.learner")
    mod.learn_from_turn = lambda profile, text, state, expr, explicit: got.append((text, explicit))
    monkeypatch.setitem(sys.modules, "acie.personalization.learner", mod)
    fb, p, s = FeedbackEngine(), CustomerProfile(customer_id="c"), SessionState(session_id="s", customer_id="c")
    turn(fb, p, s, HumanState(), text="namaste")
    assert got == [("namaste", {})]
    monkeypatch.setitem(sys.modules, "acie.personalization.learner", None)  # import fails -> skipped
    turn(fb, p, s, HumanState())


def test_validator_learning_prefer_templates():
    fb, p, s = FeedbackEngine(), CustomerProfile(customer_id="c"), SessionState(session_id="s", customer_id="c")
    bad, ok = ValidationReport(ok=False, used_fallback=True), ValidationReport(ok=True)
    turn(fb, p, s, HumanState(), val=bad)
    assert "prefer_templates" not in p.preferences
    turn(fb, p, s, HumanState(), val=ok)
    turn(fb, p, s, HumanState(), val=bad)
    assert p.preferences["prefer_templates"].value == "true"
    assert p.preferences["complexity"].value == "low"
    for _ in range(4):
        turn(fb, p, s, HumanState(), val=ok)
    assert p.preferences["prefer_templates"].confidence < 0.4 or p.preferences["prefer_templates"].value == "false"
