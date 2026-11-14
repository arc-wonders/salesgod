import sys
import types

import pytest

from acie.feedback import FeedbackEngine, validation_penalty
from acie.schemas import (CustomerProfile, ExpressionParams, HumanState, SessionState, TurnPlan, ValidationReport)

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
    assert fb.turn_reward(HumanState(), pos) == pytest.approx(0.5 * 0.3 / 3)


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


def test_pending_resolved_on_next_turn_and_delayed_credit():
    pol, fb, p, s = Pol(), FeedbackEngine(), CustomerProfile(customer_id="c"), SessionState(session_id="s", customer_id="c")
    fb.policy = pol
    turn(fb, p, s, HumanState(frustration=0.4))
    assert pol.calls == []  # needs the next turn
    turn(fb, p, s, HumanState(frustration=0.1), barge=True)
    assert len(pol.calls) == 1 and pol.calls[0][2] == pytest.approx(0.8 * 0.3 - 0.3)
    s.context.decision = "accepted"
    s.history = []
    fb.end_session(p, s)
    # last turn resolved at end (no next state: 0), then both turns credited with +1 (discounted)
    rewards = [c[2] for c in pol.calls]
    assert len(rewards) == 4 and rewards[1] == pytest.approx(0.0)
    assert rewards[2] == pytest.approx(0.9) and rewards[3] == pytest.approx(1.0)


def test_declined_negative_and_safety_excluded():
    for decision, safety, expect in (("declined", False, -0.3), ("later", False, 0.2), (None, True, None)):
        pol, fb, p = Pol(), FeedbackEngine(), CustomerProfile(customer_id="c")
        s = SessionState(session_id="s", customer_id="c")
        fb.policy = pol
        turn(fb, p, s, HumanState())
        if decision:
            s.context.decision = decision
        if safety:
            s.dialog["safety_flag"] = "1"
        fb.end_session(p, s)
        if expect is None:
            assert pol.calls == []
        else:
            assert pol.calls[-1][2] == pytest.approx(expect)


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
