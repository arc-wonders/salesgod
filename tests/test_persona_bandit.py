import random

import pytest

from acie.personalization.bandit import DIM, BanditPolicy, context_vector
from acie.schemas import CustomerProfile, SessionState
from acie.schemas import HumanState as _Base


class HumanState(_Base):  # extended dims (state lane adds them to schemas.HumanState)
    anxiety: float = 0.0
    trust: float = 0.5
    certainty: float = 0.5
    cognitive_load: float = 0.0


def mk(tmp_path, **kw):
    kw.setdefault("state_path", tmp_path / "b.json")
    kw.setdefault("rng", random.Random(1))
    return BanditPolicy(**kw)


def turn(bp, state=None, explicit=None, cid="c", turn_index=1):
    s = SessionState(session_id="s", customer_id=cid, turn_index=turn_index)
    p = CustomerProfile(customer_id=cid)
    return bp.decide("ok batao", state or HumanState(), p, s, explicit or {}), p


def simulate(bp, steps, pref="low", knob="verbosity", rng=None, state=None, explicit=None):
    rng = rng or random.Random(7)
    for _ in range(steps):
        a, p = turn(bp, state, explicit)
        r = (0.6 if getattr(a, knob) == pref else -0.2) + rng.uniform(-0.2, 0.2)
        bp.observe(state or HumanState(), a, r, p)


def test_context_vector_shape():
    x = context_vector(HumanState(frustration=0.9), "hi", 4, CustomerProfile(customer_id="c"))
    assert x.shape == (DIM,) and x.sum() == 14 + 1 + 1 + 1 + 1


def test_starts_as_rule_policy(tmp_path):
    from acie.personalization.policy import AdaptationPolicy
    bp = mk(tmp_path, epsilon=0.0)
    s = SessionState(session_id="s", customer_id="c")
    p = CustomerProfile(customer_id="c")
    assert bp.decide("hello", HumanState(), p, s, {}) == AdaptationPolicy().decide(
        "hello", HumanState(), p, SessionState(session_id="s", customer_id="c"), {})


def test_converges_to_preferred_action(tmp_path):
    bp = mk(tmp_path, epsilon=0.3, min_obs=30)
    simulate(bp, 400)
    final = turn(bp)[0]
    bp.epsilon = 0.0
    assert turn(bp)[0].verbosity == "low"  # rule says med; learned that low pays better
    assert bp.counts("verbosity")["low"] >= 30 and final is not None


def test_no_override_without_enough_observations(tmp_path):
    bp = mk(tmp_path, epsilon=0.3, min_obs=10_000)
    simulate(bp, 300)
    bp.epsilon = 0.0
    assert turn(bp)[0].verbosity == "med"


def test_no_override_when_rule_is_best(tmp_path):
    bp = mk(tmp_path, epsilon=0.3)
    simulate(bp, 400, pref="med")
    bp.epsilon = 0.0
    assert turn(bp)[0].verbosity == "med"


def test_explicit_never_overridden(tmp_path):
    bp = mk(tmp_path, epsilon=0.0, min_obs=5)
    simulate(bp, 300, pref="low")  # learn "low is great"... on knobs that aren't forced
    bp.epsilon = 1.0
    for _ in range(100):
        a, _ = turn(bp, explicit={"verbosity": "high", "explanation_style": "example", "info_density": "low"})
        assert a.verbosity == "high" and a.explanation_style == "example" and a.info_density == "low"


def test_safety_rules_never_overridden(tmp_path):
    bp = mk(tmp_path, epsilon=1.0, min_obs=1, margin=-1.0)
    simulate(bp, 200, pref="low", knob="warmth")
    bp.epsilon = 1.0
    for _ in range(80):
        a, _ = turn(bp, HumanState(frustration=0.9))
        assert a.warmth == "high" and a.verbosity == "low" and a.proactivity == "low"
        b, _ = turn(bp, HumanState(confusion=0.9, anxiety=0.9))
        assert b.complexity == "low" and b.explanation_style == "step_by_step"


def test_safety_flagged_call_all_protected(tmp_path):
    bp = mk(tmp_path, epsilon=1.0)
    s = SessionState(session_id="s", customer_id="c", turn_index=1)
    s.dialog["safety_flag"] = "1"
    from acie.personalization.policy import AdaptationPolicy
    p = CustomerProfile(customer_id="c")
    base = AdaptationPolicy().decide("x", HumanState(), p, SessionState(session_id="s", customer_id="c", turn_index=1,
                                                                        dialog={"safety_flag": "1"}), {})
    assert bp.decide("x", HumanState(), p, s, {}) == base


def test_exploration_plays_other_arms(tmp_path):
    bp = mk(tmp_path, epsilon=1.0)
    seen = set()
    for _ in range(60):
        a, p = turn(bp)
        seen.add(a.verbosity)
        bp.observe(HumanState(), a, 0.0, p)
    assert seen >= {"low", "high"}


def test_protected_knobs_not_updated(tmp_path):
    bp = mk(tmp_path, epsilon=0.0)
    a, p = turn(bp, explicit={"verbosity": "low"})
    bp.observe(HumanState(), a, 1.0, p)
    assert sum(bp.counts("verbosity").values()) == 0 and sum(bp.counts("warmth").values()) == 1


def test_reward_clipped(tmp_path):
    bp = mk(tmp_path, epsilon=0.0)
    a, p = turn(bp)
    bp.observe(HumanState(), a, 99.0, p)
    arm = bp.arms["warmth"][a.warmth]
    assert max(abs(arm.b)) <= 1.0


def test_persistence_roundtrip(tmp_path):
    bp = mk(tmp_path, epsilon=0.3, save_every=0)
    simulate(bp, 150)
    bp.save()
    assert (tmp_path / "b.json").exists() and not list(tmp_path.glob("*.tmp"))
    bp2 = mk(tmp_path, epsilon=0.0)
    assert bp2.total_obs == bp.total_obs and bp2.counts("verbosity") == bp.counts("verbosity")
    x = context_vector(HumanState(), "en", 1, CustomerProfile(customer_id="c"))
    for k in bp.arms:
        for a in bp.arms[k]:
            assert bp.arms[k][a].mean(x) == pytest.approx(bp2.arms[k][a].mean(x))
    # decisions agree
    bp.epsilon = 0.0
    assert turn(bp)[0] == turn(bp2)[0]


def test_autosave_and_corrupt_state(tmp_path):
    bp = mk(tmp_path, save_every=5, epsilon=0.0)
    simulate(bp, 5)
    assert (tmp_path / "b.json").exists()
    (tmp_path / "b.json").write_text("{not json")
    bp2 = mk(tmp_path)  # corrupt -> clean start, no crash
    assert bp2.total_obs == 0


def test_wrong_dimension_state_ignored(tmp_path):
    (tmp_path / "b.json").write_text('{"dim": 3, "arms": {}}')
    assert mk(tmp_path).total_obs == 0


def test_pipeline_wires_env(tmp_path, monkeypatch):
    from acie.pipeline import ACIEPipeline
    from acie.decision.engine import DecisionEngine
    monkeypatch.setenv("ACIE_POLICY", "bandit")
    monkeypatch.setenv("ACIE_BANDIT_STATE", str(tmp_path / "x.json"))
    pl = ACIEPipeline.__new__(ACIEPipeline)
    from acie.personalization.policy import make_default_policy
    assert hasattr(make_default_policy(), "observe")
