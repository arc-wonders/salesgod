import json
import random
import subprocess
import sys
import textwrap

import pytest

from acie.personalization.bandit import DIM, KNOB_ACTIONS, BanditPolicy, context_vector
from acie.schemas import CustomerProfile, ExpressionParams, SessionState
from acie.schemas import HumanState as _Base


class HumanState(_Base):  # extended dims (merged state lane has them; subclass keeps this test self-contained)
    anxiety: float = 0.0
    trust: float = 0.5
    certainty: float = 0.5
    cognitive_load: float = 0.0


def mk(tmp_path, **kw):
    kw.setdefault("state_path", tmp_path / "b.json")
    return BanditPolicy(**kw)


def turn(bp, state=None, explicit=None, cid="c", turn_index=1):
    s = SessionState(session_id="s", customer_id=cid, turn_index=turn_index)
    p = CustomerProfile(customer_id=cid)
    return bp.decide("ok batao", state or HumanState(), p, s, explicit or {}), p


def make_log(n, pref="low", knob="verbosity", seed=7):
    """Logged rows from a fixed logging schedule that plays every action (offline data collection)."""
    rng = random.Random(seed)
    p = CustomerProfile(customer_id="c")
    rows = []
    for i in range(n):
        st = HumanState(frustration=rng.random() * 0.3, engagement=rng.random())
        x = context_vector(st, "en", i % 5, p)
        acts = {k: KNOB_ACTIONS[k][(i + j) % 3] for j, k in enumerate(KNOB_ACTIONS)}
        r = (0.6 if acts[knob] == pref else -0.2) + rng.uniform(-0.2, 0.2)
        rows.append({"x": [int(v) for v in x.nonzero()[0]], "reward": r, "actions": acts, "protected": []})
    return rows


def test_context_vector_shape():
    x = context_vector(HumanState(frustration=0.9), "hi", 4, CustomerProfile(customer_id="c"))
    assert x.shape == (DIM,) and x.sum() == 14 + 1 + 1 + 1 + 1


def test_no_randomness_in_module():
    import acie.personalization.bandit as m
    import acie.personalization.xgb_style as xm
    for mod in (m, xm):
        src = open(mod.__file__).read().split('"""', 2)[2]
        assert "random" not in src and "epsilon" not in src and "rng" not in src


def test_starts_as_rule_policy(tmp_path):
    from acie.personalization.policy import AdaptationPolicy
    bp = mk(tmp_path)
    s = SessionState(session_id="s", customer_id="c")
    p = CustomerProfile(customer_id="c")
    assert bp.decide("hello", HumanState(), p, s, {}) == AdaptationPolicy().decide(
        "hello", HumanState(), p, SessionState(session_id="s", customer_id="c"), {})


def test_learns_preferred_action_from_log(tmp_path):
    bp = mk(tmp_path, min_obs=30)
    bp.fit_rows(make_log(400))
    assert turn(bp)[0].verbosity == "low"  # rule says med; the log shows low pays better
    assert bp.counts("verbosity")["low"] >= 30


def test_live_decisions_never_explore(tmp_path):
    bp = mk(tmp_path)
    seen = set()
    for _ in range(100):
        a, p = turn(bp)
        seen.add(a.verbosity)
        bp.observe(HumanState(), a, 0.0, p)
    assert seen == {"med"}  # no data -> always the rule action


def test_no_override_without_enough_observations(tmp_path):
    bp = mk(tmp_path, min_obs=10_000)
    bp.fit_rows(make_log(300))
    assert turn(bp)[0].verbosity == "med"


def test_no_override_when_rule_is_best(tmp_path):
    bp = mk(tmp_path)
    bp.fit_rows(make_log(400, pref="med"))
    assert turn(bp)[0].verbosity == "med"


def test_pure_posterior_mean_ignores_alpha(tmp_path):
    rows = make_log(300)
    a = mk(tmp_path, alpha=0.0, persist=False)
    b = mk(tmp_path, alpha=50.0, persist=False)
    a.fit_rows(rows)
    b.fit_rows(rows)
    assert turn(a)[0] == turn(b)[0]
    x = context_vector(HumanState(), "en", 1, CustomerProfile(customer_id="c"))
    assert b.ucb("verbosity", "low", x) > b.predict("verbosity", "low", x)


def test_explicit_never_overridden(tmp_path):
    bp = mk(tmp_path, min_obs=5, margin=-1.0)
    bp.fit_rows(make_log(300))
    a, _ = turn(bp, explicit={"verbosity": "high", "explanation_style": "example", "info_density": "low"})
    assert a.verbosity == "high" and a.explanation_style == "example" and a.info_density == "low"


def test_safety_rules_never_overridden(tmp_path):
    bp = mk(tmp_path, min_obs=1, margin=-1.0)
    bp.fit_rows(make_log(200, pref="low", knob="warmth"))
    a, _ = turn(bp, HumanState(frustration=0.9))
    assert a.warmth == "high" and a.verbosity == "low" and a.proactivity == "low"
    b, _ = turn(bp, HumanState(confusion=0.9, anxiety=0.9))
    assert b.complexity == "low" and b.explanation_style == "step_by_step"


def test_safety_flagged_call_all_protected(tmp_path):
    bp = mk(tmp_path, min_obs=1, margin=-1.0)
    bp.fit_rows(make_log(200))
    from acie.personalization.policy import AdaptationPolicy
    p = CustomerProfile(customer_id="c")
    mine = SessionState(session_id="s", customer_id="c", turn_index=1, dialog={"safety_flag": "1"})
    base = AdaptationPolicy().decide("x", HumanState(), p, SessionState(
        session_id="s", customer_id="c", turn_index=1, dialog={"safety_flag": "1"}), {})
    assert bp.decide("x", HumanState(), p, mine, {}) == base


def test_protected_knobs_not_updated(tmp_path):
    bp = mk(tmp_path)
    a, p = turn(bp, explicit={"verbosity": "low"})
    bp.observe(HumanState(), a, 1.0, p)
    assert sum(bp.counts("verbosity").values()) == 0 and sum(bp.counts("warmth").values()) == 1


def test_reward_clipped(tmp_path):
    bp = mk(tmp_path)
    a, p = turn(bp)
    bp.observe(HumanState(), a, 99.0, p)
    assert max(abs(bp.arms["warmth"][a.warmth].b)) <= 1.0


def test_persistence_roundtrip(tmp_path):
    bp = mk(tmp_path, save_every=0)
    bp.fit_rows(make_log(150))
    bp.save()
    assert (tmp_path / "b.json").exists() and not list(tmp_path.glob("*.tmp"))
    bp2 = mk(tmp_path)
    assert bp2.total_obs == bp.total_obs and bp2.counts("verbosity") == bp.counts("verbosity")
    assert turn(bp)[0] == turn(bp2)[0]


def test_autosave_and_corrupt_state(tmp_path):
    bp = mk(tmp_path, save_every=5)
    bp.fit_rows(make_log(5))
    assert (tmp_path / "b.json").exists()
    (tmp_path / "b.json").write_text("{not json")
    assert mk(tmp_path).total_obs == 0


def test_wrong_dimension_state_ignored(tmp_path):
    (tmp_path / "b.json").write_text('{"dim": 3, "arms": {}}')
    assert mk(tmp_path).total_obs == 0


# ----------------------------------------------------------------------------- determinism


def test_same_log_identical_parameters_and_decisions(tmp_path):
    rows = make_log(300)
    a, b = mk(tmp_path / "a", persist=False), mk(tmp_path / "b", persist=False)
    for bp in (a, b):
        bp.fit_rows(rows)
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)
    for ti in range(6):
        st = HumanState(engagement=ti / 6)
        assert turn(a, st, turn_index=ti)[0] == turn(b, st, turn_index=ti)[0]
    a.state_path = None
    a.path, b.path = tmp_path / "a.json", tmp_path / "b.json"
    a.save(), b.save()
    assert a.path.read_bytes() == b.path.read_bytes()


_SCRIPT = textwrap.dedent("""
    import json, sys, hashlib
    sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
    from test_persona_bandit import make_log, mk, turn, HumanState
    from pathlib import Path
    import tempfile
    d = Path(tempfile.mkdtemp())
    bp = mk(d, persist=False)
    bp.fit_rows(make_log(300))
    decs = [turn(bp, HumanState(engagement=i / 7), turn_index=i)[0].model_dump_json() for i in range(8)]
    print(hashlib.sha256(json.dumps([bp.to_dict(), decs], sort_keys=True).encode()).hexdigest())
""")


def test_identical_across_processes(tmp_path):
    from pathlib import Path
    tests = str(Path(__file__).parent)
    code = _SCRIPT.format(root=str(Path(__file__).parent.parent), tests=tests)
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={"PYTHONHASHSEED": str(seed), "PATH": ""}).stdout for seed in (1, 2)}
    assert len(outs) == 1


def test_pipeline_wires_env(tmp_path, monkeypatch):
    from acie.personalization.policy import make_default_policy
    monkeypatch.setenv("ACIE_POLICY", "bandit")
    monkeypatch.setenv("ACIE_BANDIT_STATE", str(tmp_path / "x.json"))
    assert isinstance(make_default_policy(), BanditPolicy)
