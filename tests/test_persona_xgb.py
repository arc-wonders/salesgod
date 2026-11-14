import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

from acie.personalization.train_style import XGB_PARAMS, main as train_main, train  # noqa: E402
from acie.personalization.xgb_style import XGBStylePolicy  # noqa: E402
from test_persona_bandit import HumanState, make_log, turn  # noqa: E402


def trained(tmp_path, rows, name="m.json"):
    log = tmp_path / (name + ".jsonl")
    log.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    out = tmp_path / name
    train_main(["--log", str(log), "--out", str(out), "--min-obs", "30"])
    return out


def mkx(tmp_path, model, **kw):
    return XGBStylePolicy(model_path=model, log_path=tmp_path / "log.jsonl", **kw)


def test_fixed_params():
    assert XGB_PARAMS == dict(tree_method="exact", n_estimators=200, max_depth=4, learning_rate=0.1, subsample=1.0,
                              colsample_bytree=1.0, n_jobs=1, random_state=0)


def test_no_model_is_rule_policy(tmp_path):
    from acie.personalization.policy import AdaptationPolicy
    pol = mkx(tmp_path, tmp_path / "missing.json")
    p = pol_p = None
    from acie.schemas import CustomerProfile, SessionState
    p = CustomerProfile(customer_id="c")
    got = pol.decide("hi", HumanState(), p, SessionState(session_id="s", customer_id="c"), {})
    assert got == AdaptationPolicy().decide("hi", HumanState(), p, SessionState(session_id="s", customer_id="c"), {})


def test_learns_preferred_action(tmp_path):
    pol = mkx(tmp_path, trained(tmp_path, make_log(400)))
    assert turn(pol)[0].verbosity == "low"


def test_rule_prior_wins_without_margin_or_data(tmp_path):
    rows = make_log(400, pref="med")
    assert turn(mkx(tmp_path, trained(tmp_path, rows)))[0].verbosity == "med"
    assert turn(mkx(tmp_path, trained(tmp_path, make_log(400)), min_obs=10_000))[0].verbosity == "med"
    assert turn(mkx(tmp_path, trained(tmp_path, make_log(400)), margin=5.0))[0].verbosity == "med"


def test_explicit_and_safety_never_overridden(tmp_path):
    pol = mkx(tmp_path, trained(tmp_path, make_log(400)), margin=-1.0, min_obs=1)
    a, _ = turn(pol, explicit={"verbosity": "high"})
    assert a.verbosity == "high"
    b, _ = turn(pol, HumanState(frustration=0.9))
    assert b.warmth == "high" and b.verbosity == "low"


def test_observe_only_logs_never_changes_model(tmp_path):
    model = trained(tmp_path, make_log(400))
    before = model.read_bytes()
    pol = mkx(tmp_path, model)
    d0 = turn(pol)[0]
    for _ in range(20):
        a, p = turn(pol)
        pol.observe(HumanState(), a, -1.0, p)
    assert turn(pol)[0] == d0 and model.read_bytes() == before
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 20 and "reward" in json.loads(lines[0])


def test_training_is_byte_identical(tmp_path):
    rows = make_log(300)
    assert trained(tmp_path, rows, "a.json").read_bytes() == trained(tmp_path, rows, "b.json").read_bytes()
    assert train(rows)["arms"]["verbosity"]["low"]["model"] == train(rows)["arms"]["verbosity"]["low"]["model"]


_SCRIPT = textwrap.dedent("""
    import sys, hashlib
    sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
    from pathlib import Path
    import tempfile
    from test_persona_xgb import trained, mkx, turn, HumanState, make_log
    d = Path(tempfile.mkdtemp())
    model = trained(d, make_log(300))
    pol = mkx(d, model)
    decs = [turn(pol, HumanState(engagement=i / 7), turn_index=i)[0].model_dump_json() for i in range(8)]
    print(hashlib.sha256(model.read_bytes() + "".join(decs).encode()).hexdigest())
""")


def test_identical_across_processes(tmp_path):
    here = Path(__file__).parent
    code = _SCRIPT.format(root=str(here.parent), tests=str(here))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={"PYTHONHASHSEED": str(s), "PATH": ""}).stdout.splitlines()[-1] for s in (1, 2)}
    assert len(outs) == 1


def test_env_selects_xgb(tmp_path, monkeypatch):
    from acie.personalization.policy import make_default_policy
    monkeypatch.setenv("ACIE_POLICY", "xgb")
    monkeypatch.setenv("ACIE_STYLE_MODEL", str(tmp_path / "none.json"))
    assert isinstance(make_default_policy(), XGBStylePolicy)
