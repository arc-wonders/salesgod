"""Conversation regression harness as a pytest suite (scripts/qa/run_qa.py).

* Always asserted: fixtures are well formed, no run crashes, every template run is deterministic.
* Expectation failures are reported as xfail (reason lists the failing checks and class) so the suite stays green while
  the fix lanes land; a fixture that starts passing shows as a plain pass. The overall pass rate is printed at the end.
* CRITICAL classes (fixture field "critical": true, e.g. crisis_missed) are xfail with the fixture's `xfail_reason`
  until the fixture's `"hard"` is flipped to true or QA_HARD_CRITICAL=1 is set: then they are hard failures.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qa"))
sys.path.insert(0, str(ROOT))
import run_qa  # noqa: E402

FIXTURES = run_qa.load_fixtures()
CONFIGS = ("defaults", "psyche")
FX_BY_ID = {f["id"]: f for f in FIXTURES}
TURN_KEYS = {"say", "tags", "lang", "skip", "expect_lang", "must"}


@pytest.fixture(scope="module")
def report():
    return run_qa.run_harness(FIXTURES, CONFIGS, "template")


def test_fixture_inventory():
    assert len(FIXTURES) >= 40
    assert len({f["id"] for f in FIXTURES}) == len(FIXTURES)
    langs = {f["lang"] for f in FIXTURES}
    assert {"en", "hi", "hinglish"} <= langs
    assert len({f["class"] for f in FIXTURES}) >= 20


@pytest.mark.parametrize("fx", FIXTURES, ids=lambda f: f["id"])
def test_fixture_well_formed(fx):
    assert fx["campaign"] in run_qa.load_products()
    assert fx["turns"], "no turns"
    for t in fx["turns"]:
        assert set(t) <= TURN_KEYS, set(t) - TURN_KEYS
        assert t["say"].strip()


def test_no_crashes(report):
    assert report["crashes"] == []


def test_deterministic(report):
    assert report["nondeterministic"] == []


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("fid", sorted(FX_BY_ID))
def test_expectations(report, cfg, fid):
    fx = FX_BY_ID[fid]
    fails = report["configs"][cfg][fid]["fails"]
    if not fails:
        return
    why = f"[{fx['class']}] " + "; ".join(sorted({f['check'] for f in fails}))
    first = fails[0]
    detail = f"{why} | t{first['turn']} {first['check']}: {first['detail']}"
    hard = fx.get("critical") and (fx.get("hard") or os.getenv("QA_HARD_CRITICAL") == "1")
    if hard:
        pytest.fail(detail)
    if fx.get("critical"):
        pytest.xfail(fx.get("xfail_reason", "critical") + " | " + detail)
    pytest.xfail(detail)


def test_zz_print_pass_rate(report, capsys):
    with capsys.disabled():
        print("\n\n=== QA harness pass rate (conversations with zero failed checks) ===")
        run_qa.print_table(report)
        print()
