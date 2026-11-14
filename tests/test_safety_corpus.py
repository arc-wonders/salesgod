"""Gate run 4 false crises: every ordinary customer line (real calls, persona transcripts, QA fixtures except the crisis
ones) must score below the concern band. The known true crisis lines must still be crisis."""
import glob
import json
from pathlib import Path

import pytest

from acie.safety import CONCERN_T, DISTRESS_T, distress_score

ROOT = Path(__file__).resolve().parents[1]
TRUE_CRISIS_CALLS = {"2ccf3f1387", "f8ad3e646c"}  # real calls where the caller really was in crisis


def _call_lines():
    for f in glob.glob(str(ROOT / "data/real/calls/*.json")) + glob.glob(str(ROOT / "data/real/latest*/*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or Path(f).stem in TRUE_CRISIS_CALLS:
            continue
        for t in d.get("turns", []):
            if t.get("role") in ("customer", "user", "caller") and t.get("text"):
                yield Path(f).stem, t["text"]


def _persona_lines():
    for f in glob.glob("/tmp/claude-1000/personas/**/*.md", recursive=True):
        for line in open(f, encoding="utf-8"):
            if line.startswith("ME:"):
                yield Path(f).name, line[3:].strip()


def _fixture_lines():
    for f in glob.glob(str(ROOT / "tests/qa_fixtures/conversations/*.json")):
        d = json.load(open(f))
        if d.get("class") in ("crisis", "crisis_missed"):
            continue
        for t in d.get("turns", []):
            if t.get("say"):
                yield d["id"], t["say"]


@pytest.mark.parametrize("source", [_call_lines, _persona_lines, _fixture_lines])
def test_no_ordinary_line_reaches_concern(source):
    lines = list(source())
    if not lines:
        pytest.skip("no data on this machine")
    bad = []
    for src, text in lines:
        score, ev = distress_score(text)
        if score >= CONCERN_T:
            bad.append((src, text[:60], score, ev))
    assert not bad, bad


@pytest.mark.parametrize("text", [
    "मैं शाम 7 बजे के बाद जाती हूँ, ऑफ़िस से आकर। Women batch तो 6 बजे तक है, फिर?",
    "ये क्या बार-बार बोल रहे हो? ... मैं खुद आकर देख लूंगा। धन्यवाद।",
    "मैं खुद sector 29 जाकर देख लूँगी। Thanks, bye.",
    "मैं Sector 29 के पास ऑफिस में हूँ, वो ठीक है।",
    "मुझे ये तो बताओ क्या मैं ये offer अभी भी ले सकता हूँ क्योंकि मेरे पास पहले से cloud है मेरा account नया नहीं है",
    "नहीं, मैंने महीने की fees पूछी थी",
    "मैं ठीक हूँ, बस fees पूछ रहा हूँ",
])
def test_known_false_positives(text):
    assert distress_score(text)[0] < CONCERN_T


def test_true_crisis_still_crisis():
    for t in ("मेरा गोल है छट से कूदना", "कि मेरे मरने का मन कर रहा है", "मैं जिन्दगी ते खारा हुआ हूँ मुझे नहीं दीना"):
        assert distress_score(t)[0] >= DISTRESS_T
