"""Compliance regressions: the number-source question and the removal offer (en / hi / hinglish, both configs)."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qa"))
import run_qa  # noqa: E402

PRODS = run_qa.load_products()


def play(lines, lang="hinglish", psyche="", camp="fitzone-gym"):
    old = os.environ.get("ACIE_PSYCHE")
    os.environ["ACIE_PSYCHE"] = psyche
    try:
        fx = {"id": "t", "campaign": camp, "opening_language": lang, "turns": [{"say": s} for s in lines], "class": "x"}
        return asyncio.run(run_qa.play(fx, PRODS))[1:]
    finally:
        if old is None:
            os.environ.pop("ACIE_PSYCHE", None)
        else:
            os.environ["ACIE_PSYCHE"] = old


CASES = [
    ("hinglish", "mera number kahan se mila aapko", "haan hata do"),
    ("hinglish", "kisne diya mera number", "ha hatao"),
    ("hi", "मेरा नंबर कहाँ से मिला आपको", "हाँ हटाओ नंबर"),
    ("hi", "मेरा नंबर कैसे मिला", "हाँ हटा दो"),
    ("en", "where did you get my number?", "yes remove it"),
    ("en", "which list is this number from", "yes please"),
]


@pytest.mark.parametrize("psyche", ["", "1"])
@pytest.mark.parametrize("lang,q,yes", CASES)
def test_number_source_then_remove(lang, q, yes, psyche):
    r = play(["yes", q, yes], lang, psyche)
    assert r[1]["intent"] == "number_source"
    low = r[1]["reply"].lower()
    assert "branch" not in low and "sector" not in low and "ब्रांच" not in low and "सेक्टर" not in low  # never volunteer branch facts
    assert r[2]["intent"] == "opt_out" and r[2]["end_call"]
    assert any(w in r[2]["reply"].lower() for w in ("removed", "hata diya", "हटा दिया"))
    assert any(w in r[2]["reply"].lower() for w in ("sorry", "maafi", "माफ़ी"))


@pytest.mark.parametrize("lang,q", [(c[0], c[1]) for c in CASES])
def test_number_source_never_a_branch_answer(lang, q):
    r = play([q], lang)
    assert r[0]["intent"] == "number_source"
    assert "49" not in r[0]["reply"] and "गुरुग्राम" not in r[0]["reply"]


def test_asked_twice_still_honest_with_removal_offer():
    r = play(["yes", "where did you get my number?", "no, but tell me again where you got my number"], "en")
    assert r[2]["intent"] == "number_source" and "remove" in r[2]["reply"].lower() or "off it" in r[2]["reply"].lower()


def test_where_is_the_gym_still_gets_branch_answer():
    r = play(["yes", "gym kahan hai aapka"], "hinglish")
    assert "Saket" in r[1]["reply"] or "Sector" in r[1]["reply"]
