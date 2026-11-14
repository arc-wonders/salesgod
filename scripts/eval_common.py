"""Shared helpers for the ACIE evaluation scripts (stdlib only, no acie imports).

Loads and validates data/eval/*.jsonl against docs/DATA_SCHEMA.md and provides
small metric utilities (precision / recall / F1, percentiles) so the eval
scripts don't depend on sklearn / numpy.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "eval" / "results"

LANGS = ("en", "hi", "hinglish")
INTENTS = ("greet", "check_balance", "transfer", "transfer_limit", "card_offer", "human_agent",
           "affirm", "deny", "clarify", "goodbye", "unknown")
STATE_DIMS = ("frustration", "confusion", "urgency", "engagement")
EXPLICIT_VALUES = {"language": {"en", "hi", "hinglish"}, "verbosity": {"low", "high"}, "complexity": {"low"}}
PENDING_VALUES = {None, "transfer", "card_offer"}
SLOT_VALUES = {None, "amount"}


# --------------------------------------------------------------------------- loading / validation


def validate_turn(r: dict) -> list[str]:
    """Return a list of schema problems for one turns.jsonl record (empty = valid)."""
    errs: list[str] = []
    for k in ("id", "lang", "text", "intent", "labels"):
        if k not in r:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if r["lang"] not in LANGS:
        errs.append(f"bad lang {r['lang']!r}")
    if not isinstance(r["text"], str) or not r["text"].strip():
        errs.append("empty text")
    if r["intent"] not in INTENTS:
        errs.append(f"bad intent {r['intent']!r}")
    labels = r.get("labels") or {}
    for d in STATE_DIMS:
        if labels.get(d) not in (0, 1, 2):
            errs.append(f"bad label {d}={labels.get(d)!r}")
    ctx = r.get("context") or {}
    if ctx.get("dialog_pending") not in PENDING_VALUES:
        errs.append(f"bad dialog_pending {ctx.get('dialog_pending')!r}")
    if ctx.get("pending_slot") not in SLOT_VALUES:
        errs.append(f"bad pending_slot {ctx.get('pending_slot')!r}")
    for k, v in (r.get("explicit") or {}).items():
        if k not in EXPLICIT_VALUES or v not in EXPLICIT_VALUES[k]:
            errs.append(f"bad explicit {k}={v!r}")
    amt = r.get("amount")
    if amt is not None and (not isinstance(amt, int) or isinstance(amt, bool) or amt <= 0):
        errs.append(f"bad amount {amt!r}")
    return errs


def validate_conversation(c: dict) -> list[str]:
    errs: list[str] = []
    for k in ("id", "customer_id", "turns"):
        if k not in c:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if not isinstance(c["turns"], list) or not c["turns"]:
        errs.append("no turns")
        return errs
    for i, t in enumerate(c["turns"]):
        if not isinstance(t.get("text"), str) or not t["text"].strip():
            errs.append(f"turn {i}: empty text")
        bad = set((t.get("expect") or {})) - {"intent", "language", "complexity", "verbosity",
                                              "acknowledge_emotion", "formality", "warmth", "proactivity"}
        if bad:
            errs.append(f"turn {i}: unknown expect keys {sorted(bad)}")
    return errs


def load_jsonl(path: Path, validator) -> tuple[list[dict], list[str]]:
    """Load records; invalid lines are skipped and reported, never silently 'fixed'."""
    recs, problems = [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {n}: invalid json ({e})")
                continue
            errs = validator(r)
            if errs:
                problems.append(f"line {n} ({r.get('id', '?')}): " + "; ".join(errs))
                continue
            recs.append(r)
    return recs, problems


def report_problems(problems: list[str], total_ok: int) -> None:
    if problems:
        print(f"WARNING: {len(problems)} invalid record(s) skipped ({total_ok} valid):", file=sys.stderr)
        for p in problems[:20]:
            print("  " + p, file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... {len(problems) - 20} more", file=sys.stderr)


# --------------------------------------------------------------------------- metrics


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def binary_prf(gold: list[bool], pred: list[bool]) -> dict:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    return prf(tp, fp, fn)


def multiclass(gold: list[str], pred: list[str], labels: list[str] | None = None) -> dict:
    labels = labels or sorted(set(gold) | set(pred))
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    per = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        if tp + fp + fn:
            per[lab] = prf(tp, fp, fn)
    supported = [v for v in per.values() if v["support"]]
    macro_f1 = sum(v["f1"] for v in supported) / len(supported) if supported else 0.0
    conf = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return {"n": n, "accuracy": round(acc, 3), "macro_f1": round(macro_f1, 3), "per_label": per,
            "top_confusions": [{"gold": g, "pred": p, "count": c} for (g, p), c in conf.most_common(15)],
            "confusion_matrix": {g: dict(Counter(p for gg, p in zip(gold, pred) if gg == g)) for g in labels
                                 if g in gold}}


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


def acie_fingerprint() -> str:
    """Short hash of acie/**/*.py so results can be tied to the code version that produced them."""
    h = hashlib.sha256()
    for f in sorted((ROOT / "acie").rglob("*.py")):
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]
