"""Data loading / encoding for the ACIE multi-task turn classifier. Stdlib only.

Input format (turns.jsonl, see docs/DATA_SCHEMA.md) -> a (context, customer) text
pair, encoded by the tokenizer as a sentence pair so the model sees
  [CLS] <context> [SEP] <customer text> [SEP]
where <context> carries what the rule system also sees: previous agent / customer
utterance, the pending dialog question, the pending slot, and bucketed timing.

Targets:
  intent    : softmax over INTENTS
  states    : 4 heads x 3 classes (0 absent / 1 mild / 2 clear; engagement 0 disengaged / 1 normal / 2 high)
  explicit  : multi-label over EXPLICIT_LABELS ("language=hi", ...)

Quick check without torch:
  python training/state_classifier/data.py --data scripts/fixtures/smoke_turns.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "data" / "train" / "turns.jsonl"
EVAL_DIR = ROOT / "data" / "eval"

INTENTS = ["greet", "check_balance", "transfer", "transfer_limit", "card_offer", "human_agent",
           "affirm", "deny", "clarify", "goodbye", "unknown"]
GATED_INTENTS = {"affirm", "deny"}  # only valid when a yes/no question is pending (mirrors bank_demo GATED)
STATE_DIMS = ["frustration", "confusion", "urgency", "engagement"]
EXPLICIT_LABELS = ["language=en", "language=hi", "language=hinglish", "verbosity=low", "verbosity=high",
                   "complexity=low"]


# --------------------------------------------------------------------------- input text


def _timing_words(timing: dict | None) -> list[str]:
    t = timing or {}
    w = []
    if t.get("barge_in"):
        w.append("barge-in")
    lat = t.get("response_latency_ms")
    if lat is not None:
        w.append("reply fast" if lat < 500 else "reply slow" if lat > 2500 else "reply normal")
    rate = t.get("speech_rate_wps")
    if rate is not None:
        w.append("speech fast" if rate > 3.5 else "speech slow" if rate < 1.5 else "speech normal")
    return w


def build_context(prev_agent: str | None = None, prev_customer: str | None = None,
                  pending: str | None = None, pending_slot: str | None = None,
                  timing: dict | None = None) -> str:
    """Context segment. Used identically at train time (from jsonl) and inference (from SessionState)."""
    parts = []
    if pending:
        parts.append(f"pending: {pending}")
    if pending_slot:
        parts.append(f"slot: {pending_slot}")
    tw = _timing_words(timing)
    if tw:
        parts.append("timing: " + ", ".join(tw))
    if prev_customer:
        parts.append(f"prev customer: {prev_customer}")
    if prev_agent:
        parts.append(f"prev agent: {prev_agent}")
    return " | ".join(parts) if parts else "start of call"


def record_inputs(rec: dict, drop_timing: bool = False) -> tuple[str, str]:
    """drop_timing: training-time augmentation. IntentClassifier.classify(text, session) never sees
    timing, and telephony may not provide it, so the model must work without it."""
    ctx = rec.get("context") or {}
    return (build_context(ctx.get("prev_agent_text"), ctx.get("prev_customer_text"), ctx.get("dialog_pending"),
                          ctx.get("pending_slot"), None if drop_timing else rec.get("timing")),
            rec["text"])


# --------------------------------------------------------------------------- targets


def record_targets(rec: dict) -> dict:
    ex = rec.get("explicit") or {}
    return {
        "intent": INTENTS.index(rec["intent"]),
        "states": [int(rec["labels"][d]) for d in STATE_DIMS],
        "explicit": [1.0 if ex.get(k) == v else 0.0 for k, v in (lbl.split("=") for lbl in EXPLICIT_LABELS)],
    }


def validate(rec: dict) -> str | None:
    try:
        if not isinstance(rec.get("text"), str) or not rec["text"].strip():
            return "empty text"
        if rec.get("intent") not in INTENTS:
            return f"bad intent {rec.get('intent')!r}"
        for d in STATE_DIMS:
            if rec["labels"].get(d) not in (0, 1, 2):
                return f"bad label {d}"
    except (KeyError, TypeError, AttributeError) as e:
        return f"malformed: {e}"
    return None


def load(path: str | Path) -> list[dict]:
    recs, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if validate(r):
                bad += 1
                continue
            recs.append(r)
    if bad:
        print(f"[data] skipped {bad} invalid line(s) in {path}", file=sys.stderr)
    return recs


# --------------------------------------------------------------------------- leakage guard


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[\wऀ-ॿ]+", s.lower()))


def assert_not_eval(path: str | Path) -> None:
    """Held-out eval data must never be used for training."""
    p = Path(path).resolve()
    if EVAL_DIR.resolve() in p.parents:
        raise SystemExit(f"refusing to train on {p}: data/eval/ is the held-out evaluation set. "
                         f"Train on synthetic data in data/train/ instead.")


def drop_eval_overlap(train: list[dict], eval_path: Path = EVAL_DIR / "turns.jsonl") -> list[dict]:
    """Remove training rows whose normalized text appears verbatim in the eval set."""
    if not eval_path.exists():
        return train
    eval_texts = {_norm(r["text"]) for r in load(eval_path)}
    kept = [r for r in train if _norm(r["text"]) not in eval_texts]
    if len(kept) != len(train):
        print(f"[data] dropped {len(train) - len(kept)} training rows that duplicate eval texts", file=sys.stderr)
    return kept


def split(recs: list[dict], dev_frac: float = 0.1, seed: int = 13) -> tuple[list[dict], list[dict]]:
    """Per-(lang, intent) split so the dev set covers every bucket."""
    rng = random.Random(seed)
    buckets: dict[tuple, list[dict]] = {}
    for r in recs:
        buckets.setdefault((r.get("lang"), r["intent"]), []).append(r)
    train, dev = [], []
    for rows in buckets.values():
        rng.shuffle(rows)
        k = int(round(len(rows) * dev_frac)) if len(rows) >= 5 else 0
        dev += rows[:k]
        train += rows[k:]
    rng.shuffle(train)
    return train, dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_TRAIN))
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()
    recs = load(args.data)
    print(f"{len(recs)} records from {args.data}")
    print("lang:   ", dict(Counter(r.get("lang") for r in recs)))
    print("intent: ", dict(Counter(r["intent"] for r in recs)))
    for d in STATE_DIMS:
        print(f"{d:<12}", dict(sorted(Counter(r['labels'][d] for r in recs).items())))
    print("explicit:", dict(Counter(l for r in recs for l, y in zip(EXPLICIT_LABELS, record_targets(r)["explicit"]) if y)))
    tr, dv = split(recs)
    print(f"split: train {len(tr)} / dev {len(dv)}")
    for r in recs[: args.show]:
        print("---")
        print("context :", record_inputs(r)[0])
        print("customer:", record_inputs(r)[1])
        print("targets :", record_targets(r))


if __name__ == "__main__":
    main()
