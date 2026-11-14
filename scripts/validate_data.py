#!/usr/bin/env python
"""Validate data/eval/turns.jsonl and data/eval/conversations.jsonl against docs/DATA_SCHEMA.md
and print distribution stats. Exit code 1 on any schema error."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TURNS = ROOT / "data/eval/turns.jsonl"
CONVS = ROOT / "data/eval/conversations.jsonl"

LANGS = {"en", "hi", "hinglish"}
INTENTS = {"greet", "check_balance", "transfer", "transfer_limit", "card_offer", "human_agent",
           "affirm", "deny", "clarify", "goodbye", "unknown"}
PLAN_INTENTS = {"greet", "check_balance", "transfer_ask_amount", "transfer_confirm", "transfer_over_limit",
                "transfer_insufficient", "transfer_done", "transfer_limit", "card_offer", "card_declined",
                "card_declined_again", "card_ineligible", "card_apply_started", "cancelled", "human_agent",
                "goodbye", "unknown"}
LABELS = ("frustration", "confusion", "urgency", "engagement")
EXPLICIT = {"language": LANGS, "verbosity": {"low", "high"}, "complexity": {"low"}}
EXPECT = {"intent": PLAN_INTENTS, "language": LANGS, "complexity": {"low", "high", "normal", "medium"},
          "verbosity": {"low", "high", "normal", "medium"}, "acknowledge_emotion": {True, False}}

errors = []


def err(where, msg):
    errors.append(f"{where}: {msg}")


def load(path):
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            err(f"{path.name}:{n}", f"bad JSON: {e}")
    return rows


def check_timing(where, tm):
    if not isinstance(tm, dict):
        return err(where, "timing must be an object")
    if not tm:
        return
    extra = set(tm) - {"barge_in", "response_latency_ms", "speech_rate_wps"}
    if extra:
        err(where, f"unknown timing keys {extra}")
    if "barge_in" in tm and not isinstance(tm["barge_in"], bool):
        err(where, "barge_in must be bool")
    lat = tm.get("response_latency_ms")
    if lat is not None and not (isinstance(lat, int) and 200 <= lat <= 4000):
        err(where, f"implausible response_latency_ms {lat}")
    wps = tm.get("speech_rate_wps")
    if wps is not None and not (isinstance(wps, (int, float)) and 0.8 <= wps <= 5.0):
        err(where, f"implausible speech_rate_wps {wps}")


def check_turns(rows):
    ids = set()
    required = {"id", "lang", "text", "context", "intent", "labels", "explicit", "amount", "notes"}
    for r in rows:
        w = r.get("id", "?")
        missing = required - set(r)
        if missing:
            err(w, f"missing keys {missing}")
            continue
        extra = set(r) - required - {"timing"}
        if extra:
            err(w, f"unknown keys {extra}")
        if w in ids:
            err(w, "duplicate id")
        ids.add(w)
        if r["lang"] not in LANGS:
            err(w, f"bad lang {r['lang']}")
        if not isinstance(r["text"], str) or not r["text"].strip():
            err(w, "empty text")
        if r["intent"] not in INTENTS:
            err(w, f"bad intent {r['intent']}")
        ctx = r["context"]
        if not isinstance(ctx, dict) or set(ctx) != {"prev_customer_text", "prev_agent_text", "dialog_pending", "pending_slot"}:
            err(w, f"context must have exactly the 4 schema keys, got {ctx}")
        else:
            if ctx["dialog_pending"] not in (None, "transfer", "card_offer"):
                err(w, f"bad dialog_pending {ctx['dialog_pending']}")
            if ctx["pending_slot"] not in (None, "amount"):
                err(w, f"bad pending_slot {ctx['pending_slot']}")
            if ctx["pending_slot"] == "amount" and ctx["dialog_pending"] != "transfer":
                err(w, "pending_slot amount without dialog_pending transfer")
            for k in ("prev_customer_text", "prev_agent_text"):
                if ctx[k] is not None and not isinstance(ctx[k], str):
                    err(w, f"{k} must be string or null")
        lab = r["labels"]
        if not isinstance(lab, dict) or set(lab) != set(LABELS):
            err(w, f"labels must have {LABELS}")
        else:
            for k, v in lab.items():
                if v not in (0, 1, 2) or isinstance(v, bool):
                    err(w, f"label {k}={v} not in 0/1/2")
        ex = r["explicit"]
        if not isinstance(ex, dict):
            err(w, "explicit must be an object")
        else:
            for k, v in ex.items():
                if k not in EXPLICIT:
                    err(w, f"bad explicit key {k}")
                elif v not in EXPLICIT[k]:
                    err(w, f"bad explicit {k}={v}")
        amt = r["amount"]
        if amt is not None and (not isinstance(amt, int) or isinstance(amt, bool) or amt <= 0):
            err(w, f"amount must be positive int or null, got {amt}")
        if "timing" in r:
            check_timing(w, r["timing"])


def check_convs(rows):
    ids = set()
    for c in rows:
        w = c.get("id", "?")
        for k in ("id", "customer_id", "description", "turns"):
            if k not in c:
                err(w, f"missing {k}")
        if w in ids:
            err(w, "duplicate id")
        ids.add(w)
        if c.get("customer_id") not in ("c1", "c2"):
            err(w, f"bad customer_id {c.get('customer_id')}")
        turns = c.get("turns", [])
        if not 4 <= len(turns) <= 9:
            err(w, f"{len(turns)} turns (expected 4-9)")
        for i, t in enumerate(turns):
            tw = f"{w}[{i}]"
            if not isinstance(t.get("text"), str) or not t["text"].strip():
                err(tw, "missing text")
            extra = set(t) - {"text", "timing", "expect"}
            if extra:
                err(tw, f"unknown keys {extra}")
            if "timing" in t:
                check_timing(tw, t["timing"])
            for k, v in t.get("expect", {}).items():
                if k not in EXPECT:
                    err(tw, f"bad expect key {k}")
                elif v not in EXPECT[k]:
                    err(tw, f"bad expect {k}={v}")


def pct(n, d):
    return f"{n} ({100 * n / d:.1f}%)" if d else "0"


def stats(turns, convs):
    n = len(turns)
    print(f"== turns.jsonl: {n} turns")
    print("lang:", {k: pct(v, n) for k, v in collections.Counter(r["lang"] for r in turns).most_common()})
    print("intent:", dict(collections.Counter(r["intent"] for r in turns).most_common()))
    print("intent x lang:")
    ct = collections.Counter((r["intent"], r["lang"]) for r in turns)
    for it in sorted(INTENTS):
        print(f"  {it:15s} " + " ".join(f"{l}={ct[(it, l)]:3d}" for l in ("en", "hi", "hinglish")))
    for k in LABELS:
        c = collections.Counter(r["labels"][k] for r in turns)
        print(f"{k:12s} 0={c[0]:3d} 1={c[1]:3d} 2={c[2]:3d}")
    print("frustration>=1:", sum(r["labels"]["frustration"] >= 1 for r in turns),
          " confusion>=1:", sum(r["labels"]["confusion"] >= 1 for r in turns),
          " urgency>=1:", sum(r["labels"]["urgency"] >= 1 for r in turns),
          " engagement=0:", sum(r["labels"]["engagement"] == 0 for r in turns))
    exs = [r for r in turns if r["explicit"]]
    print("explicit:", pct(len(exs), n), dict(collections.Counter(f"{k}={v}" for r in exs for k, v in r["explicit"].items())))
    print("amount non-null:", sum(r["amount"] is not None for r in turns))
    print("with context (prev_agent_text):", sum(r["context"]["prev_agent_text"] is not None for r in turns),
          " dialog_pending:", dict(collections.Counter(r["context"]["dialog_pending"] for r in turns)))
    tm = [r["timing"] for r in turns if "timing" in r]
    if tm:
        lats = sorted(t["response_latency_ms"] for t in tm)
        print(f"timing present: {len(tm)}  barge_in: {pct(sum(t['barge_in'] for t in tm), len(tm))}"
              f"  latency median={lats[len(lats) // 2]} min={lats[0]} max={lats[-1]}")
    print(f"\n== conversations.jsonl: {len(convs)} conversations")
    print("customer:", dict(collections.Counter(c["customer_id"] for c in convs)))
    lens = [len(c["turns"]) for c in convs]
    print(f"turns/conv: min={min(lens)} max={max(lens)} total={sum(lens)}")

    def script(c):
        txt = " ".join(t["text"] for t in c["turns"])
        dev = sum("ऀ" <= ch <= "ॿ" for ch in txt)
        let = sum(ch.isalpha() for ch in txt)
        r = dev / let if let else 0
        return "mainly_hi" if r > 0.7 else "mainly_en_or_roman" if r < 0.1 else "mixed"
    print("script mix:", dict(collections.Counter(script(c) for c in convs)))
    ex = collections.Counter(k for c in convs for t in c["turns"] for k in t.get("expect", {}))
    print("expect keys:", dict(ex))
    print("expect intents:", dict(collections.Counter(t["expect"]["intent"] for c in convs for t in c["turns"] if "intent" in t.get("expect", {}))))


def main():
    turns, convs = load(TURNS), load(CONVS)
    check_turns(turns)
    check_convs(convs)
    if errors:
        print(f"{len(errors)} schema error(s):")
        for e in errors:
            print("  ", e)
    stats(turns, convs)
    if errors:
        sys.exit(1)
    print("\nOK: both files pass schema validation")


if __name__ == "__main__":
    main()
