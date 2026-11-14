"""Per-turn signal audit: are ACIE's v1 signals actually correct on realistic ASR-like input?

For every labeled turn in data/eval/turns.jsonl this runs, exactly as the live
pipeline would on that turn:
  - the bank_demo intent classifier (raw, and with the domain's slot-fill rule)
  - HumanStateEstimator.estimate (single turn from the neutral prior) + raw RuleScorer scores
  - detect_explicit_requests, detect_language, parse_amount
and reports accuracy / P / R / F1 overall and per language.

  .venv/bin/python scripts/eval_signals.py                       # data/eval/turns.jsonl
  .venv/bin/python scripts/eval_signals.py --lang hi
  .venv/bin/python scripts/eval_signals.py --data scripts/fixtures/smoke_turns.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_common import (acie_fingerprint, INTENTS, LANGS, RESULTS_DIR, ROOT, STATE_DIMS, binary_prf, load_jsonl,  # noqa: E402
                         multiclass, prf, report_problems, validate_turn)

from acie.decision.domains.bank_demo import BankDemoDomain  # noqa: E402
from acie.personalization.policy import DISENGAGED_ENTER, ENTER, detect_explicit_requests  # noqa: E402
from acie.schemas import SessionState, TurnInput, TurnTiming, Utterance  # noqa: E402
from acie.state.estimator import HumanStateEstimator, RuleScorer  # noqa: E402
from acie.text import detect_language, parse_amount  # noqa: E402

SWEEP = [round(0.2 + 0.05 * i, 2) for i in range(13)]  # 0.20 .. 0.80


# --------------------------------------------------------------------------- run signals


def build_session(r: dict) -> SessionState:
    ctx = r.get("context") or {}
    s = SessionState(session_id=r["id"], customer_id="c1")
    if ctx.get("prev_customer_text"):
        s.history.append(Utterance(role="customer", text=ctx["prev_customer_text"]))
    if ctx.get("prev_agent_text"):
        s.history.append(Utterance(role="agent", text=ctx["prev_agent_text"]))
    if ctx.get("dialog_pending"):
        s.dialog["pending"] = ctx["dialog_pending"]
    if ctx.get("pending_slot"):
        s.dialog["pending_slot"] = ctx["pending_slot"]
    return s


def timing_of(r: dict) -> TurnTiming:
    t = r.get("timing") or {}
    return TurnTiming(**{k: v for k, v in t.items() if k in TurnTiming.model_fields and v is not None})


def run(records: list[dict]) -> list[dict]:
    classifier = BankDemoDomain().classifier
    estimator = HumanStateEstimator()
    scorer = RuleScorer()
    out = []
    for r in records:
        session = build_session(r)
        turn = TurnInput(session_id=r["id"], customer_id="c1", text=r["text"], timing=timing_of(r))
        intent = classifier.classify(r["text"], session)
        amount = parse_amount(r["text"])
        # bank_demo.plan() promotes `unknown` to `transfer` when an amount slot is pending and a number parses
        eff = "transfer" if (intent == "unknown" and session.dialog.get("pending_slot") == "amount"
                             and amount) else intent
        raw, ev = scorer.score(turn, session)
        smoothed = estimator.estimate(turn, session)
        out.append({
            "rec": r,
            "intent_raw": intent,
            "intent": eff,
            "state": {d: getattr(smoothed, d) for d in STATE_DIMS},
            "raw": raw,
            "signals": ev,
            "explicit": detect_explicit_requests(r["text"]),
            "language": detect_language(r["text"]),
            "amount": amount,
        })
    return out


# --------------------------------------------------------------------------- metrics


def state_score(p: dict, dim: str, which: str) -> float:
    v = p[which][dim]
    return 1.0 - v if dim == "engagement" else v  # engagement: score of being DISengaged


def state_gold(r: dict, dim: str, level: str) -> bool:
    lab = r["labels"][dim]
    if dim == "engagement":
        return lab == 0  # positive = disengaged
    return lab >= 1 if level == "ge1" else lab == 2


def state_metrics(preds: list[dict]) -> dict:
    res = {}
    for dim in STATE_DIMS:
        res[dim] = {}
        levels = ["disengaged"] if dim == "engagement" else ["ge1", "eq2"]
        for level in levels:
            gold = [state_gold(p["rec"], dim, level) for p in preds]
            block = {"positives": sum(gold), "n": len(gold)}
            for which in ("state", "raw"):
                scores = [state_score(p, dim, which) for p in preds]
                thr = DISENGAGED_ENTER if dim == "engagement" else ENTER  # policy's own threshold
                at_enter = binary_prf(gold, [s >= thr for s in scores])
                sweep = {str(t): binary_prf(gold, [s >= t for s in scores])["f1"] for t in SWEEP}
                best_t = max(SWEEP, key=lambda t: (sweep[str(t)], -abs(t - ENTER)))
                block["smoothed" if which == "state" else "raw"] = {
                    "at_enter": at_enter, "sweep_f1": sweep,
                    "best_threshold": best_t, "best_f1": sweep[str(best_t)],
                }
            res[dim][level] = block
    return res


def explicit_metrics(preds: list[dict]) -> dict:
    exact = sum(p["explicit"] == (p["rec"].get("explicit") or {}) for p in preds)
    per = {}
    for key in ("language", "verbosity", "complexity"):
        tp = fp = fn = 0
        for p in preds:
            g, q = (p["rec"].get("explicit") or {}).get(key), p["explicit"].get(key)
            if q is not None and q == g:
                tp += 1
            else:
                fp += q is not None
                fn += g is not None
        per[key] = prf(tp, fp, fn)
    with_any = [p for p in preds if p["rec"].get("explicit")]
    return {"exact_match": round(exact / len(preds), 3) if preds else 0.0,
            "exact_match_on_turns_with_request": round(
                sum(p["explicit"] == p["rec"]["explicit"] for p in with_any) / len(with_any), 3) if with_any else None,
            "turns_with_request": len(with_any),
            "false_request_rate_on_turns_without": round(
                sum(bool(p["explicit"]) for p in preds if not p["rec"].get("explicit"))
                / max(1, len(preds) - len(with_any)), 3),
            "per_key": per}


def amount_metrics(preds: list[dict]) -> dict:
    has = [p for p in preds if p["rec"].get("amount") is not None]
    none = [p for p in preds if p["rec"].get("amount") is None]
    return {
        "n_with_amount": len(has),
        "exact_match": round(sum(p["amount"] == p["rec"]["amount"] for p in has) / len(has), 3) if has else None,
        "missed_none": sum(p["amount"] is None for p in has),
        "wrong_value": sum(p["amount"] is not None and p["amount"] != p["rec"]["amount"] for p in has),
        "n_without_amount": len(none),
        "false_positive_rate": round(sum(p["amount"] is not None for p in none) / len(none), 3) if none else None,
    }


def compute(preds: list[dict]) -> dict:
    if not preds:
        return {"n": 0}
    gold_i = [p["rec"]["intent"] for p in preds]
    lang_gold = [p["rec"]["lang"] for p in preds]
    lang_pred = [p["language"] for p in preds]
    return {
        "n": len(preds),
        "intent": multiclass(gold_i, [p["intent"] for p in preds], list(INTENTS)),
        "intent_classifier_only_accuracy": round(
            sum(g == p["intent_raw"] for g, p in zip(gold_i, preds)) / len(preds), 3),
        "state": state_metrics(preds),
        "explicit": explicit_metrics(preds),
        "language": {"accuracy": round(sum(g == q for g, q in zip(lang_gold, lang_pred)) / len(preds), 3),
                     **{k: v for k, v in multiclass(lang_gold, lang_pred, list(LANGS)).items()
                        if k in ("per_label", "confusion_matrix")}},
        "amount": amount_metrics(preds),
    }


# --------------------------------------------------------------------------- reporting


def fmt(x) -> str:
    return "  -  " if x is None else f"{x:.2f}"


def print_table(metrics: dict, langs: list[str]) -> None:
    cols = ["all"] + langs
    blocks = {c: metrics["overall"] if c == "all" else metrics["by_lang"].get(c, {"n": 0}) for c in cols}

    def row(name: str, getter, pair=None) -> None:
        vals = []
        for c in cols:
            b = blocks[c]
            try:
                if not b.get("n"):
                    vals.append("  -  ")
                elif pair:
                    a, z = pair[0](b), pair[1](b)
                    vals.append("  -  " if a is None else f"{a:.2f}/{z:.2f}")
                else:
                    vals.append(fmt(getter(b)))
            except (KeyError, TypeError):
                vals.append("  -  ")
        print(f"{name:<52}" + "".join(f"{v:>11}" for v in vals))

    print(f"\n{'metric':<52}" + "".join(f"{c:>11}" for c in cols))
    print(f"{'n turns':<52}" + "".join(f"{blocks[c].get('n', 0):>11}" for c in cols))
    print("-" * (52 + 11 * len(cols)))
    row("intent accuracy (pipeline-effective)", lambda b: b["intent"]["accuracy"])
    row("intent macro-F1", lambda b: b["intent"]["macro_f1"])
    for dim in STATE_DIMS:
        levels = ["disengaged"] if dim == "engagement" else ["ge1", "eq2"]
        for lv in levels:
            def g(b, d=dim, l=lv, which="smoothed", key="f1"):
                blk = b["state"][d][l]
                if not blk["positives"]:
                    return None  # undefined without positives
                return blk[which]["best_f1"] if key == "best" else blk[which]["at_enter"][key]
            row(f"{dim}[{lv}] F1 @ENTER={ENTER} (smoothed)", g)
            row(f"{dim}[{lv}] P / R @ENTER (smoothed)", None, pair=(lambda b, g=g: g(b, key="precision"),
                                                                   lambda b, g=g: g(b, key="recall")))
            row(f"{dim}[{lv}] best F1, sweep 0.2-0.8 (raw)", lambda b, g=g: g(b, which="raw", key="best"))
    row("explicit pref exact match", lambda b: b["explicit"]["exact_match"])
    row("explicit exact (turns with a request)", lambda b: b["explicit"]["exact_match_on_turns_with_request"])
    row("language detection accuracy", lambda b: b["language"]["accuracy"])
    row("amount exact match (gold not null)", lambda b: b["amount"]["exact_match"])
    row("amount false-positive rate (gold null)", lambda b: b["amount"]["false_positive_rate"])
    print()
    conf = metrics["overall"]["intent"]["top_confusions"][:8]
    if conf:
        print("top intent confusions (gold -> pred): " + ", ".join(f"{c['gold']}->{c['pred']} x{c['count']}" for c in conf))


def _line(p: dict, extra: str) -> str:
    r = p["rec"]
    ctx = r.get("context") or {}
    c = []
    if ctx.get("dialog_pending"):
        c.append(f"pending={ctx['dialog_pending']}")
    if ctx.get("pending_slot"):
        c.append(f"slot={ctx['pending_slot']}")
    cs = f" [{', '.join(c)}]" if c else ""
    return f"| {r['id']} | {r['lang']} | {r['text'].replace('|', '/')}{cs} | {extra} |"


def write_errors(preds: list[dict], path: Path) -> None:
    L = [f"# Signal errors ({datetime.now().isoformat(timespec='seconds')}, n={len(preds)})", ""]

    def section(title: str, rows: list[str], total: int) -> None:
        L.extend([f"## {title} ({total} errors, showing {min(15, len(rows))})", "",
                  "| id | lang | text | gold vs predicted |", "|---|---|---|---|", *rows[:15], ""])

    # intent: show errors from the most common confusions first
    errs = [p for p in preds if p["intent"] != p["rec"]["intent"]]
    freq = Counter((p["rec"]["intent"], p["intent"]) for p in errs)
    # round-robin over confusion pairs (most frequent first) so one pair doesn't fill the table
    rank: dict[tuple, int] = {}
    seen: Counter = Counter()
    for p in errs:
        k = (p["rec"]["intent"], p["intent"])
        rank[p["rec"]["id"]] = seen[k]
        seen[k] += 1
    errs.sort(key=lambda p: (rank[p["rec"]["id"]], -freq[(p["rec"]["intent"], p["intent"])]))
    section("Intent", [_line(p, f"gold **{p['rec']['intent']}** / pred {p['intent']}") for p in errs], len(errs))

    for dim in STATE_DIMS:
        def gap(p, d=dim):
            g = (2 - p["rec"]["labels"][d]) / 2 if d == "engagement" else p["rec"]["labels"][d] / 2
            return g - state_score(p, d, "state")
        bad = []
        for p in preds:
            lab = p["rec"]["labels"][dim]
            pos = lab == 0 if dim == "engagement" else lab >= 1
            fired = state_score(p, dim, "state") >= ENTER
            if pos != fired:
                bad.append(p)
        bad.sort(key=lambda p: -abs(gap(p)))
        name = "disengaged (engagement label 0)" if dim == "engagement" else f"{dim} (label>=1)"
        section(f"State: {name} @ENTER", [
            _line(p, f"gold label {p['rec']['labels'][dim]} / score {state_score(p, dim, 'state'):.2f} "
                     f"(raw {state_score(p, dim, 'raw'):.2f}); signals: {'; '.join(p['signals']) or 'none'}")
            for p in bad], len(bad))

    bad = [p for p in preds if p["explicit"] != (p["rec"].get("explicit") or {})]
    section("Explicit preferences", [_line(p, f"gold {json.dumps(p['rec'].get('explicit') or {}, ensure_ascii=False)} "
                                              f"/ pred {json.dumps(p['explicit'], ensure_ascii=False)}") for p in bad], len(bad))
    bad = [p for p in preds if p["language"] != p["rec"]["lang"]]
    section("Language detection", [_line(p, f"gold {p['rec']['lang']} / pred {p['language']}") for p in bad], len(bad))
    bad = [p for p in preds if p["amount"] != p["rec"].get("amount")]
    miss = [p for p in bad if p["rec"].get("amount") is not None]
    fpos = [p for p in bad if p["rec"].get("amount") is None]
    bad = miss[:10] + fpos[:5] + miss[10:]  # show some false positives too
    section("Amount", [_line(p, f"gold {p['rec'].get('amount')} / pred {p['amount']}") for p in bad], len(bad))
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "eval" / "turns.jsonl"))
    ap.add_argument("--lang", choices=LANGS, help="evaluate only this language")
    ap.add_argument("--out", default=str(RESULTS_DIR), help="results directory")
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"no data at {path} (smoke test: --data scripts/fixtures/smoke_turns.jsonl)", file=sys.stderr)
        return 2
    records, problems = load_jsonl(path, validate_turn)
    report_problems(problems, len(records))
    if args.lang:
        records = [r for r in records if r["lang"] == args.lang]
    if not records:
        print("no valid records", file=sys.stderr)
        return 2

    preds = run(records)
    langs = [args.lang] if args.lang else [l for l in LANGS if any(r["lang"] == l for r in records)]
    metrics = {
        "meta": {"data": str(path), "n": len(records), "invalid_skipped": len(problems), "enter_threshold": ENTER,
                 "timestamp": datetime.now().isoformat(timespec="seconds"), "acie_fingerprint": acie_fingerprint(),
                 "notes": ["state metrics use HumanStateEstimator.estimate on a single turn from the neutral prior "
                           "(smoothed = 0.7*raw for rising dims), plus raw RuleScorer scores",
                           "intent = classifier output with bank_demo's slot-fill promotion (unknown + pending "
                           "amount slot + parsable number -> transfer); classifier-only accuracy reported separately",
                           "engagement positive class = gold label 0 (disengaged), score = 1 - engagement"]},
        "overall": compute(preds),
        "by_lang": {l: compute([p for p in preds if p["rec"]["lang"] == l]) for l in langs},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{args.lang}" if args.lang else "")
    jpath, epath = out / f"signals_{stamp}.json", out / f"errors_{stamp}.md"
    jpath.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_errors(preds, epath)

    print(f"data: {path}  ({len(records)} turns, {len(problems)} invalid skipped)")
    print_table(metrics, langs)
    print(f"\nmetrics -> {jpath.relative_to(ROOT) if jpath.is_relative_to(ROOT) else jpath}")
    print(f"errors  -> {epath.relative_to(ROOT) if epath.is_relative_to(ROOT) else epath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
