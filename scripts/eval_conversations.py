"""End-to-end conversation eval: scripted multi-turn calls through the full ACIEPipeline.

Each conversation gets a fresh BankDemoDomain (MockBankAPI balances mutate on
transfer), a fresh MemoryProfileStore and the TemplateBackend (no GPU / LLM),
so results are deterministic. Each turn's `expect` keys are compared with:
  intent                     -> result.plan.intent
  language / complexity / verbosity / formality / warmth / proactivity /
  acknowledge_emotion        -> result.expression.<key>

  .venv/bin/python scripts/eval_conversations.py
  .venv/bin/python scripts/eval_conversations.py --data scripts/fixtures/smoke_conversations.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_common import (acie_fingerprint, LANGS, RESULTS_DIR, ROOT, load_jsonl, percentile, report_problems,  # noqa: E402
                         validate_conversation)

from acie.decision.domains.bank_demo import BankDemoDomain  # noqa: E402
from acie.decision.engine import DecisionEngine  # noqa: E402
from acie.expression import TemplateBackend  # noqa: E402
from acie.personalization.profile_store import MemoryProfileStore  # noqa: E402
from acie.pipeline import ACIEPipeline  # noqa: E402
from acie.schemas import TurnInput, TurnTiming  # noqa: E402
from acie.text import detect_language  # noqa: E402


def conv_lang(c: dict) -> str:
    """Language bucket of a conversation = the caller's OPENING language: an explicit `lang`
    field if present, else detect_language over the first two customer turns (one word like
    "hello" is too short to tell en from hinglish). Calls that switch language mid-way are
    bucketed by where they started."""
    if c.get("lang") in LANGS:
        return c["lang"]
    return detect_language(" ".join(t["text"] for t in c["turns"][:2]))


async def run_conv(c: dict) -> list[dict]:
    pipe = ACIEPipeline(DecisionEngine(BankDemoDomain()), backend=TemplateBackend(), store=MemoryProfileStore())
    rows = []
    for i, t in enumerate(c["turns"]):
        timing = {k: v for k, v in (t.get("timing") or {}).items() if k in TurnTiming.model_fields and v is not None}
        turn = TurnInput(session_id=c["id"], customer_id=c["customer_id"], text=t["text"], timing=TurnTiming(**timing))
        res = await pipe.run_turn(turn)
        got = {"intent": res.plan.intent, **res.expression.model_dump()}
        checks = {k: {"expected": v, "got": got.get(k), "ok": got.get(k) == v} for k, v in (t.get("expect") or {}).items()}
        rows.append({"turn": i, "text": t["text"], "checks": checks, "reply": res.text,
                     "state": {k: v for k, v in res.state.model_dump().items() if k != "signals"},
                     "signals": res.state.signals, "timings_ms": res.timings_ms,
                     "used_fallback": res.validation.used_fallback})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "eval" / "conversations.jsonl"))
    ap.add_argument("--lang", choices=LANGS)
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"no data at {path} (smoke test: --data scripts/fixtures/smoke_conversations.jsonl)", file=sys.stderr)
        return 2
    convs, problems = load_jsonl(path, validate_conversation)
    report_problems(problems, len(convs))
    for c in convs:
        c["_lang"] = conv_lang(c)
    if args.lang:
        convs = [c for c in convs if c["_lang"] == args.lang]
    if not convs:
        print("no valid conversations", file=sys.stderr)
        return 2

    results = {c["id"]: asyncio.run(run_conv(c)) for c in convs}

    by_key: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_lang_key: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    conv_pass = {}
    failures = []
    totals = []
    stages = defaultdict(list)
    for c in convs:
        ok_all = True
        for row in results[c["id"]]:
            totals.append(row["timings_ms"].get("total", 0.0))
            for k, v in row["timings_ms"].items():
                stages[k].append(v)
            for k, ch in row["checks"].items():
                for bucket in (by_key[k], by_lang_key[c["_lang"]][k], by_lang_key[c["_lang"]]["ALL"], by_key["ALL"]):
                    bucket[0] += ch["ok"]
                    bucket[1] += 1
                if not ch["ok"]:
                    failures.append({"conv": c["id"], "lang": c["_lang"], "turn": row["turn"], "text": row["text"],
                                     "key": k, "expected": ch["expected"], "got": ch["got"],
                                     # a check that fails after an earlier failed turn may be knock-on damage
                                     # (e.g. no pending transfer because the amount was never parsed)
                                     "after_earlier_failure": not ok_all,
                                     "state": row["state"], "signals": row["signals"]})
            if any(not ch["ok"] for ch in row["checks"].values()):
                ok_all = False
        conv_pass[c["id"]] = ok_all

    def rate(p):
        return {"pass": p[0], "total": p[1], "rate": round(p[0] / p[1], 3) if p[1] else None}

    metrics = {
        "meta": {"data": str(path), "n_conversations": len(convs), "n_turns": len(totals),
                 "invalid_skipped": len(problems), "backend": "TemplateBackend",
                 "timestamp": datetime.now().isoformat(timespec="seconds")},
        "conversation_pass_rate": round(sum(conv_pass.values()) / len(conv_pass), 3),
        "by_key": {k: rate(v) for k, v in sorted(by_key.items())},
        "by_lang": {l: {k: rate(v) for k, v in sorted(d.items())} for l, d in sorted(by_lang_key.items())},
        "latency_ms": {"total_p50": percentile(totals, 0.5), "total_p95": percentile(totals, 0.95),
                       "total_max": round(max(totals), 2),
                       "stages_p50": {k: percentile(v, 0.5) for k, v in stages.items()}},
        "root_failures": sum(not f["after_earlier_failure"] for f in failures),
        "failures": failures,
        "turns": results,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{args.lang}" if args.lang else "")
    jpath = out / f"conversations_{stamp}.json"
    jpath.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"data: {path}  ({len(convs)} conversations, {len(totals)} turns, {len(problems)} invalid skipped)")
    print(f"conversations fully passing: {sum(conv_pass.values())}/{len(conv_pass)}")
    langs = sorted(by_lang_key)
    keys = [k for k in sorted(by_key) if k != "ALL"] + ["ALL"]
    print(f"\n{'expect key':<22}" + "".join(f"{c:>14}" for c in ["all"] + langs))
    for k in keys:
        cells = [by_key[k]] + [by_lang_key[l].get(k, [0, 0]) for l in langs]
        print(f"{k:<22}" + "".join(f"{(f'{p[0]}/{p[1]} {p[0] / p[1]:.0%}' if p[1] else '-'):>14}" for p in cells))
    lat = metrics["latency_ms"]
    print(f"\npipeline latency (TemplateBackend, no LLM): p50 {lat['total_p50']} ms, p95 {lat['total_p95']} ms, "
          f"max {lat['total_max']} ms")
    if failures:
        print(f"\nfailures ({len(failures)}; {metrics['root_failures']} are the first failure in their call, "
              f"the rest (marked *) follow an earlier failure and may be knock-on):")
        for f in failures[:60]:
            print(f"  {'*' if f['after_earlier_failure'] else ' '}{f['conv']}#{f['turn']} [{f['lang']}] {f['key']}: expected {f['expected']!r} got {f['got']!r}"
                  f"  | {f['text'][:70]}")
        if len(failures) > 60:
            print(f"  ... {len(failures) - 60} more in {jpath.name}")
    print(f"\nmetrics -> {jpath.relative_to(ROOT) if jpath.is_relative_to(ROOT) else jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
