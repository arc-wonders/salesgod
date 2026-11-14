#!/usr/bin/env python
"""Check SFT targets: runtime StreamingValidator + extra checks (see checks.py).

  .venv/bin/python training/expression/check_sft.py                      # data/sft/seed.jsonl
  .venv/bin/python training/expression/check_sft.py data/sft/train.jsonl --quiet
  .venv/bin/python training/expression/check_sft.py data/sft/seed.jsonl --show 6

Also verifies that every record's prompt is exactly what the runtime
PromptCompiler produces for its (plan, expression, history, customer text).
Exit code 1 if any record fails. Needs only acie + stdlib.
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acie.schemas import SessionState, Utterance  # noqa: E402
from checks import check_target  # noqa: E402
from sft_common import COMPILER, read_jsonl, record_plan_expr  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def recompile_matches(rec: dict) -> bool:
    plan, expr = record_plan_expr(rec)
    msgs = rec["messages"][:-1] if rec["messages"][-1]["role"] == "assistant" else rec["messages"]
    hist = [Utterance(role="customer" if m["role"] == "user" else "agent", text=m["content"]) for m in msgs[1:-1]]
    cp = COMPILER.compile(plan, expr, SessionState(session_id="x", customer_id="x", history=hist),
                          rec["meta"]["customer_text"])
    return [{"role": "system", "content": cp.system}, *cp.messages] == msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", default=[os.path.join(ROOT, "data/sft/seed.jsonl")])
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    ap.add_argument("--show", type=int, default=0, help="print N random passing examples for a read-through")
    a = ap.parse_args()

    total = bad = 0
    by_lang: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    kinds: collections.Counter = collections.Counter()
    passing = []
    for path in a.files:
        for rec in read_jsonl(path):
            if rec["messages"][-1]["role"] != "assistant":
                continue
            total += 1
            plan, expr = record_plan_expr(rec)
            target = rec["messages"][-1]["content"]
            issues = check_target(target, plan, expr)
            if not recompile_matches(rec):
                issues.append("prompt: differs from runtime PromptCompiler output")
            lang = expr.language
            by_lang[lang][0] += 1
            if issues:
                bad += 1
                by_lang[lang][1] += 1
                kinds.update(i.split(":")[0] for i in issues)
                if not a.quiet:
                    print(f"[FAIL] {rec['id']} ({rec['meta']['expr_spec']})\n  target: {target}")
                    for i in issues:
                        print(f"  - {i}")
            else:
                passing.append(rec)

    print(f"\nchecked {total} targets in {', '.join(a.files)}")
    for lang, (n, f) in sorted(by_lang.items()):
        print(f"  {lang:<9} {n - f}/{n} pass")
    if kinds:
        print("  failure kinds: " + ", ".join(f"{k}={v}" for k, v in kinds.most_common()))
    print(f"  TOTAL: {total - bad}/{total} pass")

    if a.show and passing:
        print("\n--- sample ---")
        for rec in random.Random(0).sample(passing, min(a.show, len(passing))):
            print(f"[{rec['meta']['expr_spec']}] {rec['meta']['tag']}\n  customer: {rec['meta']['customer_text']}\n"
                  f"  agent:    {rec['messages'][-1]['content']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
