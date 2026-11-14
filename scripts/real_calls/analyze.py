"""Analyse the real calls (data/real/calls, git-ignored): failure instances, counts, root causes.

Writes data/real/REPORT.md (private, has quotes) and docs/REAL_CALLS_FINDINGS.md (sanitised, committable:
no numbers, no names, only generic short quotes). Run: python scripts/real_calls/analyze.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, analyse_call, count_repetitions, load_calls  # noqa: E402

KINDS = ["repeat_text", "repeat_intent", "reasked_question", "ignored_question", "pushy", "wrong_language", "misheard",
         "frustration_words"]


def summarise(results: list[dict], calls: list[dict]) -> dict:
    tot = collections.Counter()
    per_call_rep = {}
    for r, c in zip(results, calls):
        for i in r["instances"]:
            tot[i["kind"]] += 1
        per_call_rep[r["call"]] = count_repetitions([t["text"] for t in c["turns"] if t["role"] == "agent"])
    ended = collections.Counter(c.get("end_reason") for c in calls)
    return {"calls": len(calls), "with_dialog": sum(1 for r in results if r["n_customer"] > 0),
            "customer_turns": sum(r["n_customer"] for r in results), "agent_turns": sum(r["n_agent"] for r in results),
            "totals": dict(tot), "repeats_headline": sum(per_call_rep.values()),
            "calls_with_repeat": sum(1 for v in per_call_rep.values() if v), "end_reasons": dict(ended),
            "per_call_repeats": per_call_rep}


def main() -> None:
    calls = load_calls()
    results = [analyse_call(c) for c in calls]
    s = summarise(results, calls)
    intents = collections.Counter(i["intent"] for r in results for i in r["instances"] if i["kind"] == "repeat_text")
    private = ["# Real-call report (private, git-ignored)\n", f"{s['calls']} calls, {s['with_dialog']} with a conversation, "
               f"{s['customer_turns']} customer turns, {s['agent_turns']} agent turns.\n",
               f"Ended by: {s['end_reasons']}\n", "## Totals\n"]
    private += [f"- {k}: {s['totals'].get(k, 0)}" for k in KINDS]
    private += [f"\nHeadline repetition (agent reply, ratio>0.6 to an earlier agent reply): {s['repeats_headline']} replies "
                f"in {s['calls_with_repeat']} calls\n", f"Repeated replies by intent: {dict(intents)}\n", "## Instances\n"]
    for r in results:
        if not r["instances"]:
            continue
        private.append(f"### call {r['call']} ({r['campaign']}, ended {r['end']})")
        for i in r["instances"]:
            private.append("- " + json.dumps(i, ensure_ascii=False))
    (ROOT / "data/real/REPORT.md").write_text("\n".join(private) + "\n")
    (ROOT / "data/real/analysis.json").write_text(json.dumps({"summary": s, "results": results}, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "per_call_repeats"}, indent=1))
    print("repeated replies by intent:", dict(intents))


if __name__ == "__main__":
    main()
