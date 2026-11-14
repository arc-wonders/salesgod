"""Replay every real call's customer turns through the pipeline (template backend, no network).

Runs each mode twice and asserts identical output (determinism), then reports repetition counts:
logged (what the live system said) vs current code (ACIE_PSYCHE off) vs psyche (ACIE_PSYCHE=1).
Run: python scripts/real_calls/replay.py [--json out.json]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import ROOT, count_repetitions, load_calls  # noqa: E402

from acie.decision.domains.product_sales import Product, ProductSalesDomain  # noqa: E402
from acie.decision.engine import DecisionEngine  # noqa: E402
from acie.pipeline import ACIEPipeline  # noqa: E402
from acie.schemas import TurnInput  # noqa: E402


def products() -> dict[str, Product]:
    raw = json.loads((ROOT / "data/voice/products.json").read_text())
    return {k: Product.model_validate(v) for k, v in raw.items()}


async def replay_call(call: dict, prods: dict[str, Product]) -> list[str]:
    prod = prods.get(call.get("campaign"), Product())
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(prod)))
    sid = call["_id6"]
    T = lambda t: TurnInput(session_id=sid, customer_id="+910000000000", text=t)
    sess = pipe.session(T(""))
    opening = await pipe.decision.decide("greet", T(""), sess)
    sess.context.record_turn(0, "", opening.intent, False)
    out = []
    for t in call["turns"]:
        if t["role"] == "customer":
            r = await pipe.run_turn(T(t["text"]))
            out.append(r.text)
            if r.plan.end_call:
                break
    return out


async def run_mode(calls, prods, psyche: bool, guard: bool = True) -> dict[str, list[str]]:
    os.environ["ACIE_PSYCHE"] = "1" if psyche else ""
    os.environ["ACIE_REPEAT_GUARD"] = "1" if guard else "0"
    return {c["_id6"]: await replay_call(c, prods) for c in calls}


def main() -> None:
    calls, prods = load_calls(), products()
    rows = {}
    for mode, flag, guard in (("base_code", False, False), ("current", False, True), ("psyche", True, True)):
        a = asyncio.run(run_mode(calls, prods, flag, guard))
        b = asyncio.run(run_mode(calls, prods, flag, guard))
        assert a == b, f"{mode}: replay is not deterministic"
        rows[mode] = a
    os.environ["ACIE_PSYCHE"] = ""
    res = {"logged": 0, "base_code": 0, "current": 0, "psyche": 0}
    per = []
    for c in calls:
        logged = [t["text"] for t in c["turns"] if t["role"] == "agent"]
        n = {"logged": count_repetitions(logged)}
        for m in ("base_code", "current", "psyche"):
            n[m] = count_repetitions(rows[m][c["_id6"]])
            res[m] += n[m]
        res["logged"] += n["logged"]
        per.append((c["_id6"], n))
    summary = {"calls": len(calls), "totals": res, "deterministic": True,
               "calls_with_repeat": {m: sum(1 for _, n in per if n[m]) for m in res}}
    print(json.dumps(summary, indent=1))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps({"summary": summary, "per_call": per,
                                                                           "replies": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
