#!/usr/bin/env python
"""Build SFT data for the Expression Engine (Layer 6).

Prompts are produced by the *runtime* acie.compiler.PromptCompiler from real
acie.schemas objects, so train and serve prompts are identical.

  --mode seed     compile the hand-written gold set (seed_data.py) -> data/sft/seed.jsonl
                  and run every target through check_sft's checks.
  --mode teacher  sample TurnPlans (BankDemoDomain branches on fresh instances +
                  synthetic retail-banking plans) x ExpressionParams x short histories
                  -> data/sft/prompts.jsonl (+ held-out data/sft/eval_prompts.jsonl)
                  for teacher_generate.py to complete.

Needs only acie + stdlib:  .venv/bin/python training/expression/build_sft.py --mode seed
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checks import check_target  # noqa: E402
from seed_data import SEEDS  # noqa: E402
from sft_common import (BANK_RECIPES, CLARIFIABLE, CUSTOMER, SYNTH, Scenario, compile_record,  # noqa: E402
                        amount_for, customer_text, format_amount, greet_history, parse_expr,
                        random_account, record_plan_expr, run_bank_recipe, sample_expr, synth_scenario,
                        write_jsonl)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLARIFY_PREFIX = "The customer did not understand. Re-explain more simply: "


# =========================================================================== seed mode

def build_seed_scenario(i: int, name: str, lang: str, cust: str) -> Scenario:
    rng = random.Random(1000 + i)
    if name.startswith("synth:"):
        hist = greet_history(rng, lang, rng.choice(["Rahul", "Priya", "Amit", "Neha"])) if rng.random() < 0.5 else []
        return synth_scenario(name.split(":", 1)[1], lang, rng, history=hist)
    clarify = name.startswith("clarify:")
    base = name.split(":", 1)[1] if clarify else name
    pre = None
    if not clarify and not BANK_RECIPES[base][1] and base not in ("greet",) and rng.random() < 0.4:
        cid = BANK_RECIPES[base][0]
        pre = greet_history(rng, lang, {"c1": "Rahul", "c2": "Priya"}[cid])
    sc = run_bank_recipe(name, lang, cust, rng, pre_history=pre)
    if sc.plan.intent != base:
        raise SystemExit(f"seed #{i} {name}: domain produced intent {sc.plan.intent!r}, expected {base!r} "
                         f"(check the amount in the customer text {cust!r})")
    if clarify and not sc.plan.objective.startswith(CLARIFY_PREFIX):
        raise SystemExit(f"seed #{i} {name}: not a re-explanation plan")
    return sc


def mode_seed(out: str, check: bool) -> int:
    rows, fails = [], 0
    for i, (name, spec, cust, target) in enumerate(SEEDS):
        expr = parse_expr(spec)
        sc = build_seed_scenario(i, name, expr.language, cust)
        rec = compile_record(sc, expr, cust, "seed", target=target, rid=f"seed-{i:03d}-{expr.language}-{name}")
        if check:
            plan, e = record_plan_expr(rec)
            issues = check_target(target, plan, e)
            if issues:
                fails += 1
                print(f"[FAIL] {rec['id']}: {target}\n       " + "\n       ".join(issues))
        rows.append(rec)
    write_jsonl(out, rows)
    langs = collections.Counter(r["meta"]["lang"] for r in rows)
    n = len(rows)
    print(f"wrote {n} seed examples -> {out}")
    print("  by language: " + ", ".join(f"{k}={v} ({v / n:.0%})" for k, v in sorted(langs.items())))
    print(f"  distinct scenarios: {len({r['meta']['tag'] for r in rows})}, "
          f"ack={sum(r['meta']['expr']['acknowledge_emotion'] for r in rows)}, "
          f"clarify={sum(r['meta']['tag'].startswith('clarify:') for r in rows)}")
    if check:
        print(f"  checks: {n - fails}/{n} pass")
    return 1 if fails else 0


# =========================================================================== teacher mode

BANK_WEIGHTS = {k: 1.0 for k in BANK_RECIPES}
BANK_WEIGHTS.update({"transfer_over_limit": 2.0, "transfer_confirm": 1.5, "card_offer": 1.5, "transfer_limit": 1.5,
                     "check_balance": 1.5, "unknown": 0.6, "goodbye": 0.6})


def sample_bank(rng: random.Random, lang: str, flags: dict, clarify: bool) -> tuple[Scenario, str]:
    name = rng.choice(CLARIFIABLE) if clarify else rng.choices(list(BANK_WEIGHTS), list(BANK_WEIGHTS.values()))[0]
    accounts, amount = {}, None
    for _ in range(50):
        accounts = {"c1": random_account(rng, eligible=True), "c2": random_account(rng, eligible=False)}
        cid = BANK_RECIPES[name][0]
        if "transfer" in name or name in ("cancelled",):
            amount = amount_for(rng, name, accounts[cid])
            if amount is None:
                continue
        break
    _, _, _, key = BANK_RECIPES[name]
    amt_s = format_amount(rng, amount, lang) if amount else None
    if clarify:
        final = customer_text(rng, "clarify", lang)
        full = f"clarify:{name}"
    else:
        final = customer_text(rng, key, lang, frustrated=flags["frustrated"], amt=amt_s)
        full = name
    pre = None
    if not BANK_RECIPES[name][1] and name != "greet" and rng.random() < 0.35:
        pre = greet_history(rng, lang, accounts[BANK_RECIPES[name][0]]["name"])
    sc = run_bank_recipe(full, lang, final, rng, accounts=accounts, amount=amount, pre_history=pre)
    expected = name
    if sc.plan.intent != expected:  # e.g. STT-style amount not parsed - skip
        raise ValueError(f"branch mismatch {sc.plan.intent} != {expected}")
    return sc, final


def sample_synth(rng: random.Random, lang: str, flags: dict) -> tuple[Scenario, str]:
    name = rng.choice(list(SYNTH))
    s = SYNTH[name]
    hist = greet_history(rng, lang, rng.choice(["Rahul", "Priya", "Amit", "Neha", "Sunita", "Imran"])) \
        if rng.random() < 0.4 else []
    sc = synth_scenario(name, lang, rng, history=hist)
    final = rng.choice(s.customer[lang])
    if flags["frustrated"]:
        from sft_common import FRUSTRATED_PREFIX
        p = rng.choice(FRUSTRATED_PREFIX[lang])
        final = p + (final if lang == "hi" else final[0].lower() + final[1:])
    if flags["confused"] and rng.random() < 0.5:  # synthetic re-explanation: previous agent turn + "didn't get it"
        from sft_common import agent_history_text
        from acie.schemas import Utterance
        prev_final = final
        sc.history = sc.history + [Utterance(role="customer", text=prev_final),
                                   Utterance(role="agent", text=agent_history_text(sc.plan, lang))]
        sc.plan.objective = CLARIFY_PREFIX + sc.plan.objective
        sc.tag = f"clarify:synth:{name}"
        final = customer_text(rng, "clarify", lang)
    return sc, final


def _prompt_hash(rec: dict) -> str:
    return hashlib.sha1(json.dumps(rec["messages"], ensure_ascii=False).encode()).hexdigest()


def generate(n: int, seed: int, bank_frac: float, exclude: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    rows, seen = [], set(exclude or ())
    tries = 0
    while len(rows) < n and tries < n * 5:
        tries += 1
        r = random.Random(rng.getrandbits(64))
        use_bank = r.random() < bank_frac
        clarify = r.random() < 0.12
        expr, flags = sample_expr(r, clarify=clarify)
        try:
            sc, said = sample_bank(r, expr.language, flags, clarify) if use_bank else sample_synth(r, expr.language, flags)
        except ValueError:
            continue
        rec = compile_record(sc, expr, said, "teacher")
        h = _prompt_hash(rec)
        if h in seen:
            continue
        seen.add(h)
        rows.append(rec)
    return rows


def mode_teacher(out: str, n: int, seed: int, eval_out: str | None, eval_n: int, bank_frac: float) -> int:
    rows = generate(n, seed, bank_frac)
    write_jsonl(out, rows)
    _summary(out, rows)
    if eval_out and eval_n:
        ev = generate(eval_n, seed + 7919, bank_frac, exclude={_prompt_hash(r) for r in rows})
        for r in ev:
            r["source"] = "eval"
            r["id"] = r["id"].replace("teacher-", "eval-")
        write_jsonl(eval_out, ev)
        _summary(eval_out, ev)
    return 0


def _summary(path: str, rows: list[dict]) -> None:
    n = len(rows) or 1
    langs = collections.Counter(r["meta"]["lang"] for r in rows)
    tags = collections.Counter(r["meta"]["tag"].split(":")[0] if r["meta"]["tag"].startswith("clarify")
                               else ("synth" if r["meta"]["tag"].startswith("synth") else "bank") for r in rows)
    print(f"wrote {len(rows)} prompts -> {path}")
    print("  language: " + ", ".join(f"{k}={v} ({v / n:.0%})" for k, v in sorted(langs.items())))
    print("  source:   " + ", ".join(f"{k}={v}" for k, v in sorted(tags.items())))
    for key in ("verbosity", "formality", "complexity", "warmth", "proactivity"):
        c = collections.Counter(r["meta"]["expr"][key] for r in rows)
        print(f"  {key:<11} " + " ".join(f"{k}={c[k] / n:.0%}" for k in ("low", "med", "high")))
    print(f"  ack={sum(r['meta']['expr']['acknowledge_emotion'] for r in rows) / n:.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["seed", "teacher"], required=True)
    ap.add_argument("--out", default=None, help="default: data/sft/seed.jsonl or data/sft/prompts.jsonl")
    ap.add_argument("--n", type=int, default=8000, help="teacher mode: number of prompts")
    ap.add_argument("--eval-n", type=int, default=400, help="teacher mode: held-out eval prompts")
    ap.add_argument("--eval-out", default=os.path.join(ROOT, "data/sft/eval_prompts.jsonl"))
    ap.add_argument("--bank-frac", type=float, default=0.55, help="share of BankDemo plans vs synthetic plans")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    if a.mode == "seed":
        return mode_seed(a.out or os.path.join(ROOT, "data/sft/seed.jsonl"), check=not a.no_check)
    return mode_teacher(a.out or os.path.join(ROOT, "data/sft/prompts.jsonl"), a.n, a.seed, a.eval_out, a.eval_n,
                        a.bank_frac)


if __name__ == "__main__":
    sys.exit(main())
