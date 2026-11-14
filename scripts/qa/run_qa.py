"""Offline conversation regression harness for SalesGod (ACIE).

Replays every scripted conversation in tests/qa_fixtures/conversations/*.json through ACIEPipeline with the
deterministic TemplateBackend, for two configs ("defaults" and "psyche" = ACIE_PSYCHE=1), twice each to assert
determinism, and checks every agent reply against machine-checkable expectations (scripts/qa/checks.py).

    python scripts/qa/run_qa.py [--report out.json] [--config defaults|psyche|both] [--class NAME] [-v]
                                [--backend template|openai]

`--backend openai` uses OpenAICompatibleBackend (ACIE_EXPR_* / ACIE_LLM_* env vars) instead of the templates; the
determinism assertion is then skipped (a real LLM is not bit-exact). Exit code 1 if any expectation fails, 2 if a
run crashes or the template runs are not deterministic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as C  # noqa: E402

FIXTURES = ROOT / "tests" / "qa_fixtures" / "conversations"
CRITICAL_CLASSES = {"crisis_missed"}
CONFIGS = {"defaults": {"ACIE_PSYCHE": ""}, "psyche": {"ACIE_PSYCHE": "1"}}


def load_fixtures(directory: Path = FIXTURES) -> list[dict]:
    out = []
    for f in sorted(directory.glob("*.json")):
        d = json.loads(f.read_text())
        d["_file"] = f.name
        out.append(d)
    return out


def load_products() -> dict:
    """products.json plus harness overrides for campaign records whose pitch lines name another product."""
    raw = json.loads((ROOT / "data/voice/products.json").read_text())
    from overrides import PITCH_OVERRIDES, has_wrong_pitch
    for k, ov in PITCH_OVERRIDES.items():
        if k in raw and has_wrong_pitch(raw[k]):
            raw[k].update(ov)  # repo products.json bug: see overrides.py
    return raw


def _set_env(cfg: str) -> dict:
    old = {k: os.environ.get(k) for k in ("ACIE_PSYCHE", "ACIE_REPEAT_GUARD")}
    os.environ["ACIE_PSYCHE"] = CONFIGS[cfg]["ACIE_PSYCHE"]
    os.environ.pop("ACIE_REPEAT_GUARD", None)  # measure the code as shipped (guard on by default)
    return old


def _restore_env(old: dict) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


async def play(fx: dict, prods_raw: dict, backend_name: str = "template") -> list[dict]:
    """Run one conversation. Returns one record per caller turn: the reply and its plan summary."""
    from acie.decision.domains.product_sales import Product, ProductSalesDomain
    from acie.decision.engine import DecisionEngine
    from acie.validator import StreamingValidator
    from acie.pipeline import ACIEPipeline
    from acie.schemas import ExpressionParams, TurnInput, Utterance

    camp = fx["campaign"]
    prod = Product.model_validate(prods_raw[camp])
    backend = None
    if backend_name == "openai":
        from acie.expression import OpenAICompatibleBackend
        backend = OpenAICompatibleBackend()
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(prod)), backend=backend)
    sid = "qa-" + fx["id"]
    T = lambda t: TurnInput(session_id=sid, customer_id="+910000000000", text=t)
    sess = pipe.session(T(""))
    sess.expression = ExpressionParams(language=fx.get("opening_language", "hinglish"))
    plan = await pipe.decision.decide("greet", T(""), sess)
    opening = " ".join(StreamingValidator(plan, sess.expression).finish())
    sess.history.append(Utterance(role="agent", text=opening))
    sess.context.record_turn(0, "", plan.intent, False)
    recs = [{"role": "opening", "reply": opening}]
    for turn in fx["turns"]:
        r = await pipe.run_turn(T(turn["say"]))
        recs.append({"say": turn["say"], "reply": r.text, "intent": r.plan.intent, "end_call": bool(r.plan.end_call),
                     "ask": r.plan.ask, "lang": r.expression.language})
        if r.plan.end_call:
            break
    return recs


def evaluate(fx: dict, recs: list[dict], prods_raw: dict) -> list[dict]:
    """Apply default checks + the fixture's `must` to every agent reply. Returns failures."""
    camp = fx["campaign"]
    prod = prods_raw[camp]
    allowed = C.numbers_in_product(prod)
    kbf = ROOT / "data" / "voice" / "knowledge" / "default" / f"{camp}.json"
    if os.environ.get("ACIE_KB") == "1" and kbf.exists():  # facts the knowledge base carries are product facts too
        allowed |= set(re.findall(r"\d+", C.flat(kbf.read_text())))
    others = [v["product_name"] for k, v in prods_raw.items() if k != camp and v["product_name"] != prod["product_name"]]
    fails: list[dict] = []
    prior_replies = [recs[0]["reply"]]
    prior_tags: list[set] = []
    turns = fx["turns"]
    said_numbers: set[str] = set()  # a price the caller named earlier (and we agreed to) is legitimate later in the call
    for i, rec in enumerate(recs[1:]):
        turn = turns[i]
        said_numbers |= set(re.findall(r"\d+", C.flat(turn["say"])))
        tags = set(turn.get("tags", []))
        ctx = C.Ctx(reply=rec["reply"], intent=rec["intent"], end_call=rec["end_call"], ask=rec["ask"], say=rec["say"],
                    tags=tags, lang=turn.get("lang", fx.get("lang", "hinglish")), prior_replies=list(prior_replies),
                    prior_tags=list(prior_tags), prior_intents=[r_["intent"] for r_ in recs[1:i + 1]], allowed_numbers=allowed | said_numbers, product_name=prod["product_name"],
                    other_products=others, expect_lang=turn.get("expect_lang"))
        skip = set(turn.get("skip", []))
        # an explicit end-of-call expectation implies the agent may end here
        mu = turn.get("must", {})
        while isinstance(mu.get("must"), dict):
            mu = {**{k: v for k, v in mu.items() if k != "must"}, **mu["must"]}
        if mu.get("end_call") is True:
            ctx.tags.add("end_ok")
        for name, fn in C.DEFAULT_CHECKS.items():
            if name in skip:
                continue
            d = fn(ctx)
            if d:
                fails.append({"turn": i + 1, "check": name, "detail": d, "reply": rec["reply"], "say": rec["say"]})
        for name, d in C.check_must(ctx, turn.get("must", {})):
            fails.append({"turn": i + 1, "check": name, "detail": d, "reply": rec["reply"], "say": rec["say"]})
        prior_replies.append(rec["reply"])
        prior_tags.append(tags)
    # the conversation must reach the end when the fixture says the agent has to hang up at the last turn
    if fx.get("must_end") and not recs[-1].get("end_call"):
        fails.append({"turn": len(recs) - 1, "check": "must_end_call", "detail": "call never ended", "reply": "", "say": ""})
    return fails


async def run_all(fixtures, cfg: str, backend: str, prods_raw: dict) -> dict:
    old = _set_env(cfg)
    try:
        out = {}
        for fx in fixtures:
            try:
                out[fx["id"]] = {"recs": await play(fx, prods_raw, backend)}
            except Exception as e:  # a crash is a harness failure, reported separately
                out[fx["id"]] = {"error": f"{type(e).__name__}: {e}"}
        return out
    finally:
        _restore_env(old)


def run_harness(fixtures=None, configs=("defaults", "psyche"), backend="template") -> dict:
    fixtures = fixtures if fixtures is not None else load_fixtures()
    prods_raw = load_products()
    from overrides import repo_data_issues
    raw = json.loads((ROOT / "data/voice/products.json").read_text())
    report: dict = {"backend": backend, "configs": {}, "nondeterministic": [], "crashes": [],
                    "warnings": repo_data_issues(raw)}
    for cfg in configs:
        a = asyncio.run(run_all(fixtures, cfg, backend, prods_raw))
        if backend == "template":
            b = asyncio.run(run_all(fixtures, cfg, backend, prods_raw))
            for fid in a:
                if a[fid] != b[fid]:
                    report["nondeterministic"].append({"config": cfg, "id": fid})
        res = {}
        for fx in fixtures:
            r = a[fx["id"]]
            if "error" in r:
                report["crashes"].append({"config": cfg, "id": fx["id"], "error": r["error"]})
                res[fx["id"]] = {"class": fx["class"], "error": r["error"], "fails": [], "recs": []}
                continue
            res[fx["id"]] = {"class": fx["class"], "fails": evaluate(fx, r["recs"], prods_raw), "recs": r["recs"]}
        report["configs"][cfg] = res
    return report


def summarise(report: dict) -> dict:
    """{config: {class: {total, passed, by_check: {name: n}}}} plus overall."""
    out: dict = {}
    for cfg, res in report["configs"].items():
        classes: dict = defaultdict(lambda: {"total": 0, "passed": 0, "by_check": defaultdict(int)})
        for fid, r in res.items():
            c = classes[r["class"]]
            c["total"] += 1
            if not r["fails"] and "error" not in r:
                c["passed"] += 1
            for f in r["fails"]:
                c["by_check"][f["check"]] += 1
        out[cfg] = {k: {**v, "by_check": dict(v["by_check"])} for k, v in sorted(classes.items())}
    return out


def print_table(report: dict) -> None:
    summ = summarise(report)
    cfgs = list(summ)
    classes = sorted({c for s in summ.values() for c in s})
    w = max(len(c) for c in classes) + 2
    print(f"\n{'class':<{w}}" + "".join(f"{c:>16}" for c in cfgs))
    print("-" * (w + 16 * len(cfgs)))
    for k in classes:
        row = f"{k:<{w}}"
        for cfg in cfgs:
            s = summ[cfg].get(k)
            row += f"{(str(s['passed']) + '/' + str(s['total'])) if s else '-':>16}"
        print(row)
    print("-" * (w + 16 * len(cfgs)))
    row = f"{'ALL conversations':<{w}}"
    for cfg in cfgs:
        t = sum(s["total"] for s in summ[cfg].values())
        p = sum(s["passed"] for s in summ[cfg].values())
        row += f"{f'{p}/{t} ({100 * p / max(t, 1):.0f}%)':>16}"
    print(row)
    print("\nfailing checks (count of failed replies):")
    for cfg in cfgs:
        agg: dict = defaultdict(int)
        for s in summ[cfg].values():
            for n, v in s["by_check"].items():
                agg[n] += v
        print(f"  {cfg}: " + ", ".join(f"{n}={v}" for n, v in sorted(agg.items(), key=lambda x: -x[1])))
    crit = {fid for fid, r in next(iter(report["configs"].values())).items() if r["class"] in CRITICAL_CLASSES}
    if crit:
        print("\nCRITICAL (separate, hard failure once its lane lands):")
        for cfg, res in report["configs"].items():
            ok = sum(1 for f in crit if not res[f]["fails"])
            print(f"  {cfg}: {ok}/{len(crit)} pass ({', '.join(sorted(CRITICAL_CLASSES))})")
    for w in report.get("warnings", []):
        print("\nWARNING:", w)
    if report["nondeterministic"]:
        print("\nNON-DETERMINISTIC:", report["nondeterministic"])
    if report["crashes"]:
        print("\nCRASHES:", report["crashes"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="write the full JSON report here")
    ap.add_argument("--config", default="both", choices=["defaults", "psyche", "both"])
    ap.add_argument("--class", dest="cls", help="only this class")
    ap.add_argument("--backend", default="template", choices=["template", "openai"])
    ap.add_argument("-v", "--verbose", action="store_true", help="list every failure")
    a = ap.parse_args(argv)
    fx = load_fixtures()
    if a.cls:
        fx = [f for f in fx if f["class"] == a.cls]
    cfgs = ("defaults", "psyche") if a.config == "both" else (a.config,)
    rep = run_harness(fx, cfgs, a.backend)
    print_table(rep)
    if a.verbose:
        for cfg, res in rep["configs"].items():
            for fid, r in res.items():
                for f in r["fails"]:
                    print(f"[{cfg}] {fid} t{f['turn']} {f['check']}: {f['detail']}")
    if a.report:
        Path(a.report).write_text(json.dumps({"summary": summarise(rep), **rep}, ensure_ascii=False, indent=1))
    if rep["crashes"] or rep["nondeterministic"]:
        return 2
    bad = any(r["fails"] for res in rep["configs"].values() for r in res.values())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
