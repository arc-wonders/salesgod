#!/usr/bin/env python
"""Evaluate Expression-Engine models on held-out prompts via an OpenAI-compatible endpoint.

  # base vs fine-tuned, both served by one vLLM (base + LoRA, see deploy/serve_vllm.sh MODE=lora)
  .venv/bin/python training/expression/eval_expression.py \
      --base-url http://voice:8000/v1 --models base,acie-expression \
      --prompts data/sft/eval_prompts.jsonl --out runs/eval_report.json

Streams each completion exactly like acie.expression.OpenAICompatibleBackend
(same temperature / max_tokens defaults) and feeds deltas into the runtime
StreamingValidator to measure latency to the first *validated* sentence.

Metrics per model (overall and per language):
  validator_pass   runtime validator accepted every sentence, no fallback needed
  fact_coverage    share of must_mention facts the model actually wrote as {slot}
  lang_match       language/script check (checks.language_issue)
  length_ok        length vs verbosity (checks.length_issue)
  ask_ok           ends with a question when the plan has `ask`
  all_checks       passes every check in checks.check_target (the SFT bar)
  ttft / first_sentence p50/p95 (ms), total p50/p95 (ms)

Needs only acie (+httpx).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import statistics
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acie.validator import StreamingValidator  # noqa: E402
from checks import check_target, language_issue, length_issue  # noqa: E402
from sft_common import prompt_messages, read_jsonl, record_plan_expr  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SLOT = re.compile(r"\{(\w+)\}")


async def stream_one(client: httpx.AsyncClient, a, model: str, rec: dict) -> dict:
    plan, expr = record_plan_expr(rec)
    body = {"model": model, "messages": prompt_messages(rec), "temperature": a.temperature,
            "max_tokens": a.max_tokens, "stream": True}
    v = StreamingValidator(plan, expr)
    text, ttft, first_sent, err = "", None, None, None
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", f"{a.base_url.rstrip('/')}/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {a.api_key}"}) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if not delta:
                    continue
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                text += delta
                if v.feed(delta) and first_sent is None:
                    first_sent = (time.perf_counter() - t0) * 1000
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    final = v.finish()
    if first_sent is None and final and not v.used_fallback:
        first_sent = (time.perf_counter() - t0) * 1000
    total = (time.perf_counter() - t0) * 1000

    text = text.strip()
    must = [k for k, f in plan.facts.items() if f.must_mention]
    written = set(_SLOT.findall(text))
    issues = check_target(text, plan, expr) if text else ["empty"]
    return {
        "id": rec["id"], "lang": expr.language, "tag": rec["meta"]["tag"], "expr_spec": rec["meta"]["expr_spec"],
        "text": text, "error": err, "issues": issues,
        "validator_pass": bool(text) and not v.issues and not v.used_fallback,
        "fact_coverage": (len([k for k in must if k in written]) / len(must)) if must else 1.0,
        "lang_match": bool(text) and language_issue(text, expr.language) is None,
        "length_ok": bool(text) and length_issue(text, plan, expr) is None,
        "ask_ok": (text.rstrip().endswith("?") if plan.ask else True),
        "all_checks": not issues,
        "ttft_ms": ttft, "first_sentence_ms": first_sent, "total_ms": total,
    }


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return round(xs[0], 1)
    return round(statistics.quantiles(xs, n=100, method="inclusive")[int(q) - 1], 1)


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    out = {"n": len(rows)}
    for k in ("validator_pass", "lang_match", "length_ok", "ask_ok", "all_checks"):
        out[k] = round(sum(bool(r[k]) for r in rows) / n, 4)
    out["fact_coverage"] = round(sum(r["fact_coverage"] for r in rows) / n, 4)
    for k in ("ttft_ms", "first_sentence_ms", "total_ms"):
        xs = [r[k] for r in rows]
        out[f"{k}_p50"], out[f"{k}_p95"] = pct(xs, 50), pct(xs, 95)
    out["errors"] = sum(1 for r in rows if r["error"])
    kinds = collections.Counter(i.split(":")[0] for r in rows for i in r["issues"])
    out["issue_kinds"] = dict(kinds.most_common())
    return out


async def eval_model(a, model: str, prompts: list[dict]) -> list[dict]:
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient(timeout=a.timeout) as client:
        if a.warmup:
            await asyncio.gather(*(stream_one(client, a, model, p) for p in prompts[:a.warmup]))

        async def one(p):
            async with sem:
                return await stream_one(client, a, model, p)
        return await asyncio.gather(*(one(p) for p in prompts))


def print_table(report: dict) -> None:
    cols = ["n", "validator_pass", "fact_coverage", "lang_match", "length_ok", "ask_ok", "all_checks",
            "ttft_ms_p50", "first_sentence_ms_p50", "first_sentence_ms_p95", "total_ms_p95"]
    print("\n" + " " * 26 + "".join(f"{c[:14]:>15}" for c in cols))
    for model, rep in report.items():
        for scope, s in [("all", rep["overall"]), *sorted(rep["by_lang"].items())]:
            print(f"{(model + ' / ' + scope)[:26]:<26}" + "".join(
                f"{(s.get(c) if s.get(c) is not None else '-'):>15}" for c in cols))


async def main_async(a) -> None:
    prompts = read_jsonl(a.prompts)
    if a.limit:
        prompts = prompts[:a.limit]
    report, all_rows = {}, {}
    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        print(f"evaluating {model} on {len(prompts)} prompts (concurrency={a.concurrency}) ...")
        rows = await eval_model(a, model, prompts)
        by_lang = collections.defaultdict(list)
        for r in rows:
            by_lang[r["lang"]].append(r)
        report[model] = {"overall": summarize(rows), "by_lang": {k: summarize(v) for k, v in by_lang.items()}}
        all_rows[model] = rows
    print_table(report)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(a), "report": report}, f, ensure_ascii=False, indent=1)
        with open(os.path.splitext(a.out)[0] + "_rows.jsonl", "w", encoding="utf-8") as f:
            for model, rows in all_rows.items():
                for r in rows:
                    f.write(json.dumps({"model": model, **r}, ensure_ascii=False) + "\n")
        print(f"\nreport -> {a.out} (per-example rows next to it)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.getenv("ACIE_LLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.getenv("ACIE_LLM_API_KEY", "EMPTY"))
    ap.add_argument("--models", default=os.getenv("ACIE_LLM_MODEL", "acie-expression"),
                    help="comma-separated served model names, e.g. base,acie-expression")
    ap.add_argument("--prompts", default=os.path.join(ROOT, "data/sft/eval_prompts.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/eval_report.json"))
    ap.add_argument("--temperature", type=float, default=0.4, help="same as acie OpenAICompatibleBackend")
    ap.add_argument("--max-tokens", type=int, default=160, help="same as acie OpenAICompatibleBackend")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="1 = single-call latency; raise to measure latency under load")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
