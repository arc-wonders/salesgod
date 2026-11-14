#!/usr/bin/env python
"""Complete SFT prompts with a large teacher LLM and keep only targets that pass the checks.

Talks to any OpenAI-compatible /v1/chat/completions endpoint - e.g. vLLM serving
Qwen2.5-72B-Instruct / Llama-3.3-70B-Instruct on the 4xA100 box (deploy/serve_teacher.sh).

  .venv/bin/python training/expression/teacher_generate.py \
      --base-url http://voice:8001/v1 --model teacher \
      --in data/sft/prompts.jsonl --out data/sft/train.jsonl

For every prompt it samples --n candidates, runs each through the runtime
StreamingValidator + extra checks (language/script, length vs verbosity, ask at
the end, must_not_say, number words, ack, tone; see checks.py) and keeps the
best passing one(s). Everything (incl. rejects and reasons) goes to --raw-out so
you can inspect failure modes. Resumable: prompts already in --raw-out are skipped.

The teacher sees the *student's* exact system prompt plus extra quality
guidelines and a few gold examples from data/sft/seed.jsonl; the student is
trained on the plain runtime prompt only.

Needs only acie (+httpx, already an acie dependency).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checks import check_target, words  # noqa: E402
from sft_common import prompt_messages, read_jsonl, record_plan_expr  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TEACHER_GUIDELINES = """
=== EXTRA GUIDELINES FOR THIS JOB (you are writing gold training data) ===
You are the best phone agent at an Indian retail bank. Write exactly what the agent says next, following the HARD RULES
and STYLE above to the letter. Quality bar:

- Sound like a real, calm, competent human agent on a phone call: short, natural, warm, direct. Not an IVR, not a chatbot.
- Facts appear ONLY as the given {placeholders}, copied exactly (same spelling, with braces). Never write any digit,
  amount, rupee sign, percentage, date, month, or number word (thousand, lakh, hazaar, हज़ार, ...). The placeholder will be
  replaced by the real value, so write the sentence so that it reads naturally once the value is filled in.
- Mention every placeholder in "facts". Never invent a placeholder.
- If "ask" is given, the reply MUST end with that question, phrased naturally, ending in "?".
- Respect "decision" and "must_not_say" strictly. No new offers, promises, timelines, reasons or advice.
- Length rule is a hard upper bound. Verbosity low = one short sentence (+ the question if any).
- If asked to acknowledge emotion: ONE short, specific, sincere acknowledgement as the first sentence
  (e.g. "Sorry about the wait." / "आपकी परेशानी समझ सकता हूँ।"), then straight to the point. Apologise at most once.
- If the objective says the customer did not understand: re-explain in simpler, shorter words; do not just repeat.
- Hindi (hi): everyday spoken Hindi in Devanagari, the way a Delhi call-centre agent talks. Use the English banking words
  people actually say, written in Devanagari: बैलेंस, ट्रांसफर, लिमिट, अकाउंट, कार्ड, अप्लाई, रेफ़रेंस नंबर, ड्यू डेट, चार्ज, लिंक;
  acronyms stay Latin: SMS, EMI, KYC, UPI, FD, OTP. NEVER textbook/shuddh Hindi (no शेष राशि, स्थानांतरण, सीमा, धनराशि,
  आवेदन, ऋण, संदर्भ संख्या, असुविधा के लिए खेद). Masculine first person (सकता हूँ, कर देता हूँ).
- Hinglish: Roman script only, natural Delhi/Mumbai phone Hinglish ("Aapka balance {balance} hai.").
- English: simple Indian English; no Hindi words.
- Never use: "I apologize for any inconvenience", "rest assured", "feel free to", "valued customer", "kindly note",
  "I'd be happy to assist", "Great question", "As an AI". No exclamation spam.
- Output ONLY the spoken words: no quotes, no labels, no notes, no markdown, no line breaks.
"""


def few_shot_block(seeds: list[dict], lang: str, intent: str, rng: random.Random, k: int) -> str:
    pool = [s for s in seeds if s["meta"]["lang"] == lang and s["meta"]["intent"] != intent] or seeds
    if not pool or k <= 0:
        return ""
    lines = ["\n=== GOLD EXAMPLES (style reference only; different situations) ==="]
    for s in rng.sample(pool, min(k, len(pool))):
        plan = s["meta"]["plan"]
        facts = ", ".join("{" + f + "}" for f in plan["facts"]) or "none"
        lines.append(f"[style: {s['meta']['expr_spec']}] objective: {plan['objective']} | facts: {facts} | "
                     f"ask: {plan['ask']}\ncustomer: {s['meta']['customer_text']}\nagent: {s['messages'][-1]['content']}")
    return "\n".join(lines) + "\n"


_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(text: str) -> str:
    t = _THINK.sub("", text or "").strip()
    t = re.sub(r"^(agent|reply|response|assistant)\s*:\s*", "", t, flags=re.I)
    if len(t) > 1 and t[0] in "\"'“" and t[-1] in "\"'”":
        t = t[1:-1].strip()
    return t


def score(text: str, verbosity: str) -> float:
    """Among passing candidates prefer a length near the middle of the verbosity band."""
    target = {"low": 12, "med": 26, "high": 45}[verbosity]
    return -abs(len(words(text)) - target)


async def complete(client: httpx.AsyncClient, a, messages: list[dict]) -> list[str]:
    body = {"model": a.model, "messages": messages, "temperature": a.temperature, "top_p": 0.95,
            "max_tokens": a.max_tokens, "n": a.n}
    if a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in range(4):
        try:
            r = await client.post(f"{a.base_url.rstrip('/')}/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {a.api_key}"})
            r.raise_for_status()
            return [c["message"]["content"] or "" for c in r.json()["choices"]]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            if attempt == 3:
                print(f"  request failed: {e}", file=sys.stderr)
                return []
            await asyncio.sleep(2 ** attempt)
    return []


async def run(a) -> None:
    prompts = read_jsonl(a.inp)
    if a.limit:
        prompts = prompts[:a.limit]
    seeds = read_jsonl(a.seed_file) if a.seed_file and os.path.exists(a.seed_file) else []
    done = set()
    if os.path.exists(a.raw_out) and not a.overwrite:
        done = {r["id"] for r in read_jsonl(a.raw_out)}
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(prompts)} prompts, {len(done)} already done, {len(todo)} to generate "
          f"(n={a.n}, concurrency={a.concurrency})")

    os.makedirs(os.path.dirname(a.raw_out) or ".", exist_ok=True)
    raw_f = open(a.raw_out, "w" if a.overwrite else "a", encoding="utf-8")
    sem = asyncio.Semaphore(a.concurrency)
    stats = {"prompts": 0, "kept": 0, "cands": 0, "cand_pass": 0}
    t0 = time.time()

    async with httpx.AsyncClient(timeout=a.timeout) as client:
        async def one(rec: dict) -> None:
            plan, expr = record_plan_expr(rec)
            rng = random.Random(rec["id"])
            msgs = prompt_messages(rec)
            sys_msg = msgs[0]["content"] + TEACHER_GUIDELINES + few_shot_block(seeds, expr.language, plan.intent, rng,
                                                                               a.shots)
            async with sem:
                outs = await complete(client, a, [{"role": "system", "content": sys_msg}, *msgs[1:]])
            cands = []
            for o in outs:
                t = clean(o)
                cands.append({"text": t, "issues": check_target(t, plan, expr)})
            passing = sorted({c["text"] for c in cands if not c["issues"]}, key=lambda t: -score(t, expr.verbosity))
            stats["prompts"] += 1
            stats["cands"] += len(cands)
            stats["cand_pass"] += sum(1 for c in cands if not c["issues"])
            kept = passing[:a.keep]
            stats["kept"] += bool(kept)
            raw_f.write(json.dumps({"id": rec["id"], "lang": expr.language, "expr_spec": rec["meta"]["expr_spec"],
                                    "tag": rec["meta"]["tag"], "candidates": cands, "kept": kept},
                                   ensure_ascii=False) + "\n")
            raw_f.flush()
            if stats["prompts"] % 200 == 0:
                el = time.time() - t0
                print(f"  {stats['prompts']}/{len(todo)}  kept={stats['kept']}  "
                      f"cand pass={stats['cand_pass'] / max(1, stats['cands']):.0%}  {stats['prompts'] / el:.1f} prompt/s")

        await asyncio.gather(*(one(r) for r in todo))
    raw_f.close()

    # assemble train file from everything in raw_out (so resumed runs are complete)
    by_id = {p["id"]: p for p in prompts}
    out_rows, rejects = [], []
    for r in read_jsonl(a.raw_out):
        rec = by_id.get(r["id"])
        if rec is None:
            continue
        if not r["kept"]:
            rejects.append(r)
        for j, t in enumerate(r["kept"]):
            row = json.loads(json.dumps(rec))
            row["id"] = f"{rec['id']}-{j}" if j else rec["id"]
            row["source"] = "teacher"
            row["messages"] = prompt_messages(rec) + [{"role": "assistant", "content": t}]
            out_rows.append(row)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(a.rejects_out, "w", encoding="utf-8") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"done: {len(out_rows)} train rows -> {a.out}; {len(rejects)} prompts with no passing candidate -> "
          f"{a.rejects_out}")
    if stats["cands"]:
        print(f"this run: candidate pass rate {stats['cand_pass'] / stats['cands']:.1%}, "
              f"prompt yield {stats['kept'] / max(1, stats['prompts']):.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.getenv("TEACHER_BASE_URL", "http://localhost:8001/v1"))
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "teacher"))
    ap.add_argument("--api-key", default=os.getenv("TEACHER_API_KEY", "EMPTY"))
    ap.add_argument("--in", dest="inp", default=os.path.join(ROOT, "data/sft/prompts.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/sft/train.jsonl"))
    ap.add_argument("--raw-out", default=os.path.join(ROOT, "data/sft/teacher_raw.jsonl"))
    ap.add_argument("--rejects-out", default=os.path.join(ROOT, "data/sft/teacher_rejects.jsonl"))
    ap.add_argument("--seed-file", default=os.path.join(ROOT, "data/sft/seed.jsonl"), help="few-shot source")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--n", type=int, default=4, help="candidates per prompt")
    ap.add_argument("--keep", type=int, default=1, help="max passing candidates kept per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--no-think", action="store_true", help="disable thinking for Qwen3-style teachers")
    ap.add_argument("--overwrite", action="store_true", help="ignore existing --raw-out instead of resuming")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
