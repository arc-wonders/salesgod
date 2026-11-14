# Expression Engine (Layer 6) fine-tuning

The Layer 6 LLM only turns a structured `TurnPlan` + `ExpressionParams` into spoken text. It has to:

- write facts only as `{slot}` placeholders, never as digits
- mention every fact
- end with the plan's `ask` when there is one
- obey `must_not_say`
- follow the style params (language, formality, verbosity, complexity, warmth, proactivity, acknowledge_emotion)

This folder builds the SFT data, trains a LoRA on a 7–8B instruct model, and evaluates it against the runtime validator.

**Train and serve use the same prompt.** `build_sft.py` calls `acie.compiler.PromptCompiler`, `acie.schemas` and a fresh `BankDemoDomain`/`MockBankAPI` for every prompt. `check_sft.py` recompiles each record and fails if the prompt differs from what the runtime would send. If you change `acie/compiler.py`, rebuild the data.

## Files

| file | needs | what |
|---|---|---|
| `sft_common.py` | acie | Scenario sampling: BankDemo recipes (every FSM branch, plus "didn't understand" re-explanations), 14 synthetic retail-banking plans (EMI, EMI bounce, KYC due/received, failed UPI + complaint, FD rates, card block/blocked, loan offer/ineligible, card bill, fraud check, statement), policy-like `ExpressionParams` weighting, customer utterances in en/hi/hinglish, short histories |
| `checks.py` | acie | Runtime `StreamingValidator` (fed in chunks) plus extra checks: language/script, length vs verbosity, ask at the end, must_not_say (literal text plus keyword patterns), number/month words, acknowledgement when requested, formatting, AI/IVR clichés, over-apology, textbook Hindi |
| `seed_data.py` | – | 142 hand-written gold targets (64 hi / 57 en / 21 hinglish) |
| `build_sft.py` | acie | `--mode seed` writes `data/sft/seed.jsonl` and checks it. `--mode teacher` writes `data/sft/prompts.jsonl` and `data/sft/eval_prompts.jsonl` |
| `check_sft.py` | acie | Checks any SFT jsonl: every target plus prompt identity |
| `teacher_generate.py` | acie, httpx | Completes the prompts with a big model on an OpenAI-compatible endpoint, keeps the best passing candidate, and logs rejects with reasons |
| `train_lora.py` | torch, transformers, peft | bf16 LoRA (or `--qlora`). Loss only on the final assistant turn |
| `merge_lora.py` | torch, transformers, peft | Merges the adapter into bf16 weights for vLLM |
| `eval_expression.py` | acie, httpx | Compares base vs fine-tuned on held-out prompts: validator pass, fact coverage, language match, length, ask, all checks, p50/p95 latency to first validated sentence |

Record format (one JSON per line):

```json
{"id": "...", "source": "seed|teacher|eval",
 "messages": [{"role":"system",...}, ...history..., {"role":"user","content":"CUSTOMER SAID: ...\n\nPLAN:\n{...}"}, {"role":"assistant","content":"<target>"}],
 "meta": {"tag": "transfer_over_limit", "intent": "...", "lang": "hi", "expr_spec": "hi h l l h l ack",
          "customer_text": "...", "plan": {TurnPlan}, "expr": {ExpressionParams}}}
```

History agent turns contain real values (₹…). That is what the pipeline stores, so the model learns to read digits in the history without ever writing them. Prompt files have no assistant message.

## End-to-end

Hosts: **voice** has 4× A100 80GB (teacher, training, serving). **test3-ocr** has 1× A100 40GB (extra training).

```bash
# 0. on every box: the repo + acie (pydantic, httpx) in the project venv
python -m venv .venv && .venv/bin/pip install -e .

# 1. gold seed set: compile and check (CPU, seconds)
.venv/bin/python training/expression/build_sft.py --mode seed
.venv/bin/python training/expression/check_sft.py --show 6

# 2. prompts for the teacher, plus a held-out eval set (CPU, seconds)
.venv/bin/python training/expression/build_sft.py --mode teacher --n 8000 --eval-n 400

# 3. teacher on the voice box (all 4 GPUs; stop the live expression server first)
python -m venv ~/venv-vllm && ~/venv-vllm/bin/pip install -r deploy/requirements-vllm.txt
PATH=~/venv-vllm/bin:$PATH deploy/serve_teacher.sh            # Qwen2.5-72B-Instruct, TP=4, port 8001

# 4. generate and filter. Smoke test with --limit 50 first, read data/sft/teacher_raw.jsonl, then run everything.
.venv/bin/python training/expression/teacher_generate.py --base-url http://voice:8001/v1 --model teacher --limit 50
.venv/bin/python training/expression/teacher_generate.py --base-url http://voice:8001/v1 --model teacher
.venv/bin/python training/expression/check_sft.py data/sft/train.jsonl --quiet --show 20   # must be 100% (already filtered)

# 5. train (separate training venv). One A100 80GB, about 1h for 8k rows x 2 epochs on 7B.
python -m venv ~/venv-train && ~/venv-train/bin/pip install -r training/expression/requirements.txt && ~/venv-train/bin/pip install -e .
CUDA_VISIBLE_DEVICES=0 ~/venv-train/bin/python training/expression/train_lora.py \
    --model Qwen/Qwen2.5-7B-Instruct --data data/sft/train.jsonl --seed-data data/sft/seed.jsonl \
    --out runs/acie-expr-qwen7b
#    40GB box (test3-ocr): add  --batch 4 --grad-accum 4   (bf16 LoRA fits)  or  --qlora  for more headroom

# 6. merge for production serving
~/venv-train/bin/python training/expression/merge_lora.py --adapter runs/acie-expr-qwen7b/adapter \
    --out models/acie-expression-qwen7b

# 7. serve base + LoRA side by side for the A/B eval (one GPU)
MODE=lora BASE_MODEL=Qwen/Qwen2.5-7B-Instruct ADAPTER=runs/acie-expr-qwen7b/adapter \
    PATH=~/venv-vllm/bin:$PATH deploy/serve_vllm.sh

# 8. eval base vs fine-tuned (concurrency 1 = clean latency; add --concurrency 16 for latency under load)
.venv/bin/python training/expression/eval_expression.py --base-url http://voice:8000/v1 \
    --models base,acie-expression --out runs/eval_qwen7b.json

# 9. production: serve the merged model and point acie at it
MODE=merged MODEL_PATH=models/acie-expression-qwen7b PATH=~/venv-vllm/bin:$PATH deploy/serve_vllm.sh
export ACIE_LLM_BASE_URL=http://voice:8000/v1 ACIE_LLM_MODEL=acie-expression
.venv/bin/python -m acie.cli --backend openai --debug
```

Acceptance bar before switching the demo to the LLM backend:

- `validator_pass` ≥ 0.98, `fact_coverage` ≥ 0.99, `lang_match` ≥ 0.98, `all_checks` ≥ 0.9
- `first_sentence_ms_p95` under about 350 ms on one A100 at concurrency 1

A validator failure doesn't break the call, because the pipeline falls back to the template. It does sound robotic, though, so track the pass rate.

## Choices and knobs

- **Base model.** The default is `Qwen/Qwen2.5-7B-Instruct`: strong instruction following, decent Hindi, good vLLM support, Apache-2.0. Alternatives:
  - `meta-llama/Llama-3.1-8B-Instruct`: often more natural Hindi and Hinglish. Gated, Llama licence.
  - `sarvamai/sarvam-m` (24B, Mistral-based, Indic-tuned) for the best Hindi, if latency allows. Use TP=2 to serve it.
  - `sarvamai/sarvam-1` (2B base, not instruct): very fast, but needs much more SFT.
  - `CohereForAI/aya-expanse-8b`: multilingual, non-commercial licence.
  - `google/gemma-2-9b-it` or `Qwen/Qwen3-8B` (pass `--no-think` to the teacher and disable thinking when serving).

  Run `eval_expression.py` on two candidates before committing. Hindi naturalness is worth a human listen too, not just the metrics.
- **Teacher.** `Qwen2.5-72B-Instruct` or `Llama-3.3-70B-Instruct` at bf16 on 4×A100 80GB (TP=4). The teacher gets the student's exact system prompt, plus extra guidelines (register, loanwords, no clichés) and 4 gold examples in the same language. The student trains on the plain runtime prompt only, so it has to learn those guidelines.
- **Data mix.** About 55% BankDemo branches (the demo path) and 45% synthetic plans (generalisation to new objectives and slot names). Language mix is 45% hi, 40% en, 15% hinglish. Expression params are sampled like `acie.personalization.policy`: mostly `med`, with frustrated/confused/urgent overrides, plus 25% uniform coverage. About 12% of prompts are re-explanations. Gold seeds are upsampled ×4.
- **LoRA.** r=16, alpha=32, all linear layers, lr 1.5e-4 cosine, 2 epochs, bf16, gradient checkpointing, sdpa attention (`--attn flash_attention_2` if flash-attn is installed). Rows longer than `--max-len` are dropped, never truncated.
- **Agent voice** is masculine in Hindi ("सकता हूँ"), matching the BankDemo fallback templates. If the TTS voice is female, change both the fallbacks and the seeds, and regenerate.
- `must_not_say` entries are descriptive ("limited time pressure"). `checks.MNS_PATTERNS` maps keywords to surface patterns. Add a pattern there when you add a new kind of entry.
