# Deploy: Expression Engine on vLLM

ACIE's `OpenAICompatibleBackend` (`acie/expression.py`) streams from any OpenAI-compatible `/v1/chat/completions`. It reads three variables:

| env | default | meaning |
|---|---|---|
| `ACIE_LLM_BASE_URL` | `http://localhost:8000/v1` | vLLM endpoint |
| `ACIE_LLM_MODEL` | `acie-expression` | served model name |
| `ACIE_LLM_API_KEY` | `EMPTY` | only if vLLM runs with `--api-key` |

## Setup (voice box, 4× A100 80GB)

```bash
python -m venv ~/venv-vllm && ~/venv-vllm/bin/pip install -r deploy/requirements-vllm.txt
export PATH=~/venv-vllm/bin:$PATH
```

## Serve

```bash
# production: merged weights from training/expression/merge_lora.py
MODE=merged MODEL_PATH=models/acie-expression-qwen7b deploy/serve_vllm.sh

# A/B and eval: base + LoRA in one server ("base" and "acie-expression")
MODE=lora BASE_MODEL=Qwen/Qwen2.5-7B-Instruct ADAPTER=runs/acie-expr-qwen7b/adapter deploy/serve_vllm.sh

# then, on the machine running acie
export ACIE_LLM_BASE_URL=http://voice:8000/v1 ACIE_LLM_MODEL=acie-expression
.venv/bin/python -m acie.cli --backend openai --debug
```

Every setting in `serve_vllm.sh` can be overridden with an env var: `GPUS`, `TP`, `DP`, `PORT`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_UTIL`, `SERVED_NAME`, `LORA_RANK`.

### Why these defaults

- **One GPU, TP=1.** A 7–8B bf16 model uses about 16 GB, and replies are short (at most 160 tokens). TP=1 leaves three cards free for the teacher and training. For more concurrent calls, add replicas (`DP=2 GPUS=0,1`, needs vLLM ≥ 0.8) instead of raising TP. TP=2 only shaves a little off per-token latency.
- **`--enable-prefix-caching`.** The system prompt is identical for a given style, and turn N+1's history is turn N's prompt plus two messages. Cached prefixes cut time-to-first-token, which is what the validator and TTS wait on.
- **`--max-model-len 4096`.** A compiled prompt is about 0.8–1.5k tokens (system, last 6 history turns, plan JSON), plus at most 160 output tokens. The headroom covers long histories, and a smaller value frees KV cache for concurrency.
- **Merged vs LoRA.** Merged weights are slightly faster and simpler, so use them in production. Use LoRA mode to compare base and fine-tuned on identical hardware, or to hot-swap adapters.
- **Sampling defaults.** vLLM applies the model's `generation_config.json` for anything the request leaves out. For Qwen2.5 that means top_p 0.8 and repetition_penalty 1.05; acie sets only temperature 0.4 and max_tokens 160. Eval and runtime go through the same server, so they match. Add `--generation-config vllm` to turn this off.

## Teacher (data generation only)

```bash
deploy/serve_teacher.sh                                    # Qwen2.5-72B-Instruct, TP=4, port 8001
TEACHER=meta-llama/Llama-3.3-70B-Instruct deploy/serve_teacher.sh
```

This takes all 4 GPUs, so stop the live expression server first, or run the teacher off-hours. See `training/expression/README.md` for the full pipeline.

## Health check

```bash
curl -s $ACIE_LLM_BASE_URL/models | python -m json.tool
.venv/bin/python training/expression/eval_expression.py --models "$ACIE_LLM_MODEL" --limit 50
```

## Optional infra: Redis sessions, Postgres profiles, pgvector memory, NATS events

All off by default; with no `ACIE_*` store variable set the voice server behaves exactly as before
(in-memory sessions, JSON profiles). Local test stack: `docker compose up -d` (redis 16379, postgres+pgvector 15432, nats 14222;
user/password/db `acie`). Install the client libraries with `pip install -e '.[infra]'`.

| feature | env | notes |
|---|---|---|
| Sessions in Redis (2 h TTL, a restart mid-call resumes from the last finished turn) | `ACIE_SESSION_STORE=redis`, `ACIE_REDIS_URL=redis://host:6379/0` | live call still uses an in-process cache; Redis is the durable copy |
| Profiles in Postgres | `ACIE_PROFILE_STORE=postgres`, `ACIE_PG_DSN=postgresql://acie:acie@host:5432/acie` | table `customer_profiles` is created on first use. Import existing JSON once: `python deploy/migrate_profiles.py data/voice/profiles` (`--dry-run` to check) |
| Semantic memory | `ACIE_MEMORY_STORE=pgvector`, `ACIE_PG_DSN`, `ACIE_MEMORY_DIM` (default 256) | needs the pgvector extension (the compose image has it); table `memories`, HNSW cosine index |
| Context events | `ACIE_NATS_URL=nats://host:4222` | subject `acie.calls.<call_id>.context`; bounded queue, drops on overflow, never blocks a call |

To enable on the VM (do this deliberately, not from a lane): start the services (compose or system packages), add the variables
to the `salesgod-voice` unit environment, install the `[infra]` extra into its venv, run the migration, then restart the service
in a quiet window. Roll back by removing the variables.
