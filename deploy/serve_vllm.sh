#!/usr/bin/env bash
# Serve the ACIE Expression Engine (Layer 6) with vLLM on the voice box (4x A100 80GB).
#
#   MODE=merged MODEL_PATH=/models/acie-expression-qwen7b deploy/serve_vllm.sh      # merged weights (prod)
#   MODE=lora   BASE_MODEL=Qwen/Qwen2.5-7B-Instruct ADAPTER=runs/acie-expr-qwen7b/adapter deploy/serve_vllm.sh
#                                                    # base + LoRA: serves "base" AND "acie-expression" (A/B, eval)
#
# Then point acie at it:
#   export ACIE_LLM_BASE_URL=http://<voice-box>:${PORT}/v1
#   export ACIE_LLM_MODEL=acie-expression
#   .venv/bin/python -m acie.cli --backend openai --debug
set -euo pipefail

MODE="${MODE:-merged}"                                  # merged | lora
MODEL_PATH="${MODEL_PATH:-models/acie-expression}"      # MODE=merged
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"    # MODE=lora
ADAPTER="${ADAPTER:-runs/acie-expr-lora/adapter}"       # MODE=lora
LORA_RANK="${LORA_RANK:-16}"                            # must be >= the adapter's r
SERVED_NAME="${SERVED_NAME:-acie-expression}"           # = ACIE_LLM_MODEL
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0}"                                       # 7-8B fits on one A100; keep the other GPUs for teacher/training
TP="${TP:-1}"                                           # tensor parallel; TP=2 shaves a little per-token latency
DP="${DP:-1}"                                           # data-parallel replicas (needs vLLM >= 0.8); set GPUS=0,1,.. accordingly
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"                  # prompt ~0.8-1.5k tokens (system + 6 history turns + plan) + 160 out
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"                      # concurrent calls per replica
GPU_UTIL="${GPU_UTIL:-0.85}"

export CUDA_VISIBLE_DEVICES="$GPUS"

COMMON=(
  --host "$HOST" --port "$PORT"
  --dtype bfloat16
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-prefix-caching          # the system prompt + history prefix is shared turn-to-turn -> big TTFT win
)
if [[ "$DP" -gt 1 ]]; then COMMON+=(--data-parallel-size "$DP"); fi

echo "ACIE_LLM_BASE_URL=http://$(hostname -f 2>/dev/null || hostname):${PORT}/v1"
echo "ACIE_LLM_MODEL=${SERVED_NAME}"

if [[ "$MODE" == "merged" ]]; then
  exec vllm serve "$MODEL_PATH" --served-model-name "$SERVED_NAME" "${COMMON[@]}"
elif [[ "$MODE" == "lora" ]]; then
  # base model is exposed as "base", the adapter as $SERVED_NAME
  exec vllm serve "$BASE_MODEL" --served-model-name base \
    --enable-lora --max-lora-rank "$LORA_RANK" --max-loras 2 \
    --lora-modules "${SERVED_NAME}=${ADAPTER}" \
    "${COMMON[@]}"
else
  echo "MODE must be merged or lora" >&2; exit 2
fi
