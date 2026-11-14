#!/usr/bin/env bash
# Serve a large teacher model for SFT data generation (training/expression/teacher_generate.py).
# Uses all 4 A100 80GB on the voice box - stop the live expression server first, or set GPUS to spare cards.
#
#   deploy/serve_teacher.sh                                   # Qwen2.5-72B-Instruct, TP=4
#   TEACHER=meta-llama/Llama-3.3-70B-Instruct deploy/serve_teacher.sh
#   then: TEACHER_BASE_URL=http://<voice-box>:8001/v1 TEACHER_MODEL=teacher python training/expression/teacher_generate.py
set -euo pipefail
TEACHER="${TEACHER:-Qwen/Qwen2.5-72B-Instruct}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
PORT="${PORT:-8001}"
export CUDA_VISIBLE_DEVICES="$GPUS"
exec vllm serve "$TEACHER" --served-model-name teacher \
  --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
  --tensor-parallel-size "$TP" --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --max-num-seqs 128 \
