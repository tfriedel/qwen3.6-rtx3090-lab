#!/usr/bin/env bash
# Run google DiffusionGemma-26B-A4B-it (text *diffusion* MoE, 3.8B active) under vLLM in Docker.
#
#   ./diffusion-gemma.sh start           # bring up the container
#   ./diffusion-gemma.sh stop
#   ./diffusion-gemma.sh status
#   ./diffusion-gemma.sh logs
#   ./diffusion-gemma.sh restart
#
# Endpoint after start: http://localhost:8031/v1  (model: diffusiongemma-26b-a4b-fp8)
#
# NOTE (2026-06-10): DiffusionGemma support is NOT in any released vLLM.
# This script runs the CI image built from vLLM PR #45163 ("[Model] Add
# DiffusionGemma Support", branch dgemma, commit 74b5964) pulled from vLLM's
# public ECR. Once the PR merges into a nightly, swap IMAGE for the regular
# vllm/vllm-openai nightly and drop the --entrypoint override if needed.
#
# Quant: RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 (compressed-tensors, ~13 GB)
# on a single 3090 via vLLM's Marlin weight-only FP4 fallback (native NVFP4
# kernels need SM_89+; Ampere dequants to bf16 — memory savings, no FP4 kernel
# speedup; all the speed comes from parallel canvas decoding: canvas 256
# tokens, ≤48 denoising steps).
#
# TRITON_ATTN is mandatory on Ampere: the diffusion per-sequence causal mask
# needs FA4 (Blackwell-only) on the flash-attn path.
#
# Why not the FP8-dynamic quant (~25 GB, needs 2 cards)? All multi-GPU modes
# crash on this rig — keep these failure modes in mind before "upgrading":
#   - TP=2: Marlin requires 64-aligned GEMM dims; TP halves moe_intermediate
#     704→352 and dense intermediate 2112→1056, neither 64-aligned →
#     "Invalid thread config" at startup.
#   - TP=2 + --enable-expert-parallel: fixes routed experts, but the dense
#     MLP still TP-shards → same crash.
#   - PP=2: diffusion sampler's PP broadcast is unimplemented in PR #45163
#     (AssertionError in v1/worker/gpu/pp_utils.py:181 during warmup).
# At TP=1 every dim is 64-aligned and Marlin is happy.

set -euo pipefail

CONTAINER="vllm-diffusiongemma-26b"
MODEL="RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"
SERVED_NAME="diffusiongemma-26b-a4b-nvfp4"
PORT="8031"
GPU="${GPU:-0}"                # override: GPU=2 ./diffusion-gemma.sh start
MAX_LEN="${MAX_LEN:-16384}"    # native max 262144; 32K + 4 seqs OOMs a 24 GB card
MAX_SEQS="${MAX_SEQS:-2}"      # diffusion sampler logits ≈ 270 MB fp32 per seq slot
IMAGE="public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:74b5964f02c7e023fadd3004cfac8a61c52eef1f"
URL="http://localhost:${PORT}"

is_running() {
  docker ps -q --filter "name=^${CONTAINER}$" | grep -q .
}

cmd_status() {
  if ! is_running; then
    echo "stopped"
    return
  fi
  docker ps --filter "name=^${CONTAINER}$" --format "running: {{.Status}} | {{.Ports}}"
  if curl -sf -m 3 "$URL/v1/models" >/dev/null 2>&1; then
    curl -sf "$URL/v1/models" | python3 -c "
import json, sys
m = json.load(sys.stdin)['data'][0]
print('  served:       ', m['id'])
print('  max_model_len:', f\"{m['max_model_len']:,}\")
"
  else
    echo "  (endpoint not responding on $URL/v1/models — may still be loading)"
  fi
}

cmd_stop() {
  if ! is_running; then
    echo "nothing to stop"
    return
  fi
  echo "stopping $CONTAINER..."
  docker rm -f "$CONTAINER" >/dev/null
}

wait_ready() {
  local timeout="${1:-600}" t=0
  printf "waiting for %s/v1/models" "$URL"
  until curl -sf -m 2 "$URL/v1/models" >/dev/null 2>&1; do
    if (( t >= timeout )); then
      echo " — timeout after ${timeout}s"
      docker logs --tail 30 "$CONTAINER" 2>&1 | sed 's/^/  /'
      return 1
    fi
    if ! is_running; then
      echo " — container exited"
      docker logs --tail 30 "$CONTAINER" 2>&1 | sed 's/^/  /'
      return 1
    fi
    sleep 3; t=$((t + 3)); printf "."
  done
  echo " ready (after ${t}s)"
}

cmd_start() {
  if is_running; then
    echo "already running"
    cmd_status
    return 0
  fi
  echo "starting $CONTAINER on GPU $GPU, port $PORT..."
  docker run -d \
    --name "$CONTAINER" \
    --gpus "\"device=${GPU}\"" \
    --ipc=host \
    --restart unless-stopped \
    -p "${PORT}:8000" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_NO_USAGE_STATS=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
    --entrypoint vllm \
    "$IMAGE" \
    serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --attention-backend TRITON_ATTN \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --gpu-memory-utilization 0.79 \
    --host 0.0.0.0 --port 8000 \
    >/dev/null
  wait_ready 600
  cmd_status
}

cmd_logs() {
  if ! is_running; then
    echo "container not running"
    return 1
  fi
  exec docker logs -f "$CONTAINER"
}

usage() {
  cat <<EOF
Usage: $0 {start|stop|restart|status|logs}

Env:
  GPU=<idx>      GPU device index (default: 0)
  MAX_LEN=<n>    max model length (default: 32768)
  MAX_SEQS=<n>   max concurrent seqs (default: 4; model caps at 8)
  HF_TOKEN=...   passed through if model is gated

Endpoint: $URL/v1   (model: $SERVED_NAME)
EOF
}

case "${1:-}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status)  cmd_status ;;
  logs)    cmd_logs ;;
  ""|-h|--help) usage ;;
  *) echo "unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
