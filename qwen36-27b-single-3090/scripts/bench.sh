#!/usr/bin/env bash
#
# Canonical bench against the running vLLM service.
#   - 3 warmup + 3 narrative + 2 code runs
#   - reports wall time, completion tokens, TPS per request
#   - shows MTP SpecDecoding metrics from docker logs at the end
#
# Prereq: stack is running (`cd compose && docker compose up -d`) and reports
# "Application startup complete".
#
# Env vars:
#   URL            Override endpoint. Default: http://localhost:8020
#   MODEL          Served model name. Default: qwen3.6-27b-autoround
#   CONTAINER      Docker container name for log scraping. Default: vllm-qwen36-27b

set -euo pipefail

URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-qwen3.6-27b-autoround}"
CONTAINER="${CONTAINER:-vllm-qwen36-27b}"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }
}
need curl
need python3

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  echo "  Start with: cd compose && docker compose up -d" >&2
  exit 1
fi

bench() {
  local label="$1"
  local prompt="$2"
  local max="$3"
  local start end wall comp tps
  start=$(date +%s.%N)
  local resp
  resp="$(curl -sf "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":${prompt}}],\"max_tokens\":${max},\"temperature\":0.6,\"top_p\":0.95,\"chat_template_kwargs\":{\"enable_thinking\":false}}")"
  end=$(date +%s.%N)
  wall=$(python3 -c "print(f'{${end} - ${start}:.2f}')")
  comp=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "?")
  tps=$(python3 -c "print(f'{${comp} / ${wall}:.2f}')" 2>/dev/null || echo "?")
  printf "  %-12s comp=%-4s wall=%5ss  %6s TPS\n" "$label" "$comp" "$wall" "$tps"
}

NARR='"Write a detailed 800-word essay explaining transformer attention."'
CODE='"Write a Python implementation of quicksort with comments explaining each step."'

echo "=== Warmup (3x) ==="
for i in 1 2 3; do bench "w$i" "$NARR" 1000; done
echo ""
echo "=== Narrative (3x, 1000 tok) ==="
for i in 1 2 3; do bench "narr$i" "$NARR" 1000; done
echo ""
echo "=== Code (2x, 800 tok) ==="
for i in 1 2; do bench "code$i" "$CODE" 800; done
echo ""

# GPU state
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
  echo ""
fi

# MTP stats
if command -v docker >/dev/null 2>&1 && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "=== Last 3 SpecDecoding metrics (MTP accept) ==="
  docker logs "${CONTAINER}" 2>&1 | grep "SpecDecoding metrics" | tail -3 || true
fi
