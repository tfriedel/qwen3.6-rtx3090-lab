#!/usr/bin/env bash
# Probe max stable max-model-len for Qwen3.6-35B-A3B-AWQ on 1 or 2 GPUs.
# Picks the largest value that boots, holds GPU memory, and serves a smoke request.
#
# Usage:  bash scripts/probe_moe_ctx.sh tp1
#         bash scripts/probe_moe_ctx.sh tp2
set -euo pipefail

mode="${1:-}"
case "$mode" in
  tp1) base=docker-compose.35b-a3b-awq-tp1.yml; cands=(8192 12288 16384 20480) ;;
  tp2) base=docker-compose.35b-a3b-awq.yml;     cands=(131072 196608 262144 327680 393216) ;;
  *) echo "usage: $0 tp1|tp2"; exit 2 ;;
esac

cd "$(dirname "$0")/../compose"
container=vllm-qwen36-35b-a3b
url=http://localhost:8021

probe() {
  local L=$1
  local f=docker-compose.probe.yml
  if [[ "$mode" == "tp1" ]]; then
    sed "s/MAX_MODEL_LEN_PLACEHOLDER/${L}/" "$base" > "$f"
  else
    python3 -c "
import re
s=open('$base').read()
s=re.sub(r'(--max-model-len\s*\n\s*-\s*)\"[0-9]+\"', r'\1\"$L\"', s)
open('$f','w').write(s)
"
  fi
  echo
  echo "=== Probing max-model-len=${L} (${mode}) ==="
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker compose -f "$f" up -d >/dev/null
  local t=0
  while (( t < 240 )); do
    if curl -sf -m 2 "$url/v1/models" >/dev/null 2>&1; then
      echo "  READY after ${t}s"
      # quick smoke
      local out
      out=$(curl -s "$url/v1/chat/completions" -H "Content-Type: application/json" -d '{
        "model":"qwen3.6-35b-a3b-awq",
        "messages":[{"role":"user","content":"Say hi in one short sentence."}],
        "max_tokens":40,"temperature":0.6,
        "chat_template_kwargs":{"enable_thinking":false}
      }')
      echo "  smoke: $(echo "$out" | python3 -c "import sys,json; r=json.load(sys.stdin); print('ok ->',(r['choices'][0]['message']['content'] or '')[:60])" 2>&1 | head -1)"
      local pool
      pool=$(docker logs "$container" 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1)
      echo "  $pool"
      docker rm -f "$container" >/dev/null 2>&1 || true
      return 0
    fi
    if ! docker ps -q --filter "name=^${container}$" | grep -q .; then
      echo "  FAIL: container exited"
      docker logs "$container" 2>&1 | grep -E "OutOfMemoryError|ValueError|RuntimeError|raise |maximum model length|Available KV cache|Model loading took" | grep -v "Failed core proc" | tail -6 | sed 's/^/    /'
      docker rm -f "$container" >/dev/null 2>&1 || true
      return 1
    fi
    sleep 4; t=$((t+4))
  done
  echo "  TIMEOUT after ${t}s"
  docker rm -f "$container" >/dev/null 2>&1 || true
  return 1
}

echo "Probing $mode candidates: ${cands[*]}"
last_ok=0
for L in "${cands[@]}"; do
  if probe "$L"; then
    last_ok=$L
  else
    break
  fi
done
echo
echo "=== RESULT: largest stable max-model-len for $mode = $last_ok ==="
