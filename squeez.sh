#!/usr/bin/env bash
# Run KRLabsOrg/squeez-2b under vLLM in Docker.
#
#   ./squeez.sh start           # bring up the container
#   ./squeez.sh stop
#   ./squeez.sh status
#   ./squeez.sh logs
#   ./squeez.sh restart
#
# Endpoint after start: http://localhost:8030/v1  (model: KRLabsOrg/squeez-2b)

set -euo pipefail

CONTAINER="vllm-squeez-2b"
MODEL="KRLabsOrg/squeez-2b"
PORT="8030"
GPU="${GPU:-3}"          # override: GPU=1 ./squeez.sh start
MAX_LEN="${MAX_LEN:-262144}"   # native max is 262144; override e.g. MAX_LEN=131072
IMAGE="vllm/vllm-openai@sha256:674922aae790c2cbf45f4e844098d227b80d40a74bfc7797a444d213a221879f"
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
  local timeout="${1:-300}" t=0
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
    ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
    "$IMAGE" \
    --model "$MODEL" \
    --dtype bfloat16 \
    --max-model-len "$MAX_LEN" \
    >/dev/null
  wait_ready 300
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
  GPU=<idx>      GPU device index (default: 3)
  MAX_LEN=<n>    max model length (default: 262144, native cap)
  HF_TOKEN=...   passed through if model is gated

Endpoint: $URL/v1   (model: $MODEL)
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
