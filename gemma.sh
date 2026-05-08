#!/usr/bin/env bash
# Manage the Gemma-4 vLLM / llama.cpp stack on this box.
# Sibling to ./qwen.sh and ./qwen-moe.sh — different containers, ports, GPUs.
#
#   ./gemma.sh start tp1     # vLLM AWQ + DFlash spec (Reddit recipe), 1× GPU
#   ./gemma.sh start gguf    # llama.cpp UD-Q4_K_XL + ngram-mod spec, 1× GPU (3090 sweet spot)
#   ./gemma.sh start e4b     # vLLM bf16 Gemma-4-E4B-it (small/fast), 1× GPU
#   ./gemma.sh stop
#   ./gemma.sh status
#   ./gemma.sh logs
#   ./gemma.sh restart <mode>
#
# Endpoints:
#   tp1:   http://localhost:8023/v1  (model name: gemma-4-26b-a4b-awq)
#   gguf:  http://localhost:8024/v1  (model name: gemma-4-26b-a4b-gguf)
#   e4b:   http://localhost:8025/v1  (model name: gemma-4-e4b)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
COMPOSE_DIR="${SCRIPT_DIR}/gemma/compose"

if [[ ! -d "$COMPOSE_DIR" ]]; then
  echo "ERROR: compose dir not found at $COMPOSE_DIR" >&2
  exit 1
fi

declare -A MODE_FILE=(
  [tp1]="docker-compose.26b-vllm-tp1.yml"
  [gguf]="docker-compose.26b-gguf.yml"
  [e4b]="docker-compose.e4b-vllm.yml"
)

declare -A MODE_CONTAINER=(
  [tp1]="vllm-gemma4-26b-a4b"
  [gguf]="llamacpp-gemma4-26b-a4b"
  [e4b]="vllm-gemma4-e4b"
)

declare -A MODE_URL=(
  [tp1]="http://localhost:8023"
  [gguf]="http://localhost:8024"
  [e4b]="http://localhost:8025"
)

declare -A MODE_DESC=(
  [tp1]="vLLM/AWQ + DFlash-13 — 20K ctx, 95/179/117 TPS narr/code/explain, 1× GPU (code-heavy short-ctx only)"
  [gguf]="llama.cpp/UD-Q4_K_XL + ngram-mod — 262K ctx, ~127-131 TPS flat, 1× GPU (3090 sweet spot, recommended default)"
  [e4b]="vLLM/bf16 Gemma-4-E4B-it — 128K ctx, fast small variant, 1× GPU"
)

CONTAINER=""
URL=""

resolve_mode_vars() {
  local m="$1"
  CONTAINER="${MODE_CONTAINER[$m]:-}"
  URL="${MODE_URL[$m]:-}"
}

usage() {
  cat <<EOF
Usage: $0 <command> [args]

Commands:
  start <mode>     Bring up a variant (modes: tp1, gguf, e4b)
  stop             Stop and remove the running container
  restart <mode>   stop && start <mode>
  status           Show what's currently up
  logs             Follow container logs (Ctrl-C to detach)
  modes            List available variants

Examples:
  $0 start gguf
  $0 restart tp1
  $0 status
EOF
}

current_mode() {
  for m in "${!MODE_FILE[@]}"; do
    local c="${MODE_CONTAINER[$m]}"
    if docker ps -q --filter "name=^${c}$" | grep -q .; then
      local cf
      cf="$(docker inspect "$c" --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}' 2>/dev/null || true)"
      [[ "$cf" == *"${MODE_FILE[$m]}"* ]] && { echo "$m"; return; }
    fi
  done
}

cmd_modes() {
  for m in tp1 gguf e4b; do
    printf "  %-5s %s\n" "$m" "${MODE_DESC[$m]}"
  done
}

cmd_status() {
  local cur
  cur="$(current_mode)"
  if [[ -z "$cur" ]]; then
    echo "stopped"
    return
  fi
  resolve_mode_vars "$cur"
  echo "running mode: $cur (${MODE_DESC[$cur]:-})"
  docker ps --filter "name=^${CONTAINER}$" --format "  {{.Status}} | {{.Ports}}"
  if curl -sf -m 3 "$URL/v1/models" >/dev/null 2>&1; then
    curl -sf "$URL/v1/models" | python3 -c "
import json, sys
m = json.load(sys.stdin)['data'][0]
print('  served:       ', m['id'])
ml = m.get('max_model_len')
if ml is not None:
    print('  max_model_len:', f'{ml:,}')
" 2>/dev/null || true
  else
    echo "  (endpoint not yet responding on $URL/v1/models — model may still be loading)"
  fi
}

cmd_stop() {
  local cur
  cur="$(current_mode)"
  if [[ -z "$cur" ]]; then
    echo "nothing to stop"
    return
  fi
  resolve_mode_vars "$cur"
  local f="${MODE_FILE[$cur]:-}"
  echo "stopping $cur..."
  if [[ -n "$f" ]]; then
    (cd "$COMPOSE_DIR" && docker compose -f "$f" down)
  else
    docker rm -f "$CONTAINER" >/dev/null
  fi
}

wait_ready() {
  local timeout="${1:-360}"
  local t=0
  printf "waiting for %s/v1/models" "$URL"
  until curl -sf -m 2 "$URL/v1/models" >/dev/null 2>&1; do
    if (( t >= timeout )); then
      echo " — timeout after ${timeout}s"
      docker logs --tail 30 "$CONTAINER" 2>&1 | sed 's/^/  /'
      return 1
    fi
    if ! docker ps -q --filter "name=^${CONTAINER}$" | grep -q .; then
      echo " — container exited"
      docker logs --tail 30 "$CONTAINER" 2>&1 | sed 's/^/  /'
      return 1
    fi
    sleep 3
    t=$((t + 3))
    printf "."
  done
  echo " ready (after ${t}s)"
}

cmd_start() {
  local mode="${1:-}"
  if [[ -z "$mode" || -z "${MODE_FILE[$mode]:-}" ]]; then
    echo "ERROR: unknown mode '$mode'. Available:" >&2
    cmd_modes >&2
    return 2
  fi
  local cur
  cur="$(current_mode)"
  if [[ -n "$cur" ]]; then
    if [[ "$cur" == "$mode" ]]; then
      echo "already running mode '$mode'"
      cmd_status
      return 0
    fi
    echo "switching from '$cur' to '$mode'..."
    cmd_stop
  fi
  resolve_mode_vars "$mode"
  echo "starting $mode (${MODE_DESC[$mode]})..."
  (cd "$COMPOSE_DIR" && docker compose -f "${MODE_FILE[$mode]}" up -d)
  wait_ready 480
  cmd_status
}

cmd_logs() {
  local cur
  cur="$(current_mode)"
  if [[ -z "$cur" ]]; then
    echo "container not running"
    return 1
  fi
  resolve_mode_vars "$cur"
  exec docker logs -f "$CONTAINER"
}

cmd="${1:-}"
shift || true
case "$cmd" in
  start)   cmd_start "$@" ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; cmd_start "$@" ;;
  status)  cmd_status ;;
  logs)    cmd_logs ;;
  modes)   cmd_modes ;;
  ""|-h|--help) usage ;;
  *) echo "unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
