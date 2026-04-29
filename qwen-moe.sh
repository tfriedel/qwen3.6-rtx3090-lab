#!/usr/bin/env bash
# Manage the Qwen3.6-35B-A3B (MoE, AWQ-INT4) vLLM stack on this box.
# Sibling to ./qwen.sh — the two models can run side by side (different
# container, port, GPUs) as long as combined GPU layout fits.
#
#   ./qwen-moe.sh start tp1       # vLLM/AWQ 20K text, ~18 TPS, 1× GPU (launch-bound; don't)
#   ./qwen-moe.sh start tp2       # vLLM/AWQ 200K + vision, ~149 TPS, 2× GPU (GPUs 0-1)
#   ./qwen-moe.sh start tp2-mtp   # vLLM/AWQ + MTP-3 + no-prefix-cache, ~179/264/200 TPS, 2× GPU (GPUs 1-2)
#   ./qwen-moe.sh start gguf      # llama.cpp/IQ4_XS 128K text, ~117 TPS, 1× GPU (single-GPU sweet spot)
#   ./qwen-moe.sh stop
#   ./qwen-moe.sh status
#   ./qwen-moe.sh logs
#   ./qwen-moe.sh restart <mode>
#
# Endpoints:
#   tp1, tp2, tp2-mtp:  http://localhost:8021/v1  (model name: qwen3.6-35b-a3b-awq)
#   gguf:               http://localhost:8022/v1  (model name: qwen3.6-35b-a3b-gguf)
#
# Coexistence with the 27B (./qwen.sh):
#   - tp1 (1× GPU 0): leaves GPUs 1-3 free for 27B (default/text uses 1 GPU)
#   - tp2 (2× GPU 0,1): leaves GPUs 2,3. 27B can run on its tp2 variant if you
#     pin 27B to GPUs 2,3 (edit CUDA_VISIBLE_DEVICES in compose/docker-compose.tp2.yml).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="${SCRIPT_DIR}/qwen36-27b-single-3090"
COMPOSE_DIR="${REPO_DIR}/compose"

if [[ ! -d "$COMPOSE_DIR" ]]; then
  echo "ERROR: compose dir not found at $COMPOSE_DIR" >&2
  echo "  Expected layout: <this-script> alongside qwen36-27b-single-3090/compose/" >&2
  exit 1
fi

# Each mode owns its own container/port — we read these from the mode tables below.
declare -A MODE_FILE=(
  [tp1]="docker-compose.35b-a3b-awq-tp1.yml"
  [tp2]="docker-compose.35b-a3b-awq.yml"
  [tp2-mtp]="docker-compose.35b-a3b-awq-mtp.yml"
  [gguf]="docker-compose.35b-a3b-gguf.yml"
)

declare -A MODE_CONTAINER=(
  [tp1]="vllm-qwen36-35b-a3b"
  [tp2]="vllm-qwen36-35b-a3b"
  [tp2-mtp]="vllm-qwen36-35b-a3b"
  [gguf]="llamacpp-qwen36-35b-a3b"
)

declare -A MODE_URL=(
  [tp1]="http://localhost:8021"
  [tp2]="http://localhost:8021"
  [tp2-mtp]="http://localhost:8021"
  [gguf]="http://localhost:8022"
)

declare -A MODE_DESC=(
  [tp1]="vLLM/AWQ — 20K ctx, text-only, ~18 TPS (launch-bound), 1× GPU"
  [tp2]="vLLM/AWQ — 200K ctx + vision + tools, ~149 TPS, 2× GPU (GPUs 0-1)"
  [tp2-mtp]="vLLM/AWQ + MTP-3 (no prefix cache) — ~179/264/200 TPS narrative/code/explain, 2× GPU (GPUs 1-2)"
  [gguf]="llama.cpp/IQ4_XS — 128K ctx, text-only, ~117 TPS (single-GPU sweet spot), 1× GPU"
)

# CONTAINER + URL refer to whichever mode is currently up (or the most recent target).
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
  start <mode>     Bring up a variant (modes: tp1, tp2)
  stop             Stop and remove the running container
  restart <mode>   stop && start <mode>
  status           Show what's currently up
  logs             Follow container logs (Ctrl-C to detach)
  modes            List available variants

Examples:
  $0 start tp2
  $0 restart tp1
  $0 status
EOF
}

current_mode() {
  # Look across every mode's container; first running match wins. There is at
  # most one MoE up at a time on this rig (port and GPU conflicts otherwise).
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
  for m in tp1 tp2 tp2-mtp gguf; do
    printf "  %-8s %s\n" "$m" "${MODE_DESC[$m]}"
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
  local timeout="${1:-300}"
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
  wait_ready 360
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
