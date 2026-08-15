#!/usr/bin/env bash
# Manage the Qwen3.8-27B vLLM stack on this box.
#
# Qwen3.8-27B (2026-08-13) is architecturally identical to Qwen3.6-27B
# (model_type qwen3_5, 64-layer 3:1 GatedDeltaNet hybrid, vision, 1-layer MTP
# head), so these modes are the proven 3.6 tp2 recipes with the weights
# swapped for philbert440/Qwen3.8-27B-W4A16-AWQ (INT4 g128, vision + MTP
# intact). Runs alongside the 3.6 stacks — but shares GPUs 0,1 with
# qwen.sh tp2 modes, so don't run both at once without re-pinning.
#
#   ./qwen38.sh start tp2       # 100 K ctx, 2× GPU, MTP-3 + prefix cache
#   ./qwen38.sh start tp2-mtp   # 100 K ctx, 2× GPU, MTP-4 + no-prefix-cache
#   ./qwen38.sh stop
#   ./qwen38.sh status
#   ./qwen38.sh logs
#   ./qwen38.sh restart <mode>
#
# Endpoint after start: http://localhost:8026/v1  (model name: qwen3.8-27b-awq)
#
# Bench:
#   URL=http://localhost:8026 MODEL=qwen3.8-27b-awq CONTAINER=vllm-qwen38-27b \
#     ./qwen36-27b-single-3090/scripts/bench.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="${SCRIPT_DIR}/qwen36-27b-single-3090"
COMPOSE_DIR="${REPO_DIR}/compose"

if [[ ! -d "$COMPOSE_DIR" ]]; then
  echo "ERROR: compose dir not found at $COMPOSE_DIR" >&2
  exit 1
fi
CONTAINER="vllm-qwen38-27b"
URL="http://localhost:8026"

declare -A MODE_FILE=(
  [tp2]="docker-compose.qwen38-tp2.yml"
  [tp2-mtp]="docker-compose.qwen38-tp2-mtp.yml"
)

declare -A MODE_DESC=(
  [tp2]="100K ctx + vision + tools + MTP-3 + prefix cache (2× GPU) — recommended"
  [tp2-mtp]="100K ctx + MTP-3 + no-prefix-cache (A/B experiment — no gain over tp2 on 3.8)"
)

usage() {
  cat <<EOF
Usage: $0 <command> [args]

Commands:
  start <mode>     Bring up a variant (modes: tp2, tp2-mtp)
  stop             Stop and remove the running container
  restart <mode>   stop && start <mode>
  status           Show what's currently up
  logs             Follow container logs (Ctrl-C to detach)
  modes            List available variants
EOF
}

current_mode() {
  if ! docker ps -q --filter "name=^${CONTAINER}$" | grep -q .; then
    return
  fi
  local cf
  cf="$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}' 2>/dev/null || true)"
  for m in "${!MODE_FILE[@]}"; do
    [[ "$cf" == *"${MODE_FILE[$m]}"* ]] && { echo "$m"; return; }
  done
  echo "unknown"
}

cmd_modes() {
  for m in tp2 tp2-mtp; do
    printf "  %-9s %s\n" "$m" "${MODE_DESC[$m]}"
  done
}

cmd_status() {
  local cur
  cur="$(current_mode)"
  if [[ -z "$cur" ]]; then
    echo "stopped"
    return
  fi
  echo "running mode: $cur (${MODE_DESC[$cur]:-})"
  docker ps --filter "name=^${CONTAINER}$" --format "  {{.Status}} | {{.Ports}}"
  if curl -sf -m 3 "$URL/v1/models" >/dev/null 2>&1; then
    curl -sf "$URL/v1/models" | python3 -c "
import json, sys
m = json.load(sys.stdin)['data'][0]
print('  served:       ', m['id'])
print('  max_model_len:', f\"{m['max_model_len']:,}\")
"
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
  local f="${MODE_FILE[$cur]:-}"
  echo "stopping $cur..."
  if [[ -n "$f" ]]; then
    (cd "$COMPOSE_DIR" && docker compose -f "$f" down)
  else
    docker rm -f "$CONTAINER" >/dev/null
  fi
}

wait_ready() {
  local timeout="${1:-600}"
  local t=0
  printf "waiting for %s/v1/models" "$URL"
  until curl -sf -m 2 "$URL/v1/models" >/dev/null 2>&1; do
    if (( t >= timeout )); then
      echo " — timeout after ${timeout}s"
      echo "Last 30 log lines:"
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
  if [[ ! -d "${REPO_DIR}/models/qwen3.8-27b-awq-w4a16" ]]; then
    echo "ERROR: model dir not found: ${REPO_DIR}/models/qwen3.8-27b-awq-w4a16" >&2
    echo "  Download with: hf download philbert440/Qwen3.8-27B-W4A16-AWQ --local-dir ${REPO_DIR}/models/qwen3.8-27b-awq-w4a16" >&2
    return 1
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
  echo "starting $mode (${MODE_DESC[$mode]})..."
  (cd "$COMPOSE_DIR" && docker compose -f "${MODE_FILE[$mode]}" up -d)
  wait_ready 600
  cmd_status
}

cmd_logs() {
  if [[ -z "$(current_mode)" ]]; then
    echo "container not running"
    return 1
  fi
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
