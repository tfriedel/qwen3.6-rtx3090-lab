#!/usr/bin/env bash
# Manage the Qwen3.6-27B vLLM stack on this box.
#
#   ./qwen.sh start default   # 20 K context + vision
#   ./qwen.sh start text      # 75 K context, no vision
#   ./qwen.sh start longctx   # 125 K context (slower, eager mode)
#   ./qwen.sh start tp2       # 100 K context, 2× GPU (TPS sweet-spot)
#   ./qwen.sh start tp2-mtp   # 100 K context, 2× GPU, MTP-3 + no-prefix-cache (throughput experiment)
#   ./qwen.sh start tp4       # 500 K context, 4× GPU
#   ./qwen.sh start bf16-tp4  # 100 K context, bf16 — quality ceiling, 4× GPU
#   ./qwen.sh stop
#   ./qwen.sh status
#   ./qwen.sh logs            # follow container logs
#   ./qwen.sh restart <mode>  # stop then start
#
# Endpoint after start: http://localhost:8020/v1  (model name: qwen3.6-27b-autoround)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="${SCRIPT_DIR}/qwen36-27b-single-3090"
COMPOSE_DIR="${REPO_DIR}/compose"

if [[ ! -d "$COMPOSE_DIR" ]]; then
  echo "ERROR: compose dir not found at $COMPOSE_DIR" >&2
  echo "  Expected layout: <this-script> alongside qwen36-27b-single-3090/compose/" >&2
  exit 1
fi
CONTAINER="vllm-qwen36-27b"
URL="http://localhost:8020"

declare -A MODE_FILE=(
  [default]="docker-compose.yml"
  [text]="docker-compose.tools-text.yml"
  [longctx]="docker-compose.longctx-experimental.yml"
  [tp2]="docker-compose.tp2.yml"
  [tp2-mtp]="docker-compose.tp2-mtp.yml"
  [tp4]="docker-compose.tp4.yml"
  [tp4-2]="docker-compose.tp4-2.yml"
  [bf16-tp4]="docker-compose.bf16-tp4.yml"
)

declare -A MODE_DESC=(
  [default]="20K ctx + vision + tools (~70-90 TPS, 1× GPU)"
  [text]="75K ctx, no vision, tools (~70-90 TPS, 1× GPU)"
  [longctx]="125K ctx + vision + tools (~30-42 TPS, eager, 1× GPU)"
  [tp2]="100K ctx + vision + tools (TPS sweet-spot, 2× GPU)"
  [tp2-mtp]="100K ctx + MTP-3 + no-prefix-cache (throughput experiment, 2× GPU)"
  [tp4]="500K ctx + vision + tools (max context, 4× GPU)"
  [tp4-2]="500K ctx + vision + tools (2 concurrent sessions, 4× GPU)"
  [bf16-tp4]="100K ctx, bf16, no vision (quality ceiling, 4× GPU)"
)

usage() {
  cat <<EOF
Usage: $0 <command> [args]

Commands:
  start <mode>     Bring up a variant (modes: default, text, longctx, tp2, tp2-mtp, tp4, tp4-2, bf16-tp4)
  stop             Stop and remove the running container
  restart <mode>   stop && start <mode>
  status           Show what's currently up
  logs             Follow container logs (Ctrl-C to detach)
  modes            List available variants

Examples:
  $0 start default
  $0 restart longctx
  $0 status
EOF
}

current_mode() {
  # Print the mode key whose compose file matches the running container's labels, or "" if nothing's up.
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
  for m in default text longctx tp2 tp2-mtp tp4 tp4-2 bf16-tp4; do
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
    curl -sf "$URL/v1/models" | python3 -c '
import json, sys
m = json.load(sys.stdin)["data"][0]
print(f"  served:        {m["id"]}")
print(f"  max_model_len: {m["max_model_len"]:,}")
' 2>/dev/null || curl -sf "$URL/v1/models" | python3 -c "
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
    # Fallback: container exists but we can't tell which compose file owns it.
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
  wait_ready 300
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
