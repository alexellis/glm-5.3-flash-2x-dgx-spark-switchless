#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

MODE="${1:-run}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
READY_TIMEOUT=2400
LOGFILE="$RUNTIME_DIR/service.log"
LOCKFILE="$RUNTIME_DIR/pair.lock"

mkdir -p "$RUNTIME_DIR"

log() {
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOGFILE"
}

remote() {
    # Arguments are deliberately interpreted by the remote shell.
    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$WORKER_HOST" "$@"
}

head_running() {
    docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null |
        grep -qx true
}

worker_running() {
    remote "docker inspect -f '{{.State.Running}}' '$CONTAINER_NAME' 2>/dev/null" |
        grep -qx true
}

healthy() {
    curl -fsS --max-time 15 -o /dev/null "$HEALTH_URL"
}

stop_pair() {
    log "stopping worker rank, then head rank"
    remote "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1 || true" || true
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

memory_hygiene() {
    sync
    printf '3\n' | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    printf '1\n' | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
    remote "sync; printf '3\\n' | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; printf '1\\n' | sudo -n tee /proc/sys/vm/compact_memory >/dev/null"
}

wait_for_worker() {
    local attempt
    for attempt in $(seq 1 120); do
        if ping -c 1 -W 1 "$WORKER_HOST" >/dev/null 2>&1 &&
            ping -I "$FABRIC_IFACE" -c 1 -W 1 "$WORKER_FABRIC_IP" >/dev/null 2>&1 &&
            remote true >/dev/null 2>&1; then
            return 0
        fi
        if (( attempt % 12 == 0 )); then
            log "waiting for worker (${attempt}/120)"
        fi
        sleep 5
    done
    log "worker management or fabric path did not become reachable"
    return 1
}

start_pair() {
    wait_for_worker
    if healthy && head_running && worker_running; then
        log "both ranks and /health are green"
        return 0
    fi

    stop_pair
    memory_hygiene

    log "launching worker rank 1"
    remote "/opt/glm53-tp2/scripts/rank-launcher.sh 1"
    log "launching head rank 0"
    /opt/glm53-tp2/scripts/rank-launcher.sh 0

    local waited=0
    log "waiting up to ${READY_TIMEOUT}s for /health"
    until healthy; do
        if ! head_running || ! worker_running; then
            log "a rank exited during startup"
            return 1
        fi
        sleep 15
        waited=$((waited + 15))
        if (( waited >= READY_TIMEOUT )); then
            log "startup exceeded ${READY_TIMEOUT}s"
            return 1
        fi
    done
    log "GLM TP2 is healthy after ${waited}s"
}

run_service() {
    trap 'stop_pair' EXIT
    trap 'exit 0' TERM INT
    start_pair

    local failures=0
    while true; do
        sleep 30
        if healthy && head_running && worker_running; then
            failures=0
        else
            failures=$((failures + 1))
            log "health failure ${failures}/3"
            if (( failures >= 3 )); then
                log "leaving systemd to restart the complete pair"
                return 1
            fi
        fi
    done
}

exec 9>"$LOCKFILE"
flock -w 30 9 || { log "another pair operation holds $LOCKFILE"; exit 1; }

case "$MODE" in
    run) run_service ;;
    start) start_pair ;;
    stop) stop_pair ;;
    status) head_running && worker_running && healthy ;;
    *) echo "usage: $0 {run|start|stop|status}" >&2; exit 2 ;;
esac
