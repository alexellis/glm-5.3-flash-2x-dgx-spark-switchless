#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${GLM53_CONFIG:-/etc/glm53-tp2.env}"
[[ -r "$CONFIG_FILE" ]] || {
    echo "configuration is not readable: $CONFIG_FILE" >&2
    exit 1
}
# The configuration is installed root-owned and contains shell assignments.
# shellcheck disable=SC1090
source "$CONFIG_FILE"

GID_INDEX="${GID_INDEX:-3}"

required=(
    HEAD_HOST WORKER_HOST SSH_USER FABRIC_IFACE FABRIC_HCA FABRIC_CIDR
    HEAD_FABRIC_IP WORKER_FABRIC_IP PORT MASTER_PORT CONTAINER_NAME IMAGE
    MODEL_HOST_PATH DRAFT_HOST_PATH NCCL_HOST_PATH NCCL_SHA256 RUNTIME_DIR
    KV_CACHE_BYTES MAX_MODEL_LEN GPU_MEMORY_UTILIZATION
    SPEC_TOKENS ADAPTIVE_K
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || {
        echo "missing $name in $CONFIG_FILE" >&2
        exit 1
    }
done

[[ "$SPEC_TOKENS" == "3" || "$SPEC_TOKENS" == "5" || "$SPEC_TOKENS" == "7" ]] || {
    echo "SPEC_TOKENS must be 3, 5, or 7" >&2
    exit 1
}
[[ "$ADAPTIVE_K" == "0" || "$ADAPTIVE_K" == "1" ]] || {
    echo "ADAPTIVE_K must be 0 or 1" >&2
    exit 1
}
if [[ "$ADAPTIVE_K" == "1" && "$SPEC_TOKENS" != "5" ]]; then
    echo "ADAPTIVE_K=1 requires SPEC_TOKENS=5" >&2
    exit 1
fi

# Consumed by scripts which source this library.
# shellcheck disable=SC2034
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
