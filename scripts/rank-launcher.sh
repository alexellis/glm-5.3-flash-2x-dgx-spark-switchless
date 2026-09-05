#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

NODE_RANK="${1:?usage: rank-launcher.sh 0|1}"
case "$NODE_RANK" in
    0)
        HOST_IP=$HEAD_FABRIC_IP
        PEER_IP=$WORKER_FABRIC_IP
        HEADLESS=()
        ;;
    1)
        HOST_IP=$WORKER_FABRIC_IP
        PEER_IP=$HEAD_FABRIC_IP
        HEADLESS=(--headless)
        ;;
    *)
        echo "rank must be 0 or 1" >&2
        exit 2
        ;;
esac

MODEL_PATH=/models/glm-5.3-flash-nvfp4
DRAFT_PATH=/models/dflash2-draft
PATCH_PATH=/usr/local/lib/python3.12/dist-packages/vllm
CUDA_LIB_HOST_PATH=/usr/local/cuda/targets/sbsa-linux/lib

test -f "$MODEL_HOST_PATH/config.json"
test -f "$DRAFT_HOST_PATH/model.safetensors"
test -f "$NCCL_HOST_PATH/libnccl.so.2"
echo "$NCCL_SHA256  $NCCL_HOST_PATH/libnccl.so.2" | sha256sum -c -
test -f "$RUNTIME_DIR/patches/modelopt.py"
test -f "$RUNTIME_DIR/patches/kernel_warmup.py"
test -f "$RUNTIME_DIR/templates/chat_template.jinja"
ip -4 address show dev "$FABRIC_IFACE" | grep -Fq "$HOST_IP/24"
ping -I "$FABRIC_IFACE" -c 1 -W 2 "$PEER_IP" >/dev/null

mkdir -p "$RUNTIME_DIR/cache/huggingface" \
    "$RUNTIME_DIR/cache/flashinfer" "$RUNTIME_DIR/cache/vllm"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --gpus all -d \
    --name "$CONTAINER_NAME" --restart no \
    --network host --ipc host --shm-size 32g \
    --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
    --device /dev/infiniband:/dev/infiniband \
    -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
    -v "$DRAFT_HOST_PATH:$DRAFT_PATH:ro" \
    -v "$NCCL_HOST_PATH:/opt/patched-nccl:ro" \
    -v "$RUNTIME_DIR/cache/huggingface:/cache/huggingface" \
    -v "$RUNTIME_DIR/cache/flashinfer:/root/.cache/flashinfer" \
    -v "$RUNTIME_DIR/cache/vllm:/root/.cache/vllm" \
    -v "$RUNTIME_DIR/patches/modelopt.py:$PATCH_PATH/model_executor/layers/quantization/modelopt.py:ro" \
    -v "$RUNTIME_DIR/patches/kernel_warmup.py:$PATCH_PATH/model_executor/warmup/kernel_warmup.py:ro" \
    -v "$RUNTIME_DIR/templates/chat_template.jinja:/models/chat_template.jinja:ro" \
    -v /usr/local/cuda/include/nvrtc.h:/usr/local/cuda/include/nvrtc.h:ro \
    -v "$CUDA_LIB_HOST_PATH:/opt/host-cuda-lib:ro" \
    -e LD_PRELOAD=/opt/patched-nccl/libnccl.so.2 \
    -e VLLM_NCCL_SO_PATH=/opt/patched-nccl/libnccl.so.2 \
    -e VLLM_HOST_IP="$HOST_IP" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -e MAX_JOBS=1 \
    -e LIBRARY_PATH=/opt/host-cuda-lib \
    -e LD_LIBRARY_PATH=/opt/host-cuda-lib \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e TORCH_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e GLM53_TRUST_FLASHINFER_CACHE=1 \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
    -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_ALGO=Ring \
    -e NCCL_IB_HCA="$FABRIC_HCA" -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
    -e NCCL_IB_ADDR_RANGE="$FABRIC_CIDR" \
    -e NCCL_SOCKET_IFNAME="$FABRIC_IFACE" \
    -e GLOO_SOCKET_IFNAME="$FABRIC_IFACE" \
    -e TP_SOCKET_IFNAME="$FABRIC_IFACE" -e MN_IF_NAME="$FABRIC_IFACE" \
    -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 \
    -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 6 --block-size 2304 \
    --moe-backend flashinfer_cutlass \
    --max-num-batched-tokens 8192 \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_PATH\",\"num_speculative_tokens\":7}" \
    --kv-cache-dtype fp8_e4m3 --kv-cache-memory "$KV_CACHE_BYTES" \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --chat-template /models/chat_template.jinja \
    --reasoning-parser deepseek_r1 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_FABRIC_IP" --master-port "$MASTER_PORT" \
    "${HEADLESS[@]}"

sleep 2
docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" | grep -qx true
