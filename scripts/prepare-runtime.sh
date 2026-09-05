#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

[[ $(id -u) -eq 0 ]] || { echo "run as root" >&2; exit 1; }
install -d -m 0755 "$RUNTIME_DIR/patches" "$RUNTIME_DIR/templates"

docker pull "$IMAGE"
extract_name="glm53-patch-source-$$"
docker create --name "$extract_name" "$IMAGE" >/dev/null
cleanup() { docker rm "$extract_name" >/dev/null 2>&1 || true; }
trap cleanup EXIT

modelopt=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py
warmup=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py
docker cp "$extract_name:$modelopt" "$RUNTIME_DIR/modelopt.py.orig"
docker cp "$extract_name:$warmup" "$RUNTIME_DIR/kernel_warmup.py.orig"

python3 /opt/glm53-tp2/patches/patch-modelopt-w13.py \
    "$RUNTIME_DIR/modelopt.py.orig" "$RUNTIME_DIR/patches/modelopt.py"
python3 /opt/glm53-tp2/patches/patch-autotune-cache.py \
    "$RUNTIME_DIR/kernel_warmup.py.orig" \
    "$RUNTIME_DIR/patches/kernel_warmup.py"

template_url=https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/690b705278a3a58e538fcb37c2ca8b5f9511213c/chat_template.jinja
curl -fsSL "$template_url" -o "$RUNTIME_DIR/templates/chat_template.jinja.new"
echo '0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5  '"$RUNTIME_DIR/templates/chat_template.jinja.new" | sha256sum -c -
mv "$RUNTIME_DIR/templates/chat_template.jinja.new" \
    "$RUNTIME_DIR/templates/chat_template.jinja"

chmod 0644 "$RUNTIME_DIR"/patches/*.py "$RUNTIME_DIR"/templates/*.jinja
echo "prepared verified runtime files in $RUNTIME_DIR"
