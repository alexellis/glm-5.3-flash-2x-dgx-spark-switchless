#!/usr/bin/env bash
set -euo pipefail

[[ $(id -u) -eq 0 ]] || { echo "run as root" >&2; exit 1; }
CONFIG_SOURCE="${1:?usage: sudo ./scripts/install.sh CONFIG_FILE [--head]}"
ROLE="${2:---node}"
[[ -r "$CONFIG_SOURCE" ]] || { echo "cannot read $CONFIG_SOURCE" >&2; exit 1; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_GROUP=$(id -gn "$RUN_USER")

install -d -m 0755 /opt/glm53-tp2/{scripts,patches,systemd,docs,data}
install -m 0755 "$SOURCE_DIR"/scripts/*.sh /opt/glm53-tp2/scripts/
install -m 0755 "$SOURCE_DIR"/patches/*.py /opt/glm53-tp2/patches/
install -m 0644 "$SOURCE_DIR"/systemd/*.in /opt/glm53-tp2/systemd/
install -m 0644 "$SOURCE_DIR"/docs/*.md /opt/glm53-tp2/docs/
install -m 0644 "$SOURCE_DIR"/data/*.json /opt/glm53-tp2/data/
install -m 0644 "$SOURCE_DIR/README.md" "$SOURCE_DIR/CREDITS.md" \
    "$SOURCE_DIR/LICENSE" /opt/glm53-tp2/
install -o root -g "$RUN_GROUP" -m 0640 "$CONFIG_SOURCE" /etc/glm53-tp2.env
install -d -o "$RUN_USER" -g "$RUN_GROUP" -m 0755 /var/lib/glm53-tp2

if [[ "$ROLE" == "--head" ]]; then
    sed -e "s/@RUN_USER@/$RUN_USER/g" -e "s/@RUN_GROUP@/$RUN_GROUP/g" \
        "$SOURCE_DIR/systemd/glm53-tp2.service.in" \
        >/etc/systemd/system/glm53-tp2.service
    systemctl daemon-reload
    echo "installed head service; prepare both nodes before enabling it"
elif [[ "$ROLE" == "--node" ]]; then
    echo "installed node files"
else
    echo "role must be --head or --node" >&2
    exit 2
fi
