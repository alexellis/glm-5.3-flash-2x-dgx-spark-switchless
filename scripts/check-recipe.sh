#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

for script in scripts/*.sh; do
    bash -n "$script"
done

python3 -m py_compile patches/*.py
python3 -m unittest discover -s tests -v

for receipt in data/*.json data/rigmark/*.json; do
    jq -e . "$receipt" >/dev/null
done

if grep -En \
    '192\.168\.|/home/|/tmp/|alex@|spark-[0-9a-f]{4}' \
    data/*.json data/rigmark/*.json; then
    echo 'A receipt contains a private endpoint, path, user, or hostname.' >&2
    exit 1
fi

grep -q '^HEAD_HOST=192.0.2.10$' .env.example
grep -q '^SSH_USER=your-user$' .env.example

echo 'Recipe syntax, tests, receipts, placeholders, and privacy checks passed.'
