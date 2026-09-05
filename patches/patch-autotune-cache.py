#!/usr/bin/env python3
"""Skip redundant FlashInfer dummy autotuning after a valid cache load."""

from hashlib import sha256
from pathlib import Path
import sys

SOURCE_SHA256 = "00b239d250ae180e6cb9ca622917b197ca5a2f1919594ec27db17e67c6cc8d15"
OUTPUT_SHA256 = "7fd0e6efd423e49017bf75dfa4db8553681d450e57ded02b083f9418ec6f2605"

IMPORT_OLD = "from typing import TYPE_CHECKING\n"
IMPORT_NEW = "import os\nfrom typing import TYPE_CHECKING\n"
LOAD_OLD = '''        tuner.load_configs(str(cache_path))

    group = world.cpu_group if world.world_size > 1 else None
'''
LOAD_NEW = '''        tuner.load_configs(str(cache_path))
        if os.getenv("GLM53_TRUST_FLASHINFER_CACHE") == "1":
            logger.info_once(
                "Loaded persisted FlashInfer autotune cache; skipping "
                "the redundant dummy autotune run."
            )
            return

    group = world.cpu_group if world.world_size > 1 else None
'''


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} SOURCE OUTPUT")

source, output = map(Path, sys.argv[1:])
raw = source.read_bytes()
if digest(raw) != SOURCE_SHA256:
    raise SystemExit("unexpected kernel_warmup.py; refusing to patch")

text = raw.decode()
if text.count(IMPORT_OLD) != 1 or text.count(LOAD_OLD) != 1:
    raise SystemExit("autotune patch anchors were not unique")

patched = text.replace(IMPORT_OLD, IMPORT_NEW).replace(LOAD_OLD, LOAD_NEW).encode()
if digest(patched) != OUTPUT_SHA256:
    raise SystemExit("patched kernel_warmup.py did not match the proven output")

output.write_bytes(patched)
