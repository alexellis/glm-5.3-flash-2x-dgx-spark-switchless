#!/usr/bin/env python3
"""Apply the GLM ModelOpt fused-W13 scale correction, fail closed."""

from hashlib import sha256
from pathlib import Path
import sys

SOURCE_SHA256 = "edc285a5df75b61dc935c5caeb8b69a1fd1bc487d846cc4bc65376985dd4bb21"
OUTPUT_SHA256 = "2fa021e430e7e0bbeccdabb8ec24e05bd41a57498dbbc41ea2925aae066dc6d2"

OLD = '''        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )
        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()
'''

NEW = '''        # The fused w13 kernel takes one global scale per expert, while
        # ModelOpt checkpoints carry separate global scales for w1 (gate) and
        # w3 (up). Requantize both shards onto their per-expert maximum so the
        # kernel does not incorrectly apply the gate scale to the up shard.
        gs_pair = layer.w13_weight_scale_2.to(torch.float32)
        if self.moe.is_act_and_mul and not torch.allclose(
            gs_pair[:, 0], gs_pair[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 != w3_weight_scale_2; requantizing the "
                "block scales onto a shared per-expert global scale."
            )
            gs_shared = gs_pair.max(dim=1).values
            bs = layer.w13_weight_scale
            num_experts, n2, kg = bs.shape
            num_shards = gs_pair.shape[1]
            ratio = (gs_pair / gs_shared.unsqueeze(1)).view(
                num_experts, num_shards, 1, 1
            )
            bs_f = bs.data.to(torch.float32).view(
                num_experts, num_shards, n2 // num_shards, kg
            )
            bs_f = (bs_f * ratio).view(num_experts, n2, kg)
            replace_parameter(layer, "w13_weight_scale", bs_f.to(bs.dtype))
            w13_weight_scale_2 = gs_shared.contiguous()
        else:
            w13_weight_scale_2 = gs_pair[:, 0].contiguous()
'''


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} SOURCE OUTPUT")

source, output = map(Path, sys.argv[1:])
raw = source.read_bytes()
if digest(raw) != SOURCE_SHA256:
    raise SystemExit("unexpected modelopt.py; refusing to patch")

text = raw.decode()
if text.count(OLD) != 1:
    raise SystemExit("ModelOpt patch anchor was not unique")

patched = text.replace(OLD, NEW).encode()
if digest(patched) != OUTPUT_SHA256:
    raise SystemExit("patched modelopt.py did not match the proven output")

output.write_bytes(patched)
