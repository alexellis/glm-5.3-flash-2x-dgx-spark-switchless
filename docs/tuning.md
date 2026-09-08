# Tuning boundaries

The checked-in values favour a reliable interactive and coding lane.

## KV capacity

At 4.5 GiB per rank with the selected adaptive `k=2/4/7` policy, vLLM reports
442,845 logical tokens for the service. The same seven-token engine reported
394,488 at 4 GiB; the previous five-token engine used by the adaptive `k=3/5`
policy reported 427,708 at 4 GiB. The pool is service-wide; do not add the two
rank log values together.

Linear capacity estimates are useful only for choosing the next experiment:

| KV per rank | Estimated logical pool | Status |
|---|---:|---|
| 4 GiB | 394,488 | previous cold-boot baseline |
| **4.5 GiB** | **442,845** | **current; deep-load proven** |
| 5 GiB | about 491K | not promoted; insufficient head margin |
| 6 GiB | about 588K | unsafe without reclaiming other allocations |

Model weights, drafter, vision tower, compile workspace, graphs, activations,
and host page cache all share the GB10 unified memory. During the two-by-180K
gate, available memory fell to 2.83 GiB on the head. A short reply is not an
adequate gate for a larger pin, and 5 GiB should not be inferred to be safe
from the linear estimate.

## Prefill and scheduling

`--max-num-batched-tokens 8192` is retained because the reference tests found a
prefill win without a single-stream decode regression. Mixed prefill and decode
still need realistic concurrent-agent tests: large incoming prefills can reduce
an active stream's generation substantially.

## Multimodal

The current cold-boot-proven command leaves vLLM's multimodal limits at their
image defaults. Image input works. Video is not useful to the reference
workload, but disabling its profiling is a separate change and must be measured
before becoming part of this stable recipe. Audio is not exposed by this GLM
serving path.
