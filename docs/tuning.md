# Tuning boundaries

The checked-in values favour a reliable interactive and coding lane.

## KV capacity

At 4 GiB per rank with the selected adaptive `k=3/5` policy, vLLM reported
427,708 logical tokens for the service. The former fixed-`k=7` configuration
reported 394,488. The
pool is service-wide; do not add the two rank log values together.

Linear capacity estimates are useful only for choosing the next experiment:

| KV per rank | Estimated logical pool | Status |
|---|---:|---|
| 4 GiB | 428K | cold-boot proven |
| 6 GiB | 642K | next measured trial |
| 8 GiB | 855K | memory-risk trial |

The 8 GiB option is not a free upgrade. Model weights, drafter, vision tower,
compile workspace, graphs, activations, and host page cache all share the GB10
unified memory. A short reply is not an adequate gate for a larger pin.

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
