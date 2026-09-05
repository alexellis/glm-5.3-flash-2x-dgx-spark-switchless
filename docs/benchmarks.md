# Performance and operational evidence

This document separates three kinds of evidence that answer different
questions:

1. controlled direct-API runs measure the engine without OpenCode or Toilgate;
2. an identical OpenCode task measures an agent workflow end to end; and
3. anonymised Toilgate aggregates show what sustained real use felt like.

No prompt text, user identity, session identifier, private endpoint, or raw
gateway row is included.

## How the controlled sweep was measured

The TP2 harness used the same two fixed workload shapes for every candidate:

- code: a production-quality Go implementation task; and
- prose: an approximately 700-word engineering memo.

Each run requested 512 output tokens at temperature zero. Decode rate is
completion tokens divided by the generation interval after first token, rather
than completion tokens divided by the entire request. Tables report the median
of five runs. Prefill used unique cold markers to defeat prefix reuse, then
replayed the same prefix to expose the warm-cache path.

The raw local harness contained response previews. They are deliberately not
published; aggregate values are sufficient for this recipe.

## Selected TP2 configuration

Libert ModelOpt NVFP4 at revision
`caca4e6a4ebbd66f159d3d2fc256683fd6e27177`, corrected fused-W13 scales,
native `flashinfer_cutlass`, FP8 KV, and DFlash2 `k=7`:

| Workload | Runs | Decode median | Run range | TTFT median |
|---|---:|---:|---:|---:|
| Code | 5 | **28.812 tok/s** | 23.978–30.978 | 0.483s |
| Prose | 5 | **22.919 tok/s** | 21.161–23.112 | 0.500s |

These are the numbers to use for ordinary coding and conversation. Structured
lists, JSON, and counting are easier for the speculative drafter to predict and
can be much faster; quoting that rate as “GLM speed” overstates an agent's
experience.

### Cold and warm prefill

| Prompt depth | Cold TTFT | Cold effective prefill | Warm TTFT | Warm replay |
|---:|---:|---:|---:|---:|
| 8,192 | 4.470s | **1,832.7 tok/s** | 4.514s | 1,815.0 tok/s |
| 32,768 | 17.057s | **1,921.1 tok/s** | 2.977s | **11,009.1 tok/s** |
| 65,536 | 34.051s | **1,924.6 tok/s** | 5.751s | **11,395.4 tok/s** |

The 8K replay not improving is a useful warning: fixed request and scheduling
overheads can dominate shallow prompts. Prefix reuse becomes decisive at 32K
and 64K. For an agent repeatedly resending a large code context, warm prefill
matters more to responsiveness than a structured-output decode headline.

## Red Hat W4A4 control

The same harness was run against Red Hat compressed-tensors NVFP4 with Marlin:

| Workload | Red Hat | Selected Libert | Libert difference |
|---|---:|---:|---:|
| Code decode | 27.492 | **28.812** | +4.8% |
| Prose decode | 21.914 | **22.919** | +4.6% |
| Cold prefill, 8K | 1,690.5 | **1,832.7** | +8.4% |
| Cold prefill, 32K | 1,736.5 | **1,921.1** | +10.6% |
| Cold prefill, 64K | 1,741.2 | **1,924.6** | +10.5% |
| Warm replay, 32K | 10,228.3 | **11,009.1** | +7.6% |
| Warm replay, 64K | 10,486.8 | **11,395.4** | +8.7% |

The selected checkpoint was not chosen from a counting prompt. It won both
realistic decode shapes, every cold-prefill depth, both meaningful warm
replays, and the end-to-end agent task after applying the correctness patch.

## Identical OpenCode task

The same repository, context, question, and OpenCode workflow were used before
and after the checkpoint change:

| Configuration | Wall time | Requests | Total model tokens | Result |
|---|---:|---:|---:|---|
| Red Hat W4A4 + Marlin | 423s | not retained | not retained | correct |
| Libert + native CUTLASS | **353s** | 14 | 411,676 | correct |

This is a product-level comparison rather than an isolated kernel comparison.
It includes prompt ingestion, tool turns, prefix reuse, reasoning, and decode.
The selected lane completed in about 16.5% less wall time.

## Real operational cohort

Toilgate records only the served model ID, not the checkpoint, topology, or
launcher revision. The following window was mapped from the operational change
log to periods when this TP2 configuration was serving. Treat that mapping as
an annotation, not something the database independently proves.

| Metric | Aggregate |
|---|---:|
| HTTP 2xx rows | **429 / 429** |
| Rows reporting decode rate | 420 |
| Rows reporting TTFT | 428 |
| Prompt tokens | **43,122,786** |
| Completion tokens | **344,008** |
| Prompt-to-completion ratio | **125.4:1** |
| Median completion | 265 tokens |
| Largest completion | 11,916 tokens |
| Deepest context | **207,940 tokens** |
| Sum of request wall times | 5.01 hours |
| Client-interrupted streams | 4 |
| Degenerate loops recorded | **0** |

For outputs long enough to make the rate meaningful:

| Minimum output | Samples | Decode median | Decode mean | Observed range |
|---:|---:|---:|---:|---:|
| 50 tokens | 390 | **25.21 tok/s** | 27.68 | 4.16–63.51 |
| 150 tokens | 284 | **23.54 tok/s** | 25.96 | 4.16–63.51 |

The mean is higher than the median because predictable tool, code-structure,
and list fragments draft particularly well. The controlled prose median and
the ≥150-token operational median agree closely: expect roughly 23 tok/s for
long natural output, with code and structured turns often faster.

### Behaviour by prompt depth

These cohorts are descriptive, not causal: workload mix and draft acceptance
also change between bands.

| Prompt band | Requests | Average prompt | Maximum | Median TTFT | Decode median, outputs ≥50 |
|---|---:|---:|---:|---:|---:|
| under 8K | 13 | 3,929 | 7,911 | 3.95s | 32.9 tok/s |
| 8–32K | 55 | 18,805 | 31,752 | 5.27s | 21.8 tok/s |
| 32–64K | 59 | 48,444 | 62,851 | 5.45s | 22.5 tok/s |
| 64–128K | 137 | 100,615 | 127,957 | 5.09s | 24.6 tok/s |
| 128K+ | 164 | 157,732 | 207,343 | 5.06s | 27.0 tok/s |

Toilgate did not receive per-request `cached_tokens` from this backend, so the
operational rows cannot produce an honest cold-prefill rate. Five-second median
TTFT at 128K+ strongly suggests substantial prefix reuse, but it does not tell
us how many tokens were reused. Use the controlled unique-marker sweep for
cold prefill and the replay sweep for cache speed.

## Cross-topology context

The fuller TP2, our TP4, and external TP3 comparison is in
[`topology-comparison.md`](topology-comparison.md). The short version is:

- our TP4 was about 1.77× our TP2 on controlled prose decode;
- our TP4 was about 1.19× our TP2 on cold 32K prefill; and
- Jake Harris' 66.3 tok/s TP3 code figure used an EXL3 lane and a different
  harness. Its advertised 44.6 tok/s TP2 baseline was an adapted Mia EXL3
  recipe, not this NVFP4 TP2 deployment.

## Reproduction rules

When publishing another result, include all of these:

- checkpoint and immutable revision;
- image digest and runtime patches;
- TP degree, fabric, MoE backend, KV dtype, and KV allocation;
- prompt type and prompt-token count;
- output-token count and whether reasoning is included;
- thinking effort, temperature, and draft depth;
- cold or warm prefix state;
- concurrency; and
- whether rate is provider telemetry, first-to-last-token, or whole-request
  wall time.
