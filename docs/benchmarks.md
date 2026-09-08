# Performance and operational evidence

This document separates three kinds of evidence that answer different
questions:

1. controlled direct-API runs measure the engine without OpenCode or Toilgate;
2. an identical OpenCode task measures an agent workflow end to end; and
3. anonymised Toilgate aggregates show what sustained real use felt like.

No prompt text, user identity, session identifier, private endpoint, or raw
gateway row is included.

## Current completed-output reference

The public [RigMark](https://github.com/alexellis/rigmark) sweep uses a
4,096-token ceiling, retains visible answers for audit, and rejects truncated
or reasoning-only results. Native `reasoning_effort=low` produced:

| Workload | Runs | Decode median | Run range | Gate |
|---|---:|---:|---:|---:|
| Code | 5 | **42.575 tok/s** | 38.978–44.704 | 5/5 |
| Prose | 5 | **22.162 tok/s** | 21.831–22.283 | 5/5 |
| Structured ceiling | 5 | **54.623 tok/s** | 52.956–55.722 | 5/5 |

This is the selected acceptance-guided DFlash2 policy. It begins at `k=5` and
uses a per-request EMA to drop low-acceptance output to `k=3`. The same endpoint
passed text, tool-call, 80,032-token retrieval, and image gates.

The [complete JSON receipt](../data/rigmark/adaptive-k3-k5-full.json) records a
clean RigMark `d8353e9` worktree and TP2 recipe `6163f2e`. It measured cold 64K
prefill at **1,905 tok/s**, warm 64K replay at **11,464 tok/s**, and short-code
aggregate throughput of **31.2**, **42.8**, and **61.1 tok/s** at concurrency
one, two, and four. These aggregate
figures include the whole request and use a 256-token cap; they are not prose
decode rates.

### Speculative-depth sweep

The adaptive policy was promoted only after an otherwise-identical, quiet,
five-run sweep. Every candidate completed 15/15 decode outputs:

| Policy | Code | Prose | Structured ceiling |
|---|---:|---:|---:|
| Fixed `k=3` | 38.5 | **22.4** | 43.4 |
| Fixed `k=5` | 42.5 | 20.7 | 56.1 |
| Fixed `k=7` | **44.0** | 18.6 | **66.1** |
| Adaptive `k=3/5` | 41.8 | **22.5** | 54.9 |

The adaptive candidate gained 8.7% completed-prose throughput over fixed
`k=5` while giving up 1.6% code and 2.1% structured throughput. Against the
former fixed-`k=7` default, it gained about 21% prose while giving up about 5%
code. Scheduler telemetry recorded both low and high decisions and 92 state
transitions, rather than merely accepting an unused configuration flag. The
[four receipts](../data/rigmark/) retain every output and exact setting.

## Legacy 512-token checkpoint sweep

The TP2 harness used the same two fixed workload shapes for every candidate:

- code: a production-quality Go implementation task; and
- prose: an approximately 700-word engineering memo.

Each run requested 512 output tokens at temperature zero. That was sufficient
for a controlled Libert/Red Hat checkpoint comparison, but not for the requested
roughly 700-word memo. The values below describe an early generation slice, not
completed code or prose. Decode rate is completion tokens divided by the
generation interval after first token, rather than completion tokens divided by
the entire request. Tables report the median of five runs. Prefill used unique
cold markers to defeat prefix reuse, then replayed the same prefix to expose the
warm-cache path.

The raw local harness contained response previews. They are deliberately not
published; aggregate values are sufficient for this recipe.

## Selected TP2 configuration

Libert ModelOpt NVFP4 at revision
`caca4e6a4ebbd66f159d3d2fc256683fd6e27177`, corrected fused-W13 scales,
native `flashinfer_cutlass`, FP8 KV, and fixed DFlash2 `k=7`:

| Workload | Runs | Decode median | Run range | TTFT median |
|---|---:|---:|---:|---:|
| Code | 5 | **28.812 tok/s** | 23.978–30.978 | 0.483s |
| Prose | 5 | **22.919 tok/s** | 21.161–23.112 | 0.500s |

Do not use these as the current ordinary coding or conversation headline. The
completed-output RigMark reference above supersedes them. Structured lists,
JSON, and counting are easier for the speculative drafter to predict and can be
much faster; quoting that rate as “GLM speed” overstates an agent's experience.

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

## Like-for-like Mia EXL3 E3 comparison

We repeated the public Mia EXL3 decode and cold-prefill workloads on 8 September
2026 against the selected Libert NVFP4 deployment. The upstream comparison was
pinned to Mia commit `6599585dd5ab0b5f1f68e84914f48825727ad1b3`.

| Same TP2 workload | Libert NVFP4 | Mia EXL3 E3 | Difference |
|---|---:|---:|---:|
| Prose decode, median of five | **33.164** | 27.1 | **NVFP4 +22.4%** |
| Structured decode, median of five | 58.845 | **65.1** | EXL3 +10.6% |
| Cold prefill, 8,001 tokens | **1,829.8** | 1,492.1 | **NVFP4 +22.6%** |
| Cold prefill, 16,001 tokens | **1,899.1** | 1,553.7 | **NVFP4 +22.2%** |
| Cold prefill, 256,001 tokens | **1,852.6** | 1,516.8 | **NVFP4 +22.1%** |

Decode used Mia's exact prompts, corrected chat template, temperature zero,
thinking disabled, one stream, and a 400-token ceiling. The NVFP4 prose range
was 31.349–38.064 tok/s with median TTFT 0.317s. Its structured range was
57.969–59.286 tok/s with median TTFT 0.303s and 99.4% median draft acceptance.

The structured prompt is exactly:

> Count from 1 to 200. Output only the numbers, separated by spaces. No other
> text.

This is a deliberately predictable speculative-decode ceiling, not useful
work. It says how quickly a drafter can continue an obvious sequence. It does
not measure code generation, tool use, reasoning, prose, or an agent changing a
repository. Do not present it as the model's practical speed.

Cold prefill used Mia's filler, calibration, streaming TTFT calculation, and a
fresh random salt per request. vLLM metrics recorded zero cache-hit tokens and
the full prompt-token count as local compute for every included rung. The
300,001-token rung was correctly rejected because this profile serves a
262,144-token context, so it is retained in the receipt but excluded from the
comparison table.

The production server deliberately refuses untrusted per-request chat
templates. For the comparison, the pinned Mia template was rendered locally
with `enable_thinking=false`, and its resulting prompt was submitted through
the completions endpoint. This supplies the same model input without weakening
the running server or changing any model-side setting. The only intended
differences are the recipes under comparison: Mia uses EXL3/TR3 4 bpw, FP8 KV,
and fixed DFlash2 `k=7`; this deployment uses Libert ModelOpt NVFP4, FP8 KV,
and adaptive DFlash2 `k=3/5`.

The machine-readable receipt records the benchmark checkout, dirty state,
adapter and template hashes, deployed runtime hashes, unrounded observations,
and upstream source values:
[`data/mia-exl3-e3-comparison-2026-09-08.json`](../data/mia-exl3-e3-comparison-2026-09-08.json).

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
and list fragments draft particularly well. This mixed operational median is
higher than completed long prose for the same reason. For a sustained natural
memo, use the 18–19 tok/s completed-output reference above.

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
