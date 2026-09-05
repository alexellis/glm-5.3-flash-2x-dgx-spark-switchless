# Reference measurements

Measurements below are controlled aggregates. No prompts, user identifiers,
gateway records, SQL, or private endpoint data are included.

## Current Libert NVFP4 + native CUTLASS

Five runs per decode workload:

| Workload | Median |
|---|---:|
| Code decode | 28.812 tok/s |
| Prose decode | 22.919 tok/s |
| Cold prefill, 8K | 1,832.7 tok/s |
| Cold prefill, 32K | 1,921.1 tok/s |
| Cold prefill, 64K | 1,924.6 tok/s |
| Warm replay, 64K | 11,395.4 tok/s |

## Red Hat W4A4 control

The same harness against Red Hat compressed-tensors NVFP4 with Marlin:

| Workload | Median |
|---|---:|
| Code decode | 27.492 tok/s |
| Prose decode | 21.914 tok/s |
| Cold prefill, 8K | 1,690.5 tok/s |
| Cold prefill, 32K | 1,736.5 tok/s |
| Cold prefill, 64K | 1,741.2 tok/s |
| Warm replay, 64K | 10,486.8 tok/s |

On this pair, corrected Libert was about 4.6–4.8% faster at decode and about
10.5% faster at cold 32–64K prefill. The identical OpenCode task completed in
353 seconds over 14 requests and 411,676 total tokens, versus 423 seconds on
the Red Hat control.

These are end-to-end request rates, not pure decode-loop telemetry. Draft
acceptance makes structured output substantially faster than prose. Always
publish prompt type, prompt length, cache state, thinking mode, output length,
and concurrency alongside a tokens/second result.
