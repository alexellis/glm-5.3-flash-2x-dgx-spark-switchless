# GLM-5.3-Flash on 2× DGX Spark

Serve GLM-5.3-Flash as one OpenAI-compatible TP2 endpoint across two NVIDIA
DGX Sparks joined by a direct 200 GbE RoCE cable. The reference deployment
uses the LibertAIDAI ModelOpt NVFP4 checkpoint, FP8 KV, native
`flashinfer_cutlass` MoE, and an acceptance-guided DFlash2 drafter that selects
2, 4, or 7 verified tokens from a per-request acceptance EMA.

This is the operational recipe we use, including the parts that made it
survive a real cold boot:

- worker-first, two-rank lifecycle with all-or-nothing teardown;
- a persisted FlashInfer autotune cache and a guarded cache fast path;
- the ModelOpt fused-W13 scale correction for vLLM issue 54150;
- the corrected 4 September GLM-5.3 chat template;
- `/health`, text, tool-call, long-context, and image gates;
- systemd cleanup even when startup is interrupted; and
- a real image-understanding gate alongside the text gates.

The repository contains no model weights, usage records, SQL, credentials,
private addresses, hostnames, or private container references.

## Reference configuration

| Setting | Value |
|---|---|
| Model | `LibertAIDAI/GLM-5.3-Flash-NVFP4` |
| Model revision | `caca4e6a4ebbd66f159d3d2fc256683fd6e27177` |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` |
| Drafter revision | `bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| Tensor parallelism | TP2, one GB10 per node |
| MoE backend | `flashinfer_cutlass` |
| Speculation | DFlash2, adaptive 2/4/7-token verification |
| Context | 262,144 tokens |
| KV | FP8 E4M3, 4.5 GiB per rank |
| Capacity reported by vLLM | 442,845 logical tokens across the service |
| Concurrency | 6 sequences, 8,192 batched tokens |
| Multimodal | Image input verified; default vLLM limits |

The 443K pool supports one request at the configured 262K ceiling or two deep
sessions around 180K. The promoted setting passed a cache-cold 256,001-token
request and two simultaneous, independently salted 180,001-token requests.
Those tests computed every prompt token locally and observed no cache hits.
Five GiB is not the next default: the head's available-memory low-water mark
was 2.83 GiB during the concurrent gate.

## What is actually fast

Real output is workload-dependent because speculative acceptance is
workload-dependent. These are controlled five-run results from the reference
pair, not maxima:

| Workload | Median decode | Observed range |
|---|---:|---:|
| Completed code | **44.3 tok/s** | 42.4–45.8 tok/s |
| Completed prose | **22.6 tok/s** | 22.5–22.9 tok/s |
| Valid structured ceiling | **66.5 tok/s** | 64.5–69.1 tok/s |

Cold prefill used a unique marker on every run to prevent prefix-cache reuse:

| Prompt depth | Cold TTFT | Cold effective prefill |
|---:|---:|---:|
| 8,192 tokens | 4.434s | **1,848 tok/s** |
| 32,768 tokens | 17.181s | **1,907 tok/s** |
| 65,536 tokens | 34.234s | **1,914 tok/s** |

A warm 64K prefix replay reached about 11,382 tok/s. Structured output can be
much faster than prose and must not be presented as an everyday agent speed.
The complete method and unrounded values are in
[`docs/benchmarks.md`](docs/benchmarks.md).

### Like-for-like with Mia's fixed-k7 EXL3 baseline

On 8 September 2026, we ran Mia's current decode prompts and cold-prefill
protocol against this NVFP4 recipe. The comparison used Mia's corrected chat
template, temperature zero, thinking disabled, one decode stream, 400 output
tokens, and a unique salt for every cold prompt. Cache metrics confirmed that
every reported cold token was computed locally with zero prefix-cache hits.

| Same TP2 workload | This recipe: Libert NVFP4 | Mia EXL3 E3 | Result |
|---|---:|---:|---:|
| Prose decode | **30.94 tok/s** | 27.1 tok/s | **NVFP4 +14%** |
| Structured decode | **70.47 tok/s** | 65.1 tok/s | **NVFP4 +8%** |
| Cold prefill, ~8K | **1,830 tok/s** (4.37s) | 1,492 tok/s (5.51s) | **NVFP4 +23%** |
| Cold prefill, ~16K | **1,899 tok/s** (8.43s) | 1,554 tok/s (10.56s) | **NVFP4 +22%** |
| Cold prefill, ~256K | **1,853 tok/s** (138.18s) | 1,517 tok/s (172.84s) | **NVFP4 +22%** |
| Default context | 262,144 tokens | **850,000 tokens** | EXL3 |

**Structured decode is not an agent workload.** The prompt is literally
"Count from 1 to 200. Output only the numbers, separated by spaces." It is a
useful ceiling for speculative acceptance because the next token is unusually
predictable, but it has little practical value and must not be quoted as coding,
prose, or everyday agent speed. The hash-map prose row is the more relevant
decode comparison: this NVFP4 recipe is about 14% faster there and about 22%
faster across the matched cold-prefill depths. EXL3's material advantage is its
larger default context. Mia subsequently published opt-in 2/4/7 adaptation and
dense FP8 projection work; that is a newer configuration than the pinned
fixed-k7 baseline in this table.

See the [full method and provenance](docs/benchmarks.md#like-for-like-mia-exl3-e3-comparison)
and the [machine-readable receipt](data/mia-exl3-e3-comparison-2026-09-08.json).

The evidence now includes 429 HTTP-2xx real-traffic rows carrying 43.1M prompt
tokens, contexts up to 207,940 tokens, median 25.2 tok/s across outputs of at
least 50 tokens, four client-interrupted streams, and no recorded repetition
loops. The operational cohort,
controlled five-run distributions, exact TTFT/prefill tables, Red Hat A/B, and
identical OpenCode task are all in the benchmark document. See
[`docs/topology-comparison.md`](docs/topology-comparison.md) before comparing
this lane with our TP4 ring or JSpark3's substantially different EXL3 TP3 lane.

## Run the public benchmark

Use [`alexellis/rigmark`](https://github.com/alexellis/rigmark)
for new comparisons with TP4, another quantisation, or another model. It fixes
the code, prose, structured, prefill, and concurrency workloads; records the
appliance recipe; and refuses to compare mismatched settings by default.

Use an explicit GLM reasoning effort and reuse the same comparison ID on every
appliance in the sweep:

```bash
./rigmark run \
  --base-url http://HEAD:8000 \
  --model auto \
  --label glm53-libert-nvfp4-tp2-adaptive-low \
  --comparison-id YOUR-SWEEP-ID \
  --metadata metadata.json \
  --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}'
```

The published 4,096-token reference completed every code, prose, and
structured output. The older 512-token sweep in the benchmark document remains
useful for checkpoint A/Bs, but it did not allow the requested 700-word memo to
finish and must not be cited as completed-prose throughput.

Publish the unedited result JSON. A tok/s figure is diagnostic when its
completion gate fails; it is not evidence of completed code, prose, or valid
structured output.

## Install

Start with a normal direct SparkLink/RoCE pair. The fabric interface must carry
one `/24` address on each node, MTU 9000, and the peers must ping over that
interface. The exact patched NCCL library used by the reference pair has SHA256
`ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f`;
the launcher refuses a different binary. Build notes for the skip-tree-connect
NCCL 2.30.7 patch are retained in the sibling TP4 recipe.

The rank launcher treats these as gates, not suggestions. Before loading the
model on either node it verifies the active RoCE-v2 GID and its interface
mapping, MTU 9000, an 8,972-byte no-fragment ping, the NCCL and official chat
template checksums, and an idle GPU. A stale TP4→TP2 fabric state therefore
fails in seconds rather than hanging in EngineCore or NCCL minutes later.

Run the installer on both nodes with the same configuration, selecting
`--head` only on rank 0:

```bash
git clone https://github.com/alexellis/glm-5.3-flash-2x-dgx-spark-switchless
cd glm-5.3-flash-2x-dgx-spark-switchless
cp .env.example glm53-tp2.env
$EDITOR glm53-tp2.env
sudo ./scripts/install.sh glm53-tp2.env --node  # rank 1
sudo ./scripts/install.sh glm53-tp2.env --head  # rank 0
```

Stage the pinned model and drafter revisions on **both** nodes. `hf download`
is intentionally run by the operator so the Hugging Face token stays outside
the repository and service environment:

```bash
hf download LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  --revision caca4e6a4ebbd66f159d3d2fc256683fd6e27177 \
  --local-dir /var/tmp/models/GLM-5.3-Flash-NVFP4
hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision bf582e4eacc1810f76656d1811693ff6c6737d2a \
  --local-dir /var/tmp/models/GLM-5.3-Flash-DFlash2
```

Then run `prepare-runtime.sh` on both nodes. It generates two small runtime
corrections from the pinned image, verifies their hashes, and downloads the
pinned corrected template. Start the service from the head only:

```bash
sudo /opt/glm53-tp2/scripts/prepare-runtime.sh
sudo systemctl enable --now glm53-tp2.service
journalctl -fu glm53-tp2.service
```

The first successful start may populate the FlashInfer autotune cache. Later
boots load that cache and skip the redundant dummy autotune pass that otherwise
wedged the reference deployment indefinitely.

Run the full gate before advertising the endpoint:

```bash
/opt/glm53-tp2/scripts/gate.sh http://HEAD:8000
```

## Operational contract

- Never start the head first.
- Never restart only one rank after an EngineCore failure.
- Never treat `/v1/models` as readiness; use `/health`.
- Never retry a failed mode switch until the GID, MTU, jumbo ping, and GPU-idle
  pre-flight is green on both ranks.
- Stop the unit before switching image, checkpoint, KV size, or serve flags.
- Keep the FlashInfer and vLLM caches persistent across boots.
- A stopped or failed systemd activation must reap both containers.

See [`docs/cold-boot.md`](docs/cold-boot.md) for the failure we found and
[`docs/image-provenance.md`](docs/image-provenance.md) for the remaining image
distribution caveat.

Image requests work in the proven configuration. Video and audio are not part
of our workload; explicit `--limit-mm-per-prompt` tuning is deliberately not in
this first release because it has not yet passed a cold boot on the reference
pair.
