# GLM-5.3-Flash on 2× DGX Spark

Serve GLM-5.3-Flash as one OpenAI-compatible TP2 endpoint across two NVIDIA
DGX Sparks joined by a direct 100 GbE RoCE cable. The reference deployment
uses the LibertAIDAI ModelOpt NVFP4 checkpoint, FP8 KV, native
`flashinfer_cutlass` MoE, and the DFlash2 drafter at `k=7`.

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
| Speculation | DFlash2, 7 draft tokens |
| Context | 262,144 tokens |
| KV | FP8 E4M3, 4 GiB per rank |
| Capacity reported by vLLM | 394,488 logical tokens across the service |
| Concurrency | 6 sequences, 8,192 batched tokens |
| Multimodal | Image input verified; default vLLM limits |

That 394K figure is a safe starting floor, not the target ceiling. It is enough
for one very deep session or two sessions around 180K only narrowly. A 6 GiB
trial should yield roughly 592K logical tokens, but it must pass the supplied
long-context and concurrency gates before becoming the default.

## What is actually fast

Real output is workload-dependent because speculative acceptance is
workload-dependent. These are controlled five-run results from the reference
pair, not maxima:

| Workload | Median decode | Observed range |
|---|---:|---:|
| Code | **28.8 tok/s** | 24.0–31.0 tok/s |
| Prose | **22.9 tok/s** | 21.2–23.1 tok/s |

Cold prefill used a unique marker on every run to prevent prefix-cache reuse:

| Prompt depth | Cold TTFT | Cold effective prefill |
|---:|---:|---:|
| 8,192 tokens | 4.470s | **1,833 tok/s** |
| 32,768 tokens | 17.057s | **1,921 tok/s** |
| 65,536 tokens | 34.051s | **1,925 tok/s** |

A warm 64K prefix replay reached about 11,395 tok/s. Structured output can be
much faster than prose and must not be presented as an everyday agent speed.
The complete method and unrounded values are in
[`docs/benchmarks.md`](docs/benchmarks.md).

The evidence now includes 429 HTTP-2xx real-traffic rows carrying 43.1M prompt
tokens, contexts up to 207,940 tokens, median 25.2 tok/s across outputs of at
least 50 tokens, four client-interrupted streams, and no recorded repetition
loops. The operational cohort,
controlled five-run distributions, exact TTFT/prefill tables, Red Hat A/B, and
identical OpenCode task are all in the benchmark document. See
[`docs/topology-comparison.md`](docs/topology-comparison.md) before comparing
this lane with our TP4 ring or JSpark3's substantially different EXL3 TP3 lane.

## Install

Start with a normal direct SparkLink/RoCE pair. The fabric interface must carry
one `/24` address on each node, MTU 9000, and the peers must ping over that
interface. The exact patched NCCL library used by the reference pair has SHA256
`ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f`;
the launcher refuses a different binary. Build notes for the skip-tree-connect
NCCL 2.30.7 patch are retained in the sibling TP4 recipe.

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
