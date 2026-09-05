# TP2, TP3, and TP4 comparison

This table restores the comparison made during bring-up without pretending the
columns are a single matched benchmark. “Our” rows were measured on our fleet.
The TP3 row is author-reported from JSpark3. Quantisation, thinking mode,
context window, and harness differ.

| Measurement | Our TP2 | Our TP4 | JSpark3 TP3 |
|---|---:|---:|---:|
| Checkpoint format | ModelOpt NVFP4 | Red Hat W4A4 NVFP4 | EXL3/TR3 4 bpw |
| Speculation | DFlash2 k=7 | DFlash2 k=7 | DFlash2 |
| Configured context | 262K | 262K | 1M |
| Code decode | **28.8** | not measured | **66.3** |
| Prose decode | **22.9** | **37.6–40.6** | **29.0** |
| Structured decode | not measured | not measured | **82.0** |
| Cold prefill, 8K | **1,833** | **1,881–2,233** | not published at 8K |
| Cold prefill, 32K | **1,921** | **2,278–2,288** | not published at 32K |
| Cold prefill, 64K | **1,925** | **2,263–2,273** | not published at 64K |
| Cold prefill, 114K | not measured | ~2,240 at 128K | **1,234** |
| Warm replay, 32K | **11,009** | **15,150–15,430** | not comparable |
| Code-like TTFT | **0.483s** at a shallow prompt | not measured | **0.391s** clamp-code |
| Four-stream aggregate | not measured | not measured under the controlled harness | **251** |

Rates are tokens/second. A range in our TP4 column is the baseline/restored
pair of sweeps, not run-to-run cherry-picking.

## What our TP4 result says

Using each topology's retained controlled harness:

- TP4 prose was 1.64–1.77× TP2 prose;
- TP4 cold 32K prefill was 1.19× TP2; and
- TP4 warm 32K replay was 1.38–1.40× TP2.

The decode gain is much larger than the cold-prefill gain. That is consistent
with GLM benefiting from four-way weight and expert distribution while prompt
ingestion remains constrained by work that does not scale linearly with TP.

There is an important methodology caveat: the TP2 sweep used
`glm_real_sweep.py`; the TP4 sweep used `glm_fabric_sweep.py`. Their prompt
shapes, cold markers, and output handling differ. These ratios are directional.
A publication-grade topology claim needs both deployments rerun with one frozen
harness in a quiet window.

## What the JSpark3 result does and does not say

JSpark3 publicly reports 66.257 tok/s code, 29.049 prose, 81.962 structured,
251 tok/s four-stream aggregate, and 1,234.246 tok/s on a 113,908-token prefill
proxy. Its same-agent-task result was 44.583 tok/s.

Those are substantial results, but they are not “add a third Spark to this
recipe and get 66 tok/s”. JSpark3 changes several variables at once:

- EXL3/TR3 weights instead of our ModelOpt NVFP4;
- an INT8 trunk overlay;
- TP3 plus expert parallelism;
- a 1M serving envelope;
- thinking off in its frozen screen; and
- its own estimator and prompt battery.

Most importantly, JSpark3's headline “66.3 versus 44.6 on two Sparks” compares
against its compatibility-adapted Mia EXL3 TP2 baseline. Our measured NVFP4 TP2
code rate is 28.8 tok/s. The 44.6 number was never a measurement of this
repository.

Jake's own report is unusually candid about the trade-offs: its matched 114K
prefill proxy declined by 3.38%, one C3 pairing was variable, and two internal
promotion thresholds were missed. That makes the data useful, but it still
describes a different serving product.

Source: [JSpark3 repository and benchmark summary](https://github.com/jakejharris/jspark3).

## Practical conclusion

- Choose this TP2 lane when two Sparks, a direct cable, simple topology, and a
  separate second pair matter more than maximum GLM speed.
- Choose our TP4 ring when GLM is the sole capability lane: its controlled prose
  and prefill results are materially stronger.
- Treat TP3 as a distinct EXL3 appliance recipe, not a halfway scaling point for
  our NVFP4 deployment.
