# 16-ID FP32 batch curve — stopped before saturation

Date: 2026-09-26  
Status: FP32 measurement was stopped by the user before a CUDA out-of-memory boundary was observed.

## Reading the graphs

The practical peak is user-decreed as **9,600 servings/s** for the RTX 5070 Ti. For a reproducible first-plateau point, this report marks the first completed median within 1% of 9,600: **batch 8,192**, using **5.58 GiB** device-wide peak memory. The highest completed raw median in this series was **9,793.5 servings/s** at batch 45,056; it is shown as a measured point while 9,600 remains the declared practical peak.

At batch 49,152, throughput fell to about 5,134 servings/s and at 57,344 to about 846/s, while exact first-16 sequence agreement remained 100%. The last complete row used 15.69 GiB device-wide memory. A 65,536 attempt produced two console-only repetitions near 3,791 servings/s at about 15.79 GiB, then was interrupted. It is marked as partial and excluded from completed-row peak calculations. Since no CUDA OOM was observed, this is **not a measured saturation point**.

## Metrics

- A serving is one bounded response ending on model EOS or exactly 16 generated entity IDs; no synthetic EOS is added.
- Tokens/s means returned generated entity-ID tokens/s; EOS is excluded. The measured token count is reported directly, not estimated from the serving rate.
- Servings/W and tokens/W divide the respective median rate by the median NVML device board power.
- The first-16 agreement column is exact ordered-prefix parity against the archived serial FP32 reference. The higher-batch runner also recorded protocol/cap validity and label-based macro F1 on a diagnostic eight-query validation slice; these do not control the stop rule.
- Memory and power are device-wide NVML readings under Windows WDDM. Desktop GPU processes remained resident.

## Data and evidence

- Merged rates, watts, efficiency, memory, parity and partial status: [`fp32-16id-vram-series.csv`](../../runs/learning/16id-serving-saturation-20260926/fp32-16id-vram-series.csv)
- Structured derivation and source hashes: [`fp32-16id-vram-series.json`](../../runs/learning/16id-serving-saturation-20260926/fp32-16id-vram-series.json)
- Independent recomputation audit (11 checks passed): [`2026-09-26-16id-fp32-vram-audit.json`](2026-09-26-16id-fp32-vram-audit.json)
- Stop/interruption evidence: [`fp32-user-stop-amendment.json`](../../runs/learning/16id-serving-saturation-20260926/fp32-user-stop-amendment.json)
- Full batch raw results: `runs/learning/16id-serving-saturation-20260926/results.json`, `results-large-batches.json`, `results-near-target.json`, `results-fp32-vram.json`, and `results-fp32-vram-extension.json`.

The plotted full-decode-then-cap data from the earlier 130 W campaign is excluded. Only actual `max_new_tokens=16` runs enter this series.
