# FP32 sixteen-ID batch curve: stopped before saturation

Version 1, 2026-09-26. Experiment `16id-fp32-vram-series-stopped-v1`.
This is a descriptive publication supplement; no model or inference method is promoted.

The completed measurements reached a highest median **9,793.5 decoded rows/s** at batch 45,056 on the RTX 5070 Ti. The user stopped the campaign before a CUDA out-of-memory boundary was observed. It does **not** establish VRAM saturation, online request capacity, oracle quality or protocol-valid serving capacity. The frozen files call this rate `servings_per_second`; here “decoded rows/s” avoids treating reference agreement as successful protocol service. [Merged data](../evidence/updates/16id-fp32-vram-series-v1/originals/runs/learning/16id-serving-saturation-20260926/fp32-16id-vram-series.json)

The separately declared practical peak is **9,600 rows/s**, a user choice rather than the measured maximum. The first completed median at least 99% of that value (9,504) occurs at batch 8,192: 9,555.1 rows/s and 5.58 GiB device-wide peak memory. “First plateau” means this declared threshold rule, not a statistical saturation estimate. [Stop amendment](../evidence/updates/16id-fp32-vram-series-v1/originals/runs/learning/16id-serving-saturation-20260926/fp32-user-stop-amendment.json)

| Completed point | Batch | Decoded rows/s | Device-wide peak GiB |
| --- | ---: | ---: | ---: |
| First point meeting the declared 99% rule | 8,192 | 9,555.1 | 5.58 |
| Highest completed throughput | 45,056 | 9,793.5 | 15.00 |
| Largest completed batch | 57,344 | 845.5 | 15.69 |

The decision's `highest_completed_batch: 45056` names the highest-throughput point; the largest completed batch is 57,344. A 65,536 attempt completed only two of three planned repetitions before interruption. Its console-only partial estimate is excluded from completed-row peak and plateau statistics. No CUDA OOM was observed. [Decision](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-fp32-vram-decision.json), [partial-row record](../evidence/updates/16id-fp32-vram-series-v1/originals/runs/learning/16id-serving-saturation-20260926/fp32-user-stop-amendment.json)

Each complete row summarizes three warmed repetitions of at least five seconds of an offline repeated workload. Actual generation is capped at 16 entity IDs, with earlier model EOS allowed and no synthetic EOS. This excludes the earlier full-decode-then-cap campaign. Tokens/s counts returned entity IDs and excludes EOS; rate/W divides median rate by median NVML board power. Device-wide power and memory include the Windows WDDM desktop environment. These measurements are not per-request energy isolation or online HTTP concurrency measurements. [Frozen report](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-fp32-vram-stopped-report.md), [plan amendment](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-vram-saturation-amendment.md)

All 28 completed rows record exact first-16 ordered-prefix agreement of 1.0 against the archived serial FP32 reference. **The recorded `protocol_valid_and_cap_complete_rate` is 0.0 at both the 8,192 plateau point and the 45,056 peak point.** Earlier rows have null rather than measured validity values. Prefix agreement therefore does not establish protocol validity. The cause of the zero validity field is unresolved by this frozen bundle; this supplement makes no causal diagnosis. The available label macro F1 is approximately 0.1509 on only the first eight validation queries, not a full validation-quality result. Reference parity, structural validity and agreement with correct target sets are separate measurements. [Structured metrics](../evidence/updates/16id-fp32-vram-series-v1/originals/runs/learning/16id-serving-saturation-20260926/fp32-16id-vram-series.json)

The saved audit reports 11/11 checks passing, including recomputation from local raw repetition rows. Its own attribution is **primary-task recomputation**, not a second-agent review, independent hardware execution or external review. This supplement independently checks only the frozen derived files, their byte identities, links and bounded packaging; it does not repeat the raw-row audit or hardware run. The five raw result files and checkpoint payloads remain local dependencies and are not copied here. Their hashes identify them without providing backup or public reproducibility. No retained-model or canonical-paper change is made. [Audit](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-fp32-vram-audit.json), [evidence inventory](../evidence/updates/16id-fp32-vram-series-v1/README.md)

The six original figures are preserved unchanged, including their original “servings” labels. Read those labels with the validity limitation above:

- [Decoded-row throughput](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-fp32-servings-throughput.svg)
- [Returned-ID throughput](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-fp32-tokens-throughput.svg)
- [Decoded rows per watt](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-fp32-servings-per-watt.svg)
- [Returned IDs per watt](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-fp32-tokens-per-watt.svg)
- [Memory and power](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-fp32-memory-power.svg)
- [Board power](../evidence/updates/16id-fp32-vram-series-v1/originals/docs/experiments/2026-09-26-16id-fp32-board-power.svg)
