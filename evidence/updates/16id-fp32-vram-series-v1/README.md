# User-stopped FP32 sixteen-ID batch curve

Dated 2026-09-26; experiment `16id-fp32-vram-series-stopped-v1`.
Read the [interpretation and limitations](../../../papers/fp32-sixteen-id-batch-curve.md).

This bundle contains 15 exact byte copies selected from the immutable 18-file source packet `docs/publication-queue/2026-09-26-16id-fp32-vram-series-v1.json` (SHA256 `ec8817323988b7738cbcf1f89a24cc23d26f7c2d9f2cd945653ee69ab607a4c6`). The three omitted source files are whole evolving-log/index snapshots: `docs/benchmarks.md`, `research_log.md`, and `devlog.md`; they remain in that packet. No raw repetition payload, checkpoint, database or new hardware run is included.

[Provenance map](provenance-map.json) identifies every source snapshot and copied byte hash. [SHA256SUMS.txt](SHA256SUMS.txt) lists bundle-relative checksums; it excludes itself. The original source-relative hierarchy is preserved below `originals/`, so original report links resolve without rewriting Markdown. There are no projected or modified evidence copies. The paper is a new interpretation, not an executed scientific artifact.

The [original report](originals/docs/experiments/2026-09-26-16id-fp32-vram-stopped-report.md), [audit](originals/docs/experiments/2026-09-26-16id-fp32-vram-audit.json), [decision](originals/docs/experiments/2026-09-26-16id-fp32-vram-decision.json), [CSV](originals/runs/learning/16id-serving-saturation-20260926/fp32-16id-vram-series.csv) and [JSON](originals/runs/learning/16id-serving-saturation-20260926/fp32-16id-vram-series.json) preserve the stopped series. The audit/plot scripts are exact historical copies, not self-contained runnable reproductions: the raw result files identified inside them remain local. They were not executed for this supplement.

The saved audit is a primary-task recomputation, not independent execution or external review. Zero recorded protocol/cap validity at the peak and plateau remains visible; exact reference-prefix parity does not establish valid service or oracle quality. The user-declared 9,600 practical peak is distinct from the observed 9,793.5 maximum. The partial 65,536 point is excluded from complete-row statistics; no CUDA OOM or VRAM saturation was observed.
