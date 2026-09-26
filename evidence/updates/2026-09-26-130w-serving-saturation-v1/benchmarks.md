## 130 W power-target serving calibration (2026-09-26)

Each logical served result is capped at 16 product IDs. The graph arm is loopback Uvicorn HTTP; PLM is offline fixed-group FP32 CUDA decoding followed by a post-decode cap. These rates are not online capacity comparisons.

| Path and operating point | Completed 16-ID servings/s | Sustained power |
| --- | ---: | ---: |
| SQLite graph, concurrency 4 (entry to the single-process power plateau) | 580.1 | 86.2 W CPU package |
| PLM, fixed batch 336 | 151.2 | 129.8 W GPU board |

The graph path flattened around 76.3–81.5 W and did not reach 130 W; its fastest measured point was concurrency 1 at 1015.7 servings/s and 50.7 W. PLM batch 336 was the closest tested operating point: each of three repetition medians was in the 125–135 W band; strict full-token parity held on the repeated eight-prompt workload. The cap does not shorten autoregressive decoding. Capped 16-ID validation quality over 222 archived queries was 153/222 exact sets and macro F1 0.83052; the graph oracle was exact.

The selected PLM rate is offline completed rows/s, not HTTP requests/s, hydration-inclusive throughput, or a latency SLO. See the [declared plan](experiments/2026-09-26-130w-serving-saturation-plan.md), [portable result](experiments/2026-09-26-130w-serving-saturation.json), [audit](experiments/2026-09-26-130w-serving-saturation-audit.json) and [decision](experiments/2026-09-26-130w-serving-saturation-decision.json).

#### Per-watt serving efficiency

For a 16-ID response cap, each completed decode row is counted as one returned serving-equivalent, then normalized as `median completed rows/s ÷ median GPU board watts`. The cap is post-decode in these measurements, so this is not the rate for a decoder that stops after emitting 16 IDs.

| Batch | Capped serving-equivalents/s | Median GPU board power | Serving-equivalents/W | Measurement |
| ---: | ---: | ---: | ---: | --- |
| 320 | 139.71 | 125.071 W | 1.117 | 2026-09-26 targeted run |
| 336 | 151.20 | 129.751 W | 1.165 | 2026-09-26 targeted run |
| 640 | 188.30 | 152.2 W | 1.237 | 2026-09-25 full saturation sweep |
| 656 | 188.30 | 150.9 W | 1.248 | 2026-09-25 near-peak sweep |

Batch 640 had the historical sweep's peak measured throughput; batch 656 tied it within the reported precision with a slightly higher derived efficiency. The 2025 sweep's device isolation was not established, and its results are a separate run from the 2026 targeted measurements. The [batch-size efficiency graph](experiments/2026-09-26-plm-serving-efficiency.svg) shows all points and separates the two runs.
