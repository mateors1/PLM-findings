# SQLite graph CPU saturation with a +120 W power offset — 2026-09-26

## Result

The fresh 16-ID graph HTTP sweep completed all 13 concurrency levels with exact
agreement to the frozen graph oracle and zero failed responses. Throughput was
highest at one client (954.2 servings/s), then settled at 435.4–460.9 servings/s
from concurrency 8 through 128. The measured CPU-package median entered a
roughly flat 104.1–110.9 W band over those same concurrent levels. Adding the
requested 120 W produces a derived band of 224.1–230.9 W; it is arithmetic on
RAPL data, not a wall-power measurement.

At the best concurrent point (8 clients), the graph returned 460.9 servings/s
at 108.58 W measured package power, or 228.58 W after the offset. The measured
and offset-adjusted ratios are 4.24 and 2.02 completed servings/W. At 128
clients, throughput was 435.4/s, p95 latency was 439.37 ms, measured package
power was 104.10 W, and offset-adjusted power was 224.10 W. This is an observed
throughput and package-power plateau over the tested range; it does not prove
100% CPU utilization or a hardware/thermal limit.

## Figures

- [Completed 16-ID servings/s](2026-09-26-graph-cpu-offset120-throughput.svg)
- [Measured and +120 W CPU-package power](2026-09-26-graph-cpu-offset120-power.svg)
- [Servings/W using both power series](2026-09-26-graph-cpu-offset120-efficiency.svg)

## Method

- CPU: Intel Core i9-14900K. The benchmark ran the deterministic SQLite
  `GraphOracle` through loopback Uvicorn HTTP at the existing concurrency levels
  1, 4, 8, 16, 32, 48, 56, 64, 72, 80, 96, 112 and 128.
- Each level ran for 10 timed seconds against the same eight prompts with a
  16-ID response cap. Every returned target sequence matched the frozen graph
  reference. HTTP failures: 0.
- Power came from Windows
  `Energy Meter(RAPL_Package0_PKG)` `CookedValue / 1000`. The raw samples remain
  unchanged. For each level, the second series is
  `measured package watts + 120`; servings/W uses the level's serving rate
  divided by the corresponding median watts.
- No wattage target was used. The sweep ran through the full declared
  concurrency range. RAPL is CPU-package power; the offset was not measured as
  an idle baseline and must not be described as whole-system power.

## Measurements

### Throughput and latency

| Client concurrency | Completed 16-ID servings/s | Median latency (ms) | p95 latency (ms) | Failed responses |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 954.2 | 1.02 | 1.16 | 0 |
| 4 | 565.7 | 6.56 | 10.93 | 0 |
| 8 | 460.9 | 15.65 | 27.86 | 0 |
| 16 | 459.2 | 31.11 | 55.90 | 0 |
| 32 | 450.4 | 62.71 | 113.12 | 0 |
| 48 | 448.8 | 94.29 | 170.22 | 0 |
| 56 | 448.2 | 110.59 | 198.78 | 0 |
| 64 | 442.8 | 128.05 | 229.54 | 0 |
| 72 | 442.7 | 143.72 | 257.20 | 0 |
| 80 | 446.1 | 158.73 | 282.19 | 0 |
| 96 | 441.0 | 191.78 | 336.11 | 0 |
| 112 | 437.1 | 226.14 | 389.88 | 0 |
| 128 | 435.4 | 260.89 | 439.37 | 0 |

### Measured and offset-adjusted power

The measured range is the minimum–maximum of the one-second RAPL samples
scoped to that level. The adjusted range adds 120 W to both endpoints.

| Concurrency | Measured package W (median [min–max]) | Median after +120 W | Servings/W measured | Servings/W after +120 W |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 76.75 [62.58–80.73] | 196.75 | 12.43 | 4.85 |
| 4 | 102.50 [91.76–109.69] | 222.50 | 5.52 | 2.54 |
| 8 | 108.58 [98.88–122.20] | 228.58 | 4.24 | 2.02 |
| 16 | 107.35 [95.54–114.64] | 227.35 | 4.28 | 2.02 |
| 32 | 106.99 [101.12–120.13] | 226.99 | 4.21 | 1.98 |
| 48 | 108.31 [100.54–120.07] | 228.31 | 4.14 | 1.97 |
| 56 | 106.59 [103.11–112.59] | 226.59 | 4.20 | 1.98 |
| 64 | 106.18 [100.77–116.79] | 226.18 | 4.17 | 1.96 |
| 72 | 110.91 [98.62–116.68] | 230.91 | 3.99 | 1.92 |
| 80 | 105.78 [100.23–110.60] | 225.78 | 4.22 | 1.98 |
| 96 | 106.91 [99.26–114.17] | 226.91 | 4.13 | 1.94 |
| 112 | 106.90 [98.20–118.03] | 226.90 | 4.09 | 1.93 |
| 128 | 104.10 [96.18–107.12] | 224.10 | 4.18 | 1.94 |

## Comparison with the prior clean graph sweep

The earlier clean sweep measured 1,015.7/s at 50.7 W for concurrency 1,
580.1/s at 86.2 W for concurrency 4, and 484.7/s at 86.3 W for concurrency 8.
This fresh run measured 954.2/s at 76.75 W, 565.7/s at 102.50 W, and 460.9/s
at 108.58 W at those levels. Thus throughput was 2.5–6.1% lower at these
comparison points while measured package power was higher by 16.3–26.0 W.
This is a run-to-run shift worth retaining, but the current sweep did not
capture CPU frequency, temperature, utilization, or concurrent package
activity, so it cannot attribute the difference to one cause.

## Evidence and limits

- Frozen method: [plan](2026-09-26-graph-cpu-saturation-offset120-plan.md)
- Portable per-level result, source hashes and raw/adjusted metrics:
  [structured summary](2026-09-26-graph-cpu-saturation-offset120.json)
- Secondary arithmetic and provenance checks:
  [audit](2026-09-26-graph-cpu-saturation-offset120-audit.json)
- Recorded outcome and limits:
  [decision](2026-09-26-graph-cpu-saturation-offset120-decision.json)
- Full raw run receipt and one-second RAPL samples:
  `runs/learning/power-target-saturation-20260926/graph-arm-offset120.json`
- Reusable aggregation and SVG rendering source:
  `runs/learning/power-target-saturation-20260926/aggregate_graph_cpu_offset120.py`
- CSV table:
  `runs/learning/power-target-saturation-20260926/graph-arm-offset120.csv`

The offset-adjusted watts and servings/W are derived quantities, not measured
whole-system energy efficiency. The graph path computes the oracle answer and
then caps the returned list at 16 IDs. These loopback results do not estimate
production request capacity or establish a PLM quality/serving comparison.
