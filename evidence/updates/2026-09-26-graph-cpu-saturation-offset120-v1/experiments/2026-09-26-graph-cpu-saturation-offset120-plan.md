# SQLite graph CPU saturation with a fixed 120 W power offset — 2026-09-26

## Question

What throughput and response latency does the deterministic SQLite graph HTTP path sustain as client concurrency rises, and what do the measured CPU-package RAPL readings look like after adding the requested fixed 120 W offset?

## Workload

- Reuse the immutable graph snapshot, serving receipt, protocol, and first eight validation prompts from the prior graph-arm campaign.
- Serve the graph oracle through the same loopback `/v1/predict` HTTP path with a 16-ID response cap. Require every successful response to match the frozen graph oracle targets.
- Run client concurrency 1, 4, 8, 16, 32, 48, 56, 64, 72, 80, 96, 112 and 128 for 10 timed seconds each.

## Measurement

- Sample Windows `Energy Meter(RAPL_Package0_PKG)` at its one-second interval during each load level. Preserve every raw package-power sample.
- This run has **no power target or budget**. Record the observed power and throughput curve through the highest requested concurrency.
- Apply the user's 120 W offset additively for a second derived series: `offset_adjusted_watts = measured_cpu_package_watts + 120`. Preserve raw and adjusted rates separately. The offset is a mathematical adjustment, not a direct wall-power measurement.
- Report completed 16-ID graph responses/s, median and range latency, failure counts, CPU-package power median/range, adjusted power median/range, and raw/adjusted responses per watt at each concurrency.
- Treat throughput plateau/regression and package-power plateau as observations. Do not stop or select a point based on an assumed wattage limit.

## Interpretation limits

The fixed offset is not an independently measured system baseline. The adjusted power and efficiency values are therefore offset-adjusted estimates, not whole-system wall power. The graph computes the oracle answer before the 16-ID cap. The HTTP result is a local deterministic benchmark, not a production capacity or latency SLO.
