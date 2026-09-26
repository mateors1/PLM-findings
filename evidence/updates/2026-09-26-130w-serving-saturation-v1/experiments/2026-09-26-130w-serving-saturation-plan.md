# 130 W serving saturation calibration — declared 2026-09-26

## Question

At a fixed response size of 16 product IDs (within the requested 10–20 range),
what completed servings per second can the SQLite graph path sustain at about
130 W CPU-package power and the PLM batch decoder sustain at about 130 W GPU
board power? Record the graph's request concurrency and the PLM's fixed batch
size for follow-up calibration.

## Workload

- Use the same first eight validation `(subject, dimension)` prompts as the
  existing native graph and batch-saturation runs. Do not read protected test
  examples or change graph/model data.
- The CPU arm is the deterministic SQLite graph oracle over the immutable
  Pokémon snapshot. Serve over loopback HTTP with request `limit=16`; use a
  read-only SQLite connection per request worker so concurrent graph requests
  can exercise the CPU. Report worker concurrency as the CPU operating point.
- The GPU arm is FP32 offline fixed-group PLM decoding on the existing seed-1729
  checkpoint with the established guided, constrained, unique, KV-cached policy.
  Start at batch 320, guided by the prior measured 256-row 120.8 W and 384-row
  136.2 W points. Preserve strict full-token parity with the archived serial
  responses. Count one decoded row as one serving and cap its served result to
  16 IDs after generation; retain internal generated-token counts separately.
- A serving returns exactly 16 entity IDs on both arms. The PLM's numeric
  `limit` is post-processing metadata and does not shorten model decoding.

## Measurement and stopping

- Target a sustained median of 125–135 W, centered on 130 W. Do not change CPU
  or GPU power limits. If the graph cannot reach the band by concurrency 128,
  report the closest completed level without calling it a 130 W point.
- Run CPU concurrency at 1, 4, 8, 16, 32, 64 and 128, for at least 10 timed
  seconds per level. Use the Windows `Energy Meter(RAPL_Package0_PKG)` power
  counter, cross-checked against the interval energy delta, sampled once per
  second. Increase in smaller steps around the first level that crosses 130 W.
- Run the GPU batch-size point for three timed repetitions of at least five
  seconds each, sampling NVML power and utilization. Adjust the batch in
  multiples of eight around the first point until its median is in the target
  band. Stop on token drift, invalid termination, CUDA OOM, under 2 GiB free
  VRAM, or a work unit longer than 30 seconds.
- Report completed servings/s as successful 16-ID responses divided by timed
  wall seconds, together with actual median/range power, CPU concurrency or GPU
  batch, and generated tokens/s for the PLM. Power is CPU-package or GPU-board
  telemetry, not whole-system wall power. The two arms run sequentially.

## Interpretation

This is an operating-point calibration for later experiments. It does not
establish matched-quality energy efficiency, online GPU HTTP capacity, or a
production latency SLO. The graph computes the full oracle answer before its
16-ID result cap. The PLM likewise decodes its full answer before post-processing
the response cap; its internal token count must remain visible alongside
servings/s.
