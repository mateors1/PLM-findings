# 16-ID VRAM saturation amendment — 2026-09-26

## User-directed endpoint

This amendment supersedes the earlier power-target and throughput-peak stop
rules for the 16-ID GPU serving run. Saturation means the largest feasible
fixed batch followed by a larger batch that reaches CUDA's out-of-memory
allocation boundary. A falling serving rate, power draw, GPU busy percentage,
or output score is a recorded regression and does not end the sweep.

## Arms and workload

- Run the retained guided seed-1729 checkpoint and identical validation prompt
  mix in fresh FP32 and TorchAO dynamic-E4M3 FP8 with the existing fullgraph
  TorchInductor compiled decoder. Keep the KV cache, unique targets and
  alpha-16 first-target guidance policy fixed.
- Each serving returns at most 16 generated entity IDs and stops on model EOS
  or the 16-ID bound. Do not append or count a synthetic EOS.
- Measure warmed offline fixed-group decode throughput, returned IDs/s,
  board power and per-watt rate. Record first-call/warmup time, peak PyTorch
  allocation/reservation, NVML device memory, GPU busy and power at each batch.
- Compare each FP32 batch with the archived first-16 serial reference. For FP8,
  also compare each batch with a fresh serial compiled-FP8 first-16 reference.
  Report both parity rates and first-16 precision, recall and F1 against graph
  targets for the unique eight-query validation slice. These quality measures
  are diagnostics and never a saturation stop rule.
- Use three repetitions of at least five seconds at each feasible coarse batch.
  After the first OOM, bisect the last feasible / first failing interval to a
  128-row width. Persist the failing allocation and before/after memory state.

## Interpretation boundary

The benchmark is fixed-group offline decoding, not online HTTP capacity. NVML
power, memory and utilization are device-wide under Windows WDDM, with desktop
processes resident. The eight-query label metrics are a diagnostic slice, not
a replacement for the 222-query validation. Report accuracy and throughput
regressions separately from the VRAM boundary. A CUDA OOM without measured
device memory pressure must be called an allocation-bound failure; retain its
error and memory evidence instead of describing it as full physical VRAM use.
