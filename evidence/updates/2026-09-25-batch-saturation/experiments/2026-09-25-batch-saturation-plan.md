# Increasing batch-size saturation test — declared 2026-09-25

## Question

What completed-output throughput can the current fixed-group CUDA decoder sustain
as batch size increases, and where does its throughput plateau? Track GPU busy
time, memory and power alongside TPS; do not infer GPU compute saturation from
TPS alone.

## Fixed identity and workload

- Use the RTX 5070 Ti, seed-1729 symmetric checkpoint, the exact guided policy
  from the verified serial HTTP deployment (`alpha=16`, constrained, unique,
  cached), and the same first eight validation prompts used in the native
  saturation test. No training, retraining, oracle facts or protected test data.
- Compare eight current serial completions to the archived guided HTTP full
  token IDs before timing. Every batch-size output, including duplicated prompts,
  must match those same token IDs, validity and termination. A mismatch stops the
  sweep and is saved as a failed parity result.
- Use `generate_responses` at every batch size. For sizes 1/2/4, process the
  whole eight-prompt mix in groups; at size 8 use one group; above 8 replicate
  the eight prompts evenly. Report actual completed generated tokens including
  EOS, excluding the five prompt tokens and ignored slots after EOS.

## Measurement

- Sweep 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 and 1024. Extend to 2048 and
  4096 if throughput is still rising and GPU memory permits. Never extrapolate
  beyond the highest measured batch.
- Warm each size once. Run three timed repetitions, each long enough for at least
  five seconds of work. CUDA synchronize around each timed region. Measure wall
  time for complete decoded responses, completed output TPS and RPS, PyTorch
  peak allocated memory, and sampled NVML GPU utilization, memory and power.
- Run alone on the GPU. Record checkpoint, data, source, script and runtime
  identities. Stop on output drift, CUDA out-of-memory, less than 2 GiB free
  device memory, or an impractically long work unit. Save partial results and
  the exact stop reason. Warmup and correctness checks stay outside timing.
- Call practical throughput saturation only when two successive larger sizes
  fail to improve median completed TPS by at least 5%. Report GPU utilization
  and power separately; NVML's GPU-busy percentage is not SM occupancy or a
  hardware FLOP ceiling. If the sweep ends first, report the largest measured
  throughput and that saturation was not established.

## Scope

This is offline fixed-group decoding, not the current serial HTTP scheduler or
continuous batching. The measured TPS can describe a potential batched serving
backend, but it is not a live HTTP capacity or latency SLO.

## Adaptive refinement declared after the first sweep

The first sweep completed through batch 512. Batch 1024 preserved output parity
but its warmup took 136.54 seconds, compared with 2.91 seconds at batch 512;
device memory approached capacity during that attempt. Do not attempt 2048/4096
in this FP32 run. Measure 384, 640 and 768 instead, each with the same prompt
mix and three timed repetitions. Try 896 only if 768 remains timely and has
memory headroom. Use a 15-second warmup work-unit limit and record warmup GPU
telemetry. The practical maximum is the highest stable completed TPS under
these constraints; it is not a claim of peak GPU arithmetic throughput.

## Near-limit refinement declared after the 384/640/768 run

Batch 640 completed at about 30,980 TPS and roughly 14.73 GiB peak device
memory. The same process then reported less than 2 GiB free before batch 768,
which can include PyTorch's cached reserve. Run 656 and 672 in fresh processes,
with separate output files, to inspect whether completed TPS has flattened
before device-memory pressure. Record both PyTorch allocated and reserved peaks.
Attempt 688 only if 672 completes within 15 seconds and retains meaningful
memory headroom. Keep the same frozen prompts, checkpoint, policy and parity gate.
