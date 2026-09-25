# Compiled FP8 follow-up — declared 2026-09-25

## Question

The eager torchao FP8 path retained quality but lost throughput. Does compiling
its dynamic E4M3 linear path with TorchInductor change that serving conclusion?

## Fixed inputs and comparison

- Reuse the guided seed-1729 checkpoint, all 222 validation queries, protocol
  mask, uniqueness, alpha-16 first-target guidance, maximum output length,
  graph snapshot and same eight serving prompts from the paired eager experiment.
- Fresh FP32 and compiled FP8 arms load and validate the same original checkpoint
  independently. FP8 converts the same 56 decoder-block linears as before.
  A single shared `torch.compile(..., backend='inductor', fullgraph=True)`
  functional linear callable executes those layers. Other weights and KV cache
  remain BF16. Record PyTorch, torchao, Triton, GPU, original checkpoint and
  script/source identities. No training, default change or protected test use.
- Use the same accepted seed-1730 PLM frontier macro F1 0.9730912228699782
  from shape-aware v2. Quality is postprocessed validation macro F1 divided by
  this established PLM result, multiplied by 100. Also report exact sets,
  validity, termination and raw agreement with the fresh FP32 arm.

## Execution and measurement

- Measure conversion and first calls for shapes 1, 8 and 32 separately as
  startup/compilation overhead. Compilation must succeed for all three shapes
  before timed serving or quality conclusions.
- Time 16 direct `NativePredictor.predict` requests at concurrency 1 and 4,
  after warmup, including postprocessing/hydration. Report completed output
  TPS/RPS and client p50/p95, queue p95, generation p50.
- Time two warmed synchronized fixed-group repetitions at batch 1, 8 and 32.
  Include completed outputs only, with EOS. Report TPS/RPS and PyTorch peak
  allocated memory. Keep compilation outside timed regions.
- Compare compiled FP8 with both fresh FP32 and the existing eager FP8 report.
  This tests native direct calls and offline fixed groups, not HTTP arrivals,
  TensorRT or a deployment SLO.

## Decision

Accept this compiled variant as a performance improvement only if measured
completed throughput and quality jointly support it. Record a negative result
if the graph fails to compile, produces invalid outputs, or remains slower.
