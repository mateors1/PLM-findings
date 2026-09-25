# Paired FP8 serving and quality experiment — declared 2026-09-25

## Question

For the existing guided seed-1729 checkpoint, does dynamic E4M3 FP8 inference
improve native serving throughput or fixed-group batch throughput, and how much
validation quality does it retain?

## Arms and identities

- Load the same original checkpoint through the validated inference boundary for
  each arm. FP32 uses the current native weights. FP8 converts only the 56
  decoder-block linear layers after checkpoint validation, using torchao 0.17.0
  dynamic per-tensor E4M3 activations and weights. BF16 embedding, tied output
  head, auxiliary head, norms and KV cache remain unquantized. No retraining.
- Keep protocol mask, target uniqueness, first-target guidance alpha 16,
  max output length, data split and validation prompts fixed.
- Preserve checkpoint byte hash separately from the FP8 conversion recipe and
  executed software versions. Conversion is an inference variant, not a new
  trained checkpoint.

## Quality

- Evaluate all 222 validation queries in fixed groups of eight. Compare
  postprocessed SAME sets to validation labels using macro per-query F1 and exact
  set accuracy; report protocol validity, termination and raw FP32/FP8 token
  agreement separately. Do not use protected test queries.
- Report `100 × measured macro F1 / frontier macro F1`. The frozen frontier is
  the best accepted single-seed PLM validation result in the shape-aware v2
  composition integration report: seed 1730, 222 queries, macro F1
  `0.9730912228699782`. Bind its report hash. This measures each arm's quality
  as a percentage of the established PLM result, not percentage of the oracle.

## Serving and batching

- Warm eight validation prompts, then time 16 direct `NativePredictor.predict`
  requests at concurrency 1 and 4. Include postprocessing and hydration.
  Record completed RPS, output TPS, client p50/p95, queue p95 and generation p50.
- Warm and then time two repetitions of fixed-group batches 1, 8 and 32 on the
  same eight prompts. CUDA synchronize at timing boundaries; report completed
  output TPS, RPS and peak PyTorch allocation. Check protocol validity and EOS.
- Run one arm at a time on the RTX 5070 Ti. These are native direct-call and
  offline fixed-group measurements, not HTTP or an arrival-process SLO. The
  native predictor serializes generation at concurrency 4.

## Acceptance

FP8 must complete the same 222-query evaluation with finite metrics and valid,
terminated serving responses. Treat speedup and quality retention as measured
outcomes. Preserve negative results; do not change the serving default.
