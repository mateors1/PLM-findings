# Native saturation experiment — declared 2026-09-25

## Question

How do the current native HTTP service and its internal `NativePredictor.predict`
path respond to parallel callers? Measure completed output tokens per wall second,
completed requests per wall second, client p50/p95/p99, server generation and queue
time, admission 503s, and correctness at each concurrency level.

## Fixed setup

- One RTX 5070 Ti, one checkpoint and frozen graph snapshot. Use the existing
  seed-1729 symmetric checkpoint and the recorded guided serial policy
  (`alpha=16`, uniqueness and KV cache on) for both paths.
- One predictor and one loopback Uvicorn worker. The internal path invokes that
  predictor directly; HTTP uses `/v1/predict`. No separate model copies.
- Eight fixed validation prompts, each warmed once through each path. Each
  successful timed response must match its warmup raw token IDs and deployment
  hashes. Neither protected final-test data nor oracle labels enter the workload.
- Closed, synchronized request waves at concurrency 1, 2, 4, 8 and 16, with
  16 offered requests per level and two repetitions. No retries or think time.
  `max_pending=8`; overload is allowed to return 503 with `Retry-After`.
- Alternate path order between repetitions, retaining individual level rows.
  Stop on unexpected HTTP status, invalid output, identity drift, or server
  shutdown failure. Record all accepted/rejected requests and source/runtime
  identities. All test processes must exit at the end.

## Interpretation

The paths share a serial model lock. Parallel callers are expected to increase
queue time before they increase model throughput. Throughput counts only
completed generated tokens, including EOS and excluding the five prompt tokens.
At overload, report both offered and completed rates and the 503 fraction.
These fixed-prompt, loopback, single-GPU observations do not establish a
production SLO, a maximum across workloads, an oracle comparison, or an energy
advantage. The offline batch-eight decoder is a separate path.
