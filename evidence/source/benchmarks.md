# Benchmarks (Phase D)

Measured, not extrapolated. At fixed concurrency / SLOs, report:

- **Latency:** p50 / p95 / p99.
- **Throughput:** RPS, tokens/sec.
- **Resource:** GPU memory, utilization, watts (via
  [`plm.observability`](../src/plm/observability/__init__.py)).

## Targets

1. **Project-native PyTorch inference server** — validated first.
2. **vLLM adapter** — separate Linux/WSL2 environment
   ([`serving/`](../serving/README.md)), with the exact vLLM / torch / CUDA
   versions recorded alongside every number.

Phase D exit: comparable quality with transparent, repeatable reports — not a
throughput claim from an unmeasured extrapolation.

## Native generation microbenchmark (2026-09-24)

An opt-in KV cache reproduces all 222 anchor-model validation responses exactly.
Eight prompts and three paired repetitions measured 20.352 s uncached versus
19.527 s cached summed generation time (1.0423x ratio), on one RTX 5070 Ti with
FP32 inference. This excludes HTTP, hydration and concurrent scheduling.
[Lesson 7](learning/07-kv-caching.md) documents the workload, memory results,
profiling follow-up and limitations; the [raw summary](experiments/2026-09-24-kv-cache.json)
retains individual timings and identities. This does not satisfy Phase D's exit.

## Native HTTP correctness run (2026-09-24)

A real loopback Uvicorn server reproduced all 222 archived raw validation
responses and the prior offline processed metrics (105/222 exact). Serial client
latency had median 0.690 seconds over 167.55 seconds total. No warmup, concurrency
sweep or independent repetitions were used; the first request is included.
These observations are not a comparable performance campaign and should not be
compared to the different eight-prompt microbenchmark as a speedup.

The [HTTP report](experiments/2026-09-24-native-service.json) binds deployment
identities; [Lesson 8](learning/08-native-serving.md) explains the checks and clean
server shutdown. A separate instrumented trace reduced vocabulary classification
calls from 715,638 to 15,393 after removing unnecessary grammar scans. Its
[profile](experiments/2026-09-24-native-overhead.json) describes overhead, not a
paired uninstrumented latency or GPU-utilization result. Oracle comparison,
concurrency/SLO and energy measurements remain pending.
