# 4. Serving, integration and numerical reproducibility

**Status:** selected local correctness and microbenchmark results; no quality-matched concurrent serving or energy claim.

## Correctness precedes speed

The native service binds a checkpoint, protocol and corpus identity to an immutable SQLite snapshot. A real loopback HTTP run reproduced all **222** archived raw validation responses, and its processed metrics reproduced **105/222** exact answers for that earlier checkpoint. The server stopped cleanly. These checks cover request parsing, metadata, snapshot isolation and failure behavior; they do not repair model errors ([Lesson 8](../evidence/learning/08-native-serving.md), [report](../evidence/experiments/2026-09-24-native-service.json)). A later four-path reranking application campaign independently replayed **2,664 paths/scores, 666 selections and 666 HTTP responses** with exact parity under its recorded configuration ([Lesson 24](../evidence/learning/24-integrating-set-selection.md)).

Pair-composition integration has a more specific outcome. Its frozen offline GPU replay passed all three checkpoints. At the first serial HTTP checkpoint, all **222 selected answers and raw source paths** matched the offline references, while **all 2,220 normal slot scores** differed. The maximum absolute difference was about **0.00036144**. The campaign's **exact score-parity gate failed**, and the other two HTTP checkpoints were not run. This is not application acceptance for the offline **569/666** result ([Lesson 26](../evidence/learning/26-integrating-composed-sets.md), [batch-shape report](../evidence/experiments/2026-09-24-pair-composition-batch-shape.json)).

A bounded prompt-only diagnosis reproduced the discrepancy with the same source, checkpoint and config: batch-eight/final-six scores matched the offline evidence, serial scores matched HTTP evidence, and selected slots were unchanged for the tested 222 queries. Of **227,550** compared logits, **156,003** differed by batch shape, with maximum absolute logit difference about **7.63e-06** and no membership sign changes. This establishes a numeric cause for the observed exact-score mismatch in that run, not a blanket guarantee that future batch shapes preserve decisions. An appropriate follow-up contract must compare scores at matched shapes while retaining exact checks for sources, selected sets and returned answers. The original gate was not silently relaxed ([source research log's final 2026-09-24 entry](../evidence/source/research_log.md), [portable diagnosis](../evidence/experiments/2026-09-24-pair-composition-batch-shape.json)).

## Performance results with their limits

| Local workload | Observation | What it excludes |
| --- | --- | --- |
| Opt-in KV cache, eight prompts × three repetitions | Same 222 validation responses; 20.352 s uncached versus 19.527 s cached summed generation time, about 1.04× | HTTP, concurrent scheduling, oracle comparison and energy ([Lesson 7](../evidence/learning/07-kv-caching.md)) |
| Batched generation, fixed eight-query workload | 1,332 outputs matched exactly in the broader parity check; timed serial/batch ratio 5.3071× over three paired repetitions | General traffic, end-to-end service throughput and energy ([Lesson 14](../evidence/learning/14-batched-evaluation.md)) |
| Earlier native HTTP correctness run | 222/222 raw parity; observed serial median 0.690 s | Warmup, repeats, concurrency and a fair comparison to the different microbenchmarks ([Lesson 8](../evidence/learning/08-native-serving.md)) |

The batched and cached workloads answer different questions, so their ratios must not be multiplied into an imagined service speedup. The benchmark target remains latency distribution at a stated SLO, throughput/concurrent users, resource use and energy per request **at matched answer quality** against the perfect oracle or a lookup implementation. No such end-to-end comparison appears in this snapshot.

## 2026-09-25 addition: concurrency exposes a serial scheduler

A completed [native saturation probe](../evidence/updates/2026-09-25/experiments/2026-09-25-native-saturation.json)
compared loopback HTTP with direct `NativePredictor.predict` calls. Both used the
same seed-1729 symmetric checkpoint and guided serial decoding policy, eight
fixed validation prompts, two repetitions and sixteen offered requests per
concurrency level. This is a different checkpoint/policy from the later
three-checkpoint pair-composition result. It cannot establish performance at
569/666 exactness.

| Concurrent callers | HTTP completed output tokens/s | HTTP client p95 seconds, by repetition |
| ---: | ---: | --- |
| 1 | 156.1 | 1.63, 1.66 |
| 2 | 158.8 | 2.86, 2.80 |
| 4 | 157.7 | 5.16, 4.92 |
| 8 | 156.3 | 8.37, 8.57 |

Across both paths, 288 successful requests matched their warmed raw token
sequences and 32 overload requests returned 503. The recorded server shutdown
passed. Throughput stayed approximately flat while latency rose: callers queued
behind the shared serial execution lock. Direct-call throughput was similarly
about 156–160 output tokens/s, so HTTP transport was not the observed dominant
cost for this workload. At concurrency 16, half the offered requests were
rejected on each path; the accepted prompt mix varied, preventing a clean paired
TPS comparison with lower concurrency levels.

These are fixed-prompt, closed-wave measurements. They do not establish an
open-loop production SLO, independent-host performance, matched-quality oracle
advantage or energy savings. The larger offline batch-size sweep is reported
separately below. The copied [raw report](../evidence/updates/2026-09-25/raw/native-saturation-summary.json)
matches the hash named by the portable report.

## 2026-09-25 follow-up: revised integration accepted

The [declared shape-aware contract](../evidence/updates/2026-09-25-composition-verification/experiments/2026-09-25-pair-composition-shape-aware-plan.md)
has now completed with independent audit and separate acceptance. The
[portable receipt](../evidence/updates/2026-09-25-composition-verification/experiments/2026-09-25-pair-composition-shape-aware.json)
covers 666 HTTP responses: 222 reused from seed 1729 and 444 fresh from seeds
1730/1731. All 666 serial references and 24 repeats are fresh. Offline evidence
for 666 queries is reused, and the three HTTP auxiliary suites include one
reused and two fresh executions.

All 2,664 normal HTTP source paths, 666 selected slots/sets and 6,660 normal
HTTP slot-score checks pass. Scores use exact same-shape references with zero
tolerance; decisions remain identical across shapes. Exact answers are
191/192/186 by seed, totaling 569/666, preserving the offline quality result.
The first campaign remains failed, and the policy remains opt-in.

The audit authenticates saved vectors rather than independently regenerating
neural outputs. It checks nine owned-server shutdown receipts rather than
independent socket probes. These limits are retained in acceptance. This closes
the declared integration gate; protected test, oracle parity and matched-quality
systems/energy claims remain open. Earlier copied lessons are historical and
retain their pending status; the [new lesson copy](../evidence/updates/2026-09-25-composition-verification/learning/29-shape-aware-verification.md)
describes the completed verification.

## 2026-09-25 follow-up: offline batch throughput peaks near 640

The [declared sweep](../evidence/updates/2026-09-25-batch-saturation/experiments/2026-09-25-batch-saturation-plan.md)
timed fixed-group CUDA decoding on one RTX 5070 Ti with the guided seed-1729
FP32 checkpoint. It repeated the same eight validation prompts evenly above
batch eight. Three warmed timed repetitions per size counted completed generated
tokens, including EOS; all completed outputs matched archived serial tokens.
The [portable receipt](../evidence/updates/2026-09-25-batch-saturation/experiments/2026-09-25-batch-saturation.json)
binds the checkpoint, source tree, policy and four copied raw summaries.

| Batch size | Median completed output tokens/s | Completed requests/s | Median device-wide GPU busy | Peak device memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 163 | 1.0 | 19% | 2.85 GiB |
| 8 | 843 | 5.1 | 22% | 2.99 GiB |
| 64 | 6,507 | 39.6 | 37% | 3.17 GiB |
| 256 | 21,313 | 129.6 | 68% | 5.17 GiB |
| 512 | 29,451 | 179.0 | 83% | 11.35 GiB |
| 640 | **30,980** | **188.3** | 84% | 14.73 GiB |
| 656 | 30,973 | 188.3 | 84% | 15.27 GiB |
| 672 | 15,749 | 95.7 | 92% | 15.72 GiB |

Batch 640 was the highest measured median for this fixed workload; 656 was
effectively level, and 672 slowed sharply near device-memory capacity. Batch
1024 preserved token parity but took 136.54 seconds for one untimed warmup, so
it has no accepted TPS result. The memory-pressure explanation for the slowdown
is an inference from telemetry, not a kernel-profiled cause. NVML GPU-busy time
does not measure arithmetic occupancy or prove that compute capacity was
saturated.

This is **offline batched decoder throughput**, not current live HTTP capacity.
The native HTTP scheduler remained near 156 output tokens/s in its separate
serial concurrency probe. The sweep also provides no request-arrival, latency
SLO, hydration, matched-quality oracle or energy-per-request comparison.

## The general reproducibility lesson

A Git SHA alone cannot identify runs made with an uncommitted source tree. The source project records exact source archives, resolved configuration, dataset/split/vocabulary and checkpoint hashes, seeds, runtime, output receipts and independent audits. Application identity also includes decoding policy, because a fixed checkpoint can produce different answers under different selection rules. Saving both candidate paths and selected sets prevents a derived union from masquerading as a generated sequence. The batch-shape failure shows why even numerical *scores* require an explicit comparison contract; matching outputs and matching FP32 bytes are different claims.
