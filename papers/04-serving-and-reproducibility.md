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

## The general reproducibility lesson

A Git SHA alone cannot identify runs made with an uncommitted source tree. The source project records exact source archives, resolved configuration, dataset/split/vocabulary and checkpoint hashes, seeds, runtime, output receipts and independent audits. Application identity also includes decoding policy, because a fixed checkpoint can produce different answers under different selection rules. Saving both candidate paths and selected sets prevents a derived union from masquerading as a generated sequence. The batch-shape failure shows why even numerical *scores* require an explicit comparison contract; matching outputs and matching FP32 bytes are different claims.
