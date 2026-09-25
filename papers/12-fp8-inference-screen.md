# 12. FP8 storage did not make this inference path faster

**Status:** completed paired validation and performance screen; eager FP8 serving variant rejected. The separate compiled FP8 follow-up is declared but has no result in this snapshot.

## Question and intervention

Lower-precision weights could reduce memory traffic, but an inference implementation must also execute its kernels efficiently. The source experiment held the guided seed-1729 checkpoint, validation split and decoding policy fixed. It loaded the original checkpoint independently for each arm, converting 56 decoder-block linear layers to torchao 0.17.0 dynamic per-tensor E4M3 FP8. Embeddings, tied output head, relation head, norms and KV cache remained BF16. No training or checkpoint rewrite occurred. See the [declared plan](../evidence/updates/2026-09-25-fp8-screen/2026-09-25-fp8-paired-plan.md) and [portable result](../evidence/updates/2026-09-25-fp8-screen/2026-09-25-fp8-paired.json).

## Measured outcome

| Same checkpoint, different inference implementation | FP32 native | Eager FP8 |
| --- | ---: | ---: |
| Postprocessed validation macro F1, 222 queries | 0.885711 | 0.881841 |
| Exact sets | 151/222 | 152/222 |
| Protocol-valid and terminated responses | 222/222 | 222/222 |
| Serial direct-call completed output tokens/s | 148.5 | 12.0 |
| Serial client p95 | 1.72 s | 21.98 s |
| Offline fixed batch 32 completed output tokens/s | 2,968.8 | 232.8 |

FP8 raw tokens match FP32 for 211/222 queries. Its slightly higher exact-set count does not overturn the lower macro F1 or the roughly twelve-fold throughput loss. The reported percentages of a selected PLM frontier compare against an accepted seed-1730 composition policy with a different inference path; they are **not oracle parity**. The [raw summary and executed script](../evidence/updates/2026-09-25-fp8-screen/README.md) preserve the run details.

This tests eager torchao kernels on one Windows GPU, not all FP8 implementations. Compiling the path is a [separately declared experiment](../evidence/updates/2026-09-25-fp8-screen/2026-09-25-fp8-compiled-plan.md); no compiled-kernel speedup should be inferred before its own complete measurement. Direct native calls are not HTTP arrivals, and no energy or protected-test result follows.

**Lesson:** compressed representation and faster serving are separate claims. A lower-precision variant must preserve task quality *and* win measured end-to-end throughput or latency under its actual execution backend.
