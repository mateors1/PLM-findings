# 5. Evidence ledger and next experiments

**Purpose:** state exactly what this snapshot supports and which gates remain open. The [source research log](../evidence/source/research_log.md) is the chronological decision record; the [source devlog](../evidence/source/devlog.md) records execution. These papers synthesize that history but do not replace the dated plans and reports in [`evidence/experiments`](../evidence/experiments/).

## Claim ledger

| Claim | Status at snapshot | Best local evidence |
| --- | --- | --- |
| The TYPE/COLOR SAME oracle and corpus are deterministic for the pinned graph/protocol. | Implemented and locally verified; the oracle defines correct labels. | [Manifest](../evidence/pokemon_v1_f1541479_20260924.json), [protocol](../evidence/source/protocol-spec.md), [Lesson 2](../evidence/learning/02-data-and-evaluation.md) |
| Training, checkpoint resume and prompt-only generation have tested reference paths. | Implemented and tested; correctness gates are scoped to their exercised cases. | [Lesson 1](../evidence/learning/01-training-correctness.md), [architecture](../evidence/source/architecture.md) |
| Auxiliary supervision improves this fixed validation split. | Repeated across three training seeds; no protected-test claim. | [Lessons 6](../evidence/learning/06-seed-replication.md) and [12](../evidence/learning/12-symmetry-replication.md) |
| Four-path learned set selection works in the application path. | Offline and native HTTP integration independently audited for the declared configuration. | [Lesson 24](../evidence/learning/24-integrating-set-selection.md) |
| Pair unions reach 569/666 exact validation sets. | Offline experiment and revised shape-aware application verification pass; the first score-parity campaign remains failed. | [Lesson 25](../evidence/learning/25-composing-candidate-sets.md), [accepted v2 receipt](../evidence/updates/2026-09-25-composition-verification/experiments/2026-09-25-pair-composition-shape-aware.json) |
| Direct unrestricted membership prediction should replace the candidate pool. | Rejected by its declared exact-answer gate: 339/666 exact. | [Lesson 28](../evidence/learning/28-direct-membership-ablation.md) |
| PLM matches oracle answer quality. | Open: 569/666 offline validation observations are below 666/666; the protected final test is unused for these claims. | [Lessons 25](../evidence/learning/25-composing-candidate-sets.md) and [27](../evidence/learning/27-remaining-errors-and-score-signs.md) |
| PLM has a serving or energy advantage over the oracle at matched quality. | Open: selected microbenchmarks, correctness checks and a fixed-prompt native concurrency probe do not provide this comparison. | [Source benchmarks](../evidence/source/benchmarks.md), [Paper 4](04-serving-and-reproducibility.md) |
| The deferred complementary/compositional policy experiment succeeds. | Not tested here. | [Research log](../evidence/source/research_log.md), [Paper 1](01-research-question-and-protocol.md) |
| Native serial serving scales throughput with additional callers. | Not observed in the 2026-09-25 fixed workload: approximately 156–159 HTTP output TPS through eight callers while p95 latency rises. | [Dated report](../evidence/updates/2026-09-25/experiments/2026-09-25-native-saturation.json), [Paper 4 addition](04-serving-and-reproducibility.md#2026-09-25-addition-concurrency-exposes-a-serial-scheduler) |
| Fixed-group GPU decoding reaches about 31,000 completed output TPS. | Observed offline on one RTX 5070 Ti at batch 640 (30,980 TPS, 188.3 completed requests/s); batch 656 is level and 672 slows. This is not HTTP serving capacity or proven GPU arithmetic saturation. | [Batch sweep receipt](../evidence/updates/2026-09-25-batch-saturation/experiments/2026-09-25-batch-saturation.json), [Paper 4](04-serving-and-reproducibility.md#2026-09-25-follow-up-offline-batch-throughput-peaks-near-640) |
| Eight-branch first-choice search improves validation quality at higher inference cost. | Passed the fixed three-seed gate: 603/666 exact answers, 98.35% macro F1 and one query loss; additional ranks roughly double generation time. | [Paper 14](14-wider-first-choice-search.md) |
| Eight-source pair intersections improve pooled exact answers without a seed regression. | Rejected: pooled exact answers rise 603→605/666, but seed 1729 F1 regresses and the fixed gate fails. | [Paper 15](15-eight-source-set-operations.md) |
| Historical model artifacts have traceable final-checkpoint identities. | The original 23-run snapshot and later 25-run inventory pass byte/receipt/source consistency; complete historical lineage, payload validation and durable archival remain open. | [Original inventory](../evidence/updates/2026-09-25/experiments/2026-09-25-model-version-inventory.json), [Margin follow-up](08-hardest-boundary-margin-failure.md) |
| Adding a hardest-boundary margin improves the current model. | Rejected for fixed margin 1.0 and coefficient 0.1 at seed 1729: fresh-control exact 192/222 falls to 6/222; no further-seed replication. | [Audited margin screen](08-hardest-boundary-margin-failure.md) |
| A weaker fixed margin improves both exact sets and global membership separation. | Rejected at coefficient .001: exact sets rise 191/222 to 194/222, but strict separation falls 168/222 to 164/222. | [Paired screen](10-weaker-margin-mixed-result.md) |
| Eager FP8 improves native serving throughput at similar validation quality. | Rejected on the measured guided seed-1729 path: serial output throughput falls from 148.5 to 12.0 tokens/s; compiled FP8 remains unmeasured. | [FP8 screen](12-fp8-inference-screen.md) |
| Protocolized retrieval of genuinely learned relations is a supported application. | Future hypothesis only; requires independent labels and fair simple retrieval/ranking baselines. | [Hypothesis](../evidence/updates/2026-09-25/experiments/2026-09-25-learned-retrieval-hypothesis.md) |

## Evidence rules used throughout

The [eight-branch follow-up](14-wider-first-choice-search.md) is the newer offline
quality result: 603/666 exact answers and 98.35% macro F1, with unchanged trained
weights and roughly twice the generation time. Its 63 remaining failures comprise
61 unavailable exact candidates and two selection misses. The application-verified
policy remains the earlier four-branch composition. Active research stays within
Pokémon; external datasets require an explicit user request.

The [later rank diagnosis](07-ranking-and-threshold-limits.md) refines the next
quality decision on historical four-source evidence: adding whole-source unions
creates no new exact candidate,
and even oracle-informed per-query thresholds on the frozen membership scores
are limited to 479/666 exact answers. Thus a pool-only union expansion or a
threshold-only replacement cannot beat the existing 569/666 on this evidence.
The first fixed margin training intervention subsequently failed its declared
screen, as [paper 8](08-hardest-boundary-margin-failure.md) records. Other training
changes and different candidate constructions remain untested hypotheses.

1. **Declare before measuring.** Plans fix candidates, metrics and acceptance gates. A failed gate remains a failure even when another metric improves. The [direct-membership plan](../evidence/experiments/2026-09-24-direct-membership-plan.md) and [report](../evidence/experiments/2026-09-24-direct-membership.json) illustrate this.
2. **Keep identities complete.** A run binds graph, protocol, records, vocabulary, query split, code/archive, config, checkpoint, runtime, seeds and evaluator. A new inference policy is a new result identity even with identical weights. See [Lesson 2](../evidence/learning/02-data-and-evaluation.md).
3. **Separate evidence layers.** CPU saved-score analysis, GPU generation, native HTTP replay, protected final test and controlled performance measurements prove different things. A pass in one layer does not fill a missing gate in another.
4. **Preserve the test firewall.** Development choices use train/validation. The protected final-test partition is for a declared finalization after candidate selection, not another tuning loop. The [source architecture](../evidence/source/architecture.md) and [query split lesson](../evidence/learning/02-data-and-evaluation.md) describe that contract.
5. **Audit the actual artifact.** Independent saved-output audits, score hashes and source archives matter because the source checkout had uncommitted changes. The [checksum inventory](../evidence/SHA256SUMS.txt) verifies this repository's copied evidence bytes.

## Next research decisions, in order

1. **Composition integration completed.** The two remaining HTTP checkpoints ran under the declared shape-aware contract; independent audit and separate acceptance pass. The original exact-score campaign remains failed. See [the follow-up](04-serving-and-reproducibility.md#2026-09-25-follow-up-revised-integration-accepted) for explicit fresh/reused accounting and limits.
2. **Attack missing candidates without losing exact answers.** The newer eight-source pool lacks an exact candidate in 61 of 63 remaining failures. The earlier four-source direct sign-thresholding experiment recovered members but spoiled 233 exact sets. Declare a candidate-construction hypothesis and paired per-seed nonregression gates before running it; distinguish omitted true members from unwanted extra members.
3. **Seek oracle parity on validation and preserve baseline comparisons.** Track whole-set exactness, subgroup behavior, F1, validity and termination, not only token loss. Retain the deterministic graph lookup and existing controls.
4. **Finalize once, on the protected test partition.** Freeze the selected source, data, model and inference policy; then run the protected evaluator and report its result without subsequent tuning on that partition.
5. **Measure systems benefit at comparable quality.** Use repeated latency distributions, concurrency and energy per request against a clearly specified oracle/lookup service on the same hardware and workload. Microbenchmark ratios alone do not answer the Experiment A question.

**Boundary:** these are recommendations derived from the snapshot, not completed experiments or authorization to tune against the protected test.
