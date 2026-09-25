# 5. Evidence ledger and next experiments

**Purpose:** state exactly what this snapshot supports and which gates remain open. The [source research log](../evidence/source/research_log.md) is the chronological decision record; the [source devlog](../evidence/source/devlog.md) records execution. These papers synthesize that history but do not replace the dated plans and reports in [`evidence/experiments`](../evidence/experiments/).

## Claim ledger

| Claim | Status at snapshot | Best local evidence |
| --- | --- | --- |
| The TYPE/COLOR SAME oracle and corpus are deterministic for the pinned graph/protocol. | Implemented and locally verified; the oracle defines correct labels. | [Manifest](../evidence/pokemon_v1_f1541479_20260924.json), [protocol](../evidence/source/protocol-spec.md), [Lesson 2](../evidence/learning/02-data-and-evaluation.md) |
| Training, checkpoint resume and prompt-only generation have tested reference paths. | Implemented and tested; correctness gates are scoped to their exercised cases. | [Lesson 1](../evidence/learning/01-training-correctness.md), [architecture](../evidence/source/architecture.md) |
| Auxiliary supervision improves this fixed validation split. | Repeated across three training seeds; no protected-test claim. | [Lessons 6](../evidence/learning/06-seed-replication.md) and [12](../evidence/learning/12-symmetry-replication.md) |
| Four-path learned set selection works in the application path. | Offline and native HTTP integration independently audited for the declared configuration. | [Lesson 24](../evidence/learning/24-integrating-set-selection.md) |
| Pair unions reach 569/666 exact validation sets. | Offline experiment passes its gate; separate application replay does not yet pass exact HTTP score parity. | [Lessons 25](../evidence/learning/25-composing-candidate-sets.md) and [26](../evidence/learning/26-integrating-composed-sets.md) |
| Direct unrestricted membership prediction should replace the candidate pool. | Rejected by its declared exact-answer gate: 339/666 exact. | [Lesson 28](../evidence/learning/28-direct-membership-ablation.md) |
| PLM matches oracle answer quality. | Open: 569/666 offline validation observations are below 666/666; the protected final test is unused for these claims. | [Lessons 25](../evidence/learning/25-composing-candidate-sets.md) and [27](../evidence/learning/27-remaining-errors-and-score-signs.md) |
| PLM has a serving or energy advantage over the oracle at matched quality. | Open: only selected microbenchmarks and serial HTTP correctness are measured. | [Source benchmarks](../evidence/source/benchmarks.md), [Paper 4](04-serving-and-reproducibility.md) |
| The deferred complementary/compositional policy experiment succeeds. | Not tested here. | [Research log](../evidence/source/research_log.md), [Paper 1](01-research-question-and-protocol.md) |

## Evidence rules used throughout

1. **Declare before measuring.** Plans fix candidates, metrics and acceptance gates. A failed gate remains a failure even when another metric improves. The [direct-membership plan](../evidence/experiments/2026-09-24-direct-membership-plan.md) and [report](../evidence/experiments/2026-09-24-direct-membership.json) illustrate this.
2. **Keep identities complete.** A run binds graph, protocol, records, vocabulary, query split, code/archive, config, checkpoint, runtime, seeds and evaluator. A new inference policy is a new result identity even with identical weights. See [Lesson 2](../evidence/learning/02-data-and-evaluation.md).
3. **Separate evidence layers.** CPU saved-score analysis, GPU generation, native HTTP replay, protected final test and controlled performance measurements prove different things. A pass in one layer does not fill a missing gate in another.
4. **Preserve the test firewall.** Development choices use train/validation. The protected final-test partition is for a declared finalization after candidate selection, not another tuning loop. The [source architecture](../evidence/source/architecture.md) and [query split lesson](../evidence/learning/02-data-and-evaluation.md) describe that contract.
5. **Audit the actual artifact.** Independent saved-output audits, score hashes and source archives matter because the source checkout had uncommitted changes. The [checksum inventory](../evidence/SHA256SUMS.txt) verifies this repository's copied evidence bytes.

## Next research decisions, in order

1. **Resolve the composition HTTP comparison contract.** Re-run the two unexecuted HTTP checkpoints under a declared, shape-aware score reference while retaining exact source-path, selected-set, response and failure-contract checks. The existing exact-score failure is the motivating evidence, not a passed acceptance result.
2. **Attack missing coverage without losing exact answers.** The fixed pool lacks an exact candidate in 96 of 97 remaining failures; direct sign-thresholding recovers many members but spoils 233 exact sets. Declare a candidate-construction hypothesis and paired per-seed nonregression gates before running it.
3. **Seek oracle parity on validation and preserve baseline comparisons.** Track whole-set exactness, subgroup behavior, F1, validity and termination, not only token loss. Retain the deterministic graph lookup and existing controls.
4. **Finalize once, on the protected test partition.** Freeze the selected source, data, model and inference policy; then run the protected evaluator and report its result without subsequent tuning on that partition.
5. **Measure systems benefit at comparable quality.** Use repeated latency distributions, concurrency and energy per request against a clearly specified oracle/lookup service on the same hardware and workload. Microbenchmark ratios alone do not answer the Experiment A question.

**Boundary:** these are recommendations derived from the snapshot, not completed experiments or authorization to tune against the protected test.
