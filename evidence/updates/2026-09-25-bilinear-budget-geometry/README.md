# Paired geometry across bilinear training budgets

Status: completed, independently audited and separately accepted descriptive
evidence. Primary reduction and independent audit passed on their first attempts.

The packet compares accepted saved scores from 2000 and 8000 residual
updates within each of parent seeds 1729, 1730 and 1731. It preserves all query
boundary measurements, four-category transition matrices, exact/separation
gains and losses, and per-parent/group descriptive threshold-interval bounds.
No threshold is selected and no alternative answer is generated.

Six complete seed/budget reports and three complete paired reports each retain
222 ordered queries. Their nine portable aliases are exact byte copies. The
2000 reductions must equal the previous accepted geometry diagnostic. Pooled
fresh-two/all-three counts reuse 222 shared questions; historical 1729 remains
development-selected. Pooled thresholds and cross-parent score averages are
excluded from the analysis.

The copied evidence binds the plan, primary/helper sources, independent
auditor/helper sources, synthetic tests, observed test and execution receipts,
accepted upstream JSON chains, completed diagnostic results and owner decision.
Raw full-score child reports are hash-bound local omissions. No upstream input
manifest is traversed recursively, and no weight payload, graph, training or
protected data is opened. Each included file is limited to 512 KiB.

The fresh-two transition matrix, in exact/missing/extra/overlap category order,
is [[420,0,0,0],[4,2,0,0],[8,0,1,0],[2,2,4,1]]. Fourteen exact gains and eight
separation gains have no corresponding losses. At 8000 updates the remaining
fresh errors are nine separated-but-wrong observations and one overlap/tie.
Including historical 1729 gives 13 separated-but-wrong and three overlap/tie.
Distinct questions failed by any seed decrease from 28 to 13.

No seed has a common threshold interval across its dual-TYPE queries at either
budget. Even fully separated seed 1730 at 8000 updates has A=-1.2183341979980469
and B=-0.09297823905944824, so B<A fails. This bound selects no threshold and
does not establish a deployable repair.

This diagnostic creates no checkpoint and leaves the count at 37. It does not
change a quality gate, default model, inference policy or serving support.

Read [paper 28](../../../papers/28-bilinear-budget-error-geometry.md) and
[Lesson 49](learning/49-how-error-boundaries-change-with-training.md). The
[primary summary](campaign/summary.json), [independent audit](campaign/independent-audit.json)
and [owner decision](campaign/decision.json) remain separate records. The
[portable index](experiments/2026-09-25-bilinear-budget-geometry.json) references
all nine full reports:

| Seed | 2000-update geometry | 8000-update geometry | Paired query changes |
| --- | --- | --- | --- |
| 1729, historical selected | [222 rows](experiments/2026-09-25-bilinear-budget-geometry-budget-2000-seed-1729.json) | [222 rows](experiments/2026-09-25-bilinear-budget-geometry-budget-8000-seed-1729.json) | [222 pairs](experiments/2026-09-25-bilinear-budget-geometry-paired-seed-1729.json) |
| 1730, fresh budget replication | [222 rows](experiments/2026-09-25-bilinear-budget-geometry-budget-2000-seed-1730.json) | [222 rows](experiments/2026-09-25-bilinear-budget-geometry-budget-8000-seed-1730.json) | [222 pairs](experiments/2026-09-25-bilinear-budget-geometry-paired-seed-1730.json) |
| 1731, fresh budget replication | [222 rows](experiments/2026-09-25-bilinear-budget-geometry-budget-2000-seed-1731.json) | [222 rows](experiments/2026-09-25-bilinear-budget-geometry-budget-8000-seed-1731.json) | [222 pairs](experiments/2026-09-25-bilinear-budget-geometry-paired-seed-1731.json) |

The [plan](experiments/2026-09-25-bilinear-budget-geometry-plan.md),
[model register](model-versions.md), [snapshot](snapshot.json),
[local dependency inventory](local-dependencies.json) and
[checksums](SHA256SUMS.txt) record scope, provenance and omissions. The
[documentation build receipt](campaign/docs-build-receipt.json) records the
normal MkDocs build separately from the scientific evidence checks.
