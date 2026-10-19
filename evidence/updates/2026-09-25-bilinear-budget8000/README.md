# 2026-09-25: balanced-BCE bilinear 8000-update screen

The independently audited screen passes the declared quality gate: 216/222 exact
sets, macro F1 0.9998843626799785 and group exact counts COLOR 103, single-TYPE 51,
dual-TYPE 62. Against the saved 2000-update sibling, nine answers improve and none
regress. All six remaining nonexact answers are dual-TYPE. There is one false
positive and eight false negatives; strict separation holds for 220 queries.
All 222 outputs satisfy serialization bounds. Owner acceptance does not promote
ordinary serving or authorize protected evaluation.

Read the [research synthesis](../../../papers/26-bilinear-budget8000.md),
[frozen plan](experiments/2026-09-25-bilinear-budget8000-plan.md),
[portable result](experiments/2026-09-25-bilinear-budget8000.json),
[independent audit](campaign/independent-audit.json),
[owner decision](campaign/decision.json),
[teaching note](learning/47-more-updates-with-the-same-objective.md)
and [checkpoint inventory](experiments/2026-09-25-bilinear-budget8000-model-version-inventory.json).

This is a fresh fit from the original seed-1729 parent and zero A[2,256,256],
not a resume from either trained sibling. The 93 parent tensors, scorer, balanced
BCE and optimizer settings stay fixed. Exactly 8000 full-batch updates use the
1637 training queries. Zero-start replay precedes fitting; only the final fitted
endpoint is validated. The output rule remains nonself scores strictly above
zero, sorted by identifier, without repair, truncation or threshold selection.

Training BCE falls from 0.003822767175734043 to 0.0000016234706663453835.
The first 2000 pre-update scalar losses match the historical values exactly;
old final post-update 2000 equals new pre-update 2001 at
0.00003491543247946538. This descriptive check neither proves intermediate-weight
identity nor changes the gate, training or retry policy.

The full trace is retained in four exact chunks: [1-2000](experiments/2026-09-25-bilinear-budget8000-training-part-01.json),
[2001-4000](experiments/2026-09-25-bilinear-budget8000-training-part-02.json),
[4001-6000](experiments/2026-09-25-bilinear-budget8000-training-part-03.json)
and [6001-8000](experiments/2026-09-25-bilinear-budget8000-training-part-04.json).
Each binds its original local training-file hash and update range. The builder
verifies exact concatenation, preserving all values without downsampling.

The primary passed 129 CPU tests (38 new, 91 reused), and the independent auditor
passed 70 synthetic tests. Both actual processes completed on their first attempt.
The auditor independently checks saved predictions, metrics, pairing, gates,
checkpoint tensors and optimizer lineage. It does not repeat CUDA training or
regenerate neural scores.

The child is registered as final checkpoint 35. All 35 final weight files are
freshly hash-verified; prior payloads were not revalidated in that inventory pass.
Checkpoint, inference, evaluator and publication identities remain separate.
Parent pretraining is 2000 steps; residual training is 8000 updates. The inherited
ordinary decoder ignores A and does not implement this scoring policy.

Exact copies include declarations, source/helper snapshots, recipes, portable
tests, actual/test receipts, comparisons, lineage, four training chunks and source
teaching/register context. `copied-files.json` records the first copy batch;
`snapshot.json` also binds teaching/register copies and the trace reconstruction.
`local-dependencies.json` lists authenticated omitted weights, full raw reports,
the oversized raw training file, membership and expanded runtime. Hashes do not
establish durable remote archival.

`SHA256SUMS.txt` covers this addition except itself. Older evidence remains
unchanged. This adaptive single-seed validation result does not establish oracle
parity, convergence, new-seed replication or serving/energy superiority.

Documentation validation is separate from the scientific gate. The ordinary
MkDocs build passed; strict mode failed on six pre-existing links to source or
serving files outside the docs tree. The [documentation receipt](campaign/documentation-build-receipt.json)
binds both [strict output](campaign/mkdocs-stdout.txt) and
[ordinary output](campaign/mkdocs-normal-stdout.txt).
