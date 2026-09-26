# 2026-09-25: worst-member bilinear residual screen

The independently audited screen rejects the new quality result: exact sets
increase from the balanced-BCE sibling's 207 to 208, but macro F1 decreases from
0.9997456354 to 0.9996521512 and single-TYPE exactness falls 50 to 49. These are
separate required checks; the one-answer gain cannot offset their failures.
All 222 outputs satisfy serialization bounds, and all primary execution invariants
pass. Evidence acceptance
preserves the result without promoting the child.

Read the [research synthesis](../../../papers/25-bilinear-worst-boundary.md),
[frozen declaration](experiments/2026-09-25-bilinear-worst-boundary-plan.md),
[portable result](experiments/2026-09-25-bilinear-worst-boundary.json),
[independent audit](campaign/independent-audit.json),
[owner decision](campaign/decision.json),
[full training trajectory](experiments/2026-09-25-bilinear-worst-boundary-training.json)
and [checkpoint inventory](experiments/2026-09-25-bilinear-worst-boundary-model-version-inventory.json).

The sole objective averages half the sum of softplus(-weakest true score)
and softplus(strongest false score) over training queries. Frozen masks exclude
the SAME subject and ignore control/padding labels. Masked amin/amax distributes
gradients over exact ties. Only A[2,256,256] changes; the 93 original tensors,
normalized product features, bilinear scorer and original parent remain fixed.
Fresh AdamW runs exactly 2000 full-batch updates. No mean-loss mixture, threshold
change, intermediate validation, budget extension or checkpoint selection occurs.

Worst loss falls 0.44132092595100403 to 0.003092526225373149. Separate no-gradient
mean BCE falls 0.003822767175734043 to 0.00016030404367484152, while the historical
balanced-BCE sibling finished 0.00003491543247946538. The diagnostic is measured
only at updates 0 and 2000 and never controls training or acceptance. No historical
control worst-loss endpoint was measured, so none is inferred.

The child gains 4 exact answers and loses 3 against the balanced-BCE sibling.
Groups COLOR/single TYPE/dual TYPE reach 103/49/56 against floors 103/50/54.
There are 5 extra and 15 missing member occurrences, versus 8/10; strict separation
falls 217 to 215. The earlier projection-only worst-member failure remains relevant
negative evidence. Equal optimization settings do not guarantee equal difficulty.

The primary passed 144 CPU tests (32 new/112 reused), and the independent auditor
passed 68 synthetic tests. Both actual processes completed on their first attempt.
The audit reconstructs saved predictions/metrics/pairing/gates and checks CPU
checkpoint payloads and optimizer/lineage receipts. It does not independently
replay CUDA fitting or regenerate neural scores.

The full 94-state child is registered as final checkpoint 34, including its new
objective/evaluator and separate unchanged scorer identity. All 34 final weight
files are freshly hash-verified. Parent pretraining and residual updates remain
separate. The inherited ordinary decoder ignores A and does not serve this scorer.

Exact copies include source and helper snapshots, recipes, portable tests,
actual/test receipts, declarations, comparisons, diagnostics, trajectory,
lineage and source teaching/register context. `copied-files.json` records the
first copy batch; `snapshot.json` also binds final teaching/register copies.
`local-dependencies.json` lists verified omitted weights, full score reports,
training membership and expanded runtime. Hashes are not durable remote archival.

`SHA256SUMS.txt` covers this addition except itself. Older evidence remains
unchanged. Descriptive local durations do not establish serving or energy gains.
No additional-seed replication, protected evaluation or policy/default promotion
follows from this failed adaptive single-seed gate.
