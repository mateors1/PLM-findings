# 2026-09-25: fixed 2000-update bilinear budget screen

This independently audited single-seed screen passes the unchanged quality gate:
207/222 exact answers, macro F1 0.9997456354, group exact counts COLOR103,
singleTYPE50 and dualTYPE54, and all 222 outputs within the serialization bound.
Evidence and the single-seed screen are accepted; no serving policy or default
is promoted. A replication would require a separate declaration.

Read the [research synthesis](../../../papers/22-bilinear-training-budget.md),
[frozen plan](experiments/2026-09-25-bilinear-budget2000-plan.md),
[portable result](experiments/2026-09-25-bilinear-budget2000.json),
[independent audit](campaign/independent-audit.json),
[owner decision](campaign/decision.json) and
[checkpoint inventory](experiments/2026-09-25-bilinear-budget2000-model-version-inventory.json).

Only the update budget changes relative to the historical500 recipe. The new
fit starts from the original parent with zero A and fresh AdamW state; it never
resumes the historical derivative. All 93 parent tensors remain unchanged, and
only A[2,256,256] trains for 2000 full-batch updates. A fresh full94-state reload
matches the trained tensors and optimizer. Parent pretraining2000 and residual
fitting2000 remain separate lineage fields. The inventory now records31 final
checkpoints without promising durable remote weight storage.

The longer fit gains10 exact answers and loses1 relative to the saved500-update
output, and gains13/loses7 relative to the stronger eight-branch selector. Raw
historical output is authenticated, not freshly rerun or a timing control.
The previously rejected500 gate remains unchanged. Fifteen answers remain
nonexact; this is not oracle parity or protected-test evidence.

Exact new/reused runner, scorer, recipes, portable tests/runtime fixture, test
receipts, independent auditor and its authenticated helpers are copied unchanged.
The bundle preserves campaign source snapshots, zero replay, training/run/sidecar,
comparisons, decisions and actual execution/audit receipts. The saved-output error
histogram is explicitly descriptive post-hoc evidence, not a tuning criterion.

`copied-files.json` records the initial copy batch; `snapshot.json` adds the final
lesson and model register and binds every copied source. `local-dependencies.json`
lists verified omitted inputs, including weights, full parent/child and historical
head vectors, training membership and expanded runtime. Absolute paths in copied
artifacts remain historical identities, not a turnkey replay promise.

The runner passed91 focused CPU tests and the independent auditor passed37
synthetic tests. Audit checks saved selections, metrics, split/labels, checkpoint
state and optimizer evidence on CPU; it does not independently repeat CUDA
training or regenerate neural logits. Timings are descriptive local durations,
not a serving throughput or energy claim. Adaptive validation limits broader
claims of convergence, generalization and independent confirmation.

`SHA256SUMS.txt` covers this addition except itself. Earlier evidence remains
unchanged. Copied source notes may refer to the separate source repository.
