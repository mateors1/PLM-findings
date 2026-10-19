# 2026-09-25: symmetric bilinear residual refit

This independently audited single-seed screen improves dense exact answers from
112 to 198 out of 222, with macro F1 0.99940617. The fixed quality gate still fails:
the stronger eight-branch comparator has 201 exact answers, and both TYPE group
floors are missed. Evidence is accepted; the derivative is not promoted.

Read the [research synthesis](../../../papers/21-bilinear-residual-refit.md),
[frozen plan](experiments/2026-09-25-bilinear-residual-plan.md),
[portable result](experiments/2026-09-25-bilinear-residual-refit.json),
[independent audit](campaign/independent-audit.json),
[owner decision](campaign/decision.json) and
[checkpoint inventory](experiments/2026-09-25-bilinear-residual-model-version-inventory.json).

Only a new A[2,256,256] parameter trains; all 93 original tensors remain unchanged.
The full 94-entry derivative has explicit architecture/objective identity. Its
ordinary decoder forward does not evaluate the residual. No standard serving,
protected-test measurement, additional-seed replication or default change occurs.
The inventory records 30 final neural checkpoints without promising remote weights.

The exact runner, recipe, new and reused tests, portable archived runtime fixture,
test receipts, independently authored auditor and its authenticated helper are
copied unchanged. Campaign snapshots, zero-initialization replay hashes, loss
trace, run metadata, checkpoint sidecar, decision and execution receipts preserve
the evidence chain. Historical controls remain reused evidence. The portable
error histogram is explicitly descriptive post-hoc analysis of saved outputs.

`copied-files.json` records the initial copy batch; `snapshot.json` adds the final
lesson and model register and binds every copied source. `local-dependencies.json`
lists verified omitted inputs, including raw weights, full parent/child head
vectors, training membership and expanded runtime. These hashes establish local
identity, not durable remote availability. Absolute paths inside copied artifacts
remain historical identities, not a turnkey replay promise.

The auditor reconstructs selections, metrics, labels and gate from saved evidence
and checks checkpoint tensors/optimizer on CPU. It does not independently repeat
CUDA training or regenerate neural scores. Fitting time is descriptive, not a
serving throughput or energy result. Architecture and optimization geometry both
change; this screen does not isolate a causal capacity effect.

`SHA256SUMS.txt` covers this addition except itself. Earlier evidence remains
unchanged. Copied source documents may refer to the separate source repository.
