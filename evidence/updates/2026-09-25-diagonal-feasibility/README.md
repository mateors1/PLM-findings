# 2026-09-25: frozen diagonal-head feasibility diagnostic

This addition preserves an independently audited diagnostic with an inconclusive
mathematical outcome. It tests the exact-real diagonal feature family of the
original seed-1729 parent against its training labels, without neural updates,
validation predictions or protected-test predictions. The final neural checkpoint
count remains 29; no serving policy or default is promoted.

TYPE covers 805 training queries and 824,320 nonself membership constraints;
COLOR covers 832 queries and 851,968 constraints. Exact rank two of the steering
vectors permits separate relation-vector problems. Each dimension runs one
2,048-row primal LP, then checks all its training signs exactly. Both stop because
an exact failure has a nonpositive FP64 inequality residual. Neither times out;
no dual LP, feasible witness or infeasibility certificate is produced. This
establishes neither feasibility nor a capacity limitation.

Read the [research synthesis](../../../papers/20-frozen-diagonal-feasibility.md),
[frozen plan](experiments/2026-09-25-diagonal-feasibility-plan.md),
[portable result](experiments/2026-09-25-diagonal-feasibility.json),
[independent audit](campaign/independent-audit.json) and
[owner decision](campaign/decision.json). Evidence acceptance is separate from
the diagnostic outcome; this was not a validation quality screen.

The exact runner, isolated solver, recipe, Python project/lock, tests and frozen
test receipts are preserved. The bundle includes saved numerical proposals,
exact-check reports, traces, owned-process receipts, extraction metadata, actual
audit receipt and source/config/helper snapshots. The mean-refit feature helper
is explicitly copied with its authenticated identity. Historical absolute paths
inside copied files remain evidence identities, not a turnkey replay promise.

`copied-files.json` records the initial frozen evidence copy batch; `snapshot.json`
adds the final teaching lesson and model register and maps every source copy to
its SHA256. `local-dependencies.json` records verified omitted dependencies,
including raw features.npz, neural weights, training membership labels and the
expanded runtime. No raw normalized embeddings or weight payloads are published
here. Hashes establish identity, not durable remote availability.

The auditor reconstructs the saved training-label and exact-sign evidence; it
does not independently run CUDA normalization or the LPs. Its integer matrix
checks share the pinned FLINT library. Timing/options are authenticated execution
receipts, not independently reproduced chronology. Exact-real constants do not
establish a bound on FP32 rounding-dependent behavior or held-out quality.

`SHA256SUMS.txt` covers all files in this addition except itself. Previous bundles
remain unchanged. Copied source notes may link into the separate source repository.
