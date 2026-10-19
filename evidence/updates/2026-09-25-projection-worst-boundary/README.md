# 2026-09-25: worst-boundary projection refit

This independently audited single-seed screen replaces the mean-BCE projection
objective with the average of each query's worst-positive and worst-negative
softplus penalties. It starts from the original accepted seed-1729 parent.
Only the membership projection changes across 500 full-batch updates; it does
not continue the previous rejected child.

Dense exact answers improve from 112 to 128 (the mean-BCE sibling reached 122),
but the accepted eight-branch predictor has 201. The unchanged exact/group gates
reject the new derivative despite higher macro F1. Strict head separation falls
from 165 to 160. Evidence acceptance and quality acceptance remain separate.

Read the [research synthesis](../../../papers/19-worst-boundary-projection-refit.md),
[frozen plan](experiments/2026-09-25-projection-worst-boundary-plan.md),
[portable result](experiments/2026-09-25-projection-worst-boundary.json),
[independent audit](campaign/independent-audit.json) and
[owner decision](campaign/decision.json).

The optimized worst-member training loss decreases while separately measured
mean training BCE increases. These are different objectives; validation exactness
uses a different partition again. No causal explanation or capacity-impossibility
claim follows. There is no protected-test evaluation, extra-seed replication,
generation evaluation, serving promotion or matched-quality speed claim.

The new runner, recipe and tests retain their tested bytes. The reused mean-refit
runner, recipe and lifecycle tests are copied explicitly; the required archived
runtime fixture is included. The reused independent auditor implementation is
preserved under dependencies. This makes helper provenance explicit without
changing any historical file. Local paths inside those archived files remain
historical identities, not a promise of turnkey replay from this repository.

`snapshot.json` lists every copied source artifact and its SHA256.
`copied-files.json` records the first frozen evidence batch. The final snapshot
also includes the completed lesson and model-register inventory.
`local-dependencies.json` binds omitted artifacts, including weight payloads,
full parent/child head reports and training membership labels. These remain
local dependencies; a hash or sidecar does not prove durable remote weight
archival. Authenticated historical comparator records are reused, not new control
executions. Source notes can contain links requiring the separate source repository.

`SHA256SUMS.txt` covers every file in this addition except itself. Previous dated
bundles remain unchanged. The 4.2098359-second refit is descriptive, not a serving,
energy or throughput benchmark.
