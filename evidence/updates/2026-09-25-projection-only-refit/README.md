# 2026-09-25: frozen-feature membership projection refit

This addition preserves an independently audited, single-seed training screen.
Only the symmetric membership projection is refit from the accepted seed-1729
parent; every other state tensor remains unchanged. Dense zero-threshold exact
answers improve from 112 to 122, but the stronger eight-branch comparator has
201. The fixed quality gate rejects the derivative. Lower training loss and
higher macro F1 do not replace the declared exact-answer requirements.

Read the [research synthesis](../../../papers/18-frozen-feature-projection-refit.md),
[frozen plan](experiments/2026-09-25-projection-only-refit-plan.md),
[portable result](experiments/2026-09-25-projection-only-refit.json),
[independent audit](campaign/independent-audit.json) and
[owner decision](campaign/decision.json). The result is evidence accepted,
quality rejected; no additional seed, protected-test prediction, generator
evaluation or serving promotion is included.

The 500 full-batch updates use 1,637 training queries. Parent replay and final
child evaluation cover 222 validation queries at the historical batch shapes.
All 93 state tensors are checked and only the projection changes. The final
checkpoint carries a new objective contract, separate from its parent's 2,000
training steps. Standard serving does not support the child objective.

The runner, recipe, tests and tracked archived-source fixture are copied with
their tested bytes. The fixture lets the synthetic archived-model tests run
without an ignored local source archive. The source/config snapshots, child
sidecar, training trace, audit implementation and test receipts are preserved.
These copies retain their original paths and identities inside the content;
an audit script's historical local path is not a promise of turnkey replay here.

`snapshot.json` maps every copied artifact to its source and SHA256.
`copied-files.json` records the initial frozen primary-evidence copy batch;
the final snapshot also includes the completed teaching and inventory copies.
`local-dependencies.json` binds omitted weight payloads, full parent/child head
reports, training membership labels and upstream inputs. Those files remain
local dependencies. This is not a self-contained neural reproduction bundle
or evidence of durable remote weight archival.

The synchronized 3.6429369-second refit is descriptive. Parent and child head
times have unequal warmup/cache conditions and establish no speedup. No
concurrency, energy or serving claim follows. Historical dated copies remain
unchanged. `SHA256SUMS.txt` covers every file in this addition except itself;
copied source notes may contain links that require the separate source repository.
