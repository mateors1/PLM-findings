# 2026-09-25: bilinear parent-seed replication

Both fresh fits pass their matching frozen quality gates: seed 1730 reaches
212/222 exact sets and seed 1731 reaches 208/222, versus own eight-branch counts
205 and 197. The separate owner decision accepts the audited evidence and quality
screen without promoting a policy. Fresh-two exactness is 420/444; all-three
exactness 627/666 reuses seed 1729 as historical development/selection evidence.
All observations concern 222 shared validation queries, not 666 independent tests.

Read the [research synthesis](../../../papers/23-bilinear-parent-seed-replication.md),
[frozen plan](experiments/2026-09-25-bilinear-seed-replication-plan.md),
[portable result](experiments/2026-09-25-bilinear-seed-replication.json),
[recovered aggregate](campaign/aggregate-v2.json),
[independent audit](campaign/independent-audit.json),
[owner decision](campaign/decision.json) and
[checkpoint inventory](experiments/2026-09-25-bilinear-seed-replication-model-version-inventory.json).

The recipe is unchanged: each original parent keeps all 93 tensors frozen and
fits fresh zero-initialized A[2,256,256] for 2000 full-batch AdamW updates.
Own-parent initial losses and zero-residual replay are checked separately.
Seed1730 has eight extra/five missing members; seed 1731 has 12 extra/ten missing.
Twenty-four fresh complete answers remain incorrect, all in dual TYPE. This is
below oracle parity despite fresh pooled macro F1 above 99.97%. No protected test or serving
measurement follows from the gate pass.

Original failed aggregate and audit evidence remain intact. An
[evidence repair declaration](experiments/2026-09-25-bilinear-replication-evidence-repair.md)
adds a separately versioned aggregate-only runner and auditor v2. Equivalent
aware timestamps now compare by exact instant, preserving fractional precision;
the auditor reads original-parent metadata from its authenticated sidecar.
Both fits, predictions and quality contracts remain unchanged. No GPU retraining
occurs during repair. New aggregation/auditor identities remain separate from
executed seed runner, scorer, evaluator and checkpoint identities.

The seed runner passed 136 focused CPU tests. Recovery aggregation passed 58
(13 new/45 reused), and auditor v2 passed 82 synthetic tests before actual audits.
Both per-seed and aggregate audits completed. Independent saved-set arithmetic
and CPU checkpoint checks do not independently regenerate neural logits, loss
trajectories, reload observations or GPU execution chronology.

Both full 94-state children are registered, bringing final-checkpoint count 31
to 33. All 33 final weight files were freshly rehashed. Historical 1729 was reused,
not trained again. Parent pretraining and residual-update counts stay separate.
Full trajectories are retained in exact training.json copies and portable
[1730](experiments/2026-09-25-bilinear-seed-replication-training-1730.json) and
[1731](experiments/2026-09-25-bilinear-seed-replication-training-1731.json) files.

The bundle copies exact original/recovery sources, helper scripts, tests,
recipes, declarations, snapshots, successful/failed receipts, sidecars and
versioned decision/register helpers. `copied-files.json` lists the first copy
batch; `snapshot.json` also binds the final lesson and model register.
`local-dependencies.json` verifies omitted weights, full head vectors,
training membership and runtime files. Absolute paths remain historical
identities rather than a turnkey replay promise. Local hashes are not durable
remote weight archival.

`SHA256SUMS.txt` covers this addition except itself. Earlier bundles remain
unchanged. Primary wall times are descriptive local durations; no concurrent
serving, energy, protected-test or generalization advantage is established.
