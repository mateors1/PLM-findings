# 6. Academic model identities and the next application hypothesis

**Update: 2026-09-25.** This adds provenance and a future research direction to
the initial snapshot. It does not report a newly trained model, a new quality
result or acceptance of the pending composition integration.

## A model version needs more than a package number

The source project's package version does not distinguish its trained weights.
For academic reporting, keep four identities separate:

| Identity | What a change means |
| --- | --- |
| Model recipe | Architecture, objective or training configuration changes |
| Checkpoint | A particular run, seed, step and exact saved artifact |
| Inference policy | Different constraints, guidance or set selection with potentially unchanged weights |
| Evaluation | A different plan, source, split, numerical setup or comparison contract |

For example, the four-path selector, pair composition and direct-membership
ablation reuse the same three control checkpoints. Their results concern
different inference policies. The shape-aware v2 verifier similarly changes
the comparison contract; it is not a new trained model and cannot erase the v1
failure. See the copied [version register](../evidence/updates/2026-09-25/model-versions.md)
and [verification lesson](../evidence/updates/2026-09-25/learning/29-shape-aware-verification.md).

## Historical artifact inventory

A new stdlib-only inventory discovered **23 final checkpoints** in immediate
`runs/*/run.json` directories. All 23 passed byte-hash and receipt-consistency
checks: checkpoint hashes, recorded training configurations, model metadata,
seeds, steps, corpus/split identities, source archives, archived source trees
and dependency locks. The [complete report](../evidence/updates/2026-09-25/experiments/2026-09-25-model-version-inventory.json)
retains each run ID and full historical configuration. Sixteen focused CPU tests
passed, and independent source review corrected the sidecar schema comparison
before execution. The [producing script](../evidence/updates/2026-09-25/scripts/inventory_model_versions.py)
is copied with its hash bound in the report.

This inventory does **not** deserialize checkpoint payloads, check every tensor,
rerun models, revalidate corpus files or establish quality. It excludes periodic
checkpoints and nested scratch runs. The recorded training receipts do not
establish explicit parent-checkpoint lineage, which remains unknown rather than
being guessed from run names.

The three checkpoints currently used for composition are seed replicas at step
2000. Their full hashes appear in the register. A paper should identify the run
and checkpoint hash together with the inference policy and evaluation report
hash, rather than cite an unqualified "latest model".

These copies contain provenance records, not the weights themselves. A checksum
identifies an artifact but does not make it retrievable or backed up. Durable
checkpoint/source/data archival, complete parent lineage and links from every
historical variant to its evaluation remain academic-release work.

## Learned relational retrieval is a hypothesis, not a present application claim

The strongest proposed application is an identifier-only retrieval component for
relations inferred from observations or judgments. The current TYPE/COLOR SAME
task is still Experiment A: its compiler provides a perfect oracle. Withholding
some known labels does not change that rule into independently demonstrated
compatibility or preference learning.

Experiment B should use independently meaningful labels and compare PLM with
simple lookup/popularity, neighborhood retrieval and learned ranking or
factorization baselines. The direct membership head is also an important
ablation: the decoder must earn its additional cost. All methods need comparable
information, candidate universes, tuning budgets and output constraints.
The [declared hypothesis](../evidence/updates/2026-09-25/experiments/2026-09-25-learned-retrieval-hypothesis.md)
records these requirements; no dataset or new mode is enabled by it.

Complementary compatibility is a candidate family. Co-occurrence alone does
not prove compatibility, missing observations are not automatically negatives,
and sparse positive labels may not support complete-set exactness. Split design
must prevent leakage and distinguish seen-entity generalization from cold start.
An ID-only model cannot infer a new item's semantics from an arbitrary new ID.

The falsifiable question is whether PLM yields better held-out utility, or
comparable utility at better measured cost, than the simpler methods. If a small
ranker matches it more cheaply, the decoder application has not earned its
complexity. This outcome would still be a useful research finding.

## Integration work remains open

The [shape-aware contract](../evidence/updates/2026-09-25/experiments/2026-09-25-pair-composition-shape-aware-plan.md)
requires fresh serial references for all three checkpoints, then exact
same-shape score comparisons and unchanged cross-shape decisions. It explicitly
reuses 222 saved HTTP responses and requires 444 fresh responses, all auxiliary
contracts and an independent audit. Implementation and CPU checks are progress
toward that gate; they are not completion of the GPU/HTTP campaign.

The existing offline quality remains **569/666 exact query-seed observations**.
No new protected-test, oracle-parity, energy or learned-relation result is added
by this update.
