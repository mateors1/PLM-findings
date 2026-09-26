# Canonical paper history and evidence index

The repository root [`PAPER.md`](../PAPER.md) is the findings projection of the source repository canonical
scientific synthesis, revision `PAPER-v0001`. This index keeps paper revisions distinct from model,
checkpoint, inference, evaluation, and publication identities. Historical
evidence stays at its existing immutable repository path; this index links to
it and does not move or rewrite source files.

## Paper revisions

| Revision | Date | Predecessor | Status | Preserved copy |
| --- | --- | --- | --- | --- |
| `PAPER-v0001` | 2026-09-25 (America/Bogota) | None; initial revision | Current; evidence cutoff 2026-09-26 03:12 UTC | Root `PAPER.md`; no earlier paper existed to archive |

Before replacing this revision, copy its exact bytes to a unique file such as
`PAPER-v0001-2026-09-25.md` in this directory, verify the archived checksum,
and update this table. Never overwrite an archived revision. Each later root
revision must carry its own version, date, predecessor, and evidence cutoff.

## Retained model and evidence map

| Topic | Status in `PAPER-v0001` | Source evidence |
| --- | --- | --- |
| Pokémon-only protocol, hidden graph attributes, `TYPE/COLOR SAME` | Retained scope; the graph compiler is the exact v1 oracle | [`protocol-spec.md`](../evidence/source/protocol-spec.md), [`data/evaluation`](../evidence/learning/02-data-and-evaluation.md), [`research_log.md`](../evidence/source/research_log.md) |
| Prompt-set supervision | Retained in the reference checkpoint recipe; three-seed validation gain, not generalization | [`seed replication report`](../evidence/experiments/2026-09-24-seed-replication.json), [`Lesson 6`](../evidence/learning/06-seed-replication.md) |
| Symmetric product-membership head and four-path selector | Implemented and used by the application-reference experiments; selection remains opt-in | [`symmetric relation lesson`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/11-symmetric-relations.md), [`set-selection integration`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/24-integrating-set-selection.md), [`model register`](../evidence/updates/2026-09-25-bilinear-budget8000-replication/model-versions.md) |
| Pairwise unions of candidate sets | Accepted opt-in policy; 569/666 exact on the three-seed fixed validation split; no oracle parity | [`Lesson 25`](../evidence/learning/25-composing-candidate-sets.md), [`shape-aware v2 receipt`](../evidence/updates/2026-09-25-composition-verification/experiments/2026-09-25-pair-composition-shape-aware.json), [`Lesson 29`](../evidence/updates/2026-09-25-composition-verification/learning/29-shape-aware-verification.md) |
| Bilinear 8,000-update residual replication | Accepted validation candidate; not retained or supported by ordinary serving | [`replication plan`](../evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication-plan.md), [`portable result`](../evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication.json), [`audit repair`](../evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-replication-evidence-repair.md), [`Lesson 48`](../evidence/updates/2026-09-25-bilinear-budget8000-replication/learning/48-testing-a-budget-across-parent-seeds.md), [`model register`](../evidence/updates/2026-09-25-bilinear-budget8000-replication/model-versions.md) |
| Native PLM versus SQLite oracle | Contrary, nonmatched-quality serial serving result; graph exactness and speed lead | [`portable report`](../evidence/updates/2026-09-25-canonical-paper-v0001/experiments/2026-09-25-native-graph-oracle.json), [`plan`](../evidence/updates/2026-09-25-canonical-paper-v0001/experiments/2026-09-25-native-graph-oracle-plan.md) |

## Superseded methods, negative results, and contrary evidence

The canonical paper keeps only short explanations relevant to the current
model. These original records preserve the longer reasoning, failed gates, and
superseded implementation details:

| Historical method or result | Why it is not presented as the retained model | Preserved rationale and evidence |
| --- | --- | --- |
| Initial dense/MoE and first-target-only trials | Weak exact-set quality or no retained quality benefit | [`architecture experiments`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/03-architecture-experiments.md), [`first quality campaign`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/04-first-quality-campaign.md), [`first-target guidance`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/15-first-target-guidance.md) |
| Prompt-set and symmetric-head development | Earlier recipes are provenance for the current recipe, not independent data or a parity result | [`prompt supervision`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/05-prompt-supervision.md), [`symmetric-relation replication`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/12-symmetry-replication.md) |
| Wider search, source intersections, coverage-seeking, direct-membership ablation | Changed candidate availability or set quality unevenly; some fixed gates failed | [`wider first-choice search`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/35-wider-first-choice-search.md), [`set algebra`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/36-unions-intersections-and-recoverability.md), [`coverage branch`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/38-a-branch-for-uncovered-products.md), [`direct-membership ablation`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/28-direct-membership-ablation.md) |
| Margin, projection-only, worst-boundary, and short bilinear fits | Completed negative or mixed screens; not promoted | [`hardest-boundary margin`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/31-hardest-boundary-margin.md), [`weak-margin result`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/33-loss-weight-is-not-gradient-strength.md), [`projection refit`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/39-refitting-a-frozen-feature-head.md), [`worst-boundary projection`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/40-training-the-weakest-membership-decisions.md), [`bilinear refit`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/42-learning-cross-coordinate-relations.md), [`bilinear worst-boundary`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/46-changing-which-mistakes-drive-learning.md) |
| Single-seed longer balanced-BCE fit and two-parent replication | Positive validation evidence for an experiment-local residual scorer; it does not alter normal decoder forward, standard serving, or promotion state | [`single-seed screen`](../evidence/updates/2026-09-25-bilinear-budget8000/experiments/2026-09-25-bilinear-budget8000.json), [`Lesson 47`](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition/blob/cca97ffc5491eb1cfa28ac94532d34460a75812e/docs/learning/47-more-updates-with-the-same-objective.md), [`replication result and repair`](../evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication.json), [`Lesson 48`](../evidence/updates/2026-09-25-bilinear-budget8000-replication/learning/48-testing-a-budget-across-parent-seeds.md) |
| Verifier failure during bilinear replication | The first auditor failed before checkpoint loading; versioned repairs audited the saved artifacts without retraining | [`evidence-repair declaration`](../evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-replication-evidence-repair.md), original and v2 receipts in the replication report |

The [model version register](../evidence/updates/2026-09-25-bilinear-budget8000-replication/model-versions.md) remains the detailed
checkpoint inventory. Checkpoint hashes establish artifact identity, not
durable storage or public availability. The 8,000-update candidate's results
are reported separately from the application-reference checkpoints and the
default serving policy.

## Findings projection provenance

This root paper carries the same scientific revision as the source repository's
canonical `PAPER-v0001`. Its relative citations were adapted to the evidence
paths below; the complete PLM source tree is not duplicated here.

- Source `PAPER.md` SHA256: `d8746165b57acd5d40613109f47945a1265c5aa0e5d916b2b0a6e49c0bacfa39`
- Findings projection `PAPER.md` SHA256: `a1c19721256990c54e948c7c5bfb4244b361eb0c0085cf67b2556d7bbbead5cb`
- Implementation code snapshot: source commit `cca97ffc5491eb1cfa28ac94532d34460a75812e`, copied under
  [`evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/`](../evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/README.md)
- Native oracle report and plan: copied under
  [`evidence/updates/2026-09-25-canonical-paper-v0001/`](../evidence/updates/2026-09-25-canonical-paper-v0001/experiments/2026-09-25-native-graph-oracle.json)

The publication receipt is authoritative for source and findings commit SHAs
and push status. A ready packet alone does not prove publication.
