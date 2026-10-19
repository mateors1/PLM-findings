# Shared entity offsets close the observed development gap

Campaign: `bilinear-entity-offset32000-v1`. Research date: 2026-10-18.
Status: completed and accepted after independent audit and review. This findings
note is local; publication is pending a separate publication receipt.

Learning shared entity offsets improved matched fresh controls from **660/666
to 666/666 exact development outcomes**, with six gains and no losses. Every
seed and its COLOR, single-TYPE and dual-TYPE groups passed the predeclared
exact-count and macro-F1 nonregression gates. This carries the intervention
forward as a quality candidate; it does not promote a model.

| Parent seed | Fresh control exact sets | Offset treatment exact sets | Gains | Losses |
| --- | ---: | ---: | ---: | ---: |
| 1729 | 218/222 | 222/222 | 4 | 0 |
| 1730 | 222/222 | 222/222 | 0 | 0 |
| 1731 | 220/222 | 222/222 | 2 | 0 |
| Repeated seed/query outcomes | **660/666** | **666/666** | **6** | **0** |

Each treatment has macro F1 1.0 and zero false-positive or false-negative
memberships. These are three evaluations of the **same 222 adaptively reused
development queries**, not 666 independent examples. The **191 protected
queries remain closed**. The entity universe is already present among training
candidates: this tests held-out queries over known entities, not unseen entities.

## What changed, and why it might help

The control can change how fixed entity features interact. The treatment can
also move each entity's features inside that interaction. Let frozen features
be `E: [1025,256]`, one 256-dimensional row per entity. Each dimension (TYPE or
COLOR) gets a learned matrix in `A: [2,256,256]`. Treatment additionally learns
`DeltaE: [1025,256]`:

```text
S_d = (A_d + A_d.T) / 2
control:   z(s,i,d) = z0(E)(s,i,d) + E_s.T S_d E_i / 16
treatment: F = E + DeltaE
           z(s,i,d) = z0(E)(s,i,d) + F_s.T S_d F_i / 16
```

Here `s` is the subject, `i` a candidate and `d` the selected dimension.
Symmetrizing the matrix makes the residual symmetric in subject and candidate.
The same F serves both roles and both dimensions, without renormalization.
The parent score `z0(E)` always uses the frozen E; all 93 original state entries
remain unchanged according to execution checks. Control learns 131,072 scalars;
treatment learns 393,472. Both A and DeltaE start at zero. Because A initially
vanishes, the first offset gradient is zero; later updates activate it.

Each arm starts afresh from the same seed's original 2,000-step parent and an
empty optimizer. Six sequential fits each perform 32,000 full-batch updates on
the same ordered 1,637 training queries, using balanced binary cross-entropy
(BCE), FP32 AdamW, learning rate 0.0003 and zero weight decay. There is no early
stopping, validation threshold tuning or new budget search. Both arms already
reach 1,637/1,637 exact training sets at the fixed 8,000- and 32,000-update
observations; the observed gain concerns development quality.

Prediction selects every nonself entity whose score is strictly positive,
in ascending identifier order. It uses no oracle answer size, truncation,
fallback or per-query repair. The quality gate required nonregression in exact
count and macro F1 for every seed and group, plus positive pooled paired exact
gain. All six improvements concern dual-TYPE queries.

## What was verified

The frozen readiness record binds source, recipe, inputs and **234 passing
synthetic tests with zero skips**. All six production fits completed their
32,000 updates and exited successfully. Each was audited and accepted before
the next began. Captured checks cover finite scores, losses, gradients,
parameters and Adam moments, original-state preservation, and exact model and
optimizer save/reload into a fresh factory before final evaluation.

The independent saved-evidence auditor reconstructs selections, metrics,
groups, training observations, paired changes and the aggregate gate. It does
not reopen checkpoint payloads or rerun neural inference. Weight preservation,
gradients and reload correctness therefore remain execution evidence, distinct
from independently reconstructed answer quality.

Fresh controls reproduce all historical selected sets and quality metrics,
but **not the complete floating-point trajectories**:

| Seed | Unequal pre-update losses / 32,000 | Unequal final scores / 227,550 | Maximum absolute score difference |
| --- | ---: | ---: | ---: |
| 1729 | 7,646 | 91,983 | 0.0000152587890625 |
| 1730 | 7,564 | 92,041 | 0.0000152587890625 |
| 1731 | 7,696 | 90,691 | 0.0000152587890625 |

Final residual-state hashes also differ from historical states; final
post-update losses match. The declaration treats those differences as
descriptive and gates treatment on behavioral reproduction. The scientific
comparators are the fresh controls. Exact save/reload within each new fit
is a separate check and passed in every arm.

## Interpretation and limits

The intervention passes its fixed development-quality gate across three parent
seeds. This does not establish the cause of the six recoveries or protected-test
oracle parity. The [reverse-pair coverage diagnostic](2026-10-18-bilinear-pair-coverage.md)
remains relevant to the query-holdout interpretation. Feature and matrix norms
are descriptive: scaling F by c and A by 1/c^2 leaves the residual unchanged.

The six new checkpoints remain local and are not admitted by existing serving
manifests. This experiment provides no new latency, throughput, energy or maximum
capacity measurement; earlier serving results cannot be transferred to these
weights. Checkpoint hashes establish identities, not remote backup or public
availability. Model/checkpoint, evaluator, inference and publication identities
remain separate. Serving integration requires its own declaration and checks.

## Source evidence and identities

The following links resolve in the paired local checkouts. They identify source
artifacts rather than claiming an already published evidence bundle. SHA-256
values bind the exact reviewed bytes:

| Artifact in the source repository | SHA-256 |
| --- | --- |
| [Report](../../Plm-protocolized-language-model-pika-edition/docs/experiments/2026-10-18-bilinear-entity-offset-report.md) | `0feac0394f041f0669c567f86ea3acae938cfcec4dfa0a4ccd0aca32be66e2d7` |
| [Frozen plan](../../Plm-protocolized-language-model-pika-edition/docs/experiments/2026-10-18-bilinear-entity-offset-plan.md) | `fea6a0999c8c3fb33ceadab1eab48b585b92924da5e7f5958bc6f799bb3415d3` |
| [Portable evidence](../../Plm-protocolized-language-model-pika-edition/docs/experiments/2026-10-18-bilinear-entity-offset-evidence.json) | `39c8a9b567d7b7b45337e9be0ee80bfe9d15f4105430a5a59f3c4760eefbb5b3` |
| [Independent aggregate audit](../../Plm-protocolized-language-model-pika-edition/docs/experiments/2026-10-18-bilinear-entity-offset-audit.json) | `d0b7e9ab0e52358d38e15c8d9ddd2c281a2d3a00e431d21430f0168a06c3456e` |
| [Independent review](../../Plm-protocolized-language-model-pika-edition/docs/experiments/2026-10-18-bilinear-entity-offset-review.json) | `fb0ca6fcaafbb395d77faa0a07dad7cf3d39ab93c383ed30cba36f6d4f9afd4d` |
| [Owner decision](../../Plm-protocolized-language-model-pika-edition/docs/experiments/2026-10-18-bilinear-entity-offset-decision.json) | `9e04fbc146702971e716e1ef83e3e7a1cfcaf5947d5dfc2af37aface2cb3847b` |

The owner decision accepts the experiment and fixed quality gate; the summary
and audit intentionally do not self-authorize acceptance. The decision's
`publication_status: held` records its decision-time status. Current publication
coordination is separate and pending a publication receipt, not a permanent hold.
No weights, raw run outputs or private records are included in this note.
