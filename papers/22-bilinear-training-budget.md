# A longer fixed fit passes the single-seed quality screen

**Status: independently audited evidence accepted; single-seed quality screen passed.**
The fixed 2,000-update bilinear fit produces 207/222 exact validation sets and
99.9746% macro F1. All declared primary checks pass, including the requirement
to exceed the stronger selector's 201 exact answers without regressing its
TYPE/COLOR group counts. The independent saved-evidence audit passes, and the
separate owner decision accepts the evidence and quality result without promoting
a serving policy or changing defaults.

The [frozen plan](../evidence/updates/2026-09-25-bilinear-budget2000/experiments/2026-09-25-bilinear-budget2000-plan.md)
changes only the declared optimization budget relative to the
[500-update screen](21-bilinear-residual-refit.md). Its earlier failed gate
remains part of the record. No threshold, architecture or quality requirement
was adjusted after observing the new result. The
[portable result](../evidence/updates/2026-09-25-bilinear-budget2000/experiments/2026-09-25-bilinear-budget2000.json),
[independent audit](../evidence/updates/2026-09-25-bilinear-budget2000/campaign/independent-audit.json)
and [owner decision](../evidence/updates/2026-09-25-bilinear-budget2000/campaign/decision.json)
record the separate evidence and quality decisions.

## Giving the same head more updates

The earlier run's training loss was still decreasing near update 500. That
motivated testing more optimization, without establishing that validation
quality would improve. The new experiment declares 2,000 full-batch updates
before execution and restarts from the same original seed-1729 parent with
an exactly zero residual and fresh AdamW state. It does not resume the trained
500-update derivative.

All 93 original state tensors remain frozen. The only trainable parameter is
A[2,256,256], with one matrix each for TYPE and COLOR. Frozen, scaled normalized
product embeddings have shape E[1025,256]. Scores retain the same computation:

\[
M_d=\frac{A_d+A_d^\top}{2},\qquad
z_{sdi}=z^{\mathrm{parent}}_{sdi}+\frac{E_s^\top M_dE_i}{16}.
\]

Each full training batch contains 1,637 queries and produces logits
Z[1637,1025]. The same balanced binary cross-entropy loss averages positive
and nonself-negative penalties equally within each query, then averages queries.
Differentiation produces a gradient for A[2,256,256]. More updates change how
often AdamW adjusts these same coefficients; they add no parameters or examples.

Learning rate 0.0003, zero weight decay, all other optimizer settings, FP32
operation order, subject exclusion and the strict z>0 prediction rule stay fixed.
No oracle cardinality, attribute value or evaluation-group label enters the
scorer. The new runner authenticates and calls the frozen scoring and fitting
helpers, recording the orchestration hash separately from the scorer hash.

## Restart, final endpoint and lineage

The zero-residual head exactly replays the original validation vectors in the
historical batches of eight and final batch of six. Initial full-training loss
again equals 0.003822767175734043. Only the final 2,000-update child receives a
new validation measurement; there are no intermediate validation checkpoints
from which to choose a winner.

Training loss finishes at 0.00003491543247946538, compared with the historical
500-update endpoint's 0.0002471808111295104. That comparison is between separately
declared fits. It is not a guarantee that the first 500 steps replay bit-for-bit
across GPU executions, nor evidence that AdamW reached an optimum.

The full child has 94 state tensors: the original 93, unchanged, plus trained A.
Strict reload into a fresh model with A attached reproduces tensor and optimizer
state. Parent pretraining steps (2,000) and residual fitting updates (2,000) are
separate lineage fields; calling this simply a "4,000-step model" would obscure
which parameters and objective were involved.

## What the fixed screen measured

| Predictor | Exact / 222 | Macro F1 | Strict separation / 222 |
|---|---:|---:|---:|
| Original dense parent, freshly replayed | 112 | 0.990374 | 165 |
| Historical 500-update residual | 198 | 0.999406 | 211 |
| New 2,000-update residual | 207 | 0.999746 | 217 |
| Historical eight-branch selector | 201 | 0.979926 | Not a selector metric |

The historical rows reuse accepted saved evidence, with exact query order,
labels, split and scorer compatibility checked. They are not newly executed
training runs or matched timing controls. Relative to the 500-update residual,
the longer fit gains 10 exact answers and loses one. Relative to the stronger
selector it gains 13 and loses seven. Relative to the original dense parent it
gains 95 and loses none.

| Group | Child exact | Required floor | Check |
|---|---:|---:|---|
| COLOR | 103 / 103 | 103 | Pass |
| Single TYPE | 50 / 51 | 50 | Pass |
| Dual TYPE | 54 / 68 | 48 | Pass |

All 222 outputs contain between one and 506 products. Exact count exceeds 201,
macro F1 exceeds 0.9799255176742276, group floors pass and all primary execution
invariants pass. The gate is a conjunction: improvement in one number does not
compensate for failure elsewhere. The 500-update result failed that same rule;
this independently audited result passes it.

## A pass still leaves incorrect complete answers

There are eight false-positive and ten false-negative member occurrences across
the new outputs. Fifteen of the 222 complete answers remain incorrect. A
descriptive post-hoc count of saved outputs finds 12 with one membership error
and three with two. These counts were not used for training or selection. Macro F1
is very high because most members in each large answer are correct; exact-set
accuracy asks whether even one member is missing or extra. The compiler oracle
is correct by construction, so this result remains below oracle parity.

Strict ranking separation holds for 217 queries, while only 207 answers are
exact at the fixed zero threshold; ten nonexact queries are strictly separated.
Separation describes the relative order of
true and false members; it does not establish that zero lies between those
groups. Neither the remaining errors nor that difference authorizes post-result
threshold tuning under this experiment's contract.

## What follows from this result

The independent audit and separate owner decision accept this single-seed
screen, supporting a separately declared replication. This does not
authorize serving promotion, protected-test evaluation or a hidden budget
extension. The ordinary inherited decoder forward still ignores A; this child
requires the explicit experiment-specific factory and dense scorer.

The validation split has guided multiple research choices. This adaptive result
supports the longer fixed recipe on this development screen; it does not prove
convergence, generalization or a causal capacity advantage. No new architecture
is introduced in the budget comparison. The primary passed 91 focused CPU tests,
including 57 reused arithmetic and lifecycle tests. The independent auditor
passed 37 synthetic tests and checked saved sets, metrics, split membership,
loss/update receipts, checkpoint tensors and optimizer state on CPU. It did not
independently repeat CUDA training or regenerate neural logits. The full sibling
is recorded as the 31st final checkpoint, including its explicit training identity
and unchanged scorer identity.

The approximately 4.88-second refit and 12.47-second campaign wall time are
descriptive local durations, not a serving speedup, matched-quality throughput
or energy result. Full head vectors, training labels and checkpoint payloads
remain hash-bound local dependencies. The
[evidence inventory](../evidence/updates/2026-09-25-bilinear-budget2000/README.md)
preserves the exact code, receipts and archival boundary without claiming
durable remote weight availability.
