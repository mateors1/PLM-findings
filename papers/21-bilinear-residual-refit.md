# Cross-coordinate relations improve membership but miss the exact-answer gate

**Status: independently audited evidence accepted; fixed quality gate rejected.**
The new frozen-parent bilinear head reaches 198/222 exact validation sets and
99.9406% macro F1. It improves substantially over the parent's dense predictor
(112 exact), yet falls short of the stronger eight-branch selector's 201 exact
answers and its TYPE group counts. A high F1 does not override
the declared complete-answer requirement.

The [frozen plan](../evidence/updates/2026-09-25-bilinear-residual-refit/experiments/2026-09-25-bilinear-residual-plan.md),
[portable result](../evidence/updates/2026-09-25-bilinear-residual-refit/experiments/2026-09-25-bilinear-residual-refit.json),
[independent audit](../evidence/updates/2026-09-25-bilinear-residual-refit/campaign/independent-audit.json)
and [owner decision](../evidence/updates/2026-09-25-bilinear-residual-refit/campaign/decision.json)
preserve both the progress and the failed gate.

## What changed in the head

The earlier projection refits could change each coordinate's relation weight,
but could not directly learn interactions between different embedding coordinates.
This experiment starts again from the original seed-1729 parent, rather than
either rejected refit child. All 93 original state tensors remain byte-identical.
One new FP32 parameter A[2,256,256] is initialized to zero and trained; index 0
means TYPE and index 1 COLOR. Oracle attributes and evaluation groups are never
head inputs.

With frozen, scaled normalized product embeddings E[1025,256], define

\[
M_d=\frac{A_d+A_d^\top}{2},\qquad
z_{sdi}^{\mathrm{new}}=z_{sdi}^{\mathrm{parent}}+
\frac{E_s^\top M_d E_i}{16}.
\]

An off-diagonal coefficient can connect coordinate j of the subject embedding
to coordinate k of a candidate. Symmetrization makes the relation symmetric in
real arithmetic. The two matrices store 131,072 numbers, with at most 65,792
independently effective symmetric coefficients; dependencies in E may reduce
that further. This is still not an arbitrary 1025-by-1025 relation table.
Swapping subject and candidate need not produce bitwise-identical FP32 results.

The implementation preserves the original parent's multiplication order, then
adds the residual. It transforms all entity embeddings into [2,1025,256], gathers
the requested subject/dimension rows into [B,256], and scores products into
[B,1025]. At zero A, the actual new scorer exactly reproduces all 222 saved
parent vectors in batches of eight with a final batch of six. Independent
saved-evidence checks bind the ordered raw FP32 hashes and zero residuals.

## The fixed fit and prediction rule

Only the 1,637 training queries supervise A. Each query gives equal weight to
its positive group and its nonself negative group:

\[
L=\frac1Q\sum_q\frac12\left[
\frac1{|P_q|}\sum_{i\in P_q}\operatorname{softplus}(-z_{qi})+
\frac1{|N_q|}\sum_{i\in N_q}\operatorname{softplus}(z_{qi})\right].
\]

The training score tensor is [1637,1025]. Frozen embeddings and parent logits
allow this fit to skip transformer forward passes. Exactly 500 full-batch AdamW
updates use learning rate 0.0003, no weight decay, scheduler, clipping or auxiliary
loss. Initial training loss exactly matches the parent's 0.003822767175734043;
final loss is 0.0002471808111295104.

Prediction remains every non-subject product with z>0, sorted by ID. There is no
true-cardinality input, threshold search, truncation, fallback or group routing.
The final child is reloaded into a fresh model with A explicitly attached before
strict checkpoint loading; all 94 tensor hashes and optimizer state match.

## Results and the unchanged decision

| Predictor | Exact / 222 | Macro F1 | Strict separation / 222 |
|---|---:|---:|---:|
| Original dense parent, freshly replayed | 112 | 0.990374 | 165 |
| Historical mean-BCE W-only refit | 122 | 0.991855 | 165 |
| Historical worst-boundary W-only refit | 128 | 0.992346 | 160 |
| New bilinear residual | 198 | 0.999406 | 211 |
| Historical eight-branch selector | 201 | 0.979926 | Not a selector metric |

The historical rows reuse accepted evidence; they are not new training runs or
paired latency controls. Relative to the freshly replayed dense parent, the
child gains 86 exact answers and loses none. Relative to the stronger selector,
it gains 10 and loses 13: the net difference of three hides changed queries.

| Group | Child exact | Required nonregression floor |
|---|---:|---:|
| COLOR | 103 / 103 | 103 |
| Single TYPE | 48 / 51 | 50 |
| Dual TYPE | 47 / 68 | 48 |

All 222 outputs satisfy the declared 1-to-506-product serialization bound, and
the F1 floor passes. The strict requirement of more than 201 exact sets fails,
as does TYPE group nonregression. The independent audit accepts the evidence;
the separate owner decision rejects quality promotion without changing the gate.

## Why almost-perfect F1 is not oracle parity

Across these outputs there are 25 false-positive and 11 false-negative member
occurrences. A descriptive post-hoc count of the saved outputs finds 24 nonexact
queries: 16 have one membership error, five have two, two have three and one has
four. These counts explain the result; they were not a tuning or selection rule.

For example, Absol's TYPE answer contains all 68 correct members plus one extra
product, ID 1052. Its F1 is 136/137, about 0.99270, while exact-set accuracy for
that query is zero. Its true members are strictly separated from false members,
but zero still includes the extra product. Ranking separation and placement of
the fixed zero threshold are different properties. The compiler oracle remains
correct by construction, so 198/222 is progress toward parity, not a quality win
over the oracle.

## What can be concluded

Cross-coordinate residual training works substantially better on this validation
screen than the two previous W-only fits. The intervention changes parameter
count, parameterization and optimization geometry together. Equal update count
and AdamW settings do not isolate a causal capacity effect or equal optimization
strength. The [diagonal-feasibility diagnostic](20-frozen-diagonal-feasibility.md)
was inconclusive and does not prove that the older family could not fit.

This is one adaptive validation screen from one seed. No protected-test queries,
additional-seed replication, autoregressive generation or serving integration
are measured. The roughly 1.48-second refit is descriptive local fitting time,
not a training or serving speedup claim. The runner passed 57 focused CPU tests;
the independent auditor passed 60 synthetic tests. That auditor reconstructs
saved sets, metrics, labels, gate and checkpoint lineage, but does not independently
repeat CUDA training or regenerate neural logits.

The new full derivative is registered as the 30th final checkpoint, with an
explicit architecture and objective distinct from ordinary serving. The inherited
decoder forward does not evaluate A; loading or scoring this child requires its
experiment-specific factory and scorer. No default or serving policy is promoted.
The [evidence inventory](../evidence/updates/2026-09-25-bilinear-residual-refit/README.md)
preserves exact code and receipts while keeping raw weights, full head vectors
and training labels as hash-verified local dependencies, without claiming durable
remote weight availability.
