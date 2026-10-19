# Worst-member training improves dense answers but misses the strong gate

**Status: independently audited evidence accepted; fixed quality gate rejected.**
A projection refit targeting each query's hardest correct and incorrect members
produces 128 exact validation answers out of 222. That improves on the original
parent's dense predictor (112) and the previous mean-BCE refit (122), but remains
below the accepted eight-branch predictor (201). The independent saved-evidence audit passes, and a separate owner decision
accepts the evidence while rejecting the derivative on the unchanged quality gate.

The [frozen plan](../evidence/updates/2026-09-25-projection-worst-boundary/experiments/2026-09-25-projection-worst-boundary-plan.md)
defines a sibling experiment from the original seed-1729 parent, rather than
further training the rejected [mean-BCE child](18-frozen-feature-projection-refit.md).
The optimizer, data order, 500-update budget and fixed prediction rule remain the
same. Equal optimizer settings do not establish equal optimization strength for
different objectives.

## A loss that concentrates on the boundary

Every embedding and decoder tensor stays frozen. Only the symmetric membership
projection W, shape [256,256], changes. With scaled normalized product embeddings
E and steering embeddings U, the head computes

\[
z_{s,d,i}=\frac{(E_s\odot WU_d)^\top E_i}{\sqrt{256}}.
\]

For each training query q, let P contain the correct products and N the other
products, excluding the subject from both. The new loss is

\[
a_q=\min_{i\in P}z_{qi},\qquad b_q=\max_{i\in N}z_{qi},
\]
\[
L=\frac1Q\sum_q\frac12\left[
\operatorname{softplus}(-a_q)+\operatorname{softplus}(b_q)\right].
\]

The previous loss averaged penalties over every positive and every negative.
This one concentrates each query's gradient on its worst-scoring true member
and best-scoring false member. FP32 `amin` and `amax` share gradients evenly
among exact ties. The training score tensor has shape [1637,1025]; detached
embedding features allow the fit to skip transformer forward passes. Labels
come only from the 1,637 training queries.

The output rule remains all non-subject products with z>0, sorted by ID. Exact
prediction requires a_q>0 and b_q<=0. This objective therefore targets the
absolute zero boundary; a relative-gap hinge is unchanged by a common shift, so it can be satisfied
while the zero-threshold set remains wrong.

Fixed embeddings make each logit linear in W and this objective convex in W,
with possible nondifferentiability at ties. That establishes neither a finite
minimizer nor convergence after 500 AdamW updates, and says nothing by itself
about validation generalization or representational adequacy.

## Validation outcomes

All columns use the same 222 validation queries. The mean-BCE sibling and
eight-branch results are authenticated historical comparators, not fresh
re-executed controls. The original dense parent is exactly replayed before fitting.

| Measurement | Parent dense | Mean-BCE sibling | Worst-member sibling | Eight-branch comparator |
|---|---:|---:|---:|---:|
| Exact sets | 112 | 122 | 128 | 201 |
| Macro F1 | 0.990374 | 0.991855 | 0.992346 | 0.979926 |
| COLOR exact /103 | 75 | 77 | 84 | 103 |
| Single-TYPE exact /51 | 32 | 37 | 36 | 50 |
| Dual-TYPE exact /68 | 5 | 8 | 8 | 48 |
| Strict head separation | 165 | 165 | 160 | not measured |

The new child gains 21 exact answers and loses five against its dense parent.
Against the mean-BCE sibling, it gains ten and loses four. Against the accepted
eight-branch predictor, it gains one and loses 74. Its 128 exact answers fail
the fixed requirement of strictly more than 201; every group is also below its
required count. F1 and serialization checks pass. No threshold, acceptance gate
or stopping point is changed after seeing these results.

Relative to the dense parent, false-positive memberships fall from 512 to 331
while false negatives rise from 76 to 131. Better macro F1 can coexist with many
inexact sets: one erroneous membership makes an entire answer nonexact.
All 222 outputs meet the declared serialization constraints; that does not make
them all correct.

Strict separation asks whether every true score exceeds every false score,
independent of the fixed threshold's placement. It falls from 165 to 160 despite
higher fixed-zero exactness: two queries gain separation and seven lose it
against the parent. Against the mean-BCE sibling, the counts are three gains
and eight losses. The rankings have changed; an aggregate improvement in exact
answers does not establish a uniform improvement in ranking.

## Different losses, different partitions

The optimized worst-member training loss falls from 0.44132093 to 0.31003913.
A separately declared diagnostic measures ordinary balanced mean BCE on the same
full training batch: it rises from 0.00382277 to 0.01404841. That diagnostic never
controls updates, stopping or selection. These loss magnitudes have different
definitions and should not be compared as one scale.

There is no contradiction in lower worst-member training loss, higher mean
training BCE and better validation exactness. They summarize different error
properties, and the last measurement uses a different partition. The result
does not identify a causal explanation for earlier joint-training margin
failures, prove a capacity limit, or establish that additional optimization
would succeed.

## Evidence and limits

The parent has 2,000 original training steps; this new child records 500 fresh
projection-only updates and a distinct objective,
`plm-projection-worst-boundary-softplus-v1`. All 93 state tensors are checked;
only W changes, and model and optimizer reload checks pass in the primary run.
Parent head vectors match their historical batch-eight/final-six references
exactly. Initial ordinary training BCE also exactly replays the previous value.

The synchronized refit takes 4.2098359 seconds on the recorded local GPU. This is
descriptive elapsed time, not an end-to-end serving, throughput or energy result.
No decoder generation is measured for the child. Freezing decoder weights does
not imply that head-guided generation would remain unchanged.

This is one adaptive research step on one seed after repeated validation-guided
choices. There is no protected-test evaluation, extra-seed replication,
validation early stopping, intermediate checkpoint selection or standard-serving
promotion. The checkpoint payload remains a hash-bound local dependency;
a published sidecar is not proof of durable remote weight archival.

The [portable results](../evidence/updates/2026-09-25-projection-worst-boundary/experiments/2026-09-25-projection-worst-boundary.json),
[independent audit](../evidence/updates/2026-09-25-projection-worst-boundary/campaign/independent-audit.json)
and [owner decision](../evidence/updates/2026-09-25-projection-worst-boundary/campaign/decision.json)
retain the accepted measurements and rejected quality gate separately. The runner
checks passed 47 CPU tests, including reused lifecycle coverage; the auditor
passed 39 synthetic tests. The saved-evidence audit reconstructs predictions,
metrics and gates, inspects checkpoint/optimizer identities, and authenticates
the scalar traces. It does not independently repeat neural training or regenerate
logits. Reused helper implementations and their hashes remain explicit in the
[evidence inventory](../evidence/updates/2026-09-25-projection-worst-boundary/README.md).
