# 40. Training the weakest membership decisions

**Status: completed; evidence accepted, fixed quality gate failed.** The
[worst-boundary plan](../experiments/2026-09-25-projection-worst-boundary-plan.md)
defines a new sibling refit from the original accepted parent. It leaves the
failed [mean-BCE experiment](39-refitting-a-frozen-feature-head.md) unchanged.

## An average can hide a broken answer

In the previous screen, training loss nearly halved and dense exact answers
improved from 112 to 122 out of 222. Yet 100 answers still contained an error.
The existing loss averages penalties over each query's correct products and
incorrect products. An isolated bad membership can contribute very little to
that average when hundreds of other decisions are already good.

This time, ask each query for its two weakest decisions. Let a be the smallest
score among true members, and b the largest score among nonmembers, excluding
the subject. The dense predictor selects scores greater than zero, so the
whole set is correct exactly when

\[
a>0\quad\text{and}\quad b\le0.
\]

The asymmetry at zero is intentional: a true member scored zero is missing,
while a nonmember scored zero is correctly excluded.

## A loss aimed at that boundary

Use the worst positive penalty and worst negative penalty for each query:

\[
L_q=\frac12[\operatorname{softplus}(-a_q)+\operatorname{softplus}(b_q)],
\qquad L=\frac1Q\sum_qL_q.
\]

Here Q=1,637 training queries. Softplus is log(1+exp(x)); low positive scores
and high negative scores incur larger penalties. The operation is equivalent
to taking the maximum softplus loss inside each class, then averaging the two
classes and all queries. No count or attribute value enters the model prompt.

This is a **surrogate loss**: a differentiable-almost-everywhere proxy for the
discrete exact-set objective. Exact-set correctness jumps between 0 and 1; it does
not provide a useful ordinary gradient telling us how to move a wrong score.
The surrogate supplies that direction, but improvement in it still need not
improve validation exactness.

## Why this differs from the previous margin loss

The earlier hardest-boundary hinge used relu(m+b-a). It asked whether every true
member outranked every false member by a margin. It did not anchor the scores
to zero.

For an illustrative query, take margin 1, weakest true score 2 and strongest
false score 1. The hinge is zero, because 1+1-2=0. But the positive false score
still makes the dense prediction wrong. Shift both scores by -1.5: they become
0.5 and -0.5. Their gap is unchanged, but the dense answer becomes correct.

The new loss changes under that shift because it separately pressures a toward
positive values and b toward negative values. This example explains the two
objectives; it does not give us an independently adjustable threshold for each
real query. Every query still depends on the shared learned projection W.

## Which scores receive gradients?

For a unique weakest true member and unique strongest false member,

\[
\frac{\partial L_q}{\partial a_q}=-\frac12\sigma(-a_q),
\qquad
\frac{\partial L_q}{\partial b_q}=\frac12\sigma(b_q),
\]

where sigmoid sigma(x)=1/(1+exp(-x)). These derivatives favor raising a and
lowering b; the actual shared-parameter update need not move every extreme in
that direction. Other scores get no direct contribution from that query's objective
until they become an extreme. The shared W update can nevertheless change
their values, because every score uses the same parameters.

With Z[1637,1025], masks remove ineligible columns before reducing over the
product axis. The minima and maxima each have shape [1637]. torch.amin/amax
divide gradients equally among tied extrema; indexed min/max would select a
single tied index and implement a different training rule. Tests must cover
that distinction as well as duplicate labels, controls, padding and subjects.

We retain W[256,256] as the only trainable tensor. Embeddings and transformer
weights stay frozen. Each score is linear in W, making this maximum-of-convex
penalties objective convex. It can be nonsmooth at ties. This still supplies
no guarantee of convergence, a finite minimizing W, or oracle parity.

## A controlled comparison, with limits

Start again from the original parent; do not continue the rejected child's
weights. Keep the 500 full-batch update budget and fresh AdamW recipe unchanged.
Compare dense predictions against the original parent, the completed mean-BCE
sibling, and the accepted width-eight system. The sibling's results are reused
historical evidence, not a newly executed control.

Equal optimizer settings do not mean equal gradient strength. The old loss
distributed each class's gradient over all its members; the new loss concentrates
it on the current worst members. This is a fixed-budget recipe comparison.
Record ordinary mean BCE as a separate initial/final training diagnostic so we
can see whether improving the extremes hurts the average. Do not compare the
two objectives' raw scalar magnitudes as if they were the same quantity.

The strong gate stays above 201 exact answers, with the same F1 and group floors.
All data separation, replay, finite-update, frozen-tensor and checkpoint checks
remain required. The new run gets its own objective, evaluator and checkpoint
lineage, followed by an independent audit and a separate decision. A failed
screen stays in the research record; a passing screen would need replication.

## What actually happened

All 500 updates completed. The independent audit accepted the saved evidence,
and the separate owner decision rejected the variant under the original gate.
The [portable result](../experiments/2026-09-25-projection-worst-boundary.json)
preserves both decisions and the full loss trace.

| Predictor, seed 1729 | Exact /222 | Macro F1 | Strictly separated queries |
| --- | ---: | ---: | ---: |
| Original parent, dense zero threshold | 112 | 0.990374 | 165 |
| Historical mean-BCE refit, dense | 122 | 0.991855 | 165 |
| New worst-member refit, dense | 128 | 0.992346 | 160 |
| Accepted original parent, eight-branch selector | 201 | 0.979926 | — |

Relative to the mean-BCE sibling, there are 10 exact gains and four losses,
giving six additional exact answers overall. COLOR improves from 77 to 84;
single-TYPE declines from 37 to 36; dual-TYPE remains eight. The strong gate
requires at least 103, 50 and 48 respectively, as well as more than 201 total.
All 222 sets meet the size/ID serialization contract, and F1 meets its floor,
but those facts cannot compensate for failed exact-answer requirements.

Against the original dense parent, the new child has 21 exact gains and five
losses. Against the strong selector, it has one gain and 74 losses. The graph
oracle remains correct by construction; 128 exact answers are far from parity.

## Improving one loss can worsen another

The new training objective falls from 0.441320926 to 0.310039133, about 29.75%.
The separately measured ordinary mean BCE rises from 0.003822767 to 0.014048407,
about 3.67 times its initial value. The initial mean BCE exactly reproduces the
historical starting measurement, as required before the first update.

These are two comparisons within their own loss definitions. Subtracting the
worst-member scalar from the mean-BCE scalar would not measure progress: they
aggregate different decisions. The observed divergence shows that this finite
optimization improved its chosen objective while worsening the average penalty
on the same training partition. It does not establish why any particular
validation answer changed or prove that the objective cannot work.

Nor does rising training mean BCE contradict improving validation F1. They
measure different things on different partitions: BCE penalizes score
confidence smoothly; F1 counts thresholded membership decisions. Here validation
false positives decline from the mean sibling's 389 to 331, while false
negatives increase from 109 to 131. Precision rises and recall falls.

## Better signs do not guarantee better ordering

Strict separation asks whether a>b. Exact dense prediction needs the stronger
zero-boundary condition a>0 and b<=0. Consequently, a query can have all true
members above all false members and still have the wrong zero-threshold set.

The new child has 160 strictly separated queries, of which 128 are exact and
32 remain nonexact. The original dense parent had 165 separated queries but
only 112 exact. This is how more exact sets can coexist with fewer separated
rankings. Counts also hide turnover: versus the parent, separation gains two
queries and loses seven; versus the mean sibling, it gains three and loses eight.
These are saved-score diagnostics, not evidence of an independently adjustable
threshold or a proof of a representation ceiling.

## What is preserved

The new child is checkpoint
`0abfbef2af5e12bcdaf21292a3b62a5a047fcdb7e6e77a8be5f5a0bb6fc4323b`.
It has its own objective `plm-projection-worst-boundary-softplus-v1` and evaluator
`plm-projection-worst-boundary-screen-v1`. Parent steps 2,000 and child refit
updates 500 remain separate. The audit inspects all 93 state tensors; only W
changes. Model and fresh optimizer state reload exactly. Runner tests passed
47 cases, including 26 reused lifecycle cases; the new auditor passed 39.

The audit independently reconstructs saved predictions, metrics and comparisons
and inspects CPU checkpoint payloads. It does not repeat training or regenerate
neural logits. The 4.2098359-second refit is descriptive, not a serving speedup.
No protected-test measurement, extra seed, extended budget, threshold search
or default promotion follows. The standard serving loader does not support this
new objective. All 29 registered final checkpoint file hashes were freshly
verified; local hashes and Git publication still do not establish durable weight
archival. This failed screen remains part of the research record.
