# 45. Ranking errors and zero-threshold errors

**Status: independently audited descriptive diagnostic.** The replication in
[Lesson 44](44-replicating-across-trained-parent-seeds.md) leaves 24 incorrect
answers across its two fresh models. This lesson introduces a saved-score
diagnostic to understand those mistakes before changing the training recipe.

## What does a score mean here?

For one query the experimental head produces 1025 scores, one per product.
The subject itself is excluded. The fixed output rule includes every remaining
product with a score greater than zero. The model produces these scores without
seeing the expected answer at inference time.

A **logit** is an unbounded score. The sigmoid maps it into (0,1):

\[
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

In particular, z>0 is equivalent to sigmoid(z)>0.5. That does not prove the
sigmoid values are calibrated probabilities. The completed experiments use
the sign directly; this diagnostic does not introduce a new sigmoid decision.

## Two requirements for an exact answer

Using the known validation answer only for analysis, let

\[
a=\min_{i\in T}z_i,\qquad b=\max_{i\notin T,\ i\ne s}z_i.
\]

T contains the true members, and s is the subject. Thus a is the weakest true
member's score and b is the strongest false member's score. There are two
different questions:

- **Ranking:** is every true member above every false one? This requires a>b.
- **The current decision:** are true members positive and false members
  nonpositive? This requires a>0 and b<=0.

For a synthetic example, true scores [0.4,0.2] and false scores [0.1,-0.3]
have perfect separation: 0.2>0.1. Yet the false member at 0.1 is included by
the zero rule. In another example, true scores [0.4,-0.2] and false scores
[0.1,-0.3] overlap; merely shifting one threshold cannot restore an exact set.

The strict inequalities matter. A true member at exactly zero is missed; a
false member at exactly zero is correctly excluded. A tied true/false pair
cannot be separated by any single strict-greater-than threshold.

## An interval explains the difference

For a hypothetical rule z>t, an exact answer requires

\[
b\leq t<a.
\]

The interval [b,a) exists exactly when b<a. Its width g=a-b is the separation
gap. We report the interval's bounds as a diagnostic; we do not choose a
threshold or produce alternative answers from validation labels.

For several queries to share one threshold, the stricter requirement is

\[
B=\max_q b_q < A=\min_q a_q.
\]

Every query might have its own nonempty interval while their shared intersection
is empty. Therefore, many separated mistakes alone do not show that changing
one constant would fix the model. Even a nonempty validation intersection would
not establish a deployable threshold or performance on unseen data.

## What the saved scores show

The [frozen plan](../experiments/2026-09-25-bilinear-error-geometry-plan.md)
uses the saved scores of all three completed children, with seed 1729 explicitly
labeled historical development evidence. It reconstructs the existing answers,
checks exactness and error counts, classifies separated versus overlapping
mistakes, and compares which shared questions fail across parents. For each seed,
think of the saved scores as a matrix Z of shape **[222,1025]**: one validation
query per row and one product per column. A boolean membership mask of the same
shape identifies true members; a separate mask excludes the subject. Reducing
each row over the true and false masks gives vectors a and b of shape **[222]**.

| Parent seed | Exact at zero | Separated, missing | Separated, extra | Overlap or tie |
| --- | ---: | ---: | ---: | ---: |
| 1729, historical | 207 | 4 | 6 | 5 |
| 1730, fresh | 212 | 3 | 4 | 3 |
| 1731, fresh | 208 | 3 | 5 | 6 |

The two fresh models have 24 incorrect answers: **15 are strictly separated**
(6 missing-only and 9 extra-only), while **9 have overlapping or tied ranges**.
All 24 concern dual-type subjects. Their original exact counts remain 420/444;
this analysis has not repaired any predictions.

For every seed, the shared interval is empty both across all queries and within
the dual-type group. Thus one constant threshold cannot make these saved answers
all exact. This is stronger evidence than merely noticing many separated errors:
ranking and score placement both need attention. It does not establish which
future training change will improve them.

Across the three models, 194 of the 222 shared questions are exact every time;
19 fail in one seed, 7 in two seeds, and 2 in all three. That is 28 distinct
questions accounting for 39 incorrect query-model observations. The 666 rows
are repeated measurements of 222 questions, not 666 independent test examples.
Different error sets may motivate a future experiment, but do not establish
that an ensemble works or supply a deployable way to choose the right answer.

This is **posthoc analysis**: we already know the validation results and are
exploring their structure. It can motivate a separately declared experiment;
it cannot turn that experiment into an independent confirmation retroactively.

No weights are loaded, no model is run and no new checkpoint is created. The
final-checkpoint count stays 33. A separate audit reconstructed the diagnostic
arithmetic, while the previous accepted audits remain the authority for the
saved scores and labels. This does not repeat their training or checkpoint audit.

The primary runner passed 38 synthetic tests and the independent auditor passed
39; both actual executions completed successfully on their first attempt.
The [portable result](../experiments/2026-09-25-bilinear-error-geometry.json)
binds the source, saved inputs, audit and separate evidence-acceptance decision.
Accepting the diagnostic means its accounting was verified; this campaign has
no model-quality gate and makes no oracle-parity or serving claim.
