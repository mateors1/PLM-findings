# 30. Is the ranking wrong, or is the cutoff wrong?

**Status:** completed and independently audited. Only 479/666 observations have
strictly separable membership rankings, below the current selector's 569 exact
answers. The diagnostic is accepted as evidence; no model or policy is promoted.

The [direct membership experiment](28-direct-membership-ablation.md) included
every non-subject product with a positive logit. It recovered many missing
members, yet reduced exact answers from 569 to 339 out of 666 observations.
Before changing training, we should distinguish two different kinds of error.

## A cutoff can move; an ordering cannot

Imagine a query with two true members and two nonmembers:

| Product | Expected membership | Example score |
| --- | --- | ---: |
| A | true | 4.0 |
| B | true | 2.0 |
| C | false | 1.0 |
| D | false | -3.0 |

The ranking is perfect, but selecting scores greater than zero includes C.
A threshold of 1.5 would recover the exact answer. This is an example, not a
threshold chosen for our model.

Now exchange B's and C's scores. No single score threshold can include B while
excluding C: the incorrect product ranks above a correct one. Changing a cutoff
cannot repair that ordering. We would need different scores or a selection rule
that uses information beyond one threshold.

## Derive the diagnostic

Let T contain the true members and N the nonmembers, after excluding the subject.
For a query with both kinds present, define:

\[
a=\min_{i\in T}z_i,\qquad b=\max_{j\in N}z_j,\qquad g=a-b.
\]

Here a is the weakest-scored true member and b the strongest-scored impostor.
With the strict rule `score > threshold`, including every true member requires
t<a; excluding every nonmember requires t>=b. Both can hold exactly when:

\[
b\leq t<a,\qquad\text{so an exact threshold exists iff }g>0.
\]

A positive gap means **strict rank separation**. A negative gap means rank
overlap. A zero gap means a true and false member meet at the same boundary
score; a score-only threshold must include both or exclude both. Equality is
therefore not strict separation.

Our original zero threshold is exact precisely when b<=0<a. A separable query
can fail that condition even when its ranking is correct. Empty and full expected
sets need separate handling because one of the extrema is undefined; the
diagnostic tests those cases rather than inventing an infinite JSON number.

## The tensor view

The learned symmetric head produces a matrix
Z in R^(B x 1025): one row per query, one column per product. Its embedding
width is 256. As derived in [lesson 28](28-direct-membership-ablation.md),

\[
Z=\frac{(E_{subject}\odot R)E^\top}{\sqrt{256}},
\quad E\in\mathbb{R}^{1025\times256},
\quad E_{subject},R\in\mathbb{R}^{B\times256}.
\]

For diagnosis, a boolean target mask has the same shape as Z. A separate
eligibility mask excludes each row's subject, leaving 1024 eligible columns.
Conceptually, the extrema are reductions over masked columns:

```python
# Conceptual tensor notation; the actual saved-score diagnostic uses Python.
a = Z.masked_fill(~true_member_mask, float("inf")).amin(dim=1)  # [B]
b = Z.masked_fill(~nonmember_mask, -float("inf")).amax(dim=1)  # [B]
gap = a - b                                                   # [B]
```

Explicit empty/full-set branches must precede interpreting these reductions.
The masks use expected labels, so this computation belongs to evaluation.
There is no new model call: we analyze archived FP32 logits, keeping their
original batch-eight computation identity.

## What if we knew the answer size?

A related diagnosis takes the K highest scores, where K is the true answer size.
For reproducibility, equal scores are ordered by ascending product token ID.
If there is strict separation, this **oracle-cardinality top-K** rule is exact.
At a tied boundary, the ID rule can happen to pick the correct members without
strict separation. We report those cases separately.

Knowing K is privileged information. A serving system would need to infer it or
receive it from an independently justified contract. Reporting this diagnostic
as ordinary model quality would hide a crucial input. It also does not prove
that one global threshold works: different queries can need disjoint intervals.

This is not a probability-calibration measurement. Calibration asks whether
predicted probabilities match observed frequencies. Our question concerns rank
ordering and the location of one decision boundary.

## The measured answer: threshold placement explains only part of the failure

| Seed | Current selector exact | Zero threshold exact | Strictly separable / oracle-K exact | Rank overlap |
| --- | ---: | ---: | ---: | ---: |
| 1729 | 191 | 112 | 165 | 57 |
| 1730 | 192 | 116 | 160 | 62 |
| 1731 | 186 | 111 | 154 | 68 |
| Total | 569 | 339 | 479 | 187 |

There are no boundary ties in these saved observations. Of the 479 separable
queries, 339 already pass the zero threshold and 140 need a different cutoff.
The remaining 187 have genuine rank overlap. Even a perfectly informed choice
of a separate scalar threshold for every query cannot exceed 479 exact answers
on these unchanged scores. This is below the existing selector's 569.

The oracle-K diagnosis gains 16 exact answers but loses 106 against the
selector: 569 + 16 - 106 = 479. Its macro F1 is 99.35%, again showing that a high
partial-credit score can coexist with substantially fewer complete answers.
This F1 value is a measured privileged-input diagnostic, not a ceiling over
every possible threshold rule.

| Group | Observations | Selector exact | Separable | Rank overlap |
| --- | ---: | ---: | ---: | ---: |
| COLOR | 309 | 308 | 304 | 5 |
| TYPE single | 153 | 136 | 134 | 19 |
| TYPE dual | 204 | 125 | 41 | 163 |

Dual-TYPE queries dominate the rank-overlap problem. The restriction to
generated candidate sets protects many correct answers even when their
individual membership ranking is imperfect. Removing that restriction loses
useful structure.

Among the current selector's 97 failures, 81 have rank overlap and only 16 are
strictly separable. Five of the 53 source-coverage failures are separable; 48
are not. A cutoff-only repair therefore leaves most current failures unresolved,
even before asking how to choose the cutoff without expected labels.

## How this guides the next experiment

The [declared plan](../experiments/2026-09-25-membership-separability-plan.md)
required results for each training seed and TYPE/COLOR group, with paired
comparisons against all 97 failures of the current selector. The 666 observations
reuse 222 query identities across three training seeds; they are not 666
independent queries.

The result rules out a threshold-only replacement as a way to beat the current
exact count on these scores. A next training hypothesis could penalize a
high-scoring nonmember outranking a weak true member, or a new candidate
construction could preserve structural constraints while recovering missing
products. These are hypotheses, not measured improvements or proof that the
architecture lacks capacity. Any follow-up must specify its training inputs,
validation protocol and comparison before measurement. The protected final test
remains untouched.

The [portable result](../experiments/2026-09-25-membership-separability.json)
binds the frozen plan, three checkpoint identities, saved reports, independent
audit and separate diagnostic acceptance. Thirty-one focused CPU tests pass;
the auditor independently reconstructs every row and aggregate without importing
the diagnostic's arithmetic. No neural forward was run for this measurement.
