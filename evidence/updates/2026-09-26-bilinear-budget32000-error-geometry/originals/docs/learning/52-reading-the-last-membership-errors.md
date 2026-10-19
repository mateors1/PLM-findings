# 52. Reading the last membership errors

Status: completed diagnostic, independently audited and accepted. No new model
or corrected predictions.
The [frozen plan](../experiments/2026-09-26-bilinear-budget32000-error-geometry-plan.md)
defines the saved-only analysis. The preceding campaign has six nonexact
seed-query outcomes across three evaluations of the same 222 queries.

## Two boundaries, two questions

Imagine a shelf of candidates sorted by their model score. A perfect ranking
puts every true member above every nonmember. A perfect set prediction also
needs the cutoff to fall between those two groups.

For one query, the model has a score vector `z:[1025]`. The saved labels and
predictions are Boolean masks of the same shape. Exclude the subject's own
column. Let `a` be the smallest score among true members and `b` the largest
score among nonmembers:

\[
a=\min_{i\in T}z_i,\qquad b=\max_{i\notin T,\ i\ne s}z_i.
\]

The ranking gap is `a-b`. A positive gap means that every true member ranks
above every false member. Our actual inference rule selects `z > 0`, so it
needs both `a > 0` and `b <= 0` to produce the exact set.

| Hypothetical boundary scores | What is wrong? | Can some scalar cutoff work? |
| --- | --- | --- |
| `a=0.4, b=-0.2` | Nothing; zero produces the exact set | Yes |
| `a=-0.1, b=-0.4` | A true member falls below zero | Yes, for these known labels |
| `a=0.5, b=0.2` | A false member rises above zero | Yes, for these known labels |
| `a=-0.2, b=0.1` | True/false scores overlap | No |

The second and third rows describe cutoff errors; the last describes a ranking
error. At a tie `a=b`, a single strict cutoff cannot separate the tied members.
At zero itself, a nonmember is correctly excluded, but a true member is missed.
Small sign details matter when exact set agreement is the metric.

## Why knowing the interval is not a solution

For a hypothetical rule `z > t`, an exact set requires

\[
b\le t<a.
\]

We can calculate that interval only because evaluation gives us the true
membership labels. Production inference does not have those labels. Picking a
different interval for each validation query would use the answer to repair
the prediction. It would not demonstrate learned retrieval.

There is a further distinction between each query having an interval and one
constant working for every query. The latter needs

\[
\max_q b_q < \min_q a_q.
\]

Two individually separable questions can require incompatible cutoffs. We
report whether a shared interval exists within each seed and dimension group,
but do not choose a value or score hypothetical repaired answers. Scores from
different trained parents are not pooled into a shared interval.

## What saved-score auditing buys us

For each seed, the saved validation matrix has shape `[222,1025]`. We need no
new forward pass: reconstruct `z > 0`, exclude self, and compare the result to
the already recorded selected IDs. Hashes connect those score files to the
accepted training campaign. A separate implementation recomputes the reduction.

This distinguishes a scientific result from a reporting bug. For example,
incorrect product-column alignment could make a convincing list of "hard
Pokémon" that has nothing to do with the trained model. We therefore verify
query order, product IDs, target masks, selected masks and counts before
interpreting individual errors.

The result will include every reduced query, not only the six failures. It
will also identify which failures recur across seeds. Six failed model-query
outcomes need not mean six distinct Pokémon questions.

## How this constrains the next experiment

A cutoff error suggests investigating the training objective or a calibration
method learned on training data. It does not prove that calibration will help
unseen questions. A ranking error suggests examining features and relational
scoring capacity, but it does not prove that the model needs more parameters.

The next intervention must specify its training data, fixed inference rule,
matched controls and acceptance criteria before running. The protected test
partition remains closed while we use validation to develop these hypotheses.

## The actual diagnostic: incompatible cutoffs

The independently audited reconstruction finds four separated missing-member outcomes, one
separated extra-member outcome and one ranking overlap. Six failed model-query
outcomes correspond to five distinct queries: FEZANDIPITI, FLYGON, HATTERENE,
SKARMORY and SWANNA, all under TYPE. FEZANDIPITI fails under two seeds.

Consider seed 1731. Every query has a positive ranking gap, yet two answers
are wrong at zero. Its FEZANDIPITI query needs a cutoff below approximately
`-0.597` to include a true member. Its FLYGON query needs a cutoff at least
approximately `0.218` to exclude a false member. No number meets both needs.
This remains true even if we use a separate constant for dual-TYPE queries.

That is why "the ranking is right" does not imply "just change the threshold."
A query-dependent correction might help, but it would need a rule learned
without the validation labels. The earlier affine experiment's unchanged
validation answers also remind us that adding a correction does not guarantee
it learns a useful one.

The [complete report](../experiments/2026-09-26-bilinear-budget32000-error-geometry-report.md)
preserves exact boundary values, wrong IDs, provenance and audit status.
All 666 reduced rows are retained, including the 660 exact outcomes.
