# 49. How do error boundaries change with training?

**Status: completed and independently audited.** Lesson 48
established that longer training improved both fresh parents. This lesson asks
what changed in their saved scores, using the same questions at both budgets.
We distinguish a result (more correct sets) from an explanation of its mechanism.

## Reduce a large score table to its weakest decisions

Each model has one saved score per query and candidate Pokemon. There are
222 validation queries and 1025 product columns, so one report contains a
[222,1025] score table. Two budgets and three parents give six such tables;
conceptually [2,3,222,1025]. This is an index layout, not a newly computed model
output or a claim that the diagnostic allocates one tensor of that shape.

For each query q, let T_q be its true members and N_q its false nonself members.
Keep three numbers:

\[
a_q=\min_{i\in T_q}z_{qi},\qquad
b_q=\max_{j\in N_q}z_{qj},\qquad g_q=a_q-b_q.
\]

The weakest true score a and strongest false score b determine whether every
membership decision succeeds. With the original rule z>0, the answer is exact
if and only if a>0 and b<=0. Ranking is strictly separated if g>0. The distinction
is whether the scores have the right relative order and the right location
around the fixed decision boundary.

Adding the same constant c to every score changes a and b by c, but leaves
g unchanged. That is why a larger gap alone cannot establish exact membership.
Conversely, a smaller positive gap can still produce a correct answer if it
straddles zero properly. These are mathematical properties, not new claims
about the observed diagnostic results.

## Four categories and a paired transition table

We classify each saved query into exactly one category:

| Category | What it means |
| --- | --- |
| Exact at zero | All true scores are positive and all false scores are nonpositive. |
| Separated, missing members | Relative order is correct, but one or more true scores fail the zero boundary. |
| Separated, extra members | Relative order is correct, but one or more false scores exceed zero. |
| Overlap or tie | At least one false score reaches or exceeds the weakest true score. |

For each parent, a [4,4] count table maps the old category to the new category.
Each query contributes exactly once. Its total must be 222, and transitions
into or out of the exact category must reproduce the previously audited gains
and losses. Group tables use their own query counts. This catches attractive
but incomplete explanations based on a few selected examples.

We also compare delta-a, delta-b and delta-g for each aligned query. These are
descriptive changes in a parent's score scale. They are not calibrated confidence,
proof of a causal mechanism, or directly comparable units across parents.

## Can one threshold serve every question?

For a hypothetical rule z>t, a separated query permits t in [b,a). The left
endpoint is included because a false score equal to t is excluded. The right
endpoint is excluded because a true score equal to t would also be excluded.

A single constant threshold works on all saved queries in a group only when

\[
B=\max_q b_q < A=\min_q a_q.
\]

The common interval would be [B,A). This is an intersection of constraints,
not an average of per-query thresholds. For example, [-1,-0.2) and [1,2) are
both valid individual intervals, but they have no common point. Every query
could have correct relative ordering while no shared constant threshold works.

The diagnostic reports interval existence using known validation labels. It
does not choose a threshold or generate new answers. A deployment rule cannot
consult the unknown correct answer to decide its own threshold. Any calibration
experiment would need a separately declared procedure and evaluation boundary.

## What this diagnostic can and cannot establish

The plan authenticates the accepted 2000- and 8000-update reports, replays their
original selections, and checks the older reductions against the already audited
geometry result. An independent auditor recomputes the new reductions and paired
comparisons. There is no neural execution, extra training or protected-data use.

Fresh-two totals reuse 222 questions across two parents; all-three totals add
historical development-selected seed 1729. These are repeated model-query
observations. The diagnostic cannot establish unseen-data generalization or
claim that a proposed intervention will fix the remaining errors.

The [declared plan](../experiments/2026-09-25-bilinear-budget-geometry-plan.md)
fixed these reductions before execution. The [accepted result](../experiments/2026-09-25-bilinear-budget-geometry.json)
links all six complete budget reports and three complete paired reports. The
current inventory remains 37 final checkpoints; the serving model and decision
rule remain unchanged.

## What actually changed

For the two fresh parents, rows below are categories at 2000 updates and columns
are categories at 8000 updates. Each entry counts model-query observations.

| Old category / new category | Exact | Missing only | Extra only | Overlap or tie |
| --- | ---: | ---: | ---: | ---: |
| Exact | 420 | 0 | 0 | 0 |
| Separated, missing only | 4 | 2 | 0 | 0 |
| Separated, extra only | 8 | 0 | 1 | 0 |
| Overlap or tie | 2 | 2 | 4 | 1 |

There are 14 exact-answer gains with no losses. Twelve gains came from queries
whose scores were already separated, while two came from previously overlapping
scores. Eight model-query observations gained strict separation, but six of those still failed
the zero-boundary rule. Better ordering and a correct answer set are separate
milestones, even within one training trajectory.

Including the historical development seed gives 23 exact gains: 20 from already
separated errors and three from overlap. Eleven observations gained separation;
eight of those remain nonexact. At 8000 updates, the 16 nonexact observations
comprise eight missing-only, five extra-only and three overlap/tie cases. They
cover 13 distinct questions, down from 28 at 2000 updates. All remaining errors
are dual-TYPE. These counts do not create new independent evaluation questions.

## The fully separated seed still has no shared threshold

For seed 1730 at 8000 updates, every question is strictly separated. Yet the
dual-TYPE group has A=-1.2183341979980469 from query 184 and
B=-0.09297823905944824 from query 67. To exclude every false member, a constant
t would need to be at least B. To include every true member, it would need to
be below A. Since B>A, both conditions cannot hold.

The shared interval is empty globally and within dual-TYPE for every parent at
both budgets. COLOR and single-TYPE intervals do exist. Thus one constant
threshold cannot make all these fixed-score answers exact, even if chosen with
knowledge of the validation labels. This does not rule out every possible
calibration model, and it does not test a learned query-dependent decision rule.

The next intervention must be separately declared and learned without choosing
its parameters from these validation labels. Further training, a richer scorer
or a learned calibration model are hypotheses, not conclusions of this analysis.

The primary suite passed 90 CPU tests; the independent suite passed 86. Both
actual executions passed on their first attempt. Nine bounded reports preserve
all 1332 budget-specific rows and 666 paired rows; no model was run and no
checkpoint was added. The evidence establishes descriptive geometry of saved
scores, not a causal explanation or a new model-quality result.
