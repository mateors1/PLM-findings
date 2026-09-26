# Reading the remaining bilinear errors

**Status: independently audited descriptive evidence accepted; no new quality gate.**
Among the 24 incorrect outputs from the two fresh replication seeds, 15 have
strictly separated true/false scores but fail the fixed zero threshold; nine
have overlapping or tied ranges. Including the historical selection seed gives
25 separated mistakes and 14 overlap/tie mistakes among 39 incorrect outputs.
No single constant threshold can make every saved answer exact within any
seed, even when the diagnostic restricts attention to dual-TYPE queries.

The [declared plan](../evidence/updates/2026-09-25-bilinear-error-geometry/experiments/2026-09-25-bilinear-error-geometry-plan.md)
analyzes the accepted [replication results](23-bilinear-parent-seed-replication.md).
This is posthoc explanation of existing errors, not another model quality gate.
The [independent audit](../evidence/updates/2026-09-25-bilinear-error-geometry/campaign/independent-audit.json)
and [separate owner decision](../evidence/updates/2026-09-25-bilinear-error-geometry/campaign/decision.json)
accept the saved-evidence reconstruction without selecting a new predictor.

This diagnostic examines the saved validation scores from the accepted bilinear
children: historical selection seed 1729 and fresh replication seeds 1730/1731.
It does not train a model, evaluate new examples or choose another prediction
policy. The purpose is to distinguish two reasons a complete answer can fail:
wrong placement relative to zero, or incorrect ordering of true and false members.

For one query, exclude its subject from the product columns. Let T be the true
members and N the false members. From the saved scores z define

\[
a=\min_{i\in T}z_i,\qquad b=\max_{i\in N}z_i,\qquad g=a-b.
\]

The fixed experimental rule in these saved reports includes each nonself
product with z>0. Its complete answer is exact if and only if a>0 and b<=0.
A true member scoring exactly zero is missing; a false member scoring zero is
correctly excluded. No tolerance or near-zero bucket changes these boundaries.

Strict separation asks a different question: is every true member above every
false member? That holds when a>b. A separated but wrong answer has either
missing members only, with a<=0, or extra members only, with b>0. If a<=b,
the true and false score ranges overlap or tie. Four exclusive categories
therefore cover all queries: exact at zero, separated with missing members,
separated with extra members, or overlap/tie.

For a hypothetical rule z>t, a query would be exact for t in [b,a), provided
b<a. For a whole seed/group, define A=min_query(a) and B=max_query(b). A single
constant could separate all those saved labels only if B<A. This calculation
uses the known labels. It reports whether the interval exists without choosing
a threshold, generating another answer or estimating deployable performance.

Saved FP32 scores are read as ordinary finite numbers and checked for exact
FP32 representability. The runner reconstructs the original zero-rule outputs,
errors and separation flags, then reports a, b, g, deterministic lowest-ID
extremum witnesses and incorrect-member IDs/scores. Full score vectors remain
local. Independent audit recomputes the same scientific quantities without
importing the primary classification or aggregation functions.

## Where the mistakes lie

| Seed and role | Exact at zero | Separated, missing only | Separated, extra only | Overlap or tie |
|---|---:|---:|---:|---:|
| 1729, historical selection | 207 | 4 | 6 | 5 |
| 1730, fresh replication | 212 | 3 | 4 | 3 |
| 1731, fresh replication | 208 | 3 | 5 | 6 |
| Fresh two, 444 observations | 420 | 6 | 9 | 9 |
| All three, 666 observations | 627 | 10 | 15 | 14 |

Every individual-seed row covers 222 queries. All 24 fresh-seed mistakes are
in dual TYPE. Historical seed 1729 contributes one separated-extra mistake
in single TYPE and 14 mistakes in dual TYPE. Every COLOR answer is exact.
The diagnostic exactly reconciles the original false-member counts: fresh-two
outputs have 20 extra and 15 missing member occurrences; all-three outputs
have 28 extra and 25 missing. Query counts and member counts measure different
things because one wrong query can contain more than one incorrect member.

The 15 fresh separated mistakes explain why strict-separation count 435 exceeds
exact-set count 420. Their rankings already put every true member ahead of every
false member, yet zero is outside the query's valid interval. The other nine
mistakes contain a true/false ordering failure or tie, so changing a scalar
threshold cannot make those particular saved score vectors exactly correct.
This classification does not identify the training mechanism that caused either
kind of error.

## Why a universal offset cannot repair everything

| Seed | A, lowest true-member score | B, highest false-member score | Shared interval exists? |
|---|---:|---:|---|
| 1729 | -2.868280 | 2.702680 | No |
| 1730 | -4.123544 | 1.429110 | No |
| 1731 | -1.807922 | 3.009808 | No |

These displayed values are rounded for reading; classification uses the exact
saved values without rounding or tolerance. Each overall interval is empty
because B>=A. The same extrema occur within each seed's dual-TYPE group, whose
shared interval is also empty. Consequently, one constant cannot recover every
answer on these frozen scores. This is a statement about simultaneous exact
recovery, not a claim that every threshold change would fail to improve a count.
The experiment neither searches for such improvements nor produces alternatives.

COLOR and single-TYPE groups each have nonempty shared intervals. In the fresh
seeds, their original zero rule is already exact. Historical single TYPE has
one extra-member mistake even though a label-informed shared interval exists.
The group labels belong to evaluation; these calculations do not install group
routing, choose group-specific thresholds or establish a deployable calibration.

## Which questions fail repeatedly

Across the same 222 questions, 194 fail under none of the three children,
19 fail under exactly one, seven fail under exactly two, and two fail under all
three. Thus 28 distinct questions account for the 39 wrong query-seed observations:
19*1 + 7*2 + 2*3 = 39. The two all-seed failures are the TYPE queries for
PKM_FEZANDIPITI and PKM_SKARMORY. The reduced reports preserve the complete
failure-seed lists and deterministic score-extremum witnesses.

This mixture shows that some failures depend on learned parent state while a
small subset recurs across all three. It does not justify selecting whichever
seed happens to answer each labeled query correctly. No routing rule, ensemble
or new predictor is evaluated here.

## Evidence and limits

The [portable report](../evidence/updates/2026-09-25-bilinear-error-geometry/experiments/2026-09-25-bilinear-error-geometry.json)
and [reduced per-query records](../evidence/updates/2026-09-25-bilinear-error-geometry/README.md)
preserve all reductions without copying full score vectors. The primary's 38
synthetic tests cover exact-zero boundaries, empty sets, FP32 validity, tied
extrema, shared intervals, query alignment and immutable outputs. The separate
stdlib auditor has 39 synthetic tests and uses independent reduction arithmetic.
Both actual executions completed successfully on their first attempt.
Neither implementation reads checkpoints, training examples, graph databases or
protected-test data; upstream accepted JSON evidence supplies label authority.

The two fresh models contribute 444 observations, and including the historical
selection model gives 666, all over the same 222 questions. Failure overlap
compares those aligned questions; it does not turn repeated observations into
independent examples. This is posthoc descriptive evidence, without a new quality
gate, checkpoint, selected threshold, protected-test result or serving promotion.


The final-checkpoint count remains 33. Local hashes bind omitted full score
reports but do not establish durable remote archival. A future intervention
would need its own training-only design and declared evaluation contract; these
posthoc validation measurements alone do not choose that intervention or prove
generalization.
