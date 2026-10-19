# What unions and intersections can recover from eight generated sources

**Status: independent audit passes; the fixed quality gate rejects the variant.**
This follow-up separates a label-free candidate policy from label-assisted
representability diagnostics. It uses saved Pokémon validation predictions
from the independently audited eight-branch experiment, with no new weights,
neural generation, protected-test evaluation or external dataset.

## Prediction and diagnosis answer different questions

The fixed predictor retains all 36 existing candidates and appends 28 pairwise
intersections. The same learned head scores all 64 candidates by canonical sum
of member logits. Existing slots retain tie precedence. Empty intersections
remain eligible empty sets with score zero. Labels never choose which pair or
operation to use.

The separate diagnostic enumerates each nonempty subset of eight sources: 255
pure unions and 255 pure intersections per query. It asks whether an operation
can equal the teacher set, recording a minimal-source witness in a fixed order.
These witnesses are oracle diagnostics, not predictions. They do not cover
arbitrarily nested set expressions, differences, complements or newly generated
paths. A found exact candidate is only an opportunity for a learned selector.

## A ceiling known before measurement

The existing eight-source evidence gives 636/666 observations with every true
member present somewhere among the sources. The other 30 cannot become exact
through unions or intersections alone. Two further observations already contain
an exact candidate but the unchanged scorer selects a different set. Any new
copy of the exact set has the same score and later tie precedence, so appending
it cannot repair the decision.

Consequently, any candidate extension composed from these sources by unions
and intersections, retaining the same scorer and old-first ties, has selected
exactness at most 634/666. This upper bound already rules out full parity for
that restricted approach; it does not say the bound is achievable. An exact
intersection additionally requires each participating source to contain every
true member. It can remove extras, but cannot repair a source's omissions.

## Evidence boundaries

The [declared plan](../evidence/updates/2026-09-25-eight-source-set-algebra/experiments/2026-09-25-eight-source-set-algebra-plan.md)
and [portable result](../evidence/updates/2026-09-25-eight-source-set-algebra/experiments/2026-09-25-eight-source-set-algebra.json)
bind the exact input chain, inference procedure and per-query diagnostics.

The upstream three-seed baseline is 603/666 exact answers with macro F1
0.9834927532993669. The same 222 validation queries are evaluated under each
seed; they are not 666 independent held-out queries. The fixed screen requires
strict pooled exact improvement, no per-seed exact/F1 regression, and no pooled
group exact regression. Availability improvements alone cannot pass that gate.

A separate bitset/rational-sum audit reconstructs saved set arithmetic. It
does not replay neural generation. Raw upstream reports and checkpoint payloads
remain local dependencies identified by hashes. Policy promotion, deployment
verification, final-test quality and matched-quality serving benefit remain
separate requirements.

## A pooled gain that fails the per-seed requirement

| Seed | Exact answers /222: old → new | Macro F1: old → new |
| --- | ---: | ---: |
| 1729 | 201 → 201 | 0.979925518 → 0.979542026 |
| 1730 | 205 → 206 | 0.990696934 → 0.991794120 |
| 1731 | 197 → 198 | 0.979855808 → 0.982673178 |

Pooled exactness improves 603→605/666 with two gains and no exact losses;
macro F1 improves 0.9834927532993669→0.9846697745678721. Fourteen outputs change,
all to intersections; none selects an empty set. COLOR and dual-TYPE exactness
stay at 309 and 155, while single-TYPE improves 139→141. The two gained answers
are Lycanroc TYPE under seed 1730 and Ekans TYPE under seed 1731.

Seed 1729's F1 regression fails the unchanged gate. In its Archaludon TYPE query,
the old selection has 131 correct members and 65 extras. The new selection has
68 correct members and no extras, omitting 63 correct members. Precision rises
66.84%→100%, but recall falls 100%→51.91% and F1 falls 80.12%→68.34%. Both answers
are inexact, so exact counts do not expose that deterioration.

The [independent audit](../evidence/updates/2026-09-25-eight-source-set-algebra/campaign/independent-audit.json)
reconstructs 5,328 sources, 42,624 slots and 339,660 pure-operation/subset entries.
The [separate decision](../evidence/updates/2026-09-25-eight-source-set-algebra/campaign/decision.json)
accepts the evidence while rejecting the prediction variant. Runner and auditor
passed 38 and 39 focused tests. No automatic higher-order policy was evaluated
after the failed screen.

## Exhaustive pure operations leave a small opportunity and a larger limit

Exact availability is 605 for the old pool, 607 after pair intersections, 605
among all pure unions, 515 among all pure intersections, and 610 across either
pure family. All larger pure unions therefore add no new exact opportunity.
Intersections add five beyond the old pool: two need two sources and three need
three sources. These witnesses were selected with labels; no three-way
prediction policy is established.

For the exhaustive pure families, 610 available exact answers minus the two
unchanged-scorer misses yields a selected-exact upper bound of 608/666. This is
tighter than the broader 634 bound for arbitrary union/intersection expressions,
but neither is an achievable-quality claim. More pure operations on these fixed
sources cannot establish parity. Further work needs different candidate
membership, learned scoring, or a separately justified richer construction;
all work remains within Pokémon.
