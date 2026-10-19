# More coverage, fewer exact answers

**Status: independent audit passes; the fixed quality gate rejects the policy.**
One extra generation path was started from the highest positive membership
score outside the existing eight-source union, excluding the subject. If no
eligible product existed, the old candidate pool remained. Active rows appended
the new original set and its unions with each old source: 45 slots instead of
36. Labels did not choose anchors, candidates or scores. Old slots retained
tie precedence under the unchanged sum-of-member-logits score.

The [fixed plan](../evidence/updates/2026-09-25-coverage-seeking-branch/experiments/2026-09-25-coverage-seeking-branch-plan.md)
requires per-seed exact/F1 nonregression, pooled exact improvement, dual-TYPE
exact improvement and nondecreasing pooled group exactness. No threshold,
anchor-count or group-routing sweep was performed after seeing outcomes.

## The fixed gate rejects the policy

| Seed | Exact answers /222: baseline → new | Macro F1: baseline → new | Gains / losses |
| --- | ---: | ---: | ---: |
| 1729 | 201 → 200 | 0.979925518 → 0.983592837 | 1 / 2 |
| 1730 | 205 → 202 | 0.990696934 → 0.990659956 | 0 / 3 |
| 1731 | 197 → 198 | 0.979855808 → 0.982103265 | 2 / 1 |
| Pooled | 603 → 600 | 0.983492753 → 0.985452019 | 3 / 6 |

All three seeds reproduce their fresh head vectors and historical rank-one
continuations exactly. All active new paths are valid, terminated and unique.
Nevertheless, seeds 1729 and 1730 lose exact answers, seed 1730 loses F1, and
pooled exactness declines. COLOR falls 309→306, single-TYPE 139→137, while
dual-TYPE improves 155→157. The policy fails its predeclared quality gate.

Three exact gains are Aegislash TYPE under seed 1729, and Armarouge TYPE and
Azumarill TYPE under seed 1731. All six exact losses introduce one false member
without omitting any correct member. This illustrates why a small set error can
have a small F1 effect but still destroy exactness. The
[portable report](../evidence/updates/2026-09-25-coverage-seeking-branch/experiments/2026-09-25-coverage-seeking-branch.json)
preserves all 666 compact query diagnostics and the separate decision.

## Positive scores answer different conditional questions

The preceding diagnosis found positive scores for 1,394/1,407 omitted correct
member occurrences. This experiment activates 205 anchors, of which only 30
are correct and 175 false. The former statistic conditions on already knowing
the member is correct; the latter evaluates a label-free selection rule among
all uncovered products across all queries. Their denominators and populations
differ. The 99.08% diagnostic is not precision of the anchor policy, whose
observed precision is 30/205, or 14.63%.

The new source pool adds 488 correct-member occurrences and 13,281 false-member
occurrences beyond prior coverage. These are source-pool occurrences, not the
final answer's false positives. The selected answers contain 1,089 false-positive
and 1,632 false-negative occurrences. Repeated products and queries are not
independent samples; 666 observations reuse 222 distinct validation queries.

## Availability improved while selection deteriorated

Exact candidate availability increases 605→608; all-source truth containment
increases 636→642. But available-exact selection misses increase 2→8. The
unchanged scorer selects an added slot in only 16 observations and final exact
answers fall to 600. This separates useful source expansion from reliable
final selection. More candidates cannot lower the maximum eligible model
score, but the higher-scoring candidate can be incorrect.

For a union with an old candidate S, the score increment is the sum of logits
of newly added products. A false extra with positive logit can therefore make
a wrong superset score above an exact set. The old exact candidate stays
available, yet loses selection. The failed experiment supports investigating
candidate discrimination as well as coverage; it does not validate a new
threshold, subgroup rule or training recipe.

## Execution and evidence boundary

The experiment reuses 5,328 authenticated historical source paths. It freshly
executes 666 continuation controls and 666 extra batched paths: 205 active,
461 numerical-padding rows excluded from new candidates. All original batches
remain eight rows, with six in the final batch. There are 25,821 candidate slots.
No fresh replay of the eight historical paths is claimed.

The runner passed 28 focused tests before measurement and the independent
auditor passed 38 before its actual reconstruction. The
[dated evidence addition](../evidence/updates/2026-09-25-coverage-seeking-branch/README.md)
preserves the passing evidence audit and separate policy rejection.
Controls, new-path GPU time and active/padding work are recorded separately.
The field set_scoring_seconds includes postprocessing, scoring and per-query
diagnostics. The idle LM Studio service remained resident; these are descriptive
incremental timings, not a controlled serving, energy or throughput comparison.

The baseline remains the accepted offline eight-branch 603/666 policy. There is
no promotion, training, new checkpoint, protected-test use or external dataset.
The inventory remains 27 checkpoints. Raw neural reports and weights remain
local hash-bound dependencies; publishing compact evidence is not a durable
weight archive. Overall oracle parity remains open.
