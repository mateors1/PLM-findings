# Where the six remaining bilinear errors occur

Status: primary diagnostic completed, independently audited and accepted by
the research owner. Campaign `bilinear-budget32000-error-geometry-v1` is descriptive
posthoc evidence, not a new model-quality evaluation.

## Scope

The [plan](2026-09-26-bilinear-budget32000-error-geometry-plan.md) authenticates
the three accepted 32,000-update child reports and reconstructs their original
strict-positive, nonself, ascending-ID selections. No forward pass, optimizer,
checkpoint payload, graph relation, training record or protected example is
read. The model inventory remains 41 checkpoints and the retained application
reference is unchanged.

The primary summary SHA256 is
`9d5ebfce23e58e5cef6fa9c9747db687ab921a9923c65ad6c81db23ae60c21d8`.
The root observed its successful terminal exit (`e75778`) separately from the
script's own report; the execution receipt SHA256 is
`8a76387a2d69f110ad498b379d61ed18ac147f0fec37d10c226c3e7b787b341d`.

The independent audit passed with observed exit 0 (`b29a47`), audit SHA256
`78ab0f8fbbadadf59e20a6b11dac1f537a65e76f193901c5f36ad29c77674025`.
The separate owner decision is
`a9defd6e81f0902669a73307d8eaf7107e7da6a2f16ee4250d539d6916189567`.
Primary synthetic checks passed 71 tests and auditor checks passed 41, each
also rerun by an independent reviewer before real execution. The audit checks
every reduced row, intervals, counts and overlap; it is not a neural replay.

Portable evidence: [summary](2026-09-26-bilinear-budget32000-error-geometry.json),
[seed 1729](2026-09-26-bilinear-budget32000-error-geometry-seed-1729.json),
[seed 1730](2026-09-26-bilinear-budget32000-error-geometry-seed-1730.json),
[seed 1731](2026-09-26-bilinear-budget32000-error-geometry-seed-1731.json),
[audit](2026-09-26-bilinear-budget32000-error-geometry-audit.json), and
[decision](2026-09-26-bilinear-budget32000-error-geometry-decision.json).
GitHub publication is a separate receipt, not established by these local files.

## Findings

All 666 saved model-query outcomes reconcile with the accepted upstream counts:
660 exact, four separated missing-member errors, one separated extra-member
error and one ranking-overlap error. There are seven false-negative entity
decisions and one false-positive decision. All nonexact rows are dual-TYPE.

| Seed | Subject query | Error category | Missing IDs | Extra IDs |
| --- | --- | --- | --- | --- |
| 1729 | FEZANDIPITI / TYPE | Ranking overlap | 1407, 1451 | None |
| 1729 | HATTERENE / TYPE | Separated, missing members | 1306 | None |
| 1729 | SKARMORY / TYPE | Separated, missing members | 1894 | None |
| 1729 | SWANNA / TYPE | Separated, missing members | 1138, 1825 | None |
| 1731 | FEZANDIPITI / TYPE | Separated, missing members | 1361 | None |
| 1731 | FLYGON / TYPE | Separated, extra member | None | 1452 |

These are five distinct subject/dimension queries. FEZANDIPITI fails in two
seeds, with different missing candidates. The other four fail in one seed each;
217 of the 222 shared queries are exact in all three. Seed 1730 is exact on
all 222. This is repeated development evidence, not 666 independent examples
or a protected-test result.

## A constant cutoff is insufficient

For each query, let `a` be the smallest true-member score and `b` the largest
nonmember score, excluding self. An exact threshold `t` must satisfy
`b <= t < a`. For one constant across a group, the intersection exists only if
`B=max(b) < A=min(a)`.

| Seed | A over dual-TYPE queries | B over dual-TYPE queries | Shared interval exists? |
| --- | ---: | ---: | --- |
| 1729 | -4.362809 | -0.667977 | No |
| 1730 | 0.804768 | -1.685111 | Yes; zero is inside |
| 1731 | -0.596924 | 0.217570 | No |

Table values are rounded for display only; classifications use exact saved
FP32 values and direct comparisons, without a tolerance. Full values and
boundary witnesses are preserved in the per-seed reports.

Seed 1731 is especially informative: every query has separated true/false
rankings, but FEZANDIPITI requires a cutoff below `-0.5969235897064209` to
recover its missed member, whereas FLYGON requires a cutoff at least
`0.21756982803344727` to remove its extra member. Those requirements conflict.
One constant cannot fix all its answers, even within the dual-TYPE group.

Seed 1729 has an additional limitation: FEZANDIPITI's minimum true score is
`-4.3628058433532715`, below its maximum false score `-2.393548011779785`.
No scalar cutoff can recover that query's exact set. The other three failures
in that seed are individually separated.

## Interpretation and next step

This rules out a complete repair by a single score offset on these saved
outputs. It does not establish that a learned query-dependent calibration or
feature adaptation would generalize. No threshold was selected and no repaired
predictions were generated. More parameters, longer optimization and agreement
across independent parents remain hypotheses requiring separate experiments.

The immediate engineering gap is also explicit: the accepted bilinear scorer
is outside the standard serving path. The
[integration design](2026-09-26-bilinear-serving-integration-design.md) records
how to test an opt-in adapter with faithful set provenance, label-free loading
and numerical-shape parity. It is not an implemented deployment or a speed
claim. Any future benchmark must measure the accuracy and performance of that
same inference path together.

See [Lesson 52](../learning/52-reading-the-last-membership-errors.md) for the
tensor and decision-boundary explanation. The canonical-paper owner retains
ownership of synthesis and immutable paper history; this diagnostic does not
promote a checkpoint or overwrite the queued paper revision.
