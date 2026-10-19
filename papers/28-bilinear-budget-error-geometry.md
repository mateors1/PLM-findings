# How saved error boundaries change with the training budget

Status: completed, independently audited and separately accepted as descriptive
evidence. Primary reductions and independent audit succeeded on their first
attempts. There is no new quality gate, checkpoint or promoted policy.

The [8000-update replication](27-bilinear-budget8000-replication.md) established
434/444 exact observations across the two fresh seeds, compared with 420/444
under their matched 2000-update residual fits. Those accepted quality results
motivate a more specific question: which kinds of boundary failures changed,
and which remain? This diagnostic studies the saved scores, without training,
new predictions, threshold fitting or a new quality screen.

## What a boundary measurement means

For a query q, the saved scorer provides a vector z in R^1025: one score per
product ID 1024 through 2048. Its subject is excluded from candidate membership.
Let T contain the true nonself members and N the false nonself members. Both
must be nonempty. Define

\[
a_q=\min_{i\in T_q}z_{qi},\qquad
b_q=\max_{i\in N_q}z_{qi},\qquad
g_q=a_q-b_q.
\]

The weakest true member, a, and the strongest false member, b, describe the
hardest membership boundary. A positive gap means the true members rank above
all false members. It does not ensure the model's fixed zero threshold lies
between them. For the actual rule z>0, a set is exact if and only if a>0 and
b<=0. One missed or extra member makes the entire answer nonexact even when
most memberships are correct.

The four exclusive categories, in fixed matrix order, are:

| Category | Meaning |
| --- | --- |
| exact_zero | a>0 and b<=0: every membership is correct at zero |
| separated_missing | a>b but the answer is nonexact with missing members only |
| separated_extra | a>b but the answer is nonexact with extra members only |
| overlap_or_tie | the answer is nonexact and a<=b |

All scores must be finite and exactly representable as FP32 values. Membership
is replayed from the saved scores at strict z>0, in ascending ID order, and
checked against the accepted IDs, exactness, false positives, false negatives
and strict separation. Equal extrema use the lowest product ID as witness;
shared-query extrema use the lowest query index. There is no tolerance or
rounding. The expected-set hash uses the exact historical JSON serialization.

## Pairing budgets without mixing parents

The inputs are six accepted child reports: budgets 2000 and 8000 for each of
parent seeds 1729, 1730 and 1731. Each report contains the same 222 ordered
validation queries and 1025 product columns. Query index, subject, dimension,
group, prompt IDs and expected-set IDs must align across all reports. Checkpoint
identities are compared through authenticated JSON records; payloads are never
loaded. Each 8000-update result must identify its own matched 2000-update
historical comparator and original parent.

For each parent, every query retains both budgets' a, b and g values and the
differences

\[
\Delta a_q=a_q^{8000}-a_q^{2000},\quad
\Delta b_q=b_q^{8000}-b_q^{2000},\quad
\Delta g_q=g_q^{8000}-g_q^{2000}.
\]

These are descriptive changes in that parent's score coordinates, not calibrated
confidence, cross-parent averages or causal explanations. A 4x4 count matrix
records the transition from the 2000 category (row) to the 8000 category (column).
Per-seed and per-group matrices retain all 16 cells, including zero cells.
Exact-answer gains/losses and strict-separation gains/losses are counted
separately. Exact gains/losses must reproduce the already accepted paired budget
comparisons, rather than introduce another quality estimate.

The [earlier 2000-update geometry diagnostic](../evidence/updates/2026-09-25-bilinear-budget-geometry/dependencies/runs/learning/bilinear-error-geometry-v1/summary.json)
provides an additional check. Recomputed 2000 rows, totals, groups and pooled
overlaps must equal its authenticated results exactly. Its historical checkpoint
count remains historical; this new diagnostic creates no checkpoint and leaves
the current count of 37 unchanged.

## Shared intervals are descriptive bounds

Within one seed, either across all its queries or within one group, define

\[
A=\min_q a_q,\qquad B=\max_q b_q.
\]

For a hypothetical rule z>t, all saved answers could be exact at one shared
threshold only when B<A; the label-informed feasible interval is [B,A). The
left endpoint is inclusive because false members must not satisfy z>t, whereas
the right endpoint is exclusive because true members must satisfy it strictly.
The diagnostic records existence and witnesses, not a chosen t. It generates
no alternate answers and makes no deployment or performance claim from these
intervals. No pooled or cross-parent threshold interval is computed.

At each budget the summary also records, for every shared failed query, which
seeds fail and how failure sets overlap. Fresh-two and all-three count summaries
remain distinct. Historical seed 1729 was used for development and selection;
the other two seeds supplied fresh budget replication. The 444/666 totals count
model-query observations over the same 222 validation questions. They do not
provide 444/666 independent held-out samples or a significance test.

## Observed transitions

The fresh seeds gained 14 exact answers and eight strictly separated answers,
with no exactness or separation losses. Twelve of the 14 exact gains came from
answers that were already strictly separated at 2000 updates: four with missing
members and eight with extra members. The other two exact gains came from
overlapping or tied score ranges.

The complete fresh-two transition matrix is below. Rows describe the old
2000-update category; columns describe the new 8000-update category. Every
cell counts model-query observations, including unchanged exact answers.

| 2000 category → 8000 category | Exact | Separated missing | Separated extra | Overlap/tie |
| --- | ---: | ---: | ---: | ---: |
| Exact | 420 | 0 | 0 | 0 |
| Separated missing | 4 | 2 | 0 | 0 |
| Separated extra | 8 | 0 | 1 | 0 |
| Overlap/tie | 2 | 2 | 4 | 1 |

Eight of the nine old overlap/tie observations became strictly separated, but
only two became exact. The other six moved into separated-but-wrong categories.
This explains why a ranking improvement can be real while exact-set accuracy
still has room to improve. All fresh transitions that change exactness or
separation are in dual-TYPE; COLOR and single-TYPE stay exact.

| Seed / role | Exact, 2000 → 8000 | Separated missing, 2000 → 8000 | Separated extra, 2000 → 8000 | Overlap/tie, 2000 → 8000 | Separation gains / losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1730, fresh | 212 → 220 | 3 → 2 | 4 → 0 | 3 → 0 | 3 / 0 |
| 1731, fresh | 208 → 214 | 3 → 2 | 5 → 5 | 6 → 1 | 5 / 0 |
| 1729, historical selected | 207 → 216 | 4 → 4 | 6 → 0 | 5 → 2 | 3 / 0 |

At 8000 updates, the ten remaining fresh nonexact observations comprise four
separated missing, five separated extra and one overlap/tie. Including historical
1729 gives 16 nonexact observations: eight separated missing, five separated
extra and three overlap/tie. The all-three paired comparison has 23 exact gains,
11 separation gains and no losses of either kind. Its full transition matrix,
along with every per-seed and per-group matrix, is retained in the reports.

## Why full ranking separation still does not imply one common threshold

Every seed lacks a shared interval across all queries and across its dual-TYPE
queries at both budgets. At 8000 updates the dual-TYPE interval extrema are:

| Seed | A = minimum true boundary across queries | B = maximum false boundary across queries | Shared interval exists |
| --- | ---: | ---: | --- |
| 1729 | -2.830646514892578 | 1.1218247413635254 | No |
| 1730 | -1.2183341979980469 | -0.09297823905944824 | No |
| 1731 | -0.8291666507720947 | 1.5331709384918213 | No |

Seed 1730 is the instructive case: all 222 individual queries are strictly
separated, including all 68 dual-TYPE queries. Each query separately admits
some label-informed threshold interval. Their intersection is nevertheless
empty. Including the weakest true SOLROCK member (query 184) would require
t<-1.2183341979980469, while excluding the strongest false FEZANDIPITI member
(query 67) requires t>=-0.09297823905944824. No single t satisfies both.

The result rules out a shared scalar threshold as a complete repair for these
saved dual-TYPE scores. It does not test query-conditioned thresholds, fit a
threshold policy, establish representational impossibility or predict what
further training would do. COLOR and single-TYPE shared intervals exist for
every seed at both budgets; their existence is also recorded without selecting
an endpoint or producing new answers.

## Which shared questions still fail

Across all three seeds, the number of distinct validation questions with at
least one failed answer falls from 28 to 13. At 8000 updates, 209 of the 222
questions are exact under all three saved models; ten questions fail under one
seed and three fail under two. None fails under all three. The three shared
two-seed failures are CARBINK, FEZANDIPITI and SKARMORY, each under seeds 1729
and 1731. Seed 1730's two failures are SOLROCK and TOXTRICITY.

These label-informed overlap counts describe complementary errors. They do not
supply a rule that can identify the correct model's answer at serving time.
No answer routing or ensemble predictor is constructed or evaluated here.

## Execution and evidence boundaries

The primary implementation imports a hash-pinned compatible pure geometry helper
from the earlier diagnostic. The independent stdlib auditor uses a separately
authored, separately pinned old audit helper and does not import primary
scientific arithmetic. Both authenticate the accepted summary/audit/decision
chains and the referenced score reports. Neither recursively opens upstream
input manifests, which contain model and training dependencies outside this
diagnostic's scope.

Before the real reduction, synthetic tests covered signs and ties, false membership
replay, deterministic witnesses, FP32 checks, query and column drift, all 16
transitions, historical equality, incomplete evidence, actual production receipt
schemas and immutable outputs. Test receipts bind exact helper/source/test
dependencies, observed exit code zero and stdout. The primary chain-only
preflight authenticated schemas and identities without computing new reductions.
The primary suite passed 90 synthetic tests, and the independent audit suite
passed 86, with zero skips. The primary reduction and independent actual audit
each completed once with observed exit code zero. Recomputed historical 2000
rows, totals, groups and overlaps matched the original diagnostic exactly.

The publication packet retains nine complete reduced reports: one 222-row
report per seed/budget and one 222-row paired report per seed. Portable aliases
must be byte-identical to those outputs, with complete query coverage and each
file below 512 KiB. Raw full-score child reports remain hash-bound local
dependencies. Their hashes establish identity, not durable remote archival.

No Torch, GPU, checkpoint payload, graph database, training examples, protected
data, forward pass or optimizer step is involved. Acceptance concerns the
diagnostic evidence, not a new quality gate, inference policy or serving
promotion. The owner decision accepts these evidence reductions and preserves
the current 37-checkpoint inventory and serving policy.

## Evidence

The [packet index](../evidence/updates/2026-09-25-bilinear-budget-geometry/README.md)
links all nine complete reduced reports and their exact portable copies. The
[primary summary](../evidence/updates/2026-09-25-bilinear-budget-geometry/campaign/summary.json),
[independent audit](../evidence/updates/2026-09-25-bilinear-budget-geometry/campaign/independent-audit.json)
and [owner decision](../evidence/updates/2026-09-25-bilinear-budget-geometry/campaign/decision.json)
separate calculation, verification and acceptance. The
[frozen plan](../evidence/updates/2026-09-25-bilinear-budget-geometry/experiments/2026-09-25-bilinear-budget-geometry-plan.md)
declares the allowed reductions and input boundary; the
[portable result](../evidence/updates/2026-09-25-bilinear-budget-geometry/experiments/2026-09-25-bilinear-budget-geometry.json)
provides their complete index. [Lesson 49](../evidence/updates/2026-09-25-bilinear-budget-geometry/learning/49-how-error-boundaries-change-with-training.md)
explains the difference between individual score separation and a common
threshold across queries.
