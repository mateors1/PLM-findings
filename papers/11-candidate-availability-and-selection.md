# Candidate availability limits the saved weak-margin comparison

The two retained runs select every exact answer available in their ten-slot
candidate pools. Control has 191 available and selected exact answers;
treatment has 194. A different reranker restricted to those pools cannot
increase either exact count. This is a single-seed diagnosis over the same
222 validation queries, not a new model result or a reversal of the rejected
weak-margin screen.

## Freeze candidates and substitute saved scores

Each run supplied four generated paths, six pair unions and a prompt-only
membership vector. After authenticating the earlier experiment and reconstructing
its original choices, the diagnostic scored each pool with both saved vectors.
The selector uses the canonical sum of member logits and the first eligible
slot in a tie. Labels enter metrics and availability checks only.

| Candidate pool | Control scores: exact / macro F1 | Treatment scores: exact / macro F1 |
| --- | ---: | ---: |
| Control | 191 / 0.967657 | 191 / 0.965878 |
| Treatment | 194 / 0.974821 | 194 / 0.974821 |

For exact count, both ordered decompositions allocate +3 to the pool change and
0 to the saved-score change; interaction is zero. F1 behaves differently:
substituting treatment scores on the control pool reduces macro F1 by 0.001779,
while the same substitution on treatment pools leaves the aggregate unchanged.
The resulting F1 interaction is +0.001779. These are conditional comparisons,
not unique causal contributions of independently replaceable modules. The head
also guides generation, and trained components share parameters.

The [portable report](../evidence/updates/2026-09-25-candidate-score-decomposition/experiments/2026-09-25-candidate-score-decomposition.json)
contains unrounded metrics, group tables, identities and the changed-query list.

## Missing candidates, rather than missed available exact answers

| Measure | Control | Treatment |
| --- | ---: | ---: |
| Exact availability among four originals | 160 | 166 |
| Exact availability among ten slots | 191 | 194 |
| Available exact answer missed by selection | 0 | 0 |
| No exact candidate among ten slots | 31 | 28 |
| All true entities present across four sources | 202 | 208 |

Twenty control queries and fourteen treatment queries lack at least one true
entity from every source; no union of those sources can recover it. A further
eleven and fourteen, respectively, contain all true entities collectively but
have no exact candidate in the ten slots. Containment alone does not ensure
exactness: unions retain extras and the allowed combinations are restricted.

The transition table contains 183 queries exact in both runs, 20 unavailable
in both, 11 becoming available and exact, and eight losing availability and
exactness. No query enters an available-but-missed state. Dual-TYPE accounts
for 23 of 31 control failures and 21 of 28 treatment failures. The
[learning note](../evidence/updates/2026-09-25-candidate-score-decomposition/learning/34-candidate-availability-and-selection.md)
explains the indicators, conceptual tensors and score decomposition equations.

## Evidence status and next experiment

The independent implementation checked all 888 cells, 8,880 slot scores and
1,776 raw paths, using exact rational sums to cross-check canonical floating
scores. The primary runner's 23 focused tests and auditor's 22 tests passed.
The [audit](../evidence/updates/2026-09-25-candidate-score-decomposition/campaign/independent-audit.json)
and [separate decision](../evidence/updates/2026-09-25-candidate-score-decomposition/campaign/decision.json)
accept this saved-evidence reconstruction. There was no neural replay,
training, new checkpoint or protected-test measurement. The inventory remains
27; the weak-margin gate stays failed. Four cells do not create 888 independent
test queries or four new model versions.

This result motivates testing more diverse generated candidates on the accepted
checkpoint family. One proposed next experiment widens first-choice branching
from four to eight, then scores originals and pairs with the unchanged head.
It is not yet executed or accepted. More candidates may expose selector errors
and increase cost; availability improvement alone would not prove better serving.

Separately, the [primary-source data review](../evidence/updates/2026-09-25-candidate-score-decomposition/experiments/2026-09-25-learned-retrieval-data-review.md)
distinguishes co-outfit labels, observed behavior events and website recommendation
links. It recommends a feasibility audit of a separately defined behavioral
relation, not adoption of a judged-compatibility dataset. No data was downloaded
or new task measured. The user subsequently parked this external-data track:
active work stays within Pokémon while pursuing oracle parity and the intended
evolutions, and another dataset requires an explicit request. Historical evidence
copies retain the earlier recommendation; this scope decision supersedes it.
