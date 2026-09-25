# Learned retrieval: the next data feasibility question

Date: 2026-09-25. Status: proposal from a repository review, not a selected
dataset, executed experiment, protocol extension or application claim.

Can a small context-conditioned scorer rank independently judged complementary
partners for known items better than context popularity on unseen item pairs?
This narrows the first feasibility question in the
[application hypothesis](2026-09-25-learned-retrieval-hypothesis.md). It does not
replace the unfinished oracle-parity and systems work in Experiment A.

## Data must support the question

First obtain a rights-cleared judgment dataset with a stated compatibility
rubric, positive judgments, explicit negative judgments and unknowns. No such
dataset has been selected or verified in this project. Labels computed from
TYPE/COLOR SAME would remain rule-distillation evidence.

At least two genuine contexts must support meaningful steering. Determine
whether compatibility is symmetric: the current symmetric head cannot model
directional compatibility unchanged. Record catalog version, label sources,
judgment procedure, timestamps, duplicate groups, license and available inputs.
Do not invent contexts just to populate a protocol token.

Split pair groups before constructing graph features or examples. For a
symmetric task, both directions, repeated judgments and context-specific labels
of the same pair stay together. Ensure evaluated entities occur in training;
this is unseen-pair prediction among known entities, not cold start. The current
subject/dimension query split does not provide this pair-group protection.

## Baseline gate before decoder work

Compare training-only lookup, context popularity and one small relational
scorer under the same candidate universe and permitted information. Start with
fully judged candidate panels so unlabeled pairs need not become assumed
negatives. A possible primary metric is mean average precision; the dataset
review must fix the metric, panel construction, uncertainty method and minimum
useful improvement before evaluation. Panel-ranking evidence would not establish
full-catalog retrieval quality.

Existing implementation leads require adaptation:

| Source | Current behavior / feasibility boundary |
| --- | --- |
| `src/plm/corpus/compiler.py` | Derives complete SAME answers from graph attributes; sparse judgments require a separate corpus contract. |
| `src/plm/evaluation/split.py` | Partitions subject/dimension queries; pair grouping must precede feature construction for this proposal. |
| `src/plm/baselines/popularity.py` | A popularity baseline exists; fit only permitted training judgments for the new task. |
| `src/plm/baselines/complex.py` | Samples unobserved pairs as negatives; unknown judgments cannot silently inherit this convention. |
| `src/plm/evaluation/ranking.py` | Ranking evaluator exists; candidate-panel and unknown-label handling need review before reuse. |

If simple baselines cannot be evaluated fairly because labels or split
semantics are unclear, adding decoder machinery does not resolve that problem.
If a small scorer succeeds, PLM must then compete against it with the same
information and disclosed tuning budgets. Any COMPLEMENTARY extension remains
separately versioned; v1 SAME/TYPE/COLOR and its entity IDs remain as declared.
