# A weaker margin improves answers but fails the declared gate

The new coefficient-.001 experiment did not repeat the severe answer-quality
regression observed at .1. In a new, fresh paired screen,
treatment improved exact answers from **191/222 to 194/222** and macro F1 from
**.967657 to .974821**. However, strictly separable membership rankings fell
from **168/222 to 164/222**. The fixed gate required both exact-answer and
separation improvements, so this variant was **rejected**. The independently
audited measurements are retained; the gate was not revised after the result.

## A separate coefficient intervention

The earlier .1 experiment remains a failure: its paired result was 192 versus
6 exact answers. A subsequent training-only gradient diagnosis showed why a
small-looking loss coefficient need not imply a small parameter gradient.
That evidence motivated the new .001 hypothesis; it did not identify an
optimal weight or establish the cause of the earlier collapse.

The [declared plan](../evidence/updates/2026-09-25-weak-margin-screen/experiments/2026-09-25-weak-margin-plan.md)
fixed margin m=1, coefficient .001 and seed 1729. Both control and treatment
were trained afresh for 2000 steps, sequentially, before either generated-answer
evaluation. Every other recipe field remained matched. The new control is
the comparator; historical controls were not substituted. Identical seeds do
not guarantee bitwise deterministic CUDA training.

Evaluation retained the existing policy: four guided source paths, six pair
unions, additive learned set scores, FP32 batch size eight and the final batch
of six. All 222 ordered validation queries were evaluated. Expected sets and
group labels were used only for analysis, not prediction or selection. The
[worked lesson](../evidence/updates/2026-09-25-weak-margin-screen/learning/33-loss-weight-is-not-gradient-strength.md)
explains the coefficient, gradient and tensor distinctions.

## What improved, and what did not

| Measurement | Fresh control | Treatment, .001 |
| --- | ---: | ---: |
| Exact selected sets | 191/222 | 194/222 |
| Macro precision | .978083 | .986588 |
| Macro recall | .967390 | .971723 |
| Macro F1 | .967657 | .974821 |
| Strictly separable membership rankings | 168/222 | 164/222 |

Treatment gained 11 exact answers and lost eight previously exact answers,
for a net gain of three. Group exact counts did not regress: COLOR remained
101/103, single-type queries improved 45/51 to 46/51, and dual-type queries
improved 45/68 to 47/68. All 888 source paths per arm were valid, terminated
and eligible; selected sets were eligible. The four-query separation decline
was in dual-type queries, from 19/68 to 15/68. These measurements and their
unrounded values are preserved in the
[portable report](../evidence/updates/2026-09-25-weak-margin-screen/experiments/2026-09-25-weak-margin-screen.json).

These two quality properties answer different questions. Exact selection asks
whether the chosen candidate set equals the complete teacher set. Strict
separation asks whether every true entity outranks every eligible nonmember:

\[
\min_{i\in T_q} z_{qi}>\max_{j\notin T_q,\ j\ne s_q}z_{qj},
\]

where T_q is the teacher set and s_q the excluded subject. Selecting the right
set from ten generated candidates does not require globally separating all
1024 eligible entities. Conversely, a well-separated score vector cannot
ensure that generation offers the correct candidate. The observed answer
improvement therefore does not imply improved global membership ordering.

## Decision and limits

The [primary summary](../evidence/updates/2026-09-25-weak-margin-screen/campaign/summary.json)
and [independent audit](../evidence/updates/2026-09-25-weak-margin-screen/campaign/independent-audit.json)
establish that the saved evidence satisfies the execution and reconstruction
contracts. The [separate decision](../evidence/updates/2026-09-25-weak-margin-screen/campaign/decision.json)
accepts that evidence while rejecting the fixed variant. There is no
additional-seed replication, inference-default change or model promotion.
Both final checkpoints are retained in the updated
[27-run inventory](../evidence/updates/2026-09-25-weak-margin-screen/experiments/2026-09-25-weak-margin-model-version-inventory.json);
inventory consistency does not confer model quality or acceptance.

This is one adaptively chosen development experiment on one seed and 222
reused validation queries. Earlier validation results helped motivate it;
the improvement is not an unbiased estimate of generalization or evidence
of a reliable across-seed benefit. The protected 191-query test remains
unused. No causal optimizer mechanism, serving-performance advantage or
quality advantage over the exact graph/compiler oracle is established.

The [dated evidence update](../evidence/updates/2026-09-25-weak-margin-screen/README.md)
preserves this mixed result alongside the earlier failures.
