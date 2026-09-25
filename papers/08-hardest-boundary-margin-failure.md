# A hardest-boundary training objective failed its first screen

The declared margin intervention reduced exact answers from **192/222 to 6/222**
against a fresh seed-1729 control. Independent reconstruction confirmed the
saved predictions, scores, metrics and failed gates. The fixed variant is
rejected and stops before replication at seeds 1730 and 1731.

## Hypothesis and fixed comparison

The [previous diagnosis](07-ranking-and-threshold-limits.md) found that scalar
cutoffs could not repair many overlapping membership rankings. The new objective
directly penalized the weakest true member failing to outrank the strongest
nonmember:

```text
L_margin = mean_q max(0, 1 + max_negative(z_q) - min_true(z_q))
L_treatment = L_control + 0.1 * L_margin
```

The subject is excluded from both classes. Zero hinges remain in the query
mean, the reductions use FP32, and tied extrema share their gradients. The
existing token cross-entropy and two membership losses remain in the objective.
This adds no parameters, but it updates embeddings shared with generation.

The [preregistered plan](../evidence/updates/2026-09-25-margin-screen/experiments/2026-09-25-symmetric-margin-plan.md)
fixed margin 1.0, coefficient 0.1, full 2000-step training, seed 1729 and the
existing inference policy. Both arms trained from initialization under the same
source/runtime before either final generated-answer evaluation. The historical
191/222 exact result was context; the newly trained control defined the gate.

## Outcome

| Validation measure | Fresh control | Margin treatment |
| --- | ---: | ---: |
| Exact answers, out of 222 | 192 | 6 |
| Macro F1 | 96.77% | 61.56% |
| Strictly separated membership rankings | 167 | 0 |
| Zero-threshold membership exact answers | 110 | 0 |
| COLOR exact, out of 103 | 103 | 2 |
| Single-TYPE exact, out of 51 | 47 | 0 |
| Dual-TYPE exact, out of 68 | 42 | 4 |

One exact answer was gained and 187 were lost. All 1776 source paths across
both arms were valid, terminated and eligible; all 4440 candidate slots passed
the saved-score audit. The output protocol remained intact while relation
quality degraded. The treatment selected pair unions in 210 queries versus 29
for the control, with mean selected size 232.97 versus 125.28.

The [portable result](../evidence/updates/2026-09-25-margin-screen/experiments/2026-09-25-symmetric-margin-screen.json)
binds the primary summary, separate audit, owner rejection, source archive and
checkpoint identities. Acceptance of the negative evidence is distinct from
acceptance of the model variant. No inference default changed.

## What this teaches, and what remains unknown

This was not a case of better membership ranking sacrificing generated answers:
the ranking itself failed. Final validation symmetric BCE worsened from
0.00873694 to 0.68069418; treatment margin loss was 1.13373298 and every one of
the 222 validation hinges remained active. Token CE rose from 0.12961999 to
0.14644119. These diagnostics motivate investigating optimization, but do not
establish its causal mechanism.

A small coefficient does not imply a small gradient contribution. Class-average
BCE spreads supervision over many entities; the hinge concentrates it on the
extreme scores. Shared embeddings can transmit those updates into generation.
Gradient competition is therefore a hypothesis, not a demonstrated explanation
or evidence that a smaller coefficient would succeed. Any next intervention
requires a new declared experiment; the failed gate will not be retuned.

A separate [post hoc saved-score diagnosis](../evidence/updates/2026-09-25-margin-screen/diagnosis/summary.json)
measures the compression directly: mean per-query non-subject logit range fell
from 44.80 to 0.245, and mean population standard deviation from 7.96 to 0.0376.
Mean true-boundary minus negative-boundary gap fell from 2.665 to -0.134.
The mean unit-margin hinge was actually lower for treatment (1.134 versus 1.216)
despite its much worse exactness and separation count. Averaging the surrogate
penalty therefore concealed the failed behavior. These are descriptive checks
on the saved vectors, not a new model execution or a causal experiment.
The [separate diagnosis audit](../evidence/updates/2026-09-25-margin-screen/diagnosis/independent-audit.json)
recomputed all 444 rows, group statistics and 20 logged training-step records
exactly, including independently calculated population standard deviations.
These are ten metric rows per arm; the [wording clarification](../evidence/updates/2026-09-25-margin-screen/CLARIFICATION.md)
preserves that distinction from model checkpoint artifacts in the copied lesson.

The [worked lesson](../evidence/updates/2026-09-25-margin-screen/learning/31-hardest-boundary-margin.md)
explains the tensor shapes, derivatives, loss averaging and version boundaries.

## Traceability and limits

The [updated inventory](../evidence/updates/2026-09-25-margin-screen/experiments/2026-09-25-margin-model-version-inventory.json)
verifies 25 final checkpoint records, including the two new runs. The earlier
23-checkpoint inventory remains a valid historical snapshot. Both new models
remain preserved despite rejection; neither is promoted.

Implementation verification included 818 passing CPU-suite tests, nine skips,
25 runner tests, 25 independent-auditor synthetic tests, lint/format/strict
typing, 92 exact CPU tensor comparisons against archived disabled behavior,
and three small synthetic CUDA BF16 optimization steps. Passing these checks
did not predict scientific success. The original CPU-suite stdout is not
archived; its prerequisite receipt is explicitly owner-attested.

This is one training seed and 222 repeatedly used validation queries, not a
protected-test or generalization result. The auditor reconstructs results from
saved logits and authenticated checkpoint bytes/receipts, without independently
regenerating neural outputs or replaying training. Local raw outputs and weights
are hash-bound, but this repository does not claim a durable checkpoint archive.
No other margin settings, replication seeds, HTTP behavior, throughput or energy
were tested by this screen.
