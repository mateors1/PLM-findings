# Wider first-choice search improves complete answers at a compute cost

Increasing fixed first-choice branches from four to eight raises exact answers
from 569 to 603 of 666 validation query-seed observations. Macro F1 increases
from 0.9668960529174291 to 0.9834927532993669. The same three accepted checkpoints
and learned set scorer are used throughout: this is an inference-policy result,
not a newly trained model. All predeclared quality gates and the independent
saved-evidence audit pass.

## A controlled expansion of candidates

For each seed, the archived runtime first reproduced its original four paths,
head vectors, ten candidate slots and decisions exactly. Only then did it
generate first-choice ranks five through eight. Later positions still use the
same greedy decoder and validity/uniqueness masks. Eight originals plus their
28 pair unions produce 36 slots. The original ten slots remain first, retaining
their tie precedence. Selection sums the unchanged head logits over each set;
ground-truth membership enters diagnostics only after selection.

The [frozen plan](../evidence/updates/2026-09-25-wide-first-choice/experiments/2026-09-25-wide-first-choice-plan.md)
and [portable report](../evidence/updates/2026-09-25-wide-first-choice/experiments/2026-09-25-wide-first-choice.json)
record checkpoint, corpus, split, runtime and numerical identities.

| Seed | Four-branch exact /222 | Eight-branch exact /222 | Gains / losses |
| --- | ---: | ---: | ---: |
| 1729 | 191 | 201 | 10 / 0 |
| 1730 | 192 | 205 | 14 / 1 |
| 1731 | 186 | 197 | 11 / 0 |

These are the same 222 validation queries under three training seeds, not 666
independent queries. COLOR exactness improves 308→309/309; single-TYPE improves
136→139/153 and dual-TYPE improves 125→155/204. COLOR's complete validation
agreement does not establish overall or protected-test oracle parity.

## Search creates opportunities and exposes scoring mistakes

Exact-answer availability improves 570→605, while available exact answers missed
by selection increase 1→2. There are 35 gained exact answers and one loss:
Cetitan TYPE under seed 1730. The old correct candidate remains available, but
a new wrong candidate wins. Increasing the search space therefore improves
aggregate quality without guaranteeing improvement on every query.

Of 63 remaining nonexact observations, 61 lack an exact candidate among the
36 slots. Thirty lack a true member across all eight generated sources; the
other 31 collectively contain the truth but no original/pair exactly matches
it. Two additional observations have an exact candidate that selection misses.
Larger unions of these eight sources were not evaluated. The earlier diagnosis
of four-source unions must not be generalized to this expanded pool.

## Cost and evidence boundary

Ranks 1–4 took 479.66 seconds; the additional ranks took 486.34 seconds across
the three seeds. The combined generation work took about 2.01 times the baseline
generation time. These sequential observations include order and initialization
effects and are not a randomized serving benchmark, a concurrent-user result
or an energy measurement.

The [independent audit](../evidence/updates/2026-09-25-wide-first-choice/campaign/independent-audit.json)
reconstructs all 5,328 source paths and 23,976 slots. The runner and auditor
passed 29 and 48 focused tests respectively. The
[separate decision](../evidence/updates/2026-09-25-wide-first-choice/campaign/decision.json)
accepts the offline evidence and fixed quality gate without changing deployment
defaults. The audit reconstructs saved evidence; it does not independently
rerun the neural network or prove every next-token argmax.

The inventory remains 27 final checkpoints. Large per-query neural reports
and checkpoint payloads remain local, identified by hashes; the copied compact
evidence is not a complete neural reproduction bundle or durable model backup.
Validation has guided repeated experiments; the 191-query protected test remains
unused. Active work stays within Pokémon until the user explicitly opens another
dataset. The next research question is how to recover missing exact candidates
while accounting for the additional compute and the selector's mistakes.
