# 7. Rank ordering limits threshold-only repair

The frozen membership head cannot replace the current candidate selector merely
by choosing better score thresholds. Across three training seeds, only 479 of
666 validation query-seed observations have every true member ranked above
every nonmember. The current pair selector already returns 569 exact sets.

## Two restricted directions ruled out

First, the [all-source-union proof](../evidence/updates/2026-09-25-ranking-diagnosis/experiments/2026-09-25-all-source-union-bound.json)
reconstructs all 15 nonempty subsets of the four generated source sets. Every
available exact triple/four-way union duplicates an exact candidate already in
the original ten slots. With unchanged additive scores and old slots preferred
on ties, these additions cannot create an exact gain. The existing selected
exact count of 569 is an upper bound for that pool-only change, not a measured
result of a new fifteen-slot policy. A separate bitmask audit verifies all 9,990
subsets; it is a second implementation by the same agent, not an independently
staffed replication.

Second, the [declared membership diagnosis](../evidence/updates/2026-09-25-ranking-diagnosis/experiments/2026-09-25-membership-separability-plan.md)
uses the weakest true-member score a and strongest nonmember score b. A strict
score threshold can recover the exact set iff b<a; its feasible interval is
[b,a). Labels define this diagnostic interval, and no threshold is fitted.
The [audited result](../evidence/updates/2026-09-25-ranking-diagnosis/experiments/2026-09-25-membership-separability.json)
finds no boundary ties, 479 strictly separable observations and 187 with rank
overlap. Of the separable observations, 140 fail the original zero cutoff.

| Training seed | Pair selector exact | Zero threshold exact | Oracle-cardinality top-K exact |
| --- | ---: | ---: | ---: |
| 1729 | 191 | 112 | 165 |
| 1730 | 192 | 116 | 160 |
| 1731 | 186 | 111 | 154 |
| Total | 569 | 339 | 479 |

Giving the ranking the correct answer size gains 16 exact answers but loses
106 relative to the selector. Its macro F1 is 99.35%; that privileged-input
measurement is not a general upper bound on F1 over all threshold rules.
Among the selector's 97 failures, 81 have rank overlap. Dual-TYPE accounts for
163 of the 187 rank-overlapped observations. The candidate restriction protects
many complete answers despite imperfect individual membership ordering.

## Implication and limits

These results motivate improving learned rankings or constructing candidates
that recover missing members while retaining useful set structure. They do not
establish that a new loss, architecture or hybrid policy will succeed. Each
requires a separate declared experiment. In particular, this diagnosis does
not prove an architectural capacity limit.

The 666 observations repeat the same 222 validation queries under three model
seeds. Expected labels and answer sizes are used diagnostically; no protected
test, model forward or serving change occurs. Thirty-one focused CPU tests pass.
A separately authored auditor reconstructs every membership row and aggregate;
a [separate acceptance receipt](../evidence/updates/2026-09-25-ranking-diagnosis/audit/membership-acceptance.json)
accepts diagnostic correctness without promoting a model or policy.
The [worked lesson](../evidence/updates/2026-09-25-ranking-diagnosis/learning/30-ranking-versus-threshold.md)
explains the equations, tensor masks and interpretation with examples.
