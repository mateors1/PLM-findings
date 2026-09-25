# Membership support for products absent from every generated source

This diagnostic asks whether the membership head gives favorable scores to
correct products that all eight generated paths omit. It examines the complete
coverage-defined subset before looking at scores: 30 query-seed observations,
representing 22 distinct dual-TYPE queries. No new prediction policy is tested.

For truth T and source union U, omitted truth is M = T minus U and covered truth
is C = T intersect U. Saved head vectors contain 1,025 product logits; excluding
the subject leaves 1,024 possible answers. Ranks start at one, use descending
logits, and break exact ties by smaller product ID. The
[frozen plan](../evidence/updates/2026-09-25-missing-source-membership/experiments/2026-09-25-missing-source-membership-plan.md)
defines signs, ranks, false-product comparisons and both aggregation denominators.

## Results and denominators

| Seed | Focused observations | Omitted occurrences | Positive / zero / negative | Within oracle top K |
| --- | ---: | ---: | --- | ---: |
| 1729 | 14 | 776 | 770 / 0 / 6 | 757 |
| 1730 | 6 | 183 | 180 / 0 / 3 | 174 |
| 1731 | 10 | 448 | 444 / 0 / 4 | 436 |
| Pooled | 30 | 1,407 | 1,394 / 0 / 13 | 1,367 |

Positive logits account for 99.08% of omitted member occurrences. Giving each
focused observation equal weight produces a mean positive fraction of
0.9908023047546597. Among covered truth in the same focused observations,
3,797/3,844 occurrences are positive; the mean per-query fraction is
0.9865678842678095. All 30 covered groups are nonempty. These repeated products
and queries are not independent statistical samples.

Twenty-two observations have all omitted logits positive; eight have some
positive, and none have no positive omitted logits. The count of 22 all-positive
observations is conceptually different from the 22 distinct query identities.
The [portable report](../evidence/updates/2026-09-25-missing-source-membership/experiments/2026-09-25-missing-source-membership.json)
records full-precision aggregates and provenance.

## What this supports

For Aegislash TYPE under seed 1729, product ID 1040 is absent from all sources
despite head logit 8.01483154296875 and head rank 18. No false product scores
higher or ties. This is concrete disagreement between membership support and
generated coverage, supporting a separate candidate-generation experiment.

It does not prove why generation omits that product. The opening guidance
combines a decoder logit with 16 times log-sigmoid of the membership logit;
ranking the membership head alone does not reconstruct that choice. Later
greedy decisions and stopping are not diagnosed here. Nor is a positive logit
a calibrated probability or proof that the model knows the whole relation.

The oracle top-K statistic uses K = |T|: 1,367/1,407 omitted occurrences, or
97.16%, lie inside this range. The actual request does not supply the correct
answer size. This is label-assisted diagnosis, not a usable prediction rule.
False products can still receive favorable scores; the result does not validate
returning every positive product or choosing a new threshold.

## Evidence and next work

The [dated evidence addition](../evidence/updates/2026-09-25-missing-source-membership/README.md)
preserves the independent audit, separate owner decision, focused test receipts,
scripts and full per-member diagnostic reports. The independent implementation
reconstructs source coverage, signs, ranks, false-member comparisons and
aggregates. It does not replay neural generation or independently recreate
upstream checkpoint provenance.

No checkpoint, serving default or protected-test result changes. The previous
intersection policy still fails its quality gate; the accepted offline
eight-branch result remains 603/666 exact answers. Membership support motivates
testing a fixed extra branch anchored outside existing source coverage, with
no labels in prediction and the same per-seed quality gates. That proposal is
unmeasured, and source expansion alone cannot fix existing ranking misses.
Active work remains within Pokemon. Weight payloads remain local dependencies,
not durably archived by this documentation.
