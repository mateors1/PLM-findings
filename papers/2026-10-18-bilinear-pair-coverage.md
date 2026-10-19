# Query holdout and reverse-pair supervision

Local accepted finding, 2026-10-18. Publication remains held.

The `bilinear-pair-coverage-v1` metadata diagnostic authenticates twelve existing
artifacts and joins each saved validation error candidate to the reverse
subject/dimension training-query identity. Its repaired implementation passed
68 tests; independent reconstruction matched all 666 rows and every total.
The initial schema-adapter failure is preserved separately.

Each seed has 181,491 covered and 45,837 uncovered candidate pairs: 79.8366%
of eligible validation pairs have direct supervision through the reverse
training query. All seven missed and one extra membership occurrences lack
that coverage. Exact outcomes remain218/222,222/222,220/222. The same222queries
repeat across seeds, and pairs may repeat in reverse direction.

This quantifies why query holdout is not pair or entity holdout. It is
consistent with symmetry and exact training fit; it does not identify a causal
mechanism. Uncovered means absent from admitted training identities, not a
protected-partition label or absence of indirect learning. Denominators mix
positive and negative memberships, so fractions are not class-conditioned
error rates. No weights, predictions, threshold, seed choice or protected-test
state changed. Feature adaptation requires its own controlled experiment.

Source report: `docs/experiments/2026-10-18-bilinear-pair-coverage-report.md`.
Evidence SHA256: `6379654ad76fe008771f9d351ad1d7df96aa6a4f8eb6836bfde324f9998ee821`.
Audit SHA256: `97323570c3bef5069618c8235367061b10d0dfe50c72018bb81b14e414ca2fd6`.
Decision SHA256: `e3b0d4a1fbb7589fff0e39021ccf1cbc86fd0657080bf2dc24eab7726b514226`.

Source PAPER v0010 retains this evaluation limitation and archives v0009
byte-identically. This local findings note does not assert a published paper
projection or GitHub update; publication requires a separate receipt.
