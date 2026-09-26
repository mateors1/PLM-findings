# 2026-09-25: saved bilinear error geometry

Independently audited descriptive evidence is accepted. Of the 24 wrong outputs
from fresh seeds 1730/1731, 15 are strictly separated but fail the zero threshold
(nine extra-only, six missing-only); nine have overlap or ties. Historical
selection seed 1729 contributes ten separated errors and five overlap/tie errors.
All-three totals are 25 separated mistakes and 14 overlap/tie mistakes among
39 incorrect model-query observations. No new quality gate or predictor is defined.

Read the [research synthesis](../../../papers/24-bilinear-error-geometry.md),
[frozen declaration](experiments/2026-09-25-bilinear-error-geometry-plan.md),
[portable report](experiments/2026-09-25-bilinear-error-geometry.json),
[primary summary](campaign/summary.json),
[independent audit](campaign/independent-audit.json) and
[owner decision](campaign/decision.json).
The exact reduced rows are available for
[1729](campaign/seed-1729.json), [1730](campaign/seed-1730.json) and
[1731](campaign/seed-1731.json).

For each query, a is the lowest true-member score and b the highest nonself
false-member score. Exactness at zero requires a>0 and b<=0; strict separation
requires a>b. Hypothetical valid thresholds form [b,a) only when b<a. The report
calculates interval existence without selecting a value or generating alternatives.
All overall and dual-TYPE shared intervals are empty in all three seeds. This
precludes simultaneous exact recovery with one shared constant on those saved
scores; it does not claim that every threshold change would leave every metric
unchanged. Group information is evaluation metadata, not prediction routing.

Across the same 222 queries, 194 fail in no seed, 19 in one, seven in two and two
in all three. Thus 28 distinct questions account for the 39 incorrect observations.
The 444 fresh and 666 contextual observations remain evaluations of shared queries;
historical 1729 helped select the recipe. No independent generalization sample is
created by these descriptive reductions.

The frozen stdlib-only runner passed 38 synthetic tests and the separately
authored auditor passed 39, with zero skips. Both actual executions completed
successfully on their first attempt. The auditor independently reconstructs
classification, extrema, exact zero-rule replay, errors, shared intervals and
failure overlap. It imports no primary scientific arithmetic.

Copies preserve exact runner/auditor sources, tests, successful execution
receipts, source/plan/input snapshots, reductions, owner decision and source
teaching/register context. Only the current diagnostic's permitted input chain
is authenticated. Upstream accepted JSON reports remain authority for labels;
their dependency manifests are not traversed. No checkpoint, graph database,
training examples or protected-test data is read by this diagnostic or its audit.

`copied-files.json` and `snapshot.json` bind exact source copies.
`local-dependencies.json` lists omitted authenticated inputs, principally full
local score reports. No raw weight or full score vector is copied. Hashes do
not establish durable remote archival. The final-checkpoint count remains 33.

`SHA256SUMS.txt` covers this addition except itself. Older evidence stays
unchanged. These posthoc observations may inform a separately declared training
intervention; no threshold choice, new inference rule, oracle parity, serving
advantage or protected-test result is claimed.
