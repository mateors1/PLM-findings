# Fixed pair-intersection screen and eight-source representability diagnosis

Declared 2026-09-25 before computing new candidate scores or diagnostic results.
Use only saved Pokémon validation outputs from the accepted width-eight offline
experiment. This adds no neural generation, training, external dataset or
protected-test measurement. The 27-checkpoint inventory remains unchanged.

## Inputs and reconstruction

Authenticate `runs/learning/wide-first-choice-v1/summary.json` SHA256
`1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183`,
its independent audit `eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95`,
and decision `70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601`.
Require their completed/pass/accepted status and agreement. Bind the three
per-seed report hashes from the summary, seeds 1729/1730/1731, 222 identical
ordered validation queries per seed. Bind consumed files before and after work.
The upstream audit establishes historical neural/runtime provenance; this
saved-output analysis does not independently rerun the neural network or load
checkpoint weights. Record that evidence boundary explicitly.

Reconstruct each row's eight raw source paths: five-token prompt, product-only
completion, EOS, validity, uniqueness, rank ordering and SAME subject removal.
Require all eight valid, terminated and unique, as established upstream; a
mismatch fails execution rather than filtering or silently repairing the input.
Validate canonical expected IDs, subject exclusion, TYPE/COLOR group, and finite
1025-column FP32 head logits aligned with product IDs 1024–2048.
Rebuild the existing 36 slots in historical order (legacy ten, originals five
through eight, remaining 22 pairs), their canonical sorted-ID `math.fsum` scores,
and first-maximum selection. Require exact saved-slot/decision/metric equality.
Reproduce 603 exact answers and macro F1 0.9834927532993669 before accepting any
new result. Expected rows are 666 observations on 222 distinct queries.

## Fixed prediction experiment: add only pair intersections

Keep the existing 36 slots first, then append intersections of every pair of
source sets in lexicographic rank order: (1,2), (1,3), ..., (7,8). Exactly 28
new slots produce 64 total. Preserve duplicate sets. A composed set is eligible
when both sources are eligible. Retain empty intersections as eligible empty
sets with score zero: this is the existing postprocessed-set scoring convention,
not an invented emitted sequence or EOS. Report empty selections explicitly.

Score using unchanged saved logits and canonical `math.fsum`. Choose the first
eligible maximum. Never use expected labels, group labels or observed errors
in candidate construction, scoring or selection. Do not change thresholds,
guidance, routing, weights or score normalization. Labels enter metrics only.
Intersections test whether agreement between two generated sets can remove
unwanted members; they may also delete correct members. This is one fixed
screen, not an adaptive family or a new default.

Report per seed, pooled and COLOR/single-TYPE/dual-TYPE: exact count, macro
precision/recall/F1, mean set size, exact availability, available-exact misses,
gains/losses, changed-selection count, intersection selections and empty
selections. Save each slot's IDs/provenance/score, selected slot and errors.

The quality gate requires: exact baseline reproduction and all input/path/slot
invariants; no per-seed exact or F1 regression; strict pooled exact gain above
603; and no pooled group exact regression. No gate relaxation or automatic
alternative candidate policy after failure. A pass only permits consideration
of integration under a separate verification contract.

## Diagnosis only: pure union and pure intersection bounds

Independently of the 64-slot policy, enumerate every nonempty subset of eight
sources (255 subsets), once by union and once by intersection. Use cardinality
then lexicographic rank order. Record whether each operation family can produce
the exact teacher set, the minimum number of sources required and its first
witness. These label-assisted witnesses are oracle diagnostics and are never
used by the predictor. Report availability for the baseline, 64-slot policy,
all unions, all intersections and either pure operation family.

Partition baseline failures into selected-exact, available-exact-but-missed,
unavailable-and-truth-missing-from-all-sources, and unavailable-with-truth-covered.
For each source and selected answer record false-positive/false-negative IDs.
The union of sources bounds coverage: a true member absent from every source
cannot be recovered by unions or intersections alone. Coverage is necessary,
not sufficient; extras can make pure unions inexact. These bounds concern pure
operations, not arbitrary nested union/intersection expressions, differences,
complements, new generated paths or a trainable policy.

## Evidence and verification

Use a stdlib-only runner and immutable output directory; freeze its source,
this plan and focused-test receipt before actual execution. Refuse overwrites,
write failure evidence on exceptions and rehash inputs after completion.
Focused synthetic tests must cover union/intersection identities, exactness
requiring three sources, covered truth polluted by extras, missing members,
duplicate/tied and empty candidates, negative/zero scores, a newly introduced
regression, truth-free selection, invalid/nonfinite inputs and failed gates.

A separately implemented auditor must reconstruct source sets, all 64 scores
and selections, metrics, exhaustive witnesses, per-query errors and aggregate
gate without importing runner helpers. Prefer integer bitsets for independent
set arithmetic and exact rational sums rounded to float for score checks.
Bind its source and tests before actual audit; preserve any failed attempt.
Separate evidence acceptance from the quality gate and from policy promotion.
Copy compact reports, plan, source, audit, decision and test receipts into a new
dated PLM-findings addition; retain large raw upstream reports locally with
their hashes and state the resulting reproduction limitation.
