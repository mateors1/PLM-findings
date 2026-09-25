# Membership scores for correct products absent from all eight sources

Declared 2026-09-25 before calculating the new diagnostic. Pokémon validation
only; no model forward, fitting, new prediction policy, external dataset or
protected-test measurement. This is a diagnostic, not an improvement screen.

## Question and bound inputs

In the accepted eight-branch result, 30 query-seed observations omit at least one
true member from every source. Does the saved prompt membership head assign
positive scores or high ranks to those omitted true members? Positive scores
may support investigating candidate generation; nonpositive scores or poor ranks
may implicate membership scoring as well. Neither observation identifies a causal
failure mechanism or demonstrates that an altered decoder will improve quality.

Use the audited saved set-operation reports, whose baseline remains the accepted
eight-branch policy. Authenticate the set-operation summary SHA256
`1fe6e9cb1980753e6c2315c6dac890148a52105061c881261da9bb73db1c387d`,
audit `27e0c5ff2931d88fed9fae612c73c64c7835e01282097fc290fa78cd9a7e330c`,
and separate decision `01a51a5a9d5957ec68f32b0d81d23acace29d7eb3aa5684d99939582e88cb32a`.
Its quality gate failed but evidence acceptance passed; preserve that distinction.
Bind the three per-seed reports named by the summary, before and after reading.
Upstream audits establish source/runtime identity; do not imply new neural replay
or a fresh recursive checkpoint/archive audit. No live core/config dependency.

## Fixed diagnostic

Reconstruct subject-excluded source memberships directly from each row's eight
valid, terminated, unique raw paths and require equality to saved source sets.
Validate the five-token prompt, nonempty canonical teacher IDs, finite FP32 head
vector of width 1025 and subject exclusion. Check identical ordered queries and
teacher sets across seeds and compare the computed total source union/coverage
with saved diagnosis. There are 222 distinct queries and 666 observations.

For every row let T be truth, U the union of the eight sources, M=T minus U
the omitted truth and C=T intersect U the covered truth. The focused subset
requires nonempty M, computed before using head scores. Require exactly 30 rows,
all dual-TYPE, with seed counts 14/6/10. Record all-row coverage accounting and
per-query missing counts to show selection is complete rather than hand-picked.

Within the subject-excluded 1024-product universe, rank saved head logits in
descending order with smaller token ID first on exact ties. For every omitted
true product, record ID, logit, rank, number of false products scoring strictly
higher, number of false products tied, and whether it is in the top K positions
where K=|T|. K comes from labels: this is an oracle-cardinality diagnostic and
must never enter a predictor. These are relation-head ranks, not the actual
decoder-plus-guidance first-token ranks; decoder logits are not inferred.

For M and C separately in each focused query, record count, positive/zero/negative
logit counts, min/median/max logit and rank, with null summaries for empty groups.
Use exact comparison with zero, no tolerance or selected threshold. Summarize
per seed and pooled: focused queries, omitted/covered true-member occurrences,
sign counts, top-K omitted membership counts, and number of focused queries with
all/some/no omitted members having positive scores. Report both pooled member
fractions and mean per-query positive fractions with their denominators. Do not
treat repeated products across queries or seeds as independent statistical samples.

No significance test, threshold search, temperature sweep, new candidate policy
or generated-quality claim is part of this diagnosis. Do not describe a positive
logit as proof that the model knows a relation or as a calibrated probability.

## Verification and publication

Use a small stdlib-only runner, immutable reports and bound plan/script/test
receipts. Reuse already-authenticated artifacts rather than rerunning the
preceding experiment. Focused synthetic tests cover all/some/no positive missing
members, exact zeros and ties, subject exclusion, empty covered truth, corrupted
raw/source alignment, nonfinite/non-FP32 scores, and cross-seed query mismatch.
Freeze the tested implementation before running the actual diagnosis.

An independent implementation must recompute the omitted memberships, ranks,
false-member comparisons and aggregates without importing primary arithmetic
helpers. Record its source/hash and focused tests before auditing. Preserve
failed executions/audits; evidence acceptance is separate from scientific
interpretation. Update lesson, research/dev logs, registry, model register and
PLM-findings. After verification, create a new one-experiment ready packet for
the Luna Max publication task and request its GitHub commit receipts. No new
checkpoint is created; the inventory remains 27.
