# Frozen candidate pair composition with unchanged learned scoring

Declared on 2026-09-24 after the saved-candidate coverage diagnosis. Not executed.

## Question and fixed inputs

The current learned selector reaches 501/666 exact validation sets. Of its 165
failures, 69 contain an exact union of two whole candidate sets; 43 contain all
truth members but no exact whole-path subset union; 53 omit correct products
across all four paths. Blind four-way union returns only 8/666 exact sets.

Diagnosis: `runs/learning/set-candidate-coverage-v1/summary.json`, SHA256
`67974a44184a48ecb67b62bbb5dab4c9f3ac05a2b30e4858fee771a767ea0dc4`.
This is oracle-assisted availability analysis, not a deployable selection rule.

Test whether the already trained symmetric head can choose useful pair unions
while retaining the original four sets. Use the same three original control
checkpoints' saved validation evidence, all 222 queries per seed, all four saved
paths and the saved FP32 membership logits. No training, neural recomputation,
new paths, extra first-choice ranks, threshold tuning or final-test access.

Authoritative set-score summary SHA256:
`c515fae06f7db1810e8697badf59d53ffb41675cbb8fc543d1a14feee7252c5c`.
Authoritative first-choice summary SHA256:
`7b9db0b137d53ce2c350932aec61b1a8e94295f9affbae16aa55cc52f659fa8a`.
Authenticate these summaries, per-seed reports, inputs and script receipts.
Authenticate historical source/config artifacts through their archived bytes;
the current integration's explicit source/config evolution must not cause broad
normalization, rewriting of historical reports, or false unchanged-source claims.

## Candidate pool and selector

For every query, preserve four original SAME-processed sets S1 through S4.
Add exactly six pair unions: S1 union S2, S1 union S3, S1 union S4,
S2 union S3, S2 union S4, S3 union S4. Eligibility for an original is unchanged;
a pair is eligible only if both original paths are valid, terminated and unique.
Do not filter candidates using labels, attribute counts, dimension subgroups,
expected size or product-level truth. Apply the same pool to TYPE and COLOR.

Score each eligible set with canonical ascending-product-ID math.fsum of saved
FP32 symmetric logits. No overlap penalty, cardinality correction, mixture,
calibration or coefficient is introduced. Duplicate member IDs count once.
Select highest score. Equal scores prefer original ranks 1,2,3,4, then pairs
(1,2),(1,3),(1,4),(2,3),(2,4),(3,4). Thus an equal-scoring new pair cannot displace
an original set. If none are eligible, preserve the original rank-one failure.
Retain all ten provenance slots even when their member sets coincide. Do not
deduplicate or reorder slots. Serialize a composed set in ascending product-ID
order, explicitly as a set representation rather than an emitted sequence.

First reproduce every original four-set score and selection exactly. Then select
from the ten sets without oracle inputs. Expected targets enter only measurement
after selection. Preserve all scores, eligibility flags, source ranks and chosen
sets. The fixed tie order is part of the declared policy.

## What a composed answer is

A pair union is a composition of two generated answers. It is not a sequence
the autoregressive model emitted. Store its selected set and both original raw
token sequences. Do not manufacture a raw sequence/EOS, attribute a sequence
probability to it, or report that it passed model termination/protocol checks.
Only its source paths have those checks. Set quality and source-path eligibility
are distinct evidence fields. No serving/schema/production decoder change belongs
to this experiment; any later integration needs its own explicit contract.
An ineligible fallback never receives credit as an exact valid answer, even if
its incomplete target set coincidentally matches truth. Preserve raw failure
evidence and report selected-source eligibility separately from model termination.

## Measurements and gate

Report per-seed and pooled macro precision, recall, F1 and exact-set counts;
single-TYPE, dual-TYPE and COLOR groups; paired exact gains/losses versus the
strong 501/666 four-set selector; original-versus-pair selection counts; and
counterexamples. Also report the ten-pool oracle exact availability
separately: the diagnosis predicts 570/666 (=501+69), a ceiling rather than a
learned selection result. Record scoring wall time separately from the already paid
candidate-generation cost. This CPU experiment establishes no serving latency,
concurrency, energy or oracle-quality claim.

Require no per-seed overall F1 or exact-count regression, strict pooled exact
improvement, and strict pooled dual-TYPE exact improvement versus the four-set
selector. Gate all source eligibility/fallback invariants separately. Report
failure honestly; do not tune this rule after observing outcomes. Any changed
pool or scoring rule requires a new declaration.

Keep the original four-path integration campaign intact regardless of this
result. Validate arithmetic, duplicates, tie order, failure eligibility and
truth-independent selection with focused tests and an independent CPU audit.
