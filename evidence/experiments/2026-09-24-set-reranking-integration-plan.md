# Integrate the validated four-path symmetric set selector

Declared after the completed set-reranking comparison on 2026-09-24.

## Objective and reference

Make the successful selector available through normal constrained generation,
CLI evaluation and native serving, rather than only a saved-candidate script.
Preserve the default greedy path. The measured policy generates four distinct
guided first products, greedily continues each with uniqueness, and selects a
valid completed path by canonical sum of symmetric membership logits over its
fixed SAME-processed set. No new training, model architecture or oracle inputs.

Authoritative reference: runs/learning/set-reranking-v1/summary.json, SHA256
`c515fae06f7db1810e8697badf59d53ffb41675cbb8fc543d1a14feee7252c5c`;
independent audit SHA256
`22d55b7b58b988d53e78ef4f0bf167f24a2b52e2a27d3f53b603dd00fbeeb2aa`.
The candidate paths remain bound to the previous first-choice summary
`7b9db0b137d53ce2c350932aec61b1a8e94295f9affbae16aa55cc52f659fa8a`.
Expected selected counts are168/168/165 exact, pooled501/666, mean F1
.9555846395858687, with136/57/308 single/dual/COLOR exact answers.

## Public behavior

Add an explicit default-false evaluation/serving option for symmetric set
reranking. The supported enabled policy uses four first choices; do not silently
add new widths, score mixtures, thresholds or count predictors. Reuse the current
guidance-strength setting and record it. Enabled operation requires constrained
cached decoding, unique emitted products, and a trained symmetric relation head.
Establish that training requirement from validated checkpoint model/training
metadata, not merely a positive weight in the serving configuration.
Keep core module imports Torch-free until the existing optional model boundary.

Share the candidate generation and numeric set selection between offline and
native paths. Preserve ascending token-ID first-logit ties, original rank-batch
semantics, fresh per-branch cache and seen state, and the inclusive completion
budget. Rank1 must remain the original greedy path. Only valid completed paths
are eligible; if none qualify, return the rank1 failure without manufacturing
EOS. Preserve bounded native admission and serial model execution.

Compute set scores over deterministic SAME-processed sets with canonical ID
order and math.fsum of FP32 symmetric logits. Do not let caller IGNORE or return
limits change which full answer is chosen; apply those request directives only
after selection, as before. The selector must not consult graph attributes,
expected targets, expected counts or truth-derived metrics.

Expose the enabled option and precise policy in evaluation and deployment
receipts. Keep the trained checkpoint identity separate from the new inference
source/config identity. Return selected raw evidence on failures. New successful
responses still contain identifiers and existing hydration output, not a new
user-facing research-debug format.

## Provenance during schema evolution

The old checkpoints were trained with source archive
`101d661a16e3f17ecc924c1afed3a188cea25fe8e70bb1df2a19b523f363ecd1`.
Integration intentionally creates a new inference source identity. Do not claim
that source stayed unchanged, retrain controls silently, rewrite historical
reports, or make old audits accept arbitrary new source.

New default config fields can change canonical hashes. Verify saved historical
JSON hashes before parsing it into an expanded schema. If an explicit historical
serialization adapter is necessary, limit it to the newly declared default field,
verify all old values, document the transformation, and record both identities.
Do not broadly normalize unrelated settings or suppress an identity mismatch.
Leave frozen script/helper/source archives intact. New integration verification
must compare actual outputs across the documented old/new source boundary.

## Verification and acceptance

First run meaningful tests for disabled-path compatibility, incompatible flags,
trained-head requirements, branch/cache/termination independence, score algebra,
ties, immutable candidate choice, and IGNORE/limit separation. Check CLI wiring,
config/checkpoint compatibility and failed-generation behavior.

Then load the three unchanged control checkpoints and generate all222 validation
queries per seed through the integrated offline batch8 path. Require all2,664
candidate raw token sequences, canonical set scores and selected ranks/full IDs
to agree with the frozen reference, allowing only explicitly justified arithmetic
tolerance for scores, never altered selected tokens. Also replay disabled greedy
on all666 observations against its original full-token references.

Exercise native HTTP against all222 queries per checkpoint with enabled policy
and serial query handling. Verify the entire selected raw token sequence,
including prompt and EOS, plus termination, protocol-validity and error fields.
Compare processed answers and hydration against the same reference; include
IGNORE/limit and invalid/truncated
request checks. Any batch-shape numerical mismatch must be retained and diagnosed,
not hidden by comparing only aggregate F1. Do not weaken parity after seeing a
mismatch. Stop temporary servers after verification; do not replace a permanent
deployment or make enabled operation the default in this integration.

Record immutable source/config/checkpoint/runtime/data, script/helper and report
receipts, plus wall time/memory observations. CPU audit work can run in parallel;
GPU validation and service jobs remain sequential. Do not introduce a continuous
batching or concurrency claim from this implementation.

Passing proves the opt-in application path reproduces the validated policy.
It does not establish oracle parity, protected final-test readiness, or matched
quality latency/throughput/energy. Keep those requirements open after integration.
