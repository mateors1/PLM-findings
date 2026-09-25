# Matched continuation failures: declared diagnostic comparison

2026-09-24. Follow the failed coefficient-zero/one continuation-set trial.
Use its six frozen checkpoints and all 222 validation queries per checkpoint.
No training, coefficient selection, protected-test evaluation or serving
promotion is part of this diagnosis. The prior acceptance gate remains failed.

## Question

Why do lower teacher-forced token losses and better average set overlap fail
to yield more complete answers? Locate sequence divergences and measure the
gap between actual generated prefixes and correct supplied teacher prefixes.
Distinguish ordering mistakes from set mistakes, premature stopping, and the
effect of a subject token that is removed only after generation.

## CPU analysis

Verify the completed matched campaign summary and all six full-token reports,
checkpoint/source/config/data/split identities. Reconstruct processed answers
using the existing deterministic policy; never filter by graph membership.
Report first ordered divergence, first extraneous product, missing products,
pure-prefix early EOS and set-correct reordering. Preserve raw-to-processed
position maps, subject insertion positions and whether insertion precedes each
failure boundary. Report control/candidate, seed, TYPE single/dual and COLOR,
and paired gained/lost/both-inexact observations. Correlations are not causes.

## Frozen prefix probes

Probe actual raw generated prefixes after 0, 1, 2 and 3 products, at the teacher
answer midpoint (capped to the generated length), at the generated endpoint,
and at the raw boundary corresponding to the first processed ordered divergence.
Merge coincident positions and preserve their landmark names. Exclude EOS from
the supplied product prefix. Every condition starts with a fresh full-prefix
cached prefill; no condition reuses another condition's cache.

Pair each raw prefix with a teacher prefix containing the same number of
consumed non-subject products, capped to teacher length. Count wrong products
as consumed too: matching progress must not depend on oracle correctness.
This matches positions, not remaining difficulty or semantic content. When a
raw subject was inserted, the teacher counterpart can have a shorter sequence.

Hold alpha-16 first-target guidance and uniqueness fixed. Guidance applies only
at the original prompt; do not reapply it at later states or mask the subject.
Record the next choice, probability mass on remaining correct products, EOS
behavior, and the ordered teacher-next token's rank/probability when legal.
If that teacher token was already emitted out of order, record an unavailable
exact-token score with an explicit reason; do not silently unmask it.

Observe the final normalized hidden state with a temporary hook and reuse the
existing contextual projection and entity embeddings to obtain remaining-set
logits. Use fixed threshold zero for diagnostic TP/FP/FN and coverage. No new
weights, input labels or graph facts enter the model. Candidate teacher states
after 1–3 products were explicitly supervised; later positions and generated
prefixes are different conditions. These logits are not calibrated probabilities.

At raw prefixes, compare the full-prefill greedy choice with the archived next
token from batched cached generation. Report discrepancies rather than treating
this as guaranteed full replay: floating-point execution paths can differ.
At an exhausted generation bound there is no archived next token; mark any next
score as hypothetical. Teacher-prefix scores are diagnostic, not generated quality.

## Evidence and interpretation

Keep core/model source frozen. Execute GPU jobs sequentially after focused
tests; preserve scripts, imported analysis helpers, source archive, environment,
checkpoint hashes, reference identities and full per-query probes. Refuse
existing output paths. Independently check aggregates and consistency.

A correct teacher continuation paired with an incorrect generated continuation
is evidence of a prefix-conditioned gap. It does not isolate the causal effect
of any one changed token. A good remaining-set head does not guarantee ordered
enumeration; an early extra token does not necessarily cause final set failure.
Choose and declare the next intervention only after inspecting these results.
