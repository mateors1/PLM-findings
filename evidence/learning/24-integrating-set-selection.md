# 24. Bringing set selection into the application

The previous experiment reached 501/666 exact validation answers by ranking four
saved candidates with the learned membership head. Now we move that rule into
normal generation, evaluation and native HTTP serving.

**Status:** integration completed and independently audited. All 2,664 candidate
paths/scores, 666 selections, 666 disabled greedy outputs and 666 HTTP responses
match exactly. All 21 metadata checks and three failure contracts pass; temporary
servers stopped. The completed baseline passed 554 tests. The option stays off
by default. This accepts the four-path policy, separately from pair composition.

## One policy, several entry points

Think of the experiment script as a laboratory instrument. It can tell us that
a rule works on a recorded set of inputs. The application has more responsibilities:
load the right checkpoint, form the prompt, generate the candidates, choose one,
apply request metadata, and return labels for the chosen identifiers.

The shared `generate_reranked_responses` function now owns candidate generation
and selection. Both serial `generate_response` and batched `generate_responses`
can call it. CLI evaluation and native serving use those normal entry points.
Its result retains the four candidates, their scores and eligibility, the selected
rank, and the selected raw tokens. This makes an incorrect selection inspectable.

The opt-in setting is `eval.symmetric_set_reranking=true`. It requires protocol
constraints, KV caching, target uniqueness, and a trained symmetric head. The
width is fixed at four; this integration introduces no width or score tuning.

## Follow the tensors

Let B be the number of queries in an offline group. Our ordinary validation
group has B=8; the final group has B=6; native HTTP executes B=1.

| Quantity | Shape in this model | Meaning |
| --- | --- | --- |
| Prompt IDs | [B, 5] | BOS, subject, dimension, SAME, ANSWER |
| Prompt embeddings | [B, 5, 256] | A vector for each input token |
| Prefill vocabulary logits | [B, 5, 2049] | Scores before legal-token masking |
| One-step vocabulary logits | [B, 1, 2049] | Scores for the next emitted identifier or EOS |
| Each layer's cached K and V | [B, 2, T, 32] | Two KV heads, cached length T, head width 32 |
| Symmetric membership logits | [B, 1025] | One learned score per product |

We run four separate rank batches. In the first batch each query starts with
its highest-scoring guided product; in the next it starts with its second
choice, and so on. Each branch has fresh cache and seen-target state. Later
positions use greedy constrained continuation. The original group shape is
preserved because this is a parity exercise: changing tensor batch shapes can
change floating-point arithmetic, even when the mathematical expression agrees.

For each completed candidate's SAME-processed set S, the selector computes

\[
s(S)=\sum_{i\in S}z_i,
\qquad
\hat S = \arg\max_{S\text{ eligible}}s(S).
\]

Here z is the prompt-only symmetric head output. Sorting member IDs before
`math.fsum` makes the sum independent of the candidate's output order. Equal
scores choose the earlier first-choice rank. There are no oracle targets in
this function. Lesson 23 derives the equivalent Bernoulli score and explains
why that identity does not guarantee calibrated probabilities.

## Correctness includes failure

The completion budget includes the first product and EOS. A branch that hits
the bound without EOS remains a failure. Only valid, terminated, unique paths
are eligible. If all four fail, the selector returns rank one's raw failure;
it never appends EOS to make the answer appear complete.

The native service then returns its existing generation-error response. On a
successful path it applies SAME self-exclusion, IGNORE and the requested return
limit, then hydrates product labels. IGNORE and limit must not change candidate
selection: they are caller metadata, not model tokens or membership evidence.

## Why a new false default changes identity

A configuration hash is computed from canonical serialized configuration. Adding
`"symmetric_set_reranking": false` changes those bytes even though old behavior
still runs. Overwriting the old hash would erase which configuration was used
to train the checkpoint.

`validate_saved_config` therefore checks the historical raw configuration against
its recorded hash **before** parsing it with today's schema. It allows exactly
one adaptation: insert the missing new option as false. Other missing fields,
coercions or changed values fail. The original training hash and the effective
inference hash remain separate, as do old and new source archives.

The loader also compares model metadata from the actual checkpoint payload to
the requested model configuration. A positive serving weight alone cannot prove
that a head was trained; the checkpoint must agree and have training steps.

## Evidence required for acceptance

The current CPU tests exercise branch independence, token bounds, score ties,
head requirements, migration rejection, ordinary CLI entry points, HTTP failures
and default compatibility. They also compare against frozen candidate-generation
logic on a controlled model. Their purpose is to catch contract mistakes early.

The campaign uses the three original trained checkpoints and all 222 validation
queries per checkpoint. The offline stage has reproduced 2,664 candidate token
sequences and scores, all 666 selections, and all 666 disabled greedy outputs,
with zero tolerance required for the saved score comparisons.
Real temporary HTTP servers reproduced selected raw tokens, termination,
protocol validity, errors, processed answers and hydration for all 666 queries.
The independent audit verifies complete evidence rather than substituting
aggregate F1 for these comparisons. Exact-set counts remain 168/168/165 by seed,
501 in total. The [portable receipt](../experiments/2026-09-24-set-reranking-integration.json)
binds the summary, audit, source/config archives and all six phase reports.

[The declared contract](../experiments/2026-09-24-set-reranking-integration-plan.md)
keeps model weights and protected test data untouched. Passing this integration
establishes that the application implements the measured policy. Oracle
parity and matched-quality latency, throughput and energy remain separate work.

## What remains after selection

An independent CPU diagnosis of the saved paths divides the 165 remaining
inexact observations into three groups:

| What the four candidates contain | Observations | Implication |
| --- | ---: | --- |
| Two whole candidates whose union is exact | 69 | Composition might recover missing coverage |
| Every correct member somewhere, but no exact union of whole candidates | 43 | Composition alone cannot remove the wrong members |
| Correct members absent from all four candidates | 53 | These candidates cannot supply the missing information |

The first group is entirely dual TYPE. For SOLROCK / TYPE at seed 1729, one
candidate has 101 correct members and no extras, while another has 73 correct
members and no extras. They overlap on two products:

\[
|A\cup B|=|A|+|B|-|A\cap B|=101+73-2=172.
\]

Together they match the expected set. We used the oracle to diagnose that fact;
a runtime selector would have to recognize the useful pair without those labels.

Blindly unioning all four is harmful: exact answers fall from 501 to 8, with
495 losses and two gains. For BERGMITE / TYPE at seed 1730, the selected answer
is already exact with 47 products; the full union adds 338 wrong products.

The [portable diagnosis](../experiments/2026-09-24-set-candidate-coverage.json)
records the evidence. The [next declared experiment](../experiments/2026-09-24-pair-union-plan.md)
retains the four originals, adds their six pair unions, and applies the unchanged
learned scoring rule. [Lesson 25](25-composing-candidate-sets.md) now records its
completed, independently audited result: 569/666 exact sets. A composed set is not an autoregressively
generated sequence: it must retain both source paths and cannot borrow a claim
of model-generated EOS or sequence probability. Any application integration of
composition requires the separate contract and API explained in
[lesson 26](26-integrating-composed-sets.md), now undergoing its own fresh replay.

## Self-check

Why can equal F1 hide different application behavior? Why does changing only a
default option still require a new inference identity? Trace where IGNORE acts,
and explain why moving it before selection would create a different policy.
