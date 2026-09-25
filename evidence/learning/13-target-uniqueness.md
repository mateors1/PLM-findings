# Lesson 13: preventing repetition during decoding

**2026-09-24.** Lesson 12 found stronger average answers alongside occasional
repetition loops. This experiment keeps the trained weights fixed and changes
one rule used to choose the next token. The [declared plan](../experiments/2026-09-24-target-uniqueness-plan.md)
requires a complete paired validation comparison before any serving promotion.

## Result: completion improves; exact-answer quality stays unchanged

All 666 current original-policy outputs match the archived uncached outputs
exactly. All 666 unique-target outputs are valid and terminate with EOS, compared
with 664/666 originally. The model source and weights are unchanged.

| Seed | Original processed F1 | Unique-target processed F1 | Difference (percentage points) | Exact answers, both policies |
| --- | ---: | ---: | ---: | ---: |
| 1729 | 83.800% | 83.736% | -0.0637 | 143/222 |
| 1730 | 79.190% | 79.257% | +0.0679 | 130/222 |
| 1731 | 77.747% | 77.894% | +0.1471 | 134/222 |

Mean raw F1 changes from 79.870% to 79.930%; mean processed F1 changes from
80.245% to 80.296%. The paired processed-F1 change is +0.0504 percentage points
with sample SD 0.1065 pp. Mean processed exact accuracy stays at 61.111%, and raw
exact accuracy stays zero. Fifteen answers change (4/6/5 across the seeds): no
query gains or loses processed exact correctness. Raw repetition disappears in
all 15 affected queries.

| Previously unfinished query | Original IDs / correct products | Unique IDs / correct products | Expected products | New EOS? |
| --- | ---: | ---: | ---: | --- |
| KECLEON / TYPE, seed 1730 | 507 / 2 | 157 / 19 | 130 | Yes |
| CARBINK / TYPE, seed 1731 | 507 / 6 | 92 / 15 | 135 | Yes |

The two original loops finish under the new rule, but their answers remain
mostly wrong. Output audits confirm that the first emitted ID is unchanged on
all 666 queries, and every first divergence among the 15 changed answers occurs
when the original decoder would repeat a previously emitted entity. This ties
the changed behavior to the intended mask rather than a different model.

**Decision:** the experiment passes its validity requirement but fails the
predeclared no-regression requirement because seed 1729's processed F1 decreases.
Keep the option off by default and retain the existing serving reference. The
conditional HTTP promotion campaign was not run. A small aggregate improvement
does not change the result of that declared criterion; this is a useful partial
repair, not evidence that the policy is universally better or worse.

The [portable result](../experiments/2026-09-24-target-uniqueness.json) contains
paired metrics, source/report hashes, changed queries and the eligibility decision.
The next quality work needs to improve relation selection, particularly the
initial choices, rather than treating unique IDs as sufficient correctness.

## Training and decoding solve different parts of the problem

Training changes the parameters W to reduce a loss. Decoding uses the resulting
scores to construct an answer. We can change decoding without teaching the
model any new facts. That distinction matters when reporting an improvement:
this experiment measures a **model plus decoding policy**, not a better-trained
model or a new architecture.

The standard greedy decoder chooses the highest-scoring legal next token:

```text
next_id = argmax_i logits[i]
```

A protocol mask already restricts output positions to entity IDs and, after at
least one target, EOS. That prevents text or control tokens from appearing as
products. It does not prevent choosing the same product repeatedly.

## The new rule: remember the IDs already emitted

Let S be the set of entity IDs generated **after ANSWER** in the current request.
Before choosing the next token, change its score vector:

```text
masked_logit[i] = -infinity    if i is in S
                  logit[i]    otherwise, subject to the existing protocol mask
next_id = argmax_i masked_logit[i]
if next_id is an entity:
    S = S union {next_id}
```

For example, if the model scores A=7, B=6, C=5, EOS=4 at every step, ordinary
greedy decoding repeats A indefinitely. With this rule the answer is A, B, C,
EOS: after each entity wins, it becomes ineligible to win again. Once a toy
catalog is exhausted, EOS is the only remaining legal token.

This example illustrates the mechanics, not real quality. A, B and C could all
be wrong for the query. Excluding repeated choices does not identify the correct
relation, repair an earlier wrong choice or impose the corpus's target ordering.

### An observed counterexample: AERODACTYL / TYPE, seed 1729

The two answer prefixes are:

```text
original: IRON_MOTH, IRON_MOTH, IRON_MOTH, DRAGONITE, DRAGONITE, NOIBAT, ...
unique:   IRON_MOTH, SALAZZLE, AMOONGUSS, ARBOK, ARCANINE, ARIADOS, ...
```

The actual tokens carry the `PKM_` prefix. Both decoders choose the same first
entity, IRON_MOTH, which is not in the expected answer. At position two, the
original's winning token is a repeat. The new mask forbids it and selects
SALAZZLE instead. All subsequent steps now condition on a different history.

The original emits 118 entity IDs and finds 43 distinct correct products; the
unique version emits 160 and finds only 15. Both end with EOS. The expected
answer has 178 products. Removing repetition worsens this particular answer.
It contributes to seed 1729's processed F1 dropping from 83.800% to 83.736%,
while exact answers remain 143/222.

This is why deleting duplicates after generation and forbidding them during
generation are different operations. Post-processing leaves the model's input
history unchanged. A decoding constraint can redirect the entire continuation.

## Tensor and state details

In the current model:

```text
full forward logits:     [B, T, V]
next-token logits:       [V] = [2049] for one request
entity vocabulary:       1025 IDs, starting at ID 1024
seen-target state:       a Python set of entity IDs, empty at request start
```

We select the last position of the model output, apply the protocol mask, then
set previously emitted entity entries to negative infinity. Greedy decoding
does not need to calculate softmax. If we did, `exp(-infinity)=0` would give
those entries zero probability within the allowed distribution.

The prompt's subject is not initially in S. The same subject may be generated
once; the existing SAME post-process removes it afterward. This isolates target
repetition from subject exclusion, which remains a different policy. IGNORE and
return limits remain post-processing metadata and never change this mask.

S belongs to the request, like its KV cache. Reusing it across requests could
silently forbid products because an earlier caller happened to receive them.
The tests repeat a request and check that its output starts from fresh state.

## Why this still does not guarantee termination

It also cannot fix the **first** emitted entity: S is empty then, so that choice
is unchanged. This matters because the earlier diagnosis found first-decision
errors that later steps can amplify.

EOS is not fabricated at the length bound. If the model keeps selecting fresh
wrong IDs, it can still use all 507 completion positions without ending. There
are 1025 entities, so the vocabulary is larger than that bound. Finite choices
do not imply completion within this configured context.

This is why the experiment measures validity and EOS termination separately
from duplicate rate and F1. A sequence with no duplicates can still be wrong,
unfinished, or both. The parser and failure handling keep those distinctions.

## How we isolate the intervention

We use the same three symmetric checkpoints from Lesson 12. No retraining,
threshold tuning, graph-relation lookup or answer-count hint is involved. The
new flag, `eval.prevent_repeated_targets`, defaults to false and requires
protocol-constrained decoding. Reports append `+unique-v1` to the decoder name.

For every validation query, the campaign executes two paths with the same cache
mode and source: original greedy decoding, then the unique-target variant.
Every original output must first match its archived uncached reference exactly,
including any failure. This checks cache/source compatibility across all 666
queries before interpreting the policy differences. Model source files and
runtime versions are also checked against the archived implementation.

Both raw and post-processed metrics are recomputed from generated outputs.
We record changed queries, newly exact answers, lost exact answers and recovered
termination. The original failed responses remain evidence; they are not
rewritten with the new policy's answers.

A candidate qualifies for HTTP verification only if all 666 unique-target
responses are valid and terminated and neither processed F1 nor exact accuracy
regresses in any seed. The already chosen seed 1729 is retained if eligible.
Actual HTTP output parity and metadata checks are required before updating the
serving example. Final test and oracle-quality/performance claims remain separate.

## Implementation checks

Tests cover repetition suppression in cached and uncached generation, subject
eligibility, a fresh seen set on every request, refusal of incompatible settings,
unchanged truncation behavior and policy propagation through native serving.
The full suite passed 199 tests before running the real campaign.

## A separate batching prototype

The serial campaign also makes an engineering cost visible: every token waits
for a model call for one query. In a separate, bounded prototype, eight queries
share a forward call while keeping independent histories:

```text
prefill input IDs:       [8, 5]
next decode input IDs:   [8, 1]
next-token logits:      [8, 2049]
per-layer cached K/V:   [8, n_kv_heads, cached_positions, head_dim]
```

The model computes across the batch dimension; it still cannot generate all
positions of one answer simultaneously. Some rows reach EOS earlier than
others. The prototype stops recording those answers and feeds ignored dummy
inputs for their rows while the others continue. This preserves aligned cache
positions but spends work on finished rows; it is not continuous batching.

All eight original-policy outputs and eight unique-policy outputs matched their
serial references, including different completion lengths. This is a correctness
probe on eight queries, not full-partition validation, an admitted evaluator
backend or a native serving feature. It ran alongside another evaluation, so
its runtime cannot support a speed claim. The
[prototype receipt](../experiments/2026-09-24-batch-prototype.json) preserves that
scope. It does not replace any executions in the main paired policy campaign.

Follow-up: [Lesson 14](14-batched-evaluation.md) integrates this idea into the
offline evaluator, checks the full partition and measures a separate workload.
