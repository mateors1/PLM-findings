# 29. Same model, different batch shape: what must stay identical?

**Status:** the first-checkpoint numerical diagnosis is complete and independently
audited. A new verification contract has been reviewed; its implementation and
remaining HTTP checks are pending. The original integration campaign failed.

For the first checkpoint's 222 queries, offline evaluation and HTTP gave the
same answers, but their selection scores differed. This is a useful research lesson:
an answer can be correct while an experiment fails its declared requirement.

## Start with the tensors

Each prompt contains five token IDs: `BOS SUBJECT DIMENSION SAME ANSWER`.
For a batch of `B` queries, the input has shape `[B, 5]`. The model's symmetric
membership head returns `Z` with shape `[B, 1025]`: one unnormalized membership
score, or **logit**, for each product. Product columns follow a saved ID mapping;
column position is not automatically the product's protocol token ID.

The offline evaluator uses `B=8`, with six queries in the last batch. Native
HTTP uses `B=1`. The checkpoint and mathematical scoring rule are unchanged.
The relevant reference tensors are therefore `[8, 1025]` or `[6, 1025]` offline,
and `[1, 1025]` for each serial request.

Four generated paths give us four candidate sets. Their six pairwise unions
make ten slots in total. For query `q` and candidate set `S_k`, selection uses

```text
score(q, S_k) = fsum(Z[q, column(i)] for i in sorted(S_k))
k* = earliest eligible slot attaining the largest score
```

`fsum` is Python's accurate floating-point summation. Sorting and deduplicating
IDs make the inputs explicit; eligibility and ties are part of the algorithm.
If every source is invalid, the existing rank-one failure fallback applies.
The reference calculation must reproduce that failure behavior too.

## Why equal equations can produce unequal numbers

FP32 stores a finite approximation to a real number. Rounding happens during
matrix multiplication, before the resulting logits reach `fsum`. A different
batch shape can change how the arithmetic is arranged. This is distinct from
changing the model weights or sampling a different output.

For intuition, round after each operation to FP32:

```text
(100000000 + -100000000) + 1 = 1
100000000 + (-100000000 + 1) = 0
```

Real-number addition is associative; this finite-precision calculation is not.
Accurately summing two slightly different lists of logits does not make their
sums identical.

Our [saved diagnosis](../experiments/2026-09-24-pair-composition-batch-shape.json)
recomputed the first checkpoint's 222 queries at both shapes. All serial scores
matched HTTP exactly, and all batched logits matched the offline reference
exactly. Across shapes, 156,003 of 227,550 logits differed; the largest absolute
logit difference was about `7.63e-6`. All 222 selected slots stayed unchanged.
The first eight queries repeated exactly within each shape. These are observed
results on this checkpoint and environment, not a portability guarantee.

## Separate numerical equality from decision stability

The [revised contract](../experiments/2026-09-25-pair-composition-shape-aware-plan.md)
asks two different questions:

| Comparison | Requirement |
| --- | --- |
| Serial HTTP score versus independently computed serial reference | Exact numeric equality, zero tolerance |
| Repeated serial prompt logits | Exact equality |
| Offline versus serial source paths, sets, eligibility and selection | Exact equality |
| Offline versus serial logits and scores | Record differences; do not use them as a tolerance |

We still fail if the winning slot changes, even when both slots contain the
same set. Otherwise a seemingly harmless answer comparison could hide changed
selection behavior.

A **score margin** helps explain stability. Keep candidate sets and eligibility
fixed, and suppose there is a unique winner. Let it exceed the runner-up by
`m > 0`. If every slot score changes by at most `epsilon`, the
winner is guaranteed to remain ahead when `m > 2 * epsilon`: the winner can
lose `epsilon` while a competitor gains `epsilon`. This is a mathematical
intuition, not a newly chosen acceptance threshold. Our contract checks the
actual decisions exactly, including ties, rather than assuming stability from
small observed differences.

## Repair the experiment without rewriting its history

The original contract compared serial HTTP scores with batched reference
scores and demanded equality. That requirement failed. A diagnosis can explain
the failure; it cannot turn the original report into a pass.

The next campaign gets its own plan, script snapshot and output directory.
Before fresh HTTP checks, it must compute and seal serial references for all
three checkpoints: `3 * 222 = 666` prompt forwards, plus `3 * 8 = 24` repeat
forwards. These forwards produce reference logits; they do not regenerate the
four autoregressive paths.

Evidence reuse must also be visible:

| Evidence in the new campaign | Reused | Fresh |
| --- | ---: | ---: |
| Offline query observations | 666 | 0 |
| Serial reference query observations | 0 | 666, plus 24 repeats |
| Normal HTTP responses | 222 | 444 |
| Full HTTP auxiliary suites | 1 | 2 |

The saved first-checkpoint HTTP responses are rechecked against the new serial
reference. The remaining two checkpoints require real new HTTP requests.
Filtering, zero limits, hydration, invalid requests, admission rejection,
disabled behavior, short-bound failures and server shutdown all remain required.
An independent audit must reconstruct scores, selections and evidence accounting
before acceptance. Until then, this is an implementation and verification task.

## Where this fits in the research

This verifies application behavior. It does not fix the 97 remaining validation
errors, establish oracle parity or measure production throughput. In particular,
the [direct-head ablation](28-direct-membership-ablation.md) still failed its
quality gate, despite better average F1.

The future [learned-retrieval hypothesis](../experiments/2026-09-25-learned-retrieval-hypothesis.md)
asks a different question: can an identifier-only model infer useful held-out
relations competitively against simpler methods? Reliable evidence handling is
useful for both experiments; success on this verification cannot substitute for
the future dataset, baselines or quality result.

For a self-check: if the winning slot changes to another slot containing the
same set, should the application answer check pass? Yes. Should this stricter integration contract
pass? No, because it also promises to preserve the selected slot. Different
claims need different tests.
