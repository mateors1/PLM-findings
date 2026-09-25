# 26. From a successful experiment to an explicit set API

**Status:** incorporated into the main checkout after the preceding four-path
campaign passed its complete independent audit. All 661 tests pass with CUDA
available, including the replay harness and review fixes. Ruff, formatting and
strict typing over 62 source files pass. A CPU-only replay of saved evidence
matches all 666 queries and 6,660 slots exactly. All three fresh offline GPU
checkpoints pass independently. The first HTTP checkpoint fails exact score
parity, while all 222 answers and raw source paths match. Full composition
integration acceptance remains pending; the other two HTTP checkpoints did not run.
The 569/666 exact result belongs to the completed offline experiment in
[lesson 25](25-composing-candidate-sets.md), not yet to this application path.

## What changes when the answer is assembled?

An autoregressive model produces an ordered sequence, one next-token choice at
a time. That sequence has a beginning, a last token, and possibly an EOS token.
Our application can now choose a union of two such outputs. The union has
members and source evidence, but no single generated ordering or EOS event.

This is why the implementation has two result types:

| Result | Meaning | Evidence |
| --- | --- | --- |
| `GenerationResult` | One emitted path | Full token IDs, targets, EOS/termination, protocol checks |
| `SetCompositionResult` | One selected canonical set | Four original paths, ten candidate slots, scores, source ranks and eligibility |

Canonical means serialized in ascending product-ID order. It makes two equal
sets compare identically; it does not claim that the model generated that order.
The frozen dataclasses contain tuples so later filtering cannot rewrite the
source evidence accidentally.

## The computation, with shapes

For B queries, the prompt token IDs have shape **[B, 5]**. Each of four branches
chooses a different first product and continues greedily with its own cache and
seen-product state. Keeping branches independent prevents one branch from
silencing a product or reusing an attention state from another branch.

The final prompt-only symmetric head supplies **Z [B, 1025]**, in FP32: one
membership score per product and query. We keep that pass and the existing
per-branch guidance passes unchanged. Composition shares the final scores;
it does not generate the four paths a second time.

Four originals plus six pair unions give ten slots. Conceptually, membership
could be represented by **C [B, 10, 1025]**:

```text
C[b, k, i] = 1 if product i belongs to candidate slot k for query b
score[b, k] = sum_i C[b, k, i] * Z[b, i]
```

The implementation uses canonical Python sets/tuples and `math.fsum` instead
of allocating that dense membership tensor. Each product contributes once:

```text
s(A union B) = s(A) + s(B) - s(A intersection B)
```

A pair can be selected only if both source paths are complete, protocol-valid
and unique. The highest score wins; ties prefer earlier slots, with all four
originals before the six pairs. Ground-truth targets, desired result counts and
graph relation facts never enter this selection function.

## Two endpoints with clear contracts

`POST /v1/predict` keeps its existing single-path response. A separate
`POST /v1/predict-set` exposes composition when the default-off
`eval.pair_set_composition` option is enabled. It requires the existing
`eval.symmetric_set_reranking` option and its trained-head, constrained-decoding,
uniqueness and KV-cache prerequisites.

Both endpoints use the same admission semaphore and execution lock. A semaphore
limits how many requests may be in flight; the lock serializes model execution.
Adding an endpoint must not accidentally allow a second request to run the model
outside that limit.

For a successful set request, the response contains the full selected set,
all ten scored slots, all four raw paths and the hydrated product result.
`IGNORE` and `limit` apply after selection. A request with `limit=0` can therefore
have a valid selected set and an empty displayed result. It is different from
a failed generation.

If no source is eligible, the response is a 502 error with source evidence.
It has no successful hydrated answer. The disabled endpoint returns 409;
invalid requests return 422 and exhausted admission capacity returns 503.

## Failure must remain visible in the metrics

For a successful predicted set P and target set T:

```text
precision = |P intersection T| / |P|
recall    = |P intersection T| / |T|
F1        = 2 * precision * recall / (precision + recall)
```

Undefined empty-prediction precision is zero here. A failed composition gets
zero precision, recall, F1, exact credit and answer size, even if a truncated
source happens to contain every expected product. Macro averages give each
query equal weight. Mean size includes every query in its denominator, with
zero size for failures. Successful original and pair counts plus failure count
must equal the number of queries.

Source paths retain their own termination, protocol and error statistics under
`source_path_*` names. A composed set receives no sequence-accuracy or
termination-rate metric. CLI evaluation uses the separate version
`plm-ranking-and-set-composition-v1`, with `set_generation` and `set_responses`.
This prevents old report consumers from silently treating a union as a sequence.

## What a failed test taught us about floating point

The tiny CPU checkpoint test first compared complete serial and batched reports
for exact equality. Paths, selections, memberships and metrics matched, but
some slot scores differed by at most **3.67872416973114e-08**. Across the two
tested objective settings there were 29 and 28 such differences, all confined
to slot scores. The differences were saved under `runs/staging/` in the isolated
checkout before changing the test.

FP32 arithmetic rounds intermediate results. A matrix multiplication with a
different batch shape can use a different accumulation order. `math.fsum`
stabilizes the later sum of supplied logits; it cannot undo differences already
present in those logits.

The CPU cross-batch test now requires exact non-score evidence and checks scores
with absolute tolerance 1e-7 and relative tolerance 1e-6. This change applies to
that CPU comparison only. The declared GPU replay still starts with zero score
tolerance: offline inference uses the reference batch shape, while HTTP processes
one query at a time and is also checked against the saved batched scores. This
cross-shape HTTP check is a stronger numerical requirement and may expose rounding
differences. Any discrepancy must be retained and diagnosed.
Numerically close scores are not enough if they change the selected answer.

The first real HTTP checkpoint exposes this distinction. All 2,220 normal slot
scores differ from their batched references, with maximum absolute difference
0.00036144256591796875. For one slot, the scores are 1396.0463314056396 and
1396.0466928482056. Every other composition field matches: all 888 source paths,
222 selected sets and 222 selected slots are unchanged. Exact count remains
191/222. The two filtered requests add twenty score differences and preserve
their original selections. Legacy routing, invalid requests, admission limits,
disabled behavior, short-bound failure and owned-server shutdown receipts pass.

An [independent failure audit](../experiments/2026-09-24-pair-composition-http-failure.json)
confirms these findings. The campaign correctly
stops under its declared zero-tolerance gate; it is not accepted retroactively.
A separate, [independently audited prompt-head replay](../experiments/2026-09-24-pair-composition-batch-shape.json)
now reproduces the discrepancy while keeping
checkpoint, source and prompts fixed. Across all 222 queries, batches of eight
(with the final six) reproduce the original logits and sums exactly; serial
queries reproduce all 2,220 HTTP scores exactly. The first eight queries also
repeat exactly at each shape. In total, 156,003 of 227,550 logits differ across
shapes, with maximum absolute difference about 7.63e-6. No positive-membership
sign or selected-slot decision changes in this diagnosis.

The experiment uses FP32 with float32 matrix-multiplication precision set to
`highest` and CUDA matrix-multiplication TF32 disabled. Even these settings do
not make differently shaped calculations bit-for-bit identical. Floating-point
addition rounds: in FP32, `(100000000 + -100000000) + 1` gives 1, while
`100000000 + (-100000000 + 1)` gives 0. Different accumulation orders can therefore
change a matrix product. Accurate `math.fsum` of the resulting logits cannot
recover rounding differences that already occurred inside those products.

This diagnoses the first checkpoint; it does not verify the two unexecuted HTTP
checkpoints. No score tolerance has been introduced, and the observed difference
is not a newly selected tolerance. A revised verification contract should compare
native scores with same-shape references and still require cross-shape equality
of paths, selected sets and decisions. That is a proposed next step, not a
retroactive pass for the original campaign.

Independent review caught two further verification gaps. The real-checkpoint
HTTP test now compares the actual response body against serial CLI evidence;
checking only the HTTP status and then rereading CLI fields was insufficient.
The metrics function now checks four source paths per query. Checking only the
aggregate count could hide one three-path query alongside one five-path query.
This is the difference between a local invariant and an average that looks right.

## Why parallel work needs isolation

The preceding campaign had to finish against the same source and configuration
it started with. We copied the current dirty working tree into an isolated checkout,
verified 306 file hashes, and checked that Python imports resolved there.
Tests use `CUDA_VISIBLE_DEVICES=-1` so they cannot compete for the campaign GPU.
On this Windows shell, assigning an empty environment value removed the variable;
the explicit `-1` setting was needed to make CUDA unavailable.

Workers own separate files: composition/configuration, native transport,
evaluation/documentation, and the replay harness. A local snapshot also avoided
a checkout helper's stalled remote credential lookup. These are engineering
controls for keeping the experiment interpretable, not new model techniques.
CPU work still shares machine resources with HTTP serving. These campaign times
are diagnostic observations, not controlled latency/throughput benchmarks.

The acceptance requirement is the full application replay specified in the
[integration contract](../experiments/2026-09-24-pair-composition-integration-plan.md).
Its preflight authenticated 138 inputs before model execution. Independent review
also strengthened the harness: a later offline exception now preserves completed
rows and mismatches, and HTTP checks reject unexpected top-level sequence claims.
The numerical diagnosis and remaining HTTP verification are unfinished. The
remaining 97 offline errors, protected final test and matched-quality
serving/energy comparisons remain open.
