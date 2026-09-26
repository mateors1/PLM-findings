# 47. More updates with the same objective

**Status: independently audited single-seed screen passed.** Returning to the
accepted balanced-BCE recipe and increasing the fixed budget from 2,000 to
8,000 updates improved exact answers from **207/222 to 216/222**. Nine questions
became correct and none became incorrect. The earlier worst-member experiment
remains a failed experiment with a different objective.

## What changes and what stays fixed?

The architecture, original parent, zero residual initialization, training data,
loss function, optimizer settings and prediction rule stay fixed. The treatment
is the number of residual updates. The accepted 2000-update output is the saved
historical comparator; we do not resume its weights or optimizer.

Restarting matters. It makes the 8000-update child a sibling trained from the
same starting state. Continuing a different selected child would introduce
another difference. We record parent pretraining steps and residual updates
separately because they belong to different training procedures.

The previous balanced loss was still falling: about 0.0000550 before update 1500,
0.0000380 before 1900 and0.0000349 before 2000. More optimization is therefore a
reasonable experiment. A falling training loss does not prove the model needs
more training, and it does not guarantee improvement on held-out questions.

## An update is not a new example

Each residual update uses the full training partition. The score matrix stays
**[1637,1025]**: one row per training question, one column per Pokemon candidate.
Balanced BCE first averages true-member and false-member penalties separately
within each query, then combines the classes and averages the queries.

The scalar loss differentiates into **A[2,256,256]**. AdamW tracks gradient
moment estimates with that same parameter shape. Frozen features and the 93
original state tensors do not change; the transformer is not retrained during
this residual fit.

Here one full-batch update includes one complete pass over the 1637 training
queries, so it corresponds to one epoch of this residual objective. The 8000
updates reuse the same examples; they do not create four times as much evidence.
This relationship between epochs and updates would change with minibatching or
gradient accumulation.

## Where in an update was the loss measured?

Let A_0 be the initial zero residual. A recorded **pre-update loss** at update k
is measured before the optimizer changes A:

\[
\ell_k^{\mathrm{pre}}=L(A_{k-1}),\qquad
g_k=\nabla_A L(A_{k-1}).
\]

AdamW then uses g_k and its stored moments to produce A_k. A final **post-update
loss** after K updates is

\[
\ell_K^{\mathrm{post}}=L(A_K).
\]

Thus the old run's final post-update 2000 loss aligns with the new run's
pre-update 2001 loss, not its pre-update 2000 loss. Off-by-one comparisons can
make matching training procedures appear different.

We compare the first 2000 pre-update scalar records and that boundary value as
descriptive diagnostics. Exact equality of scalar losses would not prove the
underlying tensors are identical: different states can have the same loss.
Mismatch is reported honestly, without changing the gate or restarting the run.

## Why not stop whenever validation looks best?

Repeatedly evaluating validation and choosing the best checkpoint adds another
selection process. For this experiment we declare 8000 in advance and evaluate
only the fitted child at that endpoint. The required update 0 evaluation verifies
the original parent and zero-residual replay; it does not select trained weights.

The new child must exceed 207/222 exact answers, maintain macro F1 at least
0.9997456353806234, and retain group exact counts of at least 103 COLOR,
50 single TYPE and 54 dual TYPE. All execution and serialization invariants must
pass, followed by an independent evidence audit. A near miss remains a failure.

The [frozen plan](../experiments/2026-09-25-bilinear-budget8000-plan.md) allows
one attempt at this endpoint. A pass would justify a separately declared
replication. It would not prove convergence, oracle parity, unseen-data quality
or serving efficiency. No protected evaluation or serving change is included.

## Keeping the longer trace reproducible

All 8000 update records are retained. The publication copy splits them into four
consecutive 2000-record files, with source hashes and update ranges. Concatenating
those chunks must reproduce the full original list exactly. Chunking changes
storage, not the evidence: there is no smoothing, rounding or downsampling.

## What happened?

The frozen runner passed 129 CPU tests and the independent auditor passed 70
synthetic tests before execution. The actual experiment and saved-evidence audit
each completed once, without repairs or retries. All declared gates passed.

| Measure | Historical 2,000 updates | New 8,000 updates |
| --- | ---: | ---: |
| Exact sets | 207/222 | 216/222 |
| Macro F1 | 0.9997456354 | 0.9998843627 |
| False positives / false negatives | 8 / 10 | 1 / 8 |
| COLOR exact | 103/103 | 103/103 |
| Single-TYPE exact | 50/51 | 51/51 |
| Dual-TYPE exact | 54/68 | 62/68 |
| Strictly separated score sets | 217/222 | 220/222 |

An exact answer requires every membership decision in a query to be right:

\[
\operatorname{Exact}(q)=\mathbf{1}[\widehat{Y}_q=Y_q],\qquad
F_{1,q}=\frac{2TP_q}{2TP_q+FP_q+FN_q}.
\]

Macro F1 averages the per-query F1 values. Exactness gives no partial credit:
one omitted Pokemon makes that query nonexact even if hundreds of other
memberships are correct. All six remaining failures are dual-TYPE questions.
Four of these have strictly separated true/false scores but the fixed zero
boundary still gives the wrong set; two have overlapping or tied boundaries.
This describes the saved scores, without selecting a new threshold.

Training BCE fell from 0.0038227672 to 0.0000016235. The first 2,000 recorded
pre-update losses matched the historical trace exactly, as did the aligned
post-update 2,000 / pre-update 2,001 boundary. This supports the intended
comparison while leaving the intermediate-state caveat above intact.

The audit verified all 93 original tensors unchanged, the trained residual,
8,000 optimizer steps, exact reload receipts and the independently recomputed
metrics. It inspected saved evidence; it did not repeat CUDA training or
regenerate the logits. [The portable result](../experiments/2026-09-25-bilinear-budget8000.json)
links the complete trace and separate acceptance decision. The
[model register](../model-versions.md) now includes this as final checkpoint 35.

More updates helped this seed under this fixed recipe. We have not established
that more updates always help, that 8,000 is optimal, or that the same improvement
will repeat across parent seeds. Replication needs a separate declared plan.
These are adaptively reused validation queries; the protected partition remains
untouched. Oracle parity, ordinary serving support and durable weight archival
are still unfinished.
