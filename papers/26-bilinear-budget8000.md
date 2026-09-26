# Does more balanced-BCE optimization improve complete answers?

Status: completed on the first primary attempt, independently audited on the
first audit attempt, and accepted by the owner. The fixed single-seed quality
screen passes; this is not a serving promotion.

This fixed seed-1729 screen extends balanced binary cross-entropy (BCE) training
from 2,000 to 8,000 residual updates. It restarts from the original parent with
zero residual parameters and fresh AdamW state. The accepted 2,000-update child
is a saved historical comparator, not a checkpoint resumed for another 6,000
updates. The failed worst-member sibling motivates returning to balanced BCE;
it is not a matched control for the budget comparison.

## Result

| Measure | Historical balanced BCE, 2,000 updates | New balanced BCE, 8,000 updates |
|---|---:|---:|
| Exact sets / 222 | 207 | 216 |
| Macro F1 | 0.9997456353806234 | 0.9998843626799785 |
| False positives / false negatives | 8 / 10 | 1 / 8 |
| Strictly separated queries | 217 | 220 |
| COLOR exact / 103 | 103 | 103 |
| Single-TYPE exact / 51 | 50 | 51 |
| Dual-TYPE exact / 68 | 54 | 62 |
| Final post-update training BCE | 0.00003491543247946538 | 0.0000016234706663453835 |

All primary quality floors pass. Against the historical 2,000-update sibling,
there are nine paired gains and zero losses: eight dual-TYPE queries and one
single-TYPE query. Against width eight, there are 17 gains and two losses.
Against the dense original parent, there are 104 gains and zero losses. These
comparisons concern the same 222 validation queries, not independent replications.

All six remaining nonexact answers are dual-TYPE. All 222 outputs remain
serialization-compatible. Strict separation holds on 220 queries; this is a
ranking property and does not itself guarantee that the fixed zero threshold
produces the exact answer set. No alternative threshold was selected.

Training BCE fell from 0.003822767175734043 to
0.0000016234706663453835. The first 2,000 pre-update scalar losses match the saved
historical values exactly: zero mismatches, no first mismatch. Historical final
post-update-2,000 and new pre-update-2,001 losses also match exactly at
0.00003491543247946538. This is evidence of scalar-prefix agreement, not a claim
of identical unrecorded intermediate parameters or deterministic execution.

This result supports further declared replication of the larger fixed budget.
It does not establish oracle parity (six validation answers remain wrong),
convergence, performance on new seeds or protected data, or serving superiority.

## Intuition: more practice with the same learning signal

BCE asks the model to make true products score positively and false products
negatively. The previous training loss was still falling near update 2,000.
That observation motivates a longer run, but does not prove underfitting or
predict that validation answers will improve. A lower average training loss can
coexist with a worse boundary decision on a few products. Exact-set accuracy is
especially unforgiving: a single extra or missing product makes the whole answer
wrong.

The treatment changes the fixed update budget. Parent checkpoint, zero
initialization, scorer, training partition, optimizer and output rule stay the
same. The campaign and evaluator receive new identities because the declared
execution and acceptance contract changed; the architecture and objective retain
their existing identities.

## Features, parameters and scores

The frozen token embedding table has shape `[2049, 256]`. Its 1,025 product rows
become features `E` with shape `[1025, 256]`, each normalized and rescaled:

\[
 E_i = 16\,w_i / \lVert w_i\rVert_2.
\]

The only new trainable tensor is `A` with shape `[2, 256, 256]`, one matrix for
each steering dimension. For subject `s`, dimension `d` and candidate `i`, the
unchanged scorer adds a symmetric bilinear correction to the frozen score:

\[
 S_d=(A_d+A_d^\top)/2,\qquad
 z_{sdi}=z^{\mathrm{parent}}_{sdi}+E_s^\top S_d E_i/16.
\]

The symmetry ties the correction for the pair `(s, i)` to `(i, s)` within a
dimension. It is an architectural bias, not proof of correct relations. Exact
zero `A` makes the added term zero and must reproduce every parent validation
vector before fitting. All 93 original state tensors remain unchanged. The
completed derivative contains 94 state tensors, including `A`.

For one query, let `T` be true nonself products and `N` false nonself products.
The unchanged balanced objective gives the two classes equal weight:

\[
 \ell_q=\tfrac12\frac{1}{|T|}\sum_{i\in T}\operatorname{softplus}(-z_i)
       +\tfrac12\frac{1}{|N|}\sum_{i\in N}\operatorname{softplus}(z_i),
 \qquad L=\frac1{1637}\sum_q\ell_q.
\]

Thus an abundant negative class does not dominate merely because it is larger.
Training forms scores over 1,637 queries and 1,025 products; the self position is
excluded from both terms. Full-batch AdamW performs exactly 8,000 updates with
learning rate 0.0003, betas `(0.9, 0.999)`, epsilon `1e-8`, zero decay and fresh
optimizer state. There is no schedule, clipping, accumulation or mixed objective.
The execution remains FP32 with the frozen runtime and precision settings.

## Endpoint, outputs and fixed gate

Validation points are `[0, 8000]`. Zero is required parent replay; 8,000 is the
only fitted-child endpoint. After exact checkpoint and optimizer reload, scores
are evaluated for the same 222 validation queries at batch size eight (last
batch six). A prediction consists of every nonself product with `z > 0`, sorted
by identifier. Empty or oversized outputs are retained in evidence. There is no
threshold search, fallback, cardinality oracle or truncation.

The declared gate requires all execution and serialization invariants, more
than 207 exact sets, macro F1 at least `0.9997456353806234`, and group exact counts
at least COLOR 103, single-TYPE 50 and dual-TYPE 54. The historical balanced-BCE
2,000-update comparator established those quality floors. Gains elsewhere cannot
compensate for a failed floor. Paired comparisons also include the original dense
parent and width-eight predictor, with query order and labels checked.

## A descriptive check of the shared training prefix

Each recorded scalar is measured before its update. Compare the first 2,000
new pre-update losses with the 2,000 historical pre-update losses, reporting
mismatch count and the first mismatching update if one exists. Separately compare
historical final post-update-2,000 loss to new pre-update-2,001 loss. These two
measurements align at the boundary after 2,000 updates; new pre-update-2,000 is
one update too early.

This diagnostic cannot change the gate, stop training, select an endpoint or
authorize a retry. Scalar equality does not establish identical intermediate
weights or deterministic execution. Any mismatch remains part of the record.

## Evidence and limits

The runner, recipe and tests are frozen before execution. An independent auditor
checks saved training, checkpoint lineage, predictions, comparisons and the gate;
it does not repeat CUDA training. A separate owner decision distinguishes accepted
evidence from a quality pass. A completed rejected child still receives an
additive checkpoint registration. Publication identity remains separate from the
executed model, scorer and evaluator.

All 8,000 original update records are preserved in four exact 2,000-row JSON
chunks, each below 512 KiB. Each chunk binds the full local training-file hash and
its update range. Concatenation must reproduce the full history without sampling
or rounding. The oversized raw training file, raw score reports and weights
remain local hash-bound dependencies; hashes do not establish remote archival.

This is one adaptively selected validation screen in the Pokemon TYPE/COLOR SAME
setting. It does not establish convergence, oracle parity, independent-seed
robustness, protected-test performance or a serving/energy advantage. A pass may
motivate a separately declared replication, never automatic promotion.

## Registration and evidence

The completed child is final checkpoint 35, SHA
`6979013750eb8f2780917f714192850aae411a48490757236a891fea7e351f91`.
The inventory freshly hashes all 35 final weight files; prior payloads were not
revalidated by this inventory pass. The architecture and balanced-BCE objective
remain unchanged, while the recipe, runner and evaluator identify this 8,000-update
screen. Parent pretraining remains 2,000 steps; the residual has 8,000 updates.
The ordinary inherited decoder ignores `A` and does not serve this scorer.

The primary verification passed 129 CPU tests (38 new, 91 reused), with zero
skips. The independent auditor passed 70 synthetic tests. These tests, the actual
execution receipts and the separate owner decision preserve different evidence
layers. No documentation strict-build success is claimed here.

Read the [evidence packet](../evidence/updates/2026-09-25-bilinear-budget8000/README.md),
[frozen plan](../evidence/updates/2026-09-25-bilinear-budget8000/experiments/2026-09-25-bilinear-budget8000-plan.md),
[portable result](../evidence/updates/2026-09-25-bilinear-budget8000/experiments/2026-09-25-bilinear-budget8000.json),
[independent audit](../evidence/updates/2026-09-25-bilinear-budget8000/campaign/independent-audit.json),
[owner decision](../evidence/updates/2026-09-25-bilinear-budget8000/campaign/decision.json),
[teaching note](../evidence/updates/2026-09-25-bilinear-budget8000/learning/47-more-updates-with-the-same-objective.md)
and [checkpoint inventory](../evidence/updates/2026-09-25-bilinear-budget8000/experiments/2026-09-25-bilinear-budget8000-model-version-inventory.json).

The exact history is split into [updates 1-2000](../evidence/updates/2026-09-25-bilinear-budget8000/experiments/2026-09-25-bilinear-budget8000-training-part-01.json),
[2001-4000](../evidence/updates/2026-09-25-bilinear-budget8000/experiments/2026-09-25-bilinear-budget8000-training-part-02.json),
[4001-6000](../evidence/updates/2026-09-25-bilinear-budget8000/experiments/2026-09-25-bilinear-budget8000-training-part-03.json)
and [6001-8000](../evidence/updates/2026-09-25-bilinear-budget8000/experiments/2026-09-25-bilinear-budget8000-training-part-04.json).
