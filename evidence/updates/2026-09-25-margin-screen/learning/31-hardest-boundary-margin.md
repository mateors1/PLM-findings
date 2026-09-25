# 31. Teach the weakest member to outrank the strongest impostor

**Status:** completed and independently audited; the single-seed gate failed.
Adding the margin reduced exact answers from 192/222 to 6/222 and strictly
separable rankings from 167 to zero. Replication stops; the existing checkpoint
family and default-off setting remain unchanged. The
[declared experiment](../experiments/2026-09-25-symmetric-margin-plan.md)
fixed the comparison before training. This lesson preserves the hypothesis and
explains the negative result below.

## Why target the boundary?

[Lesson 30](30-ranking-versus-threshold.md) found that 81 of the current set
selector's 97 failures have overlapping membership rankings: some nonmember
scores at least as highly as a true member. Moving a cutoff cannot fix that
ordering. We tested an additional training signal that directly targets it.

Imagine true members with scores 4.0 and 1.0, and nonmembers with scores 2.0
and -3.0. The difficult comparison is between the true member at 1.0 and the
nonmember at 2.0. Improving the already confident scores may reduce average
error while leaving that decisive mistake in place.

A **margin** is a requested gap between the two sides. We ask the weakest true
member to score at least one unit above the strongest nonmember. A **hinge
loss** charges for a shortfall and becomes zero once the requested gap is met.
The gap and its coefficient are initial experimental settings, not established
optimal values.

## From the idea to an equation

For query q, let T_q contain its true members and N_q its nonmembers. Both
exclude the query subject. Define:

\[
a_q=\min_{i\in T_q}z_{qi},\qquad
b_q=\max_{j\in N_q}z_{qj},\qquad
g_q=a_q-b_q.
\]

The per-query penalty and batch loss are:

\[
\ell_q=\max(0,m+b_q-a_q)=\max(0,m-g_q),\qquad
L_{margin}=\frac{1}{B}\sum_{q=1}^{B}\ell_q.
\]

With m=1, our example pays 1+2-1=2. If the true boundary rises to 3 while
the negative boundary remains at 2, the loss reaches zero. Strict rank
separation only requires g_q>0; this objective requests the stronger condition
g_q>=1. A zero hinge does not require any particular zero-threshold decision.

The treatment adds the penalty to the existing objective:

\[
L_{treatment}=L_{control}+0.1L_{margin}.
\]

Here the control retains token cross-entropy, prompt-set BCE and symmetric
membership BCE, with their declared weights. It has no continuation-set loss.
We preserve these existing signals rather than replacing their supervision.

**Satisfied queries remain in the denominator.** If three query penalties are
[2,0,0], their mean is 2/3, not 2. Averaging only violating queries would change
the loss scale as training improves. The implementation separately reports the
fraction of queries whose hinge is positive.

## The tensor view

The existing symmetric head produces FP32 logits
Z in R^(B x 1025): one row per query and one column per product. The campaign
uses embedding width 256; the head scores the subject against every product
using shared entity embeddings and a dimension-conditioned relation vector.
See [lesson 28](28-direct-membership-ablation.md) for that head's equation.
This new loss adds no parameters and does not enlarge the head.

Target and nonmember masks both have shape [B,1025]. Supervised product IDs
are deduplicated into the target mask. Control tokens, masked labels and padded
positions do not become members. The subject column is excluded from both
classes; a target label containing the subject is rejected. Each row must have
at least one positive and one negative product, so both extrema exist.

```python
# Conceptual view of the implemented FP32 reductions.
a = Z.masked_fill(~true_mask, float("inf")).amin(dim=-1)   # [B]
b = Z.masked_fill(~negative_mask, -float("inf")).amax(dim=-1)  # [B]
hinge = relu(1.0 + b - a)                                # [B]
loss = hinge.mean()                                     # scalar
```

The infinities only hide excluded columns during reduction; valid rows reduce
to ordinary finite scores. Empty classes raise an error. The actual operation
runs with autocast disabled and FP32 scores, including during BF16 training.
BF16 uses fewer bits for many transformer operations; retaining FP32 here keeps
the small boundary calculation at the head's existing precision.

## Which weights receive the correction?

For an active query with unique extrema, before applying the coefficient 0.1:

\[
\frac{\partial L_{margin}}{\partial a_q}=-\frac{1}{B},\qquad
\frac{\partial L_{margin}}{\partial b_q}=+\frac{1}{B}.
\]

Gradient descent therefore pushes the weakest true score up and the strongest
negative score down. Other scores receive zero direct gradient from this query's
hinge. If k true members tie for the minimum, `amin` divides that side's gradient
equally among them; `amax` does likewise for tied negatives. At the exact
zero-hinge boundary, PyTorch ReLU gives zero gradient.

The head's parameters and entity/dimension embeddings receive the resulting
gradients. Its subject score is excluded, but the **subject embedding still
participates in scores against other products** and can receive gradients.
The symmetric head does not use decoder hidden states, so this auxiliary loss
does not directly backpropagate through the transformer blocks or prompt-set
projection. However, entity embeddings are shared with the decoder and tied
output scoring. Updating them can change future generation. This is not a
head-only intervention with guaranteed unchanged generated paths.

Training labels define the masks; they are legitimate supervision. During
prediction, expected sets do not enter the head, path generation or selection.
Validation may compute labeled losses and separation diagnostics without
gradients, separately from the model's predictions.

## Ranking and calibration are different questions

Balanced binary cross-entropy (BCE) averages positive and negative penalties
separately, giving the two classes equal weight per query. It encourages true
scores upward and false scores downward across the entire set. That weighted
objective does not make sigmoid scores automatically calibrated membership
probabilities under the original class frequencies.

The new hinge compares ordering at the hardest boundary. Adding the same
constant to every score changes neither a_q-b_q nor the hinge, although it can
change which products pass a zero threshold. Thus better separation does not
by itself establish better calibration or exact unrestricted set prediction.
Nor does it guarantee better selection among generated sets: their availability
and their summed scores still matter. The experiment measures these distinctions.

## What the implementation checks establish

The implementation defaults to `model.symmetric_margin_loss_weight=0.0`, with
`model.symmetric_margin=1.0`. Positive weight requires an enabled symmetric BCE
head. Disabled execution skips the new loss entirely. Active checkpoints record
the objective contract `hard-boundary-hinge-fp32-amin-amax-query-mean-v1`.

Verification before the experiment includes:

- 818 passing CPU-suite tests, with nine tests skipped; Ruff, formatting
  and strict typing across 62 source files pass.
- Arithmetic, tied-gradient, subject exclusion, padding, empty-class and
  BF16/FP32 tests; query-weighted validation reporting and exact same-code
  training resume tests.
- 92 bit-exact CPU tensor comparisons against archived pre-change source,
  covering initialization/RNG, dropout outputs, losses, gradients, one AdamW
  update and unlabeled/cached outputs on a small model.
- Three synthetic CUDA BF16 optimization steps with FP32 margin computation
  and finite gradients. This is a numerical smoke check, not a dataset result.

The local receipts are `runs/learning/margin-default-contract-v1/summary.json`
and `gpu-smoke.json`. These checks do not establish full-training or GPU
bitwise equivalence to the historical model.

Historical configurations authenticate their original hashes before a paired
absence of the two new fields receives disabled defaults. Partial fields or
coerced metadata are rejected. The historical training identity stays intact;
the effective configuration gets its own hash. Loading old weights for
inference is distinct from resuming training: an old raw training contract
cannot silently become a new one. See the [version register](../model-versions.md).

Token cross-entropy, raw margin loss, participating query count and active
fraction are logged separately. The margin is query-averaged; token CE is
token-averaged. Their magnitudes answer different questions, so a lower combined
training loss alone cannot establish an improvement.

## The experiment we declared and ran

We trained a fresh control and treatment at seed 1729, each for the full 2000
steps, under the same frozen source and recipe. They differ only in run name
and margin coefficient, 0.0 versus 0.1; both use m=1.0. Both runs completed before
their final checkpoints were evaluated for generated answers. The old 191/222
exact result is context; the new control's 192/222 is the paired comparison.
The 92 bit-exact CPU comparisons do not promise reproduction of the historical
GPU training run: the fresh control is one exact answer above that older result.

Evaluation retains all 222 validation queries and the existing four guided
paths plus six pair unions, alpha 16, unique constrained cached decoding and
canonical additive set selection. Expected labels never choose a path or set.
The protected final test is not used.

The first screen requires a strict gain in exact answers and strict separation,
no macro-F1 loss and no exact loss in any TYPE/COLOR subgroup. Every source path
and selected set must also meet the validity/eligibility contract, with intact
identities and independent audit. Failure stops this variant without retuning.
Only a pass authorizes identical paired runs at seeds 1730 and 1731. The plan
specifies the additional per-seed and pooled acceptance gates and applies the
same validity requirements to every arm.

## The observed result: the intervention damaged both ranking and answers

| Measure, same 222 validation queries | Fresh control | Margin treatment |
| --- | ---: | ---: |
| Exact selected sets | 192 | 6 |
| Macro precision | 98.07% | 48.85% |
| Macro recall | 96.45% | 88.32% |
| Macro F1 | 96.77% | 61.56% |
| Strictly separable membership rankings | 167 | 0 |
| Direct zero-threshold exact sets, diagnostic only | 110 | 0 |
| Mean selected set size | 125.28 | 232.97 |
| Pair-composition selections | 29 | 210 |

The treatment gained one exact answer and lost 187: 192+1-187=6. All three
groups deteriorated, including COLOR, where the control answered every query
exactly. These are 222 distinct queries observed under two trained models,
not 444 independent query identities.

| Group | Queries | Control exact | Treatment exact |
| --- | ---: | ---: | ---: |
| COLOR | 103 | 103 | 2 |
| TYPE single | 51 | 47 | 0 |
| TYPE dual | 68 | 42 | 4 |

APPLIN/COLOR makes the precision failure concrete. The control selected all 129
expected products with no extras. The treatment still recovered those 129,
but added 147 incorrect products, returning 276 in total. Its recall stayed
at 100%, while precision fell to 46.74%. ARMAROUGE/TYPE was the sole exact gain:
the control omitted one of 179 members and the treatment recovered the full set.
That isolated success does not offset the widespread losses.

Every one of the 1,776 generated source paths across both arms remained valid,
terminated and eligible. All 444 selected sets were eligible too. The independent
audit reconstructed all 4,440 candidate slots and scores, selections, metrics
and gates from the saved evidence and found agreement. **Protocol correctness
does not imply relational correctness:** emitting valid product IDs and EOS
does not establish that those products belong in the answer.

## What the separate losses reveal

At the final step, the validation diagnostics were:

| Diagnostic | Control | Treatment |
| --- | ---: | ---: |
| Token cross-entropy | 0.129620 | 0.146441 |
| Symmetric membership BCE | 0.008737 | 0.680694 |
| Raw margin loss | disabled | 1.133733 |
| Active margin queries | not computed | 222/222 |

The membership BCE increase is large. For orientation, setting every binary
logit to zero gives balanced BCE log(2), approximately 0.6931. The treatment's
loss is close to that reference value, although that does not mean its logits
are all zero. Its zero-threshold rule selected an average of 586.36 products,
versus 130.19 for the control. Both the ranking diagnostic and the resulting
answers show that membership discrimination deteriorated.

All 222 treatment hinges remained active, and none of its saved membership
rankings had strict separation. The added term failed at its intended target
as well as the final task. Token CE also worsened, but much less dramatically;
teacher-forced next-token loss alone would understate the set-quality failure.

One plausible explanation is optimization interference between the hardest-pair
penalty and the existing objectives through shared embeddings. A coefficient
of 0.1 does not guarantee a small parameter update: gradient magnitude and
direction matter, and this hinge concentrates its correction on extrema.
However, this experiment did not measure gradient alignment or isolate the
optimization mechanism. Conflicting objectives are a hypothesis, not an
established cause. The result also does not prove that every margin objective,
coefficient or training schedule must fail.

### Why a small coefficient can still matter

For one query with P positive products, its balanced BCE positive term is
`-(1/(2P)) * sum(log(sigmoid(z_i)))`. The derivative for a positive score is
`(sigmoid(z_i)-1)/(2P)`. At z=0 its magnitude is 1/(4P), whereas an active
unique-minimum hinge contributes magnitude lambda to that boundary score.
The shared batch-mean factor cancels when comparing these magnitudes.

For illustration, with P=100 and lambda=0.1, the hinge's direct derivative on
the weakest positive is 40 times that score's BCE derivative at zero. That is
an illustrative score-space calculation, not a measured parameter-gradient
ratio: the shared embedding Jacobian, other objectives, tied extrema, clipping
and AdamW all affect the actual update. It explains why reading only the loss
coefficient can be misleading, without proving the cause of this failure.

## The decision and evidence boundary

A separate [post hoc saved-score diagnosis](../experiments/2026-09-25-margin-failure-diagnosis.json)
quantifies what happened to the logits, without another model run:

| Mean across validation queries, subject excluded | Control | Treatment |
| --- | ---: | ---: |
| Maximum minus minimum logit | 44.80 | 0.245 |
| Population standard deviation of logits | 7.96 | 0.0376 |
| Weakest true minus strongest negative | 2.665 | -0.134 |
| Unit-margin hinge recomputed from saved FP32 logits | 1.216 | 1.134 |

The scores became strongly compressed. The mean hinge even fell while strict
separation and exact answers collapsed: a few large violations and many small
violations can have similar average costs but very different success counts.
This demonstrates the limit of optimizing and reporting only the surrogate's
mean. It does not establish why the optimizer followed this trajectory.
An independent saved-only audit recomputed all 444 rows, group statistics and
20 logged training checkpoints exactly. Its local receipt is
`runs/learning/margin-failure-diagnosis-v1/independent-audit.json`, SHA256
`d5b85d69c087bd595772c6ffb2cdc36ab816451944db34856b510dadb16ef1b5`.

The validity gate passed; every declared quality gate failed. Under the frozen
plan, seeds 1730 and 1731 are not run for this variant. There is no coefficient
retry, default change or promotion of either new checkpoint. The prior accepted
composition family remains the application reference. These two completed
artifacts stay in the [version register](../model-versions.md), including the
negative outcome rather than disappearing from the research history.

The [portable result](../experiments/2026-09-25-symmetric-margin-screen.json)
binds the reports, independent audit and separate owner decision. The immutable
local campaign is `runs/learning/symmetric-margin-screen-v1/`.
Its summary SHA256 is
`39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701`;
the independent audit SHA256 is
`17afc9ca0fedf184ac9e9f2794947306883d4288860b22124cd9eb0c7751bcd9`.
The audit passes as a check of the evidence; it does not turn a failed scientific
gate into a successful treatment. This remains a single-seed validation result,
with no protected-test, oracle-parity, throughput, energy or Experiment B claim.
