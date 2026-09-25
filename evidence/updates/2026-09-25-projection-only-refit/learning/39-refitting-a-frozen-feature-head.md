# 39. Refitting a head while freezing its features

**Status: independently audited evidence accepted; fixed quality gate failed.** The
[projection-refit plan](../experiments/2026-09-25-projection-only-refit-plan.md)
fixed the recipe and decision gate before measurement. The
[portable result](../experiments/2026-09-25-projection-only-refit.json) records
the completed 500-update screen and its separate outcome decision.

## Why try this next?

The [extra-branch experiment](38-a-branch-for-uncovered-products.md) found more
useful candidate sets but selected fewer completely correct answers. More
search alone was insufficient. This experiment asks whether we can improve
membership decisions using the representations the model already learned.

Think of an embedding as a set of learned coordinates for a Pokemon. Freezing
embeddings fixes those coordinates. We then adjust only the small calculation
that combines a subject, a relation dimension and a candidate product into a
membership score. This is a form of **head-only fine-tuning**: a trained model
provides fixed features, and a restricted trainable head learns to use them.

The restriction helps interpretation. If validation improves, this particular
refit extracted a better decision rule from the fixed features. If it fails,
that is evidence about this recipe and budget; it does not prove that the
features contain no useful information or that all head-only fitting must fail.

## Follow the tensors

There are N=1,025 product tokens and D=256 coordinates per embedding. Normalize
each embedding to unit length, then scale it by sqrt(D)=16, as the existing
head does. Let E have shape [1025,256], containing product embeddings. Let U
contain the frozen TYPE/COLOR steering embeddings.

For a batch of B subjects, select their rows E_s [B,256] and corresponding
steering rows U_d [B,256]. The trainable projection W has shape [256,256]:

\[
R=U_dW^T\quad [B,256],
\]

\[
Z=\frac{(E_s\odot R)E^T}{\sqrt D}\quad [B,1025].
\]

The symbol odot means elementwise multiplication. Each row of Z contains one
score for every possible product. A score is a **logit**, an unrestricted real
number. Applying sigmoid would map it into (0,1), but that alone would not make
it a calibrated probability. Calibration means agreement with observed event
frequencies, which this experiment does not establish.

The full training batch has Z [1637,1025]. Validation uses the historical B=8
batches and a final B=6 batch. Matrix dimensions can affect floating-point
rounding, so replay uses the same shapes as the saved reference evidence.
There are no transformer attention states or KV caches in the refit itself.

## What loss teaches the head

For query q, let P_q be the teacher's positive products and N_q its negative
products. The SAME subject is excluded from both. The existing loss is

\[
L_q=\frac12\left[
\frac1{|P_q|}\sum_{i\in P_q}\operatorname{softplus}(-z_{qi})+
\frac1{|N_q|}\sum_{i\in N_q}\operatorname{softplus}(z_{qi})\right],
\]

\[
L=\frac1{1637}\sum_q L_q,
\qquad \operatorname{softplus}(x)=\log(1+e^x).
\]

Positive examples are penalized for low scores; negative examples are penalized
for high scores. Averaging each class separately prevents a large negative class
from winning simply by having more members. Averaging query losses gives every
training query equal weight. Duplicate target IDs count once.

We use the existing training labels only. The recipe fixes 500 full-batch AdamW
updates at learning rate 0.0003 with zero weight decay and a fresh optimizer
state. Since each update sees all 1,637 training queries, this is 500 complete
passes through that training set. It is much more exposure per update than the
parent's batches of 32, while involving far less computation per query.

An **optimizer state** is the optimizer's memory, separate from model weights.
AdamW tracks a moving average of gradients and of squared gradients for each
trainable weight, plus an update counter. Here each moving-average tensor has
shape [256,256]. Starting fresh means we inherit the parent's learned W but
discard its previous optimizer memory. The child checkpoint saves the new
optimizer state, and the reload check verifies both weights and that state.

In code, freezing disables gradient accumulation for the other parameters;
detaching cached features prevents gradients from flowing into their producer.
We also compare every saved non-W state tensor byte for byte. These checks
test what changed rather than relying only on the intended training configuration.

## What freezing does mathematically

With E and U fixed, every logit is linear in W. Softplus is convex, and positive
weighted sums preserve convexity, so this objective is convex in W. Intuitively,
the loss has no isolated bad local valleys of the kind a nonconvex objective
can have. That does not guarantee that 500 AdamW updates reach an optimum.

There is also redundancy. Only W's action on the two steering vectors matters.
Although W contains 65,536 entries, at most 2 times 256 = 512 effective directions
are identifiable through those vectors, and there may be further dependencies.
Thus convex does not mean strictly convex or a unique solution. For separable
data, logistic-style losses can even approach their infimum as weights grow
without reaching a finite minimizing weight matrix.

Lower training loss also does not guarantee better exact sets on validation.
One wrong product is enough to make an entire answer non-exact. Our scientific
gate therefore measures the outputs we care about rather than just the loss.

## The predictor and the gate

Select every non-subject product with z greater than zero, exclude exact zeros,
and sort the IDs. That is **dense membership prediction**, not autoregressive
generation. Empty or oversized predictions fail serialization compatibility,
while retaining their raw set-quality metrics; we do not hide them behind
truncation or a fallback. Sets of 1 through 506 products fit the
existing five-token prompt plus products plus EOS budget, but this experiment
does not claim that the decoder emitted those sets or EOS.

The declared gate required beating the accepted seed-1729 width-eight result: more than 201/222
exact answers, macro F1 at least 0.9799255176742276, and no group exact regression
below COLOR 103, single-TYPE 50 or dual-TYPE 48. It must also satisfy the fixed
evidence and output checks. We report the dense parent's result as a paired
comparator too. Passing this one-seed screen would justify replication, not
immediate deployment or a claim of oracle parity.

## What actually happened

All 500 updates completed. Training loss fell from 0.0038227672 to 0.0019596929,
a 48.74% reduction. The independent CPU audit inspected all 93 saved state
tensors: only W changed. The optimizer has 500 updates, and the child checkpoint
has the declared new objective and exact parent lineage.

| Predictor, seed 1729 | Exact answers / 222 | Macro F1 |
| --- | ---: | ---: |
| Parent, dense positive-logit rule | 112 | 99.0374% |
| Child, same dense rule | 122 | 99.1855% |
| Accepted width-eight candidate selection | 201 | 97.9926% |

The first two rows isolate the effect of the refit under the same prediction
rule. The third is the stronger existing system we required the child to beat;
it uses a different prediction procedure. Child versus dense parent gives 13
exact gains and 3 losses. Against width-eight selection it gives 2 gains and
81 losses. The fixed gate fails; we do not run additional seeds or promote it.

| Group | Dense parent exact | Dense child exact | Required minimum |
| --- | ---: | ---: | ---: |
| COLOR, 103 queries | 75 | 77 | 103 |
| Single-TYPE, 51 queries | 32 | 37 | 50 |
| Dual-TYPE, 68 queries | 5 | 8 | 48 |

This is a small improvement to dense membership prediction: false positives
fall from 512 to 389, while false negatives rise from 76 to 109. It does not close the gap to the accepted
candidate-selection system. All 222 child sets fit the serialization bounds,
so output length does not explain this failure.

Why can F1 exceed 99% while exactness is only 122/222? Consider Applin/COLOR:
the child returns all 129 correct products plus one false product. Its F1 is

\[
F_1=\frac{2TP}{2TP+FP+FN}=\frac{258}{259}\approx 99.61\%,
\]

yet its exact-set score is zero. There are 100 non-exact child answers, and 39
have exactly one membership error. High average membership accuracy and full
answer correctness measure different requirements.

Strict separation remains 165/222 in aggregate. That means all true products
outrank all false products for 165 queries; it does not mean zero is the right
threshold for those queries. The unchanged count also hides three separation
gains and three losses. These results do not prove a representational ceiling,
optimizer convergence or the success of a different threshold. No threshold
tuning or further training was performed after the failed gate.

The local refit took about 3.64 seconds. Parent and child head timings include
different warm-up conditions and exclude other pipeline work; their ratio is
not a serving speedup. The user-requested LM Studio backend shutdown occurred
before this run and its offline state was preserved. No protected-test query
was evaluated, and the checkpoint remains outside the standard serving contract.

Every other state tensor must remain byte-identical. However, changing the head
could change a guided decoder's choices even when decoder weights are frozen.
This experiment does not measure that separate effect.

## How the research record stays honest

The child is a new trained checkpoint with its own hash and objective identity.
Its lineage records the parent's 2,000 training steps separately from the 500
projection updates. It cannot masquerade as the old causal-training recipe to
pass the serving loader. Checkpoint loading for this screen validates the new
contract explicitly.

A checkpoint-file hash identifies the entire serialized artifact, including
weights, optimizer state and metadata. Per-tensor hashes identify individual
weight tensors. A recipe identifies how we updated them.
An evaluator identifies how we measured them. A Git commit identifies the
published source and documents. These are different parts of reproducibility.
The dedicated Luna Max publication task receives the completed, independently
audited evidence and the separate outcome decision, including a negative result.
Published hashes do not themselves provide a backup of locally stored weights.
