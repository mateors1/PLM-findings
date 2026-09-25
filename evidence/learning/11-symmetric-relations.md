# Lesson 11: a useful inductive bias for relational learning

**2026-09-24.** Earlier experiments located a weak opening decision and persistent
TYPE errors. Selecting an earlier checkpoint did not improve our retained model.
This iteration adds a direct symmetric relation objective to the existing
prompt-set decoder. The [plan](../experiments/2026-09-24-symmetric-relation-plan.md)
fixes the architecture, coefficients, seed, 2,000-step budget and final-checkpoint
comparison before the run.

**Result:** this seed improves substantially, including ordinary unassisted
generation. The next step is replication; service defaults are unchanged.

## What changed in the measured answers

| Final checkpoint, seed 1729 | Prompt-set anchor | Plus symmetric supervision |
| --- | ---: | ---: |
| Parameters | 6,624,256 | 6,689,792 |
| Raw generated F1 | 62.90% | 83.41% |
| Processed generated F1 | 63.19% | 83.80% |
| Processed exact answers | 105/222 | 143/222 |
| TYPE processed exact | 32/119 | 52/119 |
| COLOR processed exact | 73/103 | 91/103 |
| Valid EOS termination | 222/222 | 222/222 |

The [comparison report](../experiments/2026-09-24-symmetric-relations.json)
verifies identical normalized training configurations except for the new
coefficient and run name, as well as checkpoint/data/split identity and replayed
metrics. The new run finished all 2,000 steps in 66.87 seconds; the archived
anchor took 61.80 seconds. These are observed run durations, not a repeated
paired throughput measurement. New candidate generation used the cached path;
the anchor's cached/uncached response parity was verified in prior iterations.

Raw exact-set accuracy remains zero. The same deterministic self-exclusion and
deduplication policy is applied to both models to obtain the processed results.
We did not provide a correct first ID or use symmetric scores during generation.
This is a 38-answer gain in one seed, not cross-seed robustness or oracle parity.

## The helper is good, but its scores are a different output

The [classifier diagnostic](../experiments/2026-09-24-symmetric-classifiers.json)
uses a fixed zero-logit threshold on the same validation prompts:

| Prediction path in the new model | Raw set F1 | F1 after subject exclusion | Exact sets after subject exclusion |
| --- | ---: | ---: | ---: |
| Direct symmetric classifier | 98.64% | 99.08% | 114/222 |
| Existing prompt-set classifier | 86.67% | 87.05% | 120/222 |
| Autoregressive decoder | 83.41% | 83.80% | 143/222 |

Classifier rows describe sets, not emitted protocol sequences or EOS behavior.
The decoder row uses the full existing post-policy. The symmetric classifier
predicts the subject on all queries, consistent with leaving its diagonal
unsupervised. Removing the subject requires no learned graph information.

Why does the highest F1 have fewer exact answers? F1 tolerates small numbers of
wrong or missing products in long sets; exact accuracy rejects even one error.
The classifier and decoder distribute errors differently. These aggregate metrics
do not let us substitute classifier quality for generation quality.

## Use a known property without giving the answer away

If A shares a type with B, B shares a type with A. The same holds for color.
This **symmetry** is built into a small auxiliary scorer. It still has to learn
which particular pairs belong together from training labels.

An **inductive bias** is a restriction or preference built into learning. Here
the scorer cannot assign independent values to `(A, TYPE, B)` and `(B, TYPE, A)`.
Learning about either direction changes the shared score for both directions.
That may make training labels more useful for held-out queries.

Symmetry does not imply transitivity. A product with fire/grass types may share
fire with a fire/water product, which shares water with a water/rock product;
the first and third need not share any type. The new score imposes symmetry,
not a rule that all connected products belong in one answer set. Ordered answer
generation is also distinct from symmetric membership.

The diagonal bilinear form is inspired by
[Yang et al.'s relation embedding work](https://arxiv.org/abs/1412.6575).
Our normalized shared embeddings, dimension-token projection and auxiliary use
inside this decoder are project-specific choices. This is not a reproduction
of that paper's experiments or evidence of its reported performance here.

## The equation and tensors

Let D=256, N=1025 products, and B be the batch size. The model already has an
embedding table E. We reuse its product rows and its dimension-token rows:

```text
normalized product vector u_i = sqrt(D) * E_i / max(norm(E_i), epsilon)
normalized dimension vector a_d = sqrt(D) * E_d / max(norm(E_d), epsilon)
relation vector r_d = W @ a_d

z(s,d,t) = sum_k u_s[k] * r_d[k] * u_t[k] / sqrt(D)
```

Swapping `s` and `t` only swaps two scalar factors inside each product. The
mathematical score is therefore symmetric; numerical tests allow floating-point
rounding. Relation coordinates can have either sign. We normalize vectors so
their initial norm does not make the three-way product nearly vanish. This is
an explicit modeling choice, not a guarantee of good calibration.

```text
Product embeddings U:                 [1025, 256]
Batch subject embeddings:            [B, 256]
Batch dimension embeddings:          [B, 256]
New projection W:                    [256, 256]
Batch relation vectors R:            [B, 256]
Subject vectors * R:                 [B, 256]       elementwise product
Scores (subject * R) @ U.T / sqrt(D): [B, 1025]
```

Only W is new: 65,536 parameters. Its initialization happens after the existing
decoder and prompt-set head, preserving identical common initial parameters for
the same seed. The small scorer computes in FP32 even during BF16 training.

## Where the labels and gradients come from

For each training query, products already in its answer are positives; the
other eligible products are negatives. The symmetric loss excludes the subject
column: returning the subject is handled by the existing deterministic SAME
rule, so this scorer need not learn a negative diagonal.

```text
L_sym(q) = 0.5 * mean(softplus(-z_j), j in positive targets)
         + 0.5 * mean(softplus( z_j), j in negative products except subject)

L_total = L_token + L_prompt_set + L_sym
```

Duplicate target IDs count once. Both classes must be nonempty. The earlier
prompt-set loss keeps its original behavior; only the new symmetric loss masks
the subject. All three losses are logged separately because summing more terms
changes the meaning of total loss.

The new score reads only subject/dimension IDs and learned embedding weights.
It never reads future target tokens as prediction inputs. Target labels influence
gradients through the loss, which is supervision. No additional reverse-query
rows, hidden attribute names or held-out labels enter training. A product may
already appear as a target in training even when one of its subject queries is
held out; this remains the same transductive split, not unseen-product evaluation.

## The crucial transfer test

The new head does not select, filter, reorder or inject generated IDs. Cached
autoregressive decoding skips it entirely. Its route to better answers is:

```text
training relation loss
    -> gradients into shared product/dimension embeddings
    -> changed representations used by the ordinary decoder
    -> possibly better unassisted generated answers
```

We therefore measure the head's classifier scores and the decoder's full answers
separately. A near-perfect classifier can coexist with imperfect generation.
The zero-logit diagnostic threshold is fixed, not tuned on final test. Because
the loss balances positives and negatives, sigmoid scores are not automatically
calibrated probabilities under the original class prevalence.

## Verification and limits

Tests check score symmetry, exclusion of self-gradients, correct positive/negative
gradient directions, finite FP32 auxiliary computation under CPU/CUDA autocast,
invariance to future tokens, identical common initialization/token logits,
cached-forward parity, invalid inputs, exact training resume and toy overfit with
actual generated IDs. The full suite passed 194 tests before the real run.

The real candidate retains the data, split, seed, batch size and optimizer-step
budget of the prompt-set anchor, with one additional coefficient at 1. The same
number of optimizer updates is not the same computational cost or gradient
objective. Source/config/checkpoint identities preserve that distinction.
Validation evidence can justify replication, not an oracle-quality or serving
advantage claim. Final test remains reserved.

The next experiment repeats this candidate at the already used seeds 1730 and
1731 while retaining the split and final-checkpoint budget. If the improvement
repeats, examine the remaining TYPE continuation errors and the gap between
membership scoring and ordered generation. Do not assume that directly decoding
thresholded scores would preserve the protocol's target order or EOS behavior.

Follow-up: [Lesson 12](12-symmetry-replication.md) records the replication,
refreshed controls and termination failures that blocked serving promotion.
