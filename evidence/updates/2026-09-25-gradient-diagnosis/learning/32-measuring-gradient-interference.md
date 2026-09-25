# 32. When two training objectives pull in different directions

**Status:** completed; the independent saved-array audit passes. This follows the
[failed margin screen](31-hardest-boundary-margin.md). We inspected existing
weights without training another model or changing the failed experiment's gate.
The [fixed plan](../experiments/2026-09-25-margin-gradient-diagnosis-plan.md)
defines the snapshots, examples, arithmetic and evidence boundaries.

## A loss value is not an update direction

A gradient tells us how a small parameter change affects a loss. If theta is
a vector of weights and g is the gradient, ordinary gradient descent proposes
`theta_new = theta - learning_rate * g`.

Two losses can prefer different changes to the same embeddings. Improving one
may slow or reverse progress on the other. A coefficient such as 0.1 rescales a
gradient; it does not tell us how large that gradient was before rescaling.

In our fixed recipe, define

\[
g_c=\nabla L_{CE}+\nabla L_{prompt}+\nabla L_{symmetric},\qquad
g_m=0.1\nabla L_{margin}.
\]

The subscript c means the control objective. These vectors are evaluated at
the same weights and on the same batch. Comparing a control gradient at one
checkpoint with a margin gradient at another would answer a different question.

For a chosen parameter group, we inspect its gradient norm, which measures
length, and cosine, which measures alignment:

\[
\rho=\frac{\|g_m\|_2}{\|g_c\|_2},\qquad
\cos\theta=\frac{g_c\cdot g_m}{\|g_c\|_2\|g_m\|_2}.
\]

Cosine +1 means parallel; 0 means orthogonal; -1 means directly opposed. A
negative cosine is a local conflict, but does not alone mean the combined
update hurts the control objective. Magnitudes matter too.

If only the chosen parameter group moves, an ordinary gradient step using both
gives the following first-order contribution to the control loss:

\[
\Delta L_c\approx-\eta\,g_c\cdot(g_c+g_m)
=-\eta\|g_c\|_2^2(1+\rho\cos\theta).
\]

For illustration, take g_c=[1,0] and g_m=[-2,1]. Their sum is [-1,1], so
descent moves in the direction [1,-1]. Its dot product with g_c is positive:
this movement initially increases the control loss. This two-dimensional
example is not a measured PLM update.

If either vector has zero norm, its cosine is undefined. We record that fact
instead of inventing a cosine of zero. We also distinguish a parameter that
does not participate in a loss from one that participates but has zero derivative.

## Which tensors are involved?

We measured gradients on two existing parameter tensors:

| Tensor | Shape | Role |
| --- | --- | --- |
| Shared token embeddings E | [2049,256] | Input embeddings and tied output scoring; product rows and dimension rows also feed the symmetric head. |
| Symmetric relation projection W | [256,256] | Maps a dimension embedding to the head's relation vector. |

Each raw loss yields one gradient tensor of the same shape as each parameter
it uses. We save these FP32 arrays before reducing them to a few statistics.
The analysis also reports product embedding rows [1025,256] and the dimension
rows present in each batch. These are subsets of E, not extra parameters.

Token CE and prompt BCE do not use W, so their W derivatives should be unused.
For this group, the control-sum gradient must equal the symmetric-BCE gradient.
This gives us a structural check before interpreting any numerical result.

Overlapping embedding subsets must not be summed as separate contributions.
The diagnostic does not inspect all transformer-block gradients. It studies
the weights shared with the membership head where the failed treatment acts.

## A route to smaller scores without better ordering

With embeddings held fixed, the symmetric head is linear in W. For positive c,
scaling W by c scales every membership logit by c:

\[
Z(cW)=cZ(W).
\]

For a query, let a be its lowest true-member score and b its highest nonmember
score. Its margin penalty after scaling is `max(0, 1 + c*(b-a))`.
If the current ranking overlaps, b-a is positive. Reducing c then reduces
the penalty locally while preserving the incorrect ordering. At c=0, the
scores all tie and the unit-margin penalty is 1; collapse is not a successful
margin solution. This identifies an available direction, not what the optimizer
necessarily did during training.

The local derivative at c=1 has two equivalent expressions:

\[
\langle\nabla_W L_{margin},W\rangle
=\frac{1}{B}\sum_q \mathbf{1}_{\{\ell_q>0\}}(b_q-a_q).
\]

One side uses the saved parameter gradient; the other uses the saved scores
and true/false masks. Agreement is a useful check of both the calculation and
our interpretation. Inactive queries and exact ReLU-boundary queries contribute
zero under the implemented convention. We do not extrapolate this derivative
across changes in which hinges are active or confuse the all-tied derivative
at c=0 with the one-sided scaling direction.

## What this diagnostic can establish

We inspected a regenerated shared initialization, then step 500 and final
step 2000 from both previously trained arms. Independent seeded builds of both
configs produced identical initial weights. The periodic checkpoint files
received explicit new hash verification; their identity was
not implied by the earlier final-checkpoint inventory.

Every state uses the same three train-split slices: [0:32], [800:832] and
[1600:1632]. That is 96 training-query identities observed at five states,
not 480 independent queries. No validation or protected-test predictions are
measured. Eval mode is fixed and autocast is disabled; the recipe already had
zero dropout. FP32 differs from original BF16 training, so this is deliberately
not an exact replay of training numerics.

`torch.autograd.grad` obtains derivatives without filling parameter `.grad`
buffers or taking an optimizer step. We hash parameters before and after and
preserve each batch separately. Averaging three batch cosines is not the same
operation as averaging gradients and then taking a cosine; our reports name
which quantity they use.

Finally, AdamW uses moment estimates, preconditioning and weight decay; the
training loop also clips gradients. A radial shrinking component in a raw
gradient does not prove an actual AdamW norm decrease. A saved-vector audit can
rebuild norms, cosines and radial arithmetic without independently proving
autograd or reconstructing the historical optimization trajectory.

## What we measured: strength and alignment change with the state

The following entries are **means of three separately calculated batch ratios
or cosines**, all with three defined values. They are not ratios of mean norms,
cosines of averaged gradients, or measurements on one pooled batch of 96.

| Parameter state | W: weighted margin / symmetric BCE norm ratio | W: their cosine | All E: weighted margin / control-sum norm ratio | All E: their cosine |
| --- | ---: | ---: | ---: | ---: |
| Shared initialization | 6.207 | +0.507 | 1.062 | +0.057 |
| Control, step 500 | 5.799 | +0.180 | 3.445 | +0.188 |
| Treatment, step 500 | 5.086 | +0.060 | 3.026 | +0.022 |
| Control, step 2000 | 48.357 | +0.373 | 24.237 | +0.202 |
| Treatment, step 2000 | 1.606 | -0.306 | 1.419 | -0.292 |

Here “weighted margin” always means 0.1 times the raw margin gradient. On W,
the symmetric BCE is the entire control gradient, because CE and prompt BCE
do not use this projection. On all E, the comparison uses the sum of all three
control-objective gradients. These denominators answer different questions.

At initialization, the weighted margin gradient on W was already about six
times the symmetric-BCE gradient, but the two were positively aligned. A large
gradient was not automatically an opposing gradient. On the shared embeddings,
it was about as large as the entire control sum and nearly orthogonal to it.
At treatment step 500, it was about three times the control-sum gradient on E,
again with a near-zero mean cosine.

At the final treatment state, the mean cosines were negative on both parameter
blocks. That is evidence of local interference on these batches and weights.
It is not evidence that the objectives opposed each other throughout training,
nor does a negative cosine alone establish that a combined update raises the
control loss. The earlier first-order formula still requires magnitudes and
directions from the same batch. Multiplying the table's mean ratio by its mean
cosine would not reproduce the mean of those batchwise products.

The final control gives another useful warning. Its three W norm ratios were
18.84, 60.97 and 65.26, averaging 48.357. This measures a hypothetical margin
gradient at existing control weights; no margin fine-tuning was performed.
Starting from that trained control would not make coefficient 0.1 inherently
gentle relative to its remaining BCE gradient. A ratio can also become large
because the denominator is small, so it is not itself an absolute update-size
or stability measurement.

## The radial result is mixed once objectives are combined

At final treatment weights, the mean projection dot products were:

\[
\operatorname{mean}_{batch}\langle 0.1\nabla_W L_{margin},W\rangle
=+0.015187,
\qquad
\operatorname{mean}_{batch}\langle\nabla_W L_{symmetric},W\rangle
=-0.013079.
\]

A positive dot product means the ordinary negative-gradient direction has a
component toward smaller W norm; a negative one points toward larger norm.
The weighted margin contribution was positive in all three final-treatment
batches. The combined direction was not:

| Training-slice start | Weighted margin dot W | Symmetric BCE dot W | Their sum dot W |
| --- | ---: | ---: | ---: |
| 0 | +0.01809445 | -0.01105281 | +0.00704164 |
| 800 | +0.01332012 | -0.01396093 | -0.00064081 |
| 1600 | +0.01414680 | -0.01422367 | -0.00007688 |

Thus the margin has a local shrinking component here, while the control
objective partly or fully offsets it. Claiming that the combined gradient
always shrinks the projection would contradict two of these three batches.
Claiming that AdamW actually took these directions would go further still:
this probe did not reconstruct its moments, preconditioning or clipped updates.
It also holds embeddings fixed for the scaling argument; real training changes
both E and W.

The radial identity passed in all 15 snapshot/batch observations using the
declared tolerances. This links the observed gradient component to the head's
linear scaling property. It does not establish a complete causal explanation
for the failed model's compressed logits.

## What was independently checked

Fourteen primary synthetic tests and 22 auditor synthetic tests cover the
measurement and reduction contracts. The runner saved FP32 parameters and four
objective gradients, logits, labels, masks and scalar losses for every batch.
It reported unchanged full-model state hashes and empty `.grad` fields across
all 15 observations, with no optimizer construction or parameter update.

The separate auditor independently rebuilt the saved-vector reductions and
checked saved parameter blocks across batches. It also reconstructed symmetric
BCE and margin from logits/masks, checked the radial identity, and aggregated
saved unreduced token losses. The full-model immutability and checkpoint-payload
loading checks remain runner attestations bound to their executing source.
The auditor did not rerun the neural network or independently prove autograd;
the prompt-BCE scalar is authenticated rather than regenerated.

The [portable result](../experiments/2026-09-25-margin-gradient-diagnosis.json)
links the evidence and separate diagnostic decision. The local evidence is
`runs/learning/margin-gradient-diagnosis-v1/`. Summary
SHA256 is `5a0927acf533a2a170d829a781e52e45686961e909e5a7f697ad193de3dcc607`;
independent audit SHA256 is
`c2fe33de149f1ade6d28475b952833a304739c9fd32aafd0064b5ea080b0cbd5`.

This is a new evaluator result on existing weights, not a new model family.
The final-checkpoint inventory stays at 25. There is no new answer-quality
measurement, validation or protected-test result, optimizer replay or accepted
training intervention. The measurements can inform a separately declared next
experiment; they do not retroactively make the rejected margin recipe successful.
