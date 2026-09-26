# 42. Learning cross-coordinate relations

**Status: completed; evidence accepted, fixed quality gate rejected.** The
[frozen plan](../experiments/2026-09-25-bilinear-residual-plan.md) tests a richer
membership head while keeping the original embeddings and decoder unchanged.
The previous feasibility diagnostic was inconclusive. We are testing a new
hypothesis, not acting on a proof that the old head lacks capacity.

## What the current head can express

Think of a Pokémon embedding as 256 learned coordinates. These coordinates are
not named attributes such as electric or yellow. They are numbers the training
process learned to use when relating opaque product identifiers.

The existing head compares corresponding coordinates:

\[
z_{sdi}=\frac1{16}\sum_{k=1}^{256}E_{s,k}r_{d,k}E_{i,k},
\qquad r_d=WU_d.
\]

The subject's coordinate 7 interacts with candidate coordinate 7, weighted by
the dimension's coefficient. It does not directly interact with candidate
coordinate 12. In matrix notation this is a **diagonal bilinear form**:

\[
z_{sdi}=\frac1{16}E_s^T\operatorname{diag}(r_d)E_i.
\]

"Bilinear" means the expression is linear in either embedding when the other
one and the relation matrix are fixed. The matrix between the two vectors
determines which coordinate pairs can interact.

## Add a correction instead of replacing the parent

We introduce a trainable matrix A for each of TYPE and COLOR. We use its
symmetric part M=(A+A^T)/2 and add its score to the existing score:

\[
z_{\text{new}}=z_{\text{parent}}+\frac{E_s^T M_d E_i}{16}.
\]

This is a **residual**: an additive correction to an existing prediction.
Each off-diagonal entry M[j,k] permits subject coordinate j to interact with
candidate coordinate k. It can also adjust same-coordinate interactions on
the diagonal. This expands the possible functions while retaining the parent.

A two-coordinate toy example makes the difference visible. Let the subject
be (1,0) and the candidate (0,1). Every diagonal matrix gives a score of zero
before scaling, regardless of its diagonal values. But

\[
M=\begin{pmatrix}0&1\\1&0\end{pmatrix}
\]

gives E_s^T M E_i=1. The extra interaction lets one learned direction match
another. This toy example illustrates the function family; it is not a
measurement on our actual Pokémon embeddings.

For SAME, reversing the subject and candidate should preserve the relation.
Symmetric M gives E_s^T M E_i=E_i^T M E_s in real arithmetic. FP32 reductions
can differ slightly when computed in reverse order, so algebraic symmetry is
distinct from bitwise numerical equality. The experiment fixes the computation
order and does not repair scores by averaging reversed outputs afterward.

## Follow the tensors

| Tensor | Shape | Meaning |
| --- | --- | --- |
| E | [1025,256] | Frozen normalized and scaled product embeddings |
| A | [2,256,256] | New trainable raw matrices; TYPE first, COLOR second |
| M | [2,256,256] | Symmetric matrices used in scoring |
| Transformed catalog | [2,1025,256] | Each embedding transformed by each matrix |
| Gathered query vectors | [B,256] | Transformed subject for each requested dimension |
| Residual scores | [B,1025] | Corrections for every candidate product |

The computation is approximately the following; the implemented version also
authenticates dimensions, shapes, dtypes and numerical settings:

```python
M = (A + A.transpose(-1, -2)) * 0.5
transformed = torch.stack([
    F.linear(E, M[0]),
    F.linear(E, M[1]),
])
query_vectors = transformed[dimension_index, subject_index]
residual = F.linear(query_vectors, E) / 16
scores = parent_scores + residual
```

`F.linear(x, weight)` computes x @ weight.T. The transpose is consistent with
the formula because M is symmetric. We transform the catalog once per update
and gather requested rows. That uses about 2 MiB for the transformed tensor,
instead of gathering a separate 256-by-256 matrix for each training query.
Gradients still flow through the transform and gather back to A.

Training B is 1,637; validation uses the historical batches of 8, ending with 6.
Catalog embeddings exist for every product in this transductive experiment,
but only training query labels determine the updates. Merely transforming an
embedding does not evaluate a protected query.

## Zero initialization can still learn

A starts at exactly zero. The residual is then zero, and the actual new scoring
path must reproduce every parent validation logit and the original full-training
loss exactly before training starts. We retain the parent's existing operation
order rather than replacing it with an algebraically equivalent matrix product.
That matters because algebraic equivalence does not guarantee FP32 replay.

Zero A does not imply zero gradient. For one score,

\[
\frac{\partial z}{\partial A}
=\frac{E_sE_i^T+E_iE_s^T}{32}.
\]

This expression does not contain A. If the embeddings and loss derivative
provide a learning signal, the optimizer can move immediately. The zero start
preserves initial predictions without disabling the new linear correction.

We use the same balanced membership BCE as the earlier mean-loss refit: each
query gives equal total weight to its positive and nonself-negative products.
The decoder, embeddings and original W stay frozen. Only A receives 500 fresh
full-batch AdamW updates, under the fixed recipe.

## Count what is actually adjustable

The stored parameter contains 2*256*256=131,072 numbers. A symmetric
256-by-256 matrix has 256*257/2 independent coefficients, so the two matrices
have at most 65,792 effective coefficients. Antisymmetric changes to A cancel
out, and dependencies in E can reduce observable freedom further.

The previous frozen-feature head had at most 512 effective coefficients across
the two dimension vectors. That is a large change in the available functions,
but also a change in optimizer geometry. AdamW on redundant full matrices is
not equivalent to AdamW on a packed triangle or on the original W. Using the
same learning rate and number of updates makes the comparison reproducible;
it does not isolate capacity as the sole possible cause of an outcome.

The loss remains convex in A in exact arithmetic with frozen features: scores
are affine in A and logistic penalties are convex. Finite optimization can
still fall short, and fitting training relations does not guarantee validation
generalization. Even a full bilinear matrix remains restricted by the frozen
embedding space; it cannot assign arbitrary values to all product pairs.

## What would count as progress

The quality gate remains unchanged: more than 201 exact validation sets out
of 222, macro F1 at least 0.9799255176742276, group exact counts at least
103 COLOR / 50 single-TYPE / 48 dual-TYPE, and all outputs serializable under
the existing protocol limits. We also require exact replay, unchanged parent
tensors, finite updates and exact checkpoint/optimizer reload.

The output rule remains all nonself products with score strictly above zero,
sorted by ID. No validation-tuned threshold, true answer count or fallback
rescues a failed result. This screen evaluates dense set predictions; it does
not generate EOS or establish serving performance.

The completed derivative contains the original 93 state tensors plus
the new A tensor. Its architecture, objective and checkpoint have separate
identities. An explicit experiment head and loader evaluate A; the inherited
decoder's ordinary `forward` does not use it. Standard serving remains unsupported
until a separately reviewed integration. Failed results will be registered and
published too. A passing single-seed screen would motivate replication; it would
not be automatic promotion or protected-test success.

## Observed result: much closer, still below the fixed gate

The [audited result](../experiments/2026-09-25-bilinear-residual-refit.json)
records 198 exact validation sets out of 222, compared with 112 for the parent's
dense head. All 93 original tensors remained byte-identical; only the new A
tensor changed. Zero-start replay, 500 updates and full checkpoint/optimizer
reload passed. There were 57 runner tests and 60 independent auditor tests
before the single real run; the actual saved-evidence audit passed afterward.

| Predictor | Exact sets / 222 | Macro F1 |
| --- | ---: | ---: |
| Original dense head | 112 | 0.990374 |
| Earlier W-only mean-BCE refit | 122 | 0.991855 |
| Earlier W-only worst-member refit | 128 | 0.992346 |
| New bilinear residual | **198** | **0.999406** |
| Accepted eight-branch selector | 201 | 0.979926 |

The parent was freshly replayed exactly. The earlier refits and strong selector
are authenticated historical comparisons, not newly trained controls.

The new head gains 86 exact answers over the dense parent and loses none.
Against the stronger selector, it gains 10 but loses 13. The group counts are
103/103 COLOR, 48/51 single-TYPE and 47/68 dual-TYPE. The gate requires more
than 201 total and group floors of 103, 50 and 48, so the exact and TYPE group
checks fail. F1, serialization and execution checks pass. Better average overlap
does not override the declared complete-answer requirement.

There are only 25 false-positive products and 11 false-negative products across
the validation queries, yet 24 complete answers are wrong. A descriptive check
of the saved predictions finds 16 queries with one error, five with two, two
with three and one with four. Large, nearly correct sets explain how 99.94%
macro F1 can coexist with about 89.2% exact sets. No output was repaired using
these labels; these counts explain the result after the fixed evaluation.

Strict separation reaches 211 queries: in those queries every true member
scores above every nonmember. Of those, 13 still fail the fixed zero-threshold
rule. This is a distinction between **ordering** and **decision boundary**.
A per-query separating threshold exists for those saved rankings, but finding
it with the true labels is an oracle diagnostic, not an available predictor.
We did not search for thresholds, use the true answer count or change the rule.

Full-training balanced BCE falls from 0.0038227672 to 0.0002471808, a 93.53%
reduction. Training loss and validation set accuracy measure different things.
The result demonstrates improvement for this particular intervention and recipe;
it does not prove that capacity alone caused it, that AdamW reached an optimum,
or that the model has reached oracle parity.

The child is registered as final checkpoint 30. Its checkpoint SHA256 begins
`1f6e9a59`; the full identity, parent lineage and availability limits are in the
[model register](../model-versions.md). All 30 final checkpoint files were freshly
hashed; older checkpoint payloads and configs were not all revalidated. Weights
remain local, and the child is not promoted to ordinary serving.

The local fit took about 1.48 seconds and the campaign about 8.94 seconds. These
are descriptive timings. This implementation caches the frozen parent scores
and uses a different gradient computation from W-only fitting, so they do not
establish a controlled training-speed or serving-throughput improvement.

The declared 500-update run ended here. Further optimization, replication or
inference changes require a new declared experiment. This failed gate is retained
alongside the substantial dense-head improvement; the protected test remains unused.
