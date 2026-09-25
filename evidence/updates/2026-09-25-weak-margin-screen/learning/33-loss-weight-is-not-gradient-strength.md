# 33. A loss weight is not a gradient strength

The previous experiment failed badly; its follow-up diagnosis gave us a more
specific question. Can we retain useful membership scores while applying a
much weaker penalty to the hardest ranking error? The new experiment is
declared in the [weaker-margin plan](../experiments/2026-09-25-weak-margin-plan.md).
The screen is now complete and independently audited. Selected answers improve
slightly, but global ranking separation declines, so the unchanged gate rejects
this variant. The old failed experiment remains a failed experiment.

## Intuition: two teachers sharing the same parameters

Our model receives several forms of supervision. Token cross-entropy teaches
the next identifier; membership BCE teaches which entities belong to an answer;
the hardest-boundary margin focuses on the lowest-scoring true entity and
highest-scoring nonmember. They all influence some of the same parameters.

A coefficient is a volume knob on one teacher's gradient. Setting it to 0.1
does not mean that teacher contributes ten percent of the update. The raw
gradients can have very different scales because the objectives measure and
average different things. The optimizer then transforms their combined gradient.

Let the existing objective be L_c and the margin be L_m. At fixed parameters
theta and the same batch:

\[
L_\lambda=L_c+\lambda L_m,
\qquad g_\lambda=\nabla_\theta L_c+\lambda\nabla_\theta L_m.
\]

For a particular parameter block, its relative auxiliary strength is

\[
\rho_\lambda=
\frac{\|\lambda\nabla L_m\|_2}{\|\nabla L_c\|_2}.
\]

This ratio is undefined when the denominator is zero. It measures relative
gradient norm at one state and batch, not a fraction of the final AdamW step.
If both gradients are nonzero, their cosine tells us whether they align or
oppose each other. Positive rescaling changes the norm but leaves the cosine
unchanged at those same weights.

## Why reductions matter

The relation head emits Z with shape [B,1025]. Each row scores every entity.
The subject retains a score but is excluded from both loss masks.
Deduplicated supervised targets define the positive
mask P[B,1025]; the other eligible entities form N[B,1025]. The hardest-boundary
loss reduces each row to two scores, then averages B query losses:

\[
L_m=\frac1B\sum_q\max(0,1+\max_{j\in N_q}Z_{qj}
-\min_{i\in P_q}Z_{qi}).
\]

With unique extrema and an active hinge, the derivative at the hardest negative
is +1/B and at the weakest positive is -1/B. Other score entries receive zero
from this term. The implementation splits gradients across tied extrema.
Inactive hinges and exact ReLU-zero boundaries contribute zero gradient but
remain in the denominator B; we do not divide by the active-query count.
The symmetric BCE computes the positive mean and negative mean separately
within each query, weights each by one half, then averages all B queries.
These differing reductions help explain why equal-looking scalar coefficients
need not produce comparable parameter gradients. The mapping from score
gradients through the head to parameters also matters.

The previous gradient diagnosis measured shared embeddings E[2049,256] and relation projection
W[256,256]. Their gradient tensors have those same shapes. A flattened L2 norm
summarizes a block without pretending every coordinate behaves identically.
On W, CE and prompt BCE have no gradient, so the control gradient equals the
symmetric BCE gradient. On E, all three existing objectives contribute.

## What the 100-fold reduction does—and does not—predict

The fixed new coefficient is .001 instead of .1. At identical old diagnostic
weights and batches, the exact scaling relation is

\[
\rho_{.001}=.01\rho_{.1}.
\]

| Old diagnostic state/block | Mean ratio at .1 | Fixed-state ratio at .001 |
| --- | ---: | ---: |
| Initialization, W versus symmetric BCE | 6.207 | .06207 |
| Initialization, all E versus control sum | 1.062 | .01062 |
| Trained control, W versus symmetric BCE | 48.357 | .48357 |

These are means of three per-batch ratios. The last column is arithmetic on
old measurements, not a newly measured training trajectory. After even one
different update, parameters change and so can both raw gradients. AdamW's
moments, clipping and weight decay further separate a gradient coefficient
from an actual parameter-step proportion.

The value .001 is a declared hypothesis, not an optimum found by a sweep.
It might avoid the previous collapse but be too weak to improve rankings.
It might improve separation while the generated candidate sets remain incomplete.
It might simply fail again. The declared gate requires both better selected
answers and better ranking separation; neither measurement replaces the other.

## Why train another control?

We train a fresh control and treatment under the same source, environment,
seed and budget. Only the run names and margin coefficient differ. Comparing
against a convenient historical score would mix the intervention with possible
runtime variation. Same seeds align the intended stochastic setup, but do not
guarantee bitwise deterministic CUDA training.

Both runs finished at step 2000 before either generated-answer result was inspected.
There is no checkpoint shopping or new threshold. A single-seed pass requires
strict improvements in exact answers and separation, no F1 or group-exact
regression, valid predictions, authentic provenance and independent audit.
Failure stops this variant; passing only earns fixed additional-seed replication.

This is an adaptively chosen development experiment: earlier results motivated
the new coefficient. Documenting that history matters. Calling every new choice
an independent confirmation would hide how much validation guided the research.
The protected final test remains unused while these choices are being made.

## Reading a result

Keep three questions separate: did the implementation execute the stated
experiment; did the independent audit reproduce the saved measurements; did
the treatment pass the quality gate? The first two can pass while the third
fails. A useful research record preserves all three answers and both checkpoints.

## Result: better selected answers, worse global separation

The [audited result and owner decision](../experiments/2026-09-25-weak-margin-screen.json)
compare the fresh seed-1729 control with the coefficient-.001 treatment on the
same 222 validation queries. These are 444 arm-query observations, not 444
independent queries.

| Measurement | Fresh control | Weak-margin treatment |
| --- | ---: | ---: |
| Exact selected sets | 191/222 | 194/222 |
| Macro precision | .978083 | .986588 |
| Macro recall | .967390 | .971723 |
| Macro F1 | .967657 | .974821 |
| Queries with strict head separation | 168/222 | 164/222 |

There are 11 exact gains and 8 losses: the net improvement of three answers
does not mean every previous success survives. The group totals also describe
different aspects of the model:

| Group | Queries | Exact sets, control → treatment | Strict separation, control → treatment |
| --- | ---: | ---: | ---: |
| COLOR | 103 | 101 → 101 | 103 → 103 |
| Single-TYPE | 51 | 45 → 46 | 46 → 46 |
| Dual-TYPE | 68 | 45 → 47 | 19 → 15 |

The separation requirement is the only failed numerical gate. Exact count
improves, macro F1 increases and no group's exact count declines. Nevertheless,
the declared rule required a strict separation improvement too. We keep that
rule after seeing the result: the variant is rejected, and seeds 1730/1731 are
not authorized for replication. Neither checkpoint replaces the application
reference or changes inference defaults.

## Why these two measurements can disagree

Strict separation asks a demanding question about the entire eligible entity
universe. For each query, define

\[
\Delta_q=\min_{i\in P_q}Z_{qi}-\max_{j\in N_q}Z_{qj}.
\]

The query is strictly separated only if \(\Delta_q>0\). One nonmember scoring
above one true member is enough to fail, even if those entities never appear
together in a competing generated candidate. This is distinct from satisfying
the training margin of one: \(\Delta_q>0\) can hold while the hinge is active.

The current composition procedure answers a narrower question. It chooses
among four generated sets and six pair unions, scoring an eligible candidate S
with \(\sum_{i\in S}Z_{qi}\). An exact candidate can win within that restricted
pool even when individual scores are not globally separated. Conversely,
perfect separation cannot supply a true member omitted from every source path.
Training changes both the generated candidates and the head scores, so these
aggregate results alone do not isolate which component caused each gain or loss.

For a toy example, let truth be {a,b}, with scores a=2, b=-1 and c=0 for a
nonmember c. Global separation fails because c outranks b. If the available
candidates are only {a,b} and {c}, their additive scores are 1 and 0, so the
selector still returns the exact answer. This two-candidate illustration is
not a measured campaign case; the actual policy retains its ten fixed slots.

Dual-TYPE makes the distinction concrete: two more selected answers are exact,
while four fewer queries have globally separated head scores. We report both;
we do not remove the failed gate or invent a new selector to rescue the trial.

## What the loss log adds

The final step-2000 validation log retains each objective separately:

| Logged loss | Fresh control | Weak-margin treatment |
| --- | ---: | ---: |
| Token cross-entropy | .130531 | .129807 |
| Prompt membership BCE | .324739 | .305317 |
| Symmetric membership BCE | .008606 | .009272 |

The treatment's raw validation margin loss is 1.185196, with 68/222 active
hinges. The control does not log a margin objective because its coefficient is
zero. These logged losses describe the trainer's evaluation path; the final
answer and separation measurements use the separately declared inference path.
Loss values do not measure gradient strength or prove a training mechanism.

This run did not repeat the catastrophic answer-quality collapse observed at
coefficient .1. That is an observed outcome, not proof of why the trajectories
differed. The fixed-state ratios earlier in this lesson still come from the
previous training-query diagnosis; no gradients were newly measured in this
screen. The small single-seed improvement is development evidence, not an
accepted recipe, a reliable effect across seeds or protected-test performance.

## What was verified and retained

All 1,776 source paths are protocol-valid and terminated, and all selected
source evidence is eligible. The independent audit reproduces the saved set
metrics, 4,440 slot scores, selections and head diagnostics, and checks bound
checkpoint bytes and receipts. It authenticates the saved logits rather than
rerunning the neural forward pass; it does not validate checkpoint tensors by
deserializing them. The owner accepts this evidence and rejects the variant.

Both checkpoints remain in the [27-checkpoint register](../model-versions.md).
The historical failed screen, its reports and the protected-test boundary are
preserved. The new summary SHA256 is
`3d535d1e298497609169dda8791aef3920d3d8993afab9189df359644b8c7a93`;
the independent audit SHA256 is
`5cf9d5b556f06f068bcfda4709d737f351260f21144bc70ec42c305feef9caab`.
The portable result links the separate owner decision, which authorizes no
replication or promotion.
