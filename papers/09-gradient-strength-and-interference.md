# Gradient strength and interference after a failed margin intervention

A fixed diagnostic measured substantial margin gradients at initialization
and local opposition to the existing objectives at the rejected treatment's
final weights. Its independent saved-array audit passes. These measurements
describe five parameter states on 96 training queries; they do not establish
why AdamW produced the failed model or demonstrate a successful replacement.
No training, answer generation or model promotion occurred.

## The question and measurement boundary

The preceding margin screen reduced exact answers from 192/222 to 6/222.
Before choosing another intervention, the
[declared diagnostic](../evidence/updates/2026-09-25-gradient-diagnosis/experiments/2026-09-25-margin-gradient-diagnosis-plan.md)
fixed five states: regenerated common initialization, both arms at step 500,
and both final step-2000 checkpoints. Each state used the same train-split
slices [0:32], [800:832] and [1600:1632]. These are 15 snapshot/batch
observations, or 480 repeated query observations on 96 distinct training
queries. No validation or protected-test outcomes were measured.

The probe used full teacher-forced sequences, existing label masks, FP32
parameters, eval mode, no autocast and disabled TF32. The training recipe
already had zero dropout; FP32 execution still differs from its BF16 training.
It extracted four objective gradients without an optimizer or parameter update.
The [worked lesson](../evidence/updates/2026-09-25-gradient-diagnosis/learning/32-measuring-gradient-interference.md)
explains the tensor operations and a two-dimensional example.

## Compare directions as well as magnitudes

At identical weights and on one batch, define the control gradient and
weighted margin gradient:

\[
g_c=\nabla L_{CE}+\nabla L_{prompt}+\nabla L_{symmetric},
\qquad g_m=0.1\nabla L_{margin}.
\]

The norm ratio is rho=||g_m||/||g_c||; the cosine is
(g_c dot g_m)/(||g_c|| ||g_m||). A positive cosine indicates alignment and a
negative one opposition. If only the chosen parameter group moves, a small
ordinary gradient-descent step contributes approximately:

\[
\Delta L_c\approx-\eta\|g_c\|^2(1+\rho\cos\theta).
\]

Thus opposition alone does not establish an increase in control loss: its
strength matters. For g_c=[1,0] and g_m=[-2,1], the combined descent direction
is [1,-1], which initially increases that example's control loss. This is an
illustration, not a reconstructed PLM update.

The measured parameter blocks were shared token embeddings E[2049,256] and
the symmetric relation projection W[256,256]. CE and prompt BCE do not use W,
so its control gradient equals its symmetric-BCE gradient. The all-E comparison
instead includes all three control objectives. Product and dimension-row
subsets were also reported; those overlapping subsets cannot be summed as
independent contributions. Transformer-block gradients were not measured.

| State | W: weighted margin / symmetric BCE norm ratio | W: cosine | All E: weighted margin / control-sum norm ratio | All E: cosine |
| --- | ---: | ---: | ---: | ---: |
| Shared initialization | 6.207 | +0.507 | 1.062 | +0.057 |
| Control, step 500 | 5.799 | +0.180 | 3.445 | +0.188 |
| Treatment, step 500 | 5.086 | +0.060 | 3.026 | +0.022 |
| Control, step 2000 | 48.357 | +0.373 | 24.237 | +0.202 |
| Treatment, step 2000 | 1.606 | -0.306 | 1.419 | -0.292 |

Each entry is a mean of three defined per-batch ratios or cosines. It is not
a ratio of mean norms or a cosine of averaged gradients. Nor can multiplying
the two means recover the mean of batchwise products in the loss-change formula.
The [primary summary](../evidence/updates/2026-09-25-gradient-diagnosis/campaign/summary.json)
retains the individual observations and artifact identities.

Initially, the weighted margin on W was much larger than BCE but positively
aligned with it. By the final treatment state, both inspected blocks showed
negative mean cosines. This supports a state-dependent local-interference
finding, not a claim that the objectives conflicted throughout training.

At final control weights, the three W ratios were 18.84, 60.97 and 65.26.
Their mean of 48.357 cautions against assuming that adding coefficient 0.1 to
a trained control would be gentle. This was a hypothetical gradient calculation,
not a fine-tuning experiment. A large ratio can also reflect a small remaining
BCE gradient; it does not by itself quantify an absolute update or instability.

## Score-shrinking components do not determine the optimizer trajectory

Holding embeddings fixed, positive scaling W -> cW gives Z(cW)=cZ(W).
For weakest true score a and strongest negative score b, the unit-margin hinge
becomes max(0,1+c(b-a)). With overlapping rankings, decreasing c can lower this
penalty without correcting the ordering. Zero scores still pay a unit hinge.

At c=1, the implemented local derivative obeys:

\[
\langle\nabla_W L_{margin},W\rangle
=\operatorname{mean}_q\mathbf{1}_{\{\ell_q>0\}}(b_q-a_q).
\]

The identity passed for all 15 observations under the declared tolerances.
Inactive and exact ReLU-boundary rows contribute zero. This expression does
not extend unchanged across hinge crossings or define the all-tied c=0
subgradient as a one-sided scaling derivative.

At final treatment weights, mean dot products with W were +0.015187 for the
weighted margin and -0.013079 for symmetric BCE. Positive means the ordinary
negative-gradient direction has a shrinking radial component. The combined
component varied by batch:

| Slice start | Weighted margin dot W | BCE dot W | Combined dot W |
| --- | ---: | ---: | ---: |
| 0 | +0.01809445 | -0.01105281 | +0.00704164 |
| 800 | +0.01332012 | -0.01396093 | -0.00064081 |
| 1600 | +0.01414680 | -0.01422367 | -0.00007688 |

The margin pointed toward shrinking W in all three batches, but BCE offset it
enough to reverse two combined signs. Even these combined directions are not
AdamW updates: moment estimates, preconditioning, clipping and weight decay
were not replayed, and real training also changes E. Local interference and
radial components therefore do not establish the cause of the observed collapse.

## Evidence and limits

Fourteen primary synthetic tests and 22 auditor synthetic tests cover the
contracts. The runner attested unchanged full-model state hashes and empty
parameter `.grad` buffers across all observations. It regenerated both arm
initializations from the fixed seed and verified exact state equality. The
step-500 checkpoint hashes were newly pinned and checked; earlier seals of
final checkpoints did not establish those periodic identities.

The [independent audit](../evidence/updates/2026-09-25-gradient-diagnosis/campaign/independent-audit.json)
reconstructed saved-vector reductions and checked saved parameter blocks
across batches. It recomputed symmetric BCE and margin from saved logits/masks,
checked the radial identity and mask-aggregated saved unreduced token losses.
Prompt BCE remained an authenticated scalar. It did not independently prove
autograd, regenerate neural outputs or verify full-model immutability beyond
the source-bound runner attestations.

The [portable result](../evidence/updates/2026-09-25-gradient-diagnosis/experiments/2026-09-25-margin-gradient-diagnosis.json)
and [separate acceptance record](../evidence/updates/2026-09-25-gradient-diagnosis/campaign/acceptance.json)
distinguish acceptance of diagnostic evidence from a successful intervention.
The final-checkpoint inventory remains 25. The failed margin gate is unchanged;
no new coefficient, replication seed, answer-quality result, inference default,
serving benefit or model promotion follows from this diagnosis.
