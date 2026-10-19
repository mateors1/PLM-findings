# 51. More training, or different features?

Status: **all three seeds and the aggregate independently audited; fixed improvement gate passed**. The protocol is
fixed in [the campaign plan](../experiments/2026-09-26-bilinear-budget32000-plan.md).

The affine experiment gave us a useful negative result: adding linear offsets
lowered training loss slightly, but all 222 validation answers stayed the same.
The model did not earn promotion. This leaves two plausible explanations for
the remaining mistakes: the current scorer needs more optimization, or its
features need to change. These are hypotheses, not diagnoses.

## Why test training time first?

Increasing the bilinear budget from 2000 to 8000 updates previously improved
exact answers by 9, 8 and 6 across three parent seeds. Its training loss was
still declining near the endpoint. That makes one further fixed budget a
motivated test with relatively few changed variables.

The new campaign uses 32000 updates for each of seeds 1729, 1730 and 1731.
Each starts from its original parent, zero residual weights and a fresh
optimizer. We do not continue from the saved 8000 child. Restarting keeps the
initialization procedure matched; optimizer state is part of that procedure.
Adam tracks moving averages of gradients and squared gradients, so resetting
the optimizer while keeping trained weights would be a different treatment.

The scorer remains

\[
z_{s,i,d}=z^{\mathrm{parent}}_{s,i,d}
 +\frac{E_s^\top\operatorname{sym}(A_d)E_i}{16},
\qquad \operatorname{sym}(A_d)=\frac{A_d+A_d^\top}{2}.
\]

Here s is a subject, i a candidate product and d TYPE or COLOR. The frozen
feature matrix E is [1025,256]. The sole trained tensor A is [2,256,256].
The complete training score matrix is [1637,1025]: one row per training query,
one column per candidate. All 93 original model tensors remain unchanged.

An **update** means one optimizer step. Because this campaign uses the whole
training partition per update, every update sees all 1637 training queries.
The word epoch is less useful here than explicitly stating that full-batch
contract. More updates increase offline training work; they do not add matrix
multiplications to the eventual scoring formula.

## Loss, correct sets and margins answer different questions

We select candidate i when its score z is strictly positive. Let y be 1 for a
true member and 0 otherwise. Its signed margin is

\[
m=(2y-1)z.
\]

For a true member, m=z: positive scores are correct. For a nonmember, m=-z:
negative scores are correct. The smallest margin in a row describes its weakest
membership decision. Exclude the subject's own column from these calculations.

For example, consider a hypothetical query with targets [1,0,1]:

| Scores | Selected mask | Exact set? | Minimum margin |
| --- | --- | --- | ---: |
| [0.2,-0.3,-0.1] | [1,0,0] | No | -0.1 |
| [0.2,-0.3,0.1] | [1,0,1] | Yes | 0.1 |
| [0.8,-1.2,0.7] | [1,0,1] | Yes | 0.7 |

The second-to-third change improves confidence without changing the answer.
Similarly, BCE can decrease after every training set is already correct.
Zero needs care: a nonmember at z=0 is correctly excluded, whereas a member
at z=0 is missed. Thus minimum margin >0 is sufficient for an exact set, but
not necessary under the strict-positive selection rule.

We will save training scores after updates 8000 and 32000 and independently
recompute exactness, false positives/negatives and margins. The former is
observed before update 8001, which is the same parameter state. These are
training diagnostics only: they cannot change the loss, stopping rule or
acceptance gate. There is no intermediate fitted validation evaluation.

## Why keep all three seeds?

Different parent seeds produce different frozen features. A longer budget that
helps only one parent is weaker evidence than a consistent improvement. Each
new child must strictly beat its own control's exact count—216, 220 or 214—and
preserve its F1 and group floors. Every seed must pass. We run the remaining
declared seeds even if a completed seed fails quality, but stop on a broken
execution or invariant.

All runs reuse the same 222 validation queries. Three times 222 gives 666
model-query observations, not 666 independent examples. These parents and
queries have already informed development. A successful campaign would remain
development evidence; the protected partition stays unopened.

## Why not just learn a linear feature projection?

Suppose we transform every feature with the same matrix T before a fully
expressive symmetric bilinear scorer. In row-vector notation,

\[
(E_sT)A_d(E_iT)^\top=E_s(TA_dT^\top)E_i^\top.
\]

The original scorer can already represent the resulting symmetric matrix
\(TA_dT^\top\). Such a projection can change optimization or regularization,
but does not expand this score family's expressive power. A meaningful future
feature experiment could instead learn entity-specific offsets
\(\Delta E:[1025,256]\), or a nonlinear map. That introduces additional choices
and a different optimization problem, so it belongs in a separate matched
experiment. Neither option is implemented or adopted by this plan.

The new campaign's weights, evaluator and publication will receive distinct
identities. The earlier affine result remains archived evidence. No amount of
training-set fit alone establishes oracle parity, generalization or a serving
advantage.

## What the preparation checks prove

Before using the GPU, we test the training pipeline on synthetic data and
freeze its code, configuration and test receipts. A **receipt** records what
ran, its result and the hashes of its dependencies. The shared readiness
manifest binds the runner and auditor to the same 22 files. A passing synthetic
test verifies that the pipeline behaves as intended in those cases; it does
not predict whether longer training will improve this model.

We caught a provenance bug during preparation: the original model identifies
its configuration using compact JSON, while newer reports use indented JSON.
The settings can be semantically identical while the byte hashes differ.
Identity checking must reproduce the serialization used when the original
checkpoint was created. We preserve that original identity and use the newer
format only where its own contract calls for it.

The runner passed 225 CPU tests; the auditor passed 100; the aggregator passed
45. A separate comparison checked training margins and selections on 300
synthetic score vectors of shape `[1025]`, including signed zero and ties.
All three real metadata preflights then passed without importing Torch. These
are implementation/admission results, not a completed training experiment.

## What the first completed runs tell us

| Parent seed | Exact at 8000 | Reported exact at 32000 | Current evidence status |
| --- | ---: | ---: | --- |
| 1729 | 216/222 | 218/222 | Independently audited; two gains, no losses |
| 1730 | 220/222 | 222/222 | Independently audited with evidence repair v2; two gains, no losses |
| 1731 | 214/222 | 220/222 | Independently audited continuation; six gains, no losses |

For seed 1729, the new diagnostic captures show **1637/1637 training queries
already exact at step 8000**. At step 32000 they remain exact, while the weakest
signed margin increases from 6.16 to 8.92. Final training BCE is approximately
`7.57e-8`. This case supports a precise interpretation: additional optimization
strengthened correct training decisions and corrected two validation sets.
It was not repairing training-set classification errors at those endpoints.
Four validation sets remain imperfect; perfect training fit is not oracle parity.

Why does training continue after the sets are correct? For one unweighted
binary cross-entropy term,

\[
\ell(z,y)=-y\log\sigma(z)-(1-y)\log(1-\sigma(z)),
\qquad \frac{\partial\ell}{\partial z}=\sigma(z)-y.
\]

A correct positive score still has a negative derivative when y=1, so gradient
descent pushes it upward. A correct negative score has a positive derivative
when y=0, pushing it downward. The balanced objective scales these contributions
by positive class weights; the direction remains the same. The `[1637,1025]`
score matrix therefore still supplies gradients to A after every training set
is correct. Larger margins mean greater distance from the zero decision
boundary; they do not by themselves establish calibrated probabilities or
correct unseen answers.

The second audit stopped on bookkeeping: the terminal receipt represented one
timestamp in local time while the preceding receipt used UTC. The instants are
exactly equal, but the auditor compared that embedded field as literal text.
We preserve the failed audit and use a
[separate evidence revision](../experiments/2026-09-26-bilinear-budget32000-evidence-repair-v2.md)
to repair that comparison. The separate v2 audit has now passed against the
saved checkpoint, optimizer, trace, metrics and original evidence. Its SHA256 is
`83d48edb5e78ee66eeab4c9e87970b882f8d0783c5829a07b8b9cd56e5ca728d`;
the original failed audit remains unchanged. All 222 validation sets are exact,
including all 68 dual-TYPE cases, with two gains and no losses versus its control.

This establishes exact agreement on this seed's fixed validation workload. It
does not establish correctness on unseen queries, robustness across seeds or a
serving advantage. Seed 1729 still misses four sets and seed 1731 misses two.
The distinction matters: we accepted the evidence only after repairing and
rerunning the verifier, without changing the checkpoint or the quality gate.

## Completed campaign: what improved, and what did not

The final seed completed all 32000 updates and passed its independent audit.
The aggregate audit also passed, followed by the separate owner decision at
`runs/learning/bilinear-budget32000-v1/decision-v2.json`. All three seeds improve
over their own controls; pooling does not hide a failed seed.

Across the same 222 validation queries evaluated under three parent seeds,
exact outcomes rise from 650/666 to 660/666: ten gains and no losses. These are
666 seed-query outcomes, not 666 independent examples. Remaining nonexact
outcomes drop from 16 to 6; all six involve dual-TYPE subjects. COLOR and
single-TYPE remain exact for every seed. The measured aggregate macro F1 is
0.9999637956903659, with seven missed entities and one extra entity in total.

This separates two questions: **can this training budget improve the existing
scorer?** Yes, in all three declared comparisons. **Have we built a generally
oracle-equivalent serving model?** That is not established: two seeds retain
errors, the protected partition remains unopened, and this scorer is still
outside ordinary serving. We accept the positive experiment evidence without
promoting the model. The next useful step is to diagnose those saved residual
errors before declaring another intervention, rather than choosing a threshold
after looking at validation labels.

## Ordering scores versus choosing a cutoff

The audited aggregate contains another useful distinction: 665/666 outcomes
have strict separation between true and false candidates, but only 660/666
produce the exact set at the fixed cutoff zero. Thus five of the six remaining
errors have the correct ordering across the membership boundary, while one
does not. This follows from the saved aggregate counts; it is not a new fit.

For a given query, let `p_min` be the lowest true-member score and `n_max` the
highest nonmember score, excluding the subject. Selecting scores greater than
a threshold `tau` can recover the exact set only when

\[
 n_{\max}\leq\tau<p_{\min}.
\]

Strict separation means this interval exists. The current rule fixes `tau=0`,
which can fall outside that interval even when the ordering is right. For the
one nonseparable outcome, changing only a scalar cutoff cannot recover the set.
However, choosing these intervals using validation labels would be an oracle
diagnostic, not a deployable solution. Any future calibration method must learn
from training data under a separately declared comparison; these observations
do not authorize replacing the fixed evaluator or claiming corrected answers.
