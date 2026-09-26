# 46. Changing which mistakes drive learning

**Status: independently audited; quality gate failed.** We keep the same bilinear model,
the same starting parent and the same 2000-update budget. Only the training
objective changes. The aim is to find out whether paying more attention to each
query's hardest membership decisions improves complete answers.

## An average can hide an important mistake

Suppose an answer should contain ten products. Nine are confidently included,
but one is missed. The complete answer is still wrong. The balanced binary
cross-entropy objective gives every true product some gradient, averaging within
the true class; it separately averages the false class before combining them.
One difficult product can therefore contribute relatively little when its class
contains many easy products.

That does not make averaging a bug. Distributing the gradient across products
may make optimization more stable and improve generalization. We are testing a
tradeoff, not correcting a mathematical error in the old objective.

## What the new loss measures

For query q, let T_q contain its true targets and N_q contain false candidates,
excluding the subject from both. Using the scores from [Lesson 45](45-ranking-errors-and-zero-threshold-errors.md),
define

\[
a_q=\min_{i\in T_q}z_{qi},\qquad b_q=\max_{i\in N_q}z_{qi}.
\]

The **softplus** function is \(s(x)=\log(1+e^x)\). The implementation uses the
numerically stable library function rather than evaluating that expression
naively. Our replacement objective is

\[
L=\frac1Q\sum_{q=1}^Q\frac{s(-a_q)+s(b_q)}2.
\]

The weakest true score a_q should increase; the strongest false score b_q should
decrease. That aligns with our fixed output rule z>0. For unique extrema,

\[
\frac{\partial L_q}{\partial a_q}=-\frac12\sigma(-a_q),\qquad
\frac{\partial L_q}{\partial b_q}=\frac12\sigma(b_q).
\]

Gradient descent subtracts the gradient, so the negative derivative pushes a_q
up and the positive derivative pushes b_q down. The final batch mean adds a
factor1/Q. These are derivatives with respect to scores; AdamW updates the
shared model parameters, so interactions between queries still matter.

## Follow the tensors

The training score matrix is **Z[1637,1025]**. Its rows are training queries and
columns are product candidates. Boolean true/false masks have the same shape.
For the true minimum, masked-out entries become positive infinity; for the
false maximum, they become negative infinity. Both classes must be nonempty
before reduction. The extrema a and b have shape **[1637]**, and averaging the
1637 query losses produces one scalar.

Teacher labels have shape **[1637,L]**, with vocabulary token IDs and ignored
control/padding positions. The subject-exclusion vector has shape **[1637]**,
but contains column offsets: subject token ID minus1024. Confusing these two
ID spaces would exclude the wrong column or cause an indexing error.

For a unique minimum and maximum, only those two score entries receive a direct
gradient from this query's loss. At exact ties, `amin` and `amax` divide that
gradient across tied entries. We deliberately retain that behavior. The identity
of the hardest member can change from update to update.

The trainable tensor is still **A[2,256,256]**. There is one matrix for TYPE and
one for COLOR, symmetrized inside the scorer. The 93 original parameter/state
tensors remain unchanged. A gradient on one score updates A and can consequently
change many scores; two nonzero score gradients do not mean only two products'
future scores will change.

## A fair comparison requires a fresh start

Both recipes start from the same original parent, A=0 and fresh AdamW state.
We do not continue from the successful balanced-loss child, because that would
mix an objective change with another 2000 updates of prior training. The saved
balanced-loss output is the historical comparator; it is not a fresh control run.

The gate is declared before training: more than207 exact answers out of222,
macro F1 at least0.9997456353806234, and no drop below group exact counts
103 COLOR,50 single TYPE,54 dual TYPE. All execution and serialization checks
must pass too. Failing one requirement fails the quality screen.

We record initial and final ordinary BCE separately, under `no_grad`. It is a
diagnostic and cannot affect updates or stopping. Comparing the raw numerical
size of BCE with the new loss would be misleading: they average different things.

## Reasons this could fail

The earlier projection-only worst-member experiment failed its quality gate and
reduced strict separation relative to its mean-loss sibling. That result stays
part of the record. The bilinear model has more expressive interactions, but
that alone does not guarantee this objective will improve it.

Focusing on extremes may sacrifice broader fit. Shared parameters can make two
queries pull in conflicting directions. Switching which member is hardest can
make updates less smooth. Matching the learning rate and update count does not
make the two objectives equally easy to optimize.

The [frozen plan](../experiments/2026-09-25-bilinear-worst-boundary-plan.md)
allows one final validation endpoint and an independent evidence audit. No
threshold is tuned and no protected-test examples are evaluated. A successful
screen would justify a separately declared replication; it would not establish
oracle parity, serving gains or generalization by itself.

## What happened in the fixed run

The new objective did not pass the declared quality gate, despite gaining one
exact answer overall. These are the independently verified measurements:

| Measure | Balanced-BCE sibling | Worst-member sibling |
| --- | ---: | ---: |
| Exact sets, out of222 | 207 | 208 |
| Macro F1 | 0.9997456354 | 0.9996521512 |
| COLOR exact | 103 | 103 |
| Single TYPE exact | 50 | 49 |
| Dual TYPE exact | 54 | 56 |
| Extra member occurrences, FP | 8 | 5 |
| Missing member occurrences, FN | 10 | 15 |
| Strictly separated queries | 217 | 215 |

The new child fixes four queries and breaks three previously exact queries,
giving a net gain of one. Fewer false inclusions come with more missing members.
The F1 floor and single-TYPE group floor fail, so the quality result is rejected.
We keep the previous accepted child rather than redefine success after seeing
the result.

Why can exactness rise while F1 falls? Exactness counts each complete answer as
right or wrong. F1 also reflects how many members are right inside each answer.
A model can make one additional answer perfect while making its remaining wrong
answers worse. Here the total incorrect-member occurrences rose from18 to20,
even though the number of wrong queries fell from15 to14. Macro F1 is averaged
per query, so those total counts explain the direction only intuitively; they
are not themselves the formula for the reported macro F1.

The trained worst-member loss fell from0.44132092595100403 to
0.003092526225373149. Ordinary training BCE also fell from0.003822767175734043
to0.00016030404367484152. The balanced-BCE sibling's final ordinary BCE was
0.00003491543247946538. These results show that optimizing the declared training
loss succeeded while the validation tradeoff failed our acceptance rule. They
do not isolate whether learning rate, extreme-member switching, representation
or generalization explains that tradeoff.

All2000 updates completed, and the primary verified the original93 tensors
unchanged plus exact final model/optimizer reload. The separate audit confirmed
the saved evidence, including the failed quality gate. Primary coverage passed
144 CPU tests (32 new,112 reused); the independent auditor passed68 synthetic
tests. Both actual executions completed on their first attempt.

The [portable result](../experiments/2026-09-25-bilinear-worst-boundary.json)
records the separate evidence-acceptance decision. Its full loss trace is in a
[linked training artifact](../experiments/2026-09-25-bilinear-worst-boundary-training.json).
The derivative is registered as the34th final checkpoint, including its rejected
quality status. No inference rule or serving default changes. The next research
step must account for this tradeoff rather than assume harder-example weighting
alone is the solution.
