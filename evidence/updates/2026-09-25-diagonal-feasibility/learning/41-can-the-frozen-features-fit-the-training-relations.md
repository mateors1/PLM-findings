# 41. Can the frozen features fit the training relations?

**Status: completed, independently audited, inconclusive for both dimensions.** The
[declared diagnostic](../experiments/2026-09-25-diagonal-feasibility-plan.md)
asks a different question from the last two training experiments: does any
setting of this frozen-feature head fit every training relation?

The mean-BCE and worst-member refits reached 122 and 128 exact validation
answers, compared with 112 for the original dense head. Both failed the stronger
201-answer comparator. That does not tell us whether we chose poor optimization
recipes, exhausted the head's representational ability, or failed to generalize.
Another loss curve alone would not separate these possibilities.

## A large matrix can have few effective degrees of freedom

The current score for subject s, dimension d and candidate i is

\[
r_d=WU_d,\qquad
z_{s,d,i}=\frac{(E_s\odot E_i)^T r_d}{16}.
\]

Here W has shape [256,256], E has shape [1025,256], and each steering vector
U_d has shape [256]. The denominator is sqrt(256). With all embeddings frozen,
only r_TYPE and r_COLOR matter. W contains 65,536 entries, but the two vectors
contain at most 512 independently adjustable coefficients altogether.

There is a condition: the two steering vectors must be linearly independent.
Writing their columns as U[256,2] and the desired relation vectors as R[256,2],

\[
W=R(U^T U)^{-1}U^T
\]

gives WU=R when U has rank two. The run verified that rank with an exact nonzero
2-by-2 minor of the saved constants. The separate FP64 condition number is about
1.288. An ill-conditioned U can make the reconstruction numerically awkward,
even when the exact algebra works; rank and conditioning are different properties.

For one dimension, the full subject/candidate score matrix has the form

\[
S_d=\frac1{16}E\operatorname{diag}(r_d)E^T
=\frac1{16}\sum_{k=1}^{256}r_{d,k}E_{:,k}E_{:,k}^T.
\]

The head can reweight a fixed collection of 256 coordinate patterns. It cannot
freely introduce cross-coordinate interactions between subject coordinate j
and target coordinate k when j differs from k. Changing the loss changes which
weights we prefer; it does not enlarge this collection of possible scores.

## Turn the prediction rule into inequalities

For fixed embeddings, define x_si=(E_s elementwise-multiplied by E_i)/16.
It is a known vector of length 256. The unknown is r_d, also length 256.
Each labeled query/candidate pair gives one linear inequality:

\[
x_{si}^T r_d\ge1\quad\text{for true members},\qquad
x_{si}^T r_d\le0\quad\text{for nonmembers}.
\]

Why one for true members? The predictor only needs a strictly positive score.
If a finite set of true scores is all positive, its smallest value m is positive.
Replacing r_d with r_d/m makes every true score at least one without changing
any signs. Conversely, a score at least one is positive.

Why zero for negatives? Zero is excluded by the actual z>0 predictor. Requiring
every negative score to be at most -1 would ask a stronger question. For example,
a negative with x=0 always scores zero: that is a valid exclusion, but it can
never meet a -1 bound. A failed stronger test would not prove failure of our rule.

There are 1,637 training queries and 1,024 nonself products per query:
**1,676,288 inequalities** before duplicate handling. A full FP64 coefficient
matrix with 256 columns would occupy about 3.2 GiB before solver overhead.
Instead, the declared solver starts with two queries per dimension and adds
violated constraints in a fixed order, under fixed time and working-set limits.
This is a **cutting-plane procedure**: propose a solution, find constraints it
breaks, and add some of those constraints to the next solve.

This use of every product column follows the existing transductive training
setup. The solver sees training relation labels only. Validation is not used
to choose weights, thresholds, constraints or a stopping point.

## A solver's status is not a mathematical proof

A floating-point solver can return a candidate vector, report apparent
infeasibility, or exhaust its budget. Its numerical tolerances are useful for
finding proposals, but they are not our final evidence. The
[SciPy interface](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs.html)
also defaults variables to nonnegative values. We explicitly allow each r
coordinate to be any real number; otherwise we would silently test a different
feature family.

For a feasible proposal, every original training sign must be checked. The
saved FP32 embeddings and proposed FP64 vector have exact representations as
integers divided by powers of two. Multiplying by positive common denominators
lets us check the score signs using integer arithmetic, without a rounding
tolerance. A feasible result establishes training representability, not good
validation answers or a practical serving model.

For infeasibility, write the signed constraints as Ar>=b. A certificate consists
of nonnegative weights lambda such that

\[
A^T\lambda=0,\qquad b^T\lambda>0.
\]

To see why this proves a contradiction, take the weighted sum of all the
inequalities. If any solution existed, it would imply

\[
0=(A^T\lambda)^T r\ge b^T\lambda>0.
\]

That is impossible. For the tiny contradictory pair r>=1 and -r>=0, weights
(1,1) give exactly this contradiction. Our real certificate can involve more
constraints, but the logic is identical. Nonnegative weights matter: multiplying
an inequality by a negative number would reverse its direction.

The solver proposes a small support; exact integer nullspace arithmetic then
tries to construct the witness. The independent auditor must verify its equations
and bind each supported row to the real training label. A tiny residual is not
enough: with unbounded r, an apparently tiny coefficient error can be multiplied
by an arbitrarily large parameter. Failure to recover a valid exact certificate
within the declared budget means **inconclusive**, not impossible.

## What the allowed outcomes mean

There are three allowed outcomes:

- **Certified feasible:** both dimensions have independently verified vectors
  satisfying every training sign, and the steering rank condition holds.
- **Certified infeasible:** at least one dimension has a valid contradiction
  certificate for a subset of its actual training constraints.
- **Inconclusive:** the bounded computation yields neither complete witness.

The arithmetic scope is deliberately narrow. We treat the saved, normalized
FP32 embeddings as exact constants and analyze their real-arithmetic feature
family. The deployed FP32 head performs a particular sequence of rounded matrix
operations. A result here does not prove identical behavior for every possible
rounding-dependent FP32 parameter setting, nor a limit on the transformer or on
models allowed to change embeddings.

If this restricted family is certified infeasible, a richer symmetric relation
head is a better-motivated next intervention than another optimizer tweak. If a
solution is certified feasible, we have evidence that the training constraints
are representable, while optimization and generalization remain separate tasks.
Neither outcome establishes oracle parity on validation or the protected test.

## Observed result: a rounding disagreement, not a capacity proof

The [completed record](../experiments/2026-09-25-diagonal-feasibility.json)
contains the separate primary report, independent audit and owner decision
identities. The code and environment were frozen after 41 runner tests and
36 auditor tests passed. The one declared run then completed successfully.

| Dimension | Training queries | Nonself constraints checked | Primal LPs | Result |
| --- | ---: | ---: | ---: | --- |
| TYPE | 805 | 824,320 | 1 | Inconclusive |
| COLOR | 832 | 851,968 | 1 | Inconclusive |

Each LP used the initial 2,048 constraints and returned a numerical solution.
That status concerns the working subset under the numerical solver's tolerances;
it is not a certificate for all training relations. The exact sign checker then
examined every training constraint and rejected both proposed vectors.

In each dimension, at least one negative candidate had an exactly positive score
even though its FP64 inequality residual was nonpositive. The first recorded
hidden failures were train index 0/product ID 1321 for TYPE and train index
1/product ID 1246 for COLOR. Their FP64 residuals were approximately
-1.11e-16 and -2.22e-16. These are floating residuals, not the exact scores.
The exact calculation uses the same saved FP32 features and FP64 vector, treating
their binary values as exact numbers.

Why can the sign change? A dot product adds positive and negative terms:

\[
z=\frac1{16}\sum_{k=1}^{256} E_{s,k}E_{i,k}r_k.
\]

When large terms nearly cancel, the remaining value can be tiny. Rounding each
floating operation can change that remainder's sign. Our prediction rule uses
z>0, so this is a decision boundary, even when the numerical difference looks
small. Rounding a violation away would change what we mean by an exact proof.

The declared procedure stops if any exact failure is hidden by the floating
scan. It therefore stopped after the first proposal for each dimension, before
adding cuts. Neither worker timed out or crashed. There were no dual solves,
contradiction certificates or certified feasible witnesses. The independent
auditor recomputed all 1,676,288 signs and confirmed the same stopping reason.
It did not rerun the LPs or CUDA normalization; it shares the pinned FLINT integer
library while constructing its own arithmetic and provenance checks.

The scientific conclusion is narrow: **these two candidate vectors fail, and
this declared procedure did not resolve the frozen head's capacity**. We cannot
replace that statement with "the head is too small" or "the head can fit it."
More time alone would not change this run's stopping rule. A different numerical
method would need a separately declared diagnostic; a richer head remains a
separate hypothesis, not a necessity established by this result.

No neural checkpoint changed. The 29 registered final checkpoints remain; the
two rejected FP64 proposals are retained in the dimension reports as diagnostic
coefficients, not certified models. There were no validation or protected-test
predictions and no serving promotion. The 10.60-second local wall time is
descriptive, not a serving benchmark. Raw normalized features remain local
weight-derived dependencies: their hashes establish identity, not remote backup.
