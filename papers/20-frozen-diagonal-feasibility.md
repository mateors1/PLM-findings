# A frozen-head feasibility test stops without a mathematical answer

**Status: independently audited evidence accepted; diagnostic outcome inconclusive.**
The bounded test neither proves that the original diagonal membership head can
fit every training decision nor proves that it cannot. Both TYPE and COLOR stop
under the declared numerical safeguard: an exact sign failure is hidden by the
corresponding FP64 inequality calculation. This is a completed, audited diagnostic
with an inconclusive mathematical outcome, not a failed quality screen.

The [frozen plan](../evidence/updates/2026-09-25-diagonal-feasibility/experiments/2026-09-25-diagonal-feasibility-plan.md),
[portable result](../evidence/updates/2026-09-25-diagonal-feasibility/experiments/2026-09-25-diagonal-feasibility.json),
[independent audit](../evidence/updates/2026-09-25-diagonal-feasibility/campaign/independent-audit.json)
and [owner decision](../evidence/updates/2026-09-25-diagonal-feasibility/campaign/decision.json)
keep the scientific outcome separate from evidence acceptance.

## The question left by the refits

The original parent's dense predictor obtained 112/222 exact validation sets.
The [mean-BCE refit](18-frozen-feature-projection-refit.md) reached 122 and the
[worst-member refit](19-worst-boundary-projection-refit.md) reached 128; both
remained below the accepted eight-branch predictor's 201. Those finite training
runs do not establish whether optimization or expressivity limits the head.
This diagnostic asks a narrower question about the training partition alone.

The original seed-1729 checkpoint supplies normalized, scaled product embeddings
E[1025,256] and steering vectors U[2,256]. They are extracted with the authenticated
FP32 operation on CUDA, without transformer forward passes, gradients or updates.
Their saved FP32 values are then treated as exact real constants. For subject s,
dimension d and candidate product i, the proposed family is

\[
x_{si}=\frac{E_s\odot E_i}{16},\qquad
z_{sdi}=x_{si}^{\top}r_d,\qquad r_d=WU_d.
\]

The TYPE/COLOR steering vectors have exact rank two, established by a nonzero
integer 2-by-2 minor. Thus any pair of 256-element relation vectors is attainable
by some real W: TYPE and COLOR can be tested separately. The descriptive FP64
condition number is approximately 1.28760; that numerical value is not the rank
proof. This reduction does not construct a stable FP32 checkpoint.

## Matching the actual zero-threshold rule

For every training query, the subject is excluded. Correct members require a
strictly positive score; incorrect members may have score zero because prediction
uses z>0. On a finite training set, rescaling any successful relation vector gives
the equivalent feasibility constraints

\[
x_{si}^{\top}r_d\ge 1\quad\text{for true members},\qquad
x_{si}^{\top}r_d\le 0\quad\text{for false members}.
\]

A negative bound of -1 would ask a stronger question and would not be a valid
substitute. Variables are explicitly free, with no nonnegativity or magnitude
bounds, biases, new features or threshold tuning. The objective is identically
zero: the linear program seeks any admissible vector, not a better loss value.

The frozen budget permits up to 16 primal LPs and 4,096 working constraints per
dimension, starting with all nonself products from the first two training queries.
Each LP has a 30-second limit; each isolated worker has a hard 180-second wall
limit. Numerical candidates are checked against every training constraint with
exact integer arithmetic. Only a successful full sign check could certify
feasibility. Numerical infeasibility would require a separately verified Farkas
certificate: nonnegative multipliers whose weighted feature rows cancel exactly,
with positive weight on at least one positive-label right-hand side.

## What happened

| Measurement | TYPE | COLOR |
|---|---:|---:|
| Training queries | 805 | 832 |
| Nonself membership constraints | 824,320 | 851,968 |
| Primal LPs executed | 1 | 1 |
| Working constraints | 2,048 | 2,048 |
| Exact constraints checked | 824,320 | 851,968 |
| Dual LPs / certificates | 0 / 0 | 0 / 0 |
| Hard timeout | No | No |
| Mathematical outcome | Inconclusive | Inconclusive |

Together these cover 1,637 ordered training queries and 1,676,288 membership
constraints. Each initial LP produced a numerical candidate. Full exact checking
found sign failures, including at least one failure per dimension for which the
FP64 residual b-Ar was nonpositive. The frozen rule required an immediate
inconclusive stop even if other violated constraints could have been added.
Neither worker exhausted its time budget or reported a software error.

For a concrete view of the boundary, TYPE recorded a hidden exact failure at
training index 0, product ID 1321, while its FP64 residual was
-1.1102230246251565e-16. COLOR recorded one at training index 1, product ID 1246,
with residual -2.220446049250313e-16. These are saved examples, not aggregate
rounding-error estimates. Their tiny residuals cannot be treated as exact signs.
The diagnostic did not round them away, relax its rule or restart with different
solver settings after observing the outcome.

## What the evidence does and does not establish

The numerical proposals are not certified feasible vectors, and there is no
infeasibility certificate. Therefore the diagonal family's training feasibility
remains unresolved. It would be incorrect to use this result as proof that a
larger architecture is necessary, or as proof that another optimizer must succeed.

The mathematical problem concerns exact arithmetic over fixed quantized features.
The deployed FP32 head has a particular multiplication and reduction order; this
diagnostic is not a formal bound on rounding-dependent FP32 behavior. It also says
nothing directly about validation generalization, alternative embeddings, richer
heads or the transformer.

The runner passed 41 focused tests (29 orchestration and 12 isolated solver tests);
the independent auditor passed 36 synthetic tests. Python 3.12, NumPy 2.2.6,
SciPy 1.15.3 and python-flint 0.8.0 are pinned in a separate solver project. The
auditor independently reconstructs training labels and exact sign checks from
saved evidence, while sharing the pinned FLINT arithmetic library. It does not
repeat CUDA normalization or execute the LPs independently; timing and execution
order remain authenticated receipts rather than independently replayed events.

No neural training, validation prediction, protected-test prediction or policy promotion
occurs. No neural checkpoint is created; the register remains at 29 final neural
checkpoints. The roughly 10.6-second overall execution is descriptive diagnostic
time, not a serving or energy benchmark. Raw normalized embeddings, weights and
training membership labels remain hash-bound local dependencies rather than
published payloads. The [evidence inventory](../evidence/updates/2026-09-25-diagonal-feasibility/README.md)
records that boundary explicitly.
