# Frozen diagonal-head training feasibility diagnostic

Declared 2026-09-25 before extracting diagnostic features or solving any
real-data feasibility problem. Pokemon TYPE/COLOR SAME only. This is a bounded
capacity diagnostic, not another quality screen or a continuation of a rejected
checkpoint. No validation or protected-test predictions or parameter selection.

## Question and fixed input identities

The original dense parent yields112/222exact validation sets; fixed-budget mean
and worst-member projection refits yield122and128, both below the strong201
comparator. Finite optimization failures do not prove a capacity limitation.
Ask whether the original frozen diagonal feature family can represent every
training membership decision under the exact zero-threshold rule.

Use original checkpoint
e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1,
not either refit child. Authenticate the old mean-refit evidence chain:
summary0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61,
audit6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc,
decisionda0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6.
Its train-membership.json has SHA256
0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad.
Use exactly those1637ordered training queries and their deduplicated labels.
All1025product columns are eligible except the SAME subject:1,676,288
query-product constraints before any duplicate avoidance. No external dataset.

Authenticate parent checkpoint/sidecar/run, vocabulary/protocol/corpus/split
identities and training membership against that accepted evidence. The independent
auditor reconstructs training membership from the original split. Test keys may
establish partition identity; no test labels or test predictions enter the solve.

## Feature family and arithmetic boundary

Import only authenticated objective-independent feature extraction from the
frozen mean runner79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c.
Normalize original product and TYPE/COLOR token embeddings in FP32 on the recorded
CUDA device using its _features operation, autocast off and archived FP32 settings.
No transformer forward, gradients or optimizer update. Save E[1025,256] and
U[2,256] as lossless FP32 arrays with file/array hashes, token-column identities,
environment, parent tensor hash and extraction source receipt. Require finite
E/U before any conversion or solve. Preserve originals.

Treat those quantized constants as exact real numbers for this diagnostic:

    r_d = W U_d
    x_si = (E_s elementwise-multiplied by E_i) / 16
    z_sdi = x_si dot r_d

For fixed E this is a256-coordinate diagonal metric per dimension. W has65,536
entries but can affect TYPE/COLOR only through two256-element vectors. Verify
U has exact rank2 by an integer/rational nonzero2x2minor; report a descriptive
FP64 condition number separately. If rank2 fails, stop as inconclusive rather
than assuming independent relation vectors equal the shared-W family.

For rank2, every R=[r_TYPE,r_COLOR] is attainable in real arithmetic via
W=R(U_transpose U)^(-1)U_transpose, with U treated as256x2 here. This is a
mathematical reduction, not a promise of a stable FP32 reconstructed checkpoint.
The original FP32 head's multiplication order is not this exact-real problem.
Do not claim a formal limit on rounding-dependent FP32 behavior or on the
transformer, alternative embeddings, or richer relation heads.

## Inequalities matching dense selection

For each training query and nonself product, require:

    x_si dot r_d >= 1     if the product is a true member
    x_si dot r_d <= 0     otherwise

On a finite dataset, positive rescaling makes these constraints equivalent to
strictly positive true scores and nonpositive false scores. Exact zero is allowed
only for negatives. Do not replace negative0 with negative1: two-sided strict
margins ask a stronger question and their infeasibility would not answer this one.
Write the complete system as A r >= b, where b is1for positives and0for negatives.
No biases, new features, threshold search, norms or parameter magnitude bounds.

## Bounded solver and exact witnesses

Use an isolated Python3.12 solver project with a frozen lockfile; do not modify
the main or serving environments. Pin NumPy2.2.6, SciPy1.15.3 and
python-flint0.8.0. Freeze installed versions and solver source before real execution.
If dependencies cannot be resolved, report the preparation blocker without
substituting versions after measurements. SciPy linprog uses highs-ds, presolve
enabled, primal/dual tolerances1e-9, a30-second limit per LP and explicit free
primal variables (the library default nonnegative bounds would be wrong). Both
primal and dual feasibility LPs have identically zero objective vectors. Reject
nonfinite solver vectors, bounds or timing values before use or exact conversion.

Solve TYPE and COLOR separately, in that order, once each. Each dimension worker
has a hard180-second wall budget enforced by the parent process. Retain partial
traces and classify a terminated worker as inconclusive; do not restart it.
Check remaining budget before every phase. Maximum16primal LPs and4096working
inequalities per dimension; solver internal memory is not claimed to be capped.

Start with all nonself columns from the first two training queries of that
dimension in authenticated order. Avoid duplicate query/product identities.
After a candidate, scan all training constraints for that dimension in FP64
without materializing the full coefficient matrix. Append at most128previously
unselected rows with strictly positive violation magnitude max(0,b_i-A_i r)
across BOTH positive and negative labels, sorted by descending violation, then
original training index, then product ID. Preserve all existing constraints. Stop if no
new violating row can be added, a resource bound is reached, or a witness is
verified. No alternate initialization, hyperparameter sweep or post-hoc extension.

Every LP result and support selection is a proposal, not a proof. For a proposed
feasible r, interpret its saved FP64 components exactly, convert E and r to scaled
integers, and check every original training sign with exact integer arithmetic.
FLINT may accelerate this matrix product; an independent auditor must reconstruct
the same integer quantities without importing primary decision arithmetic.
Accept only true>0andfalse<=0, with subject exclusion and full query coverage.
If exact verification fails, continue only within the declared cutting-plane
budget; near-zero numerical violations are not rounded away. If exact sign
verification finds any failure whose FP64 inequality residual is nonpositive,
stop inconclusive rather than treating the floating scan as a certificate.

If the primal LP reports infeasible, make one dual feasibility solve on its
working set: lambda>=0, A_transpose lambda=0, b_transpose lambda=1. Use the same
method/tolerances/remaining time bounds. Take all strictly positive returned
support coefficients; reject support larger than257. Reconstruct an exact null
vector on that support using the signed integer feature rows and FLINT nullspace.
Require nullity1; orient its sign and reduce common integer factors. A different
nullity or mixed signs gives an inconclusive certificate attempt, not a search
over alternative supports. Preserve the approximate proposal and exact attempt.

An infeasibility certificate must independently satisfy exactly:

    lambda >= 0
    A_transpose lambda = 0
    b_transpose lambda > 0

Use integer coefficients from the common power-of-two scaling of FP32 E; the
positive common feature denominator cancels in the zero equality. A nonzero
positive-label multiplier establishes the final strict inequality. Record support
query/product IDs and normalized integer multipliers as hexadecimal strings to
avoid decimal-digit conversion limits. Limit each certificate artifact to8MiB;
excess or timeout is inconclusive. Numerical residuals alone can never certify
infeasibility for unbounded r. Any supported row must be independently rebound to
its actual training label and feature coefficients.

## Outcomes, audit and provenance

Per dimension report certified feasible, certified infeasible, or inconclusive.
The whole exact-real diagonal family is certified infeasible if either dimension
has a valid certificate; it is certified feasible only if both complete training
sign checks pass and U rank2 is verified. Otherwise the overall result is
inconclusive. A numerical infeasible status without a valid certificate stays
inconclusive. Distinguish complete diagnostic execution from mathematical outcome.

Run ID diagonal-feasibility-v1; evaluator plm-diagonal-feasibility-v1. This creates
no neural training checkpoint and changes no existing weights or serving policy.
A feasible relation vector is a new diagnostic fitted-coefficient artifact with
its own source/data/solver identities, not an unversioned model or a promoted
checkpoint. Final neural checkpoint count remains29. No validation quality or
generalization claim follows from training feasibility, and no impossibility
claim extends beyond this fixed exact-real feature family.

Before extraction/solving, freeze the plan, source, environment lock and focused
synthetic test receipts. Cover zero-threshold asymmetry, free-variable bounds,
integer feature conversion, feasible and contradictory toy systems, sign/support
binding, rejected approximate certificates, rank reduction, budgets, immutable
outputs and partial failures. Do not run actual data while tuning these tests.

An independent auditor verifies hashes, split/training labels, feature identities,
rank witness, all feasible signs or certificate equations, status reduction and
fixed limits. Check each dimension's authenticated training query count times1024,
not just a reported pooled count. It does not pretend to independently replay CUDA normalization.
Preserve separate evidence acceptance/diagnostic outcome, a teaching chapter,
model-register diagnostic entry and PLM-findings update, then notify the existing
LunaMax publisher with an immutable completed packet. Raw normalized embeddings
remain local weight-derived dependencies unless separately authorized for archive;
published hashes do not establish their availability. Keep LMStudio offline and
coordinate the brief GPU feature extraction with the benchmark owner.

## Solver references

[SciPy1.15.3 HiGHS interface](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs.html)
documents explicit variable bounds, limits and numerical status codes.
[python-flint integer matrices](https://python-flint.readthedocs.io/en/latest/fmpz_mat.html)
documents exact integer matrix operations and nullspace bases. Pin and test the
installed0.8.0 API rather than assuming the latest online documentation is identical.
