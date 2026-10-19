# Saved-error geometry after 32,000 bilinear updates

Campaign `bilinear-budget32000-error-geometry-v1` is a posthoc descriptive
diagnostic of the already accepted fixed validation results. Declared before
executing the new reductions. It does not train, select a threshold, generate
alternative predictions, promote a model or open the protected partition.

## Question and scope

Identify which saved membership decisions account for the remaining six
nonexact seed-query outcomes. Distinguish ranking overlap from displacement
relative to the fixed zero cutoff, and determine whether failures recur on the
same queries across parent seeds. This evidence informs a future separately
declared intervention; it is not independent generalization evidence.

Read only authenticated saved validation JSON and provenance source/test files.
Do not read weights, graph databases, training examples, training score matrices
or protected-test data. No Torch, GPU, model forward or optimizer is needed.
The existing 41 checkpoint identities and all inference defaults stay unchanged.

Authenticate the upstream `bilinear-budget32000-v1` evidence:

- `aggregate-v2.json`: `8f4539583080275efe53409ce6881af3b50f2b89a092e0a21801ceafe6fd0708`.
- `aggregate-audit-v2.json`: `bacc8614e11afceee2080a3faf9850c1b70de92cf5cbd3ac912025da4edb0b3c`.
- `decision-v2.json`: `4a508abaf3fac7e2e678cd5604f73636241aabcb18c876076c134468257e44af`.

Require complete primary and independent aggregate reports, successful audit,
and separate owner evidence acceptance. Resolve each of seeds 1729/1730/1731
through its bound summary and actual per-seed audit path, including the repaired
v2 audits where recorded. Authenticate child report hashes from those summaries
and reconcile child metrics/checkpoint identity with their audits and the
aggregate. Check checkpoint hashes as metadata only; never open weight files.

Require exactly 222 ordered validation queries and ordered product columns
1024 through 2048 in each child. Align query index, subject, dimension, group,
prompt IDs and expected target IDs across all three children. Reject drift,
duplicate indexes, malformed truth/selection, self membership, nonfinite or
non-FP32 scores, and disagreement with saved selections or metrics.

## Fixed arithmetic

Reuse the unchanged scientific reductions in `scripts/diagnose_bilinear_errors.py`,
SHA256 `66093bfbd06c8622b61535eeaffd0b09886c0cee05ec9eae71292a6e85925789`.
Verify the helper's exact bytes before executing them and again at completion.
Use a new wrapper and new outputs; do not modify the frozen helper or old results.

For every row, exclude the subject and compute

\[
a=\min_{i\in T}z_i,\quad b=\max_{i\notin T,\ i\ne s}z_i,\quad g=a-b.
\]

Reconstruct the original ascending, nonself `z > 0` selected IDs. The set is
exact iff `a > 0` and `b <= 0`; strict ranking separation means `a > b`.
Use the existing four exclusive categories: exact at zero, separated missing,
separated extra, and overlap or tie. Record deterministic lowest-ID witnesses
for tied extrema and all false-positive/false-negative IDs and their scores.
Publish all 222 reduced rows per seed, not only selected failures.

Report per-seed and group counts, pooled counts without pooled score intervals,
and aligned-query failure-seed lists and overlap counts. Reconcile exactness,
strict separation and FP/FN totals with the audited upstream evidence.
The 666 observations reuse 222 queries; do not describe them as independent.

For each seed and each seed/group report `A=min(a)`, `B=max(b)` and whether
`B < A`. The hypothetical threshold interval `[B,A)` describes whether any
constant could separate those saved labels. Do not choose a threshold, measure
new predictions, pool cross-seed scores or present this as deployable calibration.
No numerical tolerance, rounding-based category or near-zero bucket is allowed.

## Verification and execution

Freeze wrapper, helper, plan, tests and their successful test receipt before
the one intended primary diagnostic invocation. Preflight may authenticate
schemas and identities but must not calculate the new diagnostic reductions.
Tests must exercise genuine failure cases: forged/incomplete evidence chains,
wrong audit routing, child/metric identity drift, query/column drift, ties and
signed zero, nonfinite scores, wrong saved selections, and overwrite rejection.
Existing helper tests may supply the already covered arithmetic cases.

A separate stdlib auditor must independently reconstruct every reduced row,
group, pooled count and overlap result from authenticated children. It must
not import the primary scientific arithmetic. It may reuse independently
authored historical audit arithmetic only with an exact helper hash. Test the
actual schema contract before execution, with tampered-result rejection.

Use immutable exclusive-create outputs under
`runs/learning/bilinear-budget32000-error-geometry-v1/`. Preserve any failed
attempt and declare a versioned repair instead of overwriting. Record observed
terminal completion separately from the tool's own success fields. Final
evidence acceptance requires a passing independent audit and an owner decision;
the primary report's `acceptance` stays false. Output files must be at most
512 KiB, with complete deterministic partitioning if needed.

## Recording and next decision

Update the registry, append-only research and development logs, and a teaching
lesson. Build an immutable source and PLM-findings handoff after verification,
depending on the accepted 32,000-update campaign packet. Notify its dedicated
publisher and the canonical-paper owner; retain their file ownership. This
diagnostic changes neither the retained model nor the number of checkpoints.

The next research decision must distinguish evidence from hypotheses. A
zero-cutoff error does not prove that a train-only calibration will generalize;
a ranking overlap does not prove that more expressive features are necessary.
No new training, validation-fitted cutoff, oracle-count repair, per-query
routing or protected evaluation is authorized by this diagnostic plan.
