# Projection-only worst-member softplus screen

Declared 2026-09-25 before new training, scoring or outcome measurement.
Pokemon TYPE/COLOR SAME only; single seed1729. This is a distinct sibling
experiment from the original accepted parent, not additional optimization of
the rejected mean-BCE child. No protected-test query is evaluated.

## Fixed inherited contract and motivation

Inherit the data, original parent, frozen features, FP32 head arithmetic,
full-batch training order, optimizer, numerical settings, validation batch
shapes, dense prediction rule, checkpoint discipline and strong gate from the
[projection-only plan](2026-09-25-projection-only-refit-plan.md), SHA256
2ac4d157ad8c079561d642199fec8de2c3da1866819baa48f52080e9b0f39bfe.
The explicit changes below define the entire treatment; every other inherited
requirement remains binding. The old plan's objective identity applies only to
the old run and is replaced here as specified below.

Original parent checkpoint:
e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1.
Initialize only from that artifact, with its 2,000 parent training steps.
Only symmetric_relation_projection.weight, W[256,256], may change.
Run exactly 500 updates over all 1,637 training queries in authenticated order.
Fresh non-fused AdamW: lr0.0003, betas(.9,.999), eps1e-8, decay0. No scheduler,
clipping, additional objective, margin, temperature, threshold tuning, sweep,
validation stopping or intermediate-checkpoint selection. FP32, autocast off,
CUDA matmul TF32=false, cuDNN TF32=true; archived replay settings are unchanged.

The completed mean-BCE sibling improved dense exactness112 to122 but failed
the strong width-eight comparator201. Its accepted evidence identities are:

- Summary0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61.
- Audit6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc.
- Decisionda0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6.
- Child reportb44463dd43df234aaba5f95a8910eb5afea1bee012782f2a9d19896adb221f1f.
- Mean-refit checkpoint84a3ba02ab5f43e9073f1f8b8c7afc85ad7a5363e8cb180e1424a77e6a44ab87.

That sibling's results are authenticated historical comparators, not a newly
executed control. This compares fixed-budget recipes from the same parent;
equal AdamW settings do not establish equal optimization strength across losses.

## One replacement objective

For each query q, derive deduplicated positive products P and negative products
N from training labels, with the SAME subject excluded from both. Retain the
existing control/padding masking and empty-class rejection contract. With
scores Z[1637,1025], compute in FP32:

    a_q = min(z_qi for i in P)
    b_q = max(z_qi for i in N)
    L_q = 0.5 * (softplus(-a_q) + softplus(b_q))
    L = mean_q L_q

This replaces both per-class averages in the prior balanced BCE with their
worst-member penalty. It is the sole objective, coefficient1. Use masked
torch.amin/amax reductions, which divide gradients evenly across exact ties;
do not replace them with indexed min/max selecting one tied member. Reject
invalid masks before reductions and require finite losses, gradients and W.

Exact dense prediction needs a_q>0 and b_q<=0 because exact-zero scores are
excluded. The new objective pressures both sides of this fixed zero boundary.
The previous relative hinge relu(m+b_q-a_q) was invariant to shifting every
score by the same constant; this objective is not. No query-specific threshold
is fitted here. The shared symmetric projection still couples all queries.

With fixed features, this objective is convex in W but can be nondifferentiable
at ties. Convexity establishes neither convergence within500updates, a finite
minimizer, representational adequacy nor validation generalization. Focusing
gradients on extreme members changes their distribution and can hurt quality.
It does not prove the cause of previous joint-training margin failures.

## Reuse, diagnostics and evaluation

Authenticate and import the frozen mean-refit runner snapshot SHA256
79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c.
Reuse its objective-independent head/features, prediction, metrics, exact-reload
checks and _fit loop with an explicit new loss callback. Do not monkeypatch its
globals, private functions or archived modules, or call its objective-specific
_execute or _validate_child. Supply a new child-contract validator for the new
objective and evaluator. New orchestration owns the new recipe/identity/loss and adds only the
required comparator and diagnostic reporting. Snapshot and hash every imported
helper; no live core/serving change is needed for this screen.

Before any fitting, reproduce all222parent validation vectors exactly at the
historical ordered B8/final6 shapes and require standalone-head/model-head
agreement. Also compute the initial full-batch mean BCE using the archived
loss and require exact agreement with the previous initial value
0.003822767175734043 at that same training shape. A mismatch stops the campaign;
retain failed evidence and do not loosen the comparison after observing it.

Store all500pre-update values of the new objective and its finalpost-update
value. Separately record mean BCE on the initial and final full training batch
as a diagnostic, without gradients for those diagnostic calculations. Those
extra scalar measurements never control updates, stopping or selection. Do not
compare magnitudes of different loss definitions as though they were one metric.
The saved scalar trace is execution evidence, not independent replay of training.

Save/reload the final full child plus fresh optimizer state. Verify that only W
changed across all state tensors and optimizer state restores exactly onto the
reloaded model. Evaluate the child once at update500, with the same validation
order/shapes and dense z>0, subject-excluded, ascending-ID rule. Keep all empty
or oversized sets and their raw metrics; no fallback/truncation or generatedEOS.

Report parent/child dense metrics, groups, false positives/negatives and paired
exact gains/losses against both the historical mean-refit child and accepted
width-eight predictor. Authenticate query order and labels for every comparison.
Report strict separation and its gains/losses separately from exactness.
Every validation score vector is saved for independent arithmetic audit.

## Unchanged strong gate and new lineage

Require complete authenticated execution, exact replay,500finite updates,
train-only supervision, unchanged non-W tensors, exact model/optimizer reload,
an accepted independent audit, and all222child sets containing1through506
unique non-subject products. The strong quality requirements remain:

- Child exact answers strictly greater than201/222.
- Child macroF1 at least0.9799255176742276.
- Group exact counts at least COLOR103, single-TYPE50, dual-TYPE48.

Other comparisons are descriptive. No extra post-hoc acceptance condition or
relaxation is allowed. A single-seed pass supports proposing a separate fixed
replication, never automatic promotion. A failed gate stays failed.

Run ID projection-worst-boundary-v1; new objective
plm-projection-worst-boundary-softplus-v1; evaluator
plm-projection-worst-boundary-screen-v1. Record parentsteps2000 and newupdates500
separately; new global_step=500. This is a sibling of the rejected mean-refit
child, not its continuation. The standard serving loader remains incompatible
with this distinct objective. Inventory currently contains28final checkpoints;
register any completed new derivative even if the gate fails. No ignored weight
payload is automatically published or durably archived.

## Verification and publication

Freeze recipe/runner/source/test receipts before execution. Torch-free preflight
authenticates old and new evidence; tests use the existing required portable
source ZIP fixture. Add synthetic tests for mean-versus-extreme weighting,
absolute-zero versus relative-gap behavior, tied subgradients, subject/control
masking, nonempty classes, FP32/autocast behavior and the explicit new callback.
Retain focused lifecycle, data separation, output, gate and checkpoint tests.

An independent saved-evidence auditor must reconstruct predictions, both paired
comparisons, metrics and the strong gate, verify split/train membership and
checkpoint/optimizer/tensor lineage, and authenticate scalar traces and their
endpoint bindings. It must not import primary prediction/aggregation arithmetic
or claim an independent training replay or regenerated neural logits.

Retain immutable failures, a separate owner decision, new model-register entry,
teaching lesson and PLM-findings update. Notify the existing LunaMax publication
task with a completed immutable packet and receive its publication receipt.
Keep the user-requested LMStudio backend offline. Coordinate the single GPU
campaign; timings remain descriptive without serving/energy/concurrency claims.
