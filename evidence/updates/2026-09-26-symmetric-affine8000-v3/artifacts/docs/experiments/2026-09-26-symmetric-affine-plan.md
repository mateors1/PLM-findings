# Symmetric affine residual: fixed 8000-update pilot

Campaign `symmetric-affine8000-v1`; declared before candidate neural execution.
This is one adaptive validation screen on seed 1729, within Pokemon TYPE/COLOR
SAME. A pass permits separately declared replication, not serving promotion.

## Hypothesis and evidence

The accepted saved-score diagnostic `bilinear-budget-geometry-v1` has summary
SHA256 `3264144d880fc4fea7487e80bc0a5e5e6275cf294c723af3cb3c0e5e5f24b38d`,
audit `a3d65939a5f844f5eaf0e087bb3ec5071489e1bab8173c2c1c58544710a7c8ac`,
and decision `92b23e1d211c4775a4ec233cee4a2e3f18369536cf2c4c00aabd1cc42caebdea`.
It rules out one shared scalar threshold as a complete rescue for each fixed
seed's dual-TYPE scores. It does not establish that affine learning will help,
that the model is undertrained, or that all calibration methods must fail.

Source inspection of `scripts/refit_membership_projection.py::_head` shows the
parent head is a dimension-conditioned diagonal pair interaction. The existing
bilinear correction expands pair interactions. Add explicit tied subject and
candidate linear terms plus a dimension bias. This changes rankings as well as
boundaries; it is a richer symmetric scorer, not threshold-only calibration.

## Fixed scorer and tensor contract

Let E be the unchanged FP32 entity embedding table [1025,256], with rows equal
to 16 times normalized parent embeddings. For subject s, candidate i and
steering dimension d in {TYPE,COLOR}, use

    z(s,i,d) = z_parent(s,i,d)
             + E_s^T ((A_d + A_d^T)/2) E_i / 16
             + (u_d^T E_s + u_d^T E_i) / 16 + b_d.

Train A[2,256,256], u[2,256], b[2], all exact zero initially. The added affine
terms use the fixed divisor 16; do not select their scale using validation.
This preserves pair symmetry in real arithmetic. Separate batched FP32 paths
can differ by rounding. Synthetic symmetry checks use rtol=1e-5, atol=2e-5
with finite FP32 fixture entries bounded in absolute value by 1 and feature
dimension at most 256; this tolerance is not a production quality tolerance.
Zero affine terms must exactly recover the existing bilinear scorer operation
order. Zero A/u/b must exactly recover the original parent's score vectors.
The residual has 131586 stored trainable scalars, 514 more than A alone. A is
stored full-size but only its symmetric part affects scores; do not call all
131072 entries independent functional degrees of freedom.

Compute candidate projections once per dimension: Q=u E^T / 16 [2,1025].
For a batch B, gather the selected dimension rows [B,1025], subject values
[B,1], and biases [B,1], then add to parent plus bilinear scores [B,1025].
Do not materialize [B,1025,256]. Parameter names are
`symmetric_bilinear_residual`, `symmetric_affine_linear`, `symmetric_affine_bias`.
Freeze all 93 original state tensors; expected final state has 96 entries.
Core package import must remain torch-free. Implement experimental code under
`scripts/` with torch imported only inside runtime functions.

Architecture identity: `plm-frozen-symmetric-affine-residual-v1`.
Objective identity: `plm-affine-residual-balanced-bce-v1` (same balanced BCE
formula and masks, new parameterized scorer).
Inference identity: `plm-symmetric-affine-positive-set-v1`.
Evaluator identity: `plm-symmetric-affine8000-screen-v1`.
Publication and checkpoint identities remain separate.
This is an experimental standalone set scorer. Standard decoder forward and
ordinary serving do not execute A/u/b; retain standard_serving_supported=false
in candidate metadata. A passing screen does not change that support boundary.

## Initialization, data and optimization

Start from original parent checkpoint
`e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1`.
Jointly fit fresh zero A/u/b; never resume the trained bilinear child. Keep
parent pretraining (2000 steps) distinct from residual optimization (8000).
Use exactly the historical split
`b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d`,
1637 training queries, 222 validation queries, 1025 products, and seed 1729.
Protected examples remain unopened. Reuse the authenticated train-only label
membership and archived balanced BCE implementation. No validation supervision,
per-query exceptions, graph lookup at inference, fallback or cardinality oracle.

Run exactly 8000 full-batch updates in split order. Fresh AdamW uses lr 0.0003,
betas (0.9,0.999), eps 1e-8, weight decay 0, non-fused; no schedule, clipping or
accumulation. FP32, no autocast, highest float32 matmul precision, CUDA TF32
false, cuDNN TF32 true, deterministic algorithms false. Inherit the accepted
8000-screen environment and data bindings; authenticate dependencies before
loading them. Initial loss must exactly equal 0.003822767175734043.

Validation endpoints are zero replay and final 8000 only, B8/final B6. Keep all
8000 pre-update losses and final post-update loss; check finite scores, losses,
gradients and parameters. Original parameters must receive no gradient and
remain byte-identical. Preserve failure state and never silently restart. Save
one full final child, new identity metadata and fresh optimizer; require exact
model/optimizer reload and all three optimizer states at step 8000. Evaluate
the reloaded child once; no endpoint or seed selection after results.

## Historical comparator and fixed gate

Use accepted `bilinear-budget8000-v1`, not the earlier 2000-update child:
summary `e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8`,
audit `bdf142f48229fbcdb197bce3a019e38c3a65570ae3f8551a9c9a1f14b2198872`,
decision `2cadf9dce7ee36cc4095a10a746cc1608d5ce4437724872c1eec1c88e915c195`,
checkpoint `6979013750eb8f2780917f714192850aae411a48490757236a891fea7e351f91`.
Require query/prompt/label/product-column alignment with the comparator.

Fixed selection is nonself z > 0, ascending entity IDs, retaining raw empty or
oversized outputs as failures. No threshold tuning or truncation. Every set
must have 1..506 distinct nonself products. Passing requires all execution,
train-only, frozen-state, replay, reload and independent-audit invariants plus:

- Exact answers strictly greater than 216/222.
- Macro F1 at least 0.9998843626799785.
- Group exact counts at least COLOR 103, single-TYPE 51, dual-TYPE 62.

Report FP/FN, group metrics, paired gains/losses and strict separation. These
additional metrics cannot compensate for a failed gate. Seed 1729 has already
informed development; this is not an unbiased or protected generalization test.
The larger hypothesis class contains the older scorer at u=b=0, but optimization
and validation quality can still regress.

## Verification, provenance and stopping

Before the one production attempt, freeze plan, recipe, runner and relevant
helpers; pass synthetic CPU tests and independent methodological review. Test
explicit-loop scoring, zero reduction, symmetry, dimension isolation, masking,
finite gradients, only-three-parameter updates, input rejection, loss callback,
checkpoint/optimizer reload, complete failure receipts, immutable output and
gate boundaries. An independently authored auditor must reconstruct membership,
metrics, lineage and gates without importing primary prediction/gate arithmetic.
It audits saved artifacts and does not repeat CUDA training.

LM Studio stays offline. Use one coordinated GPU campaign at a time. No
production training until primary implementation and independent audit contracts
are ready, tested and hash-bound. Preserve any failed attempt; evidence repairs
need new identities. Do not change this plan after candidate execution.

Register any completed final checkpoint, including a gate-rejected one, after
the current 37-checkpoint inventory. Export all 8000 loss rows as four exact
2000-row chunks, each under 512KiB. Update teaching notes, logs, registry, model
register and PLM-findings. Assess PAPER.md applicability: this candidate remains
unpromoted; archive/index its evidence unless an accepted conclusion changes the
current paper. Hand completed audited results to the existing Luna Max publisher.
Hashes identify local weights; they do not establish durable backup, oracle
parity or a serving/energy advantage.
