# Bilinear residual with worst-member supervision

Declared before new fitting or validation. Campaign `bilinear-worst-boundary-v1`
tests one seed, 1729, within Pokemon TYPE/COLOR SAME. It is an adaptive validation
screen, not an independent confirmation or a protected-test result.

## Hypothesis and comparison

The completed saved-error diagnostic found both sign-placement and ranking
failures. On the two fresh bilinear children, 15 of 24 wrong answers have strict
separation and nine have overlapping/tied ranges. That diagnosis does not prove
that average loss caused the errors. It motivates testing an objective that
directly penalizes the weakest true member and strongest false member.

The earlier projection-only worst-member screen failed its strong quality gate:
128/222 exact versus 201. Its strict-separation count fell165 to160 relative to
mean fitting, and mean BCE rose despite the new loss falling. Do not conceal
that negative result or claim the loss has already worked. Here the parameterization is the more expressive bilinear
residual, and the controlled comparison is to its same-budget balanced-BCE sibling.
Equal update count and AdamW settings do not mean equal optimization difficulty.

Use the accepted seed-1729 balanced-BCE 2000-update result as the historical
comparator, without rerunning or resuming it:

- Summary `c507162902e173964c6fec1446448a074ee8fed41f4acb3a4f6d46d6744907a3`.
- Audit `475c50f91891ccd13c180472e2d019913b3e5432ab0a9d3b15162735711aac3d`.
- Decision `edaf01b58cd09ad4524123f5f817c64602d0414c4220e1c709487016cf2f89fb`.
- Checkpoint `8d9ddadc6d63f7ab8f767fb484193dbdfdf1b9fb369fad1dab38fef8bd028c2c`.
- Recipe `e454c52a315acb389b2bda3c126367e13df63b96230f9f698517705b531d51da`.

The sibling achieved 207/222 exact, macro F1 0.9997456353806234, with group exact
counts COLOR103, single-TYPE50 and dual-TYPE54. Its acceptance remains unchanged.

## Fixed model, data and optimization budget

Inherit the complete architecture, split, numerical arithmetic, replay,
checkpoint and serialization contracts of `2026-09-25-bilinear-budget2000-plan.md`,
SHA `18c08693790880740b20cf25633b56880392362ffbc6b398cd7069116b26edbe`,
except the objective, campaign/evaluator, diagnostic and comparison changes below.
The old objective and quality gate are replaced, not simultaneously required.

Restart from original parent
`e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1`.
Keep all 93 parent tensors byte-identical and initialize only A[2,256,256] to zero.
Retain architecture `plm-frozen-symmetric-bilinear-residual-v1` and the exact
scorer `5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee`.
The full token table is [2049,256]; normalized product features E[1025,256]
have length16. The residual remains E_s^T ((A_d+A_d^T)/2) E_i /16.

Use exactly 2000 full-batch updates on the same 1637 training queries in split
order. Fresh non-fused AdamW: learning rate0.0003, betas(0.9,0.999), eps1e-8,
weight decay0. No scheduler, clipping, accumulation, autocast or budget extension.
FP32, highest matmul precision, CUDA matmul TF32=false, cuDNN TF32=true,
deterministic algorithms=false, seed1729. Parent steps and residual updates are
separate provenance fields. Do not resume any derivative or use validation labels
in training. No protected predictions, external data or additional seeds.

## Sole replacement objective

For each training query q, deduplicate true products T_q and form false products
N_q, excluding the SAME subject from both. Preserve archived control/padding
masking and reject an empty class before reducing scores Z[1637,1025]:

    a_q = min_{i in T_q} Z_qi
    b_q = max_{i in N_q} Z_qi
    L_q = (softplus(-a_q) + softplus(b_q))/2
    L = mean_q L_q

Use the existing `_worst_loss(set_logits, labels, *, excluded_ids)` from frozen
`scripts/refit_worst_boundary_projection.py`, SHA
`8598f110d7cbc60085c7d099a821386cba3566f4264c91939fada5baa0f0e480`.
Its masked amin/amax distributes gradients over exact ties. Import it by checked
hash; do not copy or monkeypatch it. Use it as an explicit callback to the
frozen bilinear fitting loop. No mean-loss mixture, margin or temperature.
`excluded_ids` contains product-column offsets (subject token ID minus1024),
not vocabulary token IDs. Teacher labels remain vocabulary token IDs [B,L].

This convex-in-A loss can be nondifferentiable at ties; convexity establishes
neither finite-budget convergence nor generalization. Extreme-member gradients
may switch abruptly and may sacrifice broader fit. Keep the learning rate fixed
for this experiment; a failed result does not authorize a silent retune.

Before fitting, require exact zero-A replay of all222 validation score vectors
at historical B8/finalB6 shapes. Reproduce original full-training mean BCE
0.003822767175734043 exactly. The new initial worst-member loss is measured and
recorded, not asserted equal to BCE. Record every new-objective pre-update loss,
finite gradient/parameter checks and final post-update loss. Measure mean BCE
only at updates0 and2000, under no_grad, as a separate diagnostic unused for
updates, stopping or acceptance. Preserve complete train membership and identities.

## Final evaluation and gate

Save one final full94-entry checkpoint with optimizer state and new objective
`plm-bilinear-worst-boundary-softplus-v1`; use evaluator
`plm-bilinear-worst-boundary-screen-v1`. Fresh reload must exactly restore all
model and optimizer tensors, step2000, configuration and identity. Standard
serving remains unsupported and unchanged; inherited decoder forward ignores A.

Evaluate the reloaded final child once over222 validation queries, keeping the
fixed z>0, nonself, ascending-ID selection. Do not use oracle counts, fallback,
truncation, threshold fitting, intermediate validation or checkpoint selection.
Require complete authenticated execution, all frozen-state/replay/reload/train-only
invariants and all222 sets containing1..506 distinct nonself products. The new
gate requires **all** of:

- Exact answers strictly greater than207/222.
- Macro F1 at least0.9997456353806234.
- Group exact counts at least COLOR103, single-TYPE50 and dual-TYPE54.
- An independently accepted saved-evidence audit.

Report aggregate/group metrics, FP/FN, strict separation and paired exact
gains/losses against the historical balanced-BCE2000 sibling, original dense
parent and width-eight predictor. Verify query/label/order equality before every
historical comparison. Other comparisons are descriptive; the gate cannot be
relaxed after results. A pass supports separately declared replication, not
automatic promotion. A failure stays published as a negative result.

## Execution and provenance

Freeze plan, recipe, runner, helper snapshots and meaningful synthetic CPU tests
before the one neural attempt. Cover sole-loss callback, masks/ties/gradient
direction, diagnostics without gradients, new objective versus unchanged scorer
identity, new strict gate, immutable outputs and checkpoint/optimizer lineage.
Reuse existing arithmetic tests where applicable rather than duplicating them.
New orchestration must own objective-specific execution/validation metadata;
do not invoke an old objective-specific execute/validator with patched globals.

A separately authored auditor independently reconstructs metrics/gate/pairing
and checks saved losses, optimizer, masks, checkpoint identities and frozen
original tensors. It may reuse authenticated independent helpers, but not the
primary gate or prediction arithmetic. It does not rerun CUDA fitting. Keep
failed attempts immutable; source repairs require a distinct version.

Use one GPU campaign at a time; LM Studio remains offline. Register any completed
final child, including a rejected one, after the current33 entries; retain recipe,
checkpoint, evaluator and publication identities separately. Update Lesson46,
logs, registry, PLM-findings and notify the existing Luna Max publisher with a
completed immutable handoff. A hash binds local weights but is not a backup.
