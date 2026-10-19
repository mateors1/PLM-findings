# Bilinear balanced-BCE residual: fixed 8000-update screen

Declared before new neural execution. Campaign `bilinear-budget8000-v1` tests
one seed, 1729, within Pokemon TYPE/COLOR SAME. This is an adaptive validation
screen with a single fixed endpoint, not a protected test or a budget sweep.

## Why return to balanced BCE?

The worst-member sibling completed but failed its declared F1 and single-TYPE
floors despite reaching 208 exact answers. Its accepted evidence decision is
`6dd49574b54b0bea9d64c79650eb57c1090157254d8e2fdaec0f7dcb8cf5b8a4`.
Do not promote that rejected objective or treat its higher exact count as a win.

Return to the accepted balanced-BCE 2000-update recipe. Its saved training loss
continued to fall: update1500=0.00005501357009052299,
update1900=0.00003798709440161474 and update2000=0.000034944067010656;
final post-update loss=0.00003491543247946538. These observations motivate more
optimization, but do not establish underfitting or predict better validation.
The next fourfold budget, 8000, extends the declared 500/2000 budget sequence.
Choose this endpoint before results; no shorter/longer endpoint is selected.

Historical balanced-BCE2000 comparator, not a newly executed control:

- Summary `c507162902e173964c6fec1446448a074ee8fed41f4acb3a4f6d46d6744907a3`.
- Audit `475c50f91891ccd13c180472e2d019913b3e5432ab0a9d3b15162735711aac3d`.
- Decision `edaf01b58cd09ad4524123f5f817c64602d0414c4220e1c709487016cf2f89fb`.
- Checkpoint `8d9ddadc6d63f7ab8f767fb484193dbdfdf1b9fb369fad1dab38fef8bd028c2c`.
- Recipe `e454c52a315acb389b2bda3c126367e13df63b96230f9f698517705b531d51da`.
- Runner `f4376d4f2d838938bb9f50ae0cfa16ef319639fa0ce917c9fea3b842c0018930`.

It achieved 207/222 exact, F1 0.9997456353806234 and group exact counts
COLOR103, single-TYPE50, dual-TYPE54. Historical results remain immutable.

## Fixed treatment and inherited contracts

Inherit `2026-09-25-bilinear-budget2000-plan.md`, SHA
`18c08693790880740b20cf25633b56880392362ffbc6b398cd7069116b26edbe`,
except the explicitly replaced budget, campaign/evaluator and comparator below.
Use the same original parent
`e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1`,
exact zero A[2,256,256] and fresh AdamW. Never resume either trained derivative.
All93 original state tensors remain byte-identical; only A may change.

Architecture `plm-frozen-symmetric-bilinear-residual-v1`, objective
`plm-bilinear-residual-balanced-bce-v1` and scorer
`5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee`
remain unchanged. Use frozen archived `_prompt_set_loss` as the explicit loss
callback to the frozen bilinear fitting loop. No worst-member loss or mixture.

Run exactly8000 full-batch updates over the same1637 training queries in split
order. Keep AdamW lr0.0003, betas(0.9,0.999), eps1e-8, decay0, non-fused;
no schedule, clipping or accumulation. FP32, no autocast, highest matmul precision,
CUDA matmul TF32=false, cuDNN TF32=true, deterministic algorithms=false.
Initial original-parent mean BCE must exactly equal0.003822767175734043.
Require exact zero-A replay of all222 validation vectors at B8/finalB6 before
fitting. Preserve the previous archived runtime and unchanged scorer order.

Record every pre-update loss, finite score/gradient/parameter check and final
post-update loss. Report equality of the first2000 scalar pre-update loss records
against the historical run as a descriptive execution diagnostic only, including
the mismatch count and first mismatch index if any. Separately compare the old
final post-update2000 loss with the new pre-update2001 loss. These are aligned
measurement points; do not compare it to new pre-update2000. This diagnostic does
not assert identical intermediate weights, prove deterministic execution, change
the gate or authorize a retry. A mismatch is recorded without aborting or tuning.
Validation endpoints are [0,8000]: update0 is required parent/zero-A replay,
and8000 is the only fitted-child validation endpoint.

Save one final full94-state child with fresh optimizer state at global_step8000.
Parent pretraining remains2000 steps; record parent and residual counts separately.
Exact model/optimizer reload is required. Bind runner, scorer, objective, recipe,
data/split, environment and new evaluator `plm-bilinear-budget8000-screen-v1`.
The inherited decoder still ignores A; ordinary serving stays unchanged.

## Evaluation and fixed selection rule

Evaluate the reloaded final child once on222 validation queries. Preserve the
z>0, nonself, ascending-ID dense output policy and all raw empty/oversized sets;
no fallback, repair, threshold tuning, oracle cardinality or output truncation.
Require all execution/train-only/frozen-state/replay/reload invariants, complete
independent audit and all222 sets containing1..506 distinct nonself products.

Replace the older >201 gate with all these historical2000 comparator floors:

- Exact answers strictly greater than207.
- Macro F1 at least0.9997456353806234.
- Group exact counts at least COLOR103, single-TYPE50 and dual-TYPE54.

Report aggregate/group exactness, F1, FP/FN, strict separation, serialization
and paired gains/losses against the original dense parent, historical width-eight
predictor and historical balanced-BCE2000 sibling. Check identical query order,
labels and split for every historical comparison. Extra comparisons cannot
compensate for a failed gate. No new gate is introduced after observing results.

## Evidence, limits and stopping

Freeze new runner/recipe/tests before the one attempt. Reuse authenticated
scorer, fit and metric helpers without monkeypatching old modules or invoking
old objective/evaluator-specific execute or checkpoint validators. Own the new
8000-step identity/validator. Test budget and gate boundaries, recipe differences,
loss callback, inherited masks/gradients, checkpoint/optimizer steps, helper
identity separation, historical pairing, partial failure and immutable outputs.

An independent saved-evidence auditor checks the complete8000-step trace,
checkpoint/optimizer lineage, membership, metrics and fixed gate without importing
primary prediction/gate arithmetic. It does not repeat CUDA training. Preserve
failed attempts; repairs require new identities. Stop at8000 even if a result is
close to passing. A pass permits separately declared replication, never automatic
promotion or protected-test use. A failure remains publishable evidence.

No validation-derived supervision, external datasets, additional seeds, model
architecture change or GPU concurrency. LM Studio stays offline. Register any
completed final child after the current34 final checkpoints, including a rejected
one; retain checkpoint/inference/evaluator/publication identities separately.
Split the8000-row portable loss trace into four exact2000-row artifacts so each
published file remains under512KiB. Record indices, source trace hash and checksums
so concatenation reproduces the complete trace, without downsampling.
Update Lesson47, logs, registry, model register and PLM-findings, then hand the
completed evidence to the existing Luna Max publisher. Local weight hashes do
not establish durable archival, oracle parity or serving/energy advantage.
