# Symmetric hardest-boundary margin — declared experiment v1

Date: 2026-09-25. Status: declared before experimental training or evaluation.
The optional implementation is tested separately from its research outcome.

## Hypothesis and scope

The frozen membership diagnosis found rank overlap in 187/666 observations and
81 of the current pair selector's 97 failures. The existing balanced BCE
optimizes class-average membership errors; it does not directly penalize the
weakest true member falling below the strongest nonmember. Test whether a small
hardest-boundary hinge term improves ranking separation and complete selected
answers while retaining the current generator, membership BCE and set selector.
This is not an architectural capacity claim and remains Experiment A.

For FP32 membership logits Z[B,1025], exclude each query subject and derive
deduplicated true-member masks only from supervised target labels. Add

    L_margin = mean_q relu(1.0 + max_negative(z_q) - min_true(z_q))
    L_treatment = L_control + 0.1 * L_margin

The mean includes queries whose hinge is zero. Empty positive/negative classes
are rejected as in the existing BCE. amin/amax distribute gradients among tied
extrema; ReLU has zero gradient at its zero boundary. Version this behavior as
`hard-boundary-hinge-fp32-amin-amax-query-mean-v1`. Margin labels affect training
gradients only; validation diagnostics run without gradients. Keep validation
token CE separate from raw margin, query count and active-hinge fraction.

## Exact treatment and paired control

Freeze margin m=1.0 and coefficient lambda=0.1. No sweep, threshold fitting,
alternate margin, selected checkpoint step or hidden early stopping in this
version. These are initial hypothesis settings, not values inferred to be optimal.

Use the complete saved seed-1729 continuation-control recipe in
`runs/national_dex_continuation_control_s1729_v1/run.json`, authenticated against
its original resolved config hash and checkpoint/training receipts before use.
The original run receipt SHA256 is
`ec6c8e358586efa5808591ca0112ea13b4e52cae4b9891ccac82c299332b870d`;
its training config hash is
`6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148`.
Snapshot that original recipe and its SHA256 in the campaign. Load its named
disabled-margin migration explicitly; do not rewrite its old identity.

Train a fresh control and fresh treatment from initialization under the same
new source/config archive and runtime. No resume or fine-tuning. Both retain the
old architecture, data, split, seed, optimizer and full 2000-step training recipe:
CE first-target weight 1, prompt BCE weight 1, symmetric BCE weight 1,
continuation weight 0, BF16 training, batch 32, accumulation 1, AdamW learning
rate 0.0003. Preserve every other field from the authenticated saved recipe.
Only run names and margin coefficient differ across arms; margin is 1.0 in both.
Source/config/settings must be frozen across both arms and their evaluation.

Stage-one run names:

- `national_dex_rank_margin_control_s1729_v1` (coefficient 0.0).
- `national_dex_rank_margin_m1_w01_s1729_v1` (coefficient 0.1).

Train control then treatment, sequentially on the GPU. Use final step-2000
checkpoints only. Retain both regardless of outcome. Record new recipe/checkpoint
identities and source archives; these are new model replicas, not renamed old
weights. The fresh control is the causal comparison. The historical 191/222
seed-1729 pair result is contextual evidence, not a substitute for that control.

## Fixed validation evaluation

Use the same ordered 222 validation queries and product vocabulary. The protected
191-query test partition remains inaccessible to this campaign. Prediction uses
only five-token prompts and learned scores: four guided first-choice paths,
alpha 16, unique cached constrained decoding, four originals plus six pair unions,
canonical additive set score and original tie order. Generation batch size 8,
final batch 6, FP32 inference and max_new_tokens 507, matching the current policy.
Expected labels, graph attributes and answer sizes must not enter selection.

Save raw source paths, selected slots/sets, all candidate scores and a same-shape
FP32 membership vector per query. Validate vector-derived slot scores exactly.
Group labels COLOR/TYPE_single/TYPE_dual may reuse the authenticated old query
metadata for analysis only, with identities and expected sets checked against
the actual validation corpus. Do not feed those group labels to the predictor.

Report per arm: exact set count, macro precision/recall/F1, all source validity/
termination and selected eligibility, group metrics, pairwise exact gains/losses,
strict membership separation counts and zero-threshold membership quality.
Record token CE, margin loss and active fraction from training diagnostics.
No HTTP throughput, energy or final-test claim is produced by this experiment.

## Staged decision rule, fixed in advance

Stage one is a single-seed screen, not a three-seed result. It passes only if:

1. Every source path is valid, terminated and eligible, and every selected set
   is eligible in both arms; all provenance/identity checks and independent audit pass.
2. Treatment exact count strictly exceeds the fresh control, treatment macro F1
   does not decrease, and exact count does not decrease in any of the three groups.
3. Treatment strict-separation count strictly exceeds the fresh control.

Report gained and lost exact answers even if aggregate gates pass. A failure
stops this variant before replication; preserve the negative result and make no
claim about untested seeds. Do not relax the gate, retune or silently retry it.

If stage one passes, replicate the identical recipe at seeds 1730 and 1731,
with a fresh control and treatment for each. Reuse stage-one seed 1729 openly.
Every replication arm must also satisfy stage-one item 1: source validity,
termination and eligibility, selected eligibility, provenance/identity checks,
and independent audit. These requirements apply to every seed and both arms.
Three-seed acceptance requires exact-count and macro-F1 nonregression in each
seed, strictly higher pooled exact count, pooled group exact nonregression, and
strict-separation nonregression in each seed with strictly higher pooled count.
Independent audit remains required. No automatic inference-default promotion;
any accepted new checkpoint family needs its own application verification.

## Implementation and evidence prerequisites

Before experiment execution, require focused model/config/runtime/trainer tests,
the full test suite, Ruff, strict typing, and an archived-source comparison of
disabled behavior. GPU work is sequential and starts only after checking that
prior jobs are terminal and the device is available. CPU verification does not
stand in for a GPU numerical smoke check.

Old saved root configurations authenticate their original hash before the two
new model fields can acquire paired disabled defaults (0.0,1.0). Partial fields
or coercions are rejected, including payload model metadata. Active checkpoints
must carry the exact auxiliary objective descriptor; disabled checkpoints omit
it. Historical exact training resume remains rejected across changed raw model
configs/identities. New same-code resume must remain exact.

The immutable campaign snapshots its executing script, plan, CPU/GPU smoke
receipts, original recipe, effective arm configs, source/config archives and
runtime. Refuse existing outputs/run directories and bind raw reports/checkpoints
by SHA256; rehash input/source identities after execution. Independently rebuild
sets/scores/metrics, paired gates and recipe comparability from saved raw evidence.
Do not edit failed artifacts or overwrite prior checkpoints. The script may
create training/evaluation outputs, but only a separate audit and owner receipt
can accept the scientific comparison.
