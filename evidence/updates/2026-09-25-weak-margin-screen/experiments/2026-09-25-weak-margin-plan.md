# Weaker hardest-boundary margin — declared screen v1

Date: 2026-09-25. Declared before this campaign's training or answer evaluation.

## Hypothesis and provenance

The earlier m=1, lambda=.1 screen failed: fresh control 192/222 exact versus
treatment 6/222. Its decision remains unchanged. The subsequent training-only
gradient diagnosis found that .1 times the margin gradient initially exceeded
the relation projection's symmetric-BCE gradient by a mean factor 6.207. At
failed final weights, gradients locally opposed each other. These observations
motivate a separate coefficient intervention; they do not prove the optimizer
mechanism or identify an optimum.

Fix lambda=.001, a 100-fold reduction, with margin m=1 unchanged. At the same
diagnostic weights and batches, linear scaling would reduce the initial W
ratio to .06207, initial all-embedding/control ratio to .01062 and trained
control W ratio to .48357. These are means of per-batch ratios on the old fixed
states, not predictions about gradients or quality along a new trajectory.
The hypothesis is that weaker pressure can improve boundary separation without
destroying the useful membership representation. It can fail by being harmful,
too weak, or irrelevant to complete selected answers.

Pin the accepted diagnostic portable report SHA256
`d44e1c656a09f53f1b32f1495b9a39ebc35a03085b3fb0639dc5bdaa0b4c87ad`
and acceptance SHA256
`a66863691b2573e5fa8e1a44549b561ab2b46aaece9033e2dba64e8161700aed`.
Pin the failed screen summary SHA256
`39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701`.
Its archived executing helper is SHA256
`a5d50dc4b57277bde1cefb1cf533b4958b28b04eb515182f855b3a4aa81311b8`.
Reusable identity, training receipt, evaluation and gate functions may be
imported from that authenticated helper; no old globals or files are changed.
The new runner and independent auditor receive their own executing hashes.

## Fixed paired recipe

Use the authenticated original seed-1729 continuation-control run receipt and
its named disabled-margin configuration migration, as in the previous screen.
Original receipt SHA256:
`ec6c8e358586efa5808591ca0112ea13b4e52cae4b9891ccac82c299332b870d`;
original resolved config hash:
`6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148`.

Train both arms afresh from seed 1729, sequentially, for exactly 2000 steps.
Use BF16, batch32, accumulation1, AdamW learning rate .0003; CE first-target,
prompt BCE and symmetric BCE weights1, continuation weight0. Every other
field remains the original recipe's value after the explicit migration.
Only run names and margin coefficient differ between arms; m=1 in both.

- Control: `national_dex_rank_margin_weak_control_s1729_v1`, lambda=0.
- Treatment: `national_dex_rank_margin_m1_w0001_s1729_v1`, lambda=.001.

No resume, warm-start, sweep, alternative margin, selected checkpoint, hidden
early stop, or retuning within this campaign. Both trainings finish before
either arm's generated answers are evaluated. The fresh control is the paired
comparator; neither historical control is substituted. Retain all resulting
checkpoints even on failure. GPU jobs remain sequential.

## Evaluation and unchanged acceptance rule

Use the same 222 ordered validation queries and inference policy as the previous
screen: five-token prompts, four guided paths at alpha16, unique constrained
cached decoding, four originals plus six pair unions, canonical additive score
and original tie order; FP32 batch8/final6, max_new_tokens507. Expected targets,
answer counts and graph attributes never enter prediction or selection.
Analysis-only COLOR/TYPE_single/TYPE_dual metadata must authenticate query order
and expected sets against the corpus. The protected 191-query test is unused.

Save every source path, slot score/selection and same-shape membership vector.
Report exact sets, macro precision/recall/F1, group metrics, exact gains/losses,
strict separation, direct zero-threshold membership and training diagnostics.
No HTTP, throughput, energy, Experiment B or generalization claim is produced.

The single-seed screen passes only when all these conditions hold:

1. Both arms' source paths are valid, terminated and eligible; selected sets
   are eligible; identities and an independent saved-evidence audit pass.
2. Treatment exact count strictly improves, macro F1 does not decrease, and
   exact count does not decrease in any of the three analysis groups.
3. Treatment strict-separation count strictly improves.

Failure stops this variant before other seeds. A passing screen permits a
separately executed fixed replication at seeds1730/1731 with fresh paired arms,
the same recipe and every validity/provenance/audit requirement. Acceptance
across three seeds requires exact/F1 nonregression per seed, strict pooled exact
improvement, pooled group exact nonregression, separation nonregression per
seed and strict pooled separation improvement. No automatic default promotion:
any accepted new model family still requires application verification.

This is an adaptively chosen development experiment after earlier validation
results. Its pass is not an unbiased estimate of generalization; all failures
remain in the record and the final test is not used to choose this coefficient.

## Execution contract

Core source/config/runtime must match the previously verified margin feature
and its CPU/default-equivalence/GPU-smoke receipts. Reuse those receipts only
after checking their executing source, effective environment and raw hashes.
Run focused new-runner and independent-auditor tests before this campaign;
archive their stdout and tested script hashes. A .001-specific synthetic smoke
is optional because the same positive-weight objective path is unchanged.

Freeze and snapshot the new runner, authenticated helper, this plan, original
recipe, effective arm configs, CPU/GPU prerequisite receipts and source/config
archives. Refuse existing outputs and run directories, rehash inputs/source
throughout, and record cleanup errors. Record a training seal before evaluation.
The primary runner leaves acceptance false. Independent reconstruction must
check recipe comparability, every saved path/score/metric and the fixed gate;
a separate owner decision binds its result. Preserve raw failures unchanged.
