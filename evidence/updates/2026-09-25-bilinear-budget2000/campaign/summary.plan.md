# Bilinear residual: fixed 2000-update budget screen

Declared 2026-09-25 before new training or validation. Run ID
`bilinear-budget2000-v1`. Pokemon TYPE/COLOR SAME only. This is a separately
declared optimization-budget experiment, not an extension of the completed
500-update run or a protected evaluation.

## Rationale and controlled change

The accepted 500-update evidence gives 198/222 exact validation sets, below
the unchanged >201 gate. Its training pre-update losses at updates 400, 450
and 500 are 0.0003255420015193522, 0.0002822330570779741 and
0.000247796910116449; final post-update loss is 0.0002471808111295104.
Training loss was still falling. That motivates testing more optimization,
but does not predict improved validation exactness or prove underfitting.

Use exactly 2000 full-batch updates, a predeclared fourfold budget. Restart
from the same original seed-1729 parent and exactly zero A, with a fresh
optimizer. Do not load or resume the 500-update derivative. Change no model,
objective, learning rate, threshold, data split, query ordering or gate.
Keep only the final 2000-update checkpoint; do not evaluate intermediate
validation endpoints, select the best checkpoint or extend training on failure.
The historical 500-update output is a comparator, not another executed endpoint.

## Inherited contracts and immutable dependencies

Inherit the arithmetic, architecture, training data, optimizer, zero-start
replay, checkpoint, prediction, audit and publication contracts from
`2026-09-25-bilinear-residual-plan.md`, SHA256
`b1cd8a897e5db7d0dc0adc3e3bd40a87b0ae5830714b63ea86649f1599c9e132`,
except the explicit budget, campaign and receipt changes below. Its numerical
500-update conditions are replaced by 2000, not both required. Its original
parent remains `e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1`.
The new child is a sibling of the 500-update derivative. Parent pretraining
2000 steps and residual fitting 2000 updates must remain separate fields.

Architecture `plm-frozen-symmetric-bilinear-residual-v1`, objective
`plm-bilinear-residual-balanced-bce-v1` and evaluator
`plm-bilinear-residual-screen-v1` remain unchanged. Campaign/recipe identity
changes to `bilinear-budget2000-v1`; `residual_updates` becomes 2000.
All 93 original tensors stay byte-identical. Train only A[2,256,256] using
the exact archived scorer, balanced BCE and fresh AdamW settings. The training
logit tensor remains [1637,1025]; validation remains 222 queries in B8/finalB6.
No protected predictions, additional seeds, oracle cardinalities or thresholds.

Authenticate the completed bilinear chain before execution:

- runner: `5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee`
- summary: `92955bf21ce9dc85ce9806436417436cece618d1be9edd3c400f6d90c96bc0b5`
- audit: `2d8f3d49a4d6e25eaee319f3517b7ec69e139a566a814d4be1f01a44cdbef417`
- decision: `d7d38687e2d099348af61bd52ed62df28211464e4209cb00df43db21d1d93234`
- auditor: `a046a1e45b3c977fc556ecc74358d64e2b9d0463ebf653c7b7f7c86aed61622b`
- child: `1f6e9a593ac5c5c83f0d003f273adc02213845ab94f7d5118b1b55bf5bab43d9`

Bind all historical report dependencies through that accepted chain. Reuse
unchanged scorer/fitting/evaluation helpers by explicit hash-pinned imports.
Do not monkeypatch a frozen module, rewrite old evidence, or substitute live
core code for the authenticated archived runtime. New orchestration must
identify both its own executed bytes and the reused scorer separately in
checkpoint metadata/config/receipts. Inherited decoder forward still ignores A.

## Evidence, gates and failure behavior

Require exact zero-A replay of all original validation logits and initial
full-training loss 0.003822767175734043. Record every pre-update training loss
and finite gradient/parameter check, final post-update loss, all 2000 completed
updates, runtime environment and descriptive timing/memory. Save one full
94-entry checkpoint, optimizer state and identity. Fresh reload must reproduce
all tensor and optimizer values exactly, with global_step 2000 for this fit.

Evaluate final reloaded child once. Report aggregate/group exactness, macro F1,
false positives/negatives, strict separation, serialization compatibility and
paired gains/losses against original dense parent, historical eight-branch
selector and historical 500-update residual. Comparing saved historical output
requires exact query identity/order, split and scorer/recipe compatibility.
It does not independently rerun the older fit or establish identical timing.

Keep the quality gate unchanged: exact sets >201, macro F1 >=0.9799255176742276,
COLOR exact >=103, single TYPE >=50, dual TYPE >=48, all 222 outputs containing
1..506 products and all execution invariants passing. Do not add a post-result
rule, alter thresholds, repair sets or select a different checkpoint. If the
gate passes, declare replication separately; do not automatically promote.

Freeze new plan, recipe, orchestration and focused CPU test receipt before
any actual neural execution. Meaningful tests must cover budget enforcement,
frozen-dependency identities, scorer identity separation, exact optimizer and
checkpoint step handling, historical pairing rejection, partial failure and
immutable outputs, reusing the existing arithmetic tests. A separately authored
saved-evidence auditor must inspect 2000-step loss/optimizer/checkpoint lineage,
metrics and historical pairing without importing the primary gate/prediction.
It does not independently repeat CUDA training. Keep any failed execution
attempts and stop on nonfinite values, drift or invariant failures.

This is another adaptive validation screen. A better result would support the
longer declared fit on this screen; it would not prove convergence, optimality,
generalization, oracle parity or a serving advantage. A worse result remains
publishable and does not authorize a hidden retry. The old 500-update evidence
and its failed gate remain unchanged.

On completed training add the final sibling to the academic register, currently
30 checkpoints, regardless of quality. Keep raw weights local and distinguish
hash identity from durable archive. Update teaching Lesson 43, logs, registry,
PLM-findings and the immutable publication handoff to the existing Luna Max
task. Coordinate one GPU campaign; LMStudio stays offline.
