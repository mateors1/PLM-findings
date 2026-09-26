# Bilinear 2000-update recipe: two additional parent seeds

Declared 2026-09-25 before new neural execution. Campaign
`bilinear-seed-replication-v1`; Pokemon TYPE/COLOR SAME only. This is replication
across two existing parent training seeds, not a new corpus split, protected
evaluation, architecture search or serving integration.

## Question and fixed scope

The accepted seed-1729 budget screen reaches 207/222 exact validation sets and
passes its fixed gate. Test the same 2000-update zero-residual recipe on original
parents 1730 and 1731. Execute both in that order, in separate sequential GPU
processes, even if the first completes with a failed quality gate. Do not stop
based on quality, choose a seed, change a budget or retry on measured results.
Stop and preserve evidence on execution/invariant failure; an incomplete campaign
must not be represented as a completed two-seed replication.

Reuse the accepted 1729 result as the historical development/selection seed.
Do not retrain it or count it as a fresh replication. Its summary, audit and
owner decision are respectively:

- `c507162902e173964c6fec1446448a074ee8fed41f4acb3a4f6d46d6744907a3`
- `475c50f91891ccd13c180472e2d019913b3e5432ab0a9d3b15162735711aac3d`
- `edaf01b58cd09ad4524123f5f817c64602d0414c4220e1c709487016cf2f89fb`

Bind the complete accepted chain and dependencies before execution. The frozen
budget plan is SHA256
`18c08693790880740b20cf25633b56880392362ffbc6b398cd7069116b26edbe`;
its runner is `f4376d4f2d838938bb9f50ae0cfa16ef319639fa0ce917c9fea3b842c0018930`.
Inherit its scoring, optimizer, numeric, checkpoint and failure contracts except
the explicit seed, initial-loss, evaluator and comparison changes below.

## Authenticated parent and comparator mapping

Use each original `runs/national_dex_continuation_control_s<seed>_v1` parent,
never a residual derivative. The run, sidecar and training-result identities
must agree with actual checkpoint bytes and authenticated width-eight evidence.

| Seed | Parent checkpoint SHA256 | Training config SHA256 |
| --- | --- | --- |
| 1730 | `5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2` | `0eae5252ae9de2d1463992a319c43e695ae8bb4f24bfb2ae94acdf4da6033b31` |
| 1731 | `ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d` | `9e029d5fa280492a382b3b138eb5933a8ef8580c6a95a90deb0c87420c3601b3` |

Accepted width-eight summary/audit/decision hashes remain
`1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183`,
`eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95`,
`70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601`.
Resolve per-seed reports through that authenticated summary:

| Seed | Width-eight report SHA256 | Exact / 222 | Macro F1 | COLOR / single TYPE / dual TYPE exact |
| --- | --- | ---: | ---: | --- |
| 1730 | `0a398f2adb53877a188bf72d444c460398b1ac21e2f4317f23bc2948c0dcb88a` | 205 | 0.9906969338820507 | 103 / 45 / 57 |
| 1731 | `a2004f3a8e98043f7e55fc775e0d33069a1a46a70b13f753a816a0d2ba2d7799` | 197 | 0.9798558083418223 | 103 / 44 / 50 |

Use the shared split hash
`b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d`:
1637 train, 222 validation, 191 protected. All parent configs differ only in
run name and training seed; split seed remains 1729. Require exact query,
prompt, truth, group and product-column alignment across seed reports. Use
the same authenticated source/config ZIPs as the accepted budget experiment.
Training membership must reproduce
`0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad`.

## Same learned function and training budget

Keep architecture `plm-frozen-symmetric-bilinear-residual-v1`, objective
`plm-bilinear-residual-balanced-bce-v1` and the exact dense prediction rule.
Import unchanged scorer/fitting/evaluation helpers by pinned hash
`5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee`.
Never monkeypatch frozen modules or silently substitute live core code.

For each seed load its matching original parent, freeze all 93 original tensors,
attach zero FP32 A[2,256,256], and create a fresh AdamW optimizer. Use that seed
for the execution RNG. Exactly 2000 full-batch balanced-BCE updates train only
A. Learning rate0.0003, betas(.9,.999), epsilon1e-8, weight decay0, nonfused,
no scheduler/clipping/accumulation, FP32 with autocast off, highest matmul
precision, CUDA matmul TF32false, cuDNN TF32true and deterministic_algorithmsfalse
remain unchanged. Training logits remain [1637,1025]. No new examples or labels.

At zero A, replay every matching original parent validation logit exactly in
B8/finalB6 against that seed's authenticated width-eight vectors and archived
model forward. Compute that parent's full-training BCE directly from its frozen
parent head, then require exact equality of the actual zero-A head logits/loss.
Record both values. Do not require the different parents to reproduce the
1729 initial loss. These are training measurements, not an optimization selector.

Save only each final full94-state checkpoint plus optimizer/RNG/metadata.
Require all original tensors byte-identical to that seed's parent, changed A,
exact fresh reload of all tensors and optimizer, and optimizer/global_step2000.
Keep parent pretraining2000 and residual updates2000 separate. Bind seed,
parent hash, recipe, campaign, evaluator, new runner and unchanged scorer hashes
in config/identity/metadata. The inherited decoder forward still ignores A.

## Evaluation and replication gate

New evaluator/campaign contract: `plm-bilinear-seed-replication-v1`. This names
the per-seed comparator mapping and campaign decision. Membership arithmetic
and the inference policy do not change. Evaluate only each final child once on
the 222 validation queries, after initial parent/zero-A verification. No
intermediate validation, threshold search, cardinality oracle, truncation,
fallback, group routing or protected prediction.

For each fresh seed require all replay/frozen-state/reload/execution invariants,
222 serialization-compatible outputs with1..506 products, exact count strictly
greater than its own width-eight count, macro F1 at least its own width-eight
F1, and no regression in any of its own width-eight group exact counts.
Thus seed1730 requires >205 and floors103/45/57; seed1731 requires >197 and
floors103/44/50. Require BOTH fresh seeds to pass. A pooled gain cannot rescue
a failed seed or group. The already accepted 1729 gate is reused unchanged.

Report each fresh child's dense-parent and matching-width-eight paired gains
and losses, group metrics, membership errors and strict separation. Also report
three-seed totals/means with 1729 explicitly labeled historical selection seed,
and fresh-two-seed totals separately. These are 666 model-query observations
over 222 shared queries, not 666 independent test examples. No significance,
generalization or oracle-parity claim follows from this small adaptive sample.

## Tests, audit, chronology and publication

Freeze plan, explicit seed recipe and new orchestration after focused synthetic
CPU tests before any actual model execution. Cover seed/parent/recipe mapping,
initial-loss handling, source identities, per-seed gates, all-seed conjunction,
query alignment, final checkpoint steps, immutable artifacts and partial failure.
Reuse unchanged scorer arithmetic tests. Do not duplicate scorer implementation.

Record separate terminal completion receipts for both sequential runs. A
separately authored auditor must reconstruct per-seed labels, metrics, gate,
lineage and optimizer state without importing primary prediction/gate code.
Its final aggregate audit must bind the accepted 1729 chain, both complete
fresh audits, ordered seed coverage and all pooled arithmetic. It does not
repeat CUDA training. Keep owner evidence acceptance separate from quality.

Register every completed final child, including quality failures, additively
to the current31-checkpoint inventory. If both train successfully the count is
33, not34; reused1729 creates no checkpoint. Do not claim local hashes imply
durable remote weight archival. Update Lesson44, logs, registry, model register,
PLM-findings and the existing Luna Max publication task with immutable completed
evidence. LMStudio stays offline; one GPU campaign at a time. No serving or
default promotion, protected test or additional tuning is authorized by a pass.
