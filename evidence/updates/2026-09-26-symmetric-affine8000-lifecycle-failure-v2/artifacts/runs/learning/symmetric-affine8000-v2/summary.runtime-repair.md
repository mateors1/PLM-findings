# Symmetric affine pilot: separately versioned runtime repair v2

This declaration authorizes preparation of campaign `symmetric-affine8000-v2`
with evaluator `plm-symmetric-affine8000-screen-v2`. It does not authorize an
unreviewed execution or overwrite the failed v1 attempt.

## Preserved failure and diagnosis

Campaign `symmetric-affine8000-v1` stopped before parent runtime return, scoring,
residual attachment or optimization. Its frozen runner SHA256 is
`a9899d4c494198afcf35346903f93bc39d5771d60a6711dc164d12a42ad11568`.
The original checkpoint was loaded on CPU; no candidate checkpoint was produced.
Preserve every v1 source, test, recipe, receipt and failed output unchanged.

Read-only diagnosis `symmetric-affine8000-failure-diagnosis-v1` has report SHA256
`6fafc78ad3bb24c4f9762e7516639d6ff73adb70044816bbf8c50df3d721d696` and observed
execution receipt `13eb1ccb798a20a2cfd017b39c42896738c1418c300da23f4f2d57f3847a76ef`.
All nine admission predicates were evaluated independently. Eight passed; the
failing comparison incorrectly equated checkpoint `config` (TrainConfig) with
`run.json.config` (RootConfig). The authenticated checkpoint configuration exactly
equals `run.json.config.train`. Payload and sidecar metadata agree; the resolved
model configuration, original hash, global step, objective and identity checks pass.
Archived `Trainer` passes its TrainConfig to `save_checkpoint(config=self.config)`.

## Narrow repair and unchanged scientific contract

Compare authoritative checkpoint `config` to the authenticated original run's
`config.train`. Keep exact equality; do not normalize away differences or drop
fields. Report each parent admission predicate separately. Preserve all other
checkpoint hash, step, payload/sidecar, complete experiment identity, split/corpus,
training objective, resolved model configuration and evaluator checks.

The immutable scientific plan remains
`docs/experiments/2026-09-26-symmetric-affine-plan.md`, SHA256
`d102c6d00ab536b89c0a0021097976fb997831184a4e4c2db82d172a4c61a19a`.
The selected-partition amendment remains SHA256
`bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f`.
Architecture `plm-frozen-symmetric-affine-residual-v1`, objective
`plm-affine-residual-balanced-bce-v1`, inference
`plm-symmetric-affine-positive-set-v1`, and input adapter
`plm-authenticated-partition-membership-v1` are unchanged.

Start from the same original parent with fresh zero A/u/b and fresh AdamW. Keep
seed 1729, selected train/validation membership and order, FP32 numerical settings,
8000 updates, divisor 16, validation endpoints, exact-zero replay, initial loss,
frozen 93-tensor checks, complete 96-tensor child, optimizer checks and exact reload.
The fixed gate remains exact answers >216/222, macro F1 >=0.9998843626799785 and
COLOR/single-TYPE/dual-TYPE exact floors 103/51/62. No endpoint, threshold, seed,
optimization, model architecture or acceptance criterion changes are authorized.
No candidate-quality result informed this repair. Standard serving remains unsupported.

## Separate implementation and admission

Use `scripts/refit_symmetric_affine_v2.py`,
`tests/unit/test_symmetric_affine_runner_v2.py`,
`configs/experiments/symmetric_affine8000_v2.json`, and
`runs/learning/symmetric-affine8000-v2/` for the new attempt. Bind this declaration
in recipe, summary, checkpoint implementation/training metadata, primary test
receipt and independent auditor readiness/test receipts. Preserve independent
candidate auditing and immutable outputs. No source or receipt is reusable under
an old hash after editing.

Use realistic synthetic metadata with distinct RootConfig and TrainConfig shapes.
Prove the corrected equality admits only matching TrainConfig, and separately
reject changed optimizer settings, sequence length, update budget and other parent
identity fields. Retain the before-open allowlist tests, selected-partition firewall,
loss/gradient permutation invariance and complete affine lifecycle checks.

Freeze the separately versioned implementation and independent auditor only after
synthetic tests and independent review pass. Root coordinates any later single
attempt and GPU reservation. Do not run a success audit against failed v1 outputs;
the failure diagnosis and failure audit have separate identities. No corpus, graph
or protected partition may be opened during v2 admission or execution.
