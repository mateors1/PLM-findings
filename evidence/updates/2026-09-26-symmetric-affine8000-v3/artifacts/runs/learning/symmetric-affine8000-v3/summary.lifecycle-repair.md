# Symmetric affine pilot: lifecycle wiring repair v3

Prepare campaign `symmetric-affine8000-v3`, evaluator
`plm-symmetric-affine8000-screen-v3`, as a separate attempt. Preserve all executed
v1 and v2 sources, tests, recipes, receipts and failed outputs unchanged.

The preceding runtime repair is
`2026-09-26-symmetric-affine-runtime-repair-v2.md`, SHA256
`8ac4fbc66d40bf26e249d0ee9a0cd9117d06653e762552629c98e2b53a2c5a18`.
Its TrainConfig-versus-RootConfig correction remains in force. Frozen v2 runner
SHA256 `aab0e064a9c9729af4926626c671f28905d90aa6c29eff0cd10737222d911314`
passed parent checkpoint admission but failed at the parent evaluator call before
its body executed. The call omitted the `mean` helper argument, shifting subsequent
arguments and leaving required `bilinear` unbound. The child call already supplies
`mean`. No prediction, affine attachment or optimizer update occurred. Preserve the
v2 diagnostic state as failure evidence; it is not a candidate checkpoint.

## Repair and scientific invariants

Supply `mean` explicitly at the parent evaluator call and verify both evaluator
calls against their actual signatures and semantic argument roles. Keep the v2
TrainConfig repair, every checkpoint/identity check, unopened protected-data
boundary and immutable output policy.

The scientific plan remains SHA256
`d102c6d00ab536b89c0a0021097976fb997831184a4e4c2db82d172a4c61a19a` and input
amendment remains SHA256
`bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f`.
Architecture, objective, inference, input-adapter and frozen affine scorer identities
remain unchanged. Retain original seed-1729 parent, fresh zero A/u/b, fresh AdamW,
FP32 settings, divisor 16, selected 1637/222 memberships/order, exactly 8000 updates,
zero/final validation endpoints and fixed gate (>216 exact; F1 >=0.9998843626799785;
COLOR/single-TYPE/dual-TYPE exact floors 103/51/62). No candidate quality has informed
this repair; no threshold, model, optimization or selection change is authorized.

## Verification before another attempt

Use separate `scripts/refit_symmetric_affine_v3.py`,
`tests/unit/test_symmetric_affine_runner_v3.py`,
`configs/experiments/symmetric_affine8000_v3.json`, and
`runs/learning/symmetric-affine8000-v3/`. Bind this declaration in recipe, summary,
checkpoint implementation/training metadata, primary test receipts and independent
auditor readiness/test receipts. Keep the preceding declaration by reference above.

A synthetic full `_execute` integration must pass through parent evaluation,
attachment, zero replay, fitting, save, reload, child evaluation, gate and final
reports. Preserve real evaluator signatures and their bodies, actual archived
checkpoint save/load and optimizer state roundtrip, child/optimizer validation,
byte equality and fixed gate. Mock only runtime/environment acquisition and costly
numerical work with exact signatures, checking argument identities and shapes.
If a bounded fitting double simulates the 8000-update trajectory, label that scope
explicitly: it proves orchestration, not 8000-step numerical equivalence or model
quality. Keep existing direct scorer tests for numerical behavior.

Additionally inspect every statically resolvable internal runner call with AST and
`inspect.Signature.bind`; fail on any binding mismatch and state the coverage.
Check evaluator argument roles, not only argument count. A negative control must
reproduce v2's missing-mean failure. Retain existing configuration-schema mutation,
selected-partition, before-open admission and loss/gradient invariance tests.

Independent review must clear the new implementation and integration evidence
before root seals receipts and coordinates any new production attempt. No automatic
retry, production run, success audit of failed outputs, serving promotion, or
checkpoint archival claim follows from this declaration alone.
