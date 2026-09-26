# Symmetric affine v2: evaluation call failure

Status: production call terminal; independent failure audit passed and owner
accepted the failure evidence. This is an
implementation failure with no affine model-quality result.

V2 repaired the configuration-schema comparison from v1. Both restricted
metadata preflights passed. The reviewed runner/scorer passed 204 CPU tests;
the reviewed independent auditor passed 119. Actual parent admission then
passed, including the original checkpoint identity and all 93 tensor hashes.
The model moved to CUDA, but no forward pass was reached.

The parent evaluation call omitted the `mean` helper. Its required parameters
are `model, records, wide_rows, mean, helper, path, affine, bilinear`. Python
binds positional arguments in order, so the later values shifted left and it
reported missing `bilinear`. The error occurs before the evaluator body runs.
The child evaluation call has the correct arguments. Independent static review
checked 269 resolved calls across the runner and archived helper APIs and found
this sole arity mismatch. Arity checks cannot establish data/schema correctness.

The actual terminal exit was 1 (root-observed chunk `47269c`). The training
report has zero updates and an empty history. No residual parameters were
attached, optimizer created, predictions saved or trained child produced. A
diagnostic parent-state payload was saved locally with resume disabled; it is
not a new trained checkpoint and is excluded from publication. Inventory stays37.

| Artifact | SHA256 |
| --- | --- |
| Executed runner | `aab0e064a9c9729af4926626c671f28905d90aa6c29eff0cd10737222d911314` |
| Failure summary | `3939855979f2418f2ef2faefae2823d5a0c90b32b30ae106f963ce2c1bcab6b1` |
| Training report | `2ae347a9a4d38ee94c14c41d0168ae4227ce9ba2f8b6f48e8c3d2dbd6534c33f` |
| Stdout | `86dd0916468310ad9649ba8a7d2e9b4a0eef9f2f9790af045054ecaff7098143` |
| Local diagnostic parent state | `546e4d5274cb8f581260d690e0ae8e0bc0a898f33fe50a2510ff5a7d95c9e746` |

Evidence remains in `runs/learning/symmetric-affine8000-v2/`. No frozen v1/v2
file is repaired in place. Separate v3 preparation must test the full synthetic
production lifecycle through parent evaluation, zero replay, fitting, save,
reload, child evaluation and gating, with actual helper signatures. Expensive
training may be explicitly simulated in that integration test; it cannot stand
in for the later real 8000-update measurement. This extends test coverage while
keeping the scientific architecture, seed, objective, optimizer and gate fixed.

Independent failure-audit SHA256: `159a5fe0abfc143b49dd30d96635aa9f216f6a64de897a8760a34e8916b85d8a`.
Owner decision SHA256: `ab28fd742b3921b47811b8cf95883773be01248a9afd795f0aabcddbccda4c8e`.
The failure auditor passed 13 synthetic tests, reproduced the call-binding error,
and inspected the diagnostic payload on CPU without initializing CUDA. All 93
state tensors match the original parent byte-for-byte, and optimizer state is
absent. Root independently rehashed all 105 bound inputs before acceptance.
