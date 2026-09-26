# Symmetric affine v1: parent admission diagnosis

The failed attempt stopped because the checkpoint stores **TrainConfig** in
`config`, while `run.json` stores **RootConfig** in `config`. The adapter compared
those different schemas. The checkpoint configuration exactly equals
`run.json["config"]["train"]`.

The diagnosis evaluated all nine predicates independently. Eight passed; only
whole-root configuration equality failed. Model configuration (including resolved
vocabulary size and sequence length), original checkpoint hash, global step,
objective, evaluator and corpus identity all passed. CPU checkpoint payload
metadata exactly matches the authenticated sidecar for every compared field.

The diagnostic loaded the original checkpoint on CPU. CUDA remained uninitialized;
no forward pass, optimizer step, candidate retry, corpus, graph or protected-data
read occurred. The original attempt had loaded its parent into a local CPU model,
but failed before returning the runtime to the outer function. Consequently the
outer failure receipt correctly has no model or optimizer handle.

See [diagnosis.json](diagnosis.json) for all predicates, thirteen exact input hashes,
source pointers, and recommended repair, and [execution-receipt.json](execution-receipt.json)
for observed exit zero. Frozen v1 files and failed outputs remain unchanged.

The test gap was a synthetic fixture that used the same configuration object in
both locations. A separately declared v2 should compare payload configuration to
the authenticated train subconfiguration, preserve every other identity check,
and add realistic distinct root/train fixtures with tampered-field rejection.
This is an execution-contract defect, not a scientific failure of affine learning.
