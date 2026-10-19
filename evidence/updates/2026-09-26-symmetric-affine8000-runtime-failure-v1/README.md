# Affine v1 runtime admission failure

The production attempt terminated before training because the loader compared
checkpoint TrainConfig with enclosing RootConfig. The failure-only audit passed;
the owner accepted this software diagnosis, with no affine quality conclusion.
There were zero optimizer updates, no predictions and no new checkpoint.

See [the explanation](../../../papers/29-affine-runtime-admission.md),
[manifest](manifest.json), and [checksums](SHA256SUMS.txt).
Evidence copies preserve exact source bytes. Paths/hashes inside receipts retain
the original workspace identities and do not imply portable replay or weight backup.
The original model payload is deliberately absent. Scientific identities and
fixed gates remain unchanged for the separately declared runtime repair.
