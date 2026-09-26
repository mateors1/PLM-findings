# Affine v2 lifecycle failure

The corrected parent admission passed, but the parent evaluator call omitted a
required helper. Execution stopped before the first prediction or update.
Independent audit verifies zero updates and all 93 diagnostic tensors matching
the original parent. The local diagnostic payload is excluded from this bundle.
There is no new trained checkpoint or model-quality conclusion.

See [the report](../../../papers/30-affine-lifecycle-failure.md),
[manifest](manifest.json), and [checksums](SHA256SUMS.txt).
Exact bytes and original path/hash identities are preserved. These copies do
not claim portable replay or durable weight backup. V1 is a separate failure.
