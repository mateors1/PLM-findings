# Symmetric affine v1: failed runtime admission

Status: execution terminal; predicate-level diagnosis complete; independent
failure audit passed and owner accepted the failure evidence. This is an implementation failure, not an evaluated
negative result for the affine hypothesis.

The fixed `symmetric-affine8000-v1` attempt passed 192 runner/scorer CPU tests,
independent source review and selected-input metadata preflight. The independent
auditor passed 101 synthetic CPU tests. The production call terminated with
exit code 1 at the original-parent checkpoint admission before optimization
or predictions. The training report contains zero updates and no loss history.
There is no child checkpoint, new quality score or promotion. Inventory stays 37.

## Cause and missing coverage

The archived trainer serializes `TrainConfig` into the checkpoint. The original
run report contains the enclosing `RootConfig`, including model, data, evaluation
and training sections. The new loader compared the two dictionaries directly:
`checkpoint.config == run.config`. Their schemas differ. The correct comparison
is `checkpoint.config == run.config["train"]`, with the full experiment identity,
resolved model configuration and all remaining provenance checked separately.

CPU-only diagnosis evaluated all nine admission predicates, including those
short-circuited in the original failure. Only the full-root configuration
comparison failed. Payload and sidecar metadata agree, the original checkpoint
hash matches, and the other identity predicates pass. Diagnosis did not execute
a forward pass, optimizer step or CUDA operation.

The synthetic runtime fixture used the same configuration dictionary at both
levels. It therefore tested matching values but missed the actual storage schema.
A realistic regression fixture must represent distinct root and training scopes,
accept the exact matching training subsection, and reject tampered settings.

## Preserved identities

| Evidence | SHA256 |
| --- | --- |
| Executed runner | `a9899d4c494198afcf35346903f93bc39d5771d60a6711dc164d12a42ad11568` |
| Failure summary | `4f1471ec7880d1718ed08d1fabc6bb414f7e55f48b52cbe9ad604e57193425ec` |
| Training report | `40e783b5380236f56cc21e85563e99d82418a35c45a14ed42cf214d285f1b591` |
| Primary stdout | `c2e1f5ce1e9ac4e135c110310c78b733a34fcf1f6587468d25ada1491f3d4dd9` |
| Diagnosis | `6fafc78ad3bb24c4f9762e7516639d6ff73adb70044816bbf8c50df3d721d696` |

Local evidence directories are `runs/learning/symmetric-affine8000-v1/` and
`runs/learning/symmetric-affine8000-failure-diagnosis-v1/`. The actual terminal
receipt records exit 1, observed by the primary task in chunk `50ba53`.
The outer failure handler reports no model because the loader did not return;
the original model was instantiated and its parent weights loaded on CPU inside
the loader before rejection. These are different lifecycle observations.

All executed v1 artifacts remain immutable. A separate v2 runtime repair is
under development with unchanged scientific hypothesis, optimizer, budget,
seed and quality gates. It is not an executed result. The protected test
partition stays unopened; original corpus identities are inherited from
authenticated evidence, not freshly revalidated through the whole corpus.

This report records an implementation failure in the research pipeline. The
canonical paper retains its existing supported scientific claims. A repair
changes execution provenance; it cannot retroactively make this attempt succeed.

Failure-only audit SHA256: `f012162aae60dcd9211ce9db919864282fe7ef7c388c70113936cbf04dee6dbb`.
Owner decision SHA256: `6c421c7cbe0dfe98f828222d498a74ce2bf023cfefae8ed898bbc298011e532e`.
The independent failure auditor passed 17 synthetic tests, remained Torch-free,
and verified frozen source, output and configuration identities without loading
checkpoint payloads. Its successful audit confirms the failure record; it does
not turn the failed experiment into a model-quality result.
