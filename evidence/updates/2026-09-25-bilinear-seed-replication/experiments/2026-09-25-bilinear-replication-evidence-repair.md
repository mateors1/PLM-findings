# Replication evidence repair: unchanged model executions

Declared on 2026-09-25 (local date), after both fixed-budget model runs finished
and before executing the repaired audit or aggregation. The frozen replication
plan and all model outputs remain unchanged.

## Observed failures

1. Independent auditor v1, SHA256
   `c1170a43629111d1a9904b9c26c2ef9cba77ff46612486dd868d02b03601f0a0`,
   exited 1 on its first seed-1730 attempt. It incorrectly expected
   `training_metadata` in the original parent's `run.json`. That record contains
   `config` and `identity`; the checkpoint sidecar carries training metadata.
   No independent acceptance was produced. Failed stdout SHA256 is
   `7faf1f1c30cbb0cc815214d4479d66e8bfa595835eed4e6b07dfead045e12275`;
   the attempt receipt is
   `6545cfbe9e37519888a9d86ff188a71d7bb578e5c8486ace82eba8ffeaf982a5`.
2. The original aggregate command exited 1 with `launch/terminal chronology
   binding`. PowerShell's JSON reader converted a UTC launch timestamp into a
   local-offset datetime before the root wrote the execution receipt. For
   example, `2026-09-26T00:38:06.5251055+00:00` and
   `2026-09-25T19:38:06.5251055-05:00` name the same instant, but the frozen
   aggregator requires identical strings. The failed aggregate summary is
   `c3d79ed3effe7504b67121b25274603b2a5870f9150a5e8043b975f9dc478c2c`;
   failed stdout is
   `68736e31ddca6d572c4384273d7317dbfb02bdbb37896de8b555df40296fbce3`.

Both failed attempts, their source, original receipts and partial reports remain
immutable. These are evidence-processing failures, not failed training runs or
changes to the scientific quality gate. The original aggregate is incomplete.

## Allowed repair and verification

- Create a separately versioned, aggregate-only runner and auditor v2. Keep the
  seed execution runner and its hash separate from the aggregation runner hash.
- Validate original parent run/config/identity against the matching sidecar's
  explicit schema. Do not remove lineage checks to accommodate missing fields.
- Compare timezone-aware timestamps as instants. Reject missing offsets,
  different instants and overlapping or reversed runs. Preserve original strings
  and bytes in the authenticated input inventory.
- Bind the failed attempts and this declaration in the repaired evidence chain.
  Add schema and timestamp regressions and freeze the new scripts before running.
- Write recovered aggregation to `aggregate-v2.json`, without overwriting the
  failed `summary.json`. Version the owner decision helper to bind that output
  and auditor v2 explicitly.

No model retraining, checkpoint rewriting, new predictions, threshold search,
changed metric arithmetic, gate relaxation or protected-test access is part of
this repair. Independently reconstruct both seeds' metrics, state lineage and
quality gates before accepting evidence. Report the repairs in the educational
summary, research narrative and publication bundle.
