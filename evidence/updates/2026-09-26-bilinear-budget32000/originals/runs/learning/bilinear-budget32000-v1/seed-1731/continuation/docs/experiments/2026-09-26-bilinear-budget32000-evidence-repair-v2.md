# Bilinear 32000 campaign: evidence repair and explicit continuation v2

Status: declared for independent implementation review; no repaired audit or
seed-1731 execution has occurred. Scientific campaign: `bilinear-budget32000-v1`.
Evidence revision: `bilinear-budget32000-evidence-v2`. Execution continuation:
`bilinear-budget32000-continuation-v2`. These do not rename existing checkpoints.

## Failure, preserved evidence, and scope

The original plan remains authoritative:
`2026-09-26-bilinear-budget32000-plan.md`, SHA256
`5f1e0243b373ad1287c31dcef8a3c1cfa310cac9701795469a68677506e6f444`.
Its frozen 22-file readiness manifest is
`runs/learning/bilinear-budget32000-readiness-v1/readiness.json`, SHA256
`5266fba8ab636590c0654db31b337ba8fd806361d9703c41a1b7260fcb5d766d`.

Seed 1729 completed and was audited. Its summary SHA256 is
`b41fec77f92855db5753c97add0255e4593fa328192eb34fe3ab6ad567bbacd1`,
its passing audit is
`76ecfdb9e8e3654fe9aa7d939972c6b453079b1a907d489f5cded679e0b20850`.
Seed 1730 completed training; summary SHA256
`1237d6e266fa57b5284f76ab76d50069ec9f161f5e23add84e06d769a47b5e11`.
Its original audit exited 1 before checkpoint/state verification:
`85b193ad473d77ac10d84aa2e9138ba4d56efd7e9694ef52eeab4ed6e069c188`.
The failed audit terminal receipt is
`2195369180305bb04f7967b5877e41006f965fd8499d411229193415662fc5ce`.
The original seed-1730 execution receipt is
`97f36f6d2b1bee433082b49c5204376d4f70dd78e3abba52431d5be32781b1b8`.

PowerShell parsed and reserialized the embedded predecessor completion time
with a different timezone offset. These strings denote the same exact instant:

- `2026-09-26T14:56:49.5359481+00:00`
- `2026-09-26T09:56:49.5359481-05:00`

The frozen auditor compared the entire predecessor object as text-equivalent
JSON, despite the plan's timezone-aware chronological contract. Independent
inspection established exact Fraction equality, intact receipt/hash bindings,
and a 17.3771973-second gap from seed-1729 audit completion to seed-1730 launch.
This establishes the cause of rejection, not completion of the interrupted
seed-1730 scientific audit. Its reported 222/222 remains provisional.

The campaign stopped before seed 1731. Preserve every original file, receipt,
failed audit, stdout, source and test hash. Do not correct the receipt in place,
overwrite the failed audit, or rerun training for seeds 1729 or 1730.

## Repair and continuation contract

1. A separately versioned auditor verifies predecessor object keys, seed and
   summary hash exactly, and compares completion times as exact timezone-aware
   Fraction instants. Retain exact byte hashes of both original receipt files.
   Equivalent offset spellings pass; a 100 ns difference fails equality.
2. Re-audit the saved seed-1730 checkpoint and every scientific artifact using
   all original tensor, optimizer, trace, metric, gate and provenance checks.
   Write `seed-1730/independent-audit-v2.json` and a new observed audit terminal
   receipt `audit-execution-receipt-v2.json`. Preserve the original failure.
3. Only after that audit passes, a reviewed continuation runner may execute the
   never-run seed 1731 once. Its admission explicitly names the repaired audit,
   its actual implementation/test hashes and successful observed terminal.
   Launch must follow completion of that repaired audit. Request GPU access
   again; the original reservation was released.
4. The continuation may reuse the hash-verified original scientific runner.
   Distinguish base source from actual entrypoint: preserve original execution
   snapshots and add separately indexed continuation source/readiness snapshots
   and identity fields. The new wrapper must not masquerade as the old script.
   Any wrapper substitution is restricted to predecessor evidence admission and
   explicit provenance recording. No numerical or evaluation method changes.
   Never monkeypatch a frozen module, impersonate old files through a virtual
   reader, or rewrite returned identities to satisfy old checks. Compose its
   unchanged helpers explicitly; use a separately versioned source copy where
   clean composition is impractical. The continuation owns its admission and
   snapshots, then directly calls the unchanged base `_execute` once.
   Verify all pinned inputs before and after execution. Write a separate
   `continuation-complete.json` binding the final summary, actual wrapper and
   v2 readiness; a base summary alone cannot establish wrapper completion.
5. A versioned aggregate and aggregate audit use this explicit audit routing:
   seed 1729 original passing audit, seed 1730 repaired v2 audit, and seed 1731
   v2 audit of the continuation. Bind each seed to its actual execution and
   audit source/test/readiness identities and observed terminal receipts.
   Preserve the original two failed-audit files as part of the publication
   history. Never use a failed audit as a successful predecessor.

All three declared seeds remain mandatory. Keep their original parents,
fresh-zero residual/optimizer, 32000 full-batch updates, numerical settings,
training data, scorer, validation endpoints and strict per-seed quality gates.
No thresholds, retries, alternate seeds, early stopping, fitted validation or
promotion are authorized by this amendment. The original 218/222 and provisional
222/222 observations do not change the requirements for the remaining seed.

## Separate readiness and regression evidence

The v2 root seal is
`runs/learning/bilinear-budget32000-readiness-v2/readiness.json`.
It binds this amendment, the original readiness manifest, three new source
files, three new test files and six test-receipt/stdout files (14 exact files).
All original 22 dependencies must still match the original readiness seal.

New source paths:

- `scripts/refit_bilinear_budget32000_continuation_v2.py`
- `scripts/aggregate_bilinear_budget32000_v2.py`
- `runs/learning/bilinear-budget32000-v1/independent-audit-v2.py`

New test paths correspond to
`test_bilinear_budget32000_continuation_v2.py`,
`test_bilinear_budget32000_aggregate_v2.py`, and
`test_bilinear_budget32000_audit_v2.py` under `tests/unit/`.
Role receipts and stdout live in
`runs/learning/bilinear-budget32000-{runner,auditor,aggregate}-tests-v2/`.
The readiness revision, per-seed source routing and output filenames must agree
across all consumers before the new root seal is written.

The v2 readiness schema is `schema_version: 1`, scientific `campaign_version`,
`evidence_revision`, `amendment_sha256`, `base_readiness: {path, sha256}`,
`ready: true`, and the exact 14-entry `files` map. Role test receipts retain
the v1 passed/exit/observed-terminal/no-GPU/no-experiment/zero-skip contract,
add the evidence revision and amendment hash, and bind their new script, new
test, this amendment and original readiness as their four direct dependencies.
The original readiness separately authenticates all original dependencies.

New audit outputs identify their actual script and tested-receipt hashes,
`audit_revision`, the amendment hash, original plan hash, original
`base_readiness`, and the v2 `shared_readiness`. Keep the imported base auditor
identity separately. Seed-1731 summaries identify the base `script_sha256` and
the actual `execution_wrapper_sha256`, execution revision, amendment hash,
`continuation_readiness` and separately indexed `continuation_snapshots`.
These fields supplement the original scientific source snapshots. New audit
terminal receipts use the actual v2 auditor/test hashes. The root records
terminal exit from the process handle, rather than inferring it from files.

Regression coverage must include the actual equivalent-offset failure, a
100 ns mismatch, missing/extra predecessor keys, wrong seed/hash, reversed or
overlapping chronology, failed audit terminals, stale readiness, immutable
output refusal, and explicit per-seed audit routing. Exercise the real wrapper
orchestration and unchanged scientific stage interface using synthetic evidence;
do not infer production readiness from isolated helper tests alone.

Final publication must include both the original failure and the repaired
evidence, explain the exact amendment, retain separate checkpoint/execution/
evaluator/publication identities, and avoid claiming the campaign passed until
all three completed children and the final aggregate are independently audited.
