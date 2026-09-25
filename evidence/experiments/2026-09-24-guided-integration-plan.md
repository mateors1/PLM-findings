# Guided inference integration and HTTP parity — declared before validation

2026-09-24. Integrate the selected alpha16 + unique-target policy as an explicit
optional setting. Default alpha0 preserves existing generation and identifiers.
Keep model weights and model source unchanged. Reuse the existing prompt-only
symmetric head and its extra prefill forward; no score-computation optimization.
Positive strength requires constrained cached decoding and a trained head.
No subject mask, forced EOS, grammar change or numeric model token is added.

Before real-data runs, require unit/config/native/CLI integration coverage and
full standard checks. Then evaluate all222 validation queries at seeds1729/30/31
through the core batch-eight generator at alpha0+unique and alpha16+unique:
1,332 offline executions. Require exact full token sequences, parsed fields,
validity/termination/errors and raw/processed metrics against archived experimental
reports. Verify model-source bytes, script/report/checkpoint/data/split/runtime
identities; record new inference source and evaluation config separately from
unchanged training identity. Zero omits the experimental alpha0 descriptor suffix;
positive16 retains the canonical suffix.

Only after full offline parity, run real loopback HTTP verification for all222
queries at all three alpha16 checkpoints, sequentially, native batchsize1.
Require matching complete raw token IDs, targets, flags/errors, canonical serial
policy descriptor, guidance strength and other deployment settings, per-query
processed targets and aggregate metrics. Normalize only the expected offline
batch descriptor component. Check IGNORE/limit invariance, invalid-subject422,
readiness and deployment identity; stop every owned server afterward.
Legacy references without token IDs must be labeled partial parsed parity;
positive-guidance verification requires full token evidence.

Any mismatch is a failed integration check, not a reason to loosen the criterion
or substitute a different checkpoint. Preserve failure artifacts and diagnose.
No final-test use, retraining, commit/push or performance advantage claim.
Success establishes an opt-in application path on the tested validation data;
it does not establish oracle quality, serving SLO, concurrency or energy results.
No permanent service is started by the correctness campaign.
