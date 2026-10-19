## Bilinear 32000-update campaign (completed; all three quality gates passed)

- `docs/experiments/2026-09-26-bilinear-budget32000-plan.md` — fixed all-three
  parent-seed budget comparison to matched8000 controls, train-only margin
  diagnostics, independent audits and per-seed strict improvement gates.
- `bilinear-budget32000-v1` / `plm-bilinear-budget32000-screen-v1` — campaign
  and evaluator identities; architecture, objective and inference unchanged.
- `scripts/refit_bilinear_budget32000.py`,
  `configs/experiments/bilinear_budget32000_v1.json`,
  `tests/unit/test_bilinear_budget32000.py` — implemented and independently
  reviewed runner; 225 synthetic CPU tests passed. Requires seed, readiness
  path and hash; `--preflight-only` admits inputs without importing Torch.
- `runs/learning/bilinear-budget32000-v1/independent-audit.py` and
  `tests/unit/test_bilinear_budget32000_audit.py` — independent saved-evidence
  auditor; 100 synthetic CPU tests passed, independently rerun by root.
  CLI requires `--seed` or `--aggregate-only`, `--readiness-sha256`, and
  `CUDA_VISIBLE_DEVICES=-1`. Seed 1729 passed; the original seed-1730 audit
  failed on equivalent timestamp spellings and is preserved alongside v2.
- `docs/learning/51-optimization-budget-and-training-margins.md` — training
  loss versus exactness/margin, matched seeds and linear-adapter equivalence.

- `scripts/aggregate_bilinear_budget32000.py` and
  `tests/unit/test_bilinear_budget32000_aggregate.py` — verified metadata-only
  three-seed aggregation, strict per-seed gates and audit-before-next-run
  chronology; 45 synthetic tests passed.
- `docs/publication-checks/2026-09-26-bilinear-budget32000-method-review.json`
  — plan review clearance; not runtime admission or a production audit.

- Aggregator implementation is now reviewed: `aggregate_bilinear_budget32000.py`
  accepts `--readiness`, `--readiness-sha256`, optional `--preflight-only` and
  fixed `--out`; it remains Torch-free and requires all three audited seeds.
  Its45-test receipt is `runs/learning/bilinear-budget32000-aggregate-tests-v1/test-receipt.json`.
  Component tests use synthetic evidence, not production campaign results.
- `runs/learning/bilinear-budget32000-readiness-v1/readiness.json` — immutable
  root seal of the exact 22 implementation/recipe/test/receipt dependencies,
  SHA256 `5266fba8ab636590c0654db31b337ba8fd806361d9703c41a1b7260fcb5d766d`.
  All three primary metadata preflights and independent historical metadata
  admission passed. Seed 1729 completed and audited at 218/222 exact. Seed
  1730 completed with a provisional 222/222 report, but its audit stopped on
  equivalent-timezone predecessor text. Original files remain immutable;
  the successful versioned repair and continuation are indexed below.

### Evidence repair v2 (sealed; campaign completed)

- `docs/experiments/2026-09-26-bilinear-budget32000-evidence-repair-v2.md`
  declares exact-instant predecessor comparison, preserved failed evidence,
  saved-evidence re-audit and a single continuation for the unexecuted seed1731.
- `scripts/refit_bilinear_budget32000_continuation_v2.py` owns revised admission
  and provenance, then calls the unchanged original scientific execution body.
  `continuation-complete.json` is required in addition to the final base summary
  and root-observed terminal receipt. Seed 1731 completed once and passed its
  audit at 220/222 exact; all three seeds and the aggregate are audited.
- `runs/learning/bilinear-budget32000-v1/independent-audit-v2.py` and
  `scripts/aggregate_bilinear_budget32000_v2.py` own explicit original/v2 audit
  routing; their 120/42-test receipts and the continuation's 40-test receipt
  are bound by `runs/learning/bilinear-budget32000-readiness-v2/readiness.json`,
  SHA256 `9df87cdafc02a3449b3b78d9bf48be93f0840da4ba64b35c21c09c286c96a69a`.
  Seed 1730's `independent-audit-v2.json` passed with 222/222 exact; its hash is
  `83d48edb5e78ee66eeab4c9e87970b882f8d0783c5829a07b8b9cd56e5ca728d`.
- `runs/learning/bilinear-budget32000-planning-v1/receipt-offset-diagnosis-v1.json`
  records the exact zero instant difference and original failed-evidence hashes.
  The root terminal-receipt writer in that directory preserves JSON timestamp
  strings; it records explicitly supplied observed exits and does not infer
  process completion.

- `runs/learning/bilinear-budget32000-v1/aggregate-v2.json`,
  `aggregate-audit-v2.json`, and `decision-v2.json` — separate aggregation,
  independent audit and owner acceptance. All three gates pass; 660/666 exact
  outcomes on three repetitions of the same 222 validation queries. No promotion.
- `docs/experiments/2026-09-26-bilinear-budget32000-report.md` and
  `2026-09-26-bilinear-budget32000.json` — narrative and portable audited result.
- `docs/experiments/2026-09-26-bilinear-budget32000-model-version-inventory.json`
  — additive 41-checkpoint register; inherited 38 identities, three fresh hashes.
- `runs/learning/publication-handoffs/export-bilinear-budget32000-v2.py` —
  reviewed exporter with three explicit aggregate/audit/decision SHA arguments.
  Preserves failed-audit and continuation provenance; 48 loss chunks reconstruct
  all 96000 updates, 102 diagnostic chunks reconstruct all 9822 endpoint rows.
  No raw score matrices or weights are published. Root export verification is
  `runs/learning/bilinear-budget32000-planning-v1/export-execution-review-v2.json`.
