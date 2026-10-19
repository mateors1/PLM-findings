## Bilinear 32000 saved-error geometry (completed; descriptive evidence accepted)

- `bilinear-budget32000-error-geometry-v1` — saved-validation diagnostic identity;
  no new checkpoint, threshold policy, quality gate or serving promotion.
- `docs/experiments/2026-09-26-bilinear-budget32000-error-geometry-plan.md` —
  fixed input chain and reductions, independent audit and publication contract.
- `scripts/diagnose_bilinear_budget32000_errors.py` and
  `tests/unit/test_bilinear_budget32000_error_geometry.py` — wrapper and synthetic
  checks; reuse the immutable original geometry arithmetic. The wrapper exposes
  `Reader`, `evidence_paths`, `load_helper`, `test_receipt`, `child_schema`,
  `preflight`, `write_exclusive`, `run` and `main`; utility `require/sha/encode`
  functions support explicit input admission, hash binding and serialization.
  CLI requires `--test-receipt` and `--test-receipt-sha256`; optional
  `--preflight-only` authenticates without new reductions. 71 CPU tests passed.
- `runs/learning/bilinear-budget32000-error-geometry-v1/independent-audit.py`
  and `tests/unit/test_bilinear_budget32000_error_geometry_audit.py` — independent
  saved-score reconstruction; no primary arithmetic reuse. 41 CPU tests passed.
  `Reader`, `accepted`, `input_digest`, `authenticate_children`, `load_helper`,
  `test_receipt`, `execution_receipt`, `compare_results`, `audit`, `write` and
  `main` implement admitted source/receipt checks and independent reconstruction.
  Actual primary and audit exited 0; separate owner decision accepts evidence.
- `docs/learning/52-reading-the-last-membership-errors.md` — query versus shared
  threshold intervals, tensor shapes and the boundary between diagnosis and tuning.
- `docs/experiments/2026-09-26-bilinear-budget32000-error-geometry-report.md`
  and portable summary/seed/audit/decision JSON — six failures across five
  queries; a constant threshold cannot fix seeds 1729 or 1731.
- `docs/experiments/2026-09-26-bilinear-serving-integration-design.md` — proposed
  opt-in deployment boundary, explicit set provenance, label-free loading and
  numerical-shape parity requirements. No implementation or promoted policy.
