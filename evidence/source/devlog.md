# Dev Log

Short, skimmable session summaries — the fast catch-up for the next session.
One entry per working session, newest at the top. Deep rationale goes in
`research_log.md`; symbol changes go in `registry.md`.

Format: `## YYYY-MM-DD — <session title>` with **Done / Changed / Next**.

---

## 2026-08-13 — Phase B continuation plan

**Done**
- Re-onboarded the repository and verified the current Phase B implementation.
- Added an execution-ready multi-Luna plan for deterministic query holdout,
  shared ranking metrics, oracle/popularity/ComplEx baselines, CLI integration,
  local TYPE/COLOR labeling, and final quality gates.

**Changed**
- Recorded the query-level split and baseline decisions in `research_log.md`.
- Added a parallel labeling-webapp workstream with API acceptance tests.

**Next**
- Execute Wave 0, then dispatch the three independent Wave 1 evaluation tasks.

---

## 2026-08-13 — Phase B evaluation and labeling shipped

**Done**
- Implemented deterministic query holdout, shared ranking metrics, oracle,
  popularity, and optional ComplEx baselines.
- Added `plm baseline evaluate` JSON output and a tested local labeling webapp.
- Repaired the optional-serving CLI test and renamed the query split config field.

**Changed**
- Updated protocol/data contracts, README, Makefile, registry, and tests.

**Next**
- Use the compiled National Dex corpus to run the first recorded baseline report,
  then begin Phase C model architecture/training.

---

## 2026-07-23 — Design debate + project meta scaffolding

**Done**
- Reconciled the committed Phase-A schema, the proposed `schema.md`, `protocol-spec.md`,
  and the v0.1 project handoff into a single agreed direction (see `research_log.md`).
- Locked v1 scope: mode `SAME` only; dimensions `TYPE` + `COLOR`; ~1000 Pokémon-name
  identifiers; bucket labeling; attributes as hidden nodes; oracle-driven systems framing.
- Added project meta files: `CLAUDE.md`, `research_log.md`, `devlog.md`, `registry.md`.
- **Phase 2 graph schema landed.** `src/plm/graph/schema.py` grown 5 → 7 tables (added
  `provenance`, `aliases`; extended `nodes`/`edges`/`snapshots`) — additive, `SCHEMA_VERSION`
  stays `1` (no shipped data to migrate). `GraphRepository` gained `add_provenance`,
  `add_alias`, `node_keys(kind=...)` and richer `add_node`/`add_edge`/`add_relation_type`
  signatures. +5 tests. `registry.md` updated.
- **Tokenizer reworked.** `plm.protocol.tokenizer` rewritten as a banded, append-only
  vocabulary: uppercase specials `PAD/BOS/EOS/UNK/SEP/ANSWER`, dimensions/modes in the
  steering band (`TYPE/COLOR/SAME`), entities pinned at `ENTITY_BASE=1024`, `128–1023`
  reserved. Predicates removed from the vocab. Per-token `class` metadata + `class_of`/
  `entity_ids`. Fail-closed `encode(strict=True)`; named id props. Vocab file format
  `schema_version=2`. Tests rewritten (12). `registry.md` updated. See `research_log.md`.

**Toolchain**: green throughout (ruff / format / mypy `--strict` / 36 pytest).

**Next (protocol config + corpus compiler — Phase B)**
- Lift `DEFAULT_DIMENSIONS/MODES` out of the tokenizer into a versioned protocol config
  (`configs/protocol/pokemon_v1.*`) with the derivation rules (`TYPE/COLOR SAME`).
- Build the corpus compiler: graph → `BOS SUBJ DIM SAME ANSWER targets EOS`, deterministic
  ordering (shared-count → confidence → key), loss-mask at `ANSWER`, held-out split.
- Needs a real (even tiny) graph authored first — the labeling web-app or a seed importer.

## 2026-07-29 — National Dex graph + corpus compiler

**Done**
- Replaced the planned small Gen I seed with a full PokeAPI National Dex import:
  1,025 products, 28 hidden attributes, and 2,576 `HAS_TYPE`/`HAS_COLOR` edges.
- Added versioned `pokemon-v1` protocol config and finalized grammar/docs.
- Implemented deterministic graph → JSONL corpus compilation, vocabulary and
  manifest generation; verified 2,050 records.

**Changed**
- Added `plm graph import-national-dex` and functional `plm corpus compile`.
- Updated the data config, README, contracts, registry, and research narrative.

**Next**
- Implement an explicit held-out evaluation split and oracle/KGE baselines.

## 2026-08-23 — Phase C + AVO scaffold execution

**Done**
- Implemented the three-way split/config/protocol/vocabulary/artifact identity
  foundation, pinned graph provenance, repaired ComplEx negative sampling, and
  added immutable baseline-report handling.
- Added the reference decoder, legal output masking, synthetic overfit gate,
  training/checkpoint/resume path, validation/final-test evaluator, protocol
  parser/constrained logits, hydration, and AVO evidence/lineage infrastructure.
- Updated `README.md`, architecture/data/protocol docs, `registry.md`, and this
  repository's research narrative.

**Verification**
- `uv run pytest`: 103 passed, one upstream Starlette/httpx warning.
- `uv run ruff check`, `uv run mypy`, and `git diff --check`: green.
- Optional Torch runtime: model, trainer, checkpoint resume, synthetic overfit,
  and temporary-corpus training smoke all passed.

**Next**
- Run the real National Dex training/evaluation campaign, record repeated
  baseline variance and PLM validation receipts, then reassess AG0 honestly.

## 2026-09-24 — Code-backed project state assessment

**Done**
- Inspected source, configs, local artifacts, and Git history at `8afff36`.
- Re-ran all tests: 103 passed, one Starlette/httpx deprecation warning.
  Ruff, format checking (68 files), and strict mypy (49 files) passed.
- Verified 1,025 products, 28 attributes, 2,576 edges, 2,050 records, and
  train/validation/test counts of 1,637/222/191. CUDA is available on the
  local RTX 5070 Ti with Torch 2.11.0+cu128.
- Confirmed an unshifted training-loss defect and reproduced multi-batch
  resume divergence. Identified stale graph identity in the local corpus
  manifest and legacy import provenance; temporary recompilation validates
  while retaining identical records and vocabulary.

**Changed**
- Appended assessment evidence and corrected readiness interpretation in
  `research_log.md`. No implementation or persistent dataset changes.

**Next**
- Repair causal loss and resume data position with independent behavioral
  checks, refresh versioned corpus/provenance, then execute the recorded
  baseline/PLM validation campaign. Phase D and scientific claims remain open.

## 2026-09-24 — Learning iteration 1: causal loss and reproducible resume

**Done**
- Corrected next-token loss alignment, including prompt/padding masks and z-loss.
- Strengthened the toy overfit gate with prompt-only completion generation.
- Fixed sequential batch offset on resume and separated loader/model RNG.
- Added checkpoint objective/order metadata and incompatible-resume rejection.
- Added regression tests that fail on the earlier behavior. Full verification:
  115 tests passed (one existing Starlette/httpx warning), including CPU/CUDA
  resume, dropout, accumulation, partial batches, and a Windows worker. Ruff,
  formatting, strict mypy, and diff checks pass.
- Recorded a toy overfit result: loss 3.4326453 -> 0.00485336 after 100 steps;
  generated target/EOS fixture `(6, 7)` exactly. No National Dex quality claim.

**Changed**
- Updated model/trainer contracts, registry, architecture, and research log.
- Added an educational project overview and detailed first lesson under
  `docs/learning/`, linked from README and MkDocs. Corrected the stale Phase A
  docs landing-page status. Documentation builds with existing external-tree
  source-link warnings.

**Next**
- Teach and perform the versioned dataset/provenance refresh, preserving old
  artifacts. Resolve evaluation/identity gaps, establish comparable baselines,
  then run the reference validation campaign. No commits or pushes this session.

## 2026-09-24 — Learning iteration 2: dataset identity and architecture experiments

**Done**
- Published and verified a pinned National Dex snapshot/receipt, preserving old
  data. Added exact receipt-only reconstruction and tamper/partial-state checks.
- Added prompt-only generation and complete-answer evaluation alongside cached
  first-token ranking; removed inferred protocol validity and strengthened
  checkpoint payload/sidecar identity checks.
- Archived training source/config/environment, persisted progress/results, and
  separated token-weighted validation CE from MoE auxiliary losses.
- Implemented activation checkpointing and sparse top-k/shared-expert MoE;
  corrected RoPE coordinate pairing. Added recipes and router telemetry.
- Profiled dense/checkpointed/MoE-top1/top2 on `[8,289]`: median steps
  60.44/98.03/127.08/127.99 ms; peak allocated memory
  498.56/184.26/737.75/894.88 MiB. Raw report retained in docs/experiments.
  A separate toy MoE overfit generated the expected completion; real quality
  remains unmeasured.

**Verification and documentation**
- 130 tests passed, one existing dependency warning. Ruff/format/strict mypy
  and diff checks green; documentation builds with existing source-link warnings.
- Updated registry, architecture, quickstart and educational overview. Added
  lessons 02 (data/evaluation) and 03 (architecture equations/tensors/results).

**Next**
- Run comparable dense/MoE National Dex validation and baseline campaigns;
  inspect learned routing. Then test context-extension hypotheses and complete
  serving work. No final-test selection, commits or pushes this session.

## 2026-09-24 — Learning iteration 3: real quality campaign and failure diagnosis

**Done**
- Separated baseline initialization and split seeds; added resolved report/source
  identities, per-query generation evidence and completed-run overwrite protection.
- Recorded the campaign plan before executing 2,000-update dense and MoE runs
  plus three seeds each for oracle/popularity/ComplEx. All completed on the same
  222-query validation partition; final test untouched.
- Dense/MoE generated F1: 27.84%/27.96%; exact sets: 0/222 for both. ComplEx
  ranking MAP averaged 0.99260 (sample SD 0.00099). Measures are not interchangeable.
- Diagnosed misleading aggregate loss: first targets comprise only 0.77% of
  supervised tokens; dense first/later CE is 5.77/0.15. Both models generated all
  16 sampled training answers exactly. Prompt ablation confirms prompt dependence
  despite poor generalization; MoE traffic uses all four experts without collapse.
- Added reproducible summary, plotting, conditioning and routing scripts; retained
  raw runs/checkpoints and portable result/diagnostic artifacts. Updated registry,
  research narrative, current status and Lesson 4 with equations and a verified chart.

**Verification**
- 131 tests passed, one existing warning; Ruff, formatting, strict mypy and diff
  checks green. No commit or push. Source remains bound to both campaign runs.

**Next**
- Test prompt-conditioned set supervision, with a first-target weighting control,
  against the retained dense reference. Require complete-answer validation gains;
  oracle parity, protected-test selection and serving remain open.

## 2026-09-24 — Learning iteration 4: prompt-set supervision and deterministic output rules

**Done**
- Added independently switchable first-target weighting and balanced prompt-set
  supervision, tied to product embeddings. Preserved common initialization and
  ordinary token-loss reporting; added first-target/set telemetry and resume checks.
- Executed the predeclared two-candidate, 2,000-update comparison on the same
  validation partition. First-target weight 32 worsened raw F1 to 19.01%; auxiliary
  set supervision improved it from 27.84% to 62.90%. Raw exact sets remain zero.
- Error analysis motivated the existing protocol's deterministic post-filter:
  SAME self-exclusion, IGNORE, stable uniqueness and optional return limit. The
  same policy replay gives 42/222 exact baseline answers and 105/222 auxiliary
  answers. Raw and processed metrics remain separate; no relation lookup is used.
- Added classifier/policy analysis scripts and preserved all raw evidence;
  updated registry, architecture/protocol docs, current status and Lesson 5.

**Verification**
- 144 tests passed with one existing warning. Ruff/format/strict mypy are green;
  campaign figures visually inspected. No commit/push or protected-test use.

**Next**
- Repeat the improved candidate across seeds, address remaining relation errors
  and checkpoint selection, and integrate the native serving path. Single-seed
  quality progress does not yet establish oracle parity or serving readiness.

## 2026-09-24 — Learning iteration 5: three paired seeds

**Done**
- Predeclared and completed dense/prompt-set runs at seeds 1730/1731, retaining
  the original seed 1729, fixed split, 2,000 steps and final-checkpoint rule.
- Mean raw F1: 23.13% dense versus 60.62% prompt-set; improvement positive in
  all three pairs. Same-policy exact accuracy: 15.62% versus 45.80%. Raw exact
  remains zero; all responses terminate validly. Final test untouched.
- Identified 85 queries never exact in any prompt-set seed (72 TYPE, 13 COLOR).
- Added audited aggregation, paired plots and Lesson 6 with equations/tensor
  details. Updated registry, current status, research narrative and docs navigation.

**Verification**
- 146 tests passed, one existing warning. Ruff/format/strict mypy and diff checks
  green; comparison figure inspected. Completed all launched runs; no commit/push.

**Next**
- Diagnose persistent TYPE errors and checkpoint selection. Add and verify KV
  caching before native serving measurements; preserve the uncached reference.
- Oracle parity, broader robustness, protected-test selection and serving remain open.

## 2026-09-24 — Learning iteration 6: native KV cache

**Done**
- Added inference-only KV cache with absolute RoPE/masks, unpadded chunk support,
  ownership/layout/bounds checks and fresh state per generation request. Exposed
  opt-in `eval.use_kv_cache`; retained uncached reference and unchanged checkpoints.
- Verified 222/222 anchor-model validation responses match exactly. Benchmarked
  eight prompts across three alternating paired repetitions: summed time 20.352 s
  uncached -> 19.527 s cached; peak allocated memory 37.96 -> 35.63 MiB.
- Profiled the modest gain: repeated vocabulary scans and many small module calls
  remain. Added Lesson 7, archived evidence, registry and architecture/status updates.

**Verification**
- 159 tests passed; added CLI cache-identity/parity assertions also pass in the
  focused integration run. Ruff/format/strict mypy/docs/diff checks green, with
  existing dependency and external-source documentation warnings. No commit/push.

**Next**
- Optimize verified grammar overhead without weakening parser checks, then connect
  the native serving path. Continue the 85 persistent validation failures (mostly
  TYPE). Keep cache opt-in until broader performance evidence supports a default.
- No oracle parity, final-test selection, HTTP/concurrency or energy claim yet.

## 2026-09-24 — Learning iteration 7: native HTTP serving

**Done**
- Added shared inference loading, deployment graph copies and frozen hydration,
  strict request validation, bounded serial model execution and the `plm serve`
  CLI. Kept raw generation separate from filtering and hydrated API results.
- Fixed malformed-prefix handling and removed repeated steering-vocabulary scans.
- Verified all 222 archived raw responses over actual HTTP; reproduced processed
  F1 63.19% and 105/222 exact answers. The verification server stopped cleanly.
- Added Lesson 8, portable reports, source/deployment receipts and current docs.

**Verification**
- 185 tests passed with one existing warning. Ruff/format/strict mypy and diff
  checks passed; Torch-free imports verified. No final-test use or commit/push.

**Next**
- Address persistent relation errors and checkpoint selection. Measure native
  concurrency and latency against the deterministic oracle under declared quality
  and SLO rules. Linux/vLLM and energy claims remain separate, pending work.

## 2026-09-24 — Learning iteration 8: sequence failure diagnosis

**Done**
- Added a validation-only diagnostic for teacher-forced token accuracy and a
  clearly labeled oracle-first-ID intervention. Preserved the trained model and API.
- Reproduced all 222 unassisted outputs, then measured 105 -> 174/222 processed
  exact answers with the hint. First-target accuracy is 48.65%; later-token
  teacher-forced accuracy is 98.56%. All 48 hinted failures are TYPE.
- Stratified by training answer-family support, archived per-query evidence and
  identities, added Lesson 9 and updated the overview, navigation and registry.

**Verification**
- 186 tests passed, one existing dependency warning; lint/format/strict typing
  passed. Independently replayed raw/processed metrics and verified artifact hashes.
- Diagnostic run completed. No final-test evaluation, commit or push.

**Next**
- Compare saved checkpoints under a predeclared complete-answer selection rule.
  Address prompt conditioning and sparse TYPE continuation; oracle hints are
  diagnostic only. Oracle parity and measured serving comparisons remain open.

## 2026-09-24 — Learning iteration 9: saved-checkpoint selection

**Done**
- Predeclared and completed steps 500/1000/1500/2000 of the existing seed-1729
  prompt-set run. Evaluated 888 unassisted responses and separate FP32 losses.
- Retained step 2000: exact answers 20/76/105/105 across the four snapshots;
  final processed F1 63.19% beats step 1500's 61.27%. All responses valid/EOS.
- Verified final-checkpoint parity on all 222 archived outputs. Earlier auxiliary
  loss minima do not improve the selected generation outcome; model unchanged.
- Added selection checks, raw/portable reports, inspected PNG/SVG comparison,
  Lesson 10 and updated learning overview, registry and documentation navigation.

**Verification**
- 188 tests passed, one existing warning; Ruff/format/strict mypy passed.
- Artifact hashes and raw/processed F1/exact counts independently verified for
  all 888 responses. All launched work completed; no final-test use or commit/push.

**Next**
- Improve prompt conditioning and sparse TYPE continuation using the diagnosed
  failure patterns. A four-snapshot within-seed selection does not finalize the
  broader model/selection procedure. Oracle parity and serving comparisons remain open.

## 2026-09-24 — Learning iteration 10: symmetric relation supervision

**Done**
- Added an opt-in symmetric shared-embedding relation loss, subject exclusion,
  separate diagnostics and `symmetric_relation_lab`. Default remains off.
- Completed seed-1729 training at the same 2,000-step budget and full validation:
  raw F1 62.90% -> 83.41%; same-policy exact 105 -> 143/222. All outputs valid/EOS.
- Measured both auxiliary classifiers separately; scores never alter decoding.
- Added Lesson 11, plan/results, registry entries, architecture and overview updates.

**Verification**
- 194 tests passed; symmetry/gradients/autocast/resume/overfit/cache checks included.
  Ruff/format/strict mypy and Torch-free core imports pass. Docs build with existing warnings.
- Verified controlled configs, checkpoint/data/split identities and metric replay.
  All launched work completed; no protected-test evaluation, commit or push.

**Next**
- Repeat at seeds 1730/1731 before changing the retained serving candidate.
  Investigate remaining TYPE and membership-to-sequence errors. Oracle parity
  and the serving performance comparison remain open.

## 2026-09-24 — Learning iteration 11: replication and a failed promotion gate

**Done**
- Completed symmetric seeds 1730/1731 and three fresh same-source prompt-set
  controls, then audited all six 2,000-step runs and 1,332 validation responses.
- Mean raw F1 60.78% -> 79.87%; processed exact accuracy 45.80% -> 61.11%.
  Every pair improves; exact counts rise by 35/33/34 answers out of 222.
- Retained the existing serving reference: two candidate answers fail EOS
  termination, so the predeclared promotion gate fails. One fresh control also
  fails termination; this weakness occurs with both objectives. No promotion HTTP run.
- Preserved a failed historical-control replay (0/92 identical final tensors)
  and a passing fixed-batch CPU forward/backward comparison (95/95 tensors).
- Diagnosed the two candidate failures and their separate membership classifiers;
  added bound reports, inspected plots, Lesson 12 and current overview/navigation.

**Verification**
- All 195 tests pass, one existing warning; Ruff/format/strict mypy pass.
- All six training archives have identical source contents. Raw metrics replay
  from exact validation rows; candidate seed 1729 matches cached/uncached on 222/222.
- All launched work completed. No final-test evaluation, commit or push.

**Next**
- Investigate repetition/termination and the remaining membership-to-sequence
  gap. Keep decoding-policy changes explicit and evaluate them separately.
  Fifty-five queries are never exact across candidate seeds (52 TYPE, 3 COLOR).
  Oracle parity, finalization and matched-quality serving measurements remain open.

## 2026-09-24 — Learning iteration 12: target uniqueness and a batching probe

**Done**
- Added default-off target uniqueness to shared generation, CLI config/reports
  and native serving receipts. State is per request; EOS/bounds remain explicit.
- Completed 1,332 decoding executions across three frozen checkpoints. All 666
  original outputs replay exactly; all 666 unique outputs are valid/terminated.
- Repetition disappears in 15 affected answers, but exact counts stay 143/130/134.
  Mean processed F1 changes 80.245% -> 80.296%; seed 1729 slightly regresses.
  The declared promotion gate fails; existing serving reference and defaults retained.
- Recorded the first-divergence audit and AERODACTYL regression. Updated registry,
  protocol/architecture docs, Lesson 13, learning overview, README and navigation.
- A separate batch-size-eight prototype matches 16 serial outputs across both
  policies. It remains outside core/evaluation/serving and makes no speed claim.

**Verification**
- 199 tests pass, one existing warning; Ruff/format/strict mypy pass.
- Campaign CLI smoke and full execution pass after fixing an initial import error.
- Model/checkpoint/data/runtime provenance is checked; first emitted IDs match
  in all 666 pairs and each of 15 first divergences removes a repeated choice.
- All launched work completed. No conditional HTTP promotion, final-test use,
  commit or push. Docs build with existing repository-external link warnings.

**Next**
- Turn the batch prototype into a rigorously checked evaluator, then test ways
  to improve the initial relation choice. Completion constraints alone do not
  close the quality gap. Oracle parity, finalization and serving measurements remain open.

## 2026-09-24 — Learning iteration 13: verified batched evaluation

**Done**
- Integrated fixed-group KV generation with independent row completion/uniqueness
  state, shared parser semantics and explicit offline batch-size configuration.
- Default remains one; larger batches require constraints and caching. Native
  serving rejects this offline option, preserving its existing scheduler.
- All 1,332 full outputs and metrics match serial reports across three frozen
  symmetric checkpoints and both policies, including two original truncations.
- Three isolated paired eight-query timings yield 5.3071x throughput; median
  workload time 7.38 -> 1.39 s, peak allocated GPU memory 36.35 -> 51.71 MiB.
- Added declared plan, immutable raw receipts and source archive, portable
  evidence, Lesson 14 with a verified chart, registry and documentation links.

**Verification**
- 207 tests pass with one existing warning. Ruff, formatting and strict mypy
  passed before the campaign. Source, checkpoints, reports and exact metrics
  were independently checked afterward; all timed outputs also match.
- No retraining, checkpoint promotion, final-test use, commit or push. All model
  jobs completed. Timing is offline decoder throughput, not HTTP performance.
- Final Ruff/format/strict-mypy and whitespace checks pass. Core imports pass
  with Torch explicitly blocked; uv.lock is unchanged. Documentation builds
  with the six existing links to files outside the documentation tree.

**Next**
- Use the validated offline path to test better initial relation choices while
  keeping training and decoding interventions separately identifiable.
- Oracle quality parity, protected finalization and matched-quality serving /
  concurrency / energy measurements remain open.

## 2026-09-24 — Learning iteration 14: first-target relation guidance

**Done**
- Added an offline frozen-model experiment that applies learned log-sigmoid
  membership penalties only to the first product decision. No production changes.
- Ran all 2,664 executions: three seeds, four strengths, 222 validation queries.
  All 666 baseline outputs/metrics replay exactly before guided comparisons.
- Strength 16 raises mean processed F1 80.25% -> 88.33% and exact answers
  61.11% -> 68.17%. It gains 48 exact answers and loses one; strength 4 gains
  45 and loses none. Every setting improves per-seed aggregate quality.
- Both original 507-token loops remain unchanged. All settings fail the declared
  completion/integration gate; no strength or checkpoint is promoted.
- Added plan, complete reports/source/script receipts, portable analysis,
  verified chart, Lesson 15, learning overview and architecture/navigation links.

**Verification**
- Full suite: 216 passed before evaluation, with one existing warning.
- User requested parallel work: independent review found no blocking defect or
  answer leakage, then strengthened numerical-bias and prefill-cache tests.
  Expanded focused suite: 12 passed; experiment script unchanged.
- Final Ruff, formatting, strict mypy and whitespace checks pass. The dependency
  lock is unchanged. Docs build with the six existing links outside the docs tree.
- Report/checkpoint/source/script hashes verified. Every unchanged first choice
  produces an identical full answer. No final-test access, retraining, production
  setting changes, commit or push. All launched model/review work completed.

**Next**
- Test continuation and completion constraints in combination with learned
  guidance using declared comparisons; individual improvements do not establish
  that the combination works. Keep GPU campaigns sequential and use parallel
  agents for independent code review, diagnostics and documentation tasks.
- Oracle parity, protected finalization and matched-quality serving measurements
  remain open.

## 2026-09-24 — Learning iteration 15: combined guidance and uniqueness

**Done**
- Completed 1,998 new validation executions and reused 1,998 provenance-checked
  observations for the factorial comparison. All 666 unique-only baselines replay.
- Both combined strengths pass the declared candidate gate. Selected alpha 16
  plus uniqueness: 666/666 valid, terminated, repeat-free outputs; mean processed
  F1 88.40%, exact accuracy 68.17%. No serving promotion yet.
- Uniqueness preserves every guidance-only exact answer while repairing both
  loops. Their resulting F1 remains low; coverage is still incomplete.
- Parallel analysis found TYPE accounts for 203/214 alpha-4 residual failures;
  most start with a valid member. Parallel tests cover the experiment gates.
- Added portable evidence, verified chart, Lesson 16 and navigation/overview docs.

**Verification**
- Full suite: 241 passed, one existing warning. Ruff/format/strict mypy passed.
- All first choices match corresponding guidance-only runs; every first divergent
  original token was a repeat. Source/script/report/checkpoint hashes and
  recomputed metrics verify. All launched work completed; no commit or push.
- Final lint, formatting, strict typing and whitespace checks pass. Current source
  and lock match the executed archive. Docs build with six existing external-file
  link warnings; the chart was visually checked after correcting an encoding issue.

**Next**
- Integrate the selected policy as explicit default-zero guidance for serial and
  batched generation; prove full token parity before a real HTTP comparison.
- Strengthen HTTP verification to compare token IDs and decoding descriptors.
- Investigate TYPE continuation/coverage. Oracle parity, protected finalization
  and matched-quality serving/concurrency/energy measurements remain open.


## 2026-09-24 — Learning iteration 16: verified guided application path

**Done**
- Integrated finite default-zero `eval.first_target_guidance_alpha` into shared
  serial/batch generation, CLI evaluation and native HTTP. Positive guidance
  requires constrained cached decoding and the symmetric relation head.
- Strengthened HTTP verification to compare full token IDs, policy/config and
  deployment identity, processed targets and metrics; legacy partial evidence
  is labeled explicitly. Added full token IDs to normal evaluation reports.
- All 1,332 offline replays and 666 real HTTP comparisons pass across three
  frozen checkpoints. Metadata checks pass and every temporary server stopped.
- Added portable receipts and Lesson 17 with intuition, equations, tensor shapes,
  implementation lessons and measured results; updated navigation and registry.
- Parallel workers implemented inference and verifier paths, reviewed integration
  and updated docs while GPU experiments ran sequentially.

**Verification**
- Full suite: 315 passed, one existing warning. Ruff/format/strict typing and
  Torch-blocked core imports pass. Full-token and metric parity, hashes and
  unchanged model/checkpoint identities verify. No source changed during runs.
- Policy quality is preserved: 454/666 exact answers, mean processed F1 88.40%.
  No final-test access, retraining, commit or push.

**Next**
- Diagnose and improve TYPE continuation/coverage (153/357 exact, versus COLOR
  301/309). Keep comparisons tied to declared validation evidence.
- Oracle parity, protected finalization and matched-quality serving/concurrency/
  energy measurements remain open. Guidance stays opt-in; no service left running.


## 2026-09-24 — Learning iteration 17: dual-TYPE union diagnosis

**Done**
- Added CPU coverage analysis and frozen teacher-prefix diagnostics with immutable
  evidence and source/script/data/checkpoint checks. Parallel workers handled
  structural analysis, focused tests and independent result verification.
- Dual TYPE is 29/204 exact versus single TYPE 124/153 and COLOR 301/309.
  Seventy-five dual failures return exactly one complete attribute branch.
- The head recognizes 99% of omitted dual-TYPE targets, but is only 10/204 exact.
  Correct supplied prefixes improve local continuation: 51/204 correct first
  choices versus 201/204 midpoint choices. These are diagnostic, not generated,
  results; current overall generated quality remains 454/666 exact.
- Added portable receipts, Lesson 18, verified figure, navigation and registry.
- Declared the next training experiment: remaining-set supervision at early
  continuation positions, with fixed-budget three-seed controls. Not implemented yet.

**Verification**
- Full suite 349 passed, one existing warning; independent audit verifies 666
  query-seed observations, 3,352 probes and 9,390 condition scores.
- Source/weights and application policy unchanged. No protected test, training,
  service launch, commit or push. All model and analysis jobs completed.

**Next**
- Implement and test the declared continuation-set objective, then run fresh
  matched-source control/candidate training and full guided validation comparisons.
- Preserve exact-generation gates and report subgroup regressions. Oracle parity,
  protected finalization and matched-quality serving/energy measurements remain open.


## 2026-09-24 — Learning iteration 18: continuation objective and matched trial

**Done**
- Implemented default-zero early remaining-set supervision using the existing
  projection/embeddings; no new parameters or label-free inference computation.
  Added eligible-query-weighted logs, config/identity/resume/causality coverage.
- Completed six fresh 2,000-step control/candidate runs with fixed guided-unique
  validation. Mean F1 improves90.12%->91.52%, but exact answers fall466->463/666;
  dual TYPE34->32/204 and COLOR303->299/309. All candidates complete validly.
- Declared gate fails. Preserved the experimental feature, default zero and the
  prior serving reference. Captured33gains/36losses and concrete failure cases.
- Added38campaign-audit tests and independent CPU token/set verification; resolved
  Windows config path spelling through exact recipe reconstruction. Added portable
  receipt, Lesson19, verified chart and updated navigation, architecture and registry.
- Parallel ownership covered implementation and independent audit; GPU runs stayed
  sequential. All launched jobs finished; no service, final test, commit or push.

**Verification**
- Full suite406passed, one existing dependency warning; Ruff/format/strict typing
  and whitespace checks pass. Docs build passes with six known source-link warnings.
- Both independent audits agree on all1,332response metrics and the failed gate.
  Current core/source/lock and analysis files match their executed archives.

**Next**
- Compare first free-generation divergences with teacher-prefix/remaining-set
  predictions before choosing another objective or coefficient.
- Oracle parity, protected finalization and matched-quality serving/concurrency/
  energy measurements remain open. Failed trials remain part of the learning record.


## 2026-09-24 — Learning iteration 19: generated versus teacher prefixes

**Done**
- Added CPU matched-error analysis and frozen prefix probes with 28 focused tests.
  Preserved raw/processed position maps, condition-specific guidance, seen-token
  unavailability, head scores and atomic immutable evidence.
- Completed1,332query observations /7,980paired probes /15,960conditions. Every
  sampled raw next choice matches archived generation; independent audit agrees.
- Found two gaps: candidate dual-TYPE early head F1 is94–95% with teacher prefixes
  versus75–77% with generated ones; a correct prefix and98.7% head F1 can still
  select the wrong next product. Subject insertion and simple early EOS do not
  explain most regressions. Previous weights, quality and failed gate remain.
- Added Lesson20, portable receipts, checked figure, navigation/registry/log updates;
  corrected the protocol doc's stale statement that guided integration was pending.
- Declared weight8 first-target trial against both existing matched references.
  No new training in this session; GPU work was sequential, analysis/review parallel.

**Verification**
- Full suite434passed, one existing warning; Ruff/format/strict typing/whitespace
  checks pass. Docs build passes with six known source-link warnings.
- Source/lock and scripts/helpers match executed archives; all jobs completed.
  No protected test, service, commit or push.

**Next**
- Execute the declared three-seed first-target-weight interaction with fixed
  source/data/inference, report both references and apply the original-control gate.
- Continue addressing later ordered continuation, oracle parity, protected
  finalization and matched-quality serving/concurrency/energy evidence.

## 2026-09-24 — Learning iteration 20: first-target weight8 trial

**Done**
- Completed three declared training/evaluation runs with fixed source, data and
  decoding. Weight8 yields464/666 exact answers and89.42% meanF1, versus the
  original466/666 and90.12%. Dual TYPE falls34->31/204; quality gate fails.
- Added the CPU order-campaign auditor and10 regression tests, with a
  default-preserving optional first-weight argument in the existing auditor.
  Independent audit agrees on all9checkpoints/1,998outputs and both comparisons.
- Documented lower final-batch first-token loss but worse validation first-token
  CE in all seeds, membership versus ordering, and severe paired failures.
- Added Lesson21, portable receipts, navigation/overview/architecture updates,
  registry entry and research rationale. GPU work sequential; audits parallel.

**Verification**
- Full444tests pass, one existing dependency warning; Ruff/format/strict typing
  pass. Docs build passes with six existing source-link warnings.
- Source and runtime match frozen controls; new analysis helpers are separately
  archived. All training/evaluation/audit jobs finished. No protected test,
  service launch, commit or push; no candidate promotion.

**Next**
- Declare an early-prefix/decoding diagnostic to distinguish available good
  continuations from the ability to select them using learned scores.
- Oracle parity, protected finalization and matched-quality serving/concurrency/
  energy remain unfinished. Weight8 does not advance those acceptance claims.

## 2026-09-24 — Learning iteration 21: four first choices and selection

**Done**
- Added a frozen first-choice branching diagnostic and 13 focused tests. It
  scores four independent greedy paths per query, preserving first/EOS scores,
  token budgets, caches, deterministic ties and truth-independent selection.
- Completed 2,664 paths across three original models; all 666 rank1 sequences
  match the reference. Independent audit agrees on scores, outputs and receipts.
- Found 501/666 exact answers available, but sum/mean selection returns only
  462/465 versus greedy466. Both gates fail; all ten sum losses are dual TYPE.
- Added Lesson22, checked PNG/SVG figure, portable evidence, registry/navigation/
  overview updates, and the next reviewed frozen-set reranking declaration.

**Verification**
- Full457tests pass, one existing dependency warning; Ruff/format/strict typing
  pass. Documentation builds with six existing source-link warnings.
- Source and runtime match frozen controls. All GPU and audit jobs finished.
  No training, production decoder change, protected test, commit or push.

**Next**
- Run the declared symmetric-head selector over these same four candidate sets;
  no threshold tuning or regenerated candidates. Compare against original greedy.
- Oracle parity, protected finalization and matched-quality serving/concurrency/
  energy evidence remain open; availability alone is not a quality claim.

## 2026-09-24 — Learning iteration 22: successful set ranking

**Done**
- Added the frozen-candidate symmetric membership selector and 19 focused tests.
- Completed all three checkpoint evaluations: 501/666 exact answers versus 466,
  with 35 gains, zero exact losses and mean F1 improving from 90.12% to 95.56%.
- Independently audited every saved score, selection, output and result receipt.
- Added Lesson 23, checked figure, portable evidence, registry/navigation and
  project overview updates. Reviewed the next application integration contract.

**Verification**
- Full 476 tests pass, with one existing dependency warning; Ruff/format and
  strict typing pass. Model/config/lock source and training checkpoints unchanged.
- GPU and audit jobs finished. No protected test, commit, push or deployment.

**Next**
- Integrate the default-off policy and replay all candidate tokens and selected
  outputs through offline and native HTTP paths against the frozen evidence.
- Improve the remaining 165 inexact observations; oracle parity, protected final
  evaluation and matched-quality serving/concurrency/energy evidence remain open.

## 2026-09-24 — Learning iteration 23: shared set selector and full offline parity

**Done**
- Added default-off symmetric set reranking across shared generation, CLI and
  native serving, with trained-checkpoint checks and preserved request metadata.
- Added authenticated one-field historical configuration adaptation; retained
  separate training and effective inference identities.
- Full 514 tests, Ruff/format and strict typing of 60 source files pass.
- All 2,664 candidate paths/scores, 666 selected outputs and 666 disabled greedy
  outputs match frozen references across three checkpoints. Independent partial
  CPU audit confirms all offline comparisons and config bridges.
- Added Lesson 24, registry/navigation/overview changes, portable candidate
  coverage diagnosis and a reviewed, unexecuted pair-union experiment contract.
  Diagnosis: 69/165 failed observations admit exact pair unions; blind full union
  loses 495 exact answers. These are oracle-assisted diagnostics only.

**In progress / next**
- Live verifier session 22905 is running real HTTP checks. Keep source/config/
  script/plan frozen, resume that handle, and finish all three HTTP checkpoints,
  metadata/failure checks, terminal independent audit and final documentation.
- No full integration acceptance yet. No protected test, commit or push.
- Pair-composition experiment is declared but not implemented or run. Oracle
  parity and matched-quality serving/concurrency/energy remain open.

## 2026-09-24 — Learning iteration 24: pair-composition result and first HTTP pass

**Done**
- Added CPU-only frozen pair-composition experiment and40 focused passing tests.
- Replayed all666 original scores/choices; measured all6,660 slots. Learned
  composition reaches569/666 exact sets versus501, with68 gains/zero exact losses,
  mean F1 .96689605. Independent audit confirms every selection and receipt.
- Added Lesson25, portable results, checked PNG/SVG, registry and navigation/
  overview updates. Documented the one PILOSWINE F1 regression and97 remaining
  inexact observations. Pair composition remains offline only.
- First four-path native HTTP checkpoint passes all222 responses and metadata/
  failure checks; independent partial audit confirms the result.

**Verification / next**
- New40 CPU tests and Ruff/format pass. Prior full514 suite passed; the next full
  suite run must wait for the live GPU campaign to finish (suite includes CUDA).
- Resume live session22905, not a new run. Two HTTP checkpoints plus final audit
  and final Lesson24/serving documentation remain. Core/config/harness stay frozen.
- A separate composition integration contract is being drafted, without code
  changes. No protected test, new training, commit or push. Oracle parity and
  matched-quality serving/concurrency/energy work remain unfinished.


## 2026-09-24 — Iteration 25: isolated composition implementation

- Staged core/config, explicit set CLI metrics and separate native set endpoint
  in `../plm-pair-composition-staging`; active source/config remain unchanged.
- Parallel ownership covered core, transport, root evaluation/docs and replay
  harness. Registry updated in staging; lesson 26 is available in both checkouts.
- CPU suite: 602 passed, 9 skipped. Two review fixes passed 3 integration + 2
  metrics cases afterward. Ruff/format and strict mypy (62 source files) pass.
- Preserved tiny CPU cross-batch rounding differences; only that comparison uses
  a narrow numeric tolerance. Actual serial HTTP evidence matches the CLI.
- Existing GPU session22905 continues, HTTP1730 at200/222; third checkpoint and
  independent terminal audit pending. Next: complete prior audit, incorporate
  staged changes, then run declared pair-composition replay. No commit/push.

- Session follow-up: HTTP1730 finished and independently passed; two of three
  HTTP checkpoints complete. HTTP1731 and the terminal audit remain pending.

- Final staging verification: 639 passed, 9 skipped; 149 files lint/format clean;
  strict mypy 62 files. Saved-evidence replay exact for all 666 queries and
  6,660 slots. New replay harness ready; old HTTP1731 at 50/222, so no main
  source incorporation or new GPU campaign yet.

## 2026-09-24 — Iteration 26: accepted baseline, composition replay and error lesson

- Completed and independently audited the preceding four-path application
  campaign: all 666 HTTP responses, 2,664 candidate paths and compatibility/
  metadata/failure checks pass. Added its portable acceptance receipt and updated
  Lesson 24. The baseline then passed all 554 tests.
- Incorporated 17 staged files with verified before/after hashes. Main now passes
  661 tests including CUDA, Ruff/format over 149 files and strict mypy over 62.
  Replay harness review added preserved partial reports and strict HTTP schemas;
  50 focused harness tests pass, and preflight authenticates 138 inputs.
- Started the new frozen three-checkpoint composition replay, session20965.
  Offline seeds1729/1730 independently pass with191/192 exact answers; seed1731
  is running. Source archive1d74e018...de5376 matches all64 live members. HTTP and
  terminal acceptance remain pending. Keep GPU jobs sequential and source/config/
  harness/plan frozen; a serial-versus-batch score discrepancy must be diagnosed.
- Independently diagnosed all97 saved pair-selection errors. Only one has an
  exact candidate available;96 require a different candidate pool. Deleting every
  negative-scored member would lose62 currently exact answers and can recover at
  most17, so exact count is at most524. This is a logical bound, not a new policy
  measurement. Added Lesson27, checked figure and portable evidence. Independent
  review confirmed equations/counts and clarified the bound's exact scope in JSON;
  the original generation receipt and editorial revision receipt are retained.
- Documentation builds with six existing source-link warnings. Updated research
  rationale, teaching navigation and application status. No training, protected
  final-test evaluation, commit or push. Next: finish session20965, independently
  audit complete reports, diagnose any exact-score discrepancy, then select the
  next declared coverage experiment. Oracle parity and serving benchmarks remain
  open.

## 2026-09-24 — Iteration 27: direct-head ablation and numerical replay diagnosis

- Added the declared CPU-only direct-membership evaluator, registry entry and
  23 passing focused tests. Preflight authenticates 102 inputs and reconstructs
  the 569/666 pair baseline. Independent audit verifies all 666 new predictions,
  metrics, member changes and gates without importing the evaluator.
- Direct `z > 0` prediction fails its gate: 339 exact versus 569, three gains and
  233 losses, despite F1 improving from 96.69% to 98.96%. Retained this negative
  result without threshold tuning or runtime promotion. Added Lesson 28, portable
  evidence, visually checked chart, navigation and research rationale.
- Pair-composition session20965 is terminal (exit 1), not still running. All
  three offline checkpoints independently pass. HTTP seed1729 saves all 222
  responses but fails exact score equality; seeds1730/1731 were not executed.
  Independent failure audit confirms all 888 source paths and 222 answers/slots
  match, with all 2,220 normal slot scores differing (max absolute .00036144).
  Metadata, legacy/disabled/failure/admission and all shutdown receipts pass.
- Kept original failed reports and gate intact. Added a portable failure receipt
  and updated application documentation. A separately declared prompt-head
  batch-eight versus serial diagnosis is being prepared against the same
  checkpoint/source. No new generation or training belongs to that diagnosis.
- Evaluator lint/format and documentation build pass; the six existing source-link
  warnings remain. Full suite was not rerun during GPU work; prior 661 tests plus
  the new 23 focused tests are separate evidence. No protected test, commit or push.
  Next: resolve numerical cause, declare appropriate verification with exact
  scores for matching batch shapes, and finish the remaining application checks.

- Final follow-up: the bounded neural diagnosis and its independent audit pass.
  Batch-eight logits exactly reproduce saved offline evidence; serial logits
  exactly reproduce all 2,220 HTTP scores. All 222 selected slots remain the
  same, and first-eight repeats match within each shape. No tolerance introduced.
- Full suite exposed one test-process isolation bug, fixed by running the
  synthetic CLI drift scenario in a fresh interpreter. Final verification:
  684 tests pass; Ruff passes; 151 files formatted; strict mypy passes 62 source
  files; docs build with six existing source-link warnings. No GPU job remains
  running. Original failed campaign and successful diagnostic receipts retained;
  next work is the revised shape-aware verification contract and remaining HTTP
  checkpoints, followed by further quality research. Goal remains incomplete.
