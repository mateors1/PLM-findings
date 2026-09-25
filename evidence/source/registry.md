# Symbol Registry

A human-navigable index of every public symbol, DB table, config field, CLI
command, and protocol token in the project. **Purpose: DRY.**

- **Before** writing a new function/class/constant/table/token, search here first.
- **After** adding, renaming, or removing a public symbol, update this file in the
  same change. A stale registry is a bug.

Legend: ✅ implemented · 🚧 stub / planned · ⬜ empty package (reserved).

---

## Package map (`src/plm/`)

| Module | Responsibility | Status |
| --- | --- | --- |
| `plm` | Package root; version. | ✅ |
| `plm.cli` | Typer CLI, entry point `plm`. | ✅ (foundation cmds) |
| `plm.config` | Typed config schema (Pydantic v2). | ✅ |
| `plm.reproducibility` | Deterministic seeding (Python/NumPy/Torch). | ✅ |
| `plm.graph` | SQLite product graph — schema, repository, and National Dex importer. | ✅ |
| `plm.protocol` | Versioned protocol config + tokenizer / vocabulary. | ✅ |
| `plm.model` | Dense decoder, optional sparse MoE/activation checkpointing, interfaces, masks, overfit gate. | ✅ Phase C reference + experiments |
| `plm.corpus` | Deterministic corpus compiler (graph → protocol sequences + manifest). | ✅ |
| `plm.baselines` | Oracle, popularity, and optional ComplEx ranking scorers. | ✅ Phase B |
| `plm.evaluation` | Deterministic splits, shared metrics, PLM scorer, validation/final evaluators. | ✅ Phase C |
| `plm.training` | Dataset/collation, optimizer/scheduler, trainer, checkpoints, resume gate. | ✅ Phase C reference |
| `plm.serving` | Shared inference loader, native HTTP API, constrained decoding, output policy, frozen hydration and bounded execution. | ✅ Native API verified; performance campaign pending |
| `plm.experimentation` | AVO identity, evidence, comparison, firewall, measurement, lineage. | ✅ AVO scaffold |
| `plm.configuration` | Authoritative Hydra composition and resolved config hashing. | ✅ |
| `plm.observability` | Metrics / logging. | 🚧 |
| `plm.labeling` | Local FastAPI TYPE/COLOR curation UI. | ✅ Phase B |

---

## Public API by module

### `plm`
- `__version__: str` — `"0.1.0"`.

### `plm.config` — Pydantic models (all forbid unknown keys)
- `validate_config(raw: dict) -> RootConfig` — validate a resolved config dict.
- `RootConfig` — top level: `seed, deterministic, run_name, paths, data, model, train, eval, baseline`.
- `PathsConfig` — `data_root, run_root`.
- `DataConfig` — `name, protocol_version, tokenizer_version, graph_db, corpus_manifest, validation_query_fraction, test_query_fraction, split_seed`; legacy `held_out_query_fraction` is migration-only.
- `ComplExConfig` — `dimension, epochs, learning_rate, negative_samples, regularization, batch_size`.
- `BaselineConfig` — nested `complex: ComplExConfig`.
- `ModelConfig` — the over-engineered decoder stack (vocab/shape/norm/ffn/rope/attention/moe/numerics). Props: `kv_groups`, `resolved_head_dim`, `ffn_hidden_dim`.
- `ModelConfig.first_target_loss_weight` — finite positive first-supervised-token
  CE weight (default 1); objective normalized by total effective token weight.
- `ModelConfig.prompt_set_loss_weight` — finite nonnegative auxiliary set-loss
  coefficient (default 0/off); enabled path requires banded product vocabulary.
- `RoPEConfig` — `theta, scaling_type{none,linear,ntk,yarn}, scaling_factor, original_max_seq_len`.
- `AttentionConfig` — `backend{sdpa,flash,eager}, qk_norm, causal, sliding_window, logit_soft_cap, dropout`.
- `MoEConfig` — `enabled, num_experts, experts_per_token, aux_loss_weight, shared_experts`.
- `OptimizerConfig` — `name{adamw,adamw_fused,lion}, lr, weight_decay, betas, eps, decay_excludes_norm_and_bias`.
- `SchedulerConfig` — `name{cosine,linear,constant,wsd}, warmup_steps, min_lr_ratio`.
- `TrainConfig` — steps/batch/precision/grad-accum/compile/cadence + optimizer + scheduler;
  the reference path supports fp32/BF16, rejects fp16/compile/EMA until their
  checkpoint contracts are implemented.
- `EvalConfig` — `metrics{mrr,hits,map}, hits_at, filtered, max_new_tokens,
  constrained_decoding, use_kv_cache, prevent_repeated_targets, generation_batch_size,
  first_target_guidance_alpha, symmetric_set_reranking, pair_set_composition,
  device{auto,cpu,cuda}`;
  generation bound must be positive. Target uniqueness defaults off and requires
  protocol constraints; it masks emitted targets only, never the prompt subject or EOS.
  `generation_batch_size` is a strict positive integer, default 1; larger offline
  evaluation groups require constrained KV decoding. Native serving requires 1.
  `use_kv_cache` defaults false and selects request-local cached generation only.
  `first_target_guidance_alpha` is finite/nonnegative, defaults 0, and requires
  constrained KV decoding when positive. Loading positive guidance requires a
  trained symmetric relation head; inference config is separate from training identity.
  `pair_set_composition` is a strict boolean, default false; requires symmetric
  reranking and its trained-head/constrained/unique/KV prerequisites. It enables
  a separate set-evaluation report and native set endpoint, preserving raw APIs.

### `plm.configuration`
- `load_config(config_name="config", *, config_dir=None, overrides=()) -> RootConfig` — compose Hydra, resolve interpolations, and validate once.
- `resolved_config_dict(config) -> dict` / `config_hash(config) -> str` — canonical experiment configuration identity; timestamped `run_name` is excluded.
- `validate_saved_config(raw, expected_hash) -> RootConfig` — authenticate historical
  canonical JSON before parsing; only missing `eval.symmetric_set_reranking` and
  `eval.pair_set_composition` false defaults may be inserted. Reject unrelated coercions/defaults; retain historical
  training hash separately from the effective inference configuration hash.

### `plm.evaluation`
- `QuerySplit` — immutable ordered train/validation/test query partition with `split_hash`; legacy evaluation aliases target test only.
- `split_queries(records, validation_query_fraction, test_query_fraction, seed) -> QuerySplit` — SHA-256 deterministic query split; the train fraction is the remainder.
- `Scorer` — protocol returning a score for `(CorpusRecord, candidate)`.
- `RankingMetrics` — macro `mrr`, `hits_at`, and `map` results.
- `evaluate_ranking(records, candidates, scorer, *, ks=(1,3,10)) -> RankingMetrics` — shared multi-positive evaluator.
- `PLMScorer` — next-token scorer; caches logits once per subject/dimension query.
- `PLMEvaluation.protocol_valid` — unknown (`None`) without generation evidence.
- `GenerationMetrics` — `query_count, precision, recall, f1, exact_set_accuracy,
  exact_sequence_accuracy, protocol_valid_rate, termination_rate,
  duplicate_query_rate, self_return_rate, mean_target_count`; `to_dict()` serialization.
- `evaluate_generation(records, generate) -> GenerationMetrics` — macro query
  response metrics; duplicates consume precision denominator, exact requires EOS/validity.
- `SetCompositionMetrics` / `evaluate_set_compositions(records, results, vocabulary)`
  in `evaluation/set_composition.py` — macro set precision/recall/F1, exact count
  and accuracy, mean successful answer size over all queries, eligibility,
  failures and successful original/pair counts. Failed answers get zero quality
  and size; `source_path_*` diagnostics refer only to four actual paths per query.
  `to_dict()` serializes metrics; no composed sequence/protocol/termination metric.
  Fields: `query_count, precision, recall, f1, exact_set_accuracy, exact_set_count,
  mean_set_size, selected_source_eligibility_rate, failure_count,
  original_selection_count, pair_selection_count, source_path_count,
  source_path_protocol_valid_rate, source_path_termination_rate,
  source_path_error_count, source_path_duplicate_count`.
- `evaluate_validation(...)` / `evaluate_final_test(...)` — split-aware evaluator paths; test is finalization-only.
- `evaluate_repeated(...)` — repeated validation metrics with mean/stddev/min/max.

### `plm.baselines`
- `OracleScorer` — full-corpus oracle scorer.
- `PopularityScorer` — dimension-specific training-target frequency scorer.
- `ComplExScorer` — optional lazy-Torch knowledge-graph embedding scorer.

### `plm.labeling`
- `create_labeling_app(db_path) -> FastAPI` — local TYPE/COLOR curation application; saves atomic curated provenance-backed replacements.

### `plm.reproducibility`
- `seed_everything(seed: int, *, deterministic: bool = False) -> None` — seed all RNGs; optional deterministic-algorithms mode.
- `seed_worker(worker_id: int) -> None` — `DataLoader(worker_init_fn=...)` reseeder.

### `plm.graph.schema`
- `SCHEMA_VERSION: int` — `1`.
- `connect(db_path) -> sqlite3.Connection` — open with FK enforcement.
- `init_graph(db_path, *, protocol_version="0.0.0-dev") -> None` — create schema (idempotent) + stamp `meta`.
- `validate_graph(db_path) -> ValidationReport` — referential-integrity + counts check.
- `ValidationReport` (dataclass) — `node_count, edge_count, relation_type_count, schema_version, protocol_version, issues`; prop `ok`.

### `plm.graph.repository`
- `GraphRepository(db_path, *, create=False)` — context-managed typed CRUD (upserts).
  - `add_provenance(source_type, *, citation=None, license=None, notes=None) -> int`
  - `add_node(key, *, kind="product", label=None, namespace="PKM", status="active", provenance_id=None) -> int` — upsert; `provenance_id` preserved if omitted.
  - `add_relation_type(name, *, category=None) -> int` — `category` preserved if omitted.
  - `add_edge(src_key, dst_key, relation, *, weight=1.0, confidence=1.0, is_canonical=True, provenance_id=None) -> None`
  - `add_alias(node_key, alias, *, locale="", priority=0) -> None` — deferred use in v1.
  - `node_id(key) -> int`
  - `node_keys(*, kind=None) -> list[str]` — keys sorted by key; filter by kind (e.g. `'product'`).
  - `counts() -> tuple[int, int, int]` — `(nodes, edges, relation_types)`
  - `close()` / context-manager protocol

### `plm.protocol.tokenizer`
Banded, append-only vocabulary (ID map in the module docstring).
- Band constants: `CORE_BAND_END=32`, `STEERING_BAND_END=128`, `ENTITY_BASE=1024`.
- Class constants: `CLASS_SPECIAL`, `CLASS_DIMENSION`, `CLASS_MODE`, `CLASS_ENTITY`, `CLASS_RESERVED`.
- Token sets: `DEFAULT_SPECIALS=("PAD","BOS","EOS","UNK","SEP","ANSWER")`, `DEFAULT_DIMENSIONS=("TYPE","COLOR")`, `DEFAULT_MODES=("SAME",)`, `DEFAULT_VERSION="pokemon-v1"`.
- `UnknownTokenError(KeyError)` — raised by strict `encode`/`id_of` on an unknown token.
- `Vocabulary(tokens, classes, *, version=DEFAULT_VERSION)` — frozen token↔ID map + per-token class.
  - `Vocabulary.build_fresh(entities, *, dimensions=…, modes=…, specials=…, version=…)` / `Vocabulary.build(...)` — lays out the bands; entities sorted; **predicates not accepted** (not model-visible).
  - `Vocabulary.extend(additions, *, version=None)` — appends entity IDs without renumbering existing tokens.
  - `id_of(token) -> int` (strict, raises) · `token_of(id) -> str`
  - `encode(tokens, *, strict=True) -> list[int]` (strict = fail-closed; `strict=False` → UNK) · `decode(ids)`
  - `class_of(token_or_id) -> str` · `entity_ids() -> list[int]` (legal generation set)
  - props: `pad_id, bos_id, eos_id, sep_id, answer_id, unk_id, version`
  - `content_hash()` (SHA-256 over version+tokens+classes) · `to_dict()` · `save(path) -> str` · `Vocabulary.load(path)` (file `schema_version=2`)

### `plm.protocol.config`
- `ProtocolSpec` — `version, dimensions, modes, target_order` contract with executable target ordering and supported-mode validation.
- `UnsupportedProtocolError` — fail-closed unsupported protocol semantics.
- `load_protocol(path) -> ProtocolSpec` — load and validate protocol YAML.

### `plm.corpus.compiler`
- `CorpusRecord` — `subject, dimension, targets, input_ids, labels`; labels are loss-masked through `ANSWER`.
- `CorpusManifest` — corpus identity: protocol/tokenizer/graph/records hashes and counts.
- `compile_corpus(graph_db, *, protocol, output_dir) -> CorpusManifest` — compiles deterministic v1 JSONL records, vocabulary, and manifest.
- `canonical_graph_hash(conn) -> str` — content hash over graph nodes, relations, and semantic edges.
- `validate_corpus_artifacts(...) -> ArtifactValidationReport` / `require_valid_artifacts(...)` — fail-closed manifest, vocabulary, records, and optional graph validation.

### `plm.corpus.snapshot`
- `prepare_dataset_snapshot(data_root, name, *, protocol_path, fetch_text=...) -> Path`
  — pinned source import, graph validation, byte-stable double compilation,
  exclusive publication, nine-file hash receipt. Reverification is offline;
  receipt-only reconstruction must match all identities before publication.
  Partial/tampered snapshots are never overwritten.

### `plm.graph.importers`
- `POKEAPI_CSV_BASE_URL: str` — pinned PokeAPI CSV source root.
- `POKEAPI_SOURCE_REVISION: str` — immutable PokeAPI commit used by the default importer.
- `import_national_dex(db_path, *, protocol_version, base_url=…, source_revision=None,
  fetch_text=…, snapshot_imported_at=None) -> tuple[int, int]` — imports every
  National-Dex species with hidden TYPE/COLOR facts; optional snapshot timestamp
  reproduces original import metadata; returns product/edge counts.

### `plm.model.architecture`
- `build_model(config: ModelConfig) -> nn.Module` — lazy optional-Torch reference decoder; unsupported advanced switches fail loudly.

### `plm.model`
- `ModelInput` / `ModelOutput` — torch-light forward contracts; input-aligned labels
  are shifted exactly once inside the decoder for next-token loss.
- `ModelOutput.task_loss` — token cross-entropy before penalties; `loss` is total;
  `aux_loss` is mean router penalty; `expert_load` has shape `[layers, experts]`.
- `ModelOutput.first_target_loss` — unweighted first-supervised-token CE;
  `prompt_set_loss` — query-averaged, positive/negative-balanced binary loss;
  `prompt_set_logits` — `[batch, product_count]` logits from causal ANSWER state.
- `PLMDecoder.prompt_set_projection` — optional bias-free `[dim, dim]` projection,
  scored against tied input product embeddings; initialized after common weights.
  It does not select, filter or reorder generated tokens. Set labels use supervised
  product IDs only, collapse duplicates, and exclude prompt/padding/control/EOS.
- `ModelConfig.continuation_set_loss_weight` — finite nonnegative default-zero
  coefficient; positive requires the existing prompt-set objective. Reuses its
  projection after one/two/three teacher targets to supervise remaining members.
- `ModelOutput.continuation_set_loss` / `.continuation_set_query_count` — balanced
  loss averaged over valid stages per query, then participating queries; empty
  participation gives differentiable zero/count zero. No new parameters.
- `Trainer.validation_metrics["validation_continuation_set_loss"]` and
  `validation_continuation_set_query_count` — eligible-query-weighted diagnostic;
  history also records `last_batch_continuation_set_loss` and
  `last_batch_continuation_set_query_count`. Token CE remains separate.
- `ModelConfig.symmetric_relation_loss_weight` — finite nonnegative coefficient,
  default 0. Adds a shared-embedding symmetric relation objective; no inference filtering.
- `PLMDecoder.symmetric_relation_projection` — optional `[dim, dim]` bias-free
  map from normalized dimension-token embedding to a diagonal relation operator.
  Normalized product embeddings appear on both sides; FP32 score matrix `[B, N]`.
- `ModelOutput.symmetric_relation_logits` / `.symmetric_relation_loss` — diagnostic
  scores and query-balanced membership loss with the subject column excluded.
  Uses only prompt IDs and training target labels; no future-token inputs or graph lookup.
- `Trainer.validation_metrics["validation_symmetric_relation_loss"]` and history
  `last_batch_symmetric_relation_loss` — separate diagnostics for the new objective.
- `RMSNorm`, `RoPE`, `SwiGLU`, `CausalSelfAttention`, `DecoderBlock`, `PLMDecoder` — reference dense model components (training group).
- `KVCache(layers, position, owner)` / `DecodeOutput(logits, cache)` — torch-light
  ephemeral inference contracts; cache holds unrepeated key/value pairs by layer.
- `PLMDecoder.decode(input_ids, *, cache=None)` — unpadded inference chunks with
  absolute positions, owner/shape/context checks; token head only, no training loss.
- `CausalSelfAttention.decode(hidden, past=None)` / `DecoderBlock.decode(hidden, past=None)`
  — inference-only cached attention/block execution sharing the reference math.
- `SparseMoE(config)` — top-k SwiGLU routing with FP32 router probabilities,
  optional shared experts, padding exclusion, no token dropping; `forward`
  returns `(hidden, auxiliary_loss, expert_load)`. `DecoderBlock.forward` now
  returns the same triple, with zero auxiliary/empty load for dense blocks.
- `PLMDecoder` — optional activation checkpointing in training with dropout RNG
  preservation, optional MoE penalty; dense remains default. RoPE uses half-pair
  frequency concatenation to preserve rotations and relative-position invariance.
- `apply_legal_entity_mask(logits, vocabulary)` — separate class-aware emission mask.
- `run_synthetic_overfit(config, ...) -> SyntheticOverfitReport` — Development AVO hard gate.
- `SyntheticOverfitReport` — `initial_loss, final_loss, steps, threshold,
  generated_ids, expected_ids`; `passed` requires low loss and exact prompt-only completion.

### `plm.training`
- `CorpusDataset` / `collate_records` — label-preserving padded batches.
- `create_optimizer` / `create_scheduler` — config-driven factories.
- `Trainer` / `TrainResult` — reference CPU/CUDA training loop; sequential resume
  restores batch position accounting for gradient accumulation, with loader RNG
  isolated from model RNG. Checkpoint `training_metadata` records `objective`,
  `data_order`, `record_count`, `batch_size`, `grad_accum_steps`, `model_config`, and `train_loss`;
  missing or incompatible objective/order contracts are rejected on resume.
- `run_training` — resolved model/corpus overrides bound to identity; writes
  `run.json`, deterministic `source.zip`, `training-result.json`. Trainer writes
  `metrics.jsonl`; validation uses token-weighted CE, excluding router penalties.
  Completed results/checkpoints cannot be overwritten; existing metrics require
  explicit resume. Result includes `training_wall_seconds` (validation/checkpoint
  overhead included; not an isolated throughput benchmark).
- `Trainer.validation_metrics` — query-weighted first-target CE and optional set
  loss; history also records last-batch first-target and set losses separately.
- `save_checkpoint` / `load_checkpoint` / `validate_checkpoint_identity` — checkpoint and resume contract.
- `evaluate_resume_equivalence(...) -> ResumeEquivalenceResult` — deterministic resume hard gate.

### `plm.experimentation`
- `capture_runtime_provenance(*, source_archive=None) -> tuple[str, dict[str,str]]`
  — Git commit, working source/lock hashes, Python/Torch/CUDA/GPU environment;
  optional deterministic archive of `src/**/*.py`, `pyproject.toml`, `uv.lock`.
  Core baseline reports record `torch=not-installed` if Torch is absent.
- `ExperimentIdentity` — exact source/config/corpus/split/checkpoint/evaluator/environment identity.
- `CandidateRecord`, `Hypothesis`, `Prediction`, `HardGateResult`, `ObjectiveVector`, `EvaluationReceipt`, `SelectionDecision`, `LineageEntry` — serializable AVO evidence.
- `ComparisonRule` / `compare_objectives` — gate-aware tolerance/Pareto/protected-dimension comparison.
- `assert_candidate_split` / `assert_final_test_split` — test firewall.
- `LineageLedger` — separate `runs/avo/development` and `runs/avo/research` storage.

### `plm.serving`
- `InferenceRuntime` / `load_inference_runtime(config, checkpoint, *, corpus=None,
  graph_db=None, protocol=None)` — shared evaluation/serving loader; validates
  corpus, model/objective, checkpoint payload and training/split identity.
- `HydrationIndex.from_snapshot(graph_db)` / `.hydrate(response)` — frozen active
  product labels, independent of later database edits; no relation inference.
- `PredictRequest` — strict `subject`, `dimension{TYPE,COLOR}`, `mode{SAME}`,
  `ignore: list[str]`, nonnegative optional `limit`; unknown fields rejected.
- `NativePredictor.from_checkpoint(config, checkpoint, *, runtime_dir, corpus=None,
  graph_db=None, max_pending=8)` — SQLite backup, validated runtime, frozen hydration,
  source archive and deployment receipt. `.predict(request)` preserves raw evidence,
  applies metadata only after valid termination, hydrates and times execution.
- `InvalidRequestError`, `GenerationFailure(raw)`, `ServiceBusyError` — entity
  validation, failed generation and admission-limit boundaries.
- `create_serving_app(predictor)` — optional FastAPI `GET /health`, `POST /v1/predict`;
  invalid requests 422, generation failures 502, admission overload 503.
- `PostprocessResult` — processed `response` plus disjoint `removed_subject`,
  `removed_ignored`, `removed_duplicates`, `removed_by_limit` counts.
- `postprocess_response(response, *, ignore=(), limit=None)` — SAME self-exclusion,
  IGNORE, stable deduplication, then nonnegative return bound. No relation lookup,
  missing-target filling or raw-response mutation. Processed targets may be empty.
- `GenerationResult` — `token_ids, targets, terminated, protocol_valid, decoding,
  error`; generated evidence including failures/truncation.
- `generate_response(model, vocabulary, subject, dimension, *, max_new_tokens,
  device="cpu", constrained=True, use_cache=False, prevent_repeated_targets=False,
  first_target_guidance_alpha=0.0, symmetric_set_reranking=False)`
  — prompt-only greedy generation to EOS/bound; optional token-class and target
  uniqueness masks, no oracle filtering or manufactured EOS. `+unique-v1` records
  the distinct decoding policy in generation/evaluation reports and serving receipts.
- `ProtocolResponse`, `parse_generated`, `allowed_token_ids`, `constrain_logits`, `prompt_ids` — native grammar/state machine.
- `generate_responses(model, vocabulary, queries, *, max_new_tokens, device="cpu",
  prevent_repeated_targets=False, first_target_guidance_alpha=0.0,
  symmetric_set_reranking=False)` in
  `serving/generation_batch.py` — fixed-group
  constrained KV decoding; ordered results including duplicate queries, per-row
  seen/finished state, no returned dummy tokens or manufactured EOS. Empty input
  returns an empty tuple. Reports carry `+batch-v1` and the configured batch size.
- `validate_guidance_alpha(alpha)` / `guidance_decoding_suffix(alpha)` in
  `serving/guidance.py` — reject nonfinite/negative/nonnumeric strengths and
  identify positive guidance canonically; zero leaves decoder labels unchanged.
- `RerankedGenerationResult` in `serving/set_reranking.py` — selected generation,
  four candidate generations, canonical set scores, selected rank, eligibility,
  and explicit no-valid-path fallback evidence.
- `generate_reranked_responses(model, vocabulary, queries, *, max_new_tokens,
  device="cpu", first_target_guidance_alpha=0.0)` — four distinct guided first
  products followed by independent unique KV continuations, retaining query-group
  batch shapes; select valid completed SAME-processed sets by symmetric-logit sum.
  No IGNORE/count/oracle inputs; all-ineligible returns rank-one failure. The
  default-false `eval.symmetric_set_reranking` routes normal serial/batch/CLI/native
  generation here; requires constrained unique KV decoding and a trained head.
- `apply_first_target_guidance(model, input_ids, decoded, alpha)` — shared
  prompt-only symmetric-head forward, FP32 log-sigmoid bias to final-position
  product logits. Returns cloned logits with unchanged cache; zero bypasses
  the head. Callers invoke it only on the initial cached prefill.
- `SetCandidateSlot` in `serving/set_composition.py` — frozen `slot, kind,
  source_ranks, set_ids, source_eligible, score`; kind is `original` or `pair_composition`.
- `SetCompositionResult` — frozen `source_paths, slots, selected_slot, selected_kind,
  selected_source_ranks, selected_set_ids, selected_source_eligible,
  fallback_no_valid_source, policy`. No synthetic generation properties.
- `generate_set_compositions(model, vocabulary, queries, *, max_new_tokens,
  device="cpu", first_target_guidance_alpha=0.0)` — reuse four rank batches and
  their final FP32 scoring head pass; compare original slots and six pair unions
  by canonical `math.fsum`, original-first ties, and both-source eligibility.
- `NativePredictor.predict_set(request)` / `POST /v1/predict-set` — default-disabled
  set endpoint sharing native admission/execution limits; filter and hydrate after
  selection and return original source paths/slots. `/v1/predict` remains raw.
- `SetCompositionDisabled` — HTTP 409 `set_composition_disabled` before generation.
- `SetCompositionFailure` — HTTP 502 `set_composition_no_valid_source`, retaining
  `.composition` source evidence without a successful hydrated result.
  Native health adds `pair_set_composition` and `set_composition_policy`;
  the existing `decoding` field continues to describe the legacy endpoint.
- `hydrate_response` — read-only product hydration from an immutable SQLite snapshot.

---

## Database schema (SQLite) — `SCHEMA_VERSION = 1`

Created by `init_graph`; `provenance` is defined before `nodes`/`edges` so their
FKs resolve. FKs enforced (`PRAGMA foreign_keys = ON`).

| Table | Columns | Notes |
| --- | --- | --- |
| `meta` | `key PK, value` | `schema_version`, `protocol_version`. |
| `provenance` | `id PK, source_type, citation, license, notes` | `source_type` ∈ `canon`/`synthetic`/`curated`. |
| `nodes` | `id PK, key UNIQUE, namespace='PKM', kind='product', label, status='active', provenance_id→provenance` | `key`=SKU. `kind='product'` emittable; `'attribute'` = hidden pivot. `label` hydration-only. |
| `relation_types` | `id PK, name UNIQUE, category` | Predicate names (graph-internal; **not** model tokens). |
| `edges` | `id PK, src→nodes, dst→nodes, rel→relation_types, weight=1.0, confidence=1.0, is_canonical=1, provenance_id→provenance` | `UNIQUE(src,dst,rel)`; indexed src/dst/rel. |
| `aliases` | `id PK, node_id→nodes, alias, locale='', priority=0` | `UNIQUE(alias,locale)`. Deferred in v1 (table exists, empty). |
| `snapshots` | `id PK, name UNIQUE, protocol_version, tokenizer_version, node_count, edge_count, hash, created_at` | Immutable dataset versions. |

Indexes: `edges(src)`, `edges(dst)`, `edges(rel)`, `nodes(kind)`, `aliases(node_id)`.

The schema is stable for the Phase-C reference path; advanced model/training
switches remain explicit deferred boundaries in `registry.md` and the
architecture documentation.

---

## CLI commands (`plm`)

| Command | Purpose | Status |
| --- | --- | --- |
| `plm --version` | Print version. | ✅ |
| `plm graph init --database <path> [--protocol-version]` | Create graph schema. | ✅ |
| `plm graph validate --database <path>` | Integrity + counts. | ✅ |
| `plm graph import-national-dex --database <path> [--protocol-version]` | Import full PokeAPI National Dex, retaining `TYPE`/`COLOR` facts. | ✅ |
| `plm graph snapshot --name <version> [--data-root] [--protocol-config]` | Create, verify or reconstruct an immutable named snapshot and receipt. | ✅ |
| `plm tokenizer build --names <file> --out <json>` | Build banded vocab from names. | ✅ |
| `plm corpus compile --database <path> --protocol-config <yaml> --out <dir>` | Graph → deterministic records, vocabulary, manifest. | ✅ |
| `plm baseline evaluate --corpus <dir> [--baseline oracle|popularity|complex] [--seed] [--split-seed] [--override ...] [--out]` | Validation ranking with independent initialization/partition seeds and resolved identity; optional source archive. | ✅ |
| `plm train [--corpus <dir>] [--checkpoint <path>] [--resume <path>] [--override <Hydra expression>]` | Train and record exact config/source/data identity; overrides repeatable. | ✅ Phase C reference |
| `plm evaluate --checkpoint <path> [--override ...] [--out <json>]` | Validation ranking plus complete generation; immutable report, test rejected. | ✅ Phase C |
| `plm evaluate-final --checkpoint <path> [--override ...] [--out <json>]` | Explicitly evaluate selected checkpoint on protected test. | ✅ Finalization path |
| `plm serve --checkpoint <path> [--override ...] [--corpus] [--graph-db] [--runtime-dir] [--host] [--port] [--max-pending]` | Native API, graph snapshot, bounded serial execution; loopback/one worker by default. | ✅ native reference; vLLM deferred |

### Experimental recipes and scripts

- Baseline reports include `split_seed`, `baseline_config`, `source_commit`,
  `environment`, evaluator version `baseline-fixed-split-v2`; config hash includes
  resolved CLI seeds/fractions/corpus overrides. `--seed` now controls initialization
  only; use `--split-seed` to deliberately change query assignments.
- PLM evaluation reports include per-query `responses`: subject/dimension,
  expected/predicted IDs, complete `token_ids`, observed termination/validity/error.
  Guided reports record inference policy/config identity separately from training identity.
- `+experiment=national_dex_v1` — fixed-budget, fixed-split quality campaign;
  2,000 updates, batch 32, zero data-loader workers, split/initialization seed 1729.
- `+experiment=first_target_lab` — same campaign, first-target weight 32.
- `+experiment=prompt_set_lab` — same campaign, balanced prompt-set coefficient 1;
  adds 65,536 projection parameters at dim 256, sharing entity embeddings.
- `+experiment=checkpointing_lab` — activation checkpointing enabled.
- `+experiment=continuation_set_lab` — symmetric recipe plus early remaining-set
  supervision at coefficient one, same data/split/2,000-step budget and parameters.
- `+experiment=symmetric_relation_lab` — prompt-set objective plus symmetric
  relation loss at weight 1, same seed/split/2,000-step budget as its comparator.
- `+experiment=moe_lab` — four experts, top-1, balancing coefficient 0.01.
- `uv run --no-sync python scripts/profile_features.py --out <json>
  [--steps 5] [--warmup 2] [--batch-size 8]` — CUDA dense/checkpointed/MoE-top1/top2
  fixed-training-batch profiler; immutable report plus source archive. Reports
  parameters, raw/median times, peak allocated bytes, token/router losses and loads.
- `scripts/summarize_quality_campaign.py --reports <dir> --out <json>` — validate
  shared corpus/split identities, summarize three baseline seeds and dense/MoE
  validation reports, retain per-dimension metrics, failures and input hashes.
- `scripts/plot_quality_campaign.py --summary <json> --out <png>` — optional
  matplotlib renderer; creates PNG/SVG loss and full-answer comparison figures.
- `scripts/inspect_routing.py --run <dir> --out <json> [--batch-size 16]` —
  teacher-forced validation expert traffic by TYPE/COLOR, weighted by nonpadding
  token counts; reports entropy-derived effective expert count, exact source,
  checkpoint, split and script identities.
- `scripts/diagnose_conditioning.py --run <dir> --out <json>` — validation CE
  separated into first target, later targets and EOS; counterfactual subject/
  dimension prompt interventions; exact generation on 16 seeded training queries.
  Explicitly not final-test evaluation or a new primary quality result.
- `scripts/inspect_prompt_set.py --run <dir> --out <json>` — prompt-only auxiliary
  classifier diagnostics, threshold-logit 0 set scores plus ranking; distinct from
  autoregressive generation. Records responses, identities and an analysis-script copy.
  `--head prompt_set|symmetric_relation` chooses the trained head (default prompt_set);
  score threshold stays zero and classifier metrics remain separate from generation.
- `scripts/evaluate_postprocessing.py --report NAME=PATH [--report ...] --out <json>`
  — same deterministic policy applied to archived validation outputs across models;
  preserves raw metrics/EOS status, records per-query removal audit, source archive
  and script copy. Does not rerun the model or consult graph relations.
- `scripts/summarize_seed_replication.py --report VARIANT=PATH [--report ...] --out <json>`
  — audit paired dense/promptset validation seeds, actual checkpoint/source hashes,
  saved configurations and exact query coverage; replay the declared SAME policy,
  report per-seed metrics, sample SD, paired differences and persistent exact-answer
  failures without final-test use. Archives analysis source and its own script.
  `--comparison dense-promptset|promptset-symmetric` chooses the declared pair and
  varied coefficient; fixed coefficients, decoder, budgets and normalized configs
  must agree. Symmetric comparisons require identical training source contents.
  Default preserves the original dense/prompt-set analysis.
- `scripts/plot_seed_replication.py --summary <json> --out <png>` — three-seed
  paired plots for raw F1 and post-policy exact answers, with optional matplotlib.
- `scripts/validate_generation_batches.py --reference <json> [--reference ...] --out <json>`
  — six-report, three-seed validation of batch size eight against frozen serial
  original/unique-target outputs; checks full token/error/metric parity and
  unchanged model source/runtime. Writes `plm-batched-generation-v1` reports.
  Only after full parity, runs three alternating paired timings on eight fixed
  seed-1729 prompts, with warmup, synchronization, memory and output checks.
- `scripts/evaluate_first_guidance.py --reference-dir <batch-report-dir> --out <json>`
  — validation-only frozen symmetric-model experiment, seeds 1729/1730/1731,
  alpha 0/1/4/16. Replays original outputs at zero before applying learned
  log-sigmoid membership penalties to first-position product logits only.
  Existing forward computes prompt-only relation scores without labels. Records
  complete outputs, first-choice changes, raw/processed metrics and predeclared
  integration eligibility. No production config, training or serving change.
- `scripts/evaluate_guidance_uniqueness.py --guidance-dir <dir> --batch-dir <dir> --out <json>`
  — frozen-model guidance x uniqueness experiment: reuses verified guidance-only
  reports, replays unique-only outputs, then evaluates combined alpha 4/16 at all
  three seeds. Checks first-choice invariance and repeated-target first divergences;
  reports paired marginal effects and factorial interactions. Eligibility compares
  the combined policy with the original decoder, while retaining regressions
  relative to guidance alone. No production or serving change.
- `scripts/compare_decoding_policies.py --reference <json> [--reference ...] --out <json>`
  — compare original/unique-target cached decoding at frozen symmetric seeds
  1729/1730/1731. Requires full original-output replay against prior uncached
  validation, unchanged model source/runtime and matching checkpoint/data identity.
  Saves `plm-decoding-policy-v1` reports, paired metrics, changed queries and
  conditional HTTP eligibility. No retraining, relation lookup or final-test use.
- `scripts/profile_kv_cache.py --reference <json> --cached <json> --out <json>` —
  require full validation response parity, then time eight fixed prompts across
  three alternating paired repetitions; bind checkpoint/data/source evidence.
- `scripts/verify_native_server.py --reference <validation-json> --out <json>` —
  start an owned ephemeral loopback server, compare all validation outputs and
  metadata/error behavior over HTTP, record deployment/timing evidence, stop server.
  Requires full raw token evidence for positive guidance, compares canonical
  policy/deployment identity and raw/processed metrics/targets, and labels legacy
  no-token references as partial parsed parity. Verifies model source/runtime.
- `scripts/verify_guidance_integration.py --candidate-dir <dir> --out <json>` —
  replays alpha 0/16 unique-target validation through integrated batch generation
  at all three seeds. Requires exact complete outputs/metrics against archived
  candidates; preserves training identity and records new inference source/config.
- `scripts/analyze_type_coverage.py --reference-dir <integrated-reports> --out <json>` —
  CPU validation residuals by subject attribute count, overlap tiers, branch
  coverage and training signature support; verifies source/data/report identities.
  Graph facts are diagnostic labels only, never used to repair generation.
- `scripts/summarize_order_campaign.py --reference-dir <dir> --reports <dir> --out <json>` —
  CPU audit of the declared continuation1/first8 trial against both frozen
  continuation references. Checks exact config/checkpoint/source/input identities,
  reconstructs nine full-token reports, records first-choice membership/order,
  subgroup and paired metrics, and applies the original-control acceptance gate.
  Freezes script/helpers/source and rejects overwriting existing evidence.
- `scripts/explore_first_choices.py --reference-dir <dir> --plan <md> --out <json>` —
  frozen original-control validation diagnostic: four guided first products,
  independent greedy unique cached continuations, per-token policy log scores,
  total/mean learned selectors and separately labeled oracle availability.
  Requires full raw-token rank1 parity; preserves invalid/truncated paths,
  subgroup/paired evidence and immutable plan/script/helper/source receipts.
- `scripts/verify_set_reranking_integration.py --out <json> [--preflight-only]` —
  authenticate frozen candidate/set evidence and the explicit schema/source
  boundary; replay all three checkpoints offline and through temporary native
  HTTP servers. Requires full candidate/score/selection/disabled-greedy parity,
  raw/processed/hydrated HTTP parity and metadata/error checks. Preserves failure
  evidence, stops owned servers, and records source/config/input/report receipts.
- `scripts/verify_pair_composition_integration.py --prior-summary-sha256 <digest>
  --prior-audit-sha256 <digest> --out <json> [--evidence-root <path>]
  [--plan <path>] [--preflight-only]` — authenticate reviewed prior acceptance
  before helper/Torch loading; exact 666-query offline composition, four-path and
  greedy compatibility, then native set HTTP/source/hydration/failure replay.
  Preserve source/config/script boundaries and stop owned servers before the
  next GPU job; no protected-test access or tolerance tuning.
- `scripts/evaluate_pair_unions.py --out <json> [--preflight-only]` — CPU-only
  fixed four-original-plus-six-pair composition using frozen FP32 membership
  scores. Authenticates history, replays the original four-set choices, preserves
  ten provenance slots and source eligibility, and reports set metrics/gates
  against the strong four-set selector. Compositions have source paths, not
  invented emitted sequences/EOS. Records script/helper/source/plan/input receipts.
- `scripts/evaluate_direct_membership.py --out <json> [--root <path>] [--plan <md>]
  [--preflight-only]` — stdlib-only fixed positive-logit membership ablation over
  the three authenticated saved validation heads. Excludes subject and zero
  scores; reconstructs the pair baseline before measurement, reports exact
  gains/losses, membership changes, sign counts and fixed per-seed/pooled/dual
  gates. Authenticates historical archive members separately from current
  inference source, freezes script/plan/reports, and refuses existing evidence.
  No model execution, threshold search, synthetic sequence or application change.
- `scripts/rerank_candidate_sets.py --reference-dir <dir> --plan <md> --out <json>` —
  frozen four-path validation reranking with prompt-only symmetric membership
  logits. Canonical logit sums select unchanged eligible candidate sets; records
  every head logit, path score, selected output, group/paired metric and original
  greedy acceptance gate. No retraining, regeneration, threshold tuning or oracle
  selection. Freezes plan/script/helper/source and verifies input identities.
- `scripts/summarize_continuation_campaign.py --reports <dir> --out <json>` —
  CPU audit of six fresh control/candidate validation reports at fixed guided
  inference. Verifies training/config/source/checkpoint/data identities, rebuilds
  raw/processed metrics from complete tokens, reports single/dual TYPE and COLOR
  gains/losses, and applies the predeclared continuation-set acceptance gate.
- `scripts/analyze_matched_errors.py --reports <dir> --out <json>` — CPU
  analysis of matched continuation reports: raw/processed position maps,
  subject insertion, ordered versus set errors, and paired subgroup outcomes;
  verifies and freezes evidence without model execution.
- `scripts/probe_matched_prefixes.py --reference-dir <dir> --out <json>` —
  frozen six-checkpoint validation probes of generated versus teacher prefixes,
  next-token/remaining-set scores and archived-next agreement. Temporary hooks
  reuse existing representations; no trained weights or inference policy change.
- `scripts/diagnose_continuation.py --reference-dir <integrated-reports> --out <json>` —
  frozen validation teacher-prefix probes at first/second/third/middle/EOS and
  archived-divergence positions. Compares subject counterfactuals, token margins,
  distribution changes and prompt-only membership of omitted targets; records
  complete probes and provenance. Oracle-assisted diagnosis, not generated quality.
- `scripts/diagnose_sequence_errors.py --reference <validation-json> --out <json>` —
  validation-only teacher-forced accuracy and oracle-first-token intervention;
  requires zero-hint archived-output parity, counts the hint in the completion
  budget, records answer-family support and raw/processed results. Diagnostic
  evidence only, never unassisted candidate quality or a production decoding mode.
- `scripts/compare_checkpoints.py --reference <validation-json> --plan <md> --out <json>`
  — fixed seed-1729 prompt-set steps 500/1000/1500/2000; shared identity validation,
  FP32 losses, cached raw/processed full generation and final-reference parity.
  Eligibility requires valid EOS on every query; choose processed exact-set,
  then F1, then earlier step. Saves reports and source/script identities; no test use.
- `scripts/plot_checkpoint_selection.py --summary <json> --out <png>` — complete
  answer quality and normalized FP32 loss trajectories; also exports SVG.
- `scripts/summarize_quality_campaign.py --model-report NAME=PATH` — repeatable
  override for comparing later candidates against the retained dense reference.

---

## Protocol vocabulary (v1 — implemented in `plm.protocol.tokenizer`)

Reserved ID bands: core/control `0–31`, dimensions/modes `32–127`, `128–1023`
reserved for future content classes (materialized as `<reserved_N>` placeholders),
entities `1024+` (append-only, sorted). Uppercase `UPPERCASE_ASCII` protocol tokens.

| Class | Tokens (v1) | Model-visible | Emitted |
| --- | --- | --- | --- |
| Special | `PAD` (id 0), `BOS`, `EOS`, `UNK` (serve-time safety), `SEP` (reserved) | yes | `EOS` only |
| Control | `ANSWER` (prompt/completion + loss-mask boundary) | yes | no |
| Dimension | `TYPE`, `COLOR` | yes | no |
| Mode | `SAME` (`COMPLEMENTARY`, `RELATED` deferred) | yes | no |
| Entity | `PKM_<NAME>` (~1000, ids `1024+`) | yes | **yes** |
| Directives | `IGNORE`, counts, `SIZE`, `PAGE`, predicates | **no** | no |

Sequence grammar: `BOS SUBJECT DIMENSION MODE ANSWER TARGET… EOS`.

## Protocol config (`configs/protocol/pokemon_v1.yaml`)

- `version: str` — `pokemon-v1`.
- `dimensions: dict[str, str]` — `TYPE → HAS_TYPE`, `COLOR → HAS_COLOR`.
- `modes: list[str]` — `SAME` only.
- `target_order: list[str]` — shared count ↓, confidence ↓, product key ↑.

---

## Conventions

- Node keys: `PKM_<NAME>` (products, emittable) · `ATTR_<VALUE>` (hidden attribute
  pivots, `kind='attribute'`, never tokenized/emitted).
- Protocol tokens: `UPPERCASE_ASCII` with `_` separators.
- Config is the only source of experiment parameters; Python provides behavior, never
  hidden knobs. Hydra YAML under `configs/` → dict → `validate_config`.
- Core graph/protocol/corpus/evaluation packages stay torch-free at import time; model/training load torch only at the optional training boundary.
