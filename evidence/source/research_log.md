# Research Log

The **why** behind the project — architectural decisions, dead ends, benchmarks,
and the reasoning that produced them. Append-only and chronological: never rewrite
an entry; if something is overturned, add a new entry that says so and why.

Format per entry: `## YYYY-MM-DD — <title>`, then Context / Decision / Rationale /
Consequences as useful. This log is the source material for any eventual paper or
writeup, so favor rationale over bare outcomes.

---

## 2026-07-23 — v1 scope locked: SAME × {TYPE, COLOR}, oracle-driven systems experiment

**Context.** Reconciled three partly-conflicting descriptions of the schema:
the committed Phase-A SQLite schema (`src/plm/graph/schema.py`, 5 tables,
"protocol-agnostic, structure + integrity only"), a proposed 15-table normalized
`schema.md`, and the `protocol-spec.md` grammar — plus a v0.1 comprehensive project
handoff describing the long-term vision (generic GPU-native symbolic policy engine).

**Decisions.**
1. **Scope trimmed to the leanest viable protocol:** one mode `SAME` (COMPLEMENTARY
   deferred), two dimensions `TYPE` and `COLOR`, ~1000 Pokémon-name identifiers.
   `IGNORE` and result counts stay parser/post-process only — never model tokens.
2. **Two dimensions is the floor, not a nicety.** With a single dimension every
   training example shares the same steering tokens, so the model can ignore them and
   the protocol hypothesis is never exercised (a lookup table wins). `TYPE` vs `COLOR`
   forces the model to condition its output on the dimension token — the minimum test
   of "steered relational retrieval."
3. **Attributes are hidden nodes, not tokens.** Type/color *values* (electric, yellow)
   become `kind='attribute'` graph nodes used purely as join pivots; the model never
   sees or emits them. This collapses `schema.md`'s `PrimitiveFact(object_value XOR
   object_entity_id)` into the committed entity→entity `edges` table — no literal-value
   branch needed. `SAME` = "shares ≥1 attribute neighbor."
4. **Three-layer store split.** Graph → SQLite (source of truth); protocol
   (dimensions/modes/derivation rules) → versioned config files; corpus
   (vocab/manifest/records) → content-addressed files. `schema.md`'s ~15 tables map to
   3 stores, not 15 SQLite tables. This is also what `data-contracts.md` already
   committed to (manifests/configs committed; `.db` and processed records ignored).
5. **Bucket labeling, not pairwise.** The manual labeling web-app will assign each
   Pokémon to type/color buckets (O(N) clicks, dense + consistent + complete), rather
   than clicking pairwise "same?" judgments (O(N²), sparse, inconsistent). Bucketing is
   what makes the corpus a clean oracle.
6. **No UUIDs.** Keep integer PK + stable string `key`. Random UUIDs break "corpus is a
   pure function of the snapshot"; deterministic UUIDv5 is just a restatement of `key`.

**Key reframe — v1 is a systems experiment over an oracle.** `TYPE/COLOR SAME` is
*deterministically computable* (shared bucket), so the corpus compiler is a **perfect
oracle**. The transformer is therefore not competing on quality in v1 — it is learning
a known-correct function and being served hyper-efficiently. This splits the project
into **Exp A** (systems: match the oracle, win on concurrent-users@latency + energy)
and **Exp B** (learning/policy: COMPLEMENTARY, compositional held-out, world-evolution
— the non-computable payoff, where the "only learn what can't be computed" rule bites).
v1 deliberately lives on the computable side because that side has ground truth to
measure against. Corollary: **do not claim a quality win over the baseline in v1.**

**Reality check on signal budget.** ~1000 names × 2 dimensions ≈ ~2000 *distinct*
queries — not the "millions of samples" the v0.1 handoff imagines (those need
augmentation or a bigger graph). At that size the model largely memorizes the oracle;
a KGE/lookup baseline will be strong. Fine for Exp A (we are benchmarking a model that
*has* learned the map); Exp B will need a denser graph or compositional held-out splits.

**Contradictions found in the v0.1 handoff (superseded here).**
- Handoff shows `RETURN 10` inside the token stream → rejected. Counts/`RETURN`/`SIZE`/
  `PAGE` are parser directives; the model never sees numbers. `ANSWER` is the delimiter.
- Handoff says "the dataset should never be manually written" → not in conflict: the
  *graph* is curated (the web-app is fine); the *corpus* is generated from it.
- Handoff omits `ANSWER`/uses `RETURN` as a boundary → keep `ANSWER` as the loss-mask
  boundary.

**Pending code implications (not yet done).**
- Graph schema evolves from 5 → ~7 tables: add `provenance`, (empty) `aliases`; add
  `namespace`/`status`/`provenance_id` on `nodes`, `confidence`/`is_canonical`/
  `provenance_id` on `edges`, `protocol_version`/`tokenizer_version`/`created_at` on
  `snapshots`. Additive/backward-compatible.
- Tokenizer must change before any corpus is compiled: add `ANSWER` + dimension +
  mode tokens; **stop tokenizing predicates** (they are not model-visible); adopt the
  reserved ID ranges; add a strict encode path that fails closed (no silent `<unk>`)
  during corpus compilation.

## 2026-07-23 — Tokenizer: banded, append-only, class-manifested

**Context.** A tokenizer maps each symbol to one integer ID. Because a token ID is a
*dense row index* into the model's embedding table (and the output softmax width), the
ID layout is a permanent contract: renumbering a token silently invalidates every trained
checkpoint. Rewrote `plm.protocol.tokenizer` to make that contract explicit.

**Decisions.**
1. **Direct map, no training, no BPE.** One symbol = one ID. Kept — it's what lets the
   vocab (and thus the model) stay tiny, which is the serving thesis.
2. **Reserved ID bands, entities pinned high.** `0–31` core specials + control, `32–127`
   dimensions/modes, `128–1023` reserved for future content classes, `1024+` entities.
   Entities are pinned at `ENTITY_BASE=1024` so growth of the small control plane can never
   renumber them — entity IDs appear in every corpus record and must stay put. This is the
   one content-band reservation that buys *robustness*, not just tidiness.
3. **The number line carries no meaning; the manifest does.** Every token stores an explicit
   `class` (`special`/`dimension`/`mode`/`entity`/`reserved`). Callers use `class_of` /
   `entity_ids`, never arithmetic on the ID. Reserved gaps are materialized as
   `<reserved_N>` placeholders so the vocab stays a dense list (index == ID) and the content
   hash stays simple. Cost: ~1024 placeholder rows before entities (~0.5 MB at our dim) —
   negligible, and the price of "no ID→row remapping."
4. **Fail-closed encoding.** `encode(strict=True)` (default) raises `UnknownTokenError` on
   any unknown token — corpus compilation must never silently substitute `UNK` and poison
   training data. `strict=False` (→ `UNK`) exists only for serve-time robustness. `UNK` is
   kept in the vocab as a safety row but is never emitted during compile.
5. **Predicates removed from the vocab.** `HAS_TYPE` etc. are graph-internal and *not*
   model-visible; the old `build(relations=…)` tokenized them, which was a latent bug. `build`
   now takes `entities` + `dimensions` + `modes` only.
6. **Uppercase protocol tokens** (`BOS`, not `<bos>`) to match the spec; `PAD` pinned to id 0.
7. **Sorted entities vs append-only — the nuance.** Within a version, `build` *sorts* the
   entity set so the same set yields the same hash (reproducibility). True cross-version
   append-only (adding SKUs later without renumbering) is a *migration* operation — load the
   frozen vocab, append new tokens at the tail — not a re-sort. v1 only needs build-from-set;
   the extend/migrate path is a later concern.

**Consequence.** Vocab file format bumped to `schema_version=2` (records version + tokens +
classes). `DEFAULT_DIMENSIONS/MODES` temporarily live in the tokenizer module; they belong in
the versioned protocol config (`configs/protocol/pokemon_v1.*`) and move there when it lands.

## 2026-07-29 — Phase B: National Dex oracle graph and deterministic corpus compiler

**Context.** The initial plan called for a small curated Gen I proof graph. The
project owner instead selected the complete PokeAPI National Dex. v1 remains
strictly `SAME × {TYPE, COLOR}`: the extra catalog coverage increases the
systems workload without changing the protocol task or smuggling additional
semantic features into the model.

**Decision.** Import PokeAPI's maintained CSV dump in bulk, not thousands of
individual API responses. The importer persists each National-Dex species as a
`PKM_*` product and its default-form `HAS_TYPE` / `HAS_COLOR` links to hidden
`ATTR_*` pivots. It records PokeAPI provenance in SQLite. The compiler reads
the versioned `pokemon-v1` protocol config and emits one sequence per non-empty
subject/dimension bucket, with labels masked through `ANSWER`.

**Result.** The current import contains 1,025 product nodes, 28 hidden
attribute nodes, and 2,576 edges. It compiles to 2,050 records with a 2,049-row
banded vocabulary; its longest sequence is 289 tokens, below the tiny model's
512-token context. This is still modest data for a transformer; it strengthens
the systems benchmark and does not turn Exp A into a quality claim.

**Consequence.** The documentation's prior Gen I wording is superseded. The
next Phase B work is an explicit held-out ranking split plus oracle/KGE
baselines; only then should Phase C training begin.

## 2026-08-13 — Phase B continuation plan: query holdout and shared ranking

**Context.** The graph and corpus compiler are implemented, but evaluation had no
precise split unit or scorer contract. The configured `held_out_relation_fraction`
name also implied an edge-level split that would mutate or underdetermine the
fixed v1 oracle graph.

**Decision.** Hold out complete `(subject, dimension)` queries using a seeded
SHA-256 assignment while leaving SQLite unchanged. Evaluate all product candidates
with multi-positive MRR, Hits@K, and MAP through one shared evaluator. Require the
full-graph oracle and dimension-specific popularity baseline; add ComplEx as an
optional Torch baseline with lazy imports. Rename the config field to
`held_out_query_fraction` so the contract matches the experiment.

**Rationale.** Query holdout preserves the deterministic oracle and keeps v1's
systems experiment measurable. A common evaluator prevents baseline-specific
metric definitions from obscuring comparisons. ComplEx belongs behind the existing
training dependency boundary because core Phase A/B imports remain torch-free.

**Consequence.** Phase B now has an executable continuation plan in
`docs/plan/phase-b-evaluation-strategy.md`, `phase-b-evaluation-tasks.md`, and
`phase-b-evaluation-coordination.md`. Phase C training is gated on the resulting
split, baselines, and green quality checks.

## 2026-08-13 — Phase B evaluation and labeling implementation

**Context.** Executed the continuation plan with parallel Luna workers after the
repository re-onboarding. The missing evaluation primitives and untested local
curation path were the remaining Phase B gaps.

**Result.** Added deterministic query splits, shared multi-positive MRR/Hits@K/MAP,
oracle/popularity/optional lazy-Torch ComplEx scorers, a JSON-emitting baseline CLI,
and endpoint-level labeling tests. Label saves now replace TYPE and COLOR facts in
one transaction and attach one curated provenance row to all replacement edges.
The config field is now `held_out_query_fraction`.

**Verification.** The full suite is green: 67 tests passed. Ruff, formatting, and
strict mypy passed. FastAPI's test client emits one upstream deprecation warning
about its httpx compatibility; it does not affect correctness.

## 2026-08-13 — Phase B plan extended with local labeling workflow

**Context.** The repository already contains a minimal FastAPI labeling surface,
but it had no endpoint-level acceptance tests or explicit place in the Phase B
execution plan.

**Decision.** Treat labeling as a parallel Phase B workstream: local-only
TYPE/COLOR curation, known-value validation, atomic fact replacement, and a new
`curated` provenance row per save. Keep remote deployment, authentication, and
multi-user collaboration out of v1.

**Rationale.** The curator is the human-facing input path for improving graph
facts, while SQLite remains the sole source of truth. Endpoint tests make corpus
changes auditable before compilation and avoid coupling evaluation agents to a
running browser server.

## 2026-08-23 — Phase C reference path and AVO scaffold implemented

**Context.** Phase B had the graph, compiler, and baseline foundation, but the
split contract, artifact identity, model factory, training/checkpoint path,
and evaluator-facing experiment records were incomplete.

**Decision.** Close the Phase C reference boundary with deterministic
train/validation/test query identity, authoritative Hydra-to-Pydantic loading,
fail-closed corpus/provenance validation, append-only vocabulary migration, and
a minimal RMSNorm/RoPE/GQA/SwiGLU decoder. Add a separate AVO package for
candidate identity, evidence receipts, comparison rules, namespaces, and the
test firewall. Keep unsupported model/training switches explicit rather than
silently pretending that compile, FP16, EMA, MoE, muP, or advanced RoPE are
implemented.

**Evidence.** The final local training-group environment passed 103 tests,
strict mypy, Ruff, and `git diff --check`. Runtime gates also passed: synthetic
overfit, deterministic save/reload equivalence, a real temporary-corpus
`run_training` smoke, CUDA checkpoint resume with RNG restoration, and native
parser/hydration tests. PokeAPI import provenance is pinned to an immutable
revision with per-file hashes.

**Boundary.** This does not yet claim AG0 or a genuine National Dex quality
result: repeated baseline variance, a recorded PLM-vs-baseline validation
receipt, and protected final-test finalization still require a real corpus run.
Phase D/vLLM serving remains deferred.

## 2026-09-24 — State assessment supersedes Phase C correctness confidence

**Context.** Assessed local `main` at `8afff36` against source, ignored data,
and fresh runtime checks. The existing suite passes (103 tests), as do Ruff,
format checking, and strict mypy. These results do not establish next-token
training correctness or general multi-batch resume equivalence.

**Findings.**
- The compiler puts each target at the same position in `input_ids` and
  `labels`; collation preserves that alignment. `PLMDecoder.forward` applies
  cross entropy without a causal shift. A seven-token diagnostic produced
  reported loss = same-position loss = 2.5481743813, versus next-token loss
  2.7929649353. The ANSWER-position logits received zero loss gradient even
  though inference uses that position to predict the first target. The
  synthetic overfit gate uses the same unshifted objective. Its success cannot
  establish that the intended relational generation task has been learned.
- `Trainer.train` restores the step and optimizer but initializes a fresh
  data iterator at the first record. A four-step run over three distinct
  records, batch size one, resumed from step two, differed from the continuous
  run by maximum parameter difference 0.0293125976. The separate resume helper
  repeatedly uses one fixed batch and does not exercise this boundary.
- The saved National Dex corpus passes internal hash validation but fails
  graph-bound validation: `graph content hash does not match manifest`.
  Recompiling into a temporary directory preserves records and vocabulary
  byte-for-byte while changing the manifest. The Phase C commit expanded the
  graph-hash definition; the local manifest still carries the old identity.
  The local database also retains legacy `master`-URL provenance rather than
  the immutable revision/file hashes supported by the current importer.
- No `runs/` directory or checkpoint/baseline/receipt artifacts were found in
  the inspected checkout outside dependency/build directories. Phase D is
  still a CLI stub, and observability has no implementation. AVO record and
  comparison primitives exist, but their existence is not a completed campaign.

**Consequence.** This supersedes the August entry's implied confidence in
causal-LM training and general resume equivalence. Phase C remains a reference
implementation with correctness blockers, not an accepted experimental result.
Repair and independently test the next-token objective and data-position
resume first; preserve/rebuild versioned graph/corpus provenance before the
National Dex campaign. Then record repeated baseline and PLM validation
evidence before final-test selection and systems/serving claims. This session
did not modify implementation or replace the existing dataset.

## 2026-09-24 — Repair next-token learning and sequential resume; teach the evidence

**Context.** The owner authorized fixing and advancing the project while teaching
each step, with intuition first, equations and tensor details, and a readable
summary of the work so far. This iteration addresses the two reproduced training
defects before a dataset refresh or full experiment.

**Decisions and rationale.**
- Keep corpus labels input-aligned and perform the causal shift exactly once
  in the decoder. Logits at t predict labels at t+1; prompt/padding targets are
  ignored. Apply optional z-loss to the same supervised prediction positions.
  Reject inputs with no supervised next-token target.
- Require the synthetic overfit gate to generate the correct completion from
  the prompt alone, as well as meeting its loss threshold. Teacher-forced loss
  cannot substitute for observed autoregressive behavior.
- Derive sequential resume position as `(step * grad_accum_steps) % batches`.
  Isolate loader generators from model RNG so iterator creation cannot perturb
  dropout. This is valid for the current static dataset, deterministic collation,
  and `shuffle=False`; random augmentation/shuffling would need extra state.
- Record objective and batch-order metadata in trainer checkpoints and reject
  legacy or incompatible resume contracts. Old same-token training weights must
  not silently continue under the corrected objective.

**Evidence.** New target-alignment checks and a distinct-batch resume check failed
before the respective fixes. The complete suite now passes 115 tests, including
prompt-boundary gradients, future-token isolation, padding, optional z-loss,
autoregressive toy completion, and CPU/CUDA resume with dropout. Resume checks
cover accumulation, partial batches, epoch boundaries, and a Windows worker;
they compare losses, weights, optimizer/scheduler state, and model RNG state.
A 100-step, 16-dimensional toy run reduced loss from 3.4326453 to 0.00485336 and
generated `(6, 7)` from its five-token prompt, matching the target/EOS fixture.
This is a mechanics check, not a National Dex learning result. Ruff, formatting,
strict mypy, and diff checks pass. The documentation build succeeds with existing
links outside the docs tree producing warnings.

**Remaining boundary.** Persistent dataset identity/provenance is unchanged.
National Dex training, repeated validation measurements, protected final-test
evaluation, and serving measurements remain pending. First-token ranking is not
full-sequence quality: the current evaluator's default `protocol_valid=True`
does not measure generation validity. The campaign must address that boundary
and complete experiment identity before supporting scientific claims.

**Learning artifacts.** `docs/learning/00-project-so-far.md` explains the research
question, graph/corpus/protocol, transformer tensors, evaluation, and remaining
work. `01-training-correctness.md` explains these repairs in detail. Both are
linked from the documentation navigation and README.

## 2026-09-24 — Traceable data, full responses, and measured architecture experiments

**Context.** The owner explicitly prefers exploring advanced architecture features
as a learning/research exercise, even where they may be excessive for the small
TYPE/COLOR task. Preserve the dense reference and measure costs and failures;
adding a switch is not evidence of better quality or state-of-the-art performance.

**Data and experiment identity.** Created `pokemon_v1_f1541479_20260924` from
PokeAPI revision `f15414790832c88d784d1537658b957fd73cbbbd`, preserving old
artifacts. The new 1,025-product, 2,576-edge, 2,050-record snapshot has correct
graph/provenance identity; records and vocabulary remain unchanged. Its receipt
binds five CSVs, SQLite and three corpus files. A second compilation verifies
byte stability. Receipt-only reconstruction preserves the original snapshot
timestamp and checks every file hash before publication; tested against all nine
actual National Dex artifacts using retained raw sources. Partial/tampered
versions are rejected. This supersedes the preceding entry's pending refresh.

Training now archives exact working Python source and dependency definitions,
records effective config/data/model identities and environment, and saves progress
and final results. Checkpoint payload metadata is authoritative and conflicting
sidecars fail. These additions make dirty-worktree experiments inspectable.

**Evaluation decision.** First-token ranking is retained but cannot establish
complete-answer quality. Native greedy generation now starts from the prompt,
optionally constrains syntax, and records observed EOS or truncation without
inventing EOS or filtering answers against the oracle. Full evaluation reports
macro precision/recall/F1, exact sets/sequences, validity, termination, duplicates,
self-return and length. Standalone ranking validity is unknown rather than
defaulting to true. The scorer caches one forward per query. Validation token
loss is weighted by supervised-token counts and excludes router penalties.

**Architecture decisions.**
- Activation checkpointing uses non-reentrant per-block recomputation, preserves
  dropout RNG and only operates during gradient-enabled training. This changes
  memory/compute, not the intended model function.
- Sparse MoE replaces each dense SwiGLU with top-k routed SwiGLU experts and
  optional always-active shared experts. FP32 router probabilities and balancing
  statistics exclude padding. Selected full-softmax probabilities are not
  renormalized: top-1 renormalization would erase the task gradient through the
  selected probability. Balance penalty is `E * sum(f_i * mean(p_i))`, with
  route fractions normalized by tokens times k. No capacity dropping, fused
  dispatch, distributed placement or serving speedup is implied.
- RoPE previously combined half-vector pairing with adjacent-angle repetition.
  Corrected to half-vector frequency concatenation. Norm preservation and
  simultaneous-position-shift dot-product tests now guard the rotation identity.
  This is a correctness repair, not a validation-selected architectural option.
- Dense stays default; MoE/checkpointing have separate named lab recipes. Advanced
  RoPE scaling, muP and a KV cache remain unsupported. RoPE scaling needs an
  explicit longer-sequence experiment; current records fit within 289 tokens.

**Measured feature cost.** On RTX 5070 Ti/Torch 2.11.0+cu128, BF16 autocast with
FP32 weights and fused AdamW, profile the same eight longest training records
(`[8,289]`), two warmup updates and five timed updates per variant. Median
forward/backward/update time and peak allocated tensor memory:

| Variant | Parameters | Median ms | Peak MiB |
| --- | ---: | ---: | ---: |
| Dense | 6,558,720 | 60.44 | 498.56 |
| Checkpointing | 6,558,720 | 98.03 | 184.26 |
| Four experts/top-1 | 20,722,688 | 127.08 | 737.75 |
| Four experts/top-2 | 20,722,688 | 127.99 | 894.88 |

Checkpointing trades roughly 62% longer steps for 63% less peak allocated memory.
MoE is slower and larger here. Top-1/top-2 timing proximity needs repetitions and
order balancing, not a claim of free extra compute. This single-process,
fixed-order microbenchmark excludes data loading and establishes no quality,
serving, energy or maximum-capacity result. Raw report:
`docs/experiments/2026-09-24-feature-profile.json`; matching original source archive
retained at `runs/learning/20260924-feature-profile.source.zip`. Source was captured
before subsequent snapshot reconstruction/documentation changes; hashes preserve
that distinction.

**Validation.** 130 tests pass, one existing Starlette/httpx warning; Ruff,
format checking, strict mypy and diff checks pass. Documentation builds with
existing outside-docs-tree link warnings. New tests cover CPU/CUDA recomputation
equivalence with dropout for dense/MoE, sparse task gradients, padding exclusion,
auxiliary accounting, full-response metrics, snapshot reconstruction/tamper
refusal and a tiny train-save-evaluate integration. A separate 100-step,
16-dimensional top-1 MoE toy run reduced total loss 7.7121258 -> 0.0151458 and
generated `(6,7)` exactly. Its total includes router loss and is not comparable
to the earlier dense toy run as a controlled quality experiment.

**Next evidence.** Frozen-split dense/MoE National Dex validation, comparable
baseline measurements, router-usage inspection, then longer-context experiments
and serving. The protected final test remains untouched by these measurements.
Learning notes 02/03 explain provenance, evaluation, tensor shapes, gradients and
tradeoffs; the overview now reflects these changes. No commit or push.

## 2026-09-24 — Begin the fixed-split National Dex quality campaign

Recorded the procedure in `docs/experiments/2026-09-24-quality-campaign-plan.md`
before launching models. Compare final 2,000-update dense and four-expert/top-1
models with seed 1729 on the same 1,637/222/191 query partition. Use validation
only; equal updates intentionally do not mean equal parameters/compute. Run
oracle/popularity/ComplEx baselines at seeds 1729/1730/1731. ComplEx uses explicit
batch 512, dimension 32, 100 epochs, one negative, no regularization.

Found and repaired a campaign-confounding CLI issue: baseline initialization
seed also selected the query split. `--seed` now controls baseline randomness,
and independent `--split-seed` defaults to `data.split_seed`. Resolved CLI
arguments enter the config hash. Baseline reports archive source identity and
record environment; core execution still permits Torch to be absent. A regression
check varies initialization seed while keeping split/metrics fixed for the oracle.

Training refuses to overwrite finished runs and requires explicit resume for
existing progress. Reports retain full generated response lists for failure
analysis. Training wall time includes evaluation/checkpoint overhead; CPU
baselines and some GPU evaluation/training work overlap, so these times are not
isolated hardware-performance comparisons. Feature microbenchmarks from the
previous iteration remain the appropriate narrow cost measurement.

## 2026-09-24 — Quality campaign result: continuation learning is not relational generalization

**Completed evidence.** Both models completed the declared 2,000 updates and
full prompt-only evaluation on all 222 validation queries. All nine baseline
runs completed on the same split. Full checkpoints, source archives, configs,
responses and logs remain under `runs/`; aggregate identities/failures and
figures are in `docs/experiments/2026-09-24-quality-campaign.*`. No final test
was evaluated. The final-checkpoint choice was fixed before observing results.

| Model | Token validation CE | Generated precision | Recall | F1 | Exact sets | EOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.190831 | 0.303454 | 0.269076 | 0.278423 | 0/222 | 222/222 |
| Four experts/top-1 | 0.169906 | 0.302950 | 0.270048 | 0.279554 | 0/222 | 221/222 |

MoE's `PKM_SLOWBRO TYPE` response reached 507 generated tokens without EOS.
Duplicate-query rates were 2/222 dense and 4/222 MoE; self-return rates were
66/222 and 67/222. Dense TYPE/COLOR F1: 0.276763/0.280340; MoE:
0.270494/0.290022. The 0.11-percentage-point aggregate F1 difference from one
seed does not establish a gain. Both fail the intended oracle-quality target.

Oracle ranking MRR/MAP is 1/1 in all three repetitions. Popularity is
0.253823/0.236727, also unchanged. ComplEx mean +/- sample standard deviation
across seeds: MRR 0.986998 +/- 0.006467, MAP 0.992602 +/- 0.000995.
First-token dense MRR/MAP is 0.466399/0.137203; MoE 0.490262/0.139406.
These are ranking measures, not interchangeable with generated-set F1.

**Diagnostics and interpretation.** Dense validation CE bottoms at step 800
(0.161234) and rises while last-batch training loss keeps falling. MoE shows
the same pattern after step 1,000 (0.154061). Both generate every answer exactly
on the same 16 seeded training-query sample. This supports overfitting on this
sample, not a claim that every training query is exact.

Decompose dense FP32 teacher-forced validation loss: 222 first targets have
mean CE 5.767774; 28,221 remaining targets have CE 0.147891; 222 EOS tokens
have CE 0.069525. First-target positions are only 0.774% of supervision. Their
high loss is diluted by easy continuations. MoE's corresponding first/later
target CE is 5.373899/0.129504. Both have 43/222 correct canonical first targets.

Prompt ablation disproves total prompt neglect: replacing dense subjects with
different products increases first-target CE to 8.516666 and drops exact first
accuracy to 8.56%; swapping TYPE/COLOR gives CE 11.946355 and accuracy 4.50%.
Later-target CE changes much less (0.157547/0.236095). These counterfactual
teacher-forced probes measure information dependence, not response quality for
the changed queries. Sorted answer prefixes make continuation easy without
establishing reliable held-out query-to-relation selection. Dense/MoE outputs
are >=95% Jaccard-similar to a training answer set in 220/222 and 221/222 cases;
combined with incorrect target sets, this is consistent with selecting familiar
list patterns, but is not proof of literal retrieval.

MoE routing does not collapse onto a single expert: nonpadding-token-weighted
validation traffic has entropy-equivalent expert counts 3.50–3.93 out of four,
with different TYPE/COLOR distributions across layers. These measurements include
teacher-forced prompt/answer/EOS tokens, not autoregressive generated traffic.
Balance is not evidence of useful specialization. Conditioning/routing reports
record checkpoint, source, split and analysis-script hashes.

**Decision.** Do not scale architecture or claim scientific readiness on low
token loss. The next candidate should strengthen prompt-conditioned relational
supervision using existing training targets, e.g. an auxiliary set objective at
ANSWER, with the same held-out split and complete-generation comparison. A
first-target weighting control can distinguish supervision changes. Neither is
yet implemented or claimed to fix the failure. v1 remains a systems hypothesis
whose prerequisite oracle-quality parity has not been reached.

**Verification/learning.** 131 tests pass (one existing Starlette/httpx warning),
Ruff/format/strict mypy are green. All campaign/diagnostic scripts executed;
the chart was visually inspected. Lesson 4 explains the loss decomposition,
ablation, ranking/generation distinction and negative result with equations and
tensor shapes. README/overview reflect the actual failed quality gate.

## 2026-09-24 — Prompt supervision: a useful objective change and a failed control

**Pre-run decision.** Recorded `2026-09-24-prompt-objectives-plan.md` before
launching two dense candidates, each seed 1729, 2,000 updates, batch 32 and the
same frozen 1,637/222/191 partition. First-target reweighting uses coefficient 32
and effective-token-weight normalization. The auxiliary candidate retains ordinary
token CE and adds coefficient-1, per-query positive/negative-balanced binary set
loss. Its bias-free 256x256 projection reads only the causal ANSWER state and
scores tied product embeddings. This direct target-side relation gradient is why
we share embeddings instead of introducing an independent product output table.
It adds 65,536 parameters; common initialization is unchanged at the same seed.

The set target is derived only from supervised product IDs in the same training
record. Controls/EOS/padding are excluded and duplicates count once. Tests verify
future-token invariance and zero gradient through future input activations,
positive/negative score-gradient directions, exact weighted NLL, identical common
initial weights/logits and exact resume with auxiliary state. Ordinary token CE,
first-target CE and set loss remain separate telemetry. Both experimental switches
default off/neutral. No auxiliary scores filter the model's generated IDs.

**Full validation results.**

| Objective | Raw precision | Raw recall | Raw F1 | Raw exact sets | EOS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained dense reference | 0.303454 | 0.269076 | 0.278423 | 0/222 | 222/222 |
| First-target weight 32 | 0.204623 | 0.186254 | 0.190113 | 0/222 | 222/222 |
| Prompt-set auxiliary | 0.662100 | 0.616716 | 0.629018 | 0/222 | 222/222 |

Reweighting alone worsened generation and final first-target CE (7.064681).
Treat it as a failed improvement, retaining the evidence/recipe. Prompt-set
supervision improves raw F1 by 35.06 percentage points and first-target CE to
2.694590, with ordinary validation CE 0.146743. First-target reference CE 5.767774
was measured in a separate FP32 probe; training telemetry uses BF16 autocast,
so tiny numerical differences are not meaningful. New-model TYPE/COLOR raw F1
is 0.562840/0.705475. These are single-seed, equal-update, unequal-compute results,
not robust superiority or oracle parity. No final-test evaluation occurred.

The separate prompt-only classifier has macro F1 0.654480, MAP 0.701301, zero
exact sets, and self-return in 77.48% of queries at the declared zero-logit
threshold. Keep this distinct from autoregressive generation. The classifier's
balanced-loss sigmoid scores are not asserted to be calibrated probabilities.

**Secondary deterministic policy, after error analysis.** 105 new-model outputs
contain exactly the required set once the subject is removed. SAME's self-exclusion
was already part of graph derivation; it requires request identity, not relation
knowledge. Implemented the planned post-filter boundary: subject exclusion,
IGNORE metadata, stable deduplication, optional return bound. Count every removal;
do not fill missing IDs, query relations, fabricate EOS or mutate raw reports.
Post-filtered targets may be empty, while the original parsed grammar still
requires TARGET+ EOS.

Applied the same policy offline to all four candidates, with no IGNORE entries
and no limit. Preserve this as a secondary analysis chosen after inspecting
validation errors, not the predeclared raw-model result:

| Model | Processed F1 | Exact sets/sequences |
| --- | ---: | ---: |
| Dense | 0.279537 | 42/222 (18.92%) |
| MoE | 0.280672 | 38/222 (17.12%) |
| First-target weight 32 | 0.190871 | 27/222 (12.16%) |
| Prompt-set auxiliary | 0.631879 | 105/222 (47.30%) |

The unfinished MoE response remains unfinished. This is a verified component
and replay, not live service evidence. Raw-generation source archives precede
the new post-processing module; the secondary report archives its own source
and script. Source differences are recorded rather than silently conflated.

**Evidence and next decision.** Raw runs/checkpoints/reports remain under `runs/`.
Portable reports and visually inspected PNG/SVG figures are
`docs/experiments/2026-09-24-prompt-objectives.*`, with separate classifier and
post-processing reports. 144 tests pass (one existing Starlette/httpx warning);
Ruff, formatting, strict mypy and diff checks are green. Lesson 5 explains the
losses, gradients, tensor shapes and distinction between model learning and
deterministic rules. Repeat the auxiliary candidate across seeds, inspect the
remaining relation/completeness errors and checkpoint selection, then integrate
native generation/post-processing/hydration for measured serving. Oracle-quality
parity, final-test selection and serving performance remain open. No commit/push.

## 2026-09-24 — Paired seed replication: the objective gain repeats

**Question and procedure.** The previous prompt-set improvement had one seed.
Before training, recorded `docs/experiments/2026-09-24-seed-replication-plan.md`:
retain the seed-1729 runs and add seeds 1730/1731 for both dense token-only and
prompt-set training. Keep the canonical corpus, split seed 1729, 2,000 updates,
batch 32, architecture, optimizer/scheduler and final-checkpoint rule fixed.
No coefficient tuning, intermediate-checkpoint selection or final-test evaluation.
Dropout is zero and data order sequential, so initialization is the main seeded
variation. GPU deterministic algorithms remain disabled. Equal-update comparisons
do not establish equal-compute efficiency.

**Completed evidence.** Four new training runs and all 222-query validation
evaluations completed. The same predeclared deterministic SAME policy is applied
to all six models, without IGNORE keys or return limits. No malformed/unfinished
output is repaired. All six runs have 222/222 valid EOS-terminated responses,
but zero raw exact sets.

| Seed | Dense raw F1 | Prompt-set raw F1 | Dense processed exact | Prompt-set processed exact |
| --- | ---: | ---: | ---: | ---: |
| 1729 | 0.278423 | 0.629018 | 42/222 | 105/222 |
| 1730 | 0.195359 | 0.587455 | 32/222 | 103/222 |
| 1731 | 0.220109 | 0.601981 | 30/222 | 97/222 |

Mean raw F1 is 0.231297 (sample SD 0.042647) versus 0.606151 (0.021093).
The within-seed improvements are 0.350595, 0.392097 and 0.381873, with mean
0.374855 and sample SD 0.021623. Processed exact accuracy averages 0.156156
versus 0.457958; mean paired gain 0.301802, sample SD 0.018018. These are
descriptive statistics from three seeds on one split, not confidence intervals,
independent query repetitions or evidence across datasets.

**Error consistency.** Prompt-set training is exact after policy on 61 queries
in all seeds and 137 in at least one. The remaining 85 are never exact: 72 TYPE,
13 COLOR. Mean raw F1 is 0.547899/0.673453 for TYPE/COLOR, with mean processed
exact accuracy 0.271709/0.673139. The corresponding dense all/any counts are 5/70.
The any-seed number is an oracle-selected diagnostic, not an achievable ensemble
quality claim. This supports retaining the objective and prioritizing TYPE
failure analysis; it does not identify the cause of those failures.

**Provenance audit.** The new summarizer verifies actual checkpoint/source archive
bytes, saved raw configuration hashes, training/report identities, the exact
validation queries and targets, decoding/runtime compatibility and paired seeds.
It recomputes metrics from responses and rejects unmatched seed sets. Historical
dense source predates the optional objectives/telemetry; neutral defaults retain
its objective. Model/training code is unchanged between the first prompt-set run
and new runs. Their archive differences are only post-processing and its exports.
The summary records source differences and archives analysis code separately.

Raw artifacts are under `runs/learning/seed-replication-v1/`; the portable report
and inspected PNG/SVG are `docs/experiments/2026-09-24-seed-replication.*`.
Lesson 6 explains seeds, tensors, paired deltas, sample SD, consistent failures
and the limits of fixed-split replication. A two-seed interim audit is retained
locally with its own explicit seed list; the published result includes all three.

**Verification and next decision.** 146 tests pass, with one existing dependency
warning; lint/format/strict typing checks pass. The larger project remains open:
oracle-quality parity, selected final-test evaluation and measured serving are
unachieved. Investigate persistent errors and declared checkpoint selection.
Generation currently recomputes the full prefix per token; KV caching is the next
useful systems experiment, requiring numerical/output equivalence and measured
latency before any speed claim. No commit or push.

## 2026-09-24 — KV caching: exact output parity, modest latency benefit

**Decision and implementation.** Recorded the cache plan before implementation
and measurement. Added opt-in, request-local incremental decoding, retaining the
ordinary forward path and uncached generator as reference. Each layer stores
normalized/rotated keys and projected values at KV-head count; RoPE and masks
use absolute prefix positions. Cached one-token attention must not apply a
top-left causal mask that hides its past. Chunked append uses rectangular masks,
including the existing sliding-window rule. No parameters/checkpoint tensors are
added. Cached inference skips losses and the unused prompt-set head, supports
unpadded chunks and rejects training/autograd, incompatible owners/shapes/devices/
dtypes and context overflow. The simple cache concatenates tensors, retains the
whole prefix and does not implement batching with padding or cache eviction.

**Correctness.** CPU/CUDA tests compare full logits with prefill/chunk/single-token
decoding at FP32 atol 2e-6, rtol 2e-5, across dense, sliding-window, soft-cap, MoE
and prompt-set variants. Additional checks cover immutable prior cache contents,
unchanged state dictionaries, request isolation, EOS and truncation. CLI tests
verify separate execution/config identity while preserving training identity.
Using the fixed prompt-set seed-1729 final checkpoint, all 222 cached validation
responses exactly match their archived uncached counterparts, including status
and errors. Metrics remain identical: F1 0.629018, raw exact zero, EOS/validity
222/222. Final test remains untouched.

**Measured outcome.** Eight validation prompts span four target-length ranks per
dimension. After warmup on each path, three repetitions alternate within-pair
order and bracket each generation call with CUDA synchronization. On the same
RTX 5070 Ti, FP32 inference without autocast:

| Native generation | Uncached | Cached |
| --- | ---: | ---: |
| Sum of 24 request times | 20.352372 s | 19.526820 s |
| Median mixed-length request time | 0.852972 s | 0.811689 s |
| Maximum peak allocated memory | 39,807,488 bytes | 37,362,176 bytes |

The summed-time ratio is 1.042278 (4.06% less time). Per-prompt ratios range
approximately 0.99–1.055. This is a modest local microbenchmark result, not a
robust large speedup, HTTP/concurrency/SLO measurement, energy result or oracle
comparison. The persistent cache formula gives 1.12890625 MiB for eight layers,
two KV heads, one 289-token prefix, head width 32 and FP32; actual peak allocation
also includes model and temporary tensors. Caching preserves learned errors.

**Follow-up diagnosis.** A separate cProfile trace of cached TOXAPEX/COLOR (171
generated tokens) records 715,638 `Vocabulary.class_of` calls and 21,033 Torch
module-dispatch calls. `allowed_token_ids` cumulatively accounts for 0.191 s and
its caller `constrain_logits` 0.239 s in a 1.441 s instrumented run. These times
overlap, include instrumentation effects and do not isolate GPU utilization or
kernel time. Source inspection confirms repeated dimension/mode vocabulary scans
even during answer generation. Precomputed classes/incremental grammar state are
concrete next targets; dispatch, small kernels and synchronization are plausible
additional contributors, not proven independent causes of the small speed gain.

**Evidence and disposition.** Raw evaluation/profile/source/script/pstats artifacts
remain under `runs/learning/kv-cache-v1/`. Portable reports are
`docs/experiments/2026-09-24-kv-cache.json` and `2026-09-24-kv-overhead.json`.
Lesson 7 records tensor shapes, causal-mask pitfalls, memory arithmetic and the
measured limitations. 159 tests pass with one existing warning; lint, formatting,
strict typing and docs checks pass. Keep caching opt-in. Next, address avoidable
generation overhead and native serving integration while continuing persistent
TYPE-error analysis. Oracle parity and selected final-test/serving evidence remain
open. No commit/push.

## 2026-09-24 — Native HTTP boundary and grammar overhead

**Decision.** Connect the existing decoder to a native Windows HTTP service
before attempting Linux/vLLM or throughput claims. Evaluation and serving now
share one checkpoint/config/data/split validation loader. Deployment copies the
SQLite graph, freezes product labels in memory and saves source/config/identity
receipts. This prevents mutable curation data from silently changing hydration.
The model still receives five protocol tokens and emits IDs; Python owns JSON,
IGNORE, return limits, output filtering and hydration.

**Execution contract.** Strict request validation rejects unknown products and
hidden attributes. A bounded semaphore admits running plus waiting requests;
a lock serializes generation. This is deliberately simple scheduling, with fresh
request-local caches rather than continuous batching. Failed or incomplete raw
generation returns 502 with evidence, never a successful hydrated answer. Busy
admission returns 503; invalid requests return 422. Tests cover queued execution,
capacity limits, release after failure and frozen labels.

**Diagnosis and correction.** Profiling showed dimension/mode vocabulary scans
repeated throughout answer generation. The parser now scans steering classes only
when needed and validates earlier prefix positions, fixing malformed prefixes
that later states could previously overlook. It still scans the prefix and entity
candidates; it is not an incremental parser. For the same cached TOXAPEX/COLOR
171-token trace, class_of calls fell from 715,638 to 15,393. Instrumented cumulative
allowed_token_ids time fell from 0.1912413 to 0.0441857 seconds; constrain_logits
from 0.2387736 to 0.0925276. These overlapping cProfile times do not establish an
uninstrumented speedup or isolate GPU utilization.

**Actual HTTP evidence.** A real Uvicorn server with ordinary loopback HTTP
requests reproduced all 222 archived seed-1729 prompt-set raw responses exactly.
Processed metrics also exactly match the offline policy: F1 0.6318792717 and
105/222 exact answers. IGNORE/limits preserve raw generation; hidden attributes
return 422. Serial client median was 0.690365 seconds, total 167.552057 seconds;
no warmup or concurrency sweep, first request included. This correctness workload
cannot be compared to the earlier eight-prompt benchmark as a speedup. The owned
server shut down cleanly. Model quality remains unchanged and below the oracle.

Raw reports and deployment receipts are in `runs/learning/native-service-v1/`;
portable results are `docs/experiments/2026-09-24-native-service.json` and
`2026-09-24-native-overhead.json`. The live deployment source-tree hash is
`0b8faf55745393f534b4b2309e3fda566854fccab18dc605644584c942bb71b6`.
Lesson 8 connects tensor inputs, request metadata, snapshot identity and queueing.
185 tests pass with one existing dependency warning; core imports remain
Torch-free. Final test was not used. Oracle parity, checkpoint selection,
concurrency/SLO/energy comparisons and vLLM remain open. No commit/push.

## 2026-09-24 — Separating opening decisions from list continuation

**Question and predeclared intervention.** Native serving preserves model errors,
so investigate their location before adding capacity. The compiler orders targets
by shared-attribute count, confidence and key; a correct opening ID may cue a
familiar list. Hold the seed-1729 prompt-set final checkpoint fixed, use validation
only, measure teacher-forced accuracy by position, and inject exactly one correct
first target before greedy cached continuation. The hint consumes one position
of the unchanged 507-token completion budget. This is explicitly oracle-assisted
diagnosis, not deployable quality or a new training candidate.

**Controls and evidence.** Shared loading validates checkpoint/data/split identity.
All 222 zero-hint diagnostic responses match their archived unassisted responses
exactly, including order, EOS, validity and errors. The small scripted-model test
checks zero-hint parity with production generation, hint budget accounting,
syntax masking and preservation of truncation. Raw and same-policy processed
metrics were independently replayed from the saved responses. The exact executed
script and source archive hashes were verified. The working script subsequently
binds immediate-evaluation lambda loop variables explicitly for lint; the executed
copy remains archived unchanged. No production model/serving code was modified.

**Results.** Teacher forcing gives 108/222 correct first targets (48.65%),
27,814/28,221 correct later target tokens (98.56%) and 221/222 correct EOS decisions.
These denominators differ. With the first-ID hint, raw F1 rises from 0.629018 to
0.929725; processed F1 from 0.631879 to 0.934817. Processed exact answers rise
105 -> 174/222: 69 rescued, 105 retained, 48 still wrong. All remaining failures
are TYPE. COLOR becomes 103/103 exact; TYPE improves 32 -> 71/119. INKAY/TYPE
reaches the completion bound without EOS; no successful output is fabricated.

**Answer-family support.** Grouping by dimension and targets-union-subject set,
3 validation queries have no identical training family, 51 have 1–4 matching
training queries, and 168 have at least 5. Unassisted exact counts are 0/0/105;
hinted counts are 0/12/162. Sparse TYPE combinations remain difficult. Family
support is confounded with dimension and structure; these observational groups
do not establish frequency as the sole cause. A first ID also contains more
information than a binary hint, and may cue the answer rather than reveal a
relation representation already present before intervention.

**Disposition.** This supports an opening/conditioning bottleneck plus residual
TYPE continuation problems. It does not support feeding oracle hints in serving.
First-token weight 32 already failed in the earlier experiment; localization does
not validate that remedy. Training logs show late validation set-loss deterioration
despite declining training loss. Next compare saved checkpoints under a declared
complete-answer selection rule before enlarging the model. Final test stays
untouched; oracle parity and serving comparisons remain open.

Raw evidence: `runs/learning/sequence-diagnosis-v1/diagnosis.json`, executed script
and source archive. Portable report: `docs/experiments/2026-09-24-sequence-diagnosis.json`.
Source-tree hash remains `0b8faf55745393f534b4b2309e3fda566854fccab18dc605644584c942bb71b6`.
Lesson 9 teaches conditional factorization, tensor alignment, hint information
and causal limits. 186 tests pass; lint, formatting and strict typing pass.
No final-test use, commit or push.

## 2026-09-24 — Saved checkpoints do not improve the retained candidate

**Hypothesis and declared selection.** The auxiliary validation set loss worsened
late in training, suggesting an earlier snapshot might generalize better. Before
generation, fixed the existing seed-1729 prompt-set candidates at steps 500, 1000,
1500 and 2000. Required every response to terminate validly, then maximized
processed exact-set accuracy, breaking ties by processed F1 and then earlier step.
This is an exploratory within-run selection, not a new training run or an
independent final estimate. No oracle hints, data changes or protected-test scores.

**Implementation and controls.** The comparison script uses the shared loader,
validates checkpoint step/sidecar/payload identity, and evaluates identical cached
FP32 greedy decoding on all 222 validation prompts with the same 507-token bound.
Raw and deterministic same-policy outputs remain separate. Contextual FP32 losses
are token-count-weighted CE and query-weighted first-target/set losses. The last
checkpoint must match every archived raw response before accepting the sweep.
Unit tests cover selection priorities, eligibility, ties and incomplete/non-finite
candidate evidence. No production model or serving implementation changed.

| Step | Raw F1 | Processed F1 | Exact after policy | TYPE exact | COLOR exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.111656 | 0.111981 | 20/222 | 6/119 | 14/103 |
| 1000 | 0.439333 | 0.441107 | 76/222 | 19/119 | 57/103 |
| 1500 | 0.610035 | 0.612749 | 105/222 | 33/119 | 72/103 |
| 2000 | 0.629018 | 0.631879 | 105/222 | 32/119 | 73/103 |

All 888 outputs terminate with valid syntax; all raw exact counts remain zero.
Step 2000 matches all 222 archived responses and wins by F1 after tying step 1500
on exact count. The latter two share 98 successful queries: seven are lost and
seven gained. The selected outcome is not better on every query/dimension.

**Loss interpretation.** FP32 set loss is smallest at step 1000 (0.585584), token
CE at 1500 (0.146419), and first-target CE at 2000 (2.692895). Step-2000 set loss
is 0.703523. Thus the auxiliary-loss upturn did not identify a better snapshot for
complete generation under the declared rule. Do not infer that overfitting never
occurs or that other schedules/intermediate checkpoints cannot help. The selected
validation result is adaptive and potentially optimistic; cross-seed selection
and final-test finalization remain separate work. Retaining an earlier saved
snapshot would not recover compute already spent, so no energy-saving claim.

**Evidence and disposition.** Retain the existing final anchor; no service default
change or new quality improvement is claimed. Raw per-checkpoint reports, source
archive and executed script are under `runs/learning/checkpoint-selection-v1/`.
The portable comparison, chart and SVG are `docs/experiments/2026-09-24-checkpoint-selection.*`.
Verified all input/checkpoint/report/source/script hashes and independently
recomputed F1/exact counts and policy replay across all 888 responses. The chart
was visually inspected. Lesson 10 teaches objective mismatch, confidence-sensitive
loss, lexicographic selection, retrospective versus actual early stopping, and
validation selection bias. 188 tests pass with one existing warning; Ruff,
formatting and strict mypy pass. The campaign completed; no test evaluation,
commit or push. Next address the diagnosed conditioning/rare-TYPE errors; oracle
parity and the serving comparison remain open.

## 2026-09-24 — Symmetric relation gradients improve the decoder in one seed

**Rationale.** First-token diagnosis and the checkpoint sweep point toward weak
conditioning and sparse TYPE continuation rather than a simple checkpoint fix.
SAME membership is symmetric. Added a diagonal bilinear auxiliary scorer through
shared product and dimension-token embeddings, inspired by the relation models
in [Yang et al., ICLR 2015](https://arxiv.org/abs/1412.6575). The normalization,
dimension projection and use as a decoder auxiliary objective are our experiment,
not a reproduction of the paper. Symmetry is appropriate here; transitivity is
not imposed because overlapping TYPE membership need not be transitive.

**Implementation.** `symmetric_relation_loss_weight` defaults to zero. With it
enabled, normalize entity/steering embeddings to norm sqrt(D), project the
dimension embedding with a bias-free D-by-D matrix, and score all products with
sum_k subject[k] * relation[k] * target[k] / sqrt(D). FP32 auxiliary computation
is retained under BF16 autocast. The scorer reads no future input tokens. Its
balanced binary loss uses the same training answer labels, collapses duplicates
and excludes the subject column. The earlier prompt-set loss retains its existing
negative diagonal. No attribute labels or extra held-out supervision are added.

The projection adds 65,536 parameters (6,624,256 -> 6,689,792 total), initialized
after all common weights. Token logits and cached decoding are unchanged at
common initialization; learned effects flow through gradients into shared
embeddings. The auxiliary scores never filter or inject generated IDs. Trainer
logs the separate symmetric loss; optional classifier inspection can select either
head. Tests cover symmetry, diagonal/gradient signs, CPU/CUDA BF16 finite behavior,
future invariance, initialization/token parity, cached logits, invalid inputs,
exact resume and a toy generated-answer overfit gate. All 194 tests passed.

**Declared run and result.** The plan fixes seed 1729, the existing split and
optimizer/schedule/batch, 2,000 steps and final-checkpoint selection. New loss
coefficient is one, added to unchanged token and prompt-set coefficients. The run
completed in 66.867 seconds versus the archived anchor's 61.796; these are observed
durations, not a paired throughput study. Normalized configs match except the
new coefficient/run name. Full validation generation uses no oracle hints.

| Metric | Prompt-set anchor | Symmetric auxiliary candidate |
| --- | ---: | ---: |
| Raw F1 | 0.629018 | 0.834080 |
| Processed F1 | 0.631879 | 0.837999 |
| Processed exact | 105/222 | 143/222 |
| TYPE processed exact | 32/119 | 52/119 |
| COLOR processed exact | 73/103 | 91/103 |

All 222 candidate responses are valid and terminated. Raw exact remains zero;
the same deterministic policy is applied to both. First-target validation CE
improves 2.694590 -> 1.639457 in the BF16 training diagnostics. The candidate
checkpoint hash is `3af8aa4a7bfae722aa27b04a8b2cc9982dedbf718915ed81ea088e31558a98d1`;
source tree `5dca4c0733e9d65b26675fbc27c27f7780246837fa5455091a1811276430f1f3`.

**Separate classifier evidence.** At fixed threshold zero, symmetric scores have
raw F1 0.986384 and return the subject on all queries. Subject exclusion gives
F1 0.990837 and 114 exact sets. The trained prompt-set head has raw F1 0.866673;
subject exclusion gives 0.870496 and 120 exact sets. These are membership sets,
not generated sequences/EOS evidence. Higher F1 with fewer exact answers than
the decoder illustrates differing error distributions; it cannot replace the
generation metric. Zero-threshold calibration was not tuned using final test.

**Disposition and evidence.** This is a promising single-seed candidate with 38
more exact answers, pending replication at seeds 1730/1731. Deployment remains
on the previous anchor. Raw reports are in `runs/learning/symmetric-relation-v1/`,
training artifacts in `runs/national_dex_symmetric_s1729_v1/`, portable comparisons
in `docs/experiments/2026-09-24-symmetric-relations.json` and
`2026-09-24-symmetric-classifiers.json`. Checkpoint/data/split/run identities and
raw/processed metric replay were checked. Lesson 11 records intuition, equations,
tensor shapes, transfer and calibration limits. Lint/format/strict typing pass;
core imports remain Torch-free and docs build with existing link warnings. All
launched runs completed; no final-test use, commit or push. Oracle parity and
matched-quality serving comparisons remain open.

## 2026-09-24 — Learning iteration 11: control replay and replication amendment

**Question.** Does the symmetric relation improvement repeat at seeds 1730/1731,
using the fixed split, final 2,000-step checkpoints and unchanged coefficient?
The predeclared plan is `docs/experiments/2026-09-24-symmetric-replication-plan.md`.

**Failed replay.** Re-running the prompt-set seed-1729 control on current source
produced matching normalized settings but different final weights: 0/92 model
tensors bitwise identical. The new control checkpoint hash is
`a79daf5325179356fb49fc93d70822991a7e0a8c6f861b9244ce700e71582b7d`.
Real training uses `deterministic: false`. GPU nondeterminism and intervening
source changes are possible factors; the cause has not been isolated. Do not
interpret the replay mismatch as evidence for a specific cause.

**Bounded regression check.** With the same archived weights and four fixed
training records, padded shape `[4,178]`, old/current CPU FP32 forward/backward
results match bitwise across 95 tensors: logits, three losses and all available
parameter gradients. This supports mathematical equivalence for that batch,
not bitwise replay of an entire CUDA training trajectory.

**Amendment before refreshed control quality was evaluated.** Use fresh
prompt-set controls for all three seeds on the same source as the candidates;
reuse the completed seed-1729 bridge run. Require identical source contents in
all six training archives. Preserve historical reports and the failed replay.
Keep the same paired improvement/promotion rule. This removes source history
as a between-variant confound without claiming deterministic GPU training.
The existing candidate also passed cached/uncached response parity on all 222
validation queries. Audit receipts live in `runs/learning/symmetric-replication-v1/`.

**Candidate validity failure.** Seed 1729 terminates validly on 222/222 queries;
seeds 1730 and 1731 each do so on 221/222. The failed queries are KECLEON/TYPE
and CARBINK/TYPE. Each emits 507 IDs without EOS, containing only 31/43 distinct
products and 2/6 distinct correct products. CARBINK ends with 227 copies of
PKM_WIGLETT; KECLEON ends with 17 copies of PKM_TOXICROAK and repeats earlier too.
Both already choose a wrong non-subject entity at emitted position two. The
observations support investigating sequence recovery/termination; they do not
isolate a causal mechanism. Increasing the length bound does not address the
observed low coverage and repetition. Saved failure rows and an inspected plot
are under `docs/experiments/2026-09-24-symmetric-termination-failures.json` and
`2026-09-24-symmetric-repetition.png`.

**Exploratory membership follow-up.** After declaring this follow-up, inspected
the existing symmetric head at threshold zero on the two failed queries, with
subject exclusion. KECLEON has 130 true positives, one false positive, no false
negatives (F1 0.996169); CARBINK has 132 true positives, 12 false positives and
three false negatives (F1 0.946237). This exposes a large classifier/generation
gap on those queries. These scores do not repair generated sequences and do
not supply EOS evidence. The portable receipt is
`docs/experiments/2026-09-24-symmetric-failure-classifiers.json`.

**Promotion decision.** The all-candidates-valid gate fails, irrespective of the
remaining paired quality results. Retain the existing prompt-set serving
reference and leave the conditional HTTP promotion campaign unrun. Keep the
new symmetric checkpoints as research candidates. Do not select the apparently
safe seed after observing failures in the others or waive the declared gate.

**Completed source-controlled paired comparison.** All six runs completed 2,000
steps and all 1,332 responses were audited against the exact validation partition,
checkpoint/run/config/corpus/split/runtime identities and actual source archives.
All six training source contents match. Per-seed control -> symmetric raw F1:
0.641433 -> 0.834080; 0.582655 -> 0.788165; 0.599258 -> 0.773864. Processed exact
counts: 108 -> 143; 97 -> 130; 100 -> 134, each out of 222. Mean raw F1 is
0.607782 +/- 0.030302 sample SD versus 0.798703 +/- 0.031461. Processed F1:
0.610554 +/- 0.030569 versus 0.802454 +/- 0.031617. Processed exact accuracy:
0.457958 +/- 0.025614 versus 0.611111 +/- 0.029992. Paired improvements average
0.191900 F1 (SD 0.015605) and 0.153153 exact accuracy (SD 0.004505). All three
pairs improve in both declared quality metrics; raw exact remains zero.

The refreshed prompt-set seed 1731 also has one invalid, non-terminating output
(AERODACTYL/TYPE; 507 emissions, 49 distinct products). Controls therefore have
665/666 valid terminated responses versus candidates' 664/666. This is evidence
of termination failures under both objectives, not proof that symmetry caused
the problem. The candidate still fails its predeclared all-valid gate.

Never-exact queries decrease from 84 (71 TYPE, 13 COLOR) to 55 (52 TYPE, 3 COLOR).
Queries correct in every seed increase from 58 to 99. TYPE exact counts improve
32 -> 52, 33 -> 50, 30 -> 48; COLOR 76 -> 91, 64 -> 80, 70 -> 86. These are three
initializations on one fixed query split, not unseen-product generalization,
robustness across partitions, oracle parity or equal-compute efficiency evidence.

**Artifacts and validation.** `docs/experiments/2026-09-24-symmetric-replication.json`
binds metrics and the withheld-promotion decision; PNG/SVG plots show the paired
results and explicit unfinished counts. Lesson 12 covers controls, tensor-level
replay, seed variability, sequential failures, classifier/generation differences
and a worked F1 example. The analysis CLI now supports the declared symmetric
pair and requires matching training source. Its paired subtraction has regression
coverage. All 195 tests passed (one existing warning), with lint/format/strict
mypy passing. No core model/serving code changed in this replication iteration.
All launched training/evaluation completed; no final-test use, commit or push.

## 2026-09-24 — Learning iteration 12: an explicit target-uniqueness decoder

**Why.** Replicated symmetric models improve answer quality but two responses
repeat until the length limit; a refreshed control also fails this way. Strong
membership classification on the same failed queries does not translate into
reliable sequential output. Test a structural decoder constraint separately
from changing training, rather than treating a policy improvement as learning.

**Decision.** Add `eval.prevent_repeated_targets`, default false, requiring
protocol-constrained decoding. Maintain a request-local set of entity IDs emitted
after ANSWER; mask their subsequent logits to negative infinity. The prompt
subject remains eligible until emitted; EOS is never added to the seen set.
There is no relation lookup, answer-count input, forced EOS or checkpoint change.
The existing 507-token budget can still be exhausted. Preserve the original
parser and all invalid original responses. Reports use `+unique-v1`, and native
serving records and forwards the setting.

**Declared experiment.** `docs/experiments/2026-09-24-target-uniqueness-plan.md`
compares both policies for all 222 validation queries in each frozen symmetric
seed. First require the current cached original decoder to reproduce every
archived uncached result; then measure the unique-target decoder under the same
runtime. Require unchanged model source, matching source/checkpoint/data/runtime
identities, full validity and no per-seed processed F1/exact regression before
conditional HTTP verification. Final test stays reserved.

**Implementation checks before the real campaign.** All 199 tests pass (one
existing warning). New coverage checks cached/uncached repetition suppression,
request-local state, subject eligibility, unchanged hard truncation, invalid
configuration rejection, and native policy/metadata propagation. Ruff/format
and strict mypy pass. No commit or push.

The new campaign initially stopped at import time because its script used the
wrong module for the existing `config_hash` helper. Corrected the import to
`plm.configuration`, checked the CLI help entry point, then started evaluation.
The failed startup performed no model evaluation or artifact mutation. This is
a tooling correction, not a scientific result about target uniqueness.

**Independent bounded batching probe while the serial policy campaign runs.**
The decoder already accepts `[batch, chunk]` inputs. An isolated prototype in
`runs/learning/batched-decoding-prototype-v1/probe.py` groups eight fixed
seed-1729 validation prompts, retains batch rows after EOS using ignored dummy
inputs, and maintains per-row uniqueness state. All 16 checked outputs match:
eight original-policy outputs against archived serial results and eight
unique-policy outputs against current serial execution. Completion lengths vary,
so the probe exercises rows finishing at different times. Core/evaluator/serving
code is unchanged by this prototype, and it is not an admitted backend. It ran
concurrently with another evaluator; no timing, throughput, concurrency or energy
claim follows. Full-partition parity, interface/error tests and isolated timing
remain necessary. Portable receipt: `docs/experiments/2026-09-24-batch-prototype.json`.

**Completed policy comparison.** All 666 original-policy outputs match their
archived uncached references exactly under current cached inference. All 666
unique-target outputs are valid and terminate with EOS, versus 664/666 original
outputs. The unique rule changes 15 answers (4/6/5 across seeds), removes all
observed duplicates, and gains or loses no processed exact answers. Every first
emitted ID remains unchanged; every first divergence masks an entity already
emitted in that answer. The model source and checkpoint weights are unchanged.

Per-seed processed F1 changes: 0.837999346 -> 0.837362769;
0.791895575 -> 0.792574298; 0.777466989 -> 0.778938320. Mean processed F1 is
0.802453970 -> 0.802958462; paired delta +0.000504492 (sample SD 0.001064700).
Mean raw F1 is 0.798703004 -> 0.799299449; paired delta +0.000596445
(SD 0.001058536). Processed exact counts remain 143/130/134 out of 222;
mean 0.611111111. Raw exact accuracy remains zero.

**What improved and what did not.** KECLEON/TYPE seed 1730 now emits 157 IDs
and EOS with 19/130 correct products, versus 507 emissions and two correct
products without EOS. CARBINK/TYPE seed 1731 now emits 92 IDs and EOS with
15/135 correct products, versus 507 emissions and six correct products without
EOS. Completion improves while relation selection remains poor.

AERODACTYL/TYPE seed 1729 worsens: both policies first choose the wrong entity
IRON_MOTH. The original repeats it at position two; the unique rule chooses
SALAZZLE instead and changes the remaining history. Distinct correct products
drop from 43 to 15 while emitted IDs rise from 118 to 160. This explains why
removing duplicates afterward and forbidding them during generation are not
interchangeable. The rule cannot fix the first choice because its seen set is
empty at that position.

**Disposition.** Validity passes, but the predeclared no-per-seed-F1-regression
condition fails at seed 1729. Keep the option experimental and default-off;
retain the existing serving reference and do not run the conditional HTTP
promotion campaign. This is a partial completion repair, not a relation-quality
solution. The full portable report is
`docs/experiments/2026-09-24-target-uniqueness.json`; raw reports and executed
source/script snapshots are in `runs/learning/target-uniqueness-v1/`.
Source tree: `1e67ae6439d6baf173734af0d5663365fbf362a60b1aa6e81aa1a8510d3fbc31`.
Lesson 13 records equations, tensor shapes, the adverse example, full results and
the separate bounded batching probe. All launched processes completed; no
final-test evaluation, commit or push. Next: validate a batched evaluator to
accelerate complete-answer experiments, then address the initial relation choice.

## 2026-09-24 — Learning iteration 13: fixed-group batched evaluation

**Why.** Serial complete-answer evaluation makes each experiment expensive.
The eight-query prototype matched serial outputs, including different completion
lengths and both repetition policies. Validate and integrate this path before
using it to accelerate further quality experiments.

**Implementation.** Added `generate_responses` for constrained KV inference on
fixed groups of five-token prompts. All inputs are validated before model work;
output order and duplicate queries are preserved. Each row tracks completion
and optional seen entities. Finished rows retain cache slots and receive ignored
EOS inputs, never appended to their returned sequences. Only active logits must
be finite; malformed tensor dimensions fail explicitly. A shared result/parser
helper keeps serial and batch output contracts aligned. No model/training change.

`eval.generation_batch_size` is a strict positive integer, default one. Larger
groups require protocol constraints and KV caching, are wired through CLI
reports, and carry `+batch-v1`. Native HTTP serving rejects this offline option
rather than pretending to provide request batching. No scheduler change.

**Declared campaign.** `docs/experiments/2026-09-24-batched-evaluation-plan.md`
requires complete token/validity/error/metric parity against all 1,332 serial
outputs across three frozen symmetric checkpoints and both policies. Batch size
eight exercises final partial groups. Only if parity passes, run warmed,
alternating paired decoder timings on the fixed first eight seed-1729 queries,
with synchronization and output checks. No HTTP/SLO/energy/oracle advantage claim.

**Pre-campaign verification.** All 207 tests pass (one existing warning).
Coverage includes different EOS positions, ignored finished-row nonfinite
outputs, active-row nonfinite/shape failures, duplicate/reordered queries,
fresh state, truncation, invalid prompts/configs, native rejection and full CLI
parity on a trained integration fixture. Ruff/format/strict mypy pass, and the
campaign CLI help entry point is exercised. Defaults and retained checkpoint
remain unchanged. No final-test evaluation, commit or push.

**Campaign outcome.** All 1,332 batch-size-eight responses exactly match the six
frozen serial reports, including token IDs, termination/validity/errors and
raw/processed metrics. Both original-policy unfinished answers remain unfinished.
Checkpoint and report hashes, executed-script copy, archived/current source and
runtime identities verify. Evaluator source archive SHA-256:
`030d9ff342d732c04761aa8136f897b2b196f6e3bbd7d6796741e5c9030c9e85`.
Portable evidence: `docs/experiments/2026-09-24-batched-evaluation.json`.

**Isolated decoder benchmark.** After full parity, three warmed alternating pairs
on the fixed eight seed-1729 original-policy queries give serial times
7.37989/7.38329/7.36428 s and batch times 1.38678/1.37689/1.40573 s. Summed-time
ratio is 5.30711x; individual pairs range 5.23876x-5.36227x. Peak PyTorch allocated
GPU memory is 36.3501 versus 51.7095 MiB. Every timed output matches. Hardware:
RTX 5070 Ti, Torch 2.11.0+cu128, cached FP32. Timing does not run alongside
another launched model job. These are complete-workload throughput measurements,
not individual latency, HTTP SLO, concurrent users, total device memory, energy
or a comparison against the oracle. Three repeats on eight fixed queries limit
generalization. Completed rows retain slots: 1,150 useful returned tokens across
1,752 output slots (65.64%), which must not be called GPU utilization.

**Decision.** Admit fixed batching as an opt-in offline evaluation path for
further experiments, retaining default size one and native serving behavior.
The full campaign checks size eight, these models, and this runtime; do not
assume bitwise parity for every future model/batch configuration. No retraining,
quality gain, checkpoint promotion, final-test access, commit or push. Lesson 14
adds equations, tensor dimensions, finished-row mechanics and measured tradeoffs.

## 2026-09-24 — Learning iteration 14: learned guidance at the first target

**Hypothesis.** A frozen model's auxiliary membership head may know a relation
that its autoregressive first choice fails to express. Apply a soft learned
membership penalty only at that position, isolating an inference intervention
from retraining and from the earlier uniqueness constraint. Reuse the existing
prompt-only forward head; do not duplicate its scoring math or feed expected
answers/graph relations to generation. Log-sigmoid converts relation logits to
bounded-above evidence, scaled by alpha. It is a scoring heuristic, not a claim
of calibrated membership probability after class-balanced training.

**Declared experiment.** `docs/experiments/2026-09-24-first-guidance-plan.md`
fixes three seeds, alpha 0/1/4/16, all 222 validation queries, cached FP32 batch
eight and original repetition policy. Each zero setting must exactly replay
its frozen reference before guided runs. Record all results and first choices.
Further-integration eligibility requires 666 valid/terminated outputs, no
per-seed processed-F1 or exact-set regression, and strict mean exact-set gain.
This is validation selection, not a serving promotion or untouched final test.

**Implementation.** Offline script wraps decode: on a fresh five-token prefill,
score the existing relation head with prompt IDs only and add alpha times
log-sigmoid to product logits at the last position. No other positions or token
classes change; subsequent decoding uses the same cache. Zero strength bypasses
the head entirely. Known limitation: subject membership scores were excluded
from auxiliary loss; no subject mask is introduced in this intervention.
Nine focused tests pass, covering row independence, first-position-only changes,
no mutation of source logits, cache preservation, zero bypass, new-request state,
invalid strengths/head outputs and rejection of per-seed regressions by selection.
Registry and predeclared plan added. No production model/config changes.

**Campaign results.** Completed all 2,664 executions. All 666 zero-strength
responses and metrics replay exactly. Alpha 1/4/16 improve mean processed F1
from 0.8024539702 to 0.8670876733 / 0.8802419910 / 0.8832860157. Mean processed
exact accuracy rises from 61.111% to 66.667% / 67.868% / 68.168%. Exact counts by
seed: baseline 143/130/134; alpha1 150/144/150; alpha4 150/149/153; alpha16
151/150/153. Every setting improves per-seed aggregate F1/exact accuracy.
However, all retain 664/666 valid terminated answers. KECLEON TYPE (1730) and
CARBINK TYPE (1731) retain their entire failed 507-token sequences at every
strength. Every integration gate fails and selected_alpha is null.

**What changed.** Alpha 16 changes 90 initial choices, repairing 73 nonmember
starts and introducing two nonmember starts. First-member counts rise 578->649
of 666; exact-first counts 443->490. Every answer whose first choice stays fixed
is fully token-identical. Alpha 16 gains 48 exact answers and loses 1 (LYCANROC TYPE,
seed 1731), net 47; alpha 4 gains 45 and loses 0. For TOXTRICITY TYPE at
seed 1729, alpha 4/16 selects the subject itself
and processed F1 falls .709957->.100719. This illustrates a risk of unsupervised
subject scores, without proving a separate subject-mask intervention will fix it.
Both persistent loops already start with valid members, exposing the difference
between membership and the ordered continuation policy.

**Evidence and decision.** Portable report:
`docs/experiments/2026-09-24-first-guidance.json`; complete raw reports, executed
script/shared-helper copies and source archive: `runs/learning/first-guidance-v1/`.
Checkpoint/report/script/source hashes verify; no model source changes. Preserve
this as a useful partial quality improvement and failed completion experiment.
Next investigate continuation/constraint interactions under a new declared
comparison, rather than assuming independently useful rules combine safely.
No retraining, production config changes, checkpoint promotion, final-test use,
commit or push. Lesson 15 includes the equations, tensor shapes, all results,
regressions and the gate failure.

**Parallel review.** At the user's request, independent read-only review ran
alongside result analysis/documentation. It found no blocking code defect or
answer leakage and suggested precise numerical-bias and prefill-cache tests.
The reviewer then owned only that test file and added those checks. The full
suite passed 216 before the campaign; the expanded focused file passes 12 tests.
The experiment source remains unchanged. GPU campaigns remain sequential.

## 2026-09-24 — Learning iteration 15: guidance x uniqueness

**Why this comparison.** First guidance improves aggregate quality but leaves
both original repetition loops; uniqueness alone repairs completion but slightly
regresses one seed's F1. Test the composition directly rather than assuming
independent benefits combine. Alpha 4 and 16 are chosen from the preceding
validation grid: former avoids losing baseline exact answers, latter has the
highest mean quality. This remains validation-informed development.

**Declared gate and scope.** Plan:
`docs/experiments/2026-09-24-guidance-uniqueness-plan.md`. Run 1,998 new unique
queries across three seeds/strengths; reuse 1,998 verified no-unique observations.
Zero-guidance unique outputs must replay exactly. Combined first choices must
match guidance alone; any first divergent old choice must be a repeated target.
Report factorial interaction and marginal effects. Eligibility compares the
combined intervention to the original decoder, not to guidance alone; report
marginal regressions rather than concealing that changed comparison. The earlier
first-guidance-alone gate remains failed. Passing identifies an integration
candidate, not automatic checkpoint/HTTP promotion or oracle parity.

**Parallel diagnostic.** An independent CPU audit verified nine prior report
hashes and recomputed processed metrics. Alpha 4's 214 failures: 15 nonmember
starts, 156 member-but-wrong-position starts, 43 exact-first continuation failures.
TYPE accounts for 203 failures, COLOR 11. Of 43 exact-first failures, 42 miss targets;
25 diverge by second/third processed token. Only 15 failures contain repeats.
TYPE exact improves 150->154/357 with guidance while COLOR 257->298/309. Thus the
remaining major problem is TYPE continuation/coverage, not only first membership.
Self-starts account for 9/214 failures; masking the subject alone cannot explain
most residual errors. Full diagnostic saved in first-guidance-v1/residual-review.
This evidence informs the next modeling question without cancelling the useful
completion interaction test. A second agent owns only focused interaction tests.

**Completed campaign.** All 1,998 new executions finished; all 666 alpha-zero
unique outputs/metrics replayed exactly. First-choice invariance and all
first-divergence repeat checks pass. Combined alpha 4 and alpha 16 both produce
666/666 valid, terminated, repeat-free answers. Mean processed F1 is .8809372936
and .8839904575 respectively; exact counts remain 150/149/153 and 151/150/153.
Uniqueness changes 16/17 outputs relative to each guided setting, gaining zero
and losing zero exact answers. Its marginal F1 contribution is positive in every
seed, but some individual answers regress (DUNSPARCE COLOR/1729: .0421->.0153).
The two original loops now terminate but have only about .132 processed F1;
completion repair is not a relation-quality solution.

**Interaction and selection.** Alpha 16 paired F1 interactions are
+.0009497449 / +.0002637580 / -.0006136534; exact interactions are zero. A negative
interaction can coexist with a positive marginal gain. Both combined candidates
pass the predeclared comparison against the original decoder. Select alpha 16
plus uniqueness for integration verification by the declared exact/F1 tie-breaks.
Mean processed exact accuracy is 68.168%; TYPE 153/357 versus COLOR 301/309. The
policy remains offline, and prior standalone gate failures remain historical
failures. No HTTP promotion, final-test evaluation, training or source-model change.

**Evidence.** Portable campaign and residual reports:
`docs/experiments/2026-09-24-guidance-uniqueness.json` and
`docs/experiments/2026-09-24-guidance-residuals.json`. Complete output reports,
copied scripts, source archive and analysis scripts live under
`runs/learning/guidance-uniqueness-v1/`; residual analysis remains under the
preceding campaign's residual-review directory. Hashes and independently
recomputed processed metrics verify. Lesson 16 explains factors, tensor masks,
marginal effects, interactions, selection and the dominant TYPE coverage gap.

**Parallel verification and next integration.** Full suite passed 241 tests;
22 new interaction tests cover factorial coverage, arithmetic, gate comparisons,
selection and divergence invariants. An independent integration map recommends
a default-zero finite guidance field, one shared prompt-only head helper for
serial/batch paths, and full-token replay before HTTP checks. Existing HTTP
verification omits raw token-ID and decoding-descriptor comparisons; strengthen
those before accepting a guided deployment. Keep the existing head computation
and extra prefill forward initially, so integration does not also introduce
an unverified optimization. New config hashes must not rewrite old training
identity. No commit or push; all model and agent work completed.

## 2026-09-24 — Learning iteration 16: integrate guided inference

**Decision.** The selected combined policy passed the offline interaction gate.
Integrate a default-zero `eval.first_target_guidance_alpha` through shared serial
and batch generation, CLI reports and native deployment/prediction. Preserve
model code/weights and the existing extra prompt-forward computation. Positive
strength requires constrained KV decoding and a trained symmetric head; no
subject mask or other decoding intervention is added. Zero preserves legacy
policy descriptors and avoids the extra head call.

**Evidence boundary.** Configuration changes belong to inference config/source
identity, not old training identity. Reports now include full token IDs and
explicit decoding settings. Strength must have an unambiguous numeric descriptor.
The HTTP verifier checks complete tokens and canonical policy identity, raw and
processed metrics/targets, bound, cache/constraint/uniqueness/guidance settings,
and native batchsize1. Legacy no-token reports receive explicitly partial parity
labels. The old reference's missing full-token check is not treated as new proof.

**Declared verification.** `docs/experiments/2026-09-24-guided-integration-plan.md`:
1,332 core batch executions against alpha0/16 archived unique reports, followed
only on success by666 realHTTP comparisons at alpha16 across the three frozen
checkpoints. No model-source change, retraining, final test or performance claim.
All servers are owned ephemeral loopback instances and must stop after each run.

**Parallel implementation.** Worker1 owns shared inference helper/generators/tests;
worker2 owns the HTTP verifier and its tests. Root owns config/CLI/native/runtime,
training-fixture integration tests and docs. Fixture supervision needed two
attribute groups so symmetric training has negatives after excluding the subject;
this is a test-data correction, not a change to the real corpus. A wrapped CLI
error assertion was also normalized for terminal line wrapping. No committed or
shared model evidence was changed.


### 2026-09-24 — Guided integration verification outcome

**Result.** All 1,332 planned core replays (222 validation queries x three
checkpoints x alpha 0/16, uniqueness enabled) match archived complete tokens,
parsed fields, errors, completion flags and raw/processed metrics. The source
archive for the integrated inference implementation is
`9d65344bb13ee3844bb69de531c55503357b64364f94bd37deb8a6cfafbbefdf`;
model-source bytes and checkpoint/training identities are unchanged. The
previous experimental archive remains separately identified. The dependency
lock and runtime match the candidate campaign.

**Application evidence.** Only after offline parity, real loopback HTTP passed
all 666 planned comparisons at alpha 16 plus uniqueness, serially across seeds
1729/1730/1731. Full raw tokens, parsed fields, processed targets and metrics
match. Policy/config/deployment/readiness identities, IGNORE/limit invariance,
invalid-subject 422 and clean owned-server shutdown all pass. A separate CPU
pass verifies report/script hashes, full tokens, training/source/runtime
identities and recomputes exact counts from the HTTP outputs. All model jobs
and temporary servers finished; no permanent service was started.

**Interpretation.** Accept the opt-in application integration on this tested
validation partition. Exact processed answers remain 151/150/153 per seed
(454/666, 68.168%); mean processed F1 remains .8839904575. TYPE is 153/357 exact,
COLOR 301/309. This reproduces the selected policy rather than improving quality.
Default guidance remains zero and the older prompt-set reference remains
available. There is no oracle-quality, final-test, SLO, concurrency or energy
claim. Further TYPE continuation/coverage work is required.

**Verification and teaching.** Full suite passed 315 tests before GPU execution;
Ruff, formatting, strict typing and Torch-blocked core imports passed. A parallel
static audit found no additional actionable correctness issue. Source and
executed runner/helper bytes stayed frozen during the campaigns. Lesson 17
explains weights versus policy, log-sigmoid guidance, tensor dimensions, full-token
regression checks, inference versus training identity, fixture corrections and
results. Portable receipts: `docs/experiments/2026-09-24-guided-integration.json`
and `docs/experiments/2026-09-24-guided-http.json`; complete evidence is under
`runs/learning/guided-integration-v1/` and `runs/learning/guided-http-v1/`.
No retraining, protected test evaluation, commit or push.


## 2026-09-24 — Learning iteration 17: diagnose union coverage

**Decision before probes.** Integration preserved the selected policy, but
TYPE contributes 204 of its 212 non-exact validation observations. Inspect
membership knowledge separately from serialization before adding another
architectural feature. The compiler orders shared-attribute count descending,
then confidence and product key: binary membership alone does not encode this
ordering or the requirement to enumerate a two-attribute union.

**Declared diagnostic.** `docs/experiments/2026-09-24-continuation-diagnosis-plan.md`
fixes CPU attribute/branch/support analysis and frozen teacher-prefix probes at
all three seeds. Known prefixes are explicitly oracle-assisted diagnostic input.
Changing only the subject while keeping that prefix can create inconsistent
counterfactuals; report sensitivity rather than claiming causal forgetting or
counterfactual quality. Membership uses the existing zero-logit threshold.
No graph facts enter model scoring, no test partition is evaluated, and core
source/weights remain unchanged. Parallel work owns CPU analysis and focused
probe tests; GPU jobs remain sequential.


### 2026-09-24 — Union diagnosis outcome and next objective

**Structural result.** Single TYPE is 124/153 exact (F1 .92722), dual TYPE
29/204 (F1 .70870), COLOR 301/309 (F1 .97831). Of 175 dual-TYPE failures,
75 equal exactly one complete original-subject branch; 107 completely cover
exactly one branch. Dual TYPE accounts for 11,855 of 13,097 omitted target
observations. Shared-two retention is 890/945; shared-one retention is
22,259/34,059. The main missing mass belongs to the broader union. Same-signature
training support correlates with quality, but this does not establish a causal
oversampling intervention. Coverage v2 hardens UTF-8/exclusive receipt/source
checks and reproduces v1 analytical results; both executed script copies remain.

**Model result.** All 666 validation query-seed observations completed 3,352
unique teacher-prefix probes and 9,390 condition scores. All original first
choices match archived generation. The symmetric head marks 11,737/11,855
omitted dual-TYPE targets positive (99.00%); nevertheless its thresholded sets
are only 10/204 exact, versus generation 29/204. Membership F1 is not an exact
quality substitute. Dual-TYPE original-teacher next-token correctness is
51/204 first, 92/204 second, 118/204 third, 201/204 middle and 202/204 EOS.
These are oracle-assisted local probes, not complete generated answers.

**Sensitivity.** For dual TYPE, rotating the subject changes the next-token
distribution with mean TV .957/.421/.399/.0052 at first/second/third/middle.
The first position also applies explicit alpha-16 guidance, so that comparison
mixes policies. The unguided second/third-to-middle decrease is consistent with
target-prefix dependence; inconsistent counterfactuals do not prove internal
forgetting. A correct supplied prefix can conceal failures to construct that
prefix autonomously. Report first-divergence probes with their actual positions
rather than interpreting their mixture as pure continuation accuracy.

**Decision.** Select early continuation-set supervision as the next training
experiment. Reuse the contextual prompt-set projection after one, two and three
teacher targets, supervising still-unemitted members with balanced binary loss.
This directly tests whether early hidden states can retain both branches; no
new graph facts enter inference. Keep the original next-token/prompt/symmetric
objectives and guided-unique inference policy. The declared comparison in
`docs/experiments/2026-09-24-continuation-set-plan.md` uses default zero versus
coefficient one, fresh matched-source controls, three seeds and fixed budgets.
No such objective has been implemented or trained yet. A failed experiment
remains informative; do not replace exact-generation criteria with head F1.

**Verification and artifacts.** Full suite 349 passed, one existing warning;
independent audit reproduces all joined counts/metrics and verifies source,
script, checkpoint/reference and data identities. Executed core/source remains
`9d65344bb13ee3844bb69de531c55503357b64364f94bd37deb8a6cfafbbefdf`.
Portable coverage and continuation receipts live in `docs/experiments/`;
full evidence in `runs/learning/type-coverage-v2/` and
`runs/learning/continuation-diagnosis-v1/`. Lesson 18 adds equations, tensor
shapes, worked union example, measured limits and a visually checked figure.
No weights, protocol, corpus or serving defaults changed; no final test,
training run, commit or push. All launched jobs and agent work completed.


## 2026-09-24 — Learning iteration 18: early continuation-set objective

**Implementation decision.** Follow the predeclared continuation-set comparison.
The default-zero finite coefficient requires the contextual prompt-set head.
Reuse its projection at states after one, two and three teacher products;
construct remaining-product labels from the masked future suffix, removing
consumed IDs and the subject. Average valid stages per query, then participating
queries. Empty participation is differentiable zero with count zero. No learned
parameters are added, and label-free inference does not compute this objective.

**Evidence contract.** Log the auxiliary loss separately, weighting validation
by participating-query counts. A hand-calculated regression test distinguishes
this from incorrectly weighting by batch size. Model tests cover suffix shifts,
negative consumed IDs, short/padded rows, empty participation, common initial
parameters, unchanged zero/logit behavior, causal future-token invariance and
gradients. The trained integration fixture checks checkpoint objective identity,
exact CPU resume, CLI generation and native API behavior with the new objective.

**Campaign.** Fresh control/candidate runs use seeds 1729/1730/1731, split1729,
2,000 steps, batch32, identical parameters and all other recipe settings.
Only continuation coefficient0/1 differs within each pair (apart from run paths).
Validation keeps guided16/unique/KV/batch8/bound507. Full token/processed evidence
and subgroup gains/losses feed the declared gate, not classifier-only metrics.
No change to protected final test, serving defaults or protocol semantics.


## 2026-09-24 — Learning iteration 18 results: remaining-set supervision fails the gate

**Executed comparison.** Completed all six predeclared fresh control/candidate
runs at seeds 1729/1730/1731: 2,000 updates, batch32, continuation coefficient0/1,
prompt/symmetric coefficients1, same 6,689,792 parameters and fixed guided16,
unique-target, cached batch8 validation. Core source stayed frozen throughout;
archive `101d661a16e3f17ecc924c1afed3a188cea25fe8e70bb1df2a19b523f363ecd1`.
No protected final-test evaluation or HTTP promotion was part of the trial.

**Result.** Mean processed F1 improves .901220824953 -> .915234889845, while
exact answers fall 466/666 -> 463/666 (33 gained,36 lost). Per-seed exact counts
are 156->159,158->147,152->157. Single TYPE improves129/153->132/153; dual TYPE
falls34/204->32/204; COLOR falls303/309->299/309. Every candidate response is
valid, terminated and unique. However, no-seed-regression, mean-exact-improvement
and dual-exact-improvement all fail. Retain the declared gate: coefficient one
is not selected for further integration. The feature stays default zero.

**Interpretation.** Dual-TYPE F1 improves in all seeds without improving pooled
exact answers. All candidates also improve validation token CE (.13076->.12250,
.13143->.12134,.13220->.12465), but teacher-forced loss does not guarantee better
free generation. Seed1730 BUDEW and EKANS retain correct first choices yet omit
113 or add113 targets, respectively. DEWGONG COLOR switches first choice and
falls to1TP/93FP/86FN. These are observations, not proof of gradient conflict
or a uniquely identified causal failure. Mean recorded training time rises
68.55->75.98seconds (~10.84%); update budgets match, compute/time budgets do not.
Timings are descriptive and establish no serving or energy advantage.

**Evidence discipline.** Fresh controls outperform the historical454/666exact
reference, so comparing only candidate463 to that old result would misattribute
an improvement. CUDA nondeterminism and current matched controls remain explicit.
The CPU campaign audit checks source/archive/tree/lock, checkpoint/config/runtime,
pinned corpus/split and every full token sequence before reconstructing metrics.
A Windows slash-serialization discrepancy is resolved by reconstructing the exact
CLI recipe and separately checking the trainer-effective manifest path, without
relaxing hashes or changing core source. An independent CPU implementation using
set arithmetic agrees with raw/processed metrics, subgroups, gains/losses and gate.

**Artifacts and verification.** Full evidence is under
`runs/learning/continuation-set-v1/`; summary SHA256
`558d18391131333461efee09c3045848a8ffbca7a6a5e4b3d436ac7d49fbf819`, independent
audit `05f69fdeb8855ce469efbd79b6dcdd52a1eb1ed52b0096427979e503aa4bfe65`.
Portable receipt `docs/experiments/2026-09-24-continuation-set.json` and Lesson19
include equations, tensors, failure cases and a visually checked plot. Full suite
406passed (one existing dependency warning); Ruff/format/strict mypy59files and
whitespace checks pass. Docs build passes with six existing source-link warnings.
Parallel workers owned implementation and audit/docs; GPU runs were sequential.
All jobs completed. No commit, push, service launch or protected-test selection.

**Next decision.** Before another weight sweep, inspect paired first divergences
and remaining-set scores along generated versus teacher prefixes. The failure of
this coefficient/budget does not reject all continuation supervision; it also
does not justify changing acceptance metrics after observing outcomes. Oracle
parity and matched-quality serving/concurrency/energy remain unproven.


## 2026-09-24 — Learning iteration 19: diagnose raw versus teacher prefixes

**Decision before execution.** The continuation-set candidate improves token CE
in all seeds and mean set F1, but fails exact-answer and dual-TYPE gates. Before
another training sweep, compare first ordered divergence with actual set errors,
and probe raw generated versus position-matched teacher prefixes across all six
frozen checkpoints. The declaration is
`docs/experiments/2026-09-24-matched-prefix-plan.md`.

**Rationale and boundaries.** Subject removal cleans the final answer but cannot
remove a previously emitted subject from the autoregressive context. Track raw
and processed positions separately. Match teacher progress by all consumed
non-subject products, including wrong ones, rather than oracle overlap. Distinguish
reordering from missing/extra products. Reuse the contextual projection through
temporary hidden-state hooks; no core edits, training labels as inputs or graph
facts in inference. Record late-position and generated-prefix distribution shifts,
already-seen teacher tokens and full-prefill versus archived-cache discrepancies.
Teacher prefixes remain oracle-assisted diagnostics, not generated improvements.
Parallel workers own CPU structural analysis and frozen-probe implementation;
GPU work remains sequential. The failed candidate remains default zero.


## 2026-09-24 — Learning iteration 19 results: prefix robustness and ordered choice

**Evidence.** Completed CPU classification of all 1,332 matched outputs and
frozen probes across all six checkpoints: 7,980 paired prefixes, 15,960 condition
scores, 11,272 unique prefills. All 7,980 raw-prefix next choices agree with
archived cached generation. Independent CPU auditing verifies input hashes,
alignment, condition-specific guidance, masks, head arithmetic and aggregates;
it does not independently recompute the network. Core/model source remains
archive `101d661a16e3f17ecc924c1afed3a188cea25fe8e70bb1df2a19b523f363ecd1`.

**What changes the next action.** Both arms emit the subject in 647/666 outputs.
Only 3/36 newly lost exact cases have subject insertion in the first three raw
positions; any insertion precedes ordered divergence in 10/36 losses. Pure
correct-prefix early EOS is three control cases and zero candidate cases.
Subject insertion and simple early stopping do not explain most regressions.
Correct processed starts followed by set failure increase 39->52 overall and
25->41 for dual TYPE. A BRUTE_BONNET answer is reordered but set-correct, so
ordered diagnostics must not replace the declared exact-set criterion.

**Two gaps.** On dual TYPE, candidate head F1 at raw positions1/2/3 is
.7712/.7549/.7467 versus matched-teacher .9427/.9406/.9544. On its 203 final-set
failures the corresponding raw values are .7350/.7138/.6999, versus teacher
.9425/.9399/.9510. The auxiliary task is learned, with imperfect transfer to
self-generated context. Yet valid-member next-token mass can remain high while
ordered enumeration fails. BUDEW TYPE seed1730 has identical correct raw/teacher
prefix AMOONGUSS, head191TP/2FP/3FN, and still chooses ARBOK rather than BELLSPROUT;
the final answer omits113 targets. Thus exposure gap is not the entire explanation.

**Stopping caveat.** All 172 candidate dual-TYPE failures end with missing
products, but diverged earlier. Their endpoint remaining-product mass is about
.0000212. The head at these unsupervised late positions is weak; across all204
candidate dual endpoints it has77,298 false positives. Do not convert early
head success into a late EOS gate without a separately justified experiment.

**Next declared trial.** The36 newly lost exact cases fall34->11 in ordered raw
first-choice correctness, while correct-member first choices change34->35.
Declare first-target weight8 with continuation/prompt/symmetric coefficients1,
three seeds and unchanged2,000-step budget/inference. The continuation1/first1
runs isolate that weight interaction; the stronger continuation0/first1 runs
remain acceptance references. Reuse frozen controls only with exact unchanged
source/runtime/data/unrelated-config identities. The earlier weight32 failure
used a recipe without these membership objectives. Eight is exploratory, not
optimal; first-target weighting cannot directly fix BUDEW's wrong second choice.
Full contract: `docs/experiments/2026-09-24-continuation-order-plan.md`. Not run yet.

**Verification and artifacts.** Full suite434passed, one existing dependency
warning; Ruff/format/strict mypy59files and whitespace checks pass. Docs build
passes with six existing source-link warnings. Lesson20 contains equations,
tensor shapes, denominator caveats, failure examples and a visually checked
figure. CPU summary SHA256
`2182a155a3e1218d871d1c69ee1f21c79d8de0fc737fb285ad4f53b6736c30ca`;
probe summary `3dab79810b54fad71d01f657732c5a1f5a0e35c38702a899dea3a6539ab4a013`;
independent audit `b51af1bcc46777c41bd2dcda7931612dc8b33ad940ea295b59d84fb3266fe579`.
Portable receipts live in docs/experiments; full artifacts in
runs/learning/matched-errors-v1 and matched-prefixes-v1. No new trained weights,
final-test access, service launch, commit or push. Previous quality/gate unchanged.
All agent work, GPU probes and verification jobs completed.

## 2026-09-24 — Learning iteration 20: stronger first-target weighting fails

**Declared intervention and identity.** Completed the three-seed weight8 trial
specified in docs/experiments/2026-09-24-continuation-order-plan.md. Each new run
uses continuation/prompt/symmetric coefficients1, 2,000 updates, batch32,
fixed alpha16 guidance, uniqueness, KV cache, offline batch8 and bound507.
All source/runtime/data/unrelated resolved settings match the six frozen
references. The model remains 6,689,792 parameters; no model/config source
changes were needed. Source archive remains
`101d661a16e3f17ecc924c1afed3a188cea25fe8e70bb1df2a19b523f363ecd1`.
CUDA nondeterminism remains a limitation of paired seeds.

**Result and decision.** New exact counts are152/153/159, pooled464/666, mean
processed F1 .8941644584898988. Original controls score156/158/152, pooled466,
meanF1 .9012208249527678; continuation1/first1 scores159/147/157, pooled463,
meanF1 .9152348898453773. New single/dual/COLOR exact counts are131/31/302,
versus original129/34/303 and previous132/32/299. Against original controls,
34 cases gain exactness and36 lose it; against previous continuation runs,
35 gain and34 lose. All666 candidate outputs are valid, terminated and unique,
but all three quality gate conditions fail. Keep weight8 unpromoted and
continuation experimental/default zero. A one-answer gain against a weaker
reference does not rescue the predeclared failure or its larger F1 decline.

**What we learned.** Final-batch first-target CE falls from approximately
.137/.097/.110 to .00740/.00763/.00658, while unweighted validation first CE
worsens in every seed, from1.468/1.578/1.389 to1.579/1.940/1.702. Validation
token CE also worsens all three. This is consistent with stronger fitting of
training starts without the intended held-out improvement; it does not identify
the cause or prove that every larger weight fails. The normalized denominator
changes ordinary-token gradient weights too, so this is an objective interaction.
The final-batch statistic is not a full training-set measurement.

Raw correct-member first choices improve646->657/666 from continuation1/first1,
but ordered first choices fall502->490 and processed ordered starts514->494.
Canonical enumeration and membership are distinct. ARMALDO COLOR seed1730
changes AGGRON->ALTARIA and loses all106 expected products while adding170,
despite the original control being exact. F1 captures failure severity that a
nearly unchanged count of exact answers hides. Ordered diagnostics do not
replace exact-set acceptance: harmless reordering is still allowed by that metric.

**Cost and evidence.** Mean observed training seconds are89.82, versus75.98
for continuation1/first1 and68.55 for originals. This is not isolated timing or
serving evidence. Two parallel CPU audit/review responsibilities overlapped
sequential GPU work. Official and independent reconstructions agree on all
1,998 outputs, paired metrics, first choices, subgroups, receipt identities and
gate. Neither audit independently reruns the network. Historical script archives
remain unchanged; the new campaign snapshots its updated reusable helpers.

Official summary SHA256
`0d224d5b7008ef61be27cd26023ae0468222c9201698681f1c04877b491ca602`;
independent audit `52e8ddb732124398c64920d0061994935306c07015cefd01ca8343b6bcf1297a`.
Full evidence: runs/learning/continuation-order-w8-v1. Portable receipt:
docs/experiments/2026-09-24-continuation-order-w8.json. Lesson21 includes the
equation, tensor shapes, tables, concrete failure and interpretation limits.
Full444tests pass, one existing dependency warning; Ruff/format and strict
typing59files pass. Documentation builds with six existing source-link warnings.
No protected-test use, service launch, commit or push.

**Next research direction.** Examine whether alternative likely early prefixes
recover complete sets and whether learned scores can select them. Declare the
diagnostic before running it, retain the stronger matched controls, and separate
oracle-assisted possibility from deployable selection. Do not launch another
coefficient search on the strength of the failed first8 result. Oracle parity,
protected finalization and matched-quality serving evidence remain open.

## 2026-09-24 — Learning iteration 21 declaration: first-choice paths

After weight8 failed, test available continuations separately from the ability
to select them. Preserve the three stronger original controls and enumerate
their top4 guided legal first products, then greedily continue each. Separate
rank batches preserve the original eight-query numerical path and fresh caches.
All666 rank1 outputs must match archived raw IDs exactly. No model/source or
training change is needed; no oracle fact enters generation.

Declare total constrained-policy log probability as primary selector, mean log
probability as fixed secondary, and best validation-answer score only as an
oracle ceiling. All scores include the first product and EOS; bound507 includes
both. Only valid completed paths are eligible; failed fallback remains visible.
This tests whether the likely first alternatives contain useful paths, not
general beam search or later-branch recovery. Plan:
docs/experiments/2026-09-24-first-choice-paths-plan.md. Evaluate all three seeds,
all222validation queries and all groups. GPU work remains sequential; decoder
implementation and independent review have separate parallel ownership.

## 2026-09-24 — Learning iteration 21 results: candidate availability is not selection

**Completed experiment.** The declared first-choice branching campaign evaluated
all 222 validation queries across the three original control models. Four fresh
rank batches of eight preserve the numerical reference path. All 666 rank1
raw sequences match archived generation; all 2,664 candidate paths terminate
validly with unique products. Model source, weights, runtime, data, unrelated
settings and the protected-test boundary remain unchanged.

**Results.** Original greedy returns 466/666 exact answers, mean set F1
.9012208249527678. Primary sum scoring returns 462, F1 .9014439880258099;
secondary mean scoring returns 465, F1 .9010414153289494. Sum gains six exact
cases and loses ten; mean gains six and loses seven. Both pass validity but
fail all three quality conditions. Neither selector is promoted. A tiny overall
F1 increase for the primary score does not satisfy exactness or subgroup gates.

**Available versus selected.** At least one exact answer exists among the four
paths on 501/666 observations; mean oracle-best F1 is .9559779490923205. This
oracle uses validation answers after generation and is not a usable decoder.
Exact single/dual/COLOR counts are greedy129/34/303, sum131/24/307,
mean126/32/307, and available136/57/308. All ten sum-selector losses are dual
TYPE. Sum misses 39 available exact answers; mean misses 36. The restricted
oracle still misses 165 complete answers, so this branching cannot resolve the
entire task even with perfect selection. All counts reuse 222 queries across
three seeds; paths are not independent samples.

**Score behavior.** Sum chooses ranks1/2/3/4 on 611/41/11/3 queries; mean on
607/33/16/10. Average scored path length, including EOS, changes from120.75
to117.44/126.02. PALKIA TYPE seed1731 has an exact 220-token rank1 path starting
DRACOVISH. Sum instead chooses ALTARIA at71tokens/F1 .481; mean chooses
ALOMOMOLA at155tokens/F1 .825. INKAY TYPE seed1729 has an exact rank4 MALAMAR
path, yet both selectors choose inexact alternatives. These cases demonstrate
harmful score preferences without identifying length as the sole cause.

**Costs and checks.** Observed evaluation times are164.07/158.41/162.77seconds,
about8.1minutes total excluding checkpoint loading. Maximum allocated GPU memory
is56.63MiB. This is not a matched serving, latency, concurrency or energy result.
Independent CPU auditing reconstructs all score sums/means, masks and token
budgets, selection, metrics, groups, paired changes, gates and receipt identities;
it does not recompute neural logits. Full457tests pass (13new focused tests),
one existing dependency warning; Ruff/format and strict typing59files pass.
Docs build passes with six existing source-link warnings. Lesson22 and a checked
figure explain the probability chain rule, length effects, cache shapes and
oracle boundary. Registry, navigation and portable receipts are current.

Summary SHA256 `7b9db0b137d53ce2c350932aec61b1a8e94295f9affbae16aa55cc52f659fa8a`;
independent audit `e45dcfc7c2709ffa07bf8a719038bcabce812b8daefb62191b7caf0f01e91e48`.
Full evidence: runs/learning/first-choice-paths-v1. Portable receipt:
docs/experiments/2026-09-24-first-choice-paths.json. All GPU and audit jobs ended;
no final test, serving promotion, commit or push.

**Next declared test.** Reuse the exact same four paths and frozen symmetric
membership head. Rank each processed set by the sum of its predicted member
logits, equivalent up to a query constant to independent Bernoulli set log
likelihood. Balanced-loss calibration and independence are assumptions, not
proven probabilities. No coefficient/threshold search, training or wider
generation. Sum distinct IDs canonically so identical sets tie exactly. Require
the original controls' nonregression and strict exact/dual gains. The independently
reviewed contract is docs/experiments/2026-09-24-set-reranking-plan.md; not run yet.
This tests whether an existing set signal recognizes the alternatives that
sequence likelihood missed. Oracle parity and matched-quality serving remain open.

## 2026-09-24 — Learning iteration 22: learned set reranking succeeds

**Question and decision.** The four-path experiment contained more exact answers
than greedy, but sequence likelihood failed to recognize them. Test the declared
symmetric-head selector on the same frozen paths and original three control
checkpoints. No retraining, regenerated candidates, coefficient search or oracle
inputs enter selection. Sum the FP32 membership logits over each eligible
SAME-processed set in canonical ID order with math.fsum. This is equivalent to
full independent Bernoulli log scoring up to a query-constant exclusion term;
it does not establish calibration of the balanced-loss head.

**Evidence.** The selector reaches 168/168/165 exact validation answers across
seeds 1729/1730/1731, pooled 501/666 versus greedy 466/666: 35 gains and zero
exact-answer losses. Mean F1 rises from .9012208249527678 to .9555846395858687.
Single-TYPE/dual-TYPE/COLOR exact counts rise from 129/34/303 to 136/57/308.
Every declared gate passes. All available exact candidates are selected, but
165 observations have no exact candidate and eight select below the best
available F1. These are the same 222 queries across three trained seeds, not
666 independent queries. Exact accuracy is 75.23%, so oracle parity remains open.

The independent CPU audit reconstructs all 666 saved head vectors' candidate
scores, all 2,664 candidates' eligibility, selections, outputs, metrics and
receipts. It verifies the Bernoulli identity but does not recompute neural
logits. Summary SHA256 c515fae06f7db1810e8697badf59d53ffb41675cbb8fc543d1a14feee7252c5c;
audit SHA256 22d55b7b58b988d53e78ef4f0bf167f24a2b52e2a27d3f53b603dd00fbeeb2aa.
Full evidence lives in runs/learning/set-reranking-v1; portable results and
Lesson 23 preserve equations, tensor details, successes and remaining failures.

**Cost and limits.** Synchronized head forwards total 2.80 seconds, dominated by
the first seed (2.396 seconds versus .194 and .211); this is not a steady-state
benchmark. Peak allocation for scoring is 36.58 MiB. The previous 485.24 seconds
of candidate generation must still be counted in a full pipeline measurement.
The source archive stays 101d661a16e3f17ecc924c1afed3a188cea25fe8e70bb1df2a19b523f363ecd1.
Full 476 tests, Ruff/format and strict typing pass. No protected final test ran.

**Next decision.** Integrate the successful rule as a default-off shared decoder
option under docs/experiments/2026-09-24-set-reranking-integration-plan.md. Require
fresh complete candidate/selection parity and native HTTP replay for every
validation query and checkpoint. Authenticate historical config JSON before
schema defaults; distinguish training and inference identities. The review
requires trained-head evidence from checkpoint metadata and full raw-token,
termination/protocol/error parity at HTTP. This integration is declared, not yet
implemented. Oracle parity and matched-quality serving/energy remain unfinished.

## 2026-09-24 — Learning iteration 23 declaration: reproduce set selection in the application

The successful offline selector is now implemented behind the default-false
`eval.symmetric_set_reranking` option. Shared generation preserves four fresh
rank batches, unique targets, canonical symmetric-logit scoring and rank-one
failure fallback. Native IGNORE/limit directives still act only after selection.
Loading requires a positive trained-head model setting matched to actual
checkpoint metadata and at least one training step. No model weights, model
architecture or training recipe changed.

Adding the field intentionally changes the inference configuration/source hash.
`validate_saved_config` authenticates the saved canonical JSON before inserting
only the new false default; any unrelated schema normalization fails. Historical
training hashes remain intact. Three original run manifests passed this adapter
without changing their old evidence. The shared core has 25 focused CPU tests;
32 configuration/training/CLI/native integration checks also pass. Full National
Dex offline and HTTP parity is still pending, so this is implementation evidence,
not yet acceptance of the integrated policy. Follow the reviewed integration
contract and preserve every mismatch rather than weakening output comparisons.

## 2026-09-24 — Candidate coverage diagnosis and pair-composition declaration

While the integration replay runs, inspect saved candidate coverage using CPU
set operations only. Among 165 observations with no exact path, 69 have an exact
union of two whole paths (all dual TYPE), 43 cover all truth members collectively
but have no exact whole-path subset union, and 53 still omit correct members
across every path. The last group has 3,134 missing product occurrences. Blind
four-way union is destructive: 501 selected exact answers become eight, with
495 losses and two gains; mean F1 falls from .955585 to .583401.

The root independently recomputed the core counts against frozen per-seed paths,
in addition to the diagnostic script's receipts. Full summary:
runs/learning/set-candidate-coverage-v1/summary.json, SHA256
67974a44184a48ecb67b62bbb5dab4c9f3ac05a2b30e4858fee771a767ea0dc4.
Portable aggregates are docs/experiments/2026-09-24-set-candidate-coverage.json.
These are oracle-assisted diagnostics, not new learned-policy quality results.

Declare docs/experiments/2026-09-24-pair-union-plan.md: retain all four originals,
add all six pair unions, score with the unchanged saved symmetric logits, and
prefer originals on ties. Require improvement against the strong 501/666 selector,
with per-seed nonregression and strict pooled exact/dual gains. No implementation
or experiment has run. A composed set is explicitly not a sequence emitted by
the autoregressive model; retain source paths, do not invent EOS/probabilities,
and require a separate integration contract if composition succeeds. This future
experiment does not alter the ongoing original four-path integration acceptance.

**Integration offline result.** All three complete replays pass with exact
candidate scores and full raw outputs: 2,664 candidate paths, 666 selected
answers and 666 separately executed disabled greedy answers. Independent CPU
partial audit confirms those comparisons and all three configuration bridges.
Per-seed replay wall times are 183.8234, 186.4357 and 180.0235 seconds; peak GPU
allocations are 59,317,248, 57,596,928 and 57,269,248 bytes. These timings include
both candidate generation and disabled-greedy verification, not a serving SLO.

The owned verifier is still running (root session 22905), now in real loopback
HTTP checks. Full integration acceptance and the terminal independent audit
remain pending. Do not restart from file absence or an observation timeout;
resume the verified live process. Source/config/script/plan inputs are frozen
for this campaign. The pair-composition declaration was independently reviewed
and clarified for duplicate provenance slots, ineligible exact-answer credit,
canonical set serialization and the separate 570-answer oracle ceiling.

## 2026-09-24 — Learning iteration 24: learned pair composition succeeds offline

**Question and implementation.** Execute the already reviewed pair-union plan
using the four frozen paths and saved FP32 symmetric logits per query. Preserve
all ten provenance slots (four originals plus six pairs), canonical sums and
original-first ties. Both source paths must be eligible. The new script performs
no model forward pass and gives compositions no synthetic raw sequence/EOS or
protocol metric. It replays all 666 original selections/scores before expansion.
Forty focused CPU tests and both implementation reviews pass.

**Result.** Exact sets rise from 501/666 to 569/666: 68 gains and zero exact-answer
losses, all in dual TYPE. Seeds 1729/1730/1731 reach 191/192/186 exact sets from
168/168/165, with mean F1 .9629406553851149/.9730912228699782/.964656280497194.
Overall mean F1 rises from .9555846395858686 to .9668960529174291. Dual exact
counts rise 57->125/204; single TYPE136/153 and COLOR308/309 stay unchanged.
All declared gates pass. There are 90 pair selections and 576 originals; every
pair selection happens to be dual TYPE without subgroup labels entering selection.

**Limits and failure.** Mean precision falls .98697518->.98204612 while recall
rises .94071282->.96379647. PILOSWINE TYPE/1731 is the only per-query F1 regression
(.99578059->.99159664) and the only available exact pair missed. Exact pair(1,3)
scores877.4085; incorrect pair(1,4) scores878.3962, replacing REGICE with ANORITH.
The true member's logit is-2.572607 versus-1.584898 for the wrong member, so the
incorrect swap is penalized less. There are97 inexact observations:96 lack an
exact ten-pool candidate and one is misselected. Exact accuracy85.44% is still
short of oracle parity. The570-answer oracle availability is not learned quality.

**Evidence and cost.** Independent stdlib audit reconstructs all6,660 candidate
slots/scores/eligibility/ties, selected sets, source paths, metrics/groups/gates
and immutable input/script/helper/source/plan receipts. Summary SHA256:
4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992.
Audit SHA256:13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057.
Current analysis source archive remains e66ae65f1156f697eeb06352c360e87bf3974a70575fe7bce6b5ac016c0f070c;
historical neural training identity remains separate. Neither analysis imports
Torch. Pool construction/scoring totals .2972919 CPU seconds, excluding input
checks/reporting and all earlier neural work. The485.24-second historical path
cost remains relevant; no integrated serving/SLO claim follows from this timing.
Full evidence is runs/learning/pair-unions-v1; portable JSON and checked PNG/SVG
figure accompany Lesson25. Ruff/format pass. Core source/config is untouched.

**Parallel integration status.** The existing four-path campaign still runs in
root session22905. Its first full HTTP checkpoint passes222 raw/processed/hydrated
responses, seven metadata/invalid-request checks and a separately receipted bound1
failure; independent partial audit confirms them. The other two checkpoints and
terminal full audit remain pending. Composition integration needs a distinct
selected-set/multi-source contract; no serving change follows automatically from
this offline gate. Protected test and matched-quality serving/energy remain open.


## 2026-09-24 — Iteration 25: stage composition without changing the live replay

The successful saved-candidate union experiment warrants a distinct set API.
Implemented in an isolated local checkout, not the active inference checkout:
`C:/Users/mateo/Documents/Github/plm-pair-composition-staging`. A 306-file
hash-verified copy preserves the dirty baseline rather than starting from the
much older HEAD alone. The managed checkout helper stalled during a remote
credential lookup; its owned fetch was canceled and a local detached worktree
was created. No commit or push occurred.

Composition reuses the four existing rank batches and one final FP32 scoring
pass. Existing per-rank guidance forwards remain unchanged to preserve the
numerical path. Ten immutable slots retain originals/pairs, canonical members,
scores and source eligibility. No synthetic emitted sequence or EOS is created.
A default-off flag enables separate CLI set reports and `/v1/predict-set`;
legacy prediction remains a single raw path. Both endpoints share admission
and execution limits. Failed answers receive zero quality and size, with raw
source-path diagnostics separate from set metrics. Registry updates are in the
isolated checkout beside the implementation.

The full CPU suite passed: 602 passed, 9 skipped in 40.16 seconds; strict mypy
passed 62 source files and Ruff/format checks passed. CUDA was explicitly hidden
with CUDA_VISIBLE_DEVICES=-1; an empty PowerShell environment assignment removed
the variable and did not disable CUDA, which was caught during setup before tests.
These CPU checks are not GPU replay acceptance.

A new serial-versus-batch CPU test initially failed because slot scores differed
by at most 3.67872416973114e-08; 29 and 28 differences across the two tiny trained
objective configurations were exclusively score differences. Source paths,
selected sets and metrics were identical. Difference receipts are preserved in
staging runs/staging/. The CPU cross-batch test now permits abs1e-7/rel1e-6 for
scores only. The planned same-shape GPU replay remains zero-tolerance; no saved
reference, ranking policy or scientific gate was changed.

Independent review found two verification gaps: the HTTP integration test was
checking CLI fields after sending HTTP, and a pooled source count could hide a
3-source result paired with a 5-source result. Both are fixed. Three real tiny
training/CLI/HTTP integration cases pass with actual serial HTTP source/slot
comparison; two focused metrics cases pass with the per-query guard. Later
changes were limited to those fixes and their focused verification; static
checks remain green. Documentation builds with the six existing source-link
warnings. Lesson 26 records the contracts, shapes and failed-test lesson.

The active campaign remains frozen at source archive
`e66ae65f1156f697eeb06352c360e87bf3974a70575fe7bce6b5ac016c0f070c`;
all archived source members still match. Its second HTTP checkpoint reached
200/222 at this session checkpoint; its third and terminal audit remain pending.
The new replay harness is being implemented separately and must authenticate
successful prior campaign and audit receipts before any new GPU work. No fresh
composition quality, oracle parity, final-test or serving benchmark claim follows
from this staging work. CPU jobs share machine resources, so campaign wall times
remain diagnostic rather than controlled performance evidence.

Follow-up in the same session: HTTP seed 1730 completed all222 queries, exact
parity true and owned servers stopped. Its independent partial audit passed
888 candidates,222 chosen,222 disabled,222 HTTP responses, metadata/failure
checks and receipts. Report SHA256
`b3de91f4215764a95c3e787e8a9113fc455707cc3513f93f68dcbc36b379e75c`;
wall1884.5577006 seconds is diagnostic only. Seed1731 and terminal audit remain
pending; two of three HTTP checkpoints now independently pass.

Final staging follow-up: the completed replay harness adds 37 focused cases.
The final full CPU suite passes 639 tests with 9 skips in 35.66 seconds, including
all review fixes; Ruff/format checks pass for 149 files and strict mypy passes
62 source files. A saved-data CPU replay (no Torch import or model forward)
exactly reproduces all 666 results, 2,664 source paths, 6,660 slots/scores and
per-seed metrics. Versioned receipt:
`runs/learning/pair-composition-staging-saved-replay-v2.json` in the isolated
checkout, SHA256 `bb185034e0f3df0e3c60935a34722e929fee878d83115786ca1d8c7806b8b3ec`.
The new harness requires reviewed prior summary/audit SHA256 arguments before
loading helpers or Torch. Its full prerequisite check and fresh GPU execution
remain pending. The old campaign is now on HTTP seed 1731 (50/222 observed).

## 2026-09-24 — Iteration 26: declare a saved-evidence error diagnosis

Before proposing another decoder change, inspect the 97 remaining pair-selection
errors using only the authenticated 666 validation observations and their saved
FP32 head scores. Recompute subgroup counts and candidate availability; separate
missing source coverage from erroneous members in covered sets. Count negative
true-member scores inside already-correct selections, and score signs for false
positives and false negatives in the failed selections. This diagnoses whether
blind removal of negative-score members would necessarily discard some correct
answers. No alternative policy is evaluated or accepted, no threshold is tuned,
no model forward runs and no protected-test records are used. The existing
four-path HTTP replay and staged composition acceptance gates remain unchanged.

## 2026-09-24 — Iteration 26 results: coverage, score signs and application replay

The declared saved-evidence diagnosis independently reproduces all 97 remaining
errors: 53 lack some true member across all four sources, 43 cover every true
member but have no exact ten-slot candidate, and one overlooks an available exact
candidate. The fixed candidate pool therefore permits at most one additional
exact answer through ranking alone. Of 3,850 missing true-member occurrences,
3,810 have positive auxiliary logits; 3,100 of those are absent from every source.
This identifies a candidate-coverage limitation without establishing its cause.

Blindly deleting negative-scored members would necessarily damage 62 currently
exact answers, removing 103 true-member occurrences. The current failures comprise
63 missing-only, 17 extras-only and 17 mixed cases. A deletion-only transformation
of the current selections can repair at most the 17 extras-only cases, hence its
exact count is bounded above by 569 - 62 + 17 = 524. This is a logical bound from
saved evidence, not a measured alternative-policy result. BERGMITE / TYPE,
seed 1729, provides a concrete counterexample: the exact 47-member answer includes
ARCTIBAX despite its approximately -0.547853 logit. Query-balanced auxiliary loss
does not guarantee correct signs or probability calibration. Lesson 27 explains
the masks, loss and failure modes, with a visually checked figure.

Diagnosis summary SHA256:
`f5c826295bfc30466dc435c5a995134fe198686f69f2890cac2e3f2435053523`.
Independent audit SHA256:
`1639c4b6fd2a45600436fbbd4d9a566e43be9016ed4e618750604c3315b7f895`.
Portable results: `docs/experiments/2026-09-24-pair-error-diagnosis.json`.
Both analyses use authenticated saved validation evidence and no model forward.

The earlier four-path application campaign has now completed and independently
passed all three checkpoints: 2,664 raw candidate paths, 666 selections,
666 disabled comparisons, 666 HTTP responses, 21 metadata/invalid checks and
three short-bound failure contracts. Summary SHA256:
`780f2e219d55164acefaa3b29a6b25e803154baa057d64a5002f05733a4d99de`;
terminal audit SHA256:
`96cce95af7cfc5001a1b94ed5df34d30eeb729ae04bd7c2141cba53b47e1c3ad`.
All owned-server shutdown receipts pass. Wall times are diagnostic, not controlled
performance measurements. The unchanged baseline passed all 554 tests afterward.

Only after this acceptance, incorporated 17 staged implementation/config/test/
registry/harness files with per-file before/after hashes recorded in
`runs/learning/pair-composition-incorporation-v1.json`. Existing dirty files were
checked against their staging baseline before incorporation. The new main checkout
passes 661 tests including CUDA tests, Ruff/format checks for 149 files and strict
mypy for 62 source files. The harness preflight authenticates 138 inputs. Review
also added partial offline failure reports and rejection of unexpected HTTP
top-level fields; its focused suite now has 50 passing cases.

The declared fresh pair-composition replay is running sequentially on the GPU.
Its source archive SHA256 is
`1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376`;
all 64 archived members match the live files. Source, configs, harness and plan
remain frozen while it runs. The first two offline checkpoints completed without
harness mismatches; the first also passed independent auditing. HTTP and terminal
acceptance remain pending. Existing evidence shows small batch-shape-dependent
FP32 score differences, so the new exact serial-HTTP score gate may reveal a
numeric mismatch even with identical selections. This possibility is recorded
before observing the new HTTP result; no tolerance or gate has been changed.

Documentation builds successfully with the same six existing source-link
warnings. No training, protected-test evaluation, oracle-parity claim, serving
throughput/energy claim, commit or push occurred in this iteration.

## 2026-09-24 — Iteration 27: declare direct membership-head ablation

The saved-error diagnosis warrants one fixed ablation: predict every non-subject
product with a strictly positive frozen symmetric-head logit, using all 1,025
columns. This removes the autoregressive candidate restriction from the existing
additive scorer. It necessarily loses at least 62 already-exact observations
whose true members have negative scores; whether new recoveries offset this
damage is not yet measured. The zero threshold follows the additive-score
maximizer and is declared before execution, not selected through a search.

The plan `docs/experiments/2026-09-24-direct-membership-plan.md` fixes all inputs,
empty/zero behavior, truth-free prediction, baseline reconstruction, membership
change diagnostics and per-seed/pooled/dual-TYPE gates. Independent design review
confirmed the known-loss implication and scope. This is a CPU-only saved-score
experiment, not a runtime decoder, protocol-output or serving-performance claim.
No training, extra threshold, model forward or protected-test evaluation is
authorized by this plan. The ongoing composition application replay is unchanged.

## 2026-09-24 — Iteration 27 result: unrestricted scores improve F1 but lose exact sets

Executed the one declared direct membership rule after 23 focused CPU tests and
a successful preflight reconstruction of all 666 pair-baseline observations.
The independent stdlib audit reproduces every prediction, per-query metric,
membership change and gate from original authenticated evidence. No new model
forward, training or protected-test evaluation occurred.

The result fails its gate. Exact sets fall from 569 to 339, with three gains and
233 losses. Per-seed exact counts change191/192/186 to112/116/111. Macro F1 rises
from .9668960529174291 to .9895858651596365, precision from .9820461214803038
to .9823401536146523 and recall from .9637964716104676 to .9974930922686877.
Every seed improves F1 but regresses in exact count. Dual-TYPE exact changes
125/204 to13/204; COLOR308/309 to238/309; single-TYPE136/153 to88/153.

The intended coverage recovery occurs: direct prediction recovers3,810 missing
true-member occurrences, including3,100 absent from all four source paths.
It removes2,003 wrong-member occurrences but introduces1,556 new wrong members
and loses206 previously selected true members. Among formerly exact answers,
it adds795 wrong members and loses103 true members. Counts are query-member
occurrences. No query predicts an empty set. This explains why strong membership
F1 is compatible with poor whole-set accuracy: modest errors spread across many
otherwise exact answers. Candidate restriction preserves useful structure for
these checkpoints, even though it misses some members. Neither direct threshold
prediction nor a new threshold/hybrid is promoted from this result.

Summary SHA256:
`2cbe82aad296b6ff4a920a951de38a5999a23e0eb8507d40444254e5d404ceab`.
Independent audit SHA256:
`8111692647153787e095da94c64ec497ebc41537b6f6c3ea8ad1d5b046b8df42`.
Declared plan SHA256:
`4004639897b88d54090ce94bfecf26adf5b2a75686be4b8cdf97da2155d19da0`.
Portable results: `docs/experiments/2026-09-24-direct-membership.json`.
Lesson28 and a visually checked PNG/SVG explain the additive optimum, head
tensor formula, shared-embedding constraint, known losses and observed result.
Saved-score validation/construction took .14274820004357025 seconds, excluding
historical neural computation, authentication and reporting; no speed claim.

In parallel, all three pair-composition offline checkpoints independently pass:
2,664 sources,6,660 slots,666 selections and both compatibility paths, exact
191/192/186. Offline report SHA256 values are
`c7ad1db44d3c7e0387369b1ffc982d5e7b96afb046c4f8a68111b235ae5a3f67`,
`932ad33ad4d0b00e29cd7d64966b3f0d5820df91764d9ba1188a22cdd84e1659`,
`cd713d451a1235f7ad73d8b524acbf3c250b482f78e84c2cfd08f778e3c5ae73`.
The first HTTP checkpoint is running; no terminal application acceptance yet.
All64 frozen source-archive members remain unchanged. The GPU campaign and
these CPU analyses share host resources, so replay wall times remain diagnostic.

## 2026-09-24 — Declare batch-shape diagnosis after exact HTTP replay failure

The pair-composition campaign terminated at its first HTTP checkpoint with
`HTTP parity failed`, preserving all 222 normal responses and all metadata/
disabled/failure/admission receipts. Normal and both auxiliary owned servers
report stopped. The summary SHA256 is
`7abc3a99ea8aa0e95749da2b379893572f7f76643a53ab98f8242c2c77da7042`;
HTTP report SHA256 is
`e6759659c0eec1130088c1324c8457cc2c0bf814af7773c05a8f6b251c421e87`.
Independent failure auditing is pending; this is not integration acceptance.

Root's saved structural comparison finds 224 mismatching composition records
(222 normal and two filtered), with all 2,240 scalar differences confined to
slot scores. All 2,220 normal slot scores differ; maximum absolute difference
is .00036144256591796875, between1396.0463314056396 and1396.0466928482056.
No other composition field differs. The observed scale is not a tolerance.

Before any change to code or the gate, declare a bounded neural diagnosis using
only seed1729, the same checkpoint, config, source and222 validation prompts.
Recompute prompt-only head outputs in the exact original batches of eight
(final batch six) and individually. Compare batch-eight logits and canonical
slot sums exactly to the frozen pair evidence, and serial slot sums exactly to
the saved HTTP evidence. Recompute selected-slot/tie decisions and compare them.
Repeat the first eight queries once at each shape to check fixed-shape
repeatability. No autoregressive generation, training, alternative policy,
protected-test evaluation or tolerance change is part of this diagnosis.
Record source/runtime/config/checkpoint/input/script identities and any failed
expectation. Source/config/harness and original reports stay unchanged.

### Batch-shape diagnosis result

The bounded replay completed 259 prompt-only forwards and passed every declared
expectation. Batch-eight/final-six logits and canonical sums exactly match the
original pair evidence. Serial canonical sums exactly reproduce all 2,220 HTTP
scores. All 222 selected-slot decisions remain unchanged, and the first eight
queries repeat exactly within each shape. Of 227,550 compared logits, 156,003
differ across shapes; the maximum absolute logit difference is
7.62939453125e-06, with no positive-membership sign changes. The maximum slot-sum
difference remains .00036144256591796875. The source/runtime/checkpoint/config
remain identical. This directly reproduces the observed batch-shape discrepancy
without generation or retraining; it does not accept the failed campaign or
establish results for the two unexecuted HTTP checkpoints.

Diagnostic summary SHA256:
`d263db1e956064b1f5c597369f5c82b95a7edc2fca55901d98a2f72e4a01557b`.
The archived declaration, executing script and source accompany it under
`runs/learning/pair-composition-batch-shape-v1`. FP32 matrix precision is
`highest`, CUDA matrix-multiplication TF32 is disabled, and model eval mode is
active. The measured forward/analysis duration is 3.1329507999980706 seconds;
this is not a serving benchmark. A future contract should compare scores to
references at the same shape while preserving exact cross-shape source/answer/
selection requirements. No tolerance or runtime code was changed here.

Post-experiment full testing found a new test-isolation defect: 683 tests passed,
but the direct evaluator's source-drift test ran after other tests had imported
Torch into the shared process. Its no-Torch guard correctly rejected first.
The test now runs the synthetic CLI scenario in a fresh isolated interpreter,
without removing modules from pytest or weakening the evaluator guard. All 23
focused tests pass after the fix; a fresh full-suite run is in progress.

Final verification: the independent batch-shape audit passes, SHA256
`6bc483713c5dd4511a3bdbc455649435f6da85a63ef8756bb6998573e90c44ee`.
It independently recomputes all saved logits/sums/selection comparisons and
validates the diagnostic identities, declaration and repeatability receipts.
Portable diagnostic: `docs/experiments/2026-09-24-pair-composition-batch-shape.json`.
The full suite now passes all 684 tests after the isolation fix, with one
existing Starlette deprecation warning. Ruff passes, 151 files are formatted,
strict mypy passes 62 source files, and documentation builds with the six
existing source-link warnings. No source/config or model-weight change was
needed to explain the score mismatch. The original campaign remains failed;
appropriate reference shapes and the remaining HTTP checks are still required.
