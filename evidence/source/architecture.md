# Architecture — the over-engineered decoder (Phase C)

Over-engineered **by design**: advanced switches have typed configuration in
[`plm.config.ModelConfig`](../src/plm/config.py). The implementation
[`plm.model.architecture`](../src/plm/model/architecture.py) is the Phase-C
reference implementation. It defaults to dense blocks, with optional activation
checkpointing and sparse-expert experiments. Other advanced switches remain
explicit hypotheses until their evaluators and implementations exist.

## Block stack

| Component | Choice | Config field |
| --- | --- | --- |
| Normalization | Pre-norm **RMSNorm** | `norm`, `norm_eps` |
| Position | **RoPE** (unscaled reference path) | `rope.*` |
| Attention | **GQA** (MHA↔MQA via `n_kv_heads`) | `n_heads`, `n_kv_heads` |
| Attn extras | **QK-norm**, sliding window, logit soft-cap | `attention.*` |
| Attn kernel | PyTorch SDPA; eager math path for deterministic mode | `attention.backend` |
| Feed-forward | **SwiGLU**, hidden rounded to `ffn_multiple_of` | `activation`, `ffn_*` |
| Sparsity | Top-k SwiGLU MoE, optional shared experts, load-balancing loss | `moe.*` |
| Tying / init | Tied embeddings; **muP** remains unsupported | `tie_embeddings`, `use_mup` |
| Numerics | Configured dtype/z-loss fields; reference weights remain explicit | `dtype`, `z_loss_weight` |
| Memory | Non-reentrant activation checkpointing; opt-in native KV cache | `gradient_checkpointing`, `eval.use_kv_cache` |

## Training loop (reference)

Config-driven AdamW · optional BF16 autocast · cosine/linear/constant/WSD
schedule with warmup · grad clip + accumulation · deterministic seeding
([`plm.reproducibility`](../src/plm/reproducibility.py)) · PyTorch checkpoints
with full provenance. Compiled execution, FP16, TensorBoard, and safetensors are
deferred rather than silently implied by the reference path.

## Honest expectation

On a small structural graph, a strong KGE baseline (ComplEx/RotatE) may match or
beat a tiny PLM on filtered MRR. That's acceptable: the PLM's value is the
protocol/compositionality/serving story. Phase C exit is "trains, resumes,
writes a provenance-bound checkpoint, and produces an honest validation result";
The first real corpus run is now recorded: dense/MoE achieved roughly 28% macro
F1 and zero exact sets on 222 validation queries. Oracle-quality parity remains
unachieved; [Lesson 4](learning/04-first-quality-campaign.md) explains the failure.

## Phase-C reference boundary

`plm.model.build_model` lazily loads the optional training dependency and builds
RMSNorm + RoPE + causal MHA/GQA + SwiGLU blocks with tied embeddings when
configured. It owns masked causal-LM loss and exposes a separate legal-entity
logit mask. muP, non-default RoPE scaling, non-SwiGLU activations, and
non-causal attention fail loudly instead of silently changing reference
semantics. Sparse MoE uses full-softmax selected probabilities, padding-free
routing, and a per-layer balancing penalty. It has no capacity dropping or fused
dispatch. Activation checkpointing preserves dropout RNG during recomputation.
RoPE coordinate pairs now share angles; mathematical invariants guard the repair.
See [the feature lesson and measured costs](learning/03-architecture-experiments.md).

Training checkpoints bind model state to the resolved config, corpus manifest,
three-way split hash, exact source archive/hash, environment, and RNG state. The
runner saves resolved config and progress/results; payload metadata is authoritative
over checkpoint sidecars. Validation reports token-weighted cross-entropy, with
router penalties and last-batch loads separate. Full response evaluation accompanies
first-token ranking and observes EOS, syntax, duplicates, and exact sets. Validation
is the adaptive-search-visible evaluation; the final test path is explicit.

Corpus labels remain input-aligned; the decoder compares logits at position t
with labels at t+1, masking prompt and padding targets. The overfit gate also
requires exact generation from a prompt, not only decreasing loss. Sequential
training resume restores the batch offset implied by optimizer steps and gradient
accumulation and isolates loader randomness from dropout. This guarantee assumes
the current deterministic corpus and no shuffling or random data augmentation.
Trainer resume rejects legacy checkpoints missing the objective/order contract.
See [the worked learning note](learning/01-training-correctness.md).

## Optional prompt objectives and output policy

`first_target_loss_weight` reweights each row's first supervised token, normalized
by effective token count. `prompt_set_loss_weight` adds query-balanced binary
set supervision from the causal ANSWER state, with a small projection and tied
product embeddings. Both default off/neutral; ordinary token CE is reported
separately. Future-token invariance, gradient direction and exact resume are tested.

The first-target-only experiment worsened generation. Prompt-set supervision
improved raw validation F1 to 62.90% in one seed. A separate deterministic
`postprocess_response` applies SAME self-exclusion, IGNORE, stable deduplication
and optional return limit before hydration, with removal counts. Replaying the
same policy on baseline/new-model responses gives 18.92%/47.30% exact answers.
These are partial quality results, not oracle parity or a server performance claim.
See [Lesson 5](learning/05-prompt-supervision.md).

[Three paired seeds](learning/06-seed-replication.md) now reproduce the improvement:
mean raw F1 is 23.13% for token-only training and 60.62% for prompt-set training;
mean processed exact accuracy is 15.62% and 45.80%. Data/split/budget are held
fixed; this does not establish robustness across datasets or oracle parity.

An additional opt-in `symmetric_relation_loss_weight` trains a normalized,
diagonal bilinear SAME scorer using shared entity/dimension embeddings and a
small learned dimension projection. It uses training answer membership only,
masks the subject column and balances positive/negative classes. Enabling this
training objective does not enable inference guidance. Defaults remain off. One seed improves
raw generation F1 to 83.41% and processed exact answers to 143/222.
[Replication](learning/12-symmetry-replication.md) found one non-terminating
response in each additional seed, so the serving promotion gate failed.
[Lesson 11](learning/11-symmetric-relations.md) covers equations and limits.

## Native incremental inference

An optional target-uniqueness rule lives in the shared generation loop, outside
the transformer. `eval.prevent_repeated_targets` masks previously emitted entity
IDs, defaults off, and travels through CLI reports and native deployment receipts.
It changes decoding choices, not model parameters. It cannot establish relation
correctness or guarantee EOS within the configured length bound.
[Lesson 13](learning/13-target-uniqueness.md) explains this experiment.

The first-target guidance experiment consumed the symmetric relation
scores during inference through an offline wrapper. It added
`alpha * logsigmoid(relation_logits)` to product logits at the prefill's last
position only. The model's existing prompt-only forward supplies the scores;
later cached steps receive no direct bias. Across three seeds, alpha 16 raises
mean processed F1 80.25% -> 88.33% and exact answers 61.11% -> 68.17%, but both
original unfinished answers remain. No strength passes the declared integration
gate on its own. See
[Lesson 15](learning/15-first-target-guidance.md) for the full grid and regressions.

The subsequent offline composition with target uniqueness passes its combined
candidate gate. Alpha 16 with uniqueness yields 666/666 valid terminated outputs,
88.40% mean processed F1 and 454/666 exact answers across the three frozen seeds.
It was selected for integration verification.
[Lesson 16](learning/16-guidance-and-uniqueness.md) records the paired interaction
experiment and identifies TYPE coverage as the remaining gap.

The shared serial and batched generation paths now implement that first-position
scoring policy through `eval.first_target_guidance_alpha`, default `0.0`.
Positive strengths require constrained cached decoding and a checkpoint with
the symmetric relation head. The helper reuses the existing prompt-only forward,
adds the FP32 log-sigmoid penalty only to product logits at the prefill's final
position, and preserves the KV cache. It skips the extra forward at zero.
The setting adds no model token or learned parameter; inference configuration
identity is recorded separately from the unchanged training identity.

Native HTTP exposes the opt-in policy with batch size one. Offline evaluation
can use batch size eight; native serving rejects that offline batch setting.
Deployment and response descriptors record the guidance policy, and the HTTP
verifier checks full token IDs and processed targets as well as metrics.
All 1,332 offline replays and 666 real HTTP answers match the complete reference
tokens and metrics; all owned servers stopped. The policy retains 454/666 exact
answers, below oracle parity. These serial correctness runs establish no SLO,
concurrency or energy advantage.
[Lesson 17](learning/17-guided-integration.md) explains this integration boundary.

The [union-coverage diagnosis](learning/18-union-coverage.md) finds that dual-TYPE
answers often enumerate only one attribute branch. Teacher-prefix probes and
prompt-only membership scores suggest testing remaining-set supervision at early
continuation states. The diagnosis itself changes no inference or model weights.

`model.continuation_set_loss_weight` now enables that training-only objective,
with default zero. Positive values require the existing contextual prompt-set
projection. The model reuses it after one, two and three teacher products to
score the remaining product set: `[B, 3, D] -> [B, 3, 1025]` for this catalog.
Balanced binary loss averages eligible stages within each query, then queries;
the subject and consumed products are negatives. Empty remaining sets are skipped,
leaving EOS to the normal causal loss. Future targets construct labels but cannot
enter earlier hidden states. No learned parameters are added, and label-free
inference does not compute the objective. The trainer records its loss and
participating-query count separately. See
[Lesson 19](learning/19-continuation-set-training.md) for the matched experiment.
The six-run coefficient-zero/one comparison raises mean F1 90.12% -> 91.52%
but reduces exact answers 466/666 -> 463/666 and dual-TYPE exact answers
34/204 -> 32/204. Its declared gate fails; the default remains zero and no
serving checkpoint is promoted.

The subsequent [matched-prefix diagnosis](learning/20-generated-versus-teacher-prefixes.md)
uses temporary hooks to observe existing normalized hidden states and the
contextual projection, without adding a model API or changing weights. All
7,980 sampled full-prefill next choices agree with archived cached generation.
Early teacher-prefix membership is strong, generated-prefix membership weaker,
and some correct prefixes still select the wrong ordered next token. Late
remaining-set head states lack explicit supervision; they are not an EOS gate.

`PLMDecoder.decode` prefills or appends unpadded chunks using request-local K/V
pairs. Cached keys are normalized/rotated once, stored at KV-head count, and
combined with absolute-position causal/sliding masks. This inference-only path
uses the existing token head, skips training-only objectives, and adds no learned
parameters or checkpoint state. The ordinary forward/training path is retained.

`generate_response(..., use_cache=True)` creates fresh state per request;
`eval.use_kv_cache` defaults false. Layer/owner/shape/device/dtype/context checks
reject incompatible state. The simple cache concatenates tensors and retains
the full prefix; padding, eviction, paging and concurrent scheduling are not
implemented by this API.

All 222 cached responses match the seed-1729 prompt-set reference exactly.
The eight-prompt paired microbenchmark showed 1.0423x summed-time ratio and
37.96 -> 35.63 MiB peak allocation: a modest local result, not a serving advantage.
See [Lesson 7](learning/07-kv-caching.md) for tensor shapes, masks and measured limits.

`generate_responses` adds fixed-group constrained KV decoding for offline
evaluation. Each row has independent completion and optional uniqueness state;
finished rows keep ignored cache slots until the group ends. The strict positive
`eval.generation_batch_size` defaults to one; larger values require caching and
constraints. CLI reports record the size and `+batch-v1` decoder tag. Native
HTTP serving rejects this offline option; there is no continuous scheduler.
Batch size eight matches all 1,332 full serial outputs across three symmetric
checkpoints and two policies. A fixed eight-query workload measures 5.31x
throughput and 36.35 -> 51.71 MiB peak allocated GPU memory. See
[Lesson 14](learning/14-batched-evaluation.md) for the experiment and limitations.

## Native application boundary

`plm serve` loads one checkpoint through the same identity-validation path as
evaluation. It copies SQLite into a dedicated deployment snapshot, freezes the
product lookup in memory, and records config/data/checkpoint/source identities.
FastAPI validates request metadata; the decoder receives only the five-token
protocol prompt. IGNORE and return limits never enter the model.

A bounded admission semaphore counts running and waiting requests. A separate
lock serializes model execution; this is not continuous batching. Each generation
uses fresh cache state. Valid complete output is parsed, filtered and hydrated;
invalid/truncated generation returns HTTP 502 with raw evidence. Invalid requests
return 422; excess admission returns 503 with Retry-After.

The actual HTTP check reproduces all 222 anchor-model raw responses and the
offline processed metrics (105/222 exact). This establishes wiring correctness,
not oracle parity or Phase D performance. See [Lesson 8](learning/08-native-serving.md).

## Configs

The latest objective interaction keeps this architecture fixed: first-target
weight eight with continuation supervision scores 464/666 exact validation
answers versus the original 466/666, and fails its declared gate. It adds no
parameters and is not promoted. See [Lesson 21](learning/21-weighting-the-first-choice.md)
for the normalized loss, tensor shapes, and training-versus-validation evidence.

The subsequent offline diagnostic keeps these weights frozen and branches on
four first products. It is not a production decoding option. Although 501/666
exact answers are available, sum/mean selection yields 462/465 versus the
original 466; both gates fail. All 666 rank-one sequences match their references.
[Lesson 22](learning/22-first-choice-paths.md) explains the score and cache
contracts, subgroup failures, and separate learned-set reranking proposal.

The completed offline reranking experiment scores those same candidate sets with
the existing symmetric relation head. It selects 501/666 exact answers (35 gains,
zero losses), reaching mean F1 .95558464 and passing the declared validation gate.
This changes neither model weights nor the current application decoder.
[Lesson 23](learning/23-ranking-candidate-sets.md) explains the canonical logit
sum and remaining errors. The next integration contract requires fresh full-token
offline and native HTTP parity before exposing the policy as an opt-in option.

That opt-in implementation now exists as `eval.symmetric_set_reranking`, default
false. Serial and batched generation share the four-path selector; native serving
applies IGNORE/limit after selection. Full offline replay matches 2,664 candidates,
666 selections and 666 disabled greedy outputs. All 666 native HTTP responses,
metadata and failure contracts also pass, with independent audit acceptance. See
[Lesson 24](learning/24-integrating-set-selection.md) for cache/tensor details and
the explicit historical-training versus new-inference configuration boundary.

A subsequent CPU-only experiment expands the four saved sets to ten slots by
adding their six pair unions and reuses the same learned score. It reaches
569/666 exact sets (68 gains, zero exact losses), mean F1 .96689605, and passes
the declared gate. These are composed sets with retained source paths, not
single autoregressively emitted sequences. [Lesson 25](learning/25-composing-candidate-sets.md)
explains that evidence. The default-off `eval.pair_set_composition` option now
adds an explicit set result and separate `/v1/predict-set` endpoint. All 661 tests
pass; all three offline checkpoints match, while the first HTTP checkpoint stops
on exact score differences with answers and paths unchanged. Full HTTP acceptance
is pending. [Lesson 26](learning/26-integrating-composed-sets.md)
explains the separate source-evidence, metric and serving contracts.

A direct membership-head ablation removes that candidate restriction from the
same saved logits. It improves macro F1 to .98958587 but reduces exact sets to
339/666 (three gains, 233 losses), failing its declared gate. It adds no runtime
option. [Lesson 28](learning/28-direct-membership-ablation.md) explains the
head's shared-embedding formula and the difference between member recovery and
whole-set correctness.

- `configs/model/tiny_decoder.yaml` — the small first run; profile before scaling.
- `configs/experiment/checkpointing_lab.yaml` — activation recomputation experiment.
- `configs/experiment/moe_lab.yaml` — four experts, top-1 routing experiment.
- `configs/model/overkill.yaml` — proposed ceiling; unsupported switches still fail.
