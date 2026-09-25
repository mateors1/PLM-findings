# Integrate pair composition as an explicit set result

Implementation contract reviewed on 2026-09-24. The interfaces below describe
the next opt-in implementation, not completed behavior or integration evidence.
The existing authorization to advance the project covers isolated CPU staging;
GPU execution and incorporation follow the prerequisite below.

## Prerequisite and measured reference

Prerequisite completed before the new replay: the preceding campaign and its
independent audit passed all three checkpoints. Their reviewed SHA256 values are
`780f2e219d55164acefaa3b29a6b25e803154baa057d64a5002f05733a4d99de`
and `96cce95af7cfc5001a1b94ed5df34d30eeb729ae04bd7c2141cba53b47e1c3ad`.
After incorporation, all 661 tests pass with CUDA available; Ruff/format and
strict mypy over 62 source files pass. Preflight authenticates 138 inputs. These
checks authorize the declared replay, not a claim that the replay has passed.

Finish the existing four-path integration campaign, including all native HTTP
checks and shutdown of its owned servers, before changing source or config in
the active checkout used by that campaign. CPU-only implementation may proceed
in an isolated checkout copied from the exact frozen working tree, including
its uncommitted files. Verify imports resolve there, disable CUDA for its tests,
and keep its changes separate until the preceding campaign passes its audit.
That campaign is still running when this draft is written. Its partial progress
must not be described as completed integration evidence. Preserve its eventual
reports, scripts, source archive and any mismatch evidence unchanged.

The saved-candidate pair experiment has independently passed its declared gate:
569/666 exact sets, 68 exact gains and zero exact losses against the four-set
selector's 501/666. Macro F1 is 0.9668960529174291. Seed 1729/1730/1731 exact
counts are 191/192/186; single TYPE/dual TYPE/COLOR totals are 136/125/308.
There are 576 original selections and 90 pair compositions. The ten-slot oracle
availability is 570, a separate diagnostic ceiling, not achieved quality.

Authoritative evidence:

- `runs/learning/pair-unions-v1/summary.json`, SHA256
  `4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992`.
- Its independent audit, SHA256
  `13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057`.
- The preceding four-set summary, SHA256
  `c515fae06f7db1810e8697badf59d53ffb41675cbb8fc543d1a14feee7252c5c`.

Authenticate the summaries and every referenced per-seed report, original raw
path, saved score/logit, executed script/helper and source receipt. This step
reproduces an existing policy; it introduces no training, additional path width,
score calibration, threshold, group-specific rule or protected-test use.

## Option and shared result boundary

Add `eval.pair_set_composition: bool = False`. For this first integration, require
`eval.symmetric_set_reranking=True` when composition is enabled. This explicitly
retains the existing four-path policy as the prerequisite and legacy endpoint
behavior. The other requirements remain constrained decoding, KV caching,
unique source products and a trained symmetric relation head. Verify positive
head training configuration against loaded checkpoint metadata and a completed
training step, not merely a caller-supplied serving flag. Record guidance alpha;
the quality and parity reference here is alpha 16.

Keep `GenerationResult`, `generate_response` and `generate_responses` as
autoregressive-response interfaces. Do not make their return types conditional
on the new flag or convert a composed set into a `GenerationResult`.

Introduce a separate shared composition API/result,
`generate_set_compositions` and `SetCompositionResult`, for offline and native
callers. The result contains:

- Four original `GenerationResult` source paths in immutable rank order.
- Ten candidate slots with slot index, original/pair kind, source ranks,
  canonical product-ID set, source eligibility and canonical logit-sum score.
- Selected slot, selected kind, selected source ranks, selected set IDs,
  selected-source eligibility and explicit all-ineligible fallback status.

These field names describe the reviewed contract; record public names in the registry.
There is no synthetic raw sequence, EOS, sequence log probability, termination
flag or protocol-valid flag for a composition. Only the source paths have those
properties. Even when an original slot wins, the set representation is canonical
and its original emitted order remains separately available in its source path.

Reuse the existing four independent rank batches and canonical scoring logic.
The four original paths must be generated once per query group, with fresh
branch caches and seen-product state. Reuse prompt-only FP32 membership logits
for original and pair scores; do not obtain scores from a generated suffix or
teacher labels. Preserve batch-eight grouping offline and serial groups natively.
Do not add a second four-path generation pass merely to compute the baseline.

## Fixed pool, ordering and failure behavior

Preserve exactly these slots: original ranks 1,2,3,4, followed by pairs
(1,2),(1,3),(1,4),(2,3),(2,4),(3,4). Form unions from the original valid,
terminated, unique paths' deterministic SAME-processed sets. A pair requires
both sources to be eligible. Retain all ten provenance slots even when sets
coincide; duplicate member IDs contribute once.

Score each set by `math.fsum` of its saved-equivalent FP32 symmetric logits in
ascending product-ID order. Choose maximum score, breaking ties by the declared
slot order. Selection accepts only learned scores, product IDs and source
eligibility. No expected targets/counts, graph relation facts or diagnostic
subgroups enter candidate construction or selection.

Serialize selected set members in ascending product-ID order. This is stable
set serialization, not a claim about the model's emitted sequence. Keep original
source sequences byte-for-byte as generated. An original source retains the
inclusive completion bound, including its first product and EOS; composition
does not gain an invented completion budget or termination event.

If no source is eligible, preserve rank-one raw failure evidence and mark the
composition unsuccessful. Do not return a hydrated successful set, manufacture
EOS, or award valid-answer credit because incomplete targets match the oracle.
Keep raw overlap diagnostics separate from successful-set metrics.

`IGNORE` and return limits apply only after selecting the full canonical set.
They must not change slot scores, selected slot, source paths or the unfiltered
selected set. Hydrate the final filtered IDs from the existing frozen catalog.

## CLI and HTTP contracts

Normal CLI evaluation with composition disabled preserves its existing report
schema and generation metrics. With composition enabled, use an explicitly
versioned set-evaluation report containing `set_generation` and `set_responses`.
Do not put composition results into existing raw/processed generation fields.

`set_generation` reports query count, macro precision/recall/F1, exact-set
accuracy/count, mean set size, selected-source eligibility rate, failure count,
and original/pair selection counts. Ineligible fallbacks receive zero valid
precision/recall/F1/exact credit. Mean set size uses zero for failed answers and all queries
in its denominator; original/pair selection counts include successes only, with
failures counted separately. Incomplete member counts remain raw-source diagnostics.
Report source-path termination/protocol/error
statistics separately with names beginning `source_path_`; omit composed
exact-sequence accuracy and composed termination/protocol rates. Per-query
records retain all ten slots, four raw sources and the selected canonical set.
Any four-path baseline metrics in this report have an explicit baseline label.
Update consumers explicitly for the new opt-in report version.

Use a separate native endpoint, `POST /v1/predict-set`, accepting the existing
subject/dimension/SAME/IGNORE/limit request semantics. Enable it only for a
deployment configured for pair composition. A disabled call returns an explicit
409 `set_composition_disabled` response before model execution; health metadata
advertises the capability. The 409 status expresses a deployment capability conflict.

Successful set responses contain hydrated `result`, the unfiltered selected
canonical set, selected-slot/kind/source-rank evidence, the ten score/eligibility
slots, original raw source paths and checkpoint/data/policy identities. Put raw
evidence under `source_paths`, never under a synthetic `raw` answer. Postprocess
receipts describe removals from the selected set, not removals from an invented
sequence. Client-facing presentation can continue to show hydrated products.

Keep `/v1/predict` and its existing raw-response contract intact. It continues to
use the four-path single-answer selector when that existing option is enabled,
including in a deployment that also enables `/v1/predict-set`. It never silently
returns a composed union. Both endpoints share the same bounded admission and
serial model-execution lock; do not bypass it for scoring or branch generation.

Retain invalid-request 422 and admission 503 behavior. An all-ineligible set
request returns 502 with a distinct set-source-failure code, selected rank-one
fallback provenance and the original source failures, without a success result
or synthetic raw answer. Receipts distinguish the two endpoint policies.

## Config migration and old/new identities

The original neural training archive remains
`101d661a16e3f17ecc924c1afed3a188cea25fe8e70bb1df2a19b523f363ecd1`.
The previous four-path integration source archive is
`e66ae65f1156f697eeb06352c360e87bf3974a70575fe7bce6b5ac016c0f070c`;
its existence is not by itself proof that its HTTP campaign passed. This new
integration intentionally receives another inference source/config identity.
Model architecture, checkpoint bytes and dependency lock remain unchanged.

Extend the explicit saved-config migration by exactly the newly declared missing
`eval.pair_set_composition=False` field. Preserve the already-declared migration
for absent `eval.symmetric_set_reranking=False`. Authenticate canonical raw JSON
against its historical hash before either insertion. Accept only these named
missing defaults; existing values and every unrelated field must compare exactly
after parsing. Reject unrelated defaults, coercions or normalization. Test both
pre-four-path and four-path-era saved configurations.

Record historical training hash, any applied named default insertions, effective
inference config/hash, old/new source archives, model/checkpoint/data/split/runtime
identities and the precise set-composition policy. For changed YAML defaults,
prove that removing only the declared new false field reproduces the immediately
preceding frozen bytes. Never overwrite old reports or teach old auditors to
accept arbitrary new source. Capture new scripts/helpers and source immutably.

## Verification and acceptance

Begin with CPU tests for fixed ten-slot algebra, overlapping/identical sets,
canonical FP32 sums, original-first ties, truth-independent selection, invalid
source and fallback handling, config migration and trained-head validation.
Test immutable raw source evidence and a transport schema that cannot label a
union as emitted. Verify that both endpoints share admission/serialization and
that disabled behavior and IGNORE/limit separation hold.

After the prerequisite campaign finishes, use the unchanged three checkpoints
and all 222 validation queries per seed for fresh generation. Require:

1. Offline batch-eight equality for all 2,664 original full raw paths, all 6,660
   slot sets/scores/eligibility/source-rank records and all 666 selected slots and
   canonical sets against the pair experiment. Require exact canonical scores
   initially; any numerical discrepancy is retained for diagnosis, not silently
   tolerated or used to change the selected set.
2. With the new flag disabled, replay all 666 existing four-path selected raw
   outputs, descriptors/errors and metrics against the accepted preceding
   integration. Also preserve the ordinary disabled-reranking greedy path;
   replay its 666 full-token reference outputs as part of compatibility checks.
3. Temporary native `/v1/predict-set` HTTP replay for all 666 validation queries:
   exact selected canonical sets, source ranks, selected-source full raw tokens
   including prompt/EOS, eligibility/error fields and hydrated outputs. Retain
   all four raw source paths and ten slot receipts for independent inspection.
   A serial-versus-batch mismatch is a failure requiring diagnosis, not a reason
   to compare only aggregate F1.
4. Explicit native legacy-endpoint compatibility and new-endpoint disabled,
   invalid, all-ineligible and filtered-request checks. Include an independently
   receipted short-bound deployment for actual failed generation. Never mutate
   the main deployment's settings after writing its receipt. Close owned servers
   in `finally` paths and verify they stopped before the next GPU job.

Recompute set metrics and groups independently. Expected exact totals remain
191/192/186 and 569/666; compare per-query results, not only these totals. Record
wall time and peak memory with the complete candidate-generation cost included;
separate diagnostics from any later controlled performance benchmark. GPU jobs
remain sequential, while CPU audit and documentation may proceed in parallel.

Passing accepts an opt-in set-composition application path. It does not complete
the project goal: 97/666 observations remain inexact, the deterministic graph
oracle is still perfect, protected final evaluation remains unopened, and
matched-quality latency/throughput/energy and concurrency evidence remain open.
Do not claim oracle parity, a quality win over the compiler, SLO achievement or
default deployment from this integration.
