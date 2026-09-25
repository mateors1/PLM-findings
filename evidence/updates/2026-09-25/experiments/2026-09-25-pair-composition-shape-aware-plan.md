# Pair-composition verification with references at the same batch shape

Declared 2026-09-25. This is a proposed replacement verification contract, not
an accepted replay or a change to inference. Implementation and execution require
review of this declaration. The original failed campaign remains failed.

## Why a new contract is needed

The original integration replay passed all three offline checkpoints but failed
at HTTP seed 1729 because its serial scores were compared with batch-eight
scores. Independent failure analysis found identical source paths, candidate
sets, eligibility, selected slots and answers; all 2,220 normal HTTP slot scores
differed. The maximum absolute difference, 0.00036144256591796875, is a recorded
observation, not an allowed error or proposed tolerance.

The separately declared seed-1729 diagnosis reproduced the discrepancy with
prompt-only head forwards: batch-eight/final-six logits and sums exactly matched
offline evidence, while batch-one sums exactly matched HTTP. The first eight
queries repeated exactly within each shape, and all 222 selected slots agreed.
Its independent audit passed. This explains those saved differences; it does
not establish results for the two HTTP checkpoints that never ran.

This contract changes which numerical reference is appropriate for serial HTTP.
Every score comparison at matching shape remains exact, with absolute and
relative tolerance both zero. Every non-score composition field must remain
exact across shapes. No weights, source, config, policy, pool, threshold,
rounding operation, score precision or selected-slot tie rule may change.

## Authoritative inputs

Paths below are relative to the repository. Pin these bytes before loading
archived helpers or the model; a matching filename is insufficient.

| Artifact | SHA256 |
| --- | --- |
| `runs/learning/pair-composition-integration-v1/summary.json` (failed) | `7abc3a99ea8aa0e95749da2b379893572f7f76643a53ab98f8242c2c77da7042` |
| `pair-composition-integration-v1/failure-audit.json` under `runs/learning/` | `65ffee3f6035943cf07210f371de061e460f0a554619cb9fbd56783aab1c0253` |
| `pair-composition-integration-v1/offline-1729.json` | `c7ad1db44d3c7e0387369b1ffc982d5e7b96afb046c4f8a68111b235ae5a3f67` |
| `pair-composition-integration-v1/offline-1730.json` | `932ad33ad4d0b00e29cd7d64966b3f0d5820df91764d9ba1188a22cdd84e1659` |
| `pair-composition-integration-v1/offline-1731.json` | `cd713d451a1235f7ad73d8b524acbf3c250b482f78e84c2cfd08f778e3c5ae73` |
| `pair-composition-integration-v1/http-1729.json` (failed original comparison) | `e6759659c0eec1130088c1324c8457cc2c0bf814af7773c05a8f6b251c421e87` |
| `pair-composition-integration-v1/summary.source.zip` | `1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376` |
| `pair-composition-integration-v1/summary.configs.zip` | `51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970` |
| `pair-composition-batch-shape-v1/summary.json` | `d263db1e956064b1f5c597369f5c82b95a7edc2fca55901d98a2f72e4a01557b` |
| `pair-composition-batch-shape-v1/independent-audit.json` | `6bc483713c5dd4511a3bdbc455649435f6da85a63ef8756bb6998573e90c44ee` |
| `pair-unions-v1/summary.json` | `4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992` |
| `pair-unions-v1/independent-audit.json` | `13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057` |

All shortened campaign paths in the table share the `runs/learning/` prefix.
Recursively authenticate their bound reports, original head-logit/product-column
maps, checkpoint bytes, run receipts, raw historical training configs, corpus,
vocabulary, graph snapshot, validation split, declarations, scripts, helpers,
auditors and source/config archives. Preserve the original training and earlier
integration source boundaries; do not relax an old auditor to accept new source.

Verify all 64 members and the complete member inventory of the composition source
archive against current source, `pyproject.toml` and `uv.lock`, and every config
member against current bytes. Verify the same runtime identity and numerical
settings as the audited diagnosis: Python/Torch/CUDA/GPU, FP32 parameters, model
eval mode, autocast state, `float32_matmul_precision=highest`, CUDA matmul TF32
disabled, and the recorded cuDNN/determinism settings. Record actual values;
do not silently change global settings to force agreement. Retain historical
training hashes and the already-declared named config migrations separately
from effective inference hashes. No new migration is authorized here.

Normal inference remains alpha 16, constrained unique KV generation,
`max_new_tokens=507`, symmetric reranking and pair composition enabled, four
independent source ranks and the existing policy descriptors. Offline generation
batch size is eight (final group six); native generation and reference head
evaluation are serial. Only the already-declared auxiliary disabled and bound-one
deployments may differ, each with its own complete config/health receipt.

## Execution phases and evidence reuse

Use a new immutable directory, `runs/learning/pair-composition-integration-v2/`.
Refuse preexisting final outputs or script/plan snapshots. Never edit, replace,
relabel or copy over the v1 reports, flags, mismatch lists or diagnosis.

### 1. CPU preflight and reconstruction

Authenticate all inputs and freeze the new verifier, every imported helper, this
plan and source/config identities. Independently reconstruct the three accepted
offline reports: 666 query observations, 2,664 full raw source paths, 6,660 slots
and 666 selections, plus 666 prior four-path and 666 disabled-greedy compatibility
outputs. Check exact slot sums against saved batch-eight/final-six head logits,
canonical IDs, original-first ties, source eligibility and failure rules.

Require seeds 1729/1730/1731 exactly once, each with the same 222 unique validation
queries in the authenticated order. Expected exact counts are 191/192/186 and
569 pooled; recompute macro metrics and TYPE_single/TYPE_dual/COLOR groups from
full records. These are reused offline generation observations, not new forwards.
The original summary's false acceptance flag is expected and must stay false.

### 2. Fresh serial head references for every seed

Wait for the separate **Find current PLM serving TPS** task to release the GPU
and its owned servers. Do not run this phase concurrently with a benchmark or
another model job. CPU preflight and documentation may proceed meanwhile; no
full pytest run with CUDA belongs to that overlap.

For each frozen checkpoint, obtain fresh FP32 head logits for all 222 five-token
validation prompts, one query per forward, using the same public prompt-only
model call and numerical path as the diagnosis/native scoring pass. Do not
implement a new optimized head-only formula. No generated suffix, labels or
graph relation facts enter the forward. Save all 1,025 logits, product-column
mapping, prompt IDs, query/seed identity, checkpoint/config/runtime/source hashes
and exact batch shape for every observation; reject nonfinite or non-FP32 values.

Repeat the first eight serial prompts once per seed and require exact logit
repeatability. This is 666 reference forwards plus 24 repeat forwards, not a new
autoregressive campaign. For seed 1729, also require exact equality to all saved
serial logits from the audited diagnosis. Do not extrapolate that diagnosis to
the other two checkpoints: their references are freshly computed here.

Before any fresh HTTP checking, seal all three reference files and their hashes.
Independently compute ten slot scores from each reference using the authenticated
offline slot sets, ascending product-ID `math.fsum`, and saved eligibility. Apply
the unchanged earliest-slot tie rule and all-invalid rank-one fallback. Every
serial-reference winner, selected set and non-score slot field must equal its
offline counterpart. Any changed winner fails this contract even if set quality
is unchanged. Cross-shape logit/score differences are retained as diagnostics,
not tested against a tolerance or used to change the offline reference.

### 3. Revalidate saved HTTP seed 1729

Reinspect all 222 original HTTP responses against the newly sealed serial
reference. Require exact scores from independent sums and exact non-score
composition fields against offline evidence. Revalidate hydration, request/query
alignment, identity descriptors and every saved metadata/legacy/invalid/admission/
disabled/bound-one failure/shutdown receipt. Filtered requests use the same full
query reference, never a score recomputed on the filtered answer. Independently
reconstruct auxiliary failure slots from their saved raw sources and serial
prompt logits under the original failure-set convention.

Write a new validation receipt pointing to the unchanged HTTP report and stating
`evidence_origin=reused`, original campaign/report hash, original failed gate,
and the new reference hashes. Do not set its old `exact_parity` flag to true or
describe this as a fresh HTTP checkpoint. Missing or inconsistent auxiliary
receipts fail reuse rather than being inferred from the normal-response count.

### 4. Fresh HTTP seeds 1730 and 1731

Run the unchanged native server sequentially for the two remaining checkpoints,
with 222 real `/v1/predict-set` requests per seed. Compare every response to the
sealed reference for that seed and query. Require all four full source token
sequences, decoded targets, termination/protocol/error fields, all ten canonical
sets/source ranks/eligibility flags, selected slot/kind/ranks/set, fallback flag,
policy and identities to match offline exactly. Independently recompute each
native slot's sum from its member IDs and the serial logits; require exact
numeric equality and independently validate selection. Do not derive the
reference scores from the HTTP response itself. Timing fields are observations,
not parity targets; allow only the documented transport schema and reject any
fabricated composed-answer raw/EOS/termination/sequence-probability fields.

For each new seed, repeat the full original auxiliary coverage: IGNORE/limit
after selection on a pair-winning query (including zero limit), canonical
hydration, legacy `/v1/predict` compatibility, invalid requests returning 422,
and shared-admission rejection returning 503 on both endpoints. Preserve full
unfiltered slots, source paths and selected set during metadata checks. Create
separately receipted disabled and bound-one deployments: require disabled 409
before generation and real all-ineligible 502 with raw failure evidence and no
successful result. Derive short-bound score expectations from serial prompt
logits and the actual independently validated failure-slot sets; never award
quality credit to an incomplete answer. Do not mutate a receipted deployment.

Use owned-server `finally` shutdown for every normal and auxiliary deployment.
Verify listener/thread shutdown before any next model job. Retain partial rows,
mismatches and shutdown evidence even on exceptions. A stop failure forbids
starting another GPU job. Do not retry into the same immutable report or discard
a first mismatch; any follow-up requires a separately identified diagnostic.

## Independent audit, acceptance and explicit accounting

The audit independently parses saved logits/IDs and performs `math.fsum`, ties,
eligibility, set arithmetic, subgroup metrics and hydration comparisons without
importing the verifier's scoring/selection/metric helpers. It need not rerun the
model; state that limitation. Authenticate every report and the final verifier
summary, and rehash all inputs, live source/config and executed scripts before
completion. Preserve a manifest of reused versus newly produced evidence.

Only after every check and this independent audit passes may a new overall
receipt accept integration under this contract. Required accounting is:

| Evidence | Reused observations | Fresh observations |
| --- | ---: | ---: |
| Offline composition queries | 666 | 0 |
| Offline raw source paths / slots | 2,664 / 6,660 | 0 / 0 |
| Four-path / disabled-greedy compatibility queries | 666 / 666 | 0 / 0 |
| Serial head-reference queries | 0 | 666, plus 24 repeat forwards |
| Normal HTTP queries | 222 (seed 1729) | 444 (seeds 1730/1731) |
| HTTP auxiliary checkpoint suites | 1 | 2 |

The union must cover 666 HTTP query-seed observations, 2,664 source paths and
6,660 slots, with 666 exact matching-shape score/selection checks and all normal
and auxiliary shutdown receipts. Report distinct reference scores and checked
slot-score counts (6,660 normal HTTP scores); do not confuse query counts with
slot counts. Preserve actual metadata/invalid/admission/failure counts by seed.
Acceptance is **mixed fresh and reused evidence**, not three fresh HTTP runs,
and not a retroactive pass for v1. Separately label the unchanged offline quality
result, original failure, diagnostic success and this eventual acceptance.

## Required tests and named failure conditions

Before execution, focused CPU tests must demonstrate:

- A changed hash, archive inventory, checkpoint, query partition, config value,
  numerical setting or source file prevents reuse/launch (`identity_mismatch`).
- Duplicate, missing, wrong-seed or misordered queries and swapped product
  columns fail (`coverage_mismatch`); incomplete prerequisites cannot pass.
- Nonfinite/wrong-precision reference values or failed within-shape repeats fail
  (`invalid_serial_reference`); seed-1729 diagnostic drift also fails.
- A one-ULP score change at the same shape fails (`same_shape_score_mismatch`),
  while a fixture with different cross-shape sums passes only when each agrees
  exactly with its own reference and every other field agrees.
- Changed raw tokens, errors, canonical sets, ranks, eligibility, policy or
  selected slot fail (`cross_shape_output_mismatch`), including equal-quality
  answers and two slots containing an identical set.
- Altered canonical sums, duplicate-member counting, tie priority or fallback
  fail (`selection_arithmetic_mismatch`); no expected-answer labels enter these
  calculations.
- Missing metadata/failure evidence, wrong status, hydration or schema fails
  (`transport_contract_mismatch`). A score-only projection cannot hide extra
  synthetic sequence fields or an altered source record.
- Exceptions preserve partial reports (`incomplete_execution`); failed shutdown
  yields `owned_server_shutdown_failure` and prevents the next job.
- Reused reports cannot be counted as fresh, and any failed prerequisite or
  independent audit prevents final acceptance (`evidence_accounting_failure`).

Tests must not loosen the original verifier or its frozen artifacts. Archive
new harness/tests results separately; no change to inference is needed for this
contract. Failure remains failure: do not introduce tolerances, change batch
sizes, retune scores or switch references after seeing a mismatch.

This accepts only the opt-in application behavior at the stated checkpoints,
shapes and environment. It does not establish cross-device bitwise portability,
oracle parity, protected-test quality, SLOs, throughput, energy advantage or a
default deployment. The 97 validation errors and the wider project goal remain
open. Timing from this replay is diagnostic; controlled serving measurements
belong to their separate declared task.
