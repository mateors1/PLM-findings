# One extra branch outside existing source coverage

Declared before measurement, 2026-09-25. Pokemon validation only. No training,
protected-test prediction, external dataset or serving-default change.

## Hypothesis and fixed baseline

The missing-source diagnosis finds positive membership logits for 1,394/1,407
omitted true-member occurrences. This motivates testing whether a new starting
product outside existing coverage can generate useful candidate sets. It does
not prove a decoder cause or guarantee that the chosen starting product is true.

Use the accepted eight-source baseline, not the rejected intersection variant:
summary `1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183`,
audit `eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95`,
decision `70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601`.
Its saved three-seed reports contain 603/666 exact answers and macro F1
0.9834927532993669. Bind every report, their source archives, checkpoint/config/
training/corpus identities and the earlier runtime validation chain. The same
222 validation queries occur under each seed; these are 666 observations, not
666 independent queries. Diagnosis summary is
`47bf9d54967009a2be6e46bedc28f7949231be9bc14e18798fd66ad776c59805`,
audit `dc4bda47ac580f2c49d8cdfdd4e8cecc6c17d75c421ba970a1383268403ee16a`,
decision `ffd804c63618d79f1de986305bf31ec7cc9699a0bb965abbc1399af52261c422`.

## Truth-free intervention

For each query, let U be the union of the eight valid, terminated,
subject-excluded source sets. Exclude the subject from the 1,025 product columns.
Among products outside U with strictly positive saved head logit, choose the
largest logit; exact ties choose smaller token ID. This is the anchor. No
eligible product means no new candidate for that query. Labels, group identity,
true answer size, omitted-truth masks and oracle ranks never enter this rule.
Apply it to all queries, including COLOR and already-exact baseline answers.

Force the anchor as the first generated product, then greedily continue with the
archived protocol masks, unique-ID state and KV cache. Five prompt tokens and
max_new_tokens=507 retain the 512-token context bound, counting the forced first
product inside the 507. The first product is an explicit intervention, not
claimed to be the decoder's preferred choice. Later positions use ordinary
greedy decoding, with no later membership guidance or teacher forcing.

Keep all 36 baseline slots byte-equivalent and first in selection order. Append
the new original set, then its unions with old sources 1 through 8, in that
order: 45 slots for active rows, 36 otherwise. Duplicates remain. An added
original requires a valid, terminated, unique raw path; each added union also
requires its old source eligible. Existing SAME subject removal and canonical
ascending-ID math.fsum of saved FP32 head values remain unchanged. First eligible
maximum wins; no truth enters selection. Preserve invalid raw evidence and
never credit an unsuccessful fallback. Composed sets are not emitted sequences.

## Execution and replay contract

Use the same original control checkpoints, archived source/config bytes and
FP32 numerical settings as the width-eight experiment. Authenticate and extract
the archives into a new isolated runtime; never change live serving/core code
or monkeypatch archived helpers. A separate frozen experiment helper may adapt
the archived generation loop only to accept explicit first IDs. Record its
source and the archived implementation's identity.

Preserve the historical ordered groups of eight, last group six. Before new
anchor generation for each seed, recompute every prompt head at those shapes
and require exact saved-vector equality. Reconstruct all old raw memberships,
36 slots/scores/selection and baseline metrics exactly from authenticated data.
The eight historical source paths are reused, not claimed to be freshly replayed.

Also execute the new explicit-first helper using each saved rank-one first ID
as a control, at every original batch shape. Require equality of the full raw
token sequence, targets, termination, protocol validity and error against the
saved rank-one path; only its intervention descriptor may differ. This control
tests the new helper's continuation compatibility. A mismatch fails execution:
retain evidence and stop; do not relax tolerance or modify a frozen run.

For new anchor batches, keep all original rows and shapes. Inactive rows use
their saved rank-one first product as numerical padding; retain their traces
as padding evidence, exclude them from the new candidate pool and count their
compute separately. This avoids compacting active rows and silently changing
FP32 numerical shapes. No dummy row enters prediction. Record true batch work.

Record input hashes before and after, module origins, current dependencies,
GPU/device and numerical settings, frozen runner/helper/plan and test receipts.
No new neural measurement before focused tests and implementation freeze.
Only one GPU experiment runs at once; preserve the user's LM Studio service.
Any incidental service use makes timing descriptive rather than isolated.
Immutable outputs retain partial failures; never restart a live job after an
observation timeout without checking its handle.

## Measurements and fixed gate

Save all anchor decisions, logits, raw control/new/padding traces, old and new
slots, eligibility, selected sets, expected IDs and full per-query diagnostics.
Report per seed, pooled and COLOR/single-TYPE/dual-TYPE: exact answers, macro
precision/recall/F1, gains/losses, added-slot selections, exact availability,
available-exact selection misses and all-source truth containment. Separately
count active/inactive anchors, correct/false anchors, newly covered true/false
products and final-answer false positives/negatives. Labels enter these measures
only after the anchor and final candidate are selected.

The fixed acceptance gate requires all of:

1. Complete identity, exact saved-baseline reconstruction, fresh head equality
   and fresh control-path equality for all 666 observations.
2. Every active new source is valid, terminated and unique; every selected
   answer has eligible sources, with no credited fallback.
3. No seed regresses in overall exact count or macro F1.
4. Pooled exact count strictly exceeds 603 and dual-TYPE exact count strictly
   exceeds its baseline 155; each pooled group's exact count is nondecreasing.

No sweep of anchor count, score threshold, union width or subgroup routing is
authorized inside this screen. A failed gate remains failed. Unchanged scoring
and old-first ties cannot repair the two existing available-exact selection
misses; even success here is not proof of full oracle parity.

Measure synchronized fresh-control, head and new-anchor generation time,
set-scoring time, active versus padding decode work, useful tokens, peak GPU
allocation and wall time separately. Reused eight-source time is historical,
not a new end-to-end measurement. No throughput/energy/SLO/concurrency claim.

## Verification and publication

The runner is stdlib-only during preflight. Reuse pinned existing provenance
helpers where appropriate; snapshot and hash every imported research helper.
Focused tests cover subject exclusion, tied/zero/negative/all-covered anchors,
truth-free inputs, inactive padding, slot order and ties, invalid sources,
metrics and each quality regression. Synthetic CPU model tests cover forced
first ID, seen-state, EOS, bound accounting, row independence and rank-one
compatibility. Freeze tested source before actual execution.

Independent saved-output reconstruction must not import primary selection or
aggregation arithmetic. Check anchor policy, raw/processed alignment, all slots,
canonical scores, labels/metrics, baseline/head/control parity, identities and
gate. This audits saved neural results, not a second independent model run.
Keep primary acceptance false until a separate owner decision after the audit.
Update model register (27 checkpoints; new inference identity only), learning
lesson, registry, research/dev logs and PLM-findings. Supply a separate immutable
completed-experiment packet to Luna Max for GitHub publication, including a
negative result. No automatic serving promotion or weight archive is implied.
