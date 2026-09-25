# Fixed eight-first-choice candidate search, v1

Declared 2026-09-25 before measurement. This is an offline inference-policy
experiment on the accepted composition checkpoint family. No training or
protected-test evaluation. It does not promote the rejected weak-margin models.

## Question and fixed reference

Can additional first-choice branches supply useful complete sets that the
unchanged learned scorer selects? The saved weak-margin diagnosis found no
available-exact selection misses in its two pools. That motivates changing
candidate generation, without proving this specific intervention will work.

Use the original accepted three control checkpoints, not either weak-margin
checkpoint. All 222 validation queries for each seed are required. There are
222 distinct queries and 666 query-seed observations, not 666 independent queries.

| Seed | Run | Checkpoint SHA256 |
| --- | --- | --- |
| 1729 | national_dex_continuation_control_s1729_v1 | e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1 |
| 1730 | national_dex_continuation_control_s1730_v1 | 5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2 |
| 1731 | national_dex_continuation_control_s1731_v1 | ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d |

Authenticate the accepted shape-aware integration chain in
`runs/learning/pair-composition-integration-v2/`:

- summary SHA256 `77f9d9bba8eb9c5e0cf5df86f2eb29f522cf00ad50600ee9eea76267ec041c8f`;
- independent audit `40eba8d9dac72dbe762eb4b54e7c053128b08580f905dd7539a89961c144ebde`;
- acceptance `1d5835709646a9dbf7f8db2727e3adabef2c451ff75be932e2eb22833acdd9d4`.

Bind their cited offline reports, same-shape historical head-vector reports,
query order, corpus/vocabulary/protocol/split identity and training receipts.
The historical baseline is 569/666 selected exact sets, macro F1
0.9668960529174291. Exact availability among its ten slots is 570/666.
Fresh baseline reproduction, rather than these aggregate numbers alone, is
required. Split SHA256 is
`b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d`.

## Isolated executed source and numerical contract

Extract the authenticated archived source/config bytes from the accepted campaign
to a new campaign-local runtime directory. Source archive SHA256:
`1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376`;
configs archive SHA256:
`51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970`.
Reject unsafe archive paths, duplicate members and overwrite attempts. Import
archived `plm` only after isolation; verify loaded package modules originate
there. Do not monkeypatch helpers or edit current core/config files. Record the
current dependency/device environment separately from the archived source.

Load the pinned weights through the archived validated runtime using the saved
offline configuration. Use identical FP32 inference/head values, no autocast,
the recorded backend settings and same groups of eight (last group six).
Explicitly check the historical numerical settings before generation. Bind
checkpoint/config/receipt and corpus bytes before use and rehash afterward.
The ordinary runtime may validate the full corpus and split metadata; only the
validation partition may receive predictions or metrics. No test tuning.

## One intervention, fixed before seeing results

For every validation query generate first-product ranks 1 through 8 using the
archived `_generate_rank` implementation. Rank selection happens only at the
first position; all later tokens follow unchanged greedy decoding. Preserve
alpha=16 guidance, protocol masks, uniqueness, independent per-rank KV caches,
and max_new_tokens=507 (five prompt tokens, 512 total context). Stable descending
first-logit order breaks ties by entity ID. No beam pruning, sampling, width
sweep, subgroup routing, teacher forcing or oracle repair.

For each seed, freshly generate ranks 1-4 and the prompt-only head at the exact
historical query batch shapes. Require exact equality of full raw paths,
postprocessed sets, eligibility, head vectors, original ten slot scores and
choices against authenticated references before generating ranks 5-8 for that
seed. A mismatch is a failed execution gate; preserve it and stop without
tolerance relaxation. Prior seeds' completed evidence remains retained.

Preserve the legacy ten slots first: originals 1-4 followed by pairs (1,2),
(1,3),(1,4),(2,3),(2,4),(3,4). Append originals 5-8, then the remaining 22 pairs
in lexicographic rank order. There are exactly 36 slots. Retain duplicates and
slot provenance; unchanged legacy ties cannot be displaced by equal new scores.

Use the existing SAME postprocessor: remove subject only from protocol-valid,
terminated source sets, checking raw uniqueness before removal. An original is
eligible only if its raw source is valid, terminated and unique; a pair requires
both sources eligible. Preserve raw failure membership. Score each sorted distinct
set with `math.fsum` of the same saved FP32 head entries. The first eligible
maximum wins. All-ineligible fallback preserves rank one but gets no answer
credit. Expected sets and group labels enter diagnostics only after selection.
Composed sets are not invented emitted sequences or assigned fake EOS evidence.

## Measurements and fixed quality gate

Save every raw source path, head vector (1025 product columns), all 36 sets,
scores and flags, both baseline/new selections, expected IDs and query identities.
Report per-seed and pooled macro precision/recall/F1 and exact counts; COLOR,
single-TYPE and dual-TYPE groups; paired exact gains/losses and their queries;
original/pair/added-slot selection counts. Use the historical set-metric arithmetic
so the old baseline reproduces exactly. Averages use `math.fsum` over query values.

Separately report exact availability, available-exact selection misses, all-source
truth containment, and number of distinct eligible sets per query for widths
four and eight. Availability is an oracle diagnostic ceiling, never a substitute
for learned selected quality. The wider pool can introduce higher-scoring wrong
sets and regress despite improved availability.

All required quality conditions are:

1. Exact baseline replay and all identity/source/mask/fallback invariants pass.
2. Every new source path is valid, terminated and unique; every selected answer
   has eligible sources, with no unsuccessful fallback credited.
3. Each seed's overall exact count and macro F1 do not regress.
4. Pooled exact count strictly exceeds 569, and pooled dual-TYPE exact count
   strictly improves over its baseline.
5. Each pooled group (COLOR, single-TYPE, dual-TYPE) has nondecreasing exact count.

Strict head separation is not a success condition: weights are unchanged and
wider search cannot improve that fixed head property. Do not import or relax
the earlier training campaign's gate. No automatic additional width after failure.

Record synchronized generation time for ranks 1-4 versus 5-8, separate head and
set-scoring time, per-rank decode steps/padded row work and useful emitted tokens,
peak allocated GPU memory and total wall time. Rank batches execute sequentially,
so eight simultaneous branch caches are unnecessary. Expected full coverage is
5328 source paths and 23976 candidate slots. This is a quality/work study, not
a serving throughput, energy, SLO or concurrency benchmark; no speedup claim.

## Verification, versioning and acceptance

The new runner is stdlib-only during preflight, with lazy isolated runtime imports.
Archive its exact source, plan, input hashes, runtime archives and focused-test
receipts. Refuse overwrites; preserve partial/failure output, release model/GPU
resources and retain completed seed reports. No background job may be duplicated
after a timeout without checking its live handle. Only one GPU campaign at a time.

Focused tests cover 36-slot order, duplicate sets/ties, mask/fallback handling,
raw-subject removal, legacy replay corruption, truth-free selection, availability
and quality-gate regressions. Independent saved-output reconstruction must not
import primary selection/aggregation helpers. It checks all raw-path semantics,
slots, scores, selected metrics, query alignment, identity chain and gate.
Exact rational sums may cross-check canonical floating scores. This audits saved
neural outputs without claiming a second independent model execution.

Primary acceptance remains false pending independent audit and a separate owner
decision. A quality pass only permits later consideration of serving integration;
it does not change defaults or claim oracle parity unless exactness actually
reaches 666/666. This adds an inference experiment identity, not a model recipe
or checkpoint version. Preserve the 27-checkpoint inventory and prior failures.
