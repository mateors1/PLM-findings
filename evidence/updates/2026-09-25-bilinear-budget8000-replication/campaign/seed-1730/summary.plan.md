# Fixed 8000-update balanced-BCE replication across two parent seeds

Declared before new neural execution. Campaign `bilinear-budget8000-replication-v1`,
evaluator `plm-bilinear-budget8000-replication-v1`. Pokemon TYPE/COLOR SAME only.
The previous completed screen is evidence of progress, not completion of oracle
parity, protected evaluation, serving integration or durable weight archival.

## Question, budget and chronology

Does the seed-1729 improvement from 2000 to 8000 residual updates repeat on
original parent seeds 1730 and 1731? Use the accepted 1729 result as historical
development/selection evidence, never as a fresh replication. Execute exactly
two fresh fits, seed 1730 then 1731, in separate sequential GPU processes.
Execute the second even if the first completes with a failed quality gate.
Stop and preserve evidence on execution/invariant failure. An incomplete
campaign is not a completed two-seed replication. No result-driven retry,
additional budget, seed replacement, intermediate fitted-child validation,
threshold selection or automatic promotion.

Restart each original parent with exact-zero A and fresh AdamW. Do not resume
any 2000-update residual derivative. Parent pretraining stays 2000 steps and
the new residual training is exactly 8000 updates. LM Studio remains offline.

Inherit the numerical/scoring/data contracts of the fixed-8000 plan
`2026-09-25-bilinear-budget8000-plan.md`, SHA
`064eaa115e673caaf89196bd391005a42441b6fb54cd49f9d67a6ab9c3aaab6f`,
and the seed/replay/chronology contracts of the previous replication plan
`2026-09-25-bilinear-seed-replication-plan.md`, SHA
`d5a36d48eac3c40a30a5c29c2126f3d4f815f5ecb4d93a03f8cc47f771e75b54`.
This plan overrides their campaign/evaluator, budgets, fixed initial-loss rule,
historical references and quality-comparator floors as explicitly stated below.

## Historical development evidence and matching controls

The accepted 1729 8000-update screen achieved 216/222 exact. Authenticate:

- Summary `e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8`.
- Audit `bdf142f48229fbcdb197bce3a019e38c3a65570ae3f8551a9c9a1f14b2198872`.
- Decision `2cadf9dce7ee36cc4095a10a746cc1608d5ce4437724872c1eec1c88e915c195`.
- Checkpoint `6979013750eb8f2780917f714192850aae411a48490757236a891fea7e351f91`.
- Runner `cd593c14c429247837c26b09754509d5cfd1dd36adb317f56522a80a78a25122`.
- Recipe `f976442315433e32d5c88ce2f4e911509ec5ea56542fc2a247893cefd0ddfefc`.

The fresh seeds' matched historical comparators are their accepted 2000-update
children, authenticated through `bilinear-seed-replication-v1`:

- `aggregate-v2.json`: `338fb6b7ea08128952ce9abc02b3506639e712a9cff6ce63468ed2e7980f79a0`.
- Independent audit: `ddbecaceabdacc0808f9b4f122e0309aa3e233fe76b0d75694e31833a40083a8`.
- Owner decision: `62a8dcbb7cbe5d038984f9d57aecacec4b0ba612ba55c2a9f6fe335977304978`.

Preserve its original failed audit/aggregation attempts. Use the accepted v2
chain; do not treat failed `summary.json` as the accepted aggregate.

| Seed | Exact count to strictly exceed | Macro F1 floor | COLOR / single TYPE / dual TYPE floors |
| --- | ---: | ---: | --- |
| 1730 | 212 | 0.9997931269673187 | 103 / 51 / 58 |
| 1731 | 208 | 0.999668812076336 | 103 / 51 / 54 |

Original parent 1730:
`5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2`;
config `0eae5252ae9de2d1463992a319c43e695ae8bb4f24bfb2ae94acdf4da6033b31`.
Original parent 1731:
`ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d`;
config `9e029d5fa280492a382b3b138eb5933a8ef8580c6a95a90deb0c87420c3601b3`.
Verify checkpoint bytes, run, sidecar and training-result identities agree.

Matched historical seed-summary hashes:

- 1730: `ecc49edfa4a7b7b2428326d752a020cf09ef92eefcb69157b12a3c4a340356e1`.
- 1731: `3f99d91b87f4531941ae3af11b3f39a42746df15f86df461db7e06386ecefc9c`.

Matched complete historical training traces:

- 1730: `a7366f60a7164d8e31d1dccfb4484364cb68919f8e2449c7871138359f2d4fe5`.
- 1731: `e7f7d4fd2cbfecc7d819b8d691c3a49c9fcd2c46e8d8bf80cf46833d61ad3512`.

Bind corresponding report/checkpoint hashes through these accepted manifests.
Width-eight and original dense-parent comparisons remain descriptive secondary
comparisons, using the already authenticated seed mappings in the prior plan.

## Frozen computation and data

Architecture remains `plm-frozen-symmetric-bilinear-residual-v1`, objective
`plm-bilinear-residual-balanced-bce-v1`, with unchanged scorer/fitting helper
`5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee` and balanced
BCE helper `79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c`.
Use the explicit balanced `_prompt_set_loss` callback to the frozen fitting loop.
No worst-member loss, hidden mixture or monkeypatching of frozen modules.

Use archived source ZIP `1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376`
and config ZIP `51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970`.
Split hash is `b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d`:
1637 train, 222 validation, 191 protected. Split seed stays 1729 for all parents.
Train membership must reproduce
`0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad`.
No new labels, examples, split, external dataset or validation supervision.

All 93 original state tensors stay frozen; only zero-initialized FP32
A[2,256,256] trains. Full-batch logits remain [1637,1025]. Per-seed execution RNG
uses that parent seed. AdamW: lr 0.0003, betas (0.9,0.999), eps 1e-8, decay 0,
nonfused, fresh optimizer; no scheduler, clipping or accumulation. FP32, no
autocast, highest matmul precision, CUDA matmul TF32 false, cuDNN TF32 true,
deterministic algorithms false. Prediction stays z>0, nonself, ascending IDs.

At A=0, reproduce every matching original parent validation score exactly at
B8/final B6. Compute that parent's full-training BCE from its frozen head and
require equality of actual zero-A full-training logits and loss. Different
parents need not reproduce seed 1729's initial scalar. Require agreement with
each matching historical initial training loss as an authenticated replay check.
Validation points are [0,8000], with zero used only for parent/zero-A replay.

Retain all 8000 pre-update loss/finite-score/gradient/parameter records and a
final post-update loss. Compare each first-2000 scalar prefix to its own old
trace; record mismatch count and first mismatch. Separately compare old final
post-2000 with new pre-2001. These diagnostics cannot abort, retune or change
acceptance; scalar equality is not proof of intermediate parameter equality.

Save each final full 94-state checkpoint with fresh optimizer state and residual
global_step 8000. Verify exact parameter and optimizer reload, all original
tensors unchanged and A changed. Parent pretraining 2000, residual 8000, seed,
new evaluator, recipe, data and runner/scorer identities must be unambiguous in
checkpoint config/identity/metadata. The inherited decoder still ignores A.

## Selection and reporting

Require each fresh seed to exceed its own exact-count comparator and meet its
own F1/group floors above. Also require every execution, train-only, replay,
frozen-state, reload and evidence invariant, plus all 222 outputs containing
1..506 distinct nonself products. Retain raw empty/oversized outputs in evidence;
no threshold adjustment, fallback, truncation, oracle cardinality or routing.

Both fresh seeds must pass. A pooled improvement cannot rescue a failing seed,
metric or group; an exactness tie fails the strict improvement requirement.
Aggregate acceptance additionally requires the authenticated accepted 1729
8000-update chain, both fresh independent evidence audits and an independent
audit of aggregation. Primary summaries remain provisional until owner decision.

Report per-seed aggregate/group exactness, F1, precision/recall, FP/FN, strict
separation, serialization and paired gains/losses versus each matching 2000
child, width-eight control and original dense parent. Require identical query,
prompt, labels, group and product-column order. Report fresh-two-seed totals
separately from all-three totals, labeling 1729 historical development-selected.
444 and 666 model-query observations reuse 222 queries, not independent sample
sizes. No significance, unseen-data or protected-test claim follows.

## Tests, audit, aggregation and publication

Freeze new plan, recipe, runner, aggregator and synthetic tests before neural
execution, with observed test receipts binding actual bytes and dependencies.
Own new seed-aware 8000-step validators; do not call incompatible old execution
or checkpoint validators. Reuse compatible hash-authenticated pure helpers.
Coordinate primary/auditor schemas before freezing, including identity/config,
optimizer steps, prefix diagnostics and artifact references.

Synthetic tests must cover both seed mappings and recipe differences; exact/F1/
group gate boundaries; balanced-loss callback and inherited masks/gradients;
parent-zero replay; frozen state; 8000 optimizer/reload identity; descriptive
prefix mismatch behavior; immutable failure outputs; historical report alignment;
rejected/incomplete historical chains; missing/failed fresh audits; and aggregate
metrics/chronology. Equivalent timezone-aware timestamp representations must
compare as instants, not strings; reject naive timestamps and reversed/overlapping
execution. Preserve full fractional precision when comparing receipt instants.

Root records each launch and observed terminal completion, binding seed, summary,
stdout and chronological predecessor. Poll the same live handle until terminal;
do not restart on observation timeout. The independent CPU auditor authenticates
saved full traces, sets, metrics, all checkpoint tensors/optimizer/metadata and
the gate without using primary prediction/gate arithmetic or rerunning CUDA.
Audit both fresh results before aggregation. Aggregation binds their exact
summaries, audits and execution receipts, the accepted historical chain and
the chronology. Audit the completed aggregate before a separate owner decision.

Failed execution/audit artifacts remain immutable; any repair needs a separate
versioned identity. No neural retry is authorized by an evidence repair. A
quality failure is a completed, publishable negative result after audit.

Register completed derivatives after the current 35-checkpoint inventory,
including rejected ones; freshly verify all final checkpoint bytes. Preserve
all 8000 trace records in four exact 2000-row portable chunks per fresh seed,
with original source hash, ordered ranges and exact concatenation verification;
each published file must be <=512 KiB. Checkpoint hashes prove identity, not backup.

Update registry, research/dev logs, model register, teaching notes and
PLM-findings. Keep checkpoint, objective, inference, evaluator and publication
identities separate. Send the completed immutable handoff immediately to the
existing Luna Max publisher. No protected evaluation or serving/default change
is part of this campaign. A pass motivates the next declared research step,
not automatic promotion or a claim that the broader project is complete.
