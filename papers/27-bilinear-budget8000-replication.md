# Replicating the 8000-update budget across parent seeds

Status: completed, independently audited and separately accepted. Both fresh
quality gates pass. The two derivatives are registered in the 37-checkpoint
inventory; no default-model change or serving promotion follows.

Increasing the balanced-BCE residual budget from 2000 to 8000 updates improved
both fresh parent seeds under their own fixed quality gates. Seed 1730 improved
from 212 to 220 exact answers out of 222, and seed 1731 from 208 to 214. Neither
lost a previously exact answer against its matched 2000-update child. This
supports the budget result across these two parents; it does not establish
oracle parity, a general scaling law or a serving improvement.

## Question and controlled comparison

The earlier [seed-1729 budget screen](26-bilinear-budget8000.md) produced 216/222
exact sets, up from 207/222. That result motivated this replication. Seed 1729
is therefore historical development-selected evidence, not a third fresh run.
The two fresh controls are the accepted seed-matched children from the
[2000-update replication](23-bilinear-parent-seed-replication.md).

Each fresh fit starts again from its original parent, a zero residual and fresh
optimizer state. It does not resume the old fitted residual. Parent pretraining
and residual optimization have distinct step counts. The experiment holds the
scorer, balanced objective, optimizer settings, data split, strict zero decision
threshold and final evaluator rules fixed while increasing the residual budget.
All work remains within Pokémon TYPE/COLOR SAME.

## What is being trained

Think of the parent as a fixed map of Pokémon representations. The learned
residual changes how two representations interact for each steered dimension.
The full token embedding table has shape [2049,256]. Its 1025 product rows become
features E[1025,256] after row normalization and rescaling:

\[
E_i=16w_i/\|w_i\|_2,\qquad A\in\mathbb{R}^{2\times256\times256},
\qquad S_d=(A_d+A_d^\top)/2.
\]

For subject s, dimension d and candidate product i, the scorer is

\[
z_{sdi}=z^{\mathrm{parent}}_{sdi}+E_s^\top S_d E_i/16.
\]

Only A is trained. All 93 original state tensors stay frozen; each completed
derivative contains 94 state tensors. A stores 131072 scalar parameters, while
symmetrization makes only its symmetric component affect the scores. The
inherited ordinary decoder does not apply this residual scorer automatically.

For each query, let T be the true nonself members and N the false nonself
members. Balanced binary cross-entropy gives those two classes equal total
weight, regardless of their sizes:

\[
\ell_q=\frac{1}{2|T|}\sum_{i\in T}\operatorname{softplus}(-z_i)
       +\frac{1}{2|N|}\sum_{i\in N}\operatorname{softplus}(z_i),
\qquad L=\frac{1}{1637}\sum_q\ell_q.
\]

The complete training score matrix has shape [1637,1025]. Each fit performs
exactly 8000 full-batch FP32 AdamW updates. Validation has 222 queries and runs
at zero for replay and at update 8000 for final quality. There is no intermediate
fitted-checkpoint selection. The 191 protected queries are not evaluated.
Output membership remains z>0, excluding self, serialized in ascending entity
ID order. No threshold search, cardinality oracle or output repair is used.

## Fixed gates and observed results

Seed 1730 must exceed 212 exact sets, retain macro F1 at least
0.9997931269673187 and preserve COLOR/single-TYPE/dual-TYPE exact counts
103/51/58. Seed 1731 must exceed 208, retain macro F1 at least
0.999668812076336 and preserve 103/51/54. Both fresh seeds must pass individually;
pooling cannot rescue a failure. Execution, replay, lineage and serialization
checks apply in addition to these quality floors.

| Parent seed / role | Matched 2000 exact | 8000 exact | Macro F1 | FP / FN | Strict separation | Gains / losses vs own 2000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1730, fresh | 212/222 | 220/222 | 0.9999717006519728 | 0 / 2 | 222/222 | 8 / 0 |
| 1731, fresh | 208/222 | 214/222 | 0.9998584743646364 | 7 / 2 | 221/222 | 6 / 0 |
| 1729, historical selected | 207/222 | 216/222 | 0.9998843626799785 | 1 / 8 | 220/222 | 9 / 0 |

Both fresh seeds pass all declared primary quality checks, as independently
verified. Their fresh total is 434/444 exact observations versus 420/444 for
their own 2000-update controls: 14 paired gains and zero losses. Macro F1 is
0.9999150875083046, with 7 false positives, 4 false negatives and 443/444 strictly
separated observations. Each fresh child serializes all 222 queries correctly.

| Scope | COLOR exact | Single-TYPE exact | Dual-TYPE exact |
| --- | ---: | ---: | ---: |
| Seed 1730 | 103/103 | 51/51 | 66/68 |
| Seed 1731 | 103/103 | 51/51 | 60/68 |
| Fresh two | 206/206 | 102/102 | 126/136 |
| All three, including selected 1729 | 309/309 | 153/153 | 188/204 |

All remaining nonexact observations are dual-TYPE. Adding the historical result
gives 650/666 exact, macro F1 0.9999048458988625, FP/FN 8/12 and strict separation
663/666. The corresponding three 2000-update children had 627/666 exact, so the
descriptive comparison is 23 gains with no losses. These totals reuse the same
222 validation questions: 444 and 666 count model-query observations, not new
independent questions. No significance or held-out generalization claim follows.

The comparison against each parent's older width-8 decoder is distinct from
the budget comparison: seed 1730 has 16 gains and 1 loss; seed 1731 has 20 gains
and 3 losses. The zero-loss statement above applies specifically to the matched
2000-update residual controls.

## Training trajectory and what it does not prove

| Seed | Initial training BCE | Final post-update BCE | Old post-2000 = new pre-2001 |
| --- | ---: | ---: | ---: |
| 1730 | 0.003967047668993473 | 0.0000015859151289987494 | 0.000034819207940017805 |
| 1731 | 0.004869009833782911 | 0.0000019348922251083422 | 0.000043502172047737986 |

For both fresh seeds, every one of the first 2000 pre-update loss scalars equals
its historical counterpart, with zero mismatches; the separately aligned
boundary values above also match. Initial zero-residual scores and losses
reproduce the corresponding parent and its own historical initial loss. The
prefix comparison is descriptive, not a gate, selection rule or retry trigger.
Equality of scalar losses does not prove equality of intermediate parameters
or deterministic execution across hardware.

Exact-set accuracy is demanding: one wrong membership makes the whole query
nonexact. Strict separation asks a different question, whether the weakest true
score exceeds the strongest false score. Seed 1730 has strict separation on all
222 queries but only 220 exact sets at the fixed zero threshold. Correct ranking
can coexist with a misplaced absolute threshold. This observation does not
authorize a threshold change or establish a single threshold that fixes every
query. Training BCE, F1, separation and exactness measure different properties.

## Verifier failure and versioned repair

Both GPU training processes completed once, in seed order 1730 then 1731.
After seed 1730 completed, the original independent verifier failed before
checkpoint loading: a provenance assertion passed Python sets into a helper
that serializes its arguments as JSON. The resulting TypeError was an evidence
processing defect, not a detected scientific mismatch. Seed 1731 waited for the
repaired seed-1730 audit; seed 1730 was not retrained.

The original verifier, its 131-test receipt and its failed stdout/execution
receipt remain intact. A separately versioned v2 verifier repairs dependency
inventory comparison; 148 synthetic tests cover actual old 17/4-entry and new
6-entry receipt inventories, invalid inventories and the repair boundary.
Nineteen scientific functions remain AST-identical to the frozen original.
The original aggregator was never executed. Its separate v2 replacement binds
the v2 audit paths and repair provenance, retaining scientific arithmetic,
gates and exact-instant chronology checks. Its 46 tests include the original
34 tests. The primary runner retains its original 221-test receipt and bytes.

The v2 seed audits and aggregate audit completed with observed exit code zero.
They recompute evidence checks from saved results and authenticate checkpoint
state; they do not repeat CUDA training. Terminal receipts, source/test hashes,
the original failure and the repair declaration remain separate evidence layers.
The separate owner decision accepts the evidence and both fixed quality gates,
while explicitly preserving the current default and serving policy.

## Lineage, publication and limits

The two new checkpoint identities are
`b3e47911015a992c3c805a5643816ee262987f5c7707fa7596f284ec03aa7c27`
(1730) and
`d22a510e7a0571a0f86164f8d1dacda46f95594b9bcf8154699d04beb51bd567`
(1731). They have separate original-parent lineages. Their scorer architecture
and balanced-BCE objective are unchanged; the new replication evaluator is
`plm-bilinear-budget8000-replication-v1`. A verifier repair changes evidence
processing identity, not the trained model's recipe or quality threshold.

The evidence packet preserves each complete 8000-entry training history through
four exact 2000-entry chunks, eight chunks total. Source hashes and exact
concatenation checks bind them to the two local raw training files. Raw weights,
full score reports and oversized raw traces remain hash-bound local dependencies;
their hashes do not establish durable remote archival. Publication does not
promote the scorer to serving or change the retained default model.

These are adaptive validation findings on deterministically computable relations.
The compiler remains the perfect oracle. This campaign neither reaches oracle
parity nor evaluates protected data, throughput, concurrency, latency, energy,
external relations or a general retrieval application. It supplies a replicated
local improvement for a fixed residual budget and preserves the remaining errors.

## Evidence

The [packet index](../evidence/updates/2026-09-25-bilinear-budget8000-replication/README.md)
links the copied source, receipts and eight trace chunks. The
[aggregate](../evidence/updates/2026-09-25-bilinear-budget8000-replication/campaign/aggregate-v2.json),
[independent audit](../evidence/updates/2026-09-25-bilinear-budget8000-replication/campaign/independent-audit-v2.json)
and [owner decision](../evidence/updates/2026-09-25-bilinear-budget8000-replication/campaign/decision.json)
remain distinct records. The
[repair declaration](../evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication-evidence-repair.md)
preserves the verifier failure and authorized repair scope. The
[portable result](../evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication.json)
references exact trace chunks; the
[37-checkpoint inventory](../evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication-model-version-inventory.json)
records fresh final-file hashes without claiming all prior payloads were newly
revalidated. [Lesson 48](../evidence/updates/2026-09-25-bilinear-budget8000-replication/learning/48-testing-a-budget-across-parent-seeds.md)
provides the learning-oriented explanation.
