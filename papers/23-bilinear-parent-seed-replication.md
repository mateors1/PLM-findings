# Replicating the bilinear recipe across trained parents

**Status: independently audited evidence accepted; both fresh-seed gates passed.**
Seed 1730 reaches 212/222 exact validation sets and seed 1731 reaches 208/222,
exceeding their own historical eight-branch comparators of 205 and 197.
Both retain the required F1 and group floors. The two fresh fits produce
420/444 exact sets; including the reused selection seed gives 627/666.
The campaign supports the fixed recipe across these learned parents, while
24 fresh model-query outputs remain incorrect. No serving policy is promoted.

The [frozen plan](../evidence/updates/2026-09-25-bilinear-seed-replication/experiments/2026-09-25-bilinear-seed-replication-plan.md),
[portable result](../evidence/updates/2026-09-25-bilinear-seed-replication/experiments/2026-09-25-bilinear-seed-replication.json),
[independent audit](../evidence/updates/2026-09-25-bilinear-seed-replication/campaign/independent-audit.json)
and [owner decision](../evidence/updates/2026-09-25-bilinear-seed-replication/campaign/decision.json)
separate the scientific outcome from evidence acceptance.

The [fixed 2,000-update experiment](22-bilinear-training-budget.md) selected a
promising recipe on seed 1729. This replication asks whether that unchanged
recipe also improves two existing independently trained parents, seeds 1730
and 1731. A training seed changes initialization and the stochastic training
trajectory; it does not create a new evaluation dataset. All parents share the
same graph, split and 222 validation queries. The data-split seed remains 1729.

## Same procedure, different learned coordinates

For each parent k, freeze its original 93 tensors and attach a new zero matrix
A^(k)[2,256,256]. Each model's full token embedding table has shape [2049,256].
The scorer selects its 1,025 product rows and constructs normalized, rescaled
features E^(k)[1025,256], with E_i=16*w_i/||w_i|| for each selected row w_i. Their
coordinates were learned independently: a direction meaningful in one table
need not mean the same thing in another. Therefore each residual is fitted
freshly to its own parent's coordinates; no trained A is transplanted.

\[
M_d^{(k)}=\frac{A_d^{(k)}+A_d^{(k)\top}}{2},\qquad
z_{sdi}^{(k)}=z_{sdi}^{\mathrm{parent},(k)}+
\frac{E_s^{(k)\top}M_d^{(k)}E_i^{(k)}}{16}.
\]

With A=0, the extra term is exactly zero. Each run verifies this against its
own parent's validation logits and freshly computed full-training balanced
binary cross-entropy loss. Different parents can have different starting
losses; forcing them to match seed 1729's scalar would test the wrong invariant.

Both fresh fits use 1,637 training queries per full batch, producing
Z[1637,1025]. AdamW updates only A for 2,000 steps at learning rate 0.0003,
zero weight decay and the frozen FP32 operation order. Both optimizers start
fresh. Only the final child is evaluated; intermediate validation does not
select a stopping point. Each output uses the fixed z>0 nonself rule and
ascending product identifiers, without an oracle answer count or repair.

## Each seed must clear its own baseline

Seed 1730 must exceed its matching historical eight-branch selector's 205 exact
sets, keep macro F1 at least 0.9906969338820507, and preserve group floors
COLOR=103, single TYPE=45 and dual TYPE=57. Seed 1731 must exceed 197 exact sets,
keep F1 at least 0.9798558083418223, and preserve floors 103, 44 and 50. Both also
need all execution invariants and 222 outputs with 1..506 products.

The campaign gate is the logical AND of the two fresh-seed gates. An average
improvement cannot compensate for a failing seed or group. This contract was
frozen before execution. The historical seed-1729 pass remains separately
accepted and is reused for context, not counted as a third new replication.

## Results against each matching predictor

| Seed and role | Dense parent exact | Own width-eight exact | Residual exact | Residual macro F1 | Strict separation |
|---|---:|---:|---:|---:|---:|
| 1729, historical selection | 112 | 201 | 207 | 0.9997456354 | 217 |
| 1730, fresh replication | 116 | 205 | 212 | 0.9997931270 | 219 |
| 1731, fresh replication | 111 | 197 | 208 | 0.9996688121 | 216 |

Every row evaluates 222 queries. The fresh residuals gain 96 and 97 exact answers
against their dense parents, losing none. Against the stronger eight-branch
selectors, seed 1730 gains 12 and loses five; seed 1731 gains 18 and loses seven.
Thus a higher total does not mean every previously correct query stays correct.
The original gate protects group counts, not per-query dominance.

| Fresh seed | COLOR exact / floor | Single TYPE exact / floor | Dual TYPE exact / floor |
|---|---:|---:|---:|
| 1730 | 103 / 103 | 51 / 45 | 58 / 57 |
| 1731 | 103 / 103 | 51 / 44 | 54 / 50 |

There are 103 COLOR, 51 single-TYPE and 68 dual-TYPE queries per seed. Every fresh
COLOR and single-TYPE answer is exact; the 10 and 14 remaining errors are all
dual-TYPE answers. Seed 1730 has eight extra and five missing member occurrences;
seed 1731 has 12 extra and ten missing. These are member counts, not query counts.

Fresh-two exactness is 420/444, versus 402/444 for their own selectors, and macro
F1 is 0.9997309695. Reusing historical 1729 yields 627/666, versus 603/666, and
macro F1 0.9997358581. Across all three, 39 complete answers are still wrong.
High F1 rewards mostly correct membership in large sets; exactness requires the
entire set to match. Neither result reaches the deterministic compiler oracle.

The fresh starting losses are 0.003967047668993473 and 0.004869009833782911.
Their final losses are 0.000034819207940017805 and 0.000043502172047737986.
These decreases concern the training objective. The gate uses separate validation
measurements. Full trajectories are preserved as two individually hash-bound
[seed-1730](../evidence/updates/2026-09-25-bilinear-seed-replication/experiments/2026-09-25-bilinear-seed-replication-training-1730.json)
and [seed-1731](../evidence/updates/2026-09-25-bilinear-seed-replication/experiments/2026-09-25-bilinear-seed-replication-training-1731.json)
artifacts, with the exact original training receipts retained too.

## Evidence repairs did not rerun the experiment

Both GPU processes completed successfully. Audit and aggregation encountered
two evidence-processing failures. The original aggregate compared a UTC launch timestamp with a
PowerShell-serialized local-offset timestamp as literal strings. They represented
the same instant, but the text differed. Separately, the first auditor expected
original-parent training metadata inside run.json although that metadata lived
in its authenticated checkpoint sidecar. The original aggregate failure and
first failed audit remain preserved; neither is silently overwritten.

A separately declared [evidence repair](../evidence/updates/2026-09-25-bilinear-seed-replication/experiments/2026-09-25-bilinear-replication-evidence-repair.md)
adds an aggregate-only runner and auditor v2. Timestamp comparison uses exact
UTC instants, retaining all fractional digits; equivalent offsets pass while a
changed instant or missing timezone fails. The auditor reads the authenticated
original-parent schema. No checkpoint, prediction, metric, threshold, gate or
training budget changes, and neither GPU fit is repeated.

The frozen seed runner passed 136 focused CPU tests. The recovery runner passed
58 CPU tests (13 new, 45 reused), including timestamp precision, immutable output
and an AST comparison showing its aggregate body preserves the original logic
except explicit helper qualification and timestamp equality. Auditor v2 passed
82 synthetic tests before execution, then both per-seed and aggregate audits
completed. The new aggregation source hash is distinct from the seed runner,
scorer and evaluator identities.

The independent audit reconstructs saved-set metrics and checks CPU checkpoint
payloads. Full-training loss/logit equality, reload and process chronology remain
authenticated execution receipts, not independently regenerated neural results.
This distinction matters: repairing a verifier can make valid evidence readable,
but cannot manufacture a new scientific replication.

## What the sample means

The two fresh models supply 444 model-query observations over 222 shared
queries. Adding the historical selection seed yields 666 observations over
the same questions. This is not 666 independent held-out examples. Training
seed replication tests dependence on learned parent state; it does not undo
adaptive validation use or establish performance on unseen questions.

Every completed child receives a full 94-state checkpoint, including the
unchanged 93 parent tensors. Parent pretraining steps and residual fitting
updates remain separate lineage fields. The two new children increase the
final-checkpoint inventory from 31 to 33 after completed audits;
reusing seed 1729 creates no checkpoint. Stored local hashes do not establish
a durable remote weight archive.


The approximately 20.89- and 15.41-second primary wall times are descriptive local
measurements, not controlled speedups or evidence of concurrent serving or energy
savings. The ordinary inherited decoder forward still ignores the added A;
these children require the explicit dense experiment scorer. Protected-test
predictions, serving integration and broader generalization remain unestablished.

The [evidence inventory](../evidence/updates/2026-09-25-bilinear-seed-replication/README.md)
retains code, declarations, successful and failed receipts, final lineage and
verification boundaries. Full weights, membership labels and head vectors remain
hash-bound local dependencies. The checkpoint inventory freshly verifies all 33
final weight files; it does not promise a durable remote weight archive.
