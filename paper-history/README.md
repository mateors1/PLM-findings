# Canonical paper history and evidence index

The repository root [`PAPER.md`](../PAPER.md) is the current canonical
scientific synthesis. This index keeps paper revisions distinct from model,
checkpoint, inference, evaluation, and publication identities. Historical
evidence stays at its existing immutable repository path; this index links to
it and does not move or rewrite source files.

## Paper revisions

| Revision | Date | Predecessor | Status | Preserved copy |
| --- | --- | --- | --- | --- |
| `PAPER-v0001` | 2026-09-25 (America/Bogota) | None; initial revision | Superseded by `PAPER-v0002`; exact archived SHA256 `d8746165b57acd5d40613109f47945a1265c5aa0e5d916b2b0a6e49c0bacfa39` | [PAPER-v0001-2026-09-25.md](PAPER-v0001-2026-09-25.md) |
| `PAPER-v0002` | 2026-09-26 (America/Bogota) | `PAPER-v0001` | Superseded by `PAPER-v0003`; exact archived SHA256 `27b22b90e4c90b3b284bc773025dc9443fddbf56c8ade914ec272e18fd10d29d` | [PAPER-v0002-2026-09-26.md](PAPER-v0002-2026-09-26.md) |
| `PAPER-v0003` | 2026-10-02 (America/Bogota) | `PAPER-v0002` | Superseded by `PAPER-v0004`; exact archived SHA256 `9de795bcbea79cd6b12130fe2b25d3776e6a9983abfe5e16c44c9080c5c0e391` | [PAPER-v0003-2026-10-02.md](PAPER-v0003-2026-10-02.md) |
| `PAPER-v0004` | 2026-10-02 (America/Bogota) | `PAPER-v0003` | Superseded by `PAPER-v0005`; exact archived SHA256 `18a32906d040c840c13d336076c55648e0d3608b572e4c5e58b81d3353c2d2eb` | [PAPER-v0004-2026-10-02.md](PAPER-v0004-2026-10-02.md) |
| `PAPER-v0005` | 2026-10-02 (America/Bogota) | `PAPER-v0004` | Superseded by `PAPER-v0006`; exact archived SHA256 `12096631f48cec8b58156c9dc0e762dda6d8aeee781c224f0f0fe9b3f3ce83fb` | [PAPER-v0005-2026-10-02.md](PAPER-v0005-2026-10-02.md) |
| `PAPER-v0006` | 2026-10-02 (America/Bogota) | `PAPER-v0005` | Superseded by `PAPER-v0007`; exact archived SHA256 `54e1c804289a0b8c17141d604786ff9839db89a3b02eca163034a1ccb1ce8d84` | [PAPER-v0006-2026-10-02.md](PAPER-v0006-2026-10-02.md) |
| `PAPER-v0007` | 2026-10-02 (America/Bogota) | `PAPER-v0006` | Superseded by `PAPER-v0008`; exact archived SHA256 `d5612ec895746e27c6e22b03420c086e21cf223ecf033545cb78cbab8ea8478d` | [PAPER-v0007-2026-10-02.md](PAPER-v0007-2026-10-02.md) |
| `PAPER-v0008` | 2026-10-02 (America/Bogota) | `PAPER-v0007` | Superseded by `PAPER-v0009`; exact archived SHA256 `87981f0f68d164cc420a0b9e5fb071d243f560343775a42caffcdf76c0b320a8` | [PAPER-v0008-2026-10-02.md](PAPER-v0008-2026-10-02.md) |
| `PAPER-v0009` | 2026-10-18 (America/Bogota) | `PAPER-v0008` | Superseded by `PAPER-v0010`; exact archived SHA256 `5b91d9f164119953ede546539657e512c4660e42100df89afe21b1403b817a60` | [PAPER-v0009-2026-10-18.md](PAPER-v0009-2026-10-18.md) |
| `PAPER-v0010` | 2026-10-18 (America/Bogota) | `PAPER-v0009` | Current; adds accepted reverse-query supervision diagnostic and query-holdout limitation | Root [`PAPER.md`](../PAPER.md) |

Before replacing the current revision, copy its exact bytes to a unique,
versioned file in this directory and verify the archived checksum. Never
overwrite an archived revision. Each later root revision must carry its own
version, date, predecessor, and evidence cutoff.

## Retained model and evidence map

| Topic | Status in `PAPER-v0010` | Source evidence |
| --- | --- | --- |
| Pokémon-only protocol, hidden graph attributes, `TYPE/COLOR SAME` | Retained scope; the graph compiler is the exact v1 oracle | [`protocol-spec.md`](../docs/protocol-spec.md), [`data/evaluation`](../docs/learning/02-data-and-evaluation.md), [`research_log.md`](../research_log.md) |
| Prompt-set supervision | Retained in the reference checkpoint recipe; three-seed validation gain, not generalization | [`seed replication report`](../docs/experiments/2026-09-24-seed-replication.json), [`Lesson 6`](../docs/learning/06-seed-replication.md) |
| Symmetric product-membership head and four-path selector | Implemented and used by the application-reference experiments; selection remains opt-in | [`symmetric relation lesson`](../docs/learning/11-symmetric-relations.md), [`set-selection integration`](../docs/learning/24-integrating-set-selection.md), [`model register`](../docs/model-versions.md) |
| Pairwise unions of candidate sets | Accepted opt-in policy; 569/666 exact on the three-seed fixed validation split; no oracle parity | [`Lesson 25`](../docs/learning/25-composing-candidate-sets.md), [`shape-aware v2 receipt`](../docs/experiments/2026-09-25-pair-composition-shape-aware.json), [`Lesson 29`](../docs/learning/29-shape-aware-verification.md) |
| Bilinear 8,000-update residual replication | Accepted validation candidate; not retained or supported by ordinary serving | [`replication plan`](../docs/experiments/2026-09-25-bilinear-budget8000-replication-plan.md), [`portable result`](../docs/experiments/2026-09-25-bilinear-budget8000-replication.json), [`audit repair`](../docs/experiments/2026-09-25-bilinear-budget8000-replication-evidence-repair.md), [`Lesson 48`](../docs/learning/48-testing-a-budget-across-parent-seeds.md), [`model register`](../docs/model-versions.md) |
| Bilinear 32,000-update learning candidate | Best accepted validation finding at 660/666 versus 650/666 for the 8,000-update candidate; six residual errors are dual-`TYPE`. It remains unpromoted and outside ordinary decoder inference; the repeated seed-query outcomes use the same 222 validation questions. | [`model register`](../docs/model-versions.md) |
| Dense-set CPU serving integration | Accepted integration of the same three 32,000-update checkpoints; 1,332 score-row comparisons and 666 in-process HTTP requests preserve the existing 660/666 selections. It adds no quality gain, GPU/network capacity, or serving advantage. | [`integration report`](../docs/experiments/2026-10-02-dense-serving-integration-report.md), [`Lesson 53`](../docs/learning/53-from-learned-scores-to-a-serving-contract.md), [`readiness`](../runs/learning/dense-serving-preparation-v1/readiness-v1/readiness.json), [`summary`](../runs/learning/dense-set-serving-integration-v1/summary.json), [`audit`](../runs/learning/dense-set-serving-integration-v1/independent-audit.json), [`decision`](../runs/learning/dense-set-serving-integration-v1/decision.json) |
| Uncached dense scorer direct-call cost | Audited negative performance evidence for the warm CPU B1 complete-set path: median paired dense/SQLite ratios 12.77–13.08 across three seeds; dense selections remain 218/222, 222/222, and 220/222. A separate warm answer-cache lookup is prepared-answer reuse, not live query computation. This does not establish HTTP, concurrency, energy, or a general speed limit. | [`direct-call report`](../docs/experiments/2026-10-02-dense-oracle-direct-report.md), [`Lesson 54`](../docs/learning/54-what-a-latency-number-measures.md), [`summary`](../runs/learning/dense-oracle-direct-v1/summary.json), [`audit`](../runs/learning/dense-oracle-direct-v1/independent-audit.json), [`read-only review`](../runs/learning/dense-oracle-direct-v1/independent-review.json), [`decision`](../runs/learning/dense-oracle-direct-v1/decision.json) |
| Dense feature-cache candidate | Accepted experimental policy; passed exact-score and predeclared latency gates, using 6.37–6.64% of same-run unchanged scorer time. It remains 1.135–1.190× live SQLite and preserves the six learned errors. The later cached-serving integration adds the optional application path without changing model quality. Both baseline arms were slower than in the previous run for an unmeasured reason, so performance claims remain within-run paired. | [`feature-cache report`](../docs/experiments/2026-10-02-dense-feature-cache-report.md), [`Lesson 55`](../docs/learning/55-caching-features-without-caching-answers.md), [`summary`](../runs/learning/dense-feature-cache-v1/summary.json), [`audit`](../runs/learning/dense-feature-cache-v1/independent-audit.json), [`review`](../runs/learning/dense-feature-cache-v1/independent-review.json), [`decision`](../runs/learning/dense-feature-cache-v1/decision.json) |
| Cached dense serving integration | Accepted optional authenticated CPU manifest integration; direct scores and in-process HTTP responses preserve the existing 218/222, 222/222, and 220/222 results. This integration alone makes no performance claim. The evidence review did not replay weights or graph computation. | [`integration report`](../docs/experiments/2026-10-02-cached-serving-integration-report.md), [`readiness`](../runs/learning/cached-serving-preparation-v1/readiness-v1/readiness.json), [`summary`](../runs/learning/cached-dense-serving-integration-v1/summary.json), [`audit`](../runs/learning/cached-dense-serving-integration-v1/independent-audit.json), [`review`](../runs/learning/cached-dense-serving-integration-v1/independent-review.json), [`decision`](../runs/learning/cached-dense-serving-integration-v1/decision.json) |
| Cached HTTP concurrency | Accepted instrumented Windows loopback comparison against W32 SQLite: SQLite leads at C1/C2; the cached adapter passes the paired throughput/p95 gate at C4/C8/C16. Both meet the p95 limit through C16, a lower bound; six seed-query errors across five queries remain. This is not a matched-quality result. | [`report`](../docs/experiments/2026-10-02-cached-http-concurrency-report.md), [`summary`](../runs/learning/cached-http-concurrency-v1/summary.json), [`audit`](../runs/learning/cached-http-concurrency-v1/independent-audit.json), [`review`](../runs/learning/cached-http-concurrency-v1/independent-review.json), [`decision`](../runs/learning/cached-http-concurrency-v1/decision.json) |
| SQLite worker control | W1 passes the fixed gate over W32 at C4/C8/C16; C1/C2 are mixed. The limit applies to the whole synchronous endpoint, so the result supports a scheduling contribution but does not identify an internal cause. This qualifies the earlier cached-vs-W32 result; separate campaigns cannot rank cached inference against W1. No global optimum, maximum capacity, or energy claim. | [`report`](../docs/experiments/2026-10-02-sqlite-worker-control-report.md), [`plan`](../docs/experiments/2026-10-02-sqlite-worker-control-plan.md), [`readiness`](../runs/learning/sqlite-worker-preparation-v1/readiness-v1/readiness.json), [`summary`](../runs/learning/sqlite-worker-control-v1/summary.json), [`audit`](../runs/learning/sqlite-worker-control-v1/independent-audit.json), [`review`](../runs/learning/sqlite-worker-control-v1/independent-review.json), [`decision`](../runs/learning/sqlite-worker-control-v1/decision.json) |
| Cached HTTP against selected SQLite W1 | Accepted fresh comparison: SQLite passes all nine throughput/p95 pairs at C2/C4/C8/C16; C1 mixed; no cached advantage. Both meet 100 ms p95 through C16, a lower bound. All 81,252 requests and traces were checked; unchanged cached quality is 218/222, 222/222, and 220/222. No model promotion, energy result, or globally optimal worker count. | [`report`](../docs/experiments/2026-10-18-cached-http-w1-report.md), [`portable evidence`](../docs/experiments/2026-10-18-cached-http-w1-portable-evidence.json), [`plan`](../docs/experiments/2026-10-18-cached-http-w1-plan.md), [`readiness`](../runs/learning/cached-http-w1-preparation-v1/readiness-v1/readiness.json), [`summary`](../runs/learning/cached-http-w1-v1/summary.json), [`audit`](../runs/learning/cached-http-w1-v1/independent-audit.json), [`review`](../runs/learning/cached-http-w1-v1/independent-review.json), [`decision`](../runs/learning/cached-http-w1-v1/decision.json) |
| Native PLM versus SQLite oracle | Contrary, nonmatched-quality serial serving result; graph exactness and speed lead | [`portable report`](../docs/experiments/2026-09-25-native-graph-oracle.json), [`plan`](../docs/experiments/2026-09-25-native-graph-oracle-plan.md) |
| Graph-only concurrent CPU saturation | Added graph capacity baseline: 954.2 servings/s at one client and 435.4–460.9/s at concurrency 8–128; concurrent package medians 104.1–110.9 W. The user-specified +120 W is arithmetic, not measured wall power; no matched concurrent PLM result | [`report`](../docs/experiments/2026-09-26-graph-cpu-saturation-offset120-report.md), [`portable result`](../docs/experiments/2026-09-26-graph-cpu-saturation-offset120.json), [`audit`](../docs/experiments/2026-09-26-graph-cpu-saturation-offset120-audit.json), [`decision`](../docs/experiments/2026-09-26-graph-cpu-saturation-offset120-decision.json), [`plan`](../docs/experiments/2026-09-26-graph-cpu-saturation-offset120-plan.md) |

## Superseded methods, negative results, and contrary evidence

The canonical paper keeps only short explanations relevant to the current
model. These original records preserve the longer reasoning, failed gates, and
superseded implementation details:

| Historical method or result | Why it is not presented as the retained model | Preserved rationale and evidence |
| --- | --- | --- |
| Initial dense/MoE and first-target-only trials | Weak exact-set quality or no retained quality benefit | [`architecture experiments`](../docs/learning/03-architecture-experiments.md), [`first quality campaign`](../docs/learning/04-first-quality-campaign.md), [`first-target guidance`](../docs/learning/15-first-target-guidance.md) |
| Prompt-set and symmetric-head development | Earlier recipes are provenance for the current recipe, not independent data or a parity result | [`prompt supervision`](../docs/learning/05-prompt-supervision.md), [`symmetric-relation replication`](../docs/learning/12-symmetry-replication.md) |
| Wider search, source intersections, coverage-seeking, direct-membership ablation | Changed candidate availability or set quality unevenly; some fixed gates failed | [`wider first-choice search`](../docs/learning/35-wider-first-choice-search.md), [`set algebra`](../docs/learning/36-unions-intersections-and-recoverability.md), [`coverage branch`](../docs/learning/38-a-branch-for-uncovered-products.md), [`direct-membership ablation`](../docs/learning/28-direct-membership-ablation.md) |
| Margin, projection-only, worst-boundary, and short bilinear fits | Completed negative or mixed screens; not promoted | [`hardest-boundary margin`](../docs/learning/31-hardest-boundary-margin.md), [`weak-margin result`](../docs/learning/33-loss-weight-is-not-gradient-strength.md), [`projection refit`](../docs/learning/39-refitting-a-frozen-feature-head.md), [`worst-boundary projection`](../docs/learning/40-training-the-weakest-membership-decisions.md), [`bilinear refit`](../docs/learning/42-learning-cross-coordinate-relations.md), [`bilinear worst-boundary`](../docs/learning/46-changing-which-mistakes-drive-learning.md) |
| Single-seed longer balanced-BCE fit and two-parent replication | Positive validation evidence for an experiment-local residual scorer; it does not alter normal decoder forward, standard serving, or promotion state | [`single-seed screen`](../docs/experiments/2026-09-25-bilinear-budget8000.json), [`Lesson 47`](../docs/learning/47-more-updates-with-the-same-objective.md), [`replication result and repair`](../docs/experiments/2026-09-25-bilinear-budget8000-replication.json), [`Lesson 48`](../docs/learning/48-testing-a-budget-across-parent-seeds.md) |
| Verifier failure during bilinear replication | The first auditor failed before checkpoint loading; versioned repairs audited the saved artifacts without retraining | [`evidence-repair declaration`](../docs/experiments/2026-09-25-bilinear-budget8000-replication-evidence-repair.md), original and v2 receipts in the replication report |
| Cached HTTP advantage against W32 SQLite | Preserved historical measurement, not evidence of an advantage over the selected W1 baseline. The fresh W1 comparison supports SQLite at C2/C4/C8/C16 and no cached advantage; no cross-campaign ratios are used. | [`W32 report`](../docs/experiments/2026-10-02-cached-http-concurrency-report.md), [`W32 decision`](../runs/learning/cached-http-concurrency-v1/decision.json), [`worker-control rationale`](../docs/experiments/2026-10-02-sqlite-worker-control-report.md), [`fresh W1 plan`](../docs/experiments/2026-10-18-cached-http-w1-plan.md), [`W1 report`](../docs/experiments/2026-10-18-cached-http-w1-report.md), [`W1 audit`](../runs/learning/cached-http-w1-v1/independent-audit.json), [`W1 review`](../runs/learning/cached-http-w1-v1/independent-review.json), [`W1 decision`](../runs/learning/cached-http-w1-v1/decision.json) |

The [model version register](../docs/model-versions.md) remains the detailed
checkpoint inventory. Checkpoint hashes establish artifact identity, not
durable storage or public availability. The 8,000-update candidate's results
are reported separately from the application-reference checkpoints and the
default decoder policy. The 32,000-update candidate supersedes the 8,000-update
candidate as the best accepted validation finding, but neither is promoted to
the retained decoder. The dense-set integration is a separate CPU application
path over the 32,000-update checkpoints; it preserves their predictions and
does not change the candidate's scientific or promotion status.
The later `dense-feature-cache-v1` selected a distinct experimental policy that
reduces the measured within-run direct-call cost while preserving learned
scores, but it remains slower than live SQLite. The subsequent
`cached-dense-serving-integration-v1` verified this policy through an optional
authenticated CPU application path without changing learned quality or making
a serving-performance claim.
The subsequent HTTP experiments change the supported performance finding, not
the model: the W32-specific cached advantage remains historical evidence, while
the fresh accepted W1 comparison finds no cached advantage. `PAPER-v0009`
retains both findings with their scheduling boundaries; exact `PAPER-v0008`
preserves the earlier state in which that direct comparison was still pending.

## Historical `PAPER-v0002` PLM-findings path map

This table preserves a proposed source-to-destination path map for a prior
`PAPER-v0002` projection. It is not evidence that the projection was published,
and it does not cover `PAPER-v0003`, `PAPER-v0004`, `PAPER-v0005`,
`PAPER-v0006`, `PAPER-v0007`, `PAPER-v0008`, the current `PAPER-v0009`, or their added evidence.
Any external projection must be established from its own validated packet and
publication receipt; adapted links are not byte-identical to source files.

| Source-repository path | PLM-findings path |
| --- | --- |
| `PAPER.md` | `PAPER.md` |
| `paper-history/PAPER-v0001-2026-09-25.md` | `papers/history/PAPER-v0001-2026-09-25.md` |
| `paper-history/README.md` | `papers/history/README.md` |
| `docs/experiments/2026-09-26-graph-cpu-saturation-offset120-report.md` | `evidence/updates/2026-09-26-graph-cpu-saturation-offset120-v1/experiments/2026-09-26-graph-cpu-saturation-offset120-report.md` |
| `docs/experiments/2026-09-26-graph-cpu-saturation-offset120.json` | `evidence/updates/2026-09-26-graph-cpu-saturation-offset120-v1/experiments/2026-09-26-graph-cpu-saturation-offset120.json` |
| `docs/experiments/2026-09-26-graph-cpu-saturation-offset120-audit.json` | `evidence/updates/2026-09-26-graph-cpu-saturation-offset120-v1/experiments/2026-09-26-graph-cpu-saturation-offset120-audit.json` |
| `docs/experiments/2026-09-26-graph-cpu-saturation-offset120-decision.json` | `evidence/updates/2026-09-26-graph-cpu-saturation-offset120-v1/experiments/2026-09-26-graph-cpu-saturation-offset120-decision.json` |
| `docs/experiments/2026-09-26-graph-cpu-saturation-offset120-plan.md` | `evidence/updates/2026-09-26-graph-cpu-saturation-offset120-v1/experiments/2026-09-26-graph-cpu-saturation-offset120-plan.md` |
| `docs/experiments/2026-09-26-graph-cpu-offset120-throughput.svg` | `evidence/updates/2026-09-26-graph-cpu-saturation-offset120-v1/experiments/2026-09-26-graph-cpu-offset120-throughput.svg` |
| `docs/experiments/2026-09-26-graph-cpu-offset120-power.svg` | `evidence/updates/2026-09-26-graph-cpu-saturation-offset120-v1/experiments/2026-09-26-graph-cpu-offset120-power.svg` |
| `docs/experiments/2026-09-26-graph-cpu-offset120-efficiency.svg` | `evidence/updates/2026-09-26-graph-cpu-saturation-offset120-v1/experiments/2026-09-26-graph-cpu-offset120-efficiency.svg` |
| `docs/protocol-spec.md` | `evidence/source/protocol-spec.md` |
| `docs/learning/06-seed-replication.md` | `evidence/learning/06-seed-replication.md` |
| `docs/learning/25-composing-candidate-sets.md` | `evidence/learning/25-composing-candidate-sets.md` |
| `docs/experiments/2026-09-25-pair-composition-shape-aware.json` | `evidence/updates/2026-09-25-composition-verification/experiments/2026-09-25-pair-composition-shape-aware.json` |
| `docs/experiments/2026-09-25-bilinear-budget8000-replication.json` | `evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication.json` |
| `docs/experiments/2026-09-25-bilinear-budget8000-replication-model-version-inventory.json` | `evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication-model-version-inventory.json` |
| `docs/learning/48-testing-a-budget-across-parent-seeds.md` | `evidence/updates/2026-09-25-bilinear-budget8000-replication/learning/48-testing-a-budget-across-parent-seeds.md` |
| `docs/experiments/2026-09-25-native-graph-oracle.json` | `evidence/updates/2026-09-25-canonical-paper-v0001/experiments/2026-09-25-native-graph-oracle.json` |
| `docs/experiments/2026-09-25-native-graph-oracle-plan.md` | `evidence/updates/2026-09-25-canonical-paper-v0001/experiments/2026-09-25-native-graph-oracle-plan.md` |
| Implementation files at source commit `cca97ffc5491eb1cfa28ac94532d34460a75812e` | `evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/` |

The final paper-specific publication receipt, not this paper revision, records
source and PLM-findings commit identities. Use the receipt under
`docs/publication-receipts/` to establish push status; neither this index nor a
ready queue packet proves publication.

## 2026-10-18 — Pair-coverage finding retained

`PAPER-v0010` adds the accepted [pair-coverage result](../docs/experiments/2026-10-18-bilinear-pair-coverage-report.md): all eight saved membership errors lack reverse training-query coverage; 79.84% of eligible validation pairs have it. This qualifies query-held-out generalization without changing the model. The [failed preparation](../runs/learning/bilinear-pair-coverage-preparation-v1/failed-preparation-attempt-v1/failure-record.json), exact original source/test/plan snapshots and superseding source review remain preserved. It was an adapter failure, not a negative learning result.
