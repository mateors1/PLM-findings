# Protocolized Language Model (Pika Edition)

**Canonical research paper v0011**<br>
**Revision date:** 2026-10-18 (America/Bogota)<br>
**Evidence cutoff:** entity-offset aggregate audit and scientific review completed; owner acceptance recorded 2026-10-19 00:51:35 UTC (2026-10-18 19:51:35 America/Bogota).<br>
**Predecessor:** [`PAPER-v0010`](paper-history/PAPER-v0010-2026-10-18.md), archived byte-identically.

## Abstract

The Pika Edition studies a decoder-only transformer as a protocol-bound set
retriever over a Pokémon product graph. Given one Pokémon identifier and one
dimension, the model emits identifiers for other Pokémon that share at least
one hidden graph attribute. It does not generate prose or attribute names. In
version 1, both supported relations (`TYPE` and `COLOR`) are computable exactly
from the source graph, so the graph compiler is the quality oracle. The
scientific question is whether a compact learned model can approach that oracle
while supporting a useful serving tradeoff; no quality win over the oracle is
possible or claimed.

The retained reference is a small causal decoder with shared product embeddings,
prompt-level membership supervision and a symmetric dimension-conditioned
relation head. Its experimental serving path can select among four generated
answers and, optionally, six pairwise unions. That set-composition policy is
implemented and integration-verified, but remains opt-in. It returns a selected
set with source provenance, not a sequence the decoder generated. On the fixed
222-query validation split, the three-seed ten-candidate experiment selected
569/666 exact sets. A separate bilinear residual learning candidate, fit for
32,000 updates from each of three original parent checkpoints, reached 660/666
exact seed-query outcomes, up from 650/666 at 8,000 updates. These are repeated
evaluations of the same 222 validation queries, not 666 independent examples.
A subsequent matched six-fit experiment adds shared entity offsets to the
bilinear residual while preserving the frozen parent base. Fresh controls
reproduced 660/666 exact outcomes; treatments reached 666/666 with six paired
gains and no losses, passing every seed/group exact-count and macro-F1
nonregression check. This is accepted development-quality evidence on known
entities and adaptively reused queries, not protected-test oracle parity.
The new candidate remains unpromoted and outside ordinary or admitted serving
paths. Its changed arithmetic has no measured serving advantage.
An opt-in authenticated CPU dense-set adapter replays the earlier A-only
candidate's predictions
through a serving contract without changing any selected set. The accepted
`cached-http-w1-v1` experiment compared this cached scoring path with the
stronger tested SQLite worker policy through instrumented result-only Windows
loopback TCP/HTTP: 32 worker tokens for cached scoring and one for live SQLite.
SQLite passed the predeclared gate of at least 10% higher throughput and no
worse p95 in all nine seed/repetition pairs at two, four, eight, and sixteen
clients. One client was mixed under that gate; cached scoring passed at no
tested concurrency. Both arms met a 100 ms p95 limit through sixteen, a tested
lower bound rather than maximum capacity. The timed cached responses contained
29,700 oracle-exact results out of 29,970; SQLite returned 29,970 exact results.
The six known learned seed-query errors across five distinct validation queries
remain unchanged in those serving experiments. That serving campaign trained or
promoted no weights. No energy advantage, remote-network performance, or
production performance is established.

The earlier `cached-http-concurrency-v1` result remains valid for its declared
32-worker SQLite comparator: cached scoring passed its gate at four, eight,
and sixteen clients. A separate SQLite-only control then selected W1 as a
stronger tested baseline. The fresh W1 comparison supplies the direct evidence
that was missing between those campaigns; their rates are not combined to
calculate the new result. Neither worker experiment identifies an individual
scheduling cause or globally optimal worker count. An earlier
audited warm CPU direct-call diagnostic found the uncached
dense scorer's median paired call-time ratio was 12.77–13.08× the live SQLite
compiler across the three pinned seeds. A later frozen feature-cache experiment
used 6.37–6.64% of the contemporaneous unchanged scorer's paired median call
time, but remained 1.135–1.190× the live SQLite time. It passed exact-output
and latency gates, then a separate accepted CPU application integration
preserved its scores and selections through 666 direct score rows and 666
in-process HTTP requests.

## 1. Research question and claim boundary

The system is a deterministic pipeline:

```text
request -> protocol parse -> token prompt -> PLM -> product IDs -> post-process -> hydrate
```

The input is a product identifier, a dimension, and the `SAME` mode. The
National Dex graph and versioned protocol determine the correct set. This makes
version 1 a controlled systems experiment over a deterministic oracle, not a
benchmark in which a learned model can outperform the teacher on correctness.
Quality parity is required to claim a matched-quality serving advantage.
Before parity, latency, capacity, or energy measurements can still characterize
an explicitly labelled quality/performance tradeoff; they do not establish
matched-quality superiority. The evidence cutoff here contains no
matched-quality throughput or energy result.

The paper covers only Pokémon names as opaque product identifiers and the
`TYPE` and `COLOR` dimensions. `COMPLEMENTARY` is deferred; `RELATED`, `BIOME`,
and predator/prey relations are outside this model revision. External datasets
are not part of the active research scope. The settled scope and rationale are
recorded in the [research log](research_log.md) and
[protocol specification](docs/protocol-spec.md).

## 2. Graph, protocol, and teacher

### 2.1 Source graph

SQLite is the source of truth. Product nodes use stable `PKM_<NAME>` keys;
attribute nodes use `ATTR_<VALUE>` keys and act only as hidden join pivots. For
dimension \(d\), the protocol maps `TYPE` to `HAS_TYPE` and `COLOR` to
`HAS_COLOR`. If \(N_d(s)\) is the set of attribute neighbors of subject \(s\),
the teacher's answer is

\[
Y(s,d)=\{x\in P: x\ne s\ \land\ N_d(x)\cap N_d(s)\ne\varnothing\},
\]

where \(P\) is the product set. The model never receives the attribute values
or relation predicates as tokens. It learns a mapping between product
identifiers conditioned on a dimension token.

The compiled `pokemon-v1` snapshot records 1,025 product nodes, 28 hidden
attribute nodes, and 2,576 graph edges. It yields 2,050 non-empty
subject/dimension records and a 2,049-token vocabulary; the longest record is
289 tokens. These counts describe this snapshot, not every later PokeAPI
revision. The corpus compiler orders teacher targets by shared-attribute
count, confidence, then product key. Evaluation compares target sets, so this
ordering is a training convention rather than a correctness criterion. See
the [data/evaluation account](docs/learning/02-data-and-evaluation.md),
[compiler](src/plm/corpus/compiler.py), and [protocol config](configs/protocol/pokemon_v1.yaml).

### 2.2 Token grammar

One training record has the form

```text
BOS SUBJECT DIMENSION SAME ANSWER TARGET... EOS
```

`SUBJECT` and each `TARGET` are product tokens. `ANSWER` divides the five-token
prompt from the answer. The tokenizer uses direct symbol-to-ID lookup: control
IDs occupy 0–31, steering IDs 32–127, and product IDs begin at 1024. The
National Dex vocabulary therefore has 1,025 product rows after the reserved
band. Product IDs remain stable across corpus revisions.

Corpus labels are masked through `ANSWER`; target IDs and `EOS` are supervised
by next-token cross-entropy. There are no numeric model tokens. `IGNORE`,
return limits, pagination, graph predicates, and attribute values belong to
the parser or post-processing layer. Generation ends on `EOS` or the configured
length bound; reaching the bound does not synthesize an `EOS`. The exact
grammar, masking, and vocabulary contract are in the
[protocol specification](docs/protocol-spec.md) and
[corpus compiler](src/plm/corpus/compiler.py).

## 3. Retained reference model

### 3.1 Decoder dimensions

The reference checkpoint family uses `tiny_decoder`, not the larger proposed
`overkill` configuration. The decoder has 8 pre-normalized transformer blocks,
hidden width \(D=256\), 8 query heads, 2 key/value heads, and a maximum context
of 512 tokens. Each attention head is 32 wide. It uses RMSNorm, unscaled RoPE,
causal scaled-dot-product attention, QK normalization, SwiGLU feed-forward
layers, tied input/output embeddings, and no sparse experts. At batch size
\(B\) and sequence length \(T\):

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Input IDs | `[B, T]` | Prompt and teacher-forced product/EOS IDs |
| Decoder states | `[B, T, 256]` | Contextual representation at each position |
| Query projection | `[B, 8, T, 32]` | Eight query heads |
| Key/value projections | `[B, 2, T, 32]` each | Two shared KV heads, repeated across query groups for attention |
| Vocabulary logits | `[B, T, 2049]` | Scores over control, steering, reserved, and product IDs |
| Product embedding rows | `[1025, 256]` | Shared product vectors used by the auxiliary heads |

Native incremental decoding can retain a per-layer key/value cache with shape
`[B, 2, T, 32]` for each of keys and values; the cache belongs to one model and
one request prefix. The executable details are in
[`layers.py`](src/plm/model/layers.py),
[`tiny_decoder.yaml`](configs/model/tiny_decoder.yaml), and
[`generation.py`](src/plm/serving/generation.py).

### 3.2 Training objectives

For an input batch, token logits have shape `[B, T, 2049]`. At position \(t\),
the decoder predicts the label at \(t+1\). Prompt and padding labels are
ignored. The registered three-seed application-reference family was trained
for 2,000 updates with the same corpus and split. Its resolved configuration
uses the neutral first-target weight of 1, prompt-set loss weight 1, and
symmetric-relation loss weight 1; continuation-set and margin losses are off.
The exact seed-specific configuration hashes and checkpoint identities are in
the [model version register](docs/model-versions.md). The accepted bilinear
campaign's new child hashes and residual-update counts are in its
[sealed model-version inventory](docs/experiments/2026-09-25-bilinear-budget8000-replication-model-version-inventory.json).
The corresponding
objective settings are visible in the
[symmetric-relation experiment config](configs/experiment/symmetric_relation_lab.yaml)
and [model implementation](src/plm/model/layers.py).

The prompt-set head reads the causal state at `ANSWER`,
\(h_4\in\mathbb R^{B\times256}\), and projects it against all product rows:

\[
U=(h_4 W_p^\top)E_P^\top\in\mathbb R^{B\times1025},
\]

where \(W_p\in\mathbb R^{256\times256}\) and
\(E_P\in\mathbb R^{1025\times256}\). A balanced binary loss gives the
positive products and negative products equal weight within each query. This
teaches the shared representation which set belongs to the complete prompt,
while the autoregressive vocabulary head remains the answer generator.

The symmetric relation head produces one FP32 score per product. Let
\(e_s,e_i,e_d\in\mathbb R^{256}\) be the normalized subject, candidate, and
dimension embeddings, rescaled to norm 16, and let
\(W_r\in\mathbb R^{256\times256}\). Then

\[
r_d=W_r e_d,\qquad
z_{sdi}=\frac{e_i^\top(e_s\odot r_d)}{16},\qquad
Z\in\mathbb R^{B\times1025}.
\]

The same product vectors are used on both sides, so exchanging subject and
candidate leaves the score unchanged for a fixed dimension. The dimension
embedding supplies the steering signal. The auxiliary loss is balanced across
positive and negative product IDs and excludes the subject. This scorer is
trained alongside the token objective; its use in serving is a separate
inference-policy choice.

The combined reference objective is

\[
L=L_{\mathrm{token}}+L_{\mathrm{prompt-set}}+L_{\mathrm{symmetric}}.
\]

This describes the registered application-reference training family. Other
losses and architecture switches in the repository are experiments or
disabled options, not part of these checkpoints.

## 4. Inference and serving contract

The ordinary prediction path greedily generates a protocol-constrained
identifier sequence and stops on `EOS` or the maximum length. The class mask
allows product tokens in the answer and `EOS` after an answer begins. Optional
uniqueness masking prevents repeats; post-processing then excludes the `SAME`
subject, applies caller `IGNORE` keys, deduplicates in first-seen order, and
applies a return limit. These steps do not query the relation graph to repair
missing or incorrect predictions.

An implemented, opt-in set selector generates four independent paths, each
beginning with a different top-ranked first product, and scores each completed
set with the symmetric head:

\[
s(S)=\sum_{i\in S}z_i.
\]

The optional pair-composition extension retains those four candidates and
adds the six pairwise unions, for ten scored slots. In a conceptual Boolean
membership tensor \(C\in\{0,1\}^{B\times10\times1025}\), the slot scores are
\[
S_{bk}=\sum_{i=1}^{1025}C_{bki}Z_{bi},\qquad S\in\mathbb R^{B\times10}.
\]
The implementation stores canonical ID sets and sums their scores; it does
not allocate this dense tensor. A pair is eligible only when both source
sequences are valid, complete, and unique. Ties prefer earlier slots, so
original paths precede pair unions. The union is a deterministic selected set,
not a new autoregressive sequence or a sequence probability. Full source
provenance remains attached to it.

The native PyTorch service has a single-path `/v1/predict` endpoint and a
separate `/v1/predict-set` endpoint for the pair-composition contract. Pair
composition and symmetric set reranking default off and require explicit
configuration and a checkpoint with a trained symmetric head. The service
loads an explicitly supplied checkpoint; the repository does not identify a
promoted production checkpoint or deployment. The three-control checkpoint
family is the current application reference for composition experiments, not
a claim of an active production release. The accepted
[shape-aware integration record](docs/experiments/2026-09-25-pair-composition-shape-aware.json)
verifies validation-path and HTTP behavior. It is not an SLO, throughput, or
energy result.

An opt-in authenticated CPU dense-set path exposes the bilinear candidate
through `plm serve-dense-set` and `/v1/predict-dense-set`. It computes the
candidate score matrix directly from frozen FP32 embeddings and the learned
residual; the path does not execute transformer blocks and returns selected
product IDs, not generated tokens. The accepted `dense-set-serving-integration-v1` replay
loaded the three existing 32,000-update checkpoints on CPU. It passed 1,332
same-shape score-row comparisons, 666 FastAPI TestClient requests, three
metadata checks, and 297 required synthetic tests with no skips. CPU scores
matched the original CPU arithmetic bit-for-bit at the same shapes. The largest
CPU-to-archived-CUDA score difference was 4.57763671875e-5, with no selection
changes; exact counts remained 218/222, 222/222, and 220/222. This verifies an
opt-in CPU integration contract only: TestClient is in-process and does not
measure real-network capacity, latency, speed, energy, or GPU serving. No new
weights were trained, and the candidate was not promoted. See the
[integration report](docs/experiments/2026-10-02-dense-serving-integration-report.md),
[Lesson 53](docs/learning/53-from-learned-scores-to-a-serving-contract.md),
[frozen readiness](runs/learning/dense-serving-preparation-v1/readiness-v1/readiness.json),
[primary summary](runs/learning/dense-set-serving-integration-v1/summary.json),
[independent audit](runs/learning/dense-set-serving-integration-v1/independent-audit.json),
and [owner decision](runs/learning/dense-set-serving-integration-v1/decision.json).

The later `cached-dense-serving-integration-v1` adds optional feature-cache
admission through paired `--feature-cache-manifest` and
`--feature-cache-manifest-sha256` flags. The authenticated feature manifest
binds the cache to the base manifest, checkpoint, content digest, and numerical
policy; source archives retain separate identities. Existing
invocations keep the uncached path; cache-admission failures do not fall back.
The manifest-backed path is an accepted application integration, not a promoted
checkpoint or a claim of an active production deployment. See the [cached
integration report](docs/experiments/2026-10-02-cached-serving-integration-report.md).
The later [`cached-http-concurrency-v1`](runs/learning/cached-http-concurrency-v1/decision.json)
measured these adapters over loopback HTTP at concurrency 1, 2, 4, 8, and 16.
Its cached throughput advantage at 4/8/16 is specific to W32 SQLite
([Section 5.10](#510-cached-http-concurrency)). The fresh accepted comparison
against W1 SQLite finds an advantage for SQLite at 2/4/8/16 and none for cached
scoring ([Section 5.12](#512-cached-http-against-sqlite-w1)). Both experiments
preserve the same learned predictions and opt-in inference identity.

## 5. Evaluation and supported findings

### 5.1 Evaluation design

The split unit is a full `(subject, dimension)` query. A seeded hash assigns
queries to train, validation, or protected test while leaving the graph intact.
The current National Dex split has 1,637 training queries, 222 validation
queries, and 191 protected-test queries. Adaptive model selection uses the
validation queries. The protected test has not been used for the findings
below. All exact-set counts are strict: one missing or extra product makes a
query incorrect. The full-graph compiler oracle returns the teacher set by
construction, so it scores 222/222 on the fixed validation set. It is the
correctness ceiling, not a competitor that the learned model can beat.

An earlier native HTTP comparison exposes the current serving tradeoff. On
these same 222 validation requests, the guided PLM arm returned 151 exact sets
and completed 0.933 requests/second; the serial loopback SQLite graph arm
returned 222 exact sets and completed 449.477 requests/second (about 482 times
the measured PLM rate). The arms did not have matched quality, and this single-
host serial measurement is not a concurrent-capacity, SLO, or energy result.
It therefore neither demonstrates a learned-model serving advantage nor
settles the comparison that would be meaningful after oracle parity. See the
[native graph-oracle report](docs/experiments/2026-09-25-native-graph-oracle.json)
and its [experiment plan](docs/experiments/2026-09-25-native-graph-oracle-plan.md).

### 5.2 Prompt-level supervision

In three paired seeds on the fixed validation split, adding prompt-set
supervision raised mean raw generation F1 from 23.13% to 60.62% and mean exact
accuracy after the same subject-exclusion/deduplication rules from 15.62% to
45.80%. This supports the retained auxiliary objective on this dataset and
split. Each seed still had substantial errors; seed variation on one split is
not independent-question replication, held-out-catalog generalization, or
oracle parity. See the
[three-seed result](docs/experiments/2026-09-24-seed-replication.json) and
[analysis](docs/learning/06-seed-replication.md).

### 5.3 Four-path and pair-composition inference

The accepted four-path selector scored 501/666 exact validation sets across
the three reference seeds. Adding its six pair unions produced 569/666, with
68 exact gains and no exact losses; pooled F1 was 0.966896. These are 666
model-query observations over the same 222 questions, not 666 distinct
questions. The pair-union experiment therefore supports this opt-in selection
rule on the frozen validation outputs. It does not prove oracle parity: 97 of
the 666 selected outputs remain non-exact. The implementation/integration
identity is `plm-pair-composition-shape-aware-v2`; the portable report records
acceptance and the separate independent audit. See
[`pair-composition-shape-aware.json`](docs/experiments/2026-09-25-pair-composition-shape-aware.json),
[Lesson 25](docs/learning/25-composing-candidate-sets.md), and
[Lesson 29](docs/learning/29-shape-aware-verification.md).

### 5.4 Native serving comparison against the graph oracle

The native serving experiment compared the constrained, unique, KV-cached PLM
endpoint with first-target guidance at \(\alpha=16\) against a read-only
SQLite graph-target query and hydration path. Both arms processed the same 222
validation requests serially through loopback HTTP. The graph arm was
exact on all 222 requests and completed 449.477 requests/second; the PLM arm
was exact on 151 and completed 0.933 requests/second, approximately a 482x
measured throughput gap in favor of the graph implementation. The PLM arm's
macro F1 was 0.8857, 0.1143 below the graph arm. This is direct contrary
evidence to any claim that the current PLM already wins on serving efficiency.
Because the quality differs and the setup is a single-host serial loopback, it
is not a matched-quality capacity benchmark, an SLO test, or an energy
measurement. The [portable report](docs/experiments/2026-09-25-native-graph-oracle.json)
and [plan](docs/experiments/2026-09-25-native-graph-oracle-plan.md) preserve its
scope and environment.

A separate graph-only load curve adds concurrent capacity evidence for the
same capped graph path. With exact 16-ID oracle outputs, it measured 954.2
servings/second at one client and 435.4–460.9 servings/second from 8 through
128 clients. Median CPU-package RAPL power across that concurrent range was
104.1–110.9 W. A user-requested `+120 W` series is arithmetic on those package
readings, not whole-system power. The new sweep was 2.5–6.1% slower and
16.3–26.0 W higher in package power than the prior clean run at client counts
1, 4 and 8; missing CPU telemetry leaves that run-to-run change unexplained.
This graph-only result does not add a concurrent PLM measurement or establish
a matched-quality capacity/energy advantage. See the [experiment report](docs/experiments/2026-09-26-graph-cpu-saturation-offset120-report.md),
[audit](docs/experiments/2026-09-26-graph-cpu-saturation-offset120-audit.json),
[decision](docs/experiments/2026-09-26-graph-cpu-saturation-offset120-decision.json),
[portable per-level result](docs/experiments/2026-09-26-graph-cpu-saturation-offset120.json),
and [declared plan](docs/experiments/2026-09-26-graph-cpu-saturation-offset120-plan.md).

### 5.5 Higher-budget bilinear candidate

The accepted `bilinear-budget32000-v1` campaign continued the same symmetric
bilinear residual method. Each of three fits started from its original parent
checkpoint with a fresh zero residual and fresh optimizer; the 32,000-update
children did not resume the 8,000-update fits. The frozen parent tensors were
unchanged. Each seed passed its predeclared validation improvement gate:
seed 1729 scored 218/222 versus 216/222 at 8,000 updates, seed 1730 scored
222/222 versus 220/222, and seed 1731 scored 220/222 versus 214/222. This is
660/666 exact seed-query outcomes versus 650/666, with ten gains and no losses.
All six remaining nonexact outcomes are dual-`TYPE`. Seed 1730 is exact only on
this fixed validation workload; the same 222 questions are reused across all
three parents.

This improved the accepted validation finding at that stage, not the retained
reference decoder or its default policy. The later entity-offset experiment
in Section 5.14 supersedes its quality result while preserving these checkpoints
as the identities to which the existing serving evidence applies. The residual candidate has not been promoted;
its checkpoint metadata still says `standard_serving_supported=false`. The
separate CPU adapter described above integrates that candidate through an
explicit opt-in path, without changing its outputs or training new weights.
The full lineage and checkpoint identities are in the
[model version register](docs/model-versions.md).

For each original parent, the experiment adds a symmetric residual
\(A^{(r)}\in\mathbb R^{2\times256\times256}\) over normalized product
features \(E^{(r)}\in\mathbb R^{1025\times256}\), with row norm 16:
\[
S_d^{(r)}=\frac{A_d^{(r)}+(A_d^{(r)})^\top}{2},\qquad
z_{sdi}^{(r)}=z_{sdi}^{(r),\mathrm{parent}}
 +\frac{(E_s^{(r)})^\top S_d^{(r)}E_i^{(r)}}{16}.
\]
The fit changes only this residual. Candidate selection uses strict positive
scores, excludes the subject, and sorts product IDs. The ordinary decoder does
not apply this residual; the separately verified opt-in adapter is an
integration of the scorer, not a promotion of the learned model.

### 5.6 Dense-set integration evidence

The `dense-set-serving-integration-v1` experiment evaluates application
integration, not candidate quality. It loads the same three accepted
32,000-update checkpoints and preserves their existing validation selections:
218/222, 222/222, and 220/222. The frozen CPU replay passed 1,332 same-shape
score-row comparisons and 666 in-process HTTP requests; its separate
saved-evidence audit and read-only deployment review passed. A CPU score
matches the archived reference arithmetic exactly at the same shape, while
CPU-to-archived-CUDA differences reach 4.57763671875e-5 without changing a
selected set. The adapter is opt-in and CPU-only. It adds no model weights,
training, validation gains, protected-test evidence, or performance result.
The [report](docs/experiments/2026-10-02-dense-serving-integration-report.md)
and [Lesson 53](docs/learning/53-from-learned-scores-to-a-serving-contract.md)
preserve the contract and its limits.

### 5.7 Uncached direct-call cost against the live oracle

The accepted `dense-oracle-direct-v1` diagnostic compared the uncached dense
scorer with the live SQLite compiler and a separately labelled warm answer
cache. Its saved evidence passed an independent audit and read-only review; all
399 required tests passed with no skips. It timed complete ascending ID-set
calls on warm CPU, using B1 and all 222 admitted validation queries for five
passes across each of the three pinned seeds: 9,990 observations across the
three arms. Each arm returned the complete ascending set, with no output cap or
IGNORE filter. Dense call medians were
4.33790, 4.28820, and 4.24345 ms; live SQLite medians were 0.33320, 0.33205,
and 0.33710 ms. The median of paired per-query dense/SQLite time ratios was
13.0813, 12.9632, and 12.7655, respectively. These are medians of paired
ratios, not ratios of aggregate medians. Prepared answer-cache medians were
1.0, 1.3, and 1.3 microseconds, with cache preparation outside its timer.

This is negative performance evidence for the measured uncached direct-call
boundary: dense scoring is slower than computing this relation with the live
compiler. Dense score hashes and selected sets match the accepted B1 integration
evidence; the dense counts remain 218/222, 222/222, and 220/222, whereas the
SQLite and warm-cache arms each returned 222/222 exact sets per pass. The cache
reuses prepared complete answers; it is not a live relation query or evidence
for unseen-query inference. Its microsecond lookup times approach timer and
dispatch overhead, so their displayed precision is not an intrinsic hardware
cost estimate. Five repeated passes reuse the same queries and do not add
independent accuracy examples. This single desktop run supplies no confidence
interval or statistical-significance claim.

The timing boundary includes dense validation, full score materialization and
metadata construction; the live SQL arm includes the compiler query, ranking,
key conversion, and sorting of the full output. It excludes startup, input
authentication, hydration, HTTP, JSON serialization, output verification, and
logging. The result does not measure concurrent capacity, network latency,
energy, cold starts, optimal SQL,
or a general model speed limit. The later feature-cache experiment tests a
separate script-only policy; its within-run comparison does not replace these
earlier measurements or establish HTTP performance.
See the [report](docs/experiments/2026-10-02-dense-oracle-direct-report.md),
[Lesson 54](docs/learning/54-what-a-latency-number-measures.md),
[summary](runs/learning/dense-oracle-direct-v1/summary.json),
[independent audit](runs/learning/dense-oracle-direct-v1/independent-audit.json),
[read-only review](runs/learning/dense-oracle-direct-v1/independent-review.json),
and [owner decision](runs/learning/dense-oracle-direct-v1/decision.json).

### 5.8 Frozen-feature experimental candidate

The accepted `dense-feature-cache-v1` experiment tests caching features derived
from frozen model weights, not caching predictions. It prepares an owned entity
table and the two complete symmetric transforms once, while snapshotting the
projection and steering tensors; each request still runs the B1 steering
calculation, computes both score terms separately, materializes the scores, and
selects the positive nonself IDs. The feature snapshot stores no
answer sets, complete score vectors, graph relations, or answer table. Its
inference identity is
`dense-positive-bilinear-residual-subject-excluded-ascending-fp32-cached-features-b1-v1`.

The predeclared selection gate required each seed's median paired cached/uncached
time to be at most 0.80 and cached p95 not to worsen. Both gates passed: cached
calls used 6.37–6.64% of the same-run unchanged scorer's paired median time, and
p95 improved for every seed. Across seeds, cached medians were 0.646–0.680 ms,
unchanged dense medians 10.157–10.529 ms, and live SQLite medians 0.564–0.583 ms.
The paired cached/SQLite ratios remained 1.135–1.190, so live SQL was still
faster. Cached and unchanged learned scores and selected sets matched accepted
B1 evidence on all 7,992 warm and timed learned-score checks. Exact counts did
not change: 218/222, 222/222, and 220/222; the six residual failures remain.

Both the unchanged dense and SQL arms were slower than in the earlier direct-call
run. Host scheduling and clocks were not controlled, and the cause of this
variation was not measured. These absolute times are preserved as separate
runs; the candidate gate relies on paired comparisons within the new run.
Feature construction took 18.39–19.54 ms; the snapshot owns 3,412,992 tensor
bytes. Relative to the contemporaneous unchanged scorer, estimated warm-workload
amortization was two or three calls, excluding checkpoint loading. This is not a
SQL break-even estimate, and the byte count is not process or peak memory. All
616 required tests passed with no skips, and the saved evidence passed audit and
read-only review.

The owner selected this as an experimental candidate for later integration.
At the end of `dense-feature-cache-v1`, it existed only in experiment scripts.
The subsequent `cached-dense-serving-integration-v1`, described next, adds the
same policy to the optional authenticated CPU application path. Neither
experiment changes the retained model or learned quality. See the
[report](docs/experiments/2026-10-02-dense-feature-cache-report.md),
[Lesson 55](docs/learning/55-caching-features-without-caching-answers.md),
[summary](runs/learning/dense-feature-cache-v1/summary.json),
[independent audit](runs/learning/dense-feature-cache-v1/independent-audit.json),
[read-only review](runs/learning/dense-feature-cache-v1/independent-review.json),
and [owner decision](runs/learning/dense-feature-cache-v1/decision.json).

### 5.9 Manifest-backed cached serving integration

The accepted `cached-dense-serving-integration-v1` verifies the same feature
cache policy through the optional CPU application path. Paired feature-manifest
and SHA-256 arguments bind the cache to the authenticated base manifest,
checkpoint, cache content digest, and numerical policy; source archives retain
their separate identities. Cache identity remains unchanged through the direct
API and HTTP application. Existing invocations stay uncached, and admission
failures do not fall back. Historical checkpoint metadata is not rewritten.

Across the three pinned seeds, the campaign verified 666 direct score rows,
each containing all 1,025 scores, and 666 main FastAPI TestClient responses. It
also checked one combined `IGNORE`/limit request and one health response per
seed, plus feature manifests, product catalogs, and source archives. Before the
real execution, all 1,012 focused tests passed with no skips. Scores and
selected IDs remain identical to the accepted CPU B1 evidence; exact sets
remain 218/222, 222/222, and 220/222.
The six learned errors remain, no checkpoint was added, the 41-checkpoint
inventory is unchanged, and the 191 protected queries remain closed.

The saved-evidence audit and review accepted this scoped integration. The
reviewer authored the service/CLI adapter but not the cache candidate,
historical bridge, or primary evaluator; the review did not load checkpoint
weights, rebuild the cache from weights, or execute graph queries. The auditor
also reconstructs saved evidence without running the model or graph. This
supports integration fidelity, not independent reproduction of neural or graph
computation.

TestClient runs in-process and the campaign had no timing objective. It does
not establish real-network throughput, concurrent-user capacity, energy
savings, oracle parity, or checkpoint promotion. The earlier direct-call
comparison still records SQLite as faster; this integration result does not
replace that measurement or add a performance claim. See the
[integration report](docs/experiments/2026-10-02-cached-serving-integration-report.md),
[readiness](runs/learning/cached-serving-preparation-v1/readiness-v1/readiness.json),
[summary](runs/learning/cached-dense-serving-integration-v1/summary.json),
[independent audit](runs/learning/cached-dense-serving-integration-v1/independent-audit.json),
[independent review](runs/learning/cached-dense-serving-integration-v1/independent-review.json),
and [owner decision](runs/learning/cached-dense-serving-integration-v1/decision.json).

### 5.10 Cached HTTP concurrency

The accepted `cached-http-concurrency-v1` campaign followed the in-process
TestClient integration with instrumented result-only CPU adapters over loopback
TCP/HTTP on one Windows desktop. It compares this cached scoring adapter with
a live SQLite compiler adapter under the declared per-worker scheduling. It is
application-level loopback evidence, not deployed or unchanged public-endpoint
performance.

The predeclared cached-advantage rule required at least 10% higher throughput
and no worse p95 in every paired observation for every seed and repetition at a
given concurrency. Each table range spans nine observations (three seeds by
three repetitions); paired-ratio ranges compare corresponding observations,
not endpoints or confidence intervals.

| Concurrent clients | Cached verified RPS | SQLite verified RPS | Paired cached/SQLite RPS ratio | Cached p95 (ms) | SQLite p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 820.57-985.81 | 973.31-1057.11 | 0.843-0.962 | 1.191-1.970 | 1.199-1.335 |
| 2 | 905.42-979.47 | 1200.04-1268.84 | 0.734-0.793 | 2.267-2.686 | 1.974-2.209 |
| 4 | 886.33-973.27 | 641.54-676.70 | 1.332-1.482 | 4.501-6.570 | 9.562-10.136 |
| 8 | 861.51-965.21 | 503.37-515.40 | 1.702-1.913 | 8.958-10.354 | 25.731-26.479 |
| 16 | 906.77-976.08 | 477.72-502.62 | 1.834-1.974 | 17.181-19.399 | 53.148-55.700 |

SQLite had higher throughput at one and two clients. Cached scoring passed its
gate at four, eight, and sixteen clients. Both arms returned 100% successful
verified responses and met p95 <=100 ms through all tested points, including
sixteen, for every seed and repetition. Sixteen is only the largest tested
point: this establishes a lower bound and locates neither arm's maximum capacity.

The campaign completed 1,222 required tests with zero skips, then measured
59,940 requests across all timed cells; 19,980 warmups and 1,332 serial
preflights were separate. Both arms used the same result-only envelope, request
model, serialization, catalog, body bound, and admission limit. The cached arm
still ran the accepted predictor, score-hash construction, postprocessing, and
hydration. SQLite ran live compiler targets, converted keys to sorted product
IDs, and used persistent read-only per-worker connections without a benchmark-
imposed global query lock. Each server had 32 worker tokens in its own process;
one arm received load at a time while both servers stayed loaded for a seed's
comparison. Persistent client connections sent one outstanding request each;
response latency ended after the full body arrived and throughput used cell
wall time. Client bookkeeping and common server-trace overhead were included.

Among 29,970 timed cached responses, 29,700 were oracle-exact; all 29,970 timed
SQLite responses were exact. The cached exact counts remain 218/222, 222/222,
and 220/222 for seeds 1729, 1730, and 1731. The 270 nonexact responses repeat
six seed-query errors across five distinct validation queries; FEZANDIPITI is
one query repeated for seeds 1729 and 1731. These response repetitions do not
add independent errors or examples. The saved-score geometry diagnostic is
summarized in the [accepted report](docs/experiments/2026-09-26-bilinear-budget32000-error-geometry-report.md).
Scores, checkpoints, and protected-test access did not change, and no model
was promoted.

This is a single-desktop loopback result. It excludes remote-network conditions,
cold starts, and energy; it does not compare tuned SQL scheduling or establish
that SQLite's throughput decline is caused by contention. It is not a
matched-quality advantage because the cached predictor retains six seed-query
errors. The accepted decision and evidence are in the [report](docs/experiments/2026-10-02-cached-http-concurrency-report.md),
[plan](docs/experiments/2026-10-02-cached-http-concurrency-plan.md),
[readiness](runs/learning/cached-http-preparation-v1/readiness-v1/readiness.json),
[summary](runs/learning/cached-http-concurrency-v1/summary.json),
[independent audit](runs/learning/cached-http-concurrency-v1/independent-audit.json),
[independent review](runs/learning/cached-http-concurrency-v1/independent-review.json),
and [owner decision](runs/learning/cached-http-concurrency-v1/decision.json).

A separate SQLite-only control varied the AnyIO worker-token limit. In the
`sqlite-worker-control-v1` campaign, the cached comparison above used W32; its
accepted result must remain tied to that baseline. See [Section 5.11](#511-sqlite-worker-control)
for the independent W1/W32 comparison and its limits. The two campaigns cannot
be combined to infer a new model-versus-SQLite winner.

### 5.11 SQLite worker control

The accepted `sqlite-worker-control-v1` experiment changed the worker-token
limit for the same synchronous SQLite endpoint: W1 versus W32. It used the same
222 oracle queries and answers, client grid, and actual Windows loopback HTTP.
It did not run the learned model or checkpoints. The predeclared gate required
at least 10% higher throughput and no worse p95 in every paired repetition.
Ranges cover three observations per concurrency; paired ratios compare matched
repetitions, not range endpoints or confidence intervals.

| Clients | W1 verified RPS | W32 verified RPS | Paired W1/W32 RPS ratio | W1 p95 (ms) | W32 p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1040.80–1069.43 | 987.54–1083.71 | 0.977–1.083 | 1.148–1.185 | 1.141–1.482 |
| 2 | 1291.45–1325.49 | 1268.20–1308.19 | 0.987–1.045 | 1.758–1.848 | 1.922–2.006 |
| 4 | 1275.47–1333.24 | 652.38–688.74 | 1.908–2.022 | 3.341–3.599 | 9.320–9.882 |
| 8 | 1234.86–1306.31 | 517.77–530.62 | 2.385–2.483 | 6.695–7.362 | 25.245–25.953 |
| 16 | 1199.16–1309.38 | 491.01–503.18 | 2.442–2.627 | 12.851–14.662 | 53.249–54.351 |

W1 passed the fixed improvement gate at C4, C8, and C16, with lower p95 in
every paired repetition at those points. C1 and C2 were mixed under the gate.
This supports whole-endpoint synchronous scheduling as a contributor to the
W32 SQLite decline observed in the prior campaign, but does not identify the
responsible internal cost. The worker limit applies to the entire synchronous
endpoint, not just SQLite. W1 opened one connection and W32 opened 24 over each
server lifetime; these counts are not peak simultaneous queries.

All 19,980 timed, 6,660 warmup, and 444 preflight responses were oracle-exact.
They repeat the same 222 questions and add no independent quality examples.
The 205 focused synthetic/regression tests passed with zero skips; Ruff and
formatting passed. The first campaign and independent audit exited zero, and a
separate evidence review passed. No weights, production defaults, checkpoint
inventory, inference arithmetic, protected-test access, or model-promotion
status changed. W1 is a comparison baseline, not a demonstrated global optimum.

The W1 result narrows interpretation of the cached HTTP comparison: that run
measured the learned adapter against its declared W32 SQLite configuration.
This separate control shows a stronger tested SQLite scheduling policy at
higher concurrency. Their throughput values cannot be combined to rank the
model against W1; the subsequent fresh comparison is reported in
[Section 5.12](#512-cached-http-against-sqlite-w1). This control claims no GIL,
SQL-lock, or other individual cause, and no maximum capacity or energy result.
Evidence is in the
[report](docs/experiments/2026-10-02-sqlite-worker-control-report.md),
[plan](docs/experiments/2026-10-02-sqlite-worker-control-plan.md),
[readiness](runs/learning/sqlite-worker-preparation-v1/readiness-v1/readiness.json),
[summary](runs/learning/sqlite-worker-control-v1/summary.json),
[independent audit](runs/learning/sqlite-worker-control-v1/independent-audit.json),
[independent review](runs/learning/sqlite-worker-control-v1/independent-review.json),
and [owner decision](runs/learning/sqlite-worker-control-v1/decision.json).

### 5.12 Cached HTTP against SQLite W1

The accepted `cached-http-w1-v1` experiment directly compared the unchanged
cached scorer with the SQLite W1 policy selected by the worker control. It used
the same three checkpoints, 222 admitted development queries per seed, graph,
label-free catalog, CPU FP32 B1 arithmetic, and instrumented loopback HTTP
boundary. Cached scoring retained 32 worker tokens and its serial scoring lock;
SQLite used one worker token. Both retained an admission bound of 32. A worker
token limits the whole synchronous endpoint, including serialization and
hydration, so this comparison does not isolate time spent inside SQLite.

For each concurrency, each arm had three repetitions for each of three seeds.
The bidirectional gate requires throughput at least 1.10 times the comparator
and p95 no worse in **all nine pairs**. The following ranges describe those
pairs; they are not confidence intervals or ratios of range endpoints.

| Clients | Cached verified RPS | SQLite W1 verified RPS | Paired SQLite/cached RPS ratio | Paired SQLite/cached p95 ratio | Gate result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 972.80–1120.46 | 1074.15–1161.53 | 1.007–1.158 | 0.656–1.172 | Mixed |
| 2 | 568.82–1093.50 | 1152.29–1471.18 | 1.323–2.026 | 0.363–0.814 | SQLite improvement |
| 4 | 550.08–1063.17 | 1044.26–1445.06 | 1.309–1.898 | 0.344–0.813 | SQLite improvement |
| 8 | 824.68–1072.44 | 1277.99–1449.15 | 1.313–1.550 | 0.476–0.852 | SQLite improvement |
| 16 | 766.74–1064.67 | 1202.39–1448.88 | 1.308–1.691 | 0.406–0.862 | SQLite improvement |

SQLite passed at C2/C4/C8/C16; cached scoring passed at none. At C1, SQLite
had higher throughput in every pair, but the 10% margin and no-worse-p95
conditions did not both hold in every pair. Both arms returned successful,
verified responses throughout and met p95 <=100 ms at every tested concurrency
for every seed and repetition. C16 remains only a lower bound on capacity.

The frozen readiness suite passed 352 tests with zero skips. The campaign
completed 59,940 timed requests, 19,980 warmups, and 1,332 preflights, totaling
81,252 correlated client requests and server traces across six normally stopped
server processes. An independent audit and a separate raw-row reconstruction
verified request schedules, body bytes and hashes, cached score hashes,
persistent sockets, actual worker policies, nearest-rank latency percentiles,
throughput, exact goodput, and the directional gates. Expected answer sets
remained outside the serving processes. Timed cached answers were oracle-exact
29,700/29,970 times, versus 29,970/29,970 for SQLite. These repeat the same
218/222, 222/222, and 220/222 learned results; they add no independent quality
examples and eliminate none of the six seed-query errors.

This is negative evidence for a cached serving advantage against the selected
stronger SQLite baseline. It narrows the earlier W32 result without invalidating
that historical measurement. The new ratios come entirely from the new matched
campaign. The retained model weights, inference arithmetic, protected-test
boundary, and promotion status are unchanged. W1 is not proven globally optimal;
no energy result, maximum capacity, remote-network performance, or production
SLO follows. The accepted evidence is in the
[report](docs/experiments/2026-10-18-cached-http-w1-report.md),
[portable evidence](docs/experiments/2026-10-18-cached-http-w1-portable-evidence.json),
[plan](docs/experiments/2026-10-18-cached-http-w1-plan.md),
[readiness](runs/learning/cached-http-w1-preparation-v1/readiness-v1/readiness.json),
[summary](runs/learning/cached-http-w1-v1/summary.json),
[independent audit](runs/learning/cached-http-w1-v1/independent-audit.json),
[independent review](runs/learning/cached-http-w1-v1/independent-review.json),
and [owner decision](runs/learning/cached-http-w1-v1/decision.json).

### 5.13 Reverse-query supervision and the scope of generalization

The accepted `bilinear-pair-coverage-v1` diagnostic joined saved error IDs to
training-query identities using twelve authenticated metadata artifacts. Its
repaired preflight implementation passed 68 synthetic tests; independent
standard-library reconstruction matched all 666 rows and seed/group/pooled
totals. No predictions, weights, labels or protected examples were newly read
or generated by this diagnostic.

For validation subject `s`, candidate `i != s` and dimension `d`, coverage means
`(i,d)` is an admitted training query. Every TYPE validation row has 805 covered
and 219 uncovered candidate pairs; COLOR has 832 and 192. Each seed therefore
contains 181,491 covered and 45,837 uncovered occurrences (79.8366% covered).
All seven missed and one extra membership occurrences are uncovered. This is
consistent with symmetry and exact training fit; it does not establish a causal
mechanism or bitwise FP32 agreement across training/evaluation shapes.

The result quantifies an enduring evaluation limit: query holdout does not
imply pair holdout. These are known-entity experiments, not unseen-entity
retrieval. Absence from the admitted training identities does not identify a
protected partition or rule out indirect learning. Candidate denominators mix
positive and negative memberships, so they do not supply class-conditioned
error rates. Pooled pairs repeat queries across seeds and may repeat symmetric
pairs in both directions. The diagnostic itself changed neither the model nor
its six failed seed/query outcomes. The subsequent entity-offset experiment
is a separate learning study, reported in Section 5.14.

Evidence: [plan](docs/experiments/2026-10-18-bilinear-pair-coverage-plan.md),
[report](docs/experiments/2026-10-18-bilinear-pair-coverage-report.md),
[full rows](docs/experiments/2026-10-18-bilinear-pair-coverage-evidence.json),
[independent audit](docs/experiments/2026-10-18-bilinear-pair-coverage-audit.json),
and [owner decision](docs/experiments/2026-10-18-bilinear-pair-coverage-decision.json).

### 5.14 Shared entity offsets: accepted development-quality candidate

The accepted `bilinear-entity-offset32000-v1` experiment tests representation
flexibility against freshly rerun A-only controls. Each seed's two arms restart
from its original 2,000-update parent with zero adaptation parameters and a fresh
optimizer. Both perform exactly 32,000 full-batch updates over the same ordered
1,637 training queries. The six fits run control then treatment for seeds 1729,
1730 and 1731; each fit is audited and accepted before its successor begins.

Let \(E\in\mathbb R^{1025\times256}\) be the frozen normalized parent features,
with row norm 16, and \(z_0(E)\) the frozen parent base score. The treatment learns
\(\Delta E\in\mathbb R^{1025\times256}\) and
\(A\in\mathbb R^{2\times256\times256}\):
\[
F=E+\Delta E,\qquad S_d=(A_d+A_d^\top)/2,\qquad
z_{\mathrm{offset}}(s,i,d)=z_0(E)(s,i,d)+F_s^\top S_dF_i/16.
\]
The control uses the same residual with \(E\) instead of \(F\). The parent base
always uses \(E\); all 93 original state entries remain unchanged. The same
\(F\) serves subject/candidate roles and both dimensions, without renormalization.
Treatment stores 393,472 trainable scalars, including 262,400 offsets, versus
131,072 in control; the respective state inventories are 95 and 94 entries.
At zero \(A\), the first offset gradient is necessarily zero. Subsequent gradients
activate and the final offsets change. Nonzero-A/zero-offset score and A-gradient
parity are checked using the same authenticated residual operation order.

Both arms use the archived query/class-balanced BCE, FP32 AdamW with learning
rate 0.0003 and zero weight decay, and the fixed numerical recipe. There is no
scheduler, clipping, early stopping, budget search or threshold adjustment.
All six fits already have 1,637/1,637 exact training sets at the fixed observations
after updates 8,000 and 32,000. The accepted gain is in development predictions,
not merely training exactness or decreasing BCE.

| Original parent seed | Fresh control exact | Offset treatment exact | Paired gains | Paired losses |
| --- | ---: | ---: | ---: | ---: |
| 1729 | 218/222 | 222/222 | 4 | 0 |
| 1730 | 222/222 | 222/222 | 0 | 0 |
| 1731 | 220/222 | 222/222 | 2 | 0 |
| Repeated seed/query outcomes | 660/666 | 666/666 | 6 | 0 |

Every seed and its COLOR, single-TYPE and dual-TYPE groups passes exact-count
and macro-F1 nonregression; the pooled paired exact gain is strictly positive.
All treatments have macro F1 1.0 and zero false-positive/false-negative memberships
on this slice. The same 222 known-entity development queries recur across seeds:
666 outcomes are not 666 independent examples, and this is not unseen-entity
generalization. Candidate appearances can supervise entity offsets even when
their subject queries are held out. The 191 protected queries remain closed.

Fresh controls reproduce every historical selected set and required quality
metric, but not full score bytes, full loss traces or final residual-state hashes.
Their first pre-update loss mismatches occur at updates 9, 7 and 13; maximum
absolute endpoint score differences are 0.0000152587890625 for each seed.
These are descriptive numerical replay findings, separate from exact model and
optimizer save/reload within every new fit. Final evaluation uses reloaded
checkpoints at batches of eight/final six, strict-positive nonself selection,
and ascending IDs.

The accepted aggregate audit reconstructs sets, metrics, observations and paired
gates from saved evidence. Frozen weights, gradient history and optimizer/reload
equality remain authenticated execution receipts rather than independent neural
replay. The owner retains this method as a development-quality candidate, not a
promoted decoder or served model. Existing cached-serving latency and HTTP
results cannot transfer to these new checkpoints: their arithmetic and artifacts
require separate integration, correctness checks and matched measurements.
Scaling \(F\) by \(c\) and \(A\) by \(1/c^2\) preserves the residual, so drift norms
are descriptive rather than a unique explanation of learned geometry.

Evidence: [declaration](docs/experiments/2026-10-18-bilinear-entity-offset-plan.md),
[report](docs/experiments/2026-10-18-bilinear-entity-offset-report.md),
[portable evidence and lineage](docs/experiments/2026-10-18-bilinear-entity-offset-evidence.json),
[aggregate audit](docs/experiments/2026-10-18-bilinear-entity-offset-audit.json),
[independent review](docs/experiments/2026-10-18-bilinear-entity-offset-review.json),
and [owner decision](docs/experiments/2026-10-18-bilinear-entity-offset-decision.json).

## 6. Limitations and remaining work

1. **Development exactness is not general oracle parity.** The entity-offset
   candidate reaches 222/222 on each of three seeds, while the pair-composition
   path and older served A-only checkpoints retain their documented errors.
   The new candidate remains unpromoted and has no accepted serving integration.
   Its development result establishes neither protected-test nor unseen-entity
   accuracy and supplies no matched-quality serving measurement.
2. **Validation is not protected testing.** The experiments repeatedly use the
   same 222 queries for adaptive development. The 191-query test partition is
   still protected; the reported candidate results do not establish its
   accuracy.
3. **The graph is small and task-specific.** It contains 1,025 product IDs and
   two fixed dimensions. This does not establish performance on new products,
   changed graph snapshots, other domains, or uncomputable relations.
4. **Matched-quality serving superiority remains unmeasured.** The earlier
   uncached scorer was slower than live SQLite at the measured warm CPU B1
   direct-call boundary: 4.24–4.34 ms versus 0.332–0.337 ms, with median paired
   ratios of 12.77–13.08× across three seeds. The feature-cache candidate
   remained 1.135–1.190× live SQLite in its own campaign. The cached HTTP
   campaign passed its gate at C4/C8/C16 against W32 SQLite, while SQLite led
   at C1/C2. A separate SQLite-only W1/W32 control passed its gate for W1 at
   C4/C8/C16 and was mixed at C1/C2. The subsequent fresh W1 comparison found
   SQLite passes its gate at C2/C4/C8/C16, with C1 mixed and no cached advantage.
   This conclusion uses within-run pairs, not rates combined across campaigns.
   Both previously measured learned paths retain six
   seed-query errors across five distinct validation queries. Those serving
   results do not apply to the new development-exact offset checkpoints. W1 changes
   scheduling for the whole synchronous
   endpoint; it does not isolate an SQL, GIL, or lock cause and does not prove
   an optimal worker count. No maximum-capacity, remote-network, energy,
   matched-quality, GPU-serving, or production-SLO claim follows. The separate
   vLLM/Phase-D path is not the evidence reported here.
5. **Candidate weights are not durably archived.** Checkpoint hashes and local
   receipts identify saved artifacts but do not make the ignored local weight
   files publicly retrievable or backed up. No DOI or external weight archive
   is established.
6. **The source of the quality gain is not identified.** Earlier residual errors
   concentrated in dual-`TYPE`; the new matched offset intervention removes them
   on this development slice. This does not isolate a unique learned geometry,
   prove a general remedy for that relation, or erase the prior negative affine
   and threshold findings. Feature/matrix scale compensation limits norm-based
   interpretations.

These limitations keep the result a systems-and-protocol study. The current
research direction is to verify the new candidate in a separately declared
serving integration, preserve the protected evaluation boundary, and require
matched quality before claiming a serving advantage. Pre-parity measurements can still report
an explicitly labelled quality/performance tradeoff, as the direct-call result
above does. Network capacity, energy, and SLO claims need their own measurements
and durable, versioned artifacts.

## 7. Revision and identity ledger

The paper revision is not a model release. The identities below are deliberately
separate:

| Layer | Identity for this paper |
| --- | --- |
| Paper | `PAPER-v0011`, dated 2026-10-18; predecessor `PAPER-v0010` |
| Protocol/data | `pokemon-v1`; National Dex corpus snapshot `pokemon_v1_f1541479_20260924` |
| Baseline implementation source | Code and configs for the retained decoder are from source commit `cca97ffc5491eb1cfa28ac94532d34460a75812e` |
| Reference checkpoints | `national_dex_continuation_control_s1729_v1`, `_s1730_v1`, `_s1731_v1`; hashes and resolved-config hashes are in [model-versions.md](docs/model-versions.md) |
| Inference policy | Ordinary single-path generation is the default; pair-composition and the CPU dense-set adapter are separate opt-in policies |
| Pair-composition evaluator | `plm-pair-composition-shape-aware-v2` |
| Bilinear learning campaign | `bilinear-budget32000-v1`; accepted 660/666 validation evidence, candidate not promoted; full lineage is in [model-versions.md](docs/model-versions.md) |
| Entity-offset learning campaign | `bilinear-entity-offset32000-v1`; six fresh fits; accepted development-quality candidate at 666/666 versus matched controls 660/666, +6/-0; no promotion or serving admission |
| Entity-offset architecture | `plm-frozen-parent-bilinear-entity-offset-v1`; frozen base plus shared entity offsets in the symmetric residual |
| Entity-offset evaluation/inference | `plm-entity-offset-fullscore-b8-v1`; `strict-positive-nonself-ascending-ID-batch8-v1`; distinct from previous CPU cached-serving policies |
| Entity-offset checkpoints and evidence | Six new local children with lineage in the [portable evidence](docs/experiments/2026-10-18-bilinear-entity-offset-evidence.json) and [model register](docs/model-versions.md); [audit](docs/experiments/2026-10-18-bilinear-entity-offset-audit.json), [review](docs/experiments/2026-10-18-bilinear-entity-offset-review.json), [decision](docs/experiments/2026-10-18-bilinear-entity-offset-decision.json) |
| Dense integration campaign | `dense-set-serving-integration-v1`; CPU integration replay of existing checkpoints, no new model or weights |
| Dense native inference identity | `dense-positive-bilinear-residual-subject-excluded-ascending-fp32-native-b1-v1` |
| Integration evidence | [Report](docs/experiments/2026-10-02-dense-serving-integration-report.md), [summary](runs/learning/dense-set-serving-integration-v1/summary.json), [audit](runs/learning/dense-set-serving-integration-v1/independent-audit.json), [decision](runs/learning/dense-set-serving-integration-v1/decision.json) |
| Direct-call performance diagnostic | `dense-oracle-direct-v1`; earlier uncached warm serial CPU B1 comparison, complete-set calls, five passes over the same 222 queries for three seeds; uncached dense scorer was slower than live SQLite, with warm answer-cache reuse reported separately |
| Direct-call evidence | [Report](docs/experiments/2026-10-02-dense-oracle-direct-report.md), [Lesson 54](docs/learning/54-what-a-latency-number-measures.md), [summary](runs/learning/dense-oracle-direct-v1/summary.json), [audit](runs/learning/dense-oracle-direct-v1/independent-audit.json), [read-only review](runs/learning/dense-oracle-direct-v1/independent-review.json), [decision](runs/learning/dense-oracle-direct-v1/decision.json) |
| Feature-cache candidate | `dense-feature-cache-v1`; selected experimental policy; exact score/selection checks and predeclared latency gate passed; learned quality is unchanged |
| Feature-cache inference identity | `dense-positive-bilinear-residual-subject-excluded-ascending-fp32-cached-features-b1-v1` |
| Feature-cache evidence | [Report](docs/experiments/2026-10-02-dense-feature-cache-report.md), [Lesson 55](docs/learning/55-caching-features-without-caching-answers.md), [summary](runs/learning/dense-feature-cache-v1/summary.json), [audit](runs/learning/dense-feature-cache-v1/independent-audit.json), [review](runs/learning/dense-feature-cache-v1/independent-review.json), [decision](runs/learning/dense-feature-cache-v1/decision.json) |
| Cached-feature serving integration | `cached-dense-serving-integration-v1`; optional authenticated CPU manifest integration preserves accepted scores and selections through direct and in-process HTTP checks; no model promotion or timing claim |
| Cached-serving evidence | [Report](docs/experiments/2026-10-02-cached-serving-integration-report.md), [readiness](runs/learning/cached-serving-preparation-v1/readiness-v1/readiness.json), [summary](runs/learning/cached-dense-serving-integration-v1/summary.json), [audit](runs/learning/cached-dense-serving-integration-v1/independent-audit.json), [review](runs/learning/cached-dense-serving-integration-v1/independent-review.json), [decision](runs/learning/cached-dense-serving-integration-v1/decision.json) |
| Cached HTTP concurrency campaign | `cached-http-concurrency-v1`; cached scoring passes the paired throughput/p95 gate at C4/C8/C16, while SQLite leads at C1/C2; both meet p95 <=100 ms through C16, a tested lower bound only. Six seed-query errors across five validation queries remain; no model promotion or matched-quality advantage. |
| Cached HTTP concurrency evidence | [Report](docs/experiments/2026-10-02-cached-http-concurrency-report.md), [plan](docs/experiments/2026-10-02-cached-http-concurrency-plan.md), [readiness](runs/learning/cached-http-preparation-v1/readiness-v1/readiness.json), [summary](runs/learning/cached-http-concurrency-v1/summary.json), [audit](runs/learning/cached-http-concurrency-v1/independent-audit.json), [review](runs/learning/cached-http-concurrency-v1/independent-review.json), [decision](runs/learning/cached-http-concurrency-v1/decision.json) |
| SQLite worker-control campaign | `sqlite-worker-control-v1`; W1 passes the worker-limit gate over W32 at C4/C8/C16 and is mixed at C1/C2. This varies scheduling for the whole synchronous endpoint and identifies no internal cause or global optimum. It limits the prior cached result to its W32 SQLite baseline; separate campaigns cannot establish a model-versus-W1 winner. |
| SQLite worker-control evidence | [Report](docs/experiments/2026-10-02-sqlite-worker-control-report.md), [plan](docs/experiments/2026-10-02-sqlite-worker-control-plan.md), [readiness](runs/learning/sqlite-worker-preparation-v1/readiness-v1/readiness.json), [summary](runs/learning/sqlite-worker-control-v1/summary.json), [audit](runs/learning/sqlite-worker-control-v1/independent-audit.json), [review](runs/learning/sqlite-worker-control-v1/independent-review.json), [decision](runs/learning/sqlite-worker-control-v1/decision.json) |
| Cached HTTP versus W1 campaign | `cached-http-w1-v1`; fresh within-run comparison with unchanged cached32 and selected SQLite1. SQLite passes all nine paired gates at C2/C4/C8/C16; C1 mixed; no cached advantage. Quality, checkpoint and inference identities remain unchanged. |
| Cached HTTP versus W1 evidence | [Report](docs/experiments/2026-10-18-cached-http-w1-report.md), [portable evidence](docs/experiments/2026-10-18-cached-http-w1-portable-evidence.json), [summary](runs/learning/cached-http-w1-v1/summary.json), [audit](runs/learning/cached-http-w1-v1/independent-audit.json), [review](runs/learning/cached-http-w1-v1/independent-review.json), [decision](runs/learning/cached-http-w1-v1/decision.json) |
| Pair-coverage diagnostic | `bilinear-pair-coverage-v1`, implementation `bilinear-pair-coverage-preflight-v2`; accepted descriptive metadata join, no new model or predictions. [Report and evidence](docs/experiments/2026-10-18-bilinear-pair-coverage-report.md). |
| Publication | This is a local paper revision; publication identity and status require a separate receipt |

The version and historical-evidence index is
[`paper-history/README.md`](paper-history/README.md). The PLM-findings synthesis
is separately versioned; this local source revision does not assert that an
external projection was updated. Any future projection needs adapted relative
links and its own publication receipt. A publication commit or receipt does not
change the evidence cutoff or imply a software release.

## Conclusion

Pika v1 makes a narrow, testable proposal: condition a compact decoder on an
explicit relation dimension and emit only product identifiers for the
corresponding set. The data compiler defines a perfect oracle, the model and
serving contracts are implemented, and set composition improves fixed-split
validation exactness. The accepted 32,000-update residual candidate improves
fixed-split validation performance to 660/666. The subsequent matched entity-offset
experiment reaches 666/666 with six paired gains and no losses, passing the
predeclared development-quality gate across all seeds and groups. This retained
quality candidate remains unpromoted, reuses the same 222 development queries
across three seeds, and leaves the 191 protected queries closed. Its new
checkpoints are not admitted by existing serving manifests.
An opt-in authenticated CPU adapter for the older A-only checkpoints accepts
feature-cache manifests and preserves
accepted scores and selections through direct and in-process HTTP checks. A
later instrumented loopback HTTP campaign found a cached-adapter throughput
advantage over its W32 SQLite comparator at C4/C8/C16, while SQLite led at
C1/C2; the cached path still repeats six seed-query errors across five
validation queries. A separate SQLite-only control selected W1 as a stronger
tested baseline. The fresh `cached-http-w1-v1` comparison then found SQLite
passes the throughput/p95 gate at C2/C4/C8/C16; C1 is mixed and cached scoring
passes at none. This direct comparison limits the earlier advantage to the
W32 scheduling regime. Both arms met the 100 ms p95 limit through C16, a tested
lower bound rather than maximum capacity. Those serving campaigns preserve the
older weights and their six errors. Their latency and HTTP evidence cannot
transfer to the new offset arithmetic. Protected-test oracle parity and a
matched-quality systems advantage remain open; no energy savings,
remote-network or production performance, or model promotion follows.
