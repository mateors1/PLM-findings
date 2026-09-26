# Protocolized Language Model (Pika Edition)

**Canonical research paper v0001**  
**Revision date:** 2026-09-25 (America/Bogota)  
**Evidence cutoff:** 2026-09-26 03:12 UTC (2026-09-25 22:12 America/Bogota)  
**Predecessor:** none; this is the initial canonical revision.  
**Publication projection:** PLM-findings mirror of source revision `PAPER-v0001`; citation paths adapted.

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
569/666 exact sets. A separate, independently accepted 8,000-update bilinear
residual candidate reached 434/444 exact sets on two fresh seeds, or 650/666
when historical seed 1729 is included. That candidate is not supported by the
ordinary decoder or standard serving path and has not been promoted. Neither
result establishes oracle parity, protected-test performance, or serving
efficiency.

## 1. Research question and claim boundary

The system is a deterministic pipeline:

```text
request -> protocol parse -> token prompt -> PLM -> product IDs -> post-process -> hydrate
```

The input is a product identifier, a dimension, and the `SAME` mode. The
National Dex graph and versioned protocol determine the correct set. This makes
version 1 a controlled systems experiment over a deterministic oracle, not a
benchmark in which a learned model can outperform the teacher on correctness.
Quality parity is a prerequisite for comparing latency, concurrent-user
capacity, or energy per request. The evidence cutoff here contains no
matched-quality throughput or energy result.

The paper covers only Pokémon names as opaque product identifiers and the
`TYPE` and `COLOR` dimensions. `COMPLEMENTARY` is deferred; `RELATED`, `BIOME`,
and predator/prey relations are outside this model revision. External datasets
are not part of the active research scope. The settled scope and rationale are
recorded in the [research log](evidence/source/research_log.md) and
[protocol specification](evidence/source/protocol-spec.md).

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
the [data/evaluation account](evidence/learning/02-data-and-evaluation.md),
[compiler](evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/src/plm/corpus/compiler.py), and [protocol config](evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/configs/protocol/pokemon_v1.yaml).

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
[protocol specification](evidence/source/protocol-spec.md) and
[corpus compiler](evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/src/plm/corpus/compiler.py).

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
[`layers.py`](evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/src/plm/model/layers.py),
[`tiny_decoder.yaml`](evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/configs/model/tiny_decoder.yaml), and
[`generation.py`](evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/src/plm/serving/generation.py).

### 3.2 Training objectives

For an input batch, token logits have shape `[B, T, 2049]`. At position \(t\),
the decoder predicts the label at \(t+1\). Prompt and padding labels are
ignored. The registered three-seed application-reference family was trained
for 2,000 updates with the same corpus and split. Its resolved configuration
uses the neutral first-target weight of 1, prompt-set loss weight 1, and
symmetric-relation loss weight 1; continuation-set and margin losses are off.
The exact seed-specific configuration hashes and checkpoint identities are in
the [model version register](evidence/updates/2026-09-25-bilinear-budget8000-replication/model-versions.md). The accepted bilinear
campaign's new child hashes and residual-update counts are in its
[sealed model-version inventory](evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication-model-version-inventory.json).
The corresponding
objective settings are visible in the
[symmetric-relation experiment config](evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/configs/experiment/symmetric_relation_lab.yaml)
and [model implementation](evidence/source-code/cca97ffc5491eb1cfa28ac94532d34460a75812e/src/plm/model/layers.py).

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
[shape-aware integration record](evidence/updates/2026-09-25-composition-verification/experiments/2026-09-25-pair-composition-shape-aware.json)
verifies validation-path and HTTP behavior. It is not an SLO, throughput, or
energy result.

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
[native graph-oracle report](evidence/updates/2026-09-25-canonical-paper-v0001/experiments/2026-09-25-native-graph-oracle.json)
and its [experiment plan](evidence/updates/2026-09-25-canonical-paper-v0001/experiments/2026-09-25-native-graph-oracle-plan.md).

### 5.2 Prompt-level supervision

In three paired seeds on the fixed validation split, adding prompt-set
supervision raised mean raw generation F1 from 23.13% to 60.62% and mean exact
accuracy after the same subject-exclusion/deduplication rules from 15.62% to
45.80%. This supports the retained auxiliary objective on this dataset and
split. Each seed still had substantial errors; seed variation on one split is
not independent-question replication, held-out-catalog generalization, or
oracle parity. See the
[three-seed result](evidence/experiments/2026-09-24-seed-replication.json) and
[analysis](evidence/learning/06-seed-replication.md).

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
[`pair-composition-shape-aware.json`](evidence/updates/2026-09-25-composition-verification/experiments/2026-09-25-pair-composition-shape-aware.json),
[Lesson 25](evidence/learning/25-composing-candidate-sets.md), and
[Lesson 29](evidence/updates/2026-09-25-composition-verification/learning/29-shape-aware-verification.md).

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
measurement. The [portable report](evidence/updates/2026-09-25-canonical-paper-v0001/experiments/2026-09-25-native-graph-oracle.json)
and [plan](evidence/updates/2026-09-25-canonical-paper-v0001/experiments/2026-09-25-native-graph-oracle-plan.md) preserve its
scope and environment.

### 5.5 A stronger but unpromoted research candidate

After the accepted 2,000-update screen, the
`bilinear-budget8000-replication-v1` experiment fitted an additional symmetric
residual scorer \(A\in\mathbb R^{2\times256\times256}\) to each of two frozen
parent seeds. It left each parent's 93 tensors unchanged and trained the two
dimension-specific residual matrices from zero with a fresh optimizer for
8,000 updates per child. Each
fresh child passed its predeclared validation gate: seed 1730 reached 220/222
(8 gains, 0 losses over its 2,000-update comparator), and seed 1731 reached
214/222 (6 gains, 0 losses). Together the two fresh runs scored 434/444; adding
historical development seed 1729 gives 650/666. These reuse the same 222
questions. All remaining errors are dual-`TYPE` queries.

This is accepted validation evidence for a research candidate, not a change to
the retained decoder or default inference. The ordinary decoder and standard
serving loader do not apply the added residual, so these scores are not a
serving result. No protected test, oracle parity, serving promotion, or durable
weight backup follows. The first v1 audit failed on set-to-JSON serialization
before loading checkpoint payloads; versioned v2 audits repaired evidence
processing without retraining, and the original failure is preserved. The
[portable campaign report](evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication.json),
[evidence-repair declaration](evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-replication-evidence-repair.md),
[Lesson 48](evidence/updates/2026-09-25-bilinear-budget8000-replication/learning/48-testing-a-budget-across-parent-seeds.md), and
[model register](evidence/updates/2026-09-25-bilinear-budget8000-replication/model-versions.md) keep the candidate, verifier, and
checkpoint identities distinct. The separate
[model-version inventory](evidence/updates/2026-09-25-bilinear-budget8000-replication/experiments/2026-09-25-bilinear-budget8000-replication-model-version-inventory.json)
records the two new checkpoint byte hashes, each parent hash, 8,000 residual
updates, and the absence of a durable weight archive.

For the residual scorer, each frozen parent has normalized product features
\(E^{(r)}\in\mathbb R^{1025\times256}\), with row norm 16. Each seed-specific
fit adds \(A^{(r)}\in\mathbb R^{2\times256\times256}\), one matrix per
dimension, and scores
\[
S_d^{(r)}=\frac{A_d^{(r)}+(A_d^{(r)})^\top}{2},\qquad
z_{sdi}^{(r)}=z_{sdi}^{(r),\mathrm{parent}}
 +\frac{(E_s^{(r)})^\top S_d^{(r)}E_i^{(r)}}{16}.
\]
The initial \(A^{(r)}=0\) preserves the parent's scores; the experiment trains
only this residual. The ordinary model forward has no corresponding residual
application, which is why the candidate remains outside standard serving.

## 6. Limitations and remaining work

1. **Oracle parity is not achieved.** The best accepted pair-composition
   validation result misses 97 of 666 model-query outputs. The separate
   bilinear candidate misses 16 of 666 observations over the same queries, but
   is not a retained or served policy.
2. **Validation is not protected testing.** The experiments repeatedly use the
   same 222 queries for adaptive development. The 191-query test partition is
   still protected; the reported candidate results do not establish its
   accuracy.
3. **The graph is small and task-specific.** It contains 1,025 product IDs and
   two fixed dimensions. This does not establish performance on new products,
   changed graph snapshots, other domains, or uncomputable relations.
4. **Matched-quality serving superiority remains unmeasured.** The available
   single-host serial comparison is nonmatched on quality and favors the graph
   oracle on both exactness and measured throughput. HTTP integration checks
   verify behavior and numerical agreement for the tested contract, but there
   is no matched-quality latency, concurrency, energy/request, or production
   SLO claim in this revision. Native HTTP batch behavior is limited; the
   separate vLLM/Phase-D serving path is not the evidence reported here.
5. **Candidate weights are not durably archived.** Checkpoint hashes and local
   receipts identify saved artifacts but do not make the ignored local weight
   files publicly retrievable or backed up. No DOI or external weight archive
   is established.
6. **Exact membership is difficult for dual-`TYPE` queries.** The residual
   candidate's remaining errors concentrate there; changing thresholds or
   claiming a separated ranking is not the same as obtaining exact answers.

These limitations keep the initial result a systems-and-protocol study. They
also define the current research direction: pursue oracle parity within the
Pokémon realm, then measure serving only at matched quality and with durable,
versioned artifacts.

## 7. Revision and identity ledger

The paper revision is not a model release. The identities below are deliberately
separate:

| Layer | Identity for this paper |
| --- | --- |
| Paper | `PAPER-v0001`, dated 2026-09-25; no predecessor |
| Protocol/data | `pokemon-v1`; National Dex corpus snapshot `pokemon_v1_f1541479_20260924` |
| Implementation source | Code and configs cited here are from source commit `cca97ffc5491eb1cfa28ac94532d34460a75812e` |
| Reference checkpoints | `national_dex_continuation_control_s1729_v1`, `_s1730_v1`, `_s1731_v1`; hashes and resolved-config hashes are in [model-versions.md](evidence/updates/2026-09-25-bilinear-budget8000-replication/model-versions.md) |
| Inference policy | Ordinary single-path generation is the default; `+first4-pair6-symmetric-set-logit-sum-v1` is an opt-in policy |
| Pair-composition evaluator | `plm-pair-composition-shape-aware-v2` |
| Bilinear experiment/campaign | `bilinear-budget8000-replication-v1`; two audited candidate children, not promoted |
| Bilinear evaluator | `plm-bilinear-budget8000-replication-v1`; report, audit, and aggregation identities remain separately hashed |
| Publication | Assigned by the dedicated publication receipt; it is not a paper, checkpoint, inference, or evaluator identifier |

The version and historical-evidence index is
[`paper-history/README.md`](paper-history/README.md). The PLM-findings root
projection carries this same scientific revision with adapted relative links;
the path mapping is documented in that index. A publication commit or receipt
does not change the evidence cutoff or imply a software release.

## Conclusion

Pika v1 makes a narrow, testable proposal: condition a compact decoder on an
explicit relation dimension and emit only product identifiers for the
corresponding set. The data compiler defines a perfect oracle, the model and
serving contracts are implemented, and set composition improves fixed-split
validation exactness. An independently accepted residual scorer improves that
validation score further, yet remains outside the retained serving model.
Oracle parity and a matched-quality systems advantage remain open.
