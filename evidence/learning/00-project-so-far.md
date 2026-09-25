# PLM/Pika so far: a guided research and ML overview

**Updated: 2026-09-24.** This guide assumes you know Python but are learning ML.
It describes the current implementation and separates it from future experiments.
Read this first, then [Lesson 1: training correctness](01-training-correctness.md),
[Lesson 2: data and evaluation](02-data-and-evaluation.md), and
[Lesson 3: architecture experiments](03-architecture-experiments.md).
Then read [Lesson 4: the first real-data failure](04-first-quality-campaign.md).
Continue with [Lesson 5: improving prompt supervision](05-prompt-supervision.md)
and [Lesson 6: repeating across initializations](06-seed-replication.md).
[Lesson 7: KV caching](07-kv-caching.md) begins the inference optimization work.
[Lesson 8: native serving](08-native-serving.md) connects the model to a verified
HTTP application and explains snapshots, queues and failure handling.
[Lesson 9: sequence failure diagnosis](09-sequence-diagnosis.md) tests whether
the model can continue correctly after receiving an oracle-supplied first ID.
[Lesson 10: checkpoint selection](10-checkpoint-selection.md) compares saved
training stages using complete answers and explains why loss minima can disagree.
[Lesson 11: symmetric relations](11-symmetric-relations.md) adds a structural
learning bias and measures whether its benefits transfer to generated answers.
[Lesson 12: symmetry replication](12-symmetry-replication.md) audits fresh controls
and explains why repetition failures prevent promoting a stronger candidate.
[Lesson 13: target uniqueness](13-target-uniqueness.md) tests a decoding constraint
that forbids repeated target IDs while keeping the trained weights fixed.
[Lesson 14: batched evaluation](14-batched-evaluation.md) accelerates independent
queries, [Lesson 15: first-target guidance](15-first-target-guidance.md) uses
learned membership scores, and [Lesson 16: guidance and uniqueness](16-guidance-and-uniqueness.md)
tests their interaction. [Lesson 17: guided integration](17-guided-integration.md)
connects that experimental policy to the normal application paths.

## 1. The question we are investigating

We are building a small transformer that answers structured relationship queries
by generating product identifiers. Pokemon names are our first product catalog.
Each name is an opaque SKU: an identifier, not a word whose spelling the model
needs to understand. We train the reference model from scratch.

A query is roughly: "Given product A, return products that share its TYPE" or
"return products that share its COLOR." The answer contains identifiers only.
The surrounding application can look those identifiers up and display details.

The first research question is about **systems performance**: can a trained model
match a deterministic reference's answer quality and provide a useful advantage
in concurrent requests, latency, or energy per request? We have not established
that advantage. A database or precomputed lookup is a serious competitor here.

The current reference knows the right answers by construction. It is our
**oracle**. Beating its correctness is not a meaningful goal. More uncertain
relations, such as COMPLEMENTARY, belong to a later experiment about learning
policy. They are not part of today's TYPE/COLOR SAME task.

## 2. The data starts as a graph, not a paragraph

A **graph** contains nodes and edges. Nodes represent products and hidden
attributes; edges connect products to their attributes. An illustrative graph:

```text
PKM_A --HAS_COLOR--> ATTR_RED <--HAS_COLOR-- PKM_B
PKM_A --HAS_TYPE---> ATTR_FIRE <--HAS_TYPE--- PKM_C
```

The query `(PKM_A, COLOR, SAME)` therefore includes PKM_B. The TYPE query includes
PKM_C. We exclude the subject itself. For TYPE, sharing at least one type is
sufficient; we do not require identical sets of types.

The graph lives in **SQLite**, an on-disk relational database. SQL joins follow
the shared attributes to derive answers. The model never receives ATTR_RED,
ATTR_FIRE, HAS_COLOR, or HAS_TYPE as tokens. It sees the steering tokens COLOR
and TYPE plus product IDs. It can learn latent groupings without explicit
attribute-value tokens.

The local data inspected on September 24 contains:

| Item | Count | Meaning |
| --- | ---: | --- |
| Products | 1,025 | Possible entity identifiers |
| Hidden attributes | 28 | Type/color grouping nodes |
| Edges | 2,576 | Product-to-attribute facts |
| Corpus records | 2,050 | One nonempty answer sequence per subject/dimension query |
| Vocabulary rows | 2,049 | Products plus control/steering/reserved positions |
| Longest record | 289 tokens | Fits the configured 512-token context |

There is also a local labeling UI. It edits each product's attribute buckets;
the compiler derives the resulting relationships. This avoids asking a person
to label every possible product pair separately.

## 3. A compiler creates the training examples

A **corpus** is the collection of training examples. Our compiler derives it
from the graph using a versioned **protocol**, the exact allowed token grammar:

```text
BOS SUBJECT DIMENSION SAME ANSWER TARGET... EOS
```

For the toy graph, a record could be:

```text
BOS PKM_A COLOR SAME ANSWER PKM_B EOS
```

BOS marks the start, ANSWER separates the prompt from the completion, and EOS
marks the end. Answers can contain many target identifiers. The compiler orders
them consistently using shared-attribute count, confidence, and product key.

A **tokenizer** maps each symbol to one integer. This is a direct lookup, not the
subword tokenizer used by many text models. Product IDs begin at 1024; lower
positions reserve space for the protocol. Existing product IDs must remain
stable when extending a frozen vocabulary, because an ID selects a learned
embedding row. Renumbering IDs would change what existing weights refer to.

We deliberately keep three different responsibilities separate:

| Store | Responsibility |
| --- | --- |
| SQLite graph | Facts: which products have which attributes |
| Protocol configuration | Rules: how facts become queries and ordered answers |
| Corpus files | Derived examples, vocabulary, and manifest |

A **manifest** records counts and content hashes. A hash fingerprints content
so we can detect mismatches. **Provenance** records where that content came from.
These solve related but different problems: unchanged bytes can still come from
an inadequately documented source.

We created the versioned snapshot `pokemon_v1_f1541479_20260924` using the pinned
PokeAPI revision and source hashes. Its records and vocabulary match the earlier
corpus, with corrected graph identity and provenance. The old dataset remains
preserved. The snapshot command verifies existing artifacts or reconstructs a
receipt-only checkout, requiring exact hash agreement before publication.

## 4. What the transformer does with those integers

An **embedding** is a learned vector for a token. The configured reference model
uses vectors of length 256 and eight decoder blocks. For an illustrative batch
of two sequences padded to seven positions:

```text
input IDs:          [2, 7]         integers
embedding table:    [2049, 256]    learned floating-point values
embedded inputs:    [2, 7, 256]
decoder output:     [2, 7, 256]
output logits:      [2, 7, 2049]   a score for each possible next token
```

The **decoder** transforms each position using its preceding context. In
**self-attention**, positions compute weights describing how much to use other
visible positions. A simplified attention equation is:

```text
attention(Q, K, V) = softmax(Q @ K.T / sqrt(head_dim) + causal_mask) @ V
```

Q, K, and V are learned projections called queries, keys, and values. Here V
means attention values, not vocabulary size. The causal mask blocks future
positions. In the eight-head configuration, each query head has dimension 32
because `256 / 8 = 32`. **Grouped-query attention (GQA)** lets those eight query
heads share two sets of key/value heads. That sharing is implemented; a serving
speed or memory advantage for this project still needs measurement.

Other implemented components have specific jobs:

- **RoPE** represents position by rotating query/key components.
- **RMSNorm** rescales activations using their root-mean-square magnitude.
- **SwiGLU** is the gated feed-forward transformation inside each block.
- **Residual connections** add a block's transformation back to its input.
- **Tied embeddings** reuse the input embedding weights for output scoring.

These are architectural choices, not evidence that this model is state of the
art. Optional activation checkpointing and sparse mixture-of-experts now work.
We measured their training cost, and repaired an error in RoPE coordinate pairing.
The dense reference remains the default. RoPE scaling, muP and optimized serving
remain future work; see Lesson 3 for equations and the first measurements.

## 5. Training is an optimization problem

The model converts logits into probabilities and is penalized for assigning
low probability to the correct next token. In notation:

```text
L = mean(-log P(correct next token | preceding tokens))
```

**Backpropagation** computes gradients: how changing each weight would change
the loss. An **optimizer**, AdamW here, uses gradients and accumulated statistics
to update weights. A **learning rate** controls the update scale; a **scheduler**
changes that rate over training. A batch groups examples, and gradient
accumulation combines several batches before an optimizer update.

The model is trained with **teacher forcing**: the true earlier answer tokens
are supplied. During generation it instead appends its own predictions. This
difference makes prompt-only generation an essential check.

On September 24, we found and repaired two correctness issues:

1. **Wrong target alignment.** The old loss rewarded reconstructing a token
   already visible at that position. We shifted the loss alignment so ANSWER
   predicts the first target, and each target predicts the following token.
   Prompt and padding targets remain excluded from scoring.
2. **Incomplete resume.** The old trainer restored weights and optimizer state
   but restarted the dataset at its first batch. It now restores the correct
   batch offset and keeps loader randomness separate from model randomness.

The tiny overfit gate now requires correct completion generated from the prompt,
as well as low loss. New checkpoint metadata identifies the training objective
and batch-order contract; the trainer rejects older incompatible checkpoints.

The correctness iteration passed **115 tests**, including CPU and GPU resume
comparisons, dropout, accumulation, and a Windows data-loader worker. Subsequent
iterations add generation, provenance and architecture checks. This is evidence for
the tested mechanics. It does not prove that the model learned the National Dex
relationships or that it will generalize.

## 6. How we will judge learning

We split complete `(subject, dimension)` queries into three groups:

| Partition | Purpose | Current local count |
| --- | --- | ---: |
| Train | Update model weights | 1,637 |
| Validation | Compare choices during development | 222 |
| Test | Evaluate the selected result at finalization | 191 |

Assignment uses a seeded hash, so counts are approximate fractions rather than
exact quotas. The same configuration reproduces the same assignment. A held-out
query is not necessarily an unseen product: its entities can appear in other
queries. This experiment does not establish generalization to brand-new SKUs.

Repeatedly choosing changes based on the test result would make the test part
of development. That is why the adaptive path uses validation and there is an
explicit final-test path. It is an experimental discipline supported by API
checks, not a guarantee that arbitrary calling code cannot misuse test records.

A **baseline** is a reference method against which the model is compared:

- The oracle derives the correct relationships from the graph/corpus.
- Popularity ranks frequently occurring training targets within each dimension.
- ComplEx learns entity/relation embeddings to score relationships.

Several answers may be correct. Our shared ranking metrics reflect that:

- **MRR**, mean reciprocal rank, rewards placing a correct answer early.
- **Hits@K** asks whether at least one correct answer appears in the first K.
- **MAP**, mean average precision, also rewards ranking the other correct
  answers highly. For one query with exactly two positives at ranks 2 and 4,
  reciprocal rank is `1/2`; average precision is `(1/2 + 2/4) / 2 = 1/2`.

The PLM scorer ranks candidates using logits immediately after ANSWER. The CLI
now additionally generates whole answers and measures precision/recall, exact
sets/sequences, valid termination, duplicates and self-return. We removed the
unmeasured `protocol_valid=True` default. First-token ranking, complete-answer
quality, latency and energy remain distinct measurements. The protocol mask
enforces allowed syntax; it cannot establish correct relationships.

## 7. What has been built, and what comes next

| Stage | What exists | What it establishes |
| --- | --- | --- |
| Foundation | Environment, schema, configs, tokenizer | A consistent development substrate |
| Graph/corpus | Importer, compiler, labeling, splits, baselines | The data and evaluation building blocks |
| Reference model | Dense decoder, optional MoE/checkpointing, corrected mechanics | A testable training implementation and measured feature costs |
| Experimentation | Identity, receipts, comparison, lineage helpers | Infrastructure for documenting experiments |
| Serving | Native HTTP API, snapshot validation, output policy and bounded execution | Verified application wiring; performance comparisons remain open |

**Hydration** means looking up generated IDs to recover display information.
The full intended application is parse request -> tokenize -> generate IDs ->
parse/validate output -> hydrate. That end-to-end server now reproduces all 222
archived raw validation responses exactly. Filtering reproduces 105/222 exact
answers for the anchor checkpoint; the remaining learned errors persist.

**AVO**, Agentic Variation Operator, refers to a proposed evidence-guided
improvement loop: propose a candidate change, evaluate it, compare under declared
rules, and retain its lineage. The repository has supporting record types and
comparison helpers. This is not yet evidence of a completed autonomous campaign.

The dataset refresh, feature microbenchmark and first quality campaign are
complete. Dense and four-expert MoE each trained for 2,000 updates; validation
generation achieved roughly 28% macro F1 and zero exact sets out of 222 queries.
Both generated all 16 sampled training answers exactly. Low token loss hid a
first-target generalization failure, while ComplEx ranking was strong across
three seeds. Prompt-conditioned set supervision raises mean raw generated F1
from 23.13% to 60.62% across three paired seeds on the same split. With the same
deterministic self-exclusion/deduplication policy, mean exact-answer accuracy
increases from 15.62% to 45.80%; raw exact accuracy remains zero. The improvement
repeats across these initializations, but 85 validation queries are never exact
in any improved-model seed. Oracle-quality parity, broader robustness, final-test
selection and measured serving remain unachieved. Lessons 4–6 explain the
failure, improvement and replication.

Native KV caching now preserves all 222 anchor-model validation responses and
shows about 4% less time in a small paired generation benchmark. It remains
opt-in. Native HTTP serving is implemented and verified; concurrency and oracle
performance comparisons are still open. Lesson 7 explains why fewer computations
gave only a modest speedup; Lesson 8 follows the request through the application.

A later diagnostic supplies exactly the correct first answer ID: processed
exact answers rise from 105 to 174/222, while all 48 remaining failures are TYPE.
This is evidence of a first-decision bottleneck and residual continuation errors,
not improved unassisted model quality. Lesson 9 explains the intervention,
teacher-forced accuracy and the concentration of failures in sparse answer families.

A four-checkpoint comparison retains step 2000: both it and step 1500 yield
105/222 processed exact answers, while final F1 is higher. The minimum auxiliary
set loss occurs at step 1000, which produces only 76 exact answers. Lesson 10
explains selection rules, differing objectives and why this does not improve
the already retained model or establish untouched final-test performance.

The next candidate adds a symmetric relation loss through the shared embeddings.
In one controlled seed, raw generated F1 improves from 62.90% to 83.41% and
processed exact answers from 105 to 143/222. This is actual unassisted generation,
unlike the oracle-hint diagnosis. The
direct classifier's higher F1 does not establish equal generation performance;
Lesson 11 separates these output paths and explains the remaining gap.

With fresh controls on identical source, three-seed mean raw F1 improves from
60.78% to 79.87% and processed exact accuracy from 45.80% to 61.11%. Every pair
improves. Replication also exposes one invalid, non-terminating answer in each
additional candidate seed; the refreshed controls have one unfinished answer.
Both candidate failures consume 507 tokens while repeating a small set of IDs. A separate
membership classifier nearly solves one of those queries, demonstrating that
knowing much of the relation does not guarantee successful sequential output.
The candidate fails the predeclared serving promotion gate. Lesson 12 records
the paired comparison, an unsuccessful historical-control replay, the refreshed
controls and the termination diagnosis; the existing serving reference is retained.

An optional target-uniqueness decoder then makes all 666 candidate responses
terminate validly, but yields no additional exact answers. One seed's processed
F1 drops slightly, so its declared promotion criterion fails and the flag stays
off. Lesson 13 explains why preventing repetition can redirect a sequence into
a different wrong answer. A separate eight-query batching prototype matches
16 serial outputs across both policies; it is not yet an admitted evaluator or
serving backend, and no speed claim has been measured.

That prototype is subsequently integrated as an optional offline evaluator in
[Lesson 14](14-batched-evaluation.md). It preserves all 1,332 serial outputs
across three frozen checkpoints and both decoding policies. Three paired timings
of eight fixed queries yield 5.31x throughput: median total time falls from
7.38 to 1.39 seconds, while peak allocated GPU memory rises from 36.35 to
51.71 MiB. This makes further experiments cheaper to run; it does not solve the
remaining quality problem or measure HTTP performance. The lesson explains the
batch axis, independent caches, finished rows and throughput versus latency.

[Lesson 15](15-first-target-guidance.md) uses the learned relation head during
the first decoding step. All three seeds improve: at strength 16, mean processed
F1 rises from 80.25% to 88.33%, and exact answers from 61.11% to 68.17%. The
strongest setting gains 48 exact answers and loses one across 666 executions.
Both existing repetition failures remain exactly unchanged, so every tested
strength fails the declared integration gate. This teaches the difference
between learning membership, choosing an ordered starting point, and completing
an answer. Independent review and focused test improvements ran in parallel
with the results analysis; no serving promotion follows from these results.

[Lesson 16](16-guidance-and-uniqueness.md) tests guidance and uniqueness together.
Strength 16 plus uniqueness preserves 454/666 exact answers (68.17%), reaches
88.40% mean processed F1, and makes all 666 outputs valid, terminated and
repeat-free. The combination passes its declared candidate gate and is selected
for integration verification; this does not change the old standalone failures
or promote the service. Parallel residual analysis shows the remaining gap is
mostly TYPE coverage: only 153/357 TYPE observations are exact, compared with
301/309 COLOR. The lesson explains interactions, evidence reuse and why clean
completion is different from correct enumeration.

[Lesson 17](17-guided-integration.md) moves the selected policy into shared
generation, evaluation and native HTTP through `eval.first_target_guidance_alpha`.
It defaults to zero. A positive value requires constrained cached decoding and
a model with the symmetric relation head; it changes inference scores, not the
checkpoint's learned weights. Serial generation uses a `[1, 5]` prompt and
offline batch size eight uses `[8, 5]`. The same head produces product scores
of shape `[B, 1025]`, applied only at the first generated position.

Native HTTP still processes one model request at a time. Full real-data token
replay now matches all 1,332 archived answers. All 666 real HTTP responses also
match full tokens and metrics, and every temporary server stopped cleanly.
The recorded candidate quality remains 454/666 exact answers; improving TYPE
coverage and reaching oracle parity remain open research work.

[Lesson 18](18-union-coverage.md) narrows the remaining quality problem to
union coverage. Single-TYPE answers are 124/153 exact, dual-TYPE 29/204, and
COLOR 301/309 across the three seeds. Seventy-five failing dual-TYPE answers
return exactly one complete branch. The symmetric head recognizes 99% of the
omitted dual-TYPE members, but its own thresholded sets are rarely exact.
Teacher-prefix probes show weak early decisions and strong later continuation
when given correct context. This selects supervision of the remaining set at
early continuation states as the next experiment.

[Lesson 19](19-continuation-set-training.md) implements and tests that objective.
It reuses existing weights at three early positions, adds class-balanced binary
loss, and compares six fresh runs with the same training budget and inference
policy. Mean F1 improves 90.12% -> 91.52%, but exact answers fall 466/666 ->
463/666, with dual TYPE falling 34/204 -> 32/204. The gate fails, so the feature
remains experimental and defaults to zero. All three candidates improve
teacher-forced token loss; this does not guarantee better free generation.
The lesson includes equations, tensor shapes, concrete omissions/additions,
independent evidence checks and the observed increase in training cost.

[Lesson 20](20-generated-versus-teacher-prefixes.md) compares frozen generated
and teacher prefixes across all six models. All 7,980 sampled next choices
match archived generation. The candidate's early dual-TYPE set head is much
stronger with teacher prefixes (94–95% F1) than generated ones (75–77%). A
strong set prediction can also coexist with a wrong next choice on an identical
correct prefix. The diagnosis separates those two problems and motivates the
next ordered-start loss experiment; it changes no weights or generation results.

[Lesson 21](21-weighting-the-first-choice.md) completes the ordered-start trial.
Increasing first-target loss weight to eight improves final-training-batch fit
but worsens held-out first-token CE in every seed. Generated exact answers are
464/666 versus 466/666 for the original control, and dual-TYPE answers decline
to 31/204. The gate fails. This gives us a concrete generalization lesson:
emphasizing a training decision does not guarantee better held-out decisions,
and choosing a valid first member does not guarantee a complete enumeration.

[Lesson 22](22-first-choice-paths.md) changes decoding while freezing the model.
Four first-choice paths contain 501/666 exact answers, up from greedy 466/666,
but their sum/mean probability selectors return only 462/465. Neither gate
passes. All 666 greedy replays match exactly; independent checks cover all
2,664 paths. This separates finding useful alternatives from recognizing them.
The next declared test uses the learned membership head to rank these same sets.

[Lesson 23](23-ranking-candidate-sets.md) completes that test successfully.
The learned head assigns a membership logit to every product; adding the logits
of a candidate's members ranks whole sets. It selects all 501 available exact
answers, with 35 gains and zero exact-answer losses versus greedy. Mean F1 rises
from 90.12% to 95.56%. There are still 165 observations without an exact candidate,
and eight where another candidate has better F1. This is an audited offline
result; reproducing it through normal generation and native serving comes next.

[Lesson 24](24-integrating-set-selection.md) implements that shared opt-in path.
All 554 baseline tests and all three complete offline checkpoint replays pass: 2,664
candidate paths/scores, 666 selected outputs and 666 disabled greedy outputs
match their references. All 666 native HTTP responses and failure contracts also
match, and the independent audit accepts the opt-in application path.
A separate saved-path diagnosis finds 69 remaining failures with an exact pair
union available, motivating a declared composition experiment without changing
the current integration's checks or claiming that unions are model-emitted paths.

[Lesson 25](25-composing-candidate-sets.md) completes the pair-composition test:
569/666 exact sets versus 501, with 68 gains, zero exact losses and mean F1 of
96.69%. All gains are dual TYPE; single TYPE and COLOR stay unchanged. The
independent audit verifies all 6,660 slots and preserves each composition's raw
source paths. One PILOSWINE choice is worse in F1 and misses an available exact
pair. There are 97 inexact observations remaining. This result is offline;
the application contract and verification status follow in lesson 26.

[Lesson 26](26-integrating-composed-sets.md) implements that separate set API and
report contract. All 661 tests pass after incorporation; its fresh three-checkpoint
offline replay passes. The first HTTP checkpoint fails exact score equality,
although all answers and paths match. The original four paths remain explicit evidence, and no
composed sequence/EOS event is manufactured.

[Lesson 27](27-remaining-errors-and-score-signs.md) diagnoses the 97 remaining
saved errors: 53 lack source coverage, 43 have coverage but no exact candidate,
and one misses an available exact candidate. Negative-score deletion would
damage 62 already-exact answers; a deletion-only repair can recover at most 17
existing failures. The resulting upper bound is 524 exact answers, below 569.

[Lesson 28](28-direct-membership-ablation.md) completes the fixed ablation:
select every non-subject product with a positive membership logit. F1 improves
to 98.96%, but exact answers fall from 569 to 339, with three gains and 233 losses.
The declared gate fails. The lesson derives the rule and tensor shapes and
explains why coverage gains alone cannot justify unrestricted set prediction.

## Where to read the code

| Question | Starting point |
| --- | --- |
| How do graph facts become sequences? | `src/plm/corpus/compiler.py` |
| Where are symbols mapped to IDs? | `src/plm/protocol/tokenizer.py` |
| How does the model compute predictions and loss? | `src/plm/model/layers.py` |
| How does training advance and resume? | `src/plm/training/trainer.py` |
| How are predictions scored? | `src/plm/evaluation/plm.py` and `src/plm/evaluation/ranking.py` |
| Why were these decisions made? | `research_log.md` |

For a useful self-check, explain why a model could achieve very low loss under
the old implementation yet fail when given only a prompt. Then explain why
restoring the same random seed is not the same as restoring RNG state midway
through training. Both answers connect the implementation to the experiment.
