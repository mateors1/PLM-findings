# PLM findings

Research synthesis for **PLM / Pika Edition**, a decoder-only model that emits product identifiers for structured relation queries. Pokémon identifiers are the first opaque product catalog. This repository is a dated research snapshot of the source project's working tree, not a software release.

## Current model size and retained techniques

**Verified 2026-09-25.** The measured family uses the dense `tiny_decoder`: **8 transformer layers**, width **256**, **8 query heads / 2 KV heads** (head width 32), SwiGLU hidden width **768**, a **2,049-entry vocabulary**, and a **512-token context**. The catalog contains 1,025 Pokémon entities.

| Component / variant | Unique learned parameters |
| --- | ---: |
| Dense decoder with tied input/output embeddings | 6,558,720 |
| Prompt-set projection, `[256,256]` | +65,536 |
| Symmetric relation projection, `[256,256]` | +65,536 |
| **Parent model used by the verified composition application** | **6,689,792 (~6.69M)** |
| New per-dimension bilinear residual, `[2,256,256]` | +131,072 |
| **Latest bilinear research derivative, including frozen parent** | **6,820,864 (~6.82M)** |

Counts were checked by loading the local parent checkpoint on CPU, reconstructing its recorded model config, and counting unique model parameters. Tied embeddings count once; optimizer state and positional buffers are excluded. The derivative adds only the stated residual. During that refit, **131,072 parameters are trainable** and the original 93 saved tensors remain unchanged. The latest derivative is a research variant, not a promoted serving default.

The main techniques retained so far are:

- **Dense decoder architecture:** pre-norm RMSNorm, RoPE, grouped-query attention with QK normalization, SwiGLU, and tied embeddings. The measured model is distinct from the larger `overkill` configuration and separate MoE trials.
- **Protocol and output constraints:** a subject/dimension/`SAME` prefix, an `ANSWER` loss boundary, legal entity-ID/EOS decoding, and deterministic parsing, post-processing and hydration. Attribute values remain outside the model vocabulary.
- **Auxiliary supervision:** masked causal next-token loss plus prompt-set membership supervision and a shared-embedding symmetric relation objective. The parent uses AdamW with BF16 mixed-precision training; its learned weights are stored in FP32.
- **Verified autoregressive inference variants:** KV caching, first-target relation guidance, candidate branching, and learned set selection/composition. These are separately configured inference procedures, not extra transformer parameters. See the [set study](papers/03-set-prediction.md) and [serving study](papers/04-serving-and-reproducibility.md).
- **Newest research direction:** freeze the parent and fit a symmetric bilinear residual for TYPE/COLOR using query-balanced positive/negative binary cross-entropy. Its dedicated scorer predicts membership from frozen embeddings and learned relation matrices; it does **not** apply the residual through ordinary autoregressive decoder forward calls. Its accepted validation screen is separate from application integration, protected-test evaluation and serving promotion.

The architecture and auxiliary-head design are documented in the copied [architecture](evidence/source/architecture.md), [prompt-set lesson](evidence/learning/05-prompt-supervision.md) and [symmetric-relation lesson](evidence/learning/11-symmetric-relations.md). This size snapshot identifies parent checkpoint `e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1` and local `bilinear-budget8000-v1` derivative `6979013750eb8f2780917f714192850aae411a48490757236a891fea7e351f91`. These hashes identify local artifacts; they do not imply that weights are published or backed up.

## What the research aims to prove

PLM tests whether a constrained vocabulary and an explicit query protocol can support reliable **relation-to-ID retrieval**. A deterministic pipeline parses a request, supplies the model with a subject and relation, validates its generated product IDs, and hydrates those IDs outside the model. The model does not generate prose or attribute descriptions.

The current controlled experiment uses `TYPE` and `COLOR` with the `SAME` relation. A graph lookup can compute every correct answer, so the goal is to **match that oracle's complete-answer quality** and then measure whether model serving offers an advantage in concurrent throughput, latency, or energy per request **at matched quality**. A later experiment would ask whether a learned policy helps with relations that cannot be fully compiled from fixed graph rules, such as complementary products. The current findings establish neither oracle parity nor a quality-matched serving advantage, and they do not test that later learning claim. See the [research question and protocol](papers/01-research-question-and-protocol.md) and [evidence ledger](papers/05-evidence-ledger.md).

## Possible applications

- **Catalog relation queries:** return product IDs that share a specified attribute with a subject. This is the present test case; exact graph lookup remains the correct baseline for deployment.
- **Human-reviewed candidate discovery:** propose related IDs for a curator or merchandiser to assess. This could tolerate incomplete suggestions, but no user-workflow benefit has been measured.
- **Learned complementary or preference retrieval:** explore relationships that require judgments beyond deterministic attribute matching. This is a future hypothesis requiring new labels, careful splits, and comparisons with simpler retrieval methods.
- **High-volume structured services:** serve bounded, ID-only responses for many requests. Offline batching shows capacity in a fixed workload; live concurrency, latency targets, energy, and quality-matched comparisons remain open.

These are application directions, not validated product claims. The [architecture and applications paper](papers/13-decoder-and-applications.md) details the fit and evidence boundary for each.

## Read the findings

1. [Research question and protocol](papers/01-research-question-and-protocol.md) — what the experiment can establish.
2. [Learning and quality results](papers/02-learning-and-quality.md) — the measured progression and failed interventions.
3. [Set prediction and failure analysis](papers/03-set-prediction.md) — why complete answers remain difficult.
4. [Serving and numerical reproducibility](papers/04-serving-and-reproducibility.md) — what application and performance tests established.
5. [Evidence ledger and next experiments](papers/05-evidence-ledger.md) — claim status, source map, and acceptance work still open.
6. [Academic versioning and the next hypothesis](papers/06-academic-versioning-and-next-hypothesis.md) — 23 final-checkpoint identities, remaining archival work, and learned relational retrieval.
7. [Rank ordering and threshold limits](papers/07-ranking-and-threshold-limits.md) — why adding larger unions or tuning cutoffs alone cannot exceed the current selector's exact-set count on frozen evidence.
8. [A failed hardest-boundary margin](papers/08-hardest-boundary-margin-failure.md) — a fresh paired training screen regressed from 192 to 6 exact answers, with both checkpoints and the failed gates preserved.
9. [Gradient strength and interference](papers/09-gradient-strength-and-interference.md) — fixed training-query probes reveal substantial gradients and local opposition, with explicit limits on causal and optimizer claims.
10. [A weaker margin gives a mixed result](papers/10-weaker-margin-mixed-result.md) — exact answers improve from 191 to 194, but separation falls from 168 to 164; the unchanged gate rejects the variant.
11. [Candidate availability and selection](papers/11-candidate-availability-and-selection.md) — both saved selectors find every available exact answer; missing candidates limit the remaining exactness.
12. [FP8 inference screen](papers/12-fp8-inference-screen.md) — eager FP8 kept similar validation quality but was much slower on the measured native path.
13. [Decoder-only architecture and applications](papers/13-decoder-and-applications.md) — why this reference model generates IDs sequentially, and which uses remain hypotheses.

The [`evidence`](evidence/) directory preserves the 2026-09-24 learning notes, experiment plans and portable reports, figures, and the National Dex manifest from the source checkout. The evidence files are copied without editorial changes. Research papers cite those local copies; consult them for methods, exact run identities and detailed measurements. The evidence notes' links into `src/` refer to the separate [PLM source repository](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition) and may not resolve here.

## Reading the numbers

The main quality comparisons reuse **222 validation queries** under each of three training seeds. Thus **666 query-seed observations** represent repeated model evaluations on the same queries, not 666 independent held-out queries. The protected final-test partition has **191 queries** and has not been used to support the findings here. Validation has guided many choices, so its results are developmental evidence. The graph/compiler oracle is correct by construction for the scoped task; a model result below 100% exactness is below oracle parity.

The composition result is **569/666 exact sets** with **96.69% macro F1**, now preserved by the accepted shape-aware application verification. The first application campaign failed exact HTTP score parity because FP32 scores changed with batch shape; that failed gate remains failed. The revised contract uses exact same-shape score references and unchanged cross-shape decisions, with explicit mixed fresh/reused evidence. No matched-quality concurrent throughput or energy advantage is established.

## Snapshot and reuse

The [2026-09-25 update](evidence/updates/2026-09-25/README.md) adds a historical
checkpoint inventory, the revised verification contract, a completed native
concurrency probe and the future learned-retrieval hypothesis. Original evidence
copies remain unchanged; the addition has its own checksum inventory.

The later [rank-diagnosis follow-up](evidence/updates/2026-09-25-ranking-diagnosis/README.md)
adds an audited 479/666 exact-recovery bound for per-query scalar thresholds on
the saved membership scores, and the proof that triple/four-way unions create
no new exact-answer availability. These are diagnostic limits, not new policies.

The [completed composition verification](evidence/updates/2026-09-25-composition-verification/README.md)
records 666 HTTP responses, fresh serial references, independent audit and
separate acceptance. Earlier dated copies retain their original pending status.

The [margin training screen](evidence/updates/2026-09-25-margin-screen/README.md)
records an independently audited negative result and a new 25-checkpoint
inventory. The failed variant was stopped before additional-seed replication;
the earlier 23-checkpoint inventory remains unchanged.

The [gradient-diagnosis addition](evidence/updates/2026-09-25-gradient-diagnosis/README.md)
records 15 audited observations on existing parameter states and 96 fixed
training queries. It examines gradient strength, alignment and radial score
scaling without training another model or changing the rejected margin gate.

The [weaker-margin screen](evidence/updates/2026-09-25-weak-margin-screen/README.md)
records a separately declared .001-coefficient intervention with fresh paired
training. Complete-answer quality improved, but the required separation gain
failed. The audited result is retained without additional-seed replication
or promotion, together with a new 27-checkpoint inventory.

The [candidate-score diagnosis](evidence/updates/2026-09-25-candidate-score-decomposition/README.md)
reconstructs all four fixed pool/score combinations and separates candidate
availability from selection errors. It adds no checkpoint or promoted policy.
The same dated addition includes the future learned-retrieval data review.

The [offline batch-saturation sweep](evidence/updates/2026-09-25-batch-saturation/README.md)
reached 30,980 completed output tokens/s and 188.3 completed requests/s at
batch 640 on one RTX 5070 Ti. This fixed-group decoder result does not change
the measured serial HTTP throughput or establish GPU arithmetic saturation.

The [eager FP8 screen](evidence/updates/2026-09-25-fp8-screen/README.md)
retained similar validation quality on one guided checkpoint but reduced serial
native throughput from 148.5 to 12.0 completed output tokens/s. The compiled
FP8 follow-up is a declared experiment without a result in this snapshot.

Source checkout: `mateors1/Plm-protocolized-language-model-pika-edition`, `HEAD` `8afff361447b1357a041aa65f5b588af5b7ba5b5`; the copied source material also contained uncommitted changes. The copy was made on 2026-09-25 from the 2026-09-24 research state. [`evidence/SHA256SUMS.txt`](evidence/SHA256SUMS.txt) records the copied file hashes, allowing exact verification without pretending that the source `HEAD` alone identifies this snapshot. The papers are interpretive syntheses; the evidence copies remain the detailed record.

This public repository includes research material copied from a source working
tree with uncommitted changes. The dated evidence inventories identify the copied
bytes; source `HEAD` alone does not identify them. Results and status statements
should be read with their original dates and declared acceptance gates.

## Pokémon names and IP

Pokémon and individual Pokémon names appear here as reference labels for a research corpus. The model treats names as opaque `PKM_<NAME>` identifiers to study relational retrieval and serving, rather than as characters in a game or story. This is independent research, with no affiliation with or endorsement by the owners of Pokémon or related marks. No rights in those marks are claimed.

The intended rationale is limited, referential use of names to identify the benchmark entities and make the experiments reproducible. Under U.S. guidance, individual names are not protected by copyright, although names may be protected as trademarks ([U.S. Copyright Office](https://www.copyright.gov/help/faq/faq-protect.html); [USPTO](https://www.uspto.gov/trademarks/basics/what-trademark)). Copyright fair use and trademark defenses depend on the specific use and jurisdiction; this notice is not a legal determination or permission to reuse other Pokémon material ([U.S. Copyright Office fair-use guidance](https://copyright.gov/fair-use/)).

## Support

To support the maintainer, [sponsor @mateors1](https://github.com/sponsors/mateors1).
The repository's GitHub Sponsor button is configured in [`.github/FUNDING.yml`](.github/FUNDING.yml).
