# Reader guide to the PLM snapshot

[Back to the overview](../README.md) · [Full paper catalog](../papers/README.md)

This guide is an editorial wrapper around the archived research. It changes no experimental result, acceptance gate or archived file. Read dated status statements as a record of what was known then.

## Navigating archived links

The copied notes retain paths from the source checkout. A broken relative link does not imply that every target is missing: many learning notes are public here, while the full implementation and run artifacts are not.

Start with the [protocol](../evidence/source/protocol-spec.md), [data contracts](../evidence/source/data-contracts.md), [learning overview](../evidence/learning/00-project-so-far.md), [evidence inventory](../evidence/README.md), or [claim ledger](../papers/05-evidence-ledger.md). For the application acceptance status, use the later [composition verification and acceptance](../evidence/updates/2026-09-25-composition-verification/README.md).

### Architecture note: public lesson links

The [archived architecture note](../evidence/source/architecture.md) contains 24 `learning/...` links that resolve incorrectly from its copied location. Use these public copies instead. They retain their original dates, including historical pending and failed statuses.

| Path written in the archive | Public copy |
| --- | --- |
| `learning/04-first-quality-campaign.md` | [04 first quality campaign](../evidence/learning/04-first-quality-campaign.md) |
| `learning/03-architecture-experiments.md` | [03 architecture experiments](../evidence/learning/03-architecture-experiments.md) |
| `learning/01-training-correctness.md` | [01 training correctness](../evidence/learning/01-training-correctness.md) |
| `learning/05-prompt-supervision.md` | [05 prompt supervision](../evidence/learning/05-prompt-supervision.md) |
| `learning/06-seed-replication.md` | [06 seed replication](../evidence/learning/06-seed-replication.md) |
| `learning/12-symmetry-replication.md` | [12 symmetry replication](../evidence/learning/12-symmetry-replication.md) |
| `learning/11-symmetric-relations.md` | [11 symmetric relations](../evidence/learning/11-symmetric-relations.md) |
| `learning/13-target-uniqueness.md` | [13 target uniqueness](../evidence/learning/13-target-uniqueness.md) |
| `learning/15-first-target-guidance.md` | [15 first target guidance](../evidence/learning/15-first-target-guidance.md) |
| `learning/16-guidance-and-uniqueness.md` | [16 guidance and uniqueness](../evidence/learning/16-guidance-and-uniqueness.md) |
| `learning/17-guided-integration.md` | [17 guided integration](../evidence/learning/17-guided-integration.md) |
| `learning/18-union-coverage.md` | [18 union coverage](../evidence/learning/18-union-coverage.md) |
| `learning/19-continuation-set-training.md` | [19 continuation set training](../evidence/learning/19-continuation-set-training.md) |
| `learning/20-generated-versus-teacher-prefixes.md` | [20 generated versus teacher prefixes](../evidence/learning/20-generated-versus-teacher-prefixes.md) |
| `learning/07-kv-caching.md` | [07 kv caching](../evidence/learning/07-kv-caching.md) |
| `learning/14-batched-evaluation.md` | [14 batched evaluation](../evidence/learning/14-batched-evaluation.md) |
| `learning/08-native-serving.md` | [08 native serving](../evidence/learning/08-native-serving.md) |
| `learning/21-weighting-the-first-choice.md` | [21 weighting the first choice](../evidence/learning/21-weighting-the-first-choice.md) |
| `learning/22-first-choice-paths.md` | [22 first choice paths](../evidence/learning/22-first-choice-paths.md) |
| `learning/23-ranking-candidate-sets.md` | [23 ranking candidate sets](../evidence/learning/23-ranking-candidate-sets.md) |
| `learning/24-integrating-set-selection.md` | [24 integrating set selection](../evidence/learning/24-integrating-set-selection.md) |
| `learning/25-composing-candidate-sets.md` | [25 composing candidate sets](../evidence/learning/25-composing-candidate-sets.md) |
| `learning/26-integrating-composed-sets.md` | [26 integrating composed sets](../evidence/learning/26-integrating-composed-sets.md) |
| `learning/28-direct-membership-ablation.md` | [28 direct membership ablation](../evidence/learning/28-direct-membership-ablation.md) |

### Architecture note: source-only links

These three linked implementation files are absent from this public snapshot. The separate source repository referenced in the archive is not publicly accessible; these paths are provenance, not working public code links:

- `../src/plm/config.py`
- `../src/plm/model/architecture.py`
- `../src/plm/reproducibility.py`

The note also names configuration files under `configs/`; the full source checkout is not included. Selected later additions contain copied audit scripts and tests, but that does not make the whole training or serving application available. Raw run paths, checkpoint hashes and source commit IDs identify artifacts; they do not guarantee public access to those artifacts.

## Model size and retained techniques

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
- **Verified autoregressive inference variants:** KV caching, first-target relation guidance, candidate branching, and learned set selection/composition. These are separately configured inference procedures, not extra transformer parameters. See the [set study](../papers/03-set-prediction.md) and [serving study](../papers/04-serving-and-reproducibility.md).
- **Newest research direction:** freeze the parent and fit a symmetric bilinear residual for TYPE/COLOR using query-balanced positive/negative binary cross-entropy. Its dedicated scorer predicts membership from frozen embeddings and learned relation matrices; it does **not** apply the residual through ordinary autoregressive decoder forward calls. Its accepted validation screen is separate from application integration, protected-test evaluation and serving promotion.

The architecture and auxiliary-head design are documented in the copied [architecture](../evidence/source/architecture.md), [prompt-set lesson](../evidence/learning/05-prompt-supervision.md) and [symmetric-relation lesson](../evidence/learning/11-symmetric-relations.md). This size snapshot identifies parent checkpoint `e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1` and local `bilinear-budget8000-v1` derivative `6979013750eb8f2780917f714192850aae411a48490757236a891fea7e351f91`. These hashes identify local artifacts; they do not imply that weights are published or backed up.

## Snapshot history and reuse

Three boundaries matter when reading the chronology:

1. **Historical evidence:** the original copied notes include the failed HTTP score-parity campaign and statements that acceptance was pending. Those records remain unchanged.
2. **Accepted composition integration:** the later shape-aware v2 verification retains 569/666 exact sets on validation under its declared contract. The first campaign still failed, and the inference option remains disabled by default.
3. **Bilinear research derivative:** the ~6.82M variant has a separate scorer and validation screen. It is not the accepted composition application's model or a promoted serving default.

The [2026-09-25 update](../evidence/updates/2026-09-25/README.md) adds a historical
checkpoint inventory, the revised verification contract, a completed native
concurrency probe and the future learned-retrieval hypothesis. Original evidence
copies remain unchanged; the addition has its own checksum inventory.

The later [rank-diagnosis follow-up](../evidence/updates/2026-09-25-ranking-diagnosis/README.md)
adds an audited 479/666 exact-recovery bound for per-query scalar thresholds on
the saved membership scores, and the proof that triple/four-way unions create
no new exact-answer availability. These are diagnostic limits, not new policies.

The [completed composition verification](../evidence/updates/2026-09-25-composition-verification/README.md)
records 666 HTTP responses, fresh serial references, independent audit and
separate acceptance. Earlier dated copies retain their original pending status.

The [margin training screen](../evidence/updates/2026-09-25-margin-screen/README.md)
records an independently audited negative result and a new 25-checkpoint
inventory. The failed variant was stopped before additional-seed replication;
the earlier 23-checkpoint inventory remains unchanged.

The [gradient-diagnosis addition](../evidence/updates/2026-09-25-gradient-diagnosis/README.md)
records 15 audited observations on existing parameter states and 96 fixed
training queries. It examines gradient strength, alignment and radial score
scaling without training another model or changing the rejected margin gate.

The [weaker-margin screen](../evidence/updates/2026-09-25-weak-margin-screen/README.md)
records a separately declared .001-coefficient intervention with fresh paired
training. Complete-answer quality improved, but the required separation gain
failed. The audited result is retained without additional-seed replication
or promotion, together with a new 27-checkpoint inventory.

The [candidate-score diagnosis](../evidence/updates/2026-09-25-candidate-score-decomposition/README.md)
reconstructs all four fixed pool/score combinations and separates candidate
availability from selection errors. It adds no checkpoint or promoted policy.
The same dated addition includes the future learned-retrieval data review.

The [offline batch-saturation sweep](../evidence/updates/2026-09-25-batch-saturation/README.md)
reached 30,980 completed output tokens/s and 188.3 completed requests/s at
batch 640 on one RTX 5070 Ti. This fixed-group decoder result does not change
the measured serial HTTP throughput or establish GPU arithmetic saturation.

The [eager FP8 screen](../evidence/updates/2026-09-25-fp8-screen/README.md)
retained similar validation quality on one guided checkpoint but reduced serial
native throughput from 148.5 to 12.0 completed output tokens/s. The compiled
FP8 follow-up is a declared experiment without a result in this snapshot.

Source checkout: `mateors1/Plm-protocolized-language-model-pika-edition`, `HEAD` `8afff361447b1357a041aa65f5b588af5b7ba5b5`; the copied source material also contained uncommitted changes. The copy was made on 2026-09-25 from the 2026-09-24 research state. [`evidence/SHA256SUMS.txt`](../evidence/SHA256SUMS.txt) records the copied file hashes, allowing exact verification without pretending that the source `HEAD` alone identifies this snapshot. The papers are interpretive syntheses; the evidence copies remain the detailed record.

This public repository includes research material copied from a source working
tree with uncommitted changes. The dated evidence inventories identify the copied
bytes; source `HEAD` alone does not identify them. Results and status statements
should be read with their original dates and declared acceptance gates.

