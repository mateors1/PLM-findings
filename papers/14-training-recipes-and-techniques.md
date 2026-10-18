# 14. Training recipes: what helped, how it works, and what did not

**Published 2026-10-18.** This is an explanatory synthesis of existing experiments, not a new run. Newer results come from the source snapshot `de8338f0dbd644bda1e9746d84a36fe9fb53dc22`, whose canonical paper v0008 has an evidence cutoff of 2026-10-02 16:53 UTC. [Provenance and curated evidence](../evidence/updates/2026-10-18-training-recipes/README.md).

## How to read this ranking

Ranked by usefulness supported in this particular research program, not by novelty or universal ML merit:

1. **Frozen-feature symmetric bilinear residual + balanced BCE, with a longer declared fitting budget:** best observed complete-set validation result.
2. **Prompt-set auxiliary supervision:** foundational, replicated improvement over token-only training.
3. **Symmetric relation auxiliary supervision:** a second replicated representation-learning gain, with historical generation failures preserved.
4. **Learned candidate ranking and pair composition:** valuable downstream use of those representations, but an inference technique rather than another training recipe.

The best observed output is not automatically the best deployable model. All results use the same small graph with known catalog entities. A held-out query can contain entities seen elsewhere during training: this is **transductive query holdout**, not unseen-product generalization. There are 1,637 training queries, 222 reused development queries, and 191 protected test queries not used to support these findings. Three seeds yield 666 query-seed outcomes, not 666 independent questions. Repeated validation-guided choices create adaptive-selection risk.

## 1. Best observed recipe: freeze the representation, learn a richer relation

An embedding is a learned list of 256 numbers describing an opaque product ID. Those coordinates are not labeled properties such as “red” or “fire.” The parent learns them jointly with next-token prediction and two auxiliary tasks. The later fit freezes all 93 original state tensors and changes only a small new relation component.

### Diagonal versus full bilinear interaction

The original symmetric relation head multiplies matching coordinates of subject and candidate, weighted by the relation dimension. Coordinate 7 compares with coordinate 7. In shorthand, its score is `E_subject.T × diagonal(relation) × E_candidate / 16`.

The residual adds a full matrix for each dimension: `parent_score + E_subject.T × S_dimension × E_candidate / 16`, where `S = (A + A.T) / 2`. Now subject coordinate 7 can interact with candidate coordinate 12. Symmetrizing makes swapping subject and candidate preserve the mathematical relation, appropriate for SAME. Floating-point reversal can still differ slightly; symmetry is not a promise of bitwise equality.

A teaching example: subject `(1,0)` and candidate `(0,1)` have zero diagonal interaction. A matrix with off-diagonal entries equal to one gives a nonzero interaction. That illustrates the additional flexibility; it is not an actual product prediction.

**Residual** means additive correction. A starts at zero, so the initial scorer must exactly replay the parent before learning. Zero initialization does not prevent learning: the score's gradient depends on the two embedding vectors, not on a nonzero A.

The stored tensor has shape `[2,256,256]`: **131,072 trainable scalars**, added to the ~6.69M parent for ~6.82M stored model parameters. Symmetry leaves at most **65,792 independent coefficients** in those two matrices; embedding dependencies may reduce observable freedom further. This is not LoRA and no low-rank factorization is being claimed. Freezing prevents this fit from changing the parent representation, but does not prove that freezing itself caused the gain or that the decoder pretraining is necessary versus simpler baselines.

### Balanced binary cross-entropy (BCE)

Rather than ask “which token comes next?”, the scorer asks a yes/no question about each of 1,025 IDs: “does this product belong in this answer?” A sigmoid maps each score to a membership probability for the loss. BCE penalizes confident wrong membership decisions.

For each query, the loss separately averages true members and nonmembers (excluding the subject), then gives those two groups equal weight. Finally it averages queries. Otherwise, the numerous negative products could overwhelm a small positive set, or larger answer sets could dominate learning. This is **query-balanced positive/negative supervision**, not validation-based threshold tuning.

Frozen features make scores affine in A and the ideal logistic objective convex in A. That mathematical property does not prove finite AdamW optimization reaches an optimum, nor that a training optimum generalizes. Redundant full-matrix parameterization also affects optimizer geometry.

### The actual recipe and budget comparison

- Start each fit from its original trained parent, with an exact-zero residual and fresh AdamW state.
- Full batch: all 1,637 training queries per update; only training labels supply gradients.
- FP32 residual and arithmetic; autocast off; learning rate 0.0003, AdamW betas 0.9/0.999, epsilon 1e-8, weight decay zero, no scheduler or gradient clipping.
- Compare declared 2,000, 8,000 and 32,000 residual updates. These are separate fits, **not resuming the 8,000-update checkpoint to reach 32,000**. The parent's own 2,000 training steps are separate.
- Select all non-subject IDs whose final score is strictly greater than zero, in ascending ID order. No true answer size, oracle repair or validation-tuned threshold is supplied.

| Original parent seed | 2,000 residual updates | 8,000 | 32,000 | 32k gains / losses versus 8k |
| --- | ---: | ---: | ---: | ---: |
| 1729 | 207/222 | 216/222 | 218/222 | 2 / 0 |
| 1730 | 212/222 | 220/222 | 222/222 | 2 / 0 |
| 1731 | 208/222 | 214/222 | 220/222 | 6 / 0 |
| **Same 222 queries across three seeds** | **627/666** | **650/666** | **660/666** | **10 / 0** |

The 32k result is **99.10% exact sets**, with macro F1 **0.9999637956903659**. Seven omitted members and one extra member produce six wrong seed-query answers across five distinct queries, all dual-TYPE. This shows why near-perfect F1 can coexist with imperfect whole answers. Strict score separation holds for 665/666 outcomes, yet only 660 are exact at zero: ranking all true members above all false ones is not the same as putting the zero boundary between them.

The original 500-update pilot reached 198/222 versus its dense parent’s 112, but failed the stronger predeclared 201-answer comparator and TYPE-group floors. That failure remains a failure. Later separately declared budgets passed their own fixed gates. Do not reinterpret a failed pilot after seeing its descendants succeed.

**Status:** accepted developmental evidence, **unpromoted candidate**. It now has a separately verified opt-in CPU dense-set adapter and feature-cache path. This scorer bypasses the transformer blocks during inference; it is not a 660/666 autoregressive decoder. The historical 569/666 composition result uses a different predictor, so the difference is not an isolated training-only effect.

## 2. Prompt-set supervision: teach the complete destination as well as the next step

Teacher-forced token training gives the correct previous tokens and asks for the next one. At deployment the model must follow its own decisions. Low token loss can therefore coexist with poor complete answers.

The prompt-set head reads the decoder state at ANSWER, before any target tokens, projects it against all product embeddings, and receives balanced membership supervision for the entire target set. A `[256,256]` projection adds 65,536 parameters. Its gradients teach the shared representation about the full answer, while token loss still teaches ordering and stopping.

Analogy: a student practicing a route gets both next-turn practice and a checklist of every place that must be visited. The checklist does not guarantee the student can execute the whole route.

Across three paired seeds, raw generation F1 improved **23.13% → 60.62%**, and exact answers after the same subject-exclusion/deduplication policy improved **15.62% → 45.80%**. This is the clearest foundational training gain here. Those raw-F1 and processed-exact metrics are different measurements, not one interchangeable score. [Public replication lesson](../evidence/learning/06-seed-replication.md).

## 3. Symmetric relation supervision: build the known relationship into the score

For SAME, if A shares a type with B, B shares it with A. The symmetric head uses the same product embeddings on both sides, with dimension-conditioned coordinate weights. Reversing subject and candidate preserves the algebraic score. The additional `[256,256]` projection contributes another 65,536 parameters.

This is an **inductive bias**: the architecture restricts learning to a relation family appropriate for this task. It is not suitable unchanged for directional relations such as “parent of.” The retained parent combines token loss + prompt-set loss + symmetric-relation loss, each at weight one; experimental continuation and margin losses are off.

Fresh paired controls showed raw F1 **60.78% → 79.87%** and processed exact accuracy **45.80% → 61.11%**. However, two candidate seeds each had one nonterminating answer, so the historical promotion gate failed. Later inference constraints addressed completion without retroactively changing that failed gate. [Replication lesson](../evidence/learning/12-symmetry-replication.md).

## 4. Candidate ranking and composition: use what was learned without retraining

Generate four answer paths with different first products. A learned membership head then scores each candidate set by summing the logits of its members. The ranking follows a factorized Bernoulli membership model: nonmember terms shared by all candidates cancel, leaving the member-logit sum. That is a scoring construction, not proof that membership events are independent or calibrated.

The learned selector improved **466 → 501/666 exact answers**. Adding the six pairwise unions gives ten slots and improved **501 → 569/666**, with 68 gains and no exact losses. A dual-TYPE query can need two groups, and a union can recover coverage missing from either path. The output retains both source paths; it is a selected set, not a fabricated decoder sequence. [Ranking](../evidence/learning/23-ranking-candidate-sets.md) and [composition](../evidence/learning/25-composing-candidate-sets.md).

This explains the complementary roles: training shapes membership scores; branching supplies alternatives; selection chooses; composition changes the available answer sets. Adding more candidates is not automatically beneficial, and a correct answer must be available and correctly ranked.

## What did not earn a winning-recipe label

| Experiment | What it tries | Observed boundary |
| --- | --- | --- |
| Continuation-set supervision | Teach the remaining members at early teacher-forced continuation states | F1 rose, but exact answers fell 466 → 463/666; gate failed. |
| First-token weight eight | Put more token-loss weight on the initial answer choice | 466 → 464/666; held-out first-token loss worsened across seeds. |
| Hardest-boundary margin | Push the weakest true member above the strongest false member | Strong coefficient: 192 → 6/222, failed. Weaker coefficient: 191 → 194, but required separation worsened; still failed. |
| Unrestricted positive-score selection on the old head | Remove the candidate-pool restriction | F1 reached 98.96%, but exact sets fell 569 → 339/666. It cannot be confused with the later refitted bilinear scorer. |
| Frozen original-head refits | Optimize only the old membership projection | Mean-BCE 122/222 and worst-member 128/222, below the strong 201 comparator in their single-seed screens. |
| Symmetric affine offsets | Add query/candidate linear shifts and bias to the bilinear family | 514 additional parameters; all 222 sets unchanged at 216/222 after 8k updates; strict improvement gate failed. |

These negative findings concern the declared architecture, budget and data, not proof that each idea can never work. [Historical failures](02-learning-and-quality.md), [margin result](08-hardest-boundary-margin-failure.md), [weaker margin](10-weaker-margin-mixed-result.md), and [new summary provenance](../evidence/updates/2026-10-18-training-recipes/README.md).

## Systems techniques are a separate ranking

KV caching reuses earlier attention computations; batching amortizes work across queries; FP8 changes arithmetic precision; feature caching reuses transformations of frozen learned embeddings. None is evidence of improved learning by itself. The latest dense feature cache stores learned transformed features, not a table of complete oracle answers.

The recorded CPU feature-cache campaign used **6.37–6.64%** of the contemporaneous uncached scorer’s paired median call time, but still **1.135–1.190×** live SQLite time. Later loopback HTTP favored the cached adapter at 4/8/16 clients against the declared 32-worker SQLite endpoint, with SQLite ahead at 1/2. A separate SQLite-only control found a stronger one-worker policy at higher concurrency. Those separate campaigns cannot establish a cached-model-versus-one-worker winner. The learned errors remain; no matched-quality, energy, edge-device or production superiority follows.

## What is the transferable lesson?

The strongest supported pattern is: learn a useful shared representation, supervise the set structure explicitly, freeze that representation, and fit a relation-specific correction under a declared objective and budget. Keep whole-answer correctness, score separation, execution validity and serving performance as separate questions.

A proposed mechanism still needs comparisons with simpler learned relational baselines, independent evaluation, broader datasets and a precise account of what is actually new.
