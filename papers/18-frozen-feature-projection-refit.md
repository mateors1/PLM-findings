# A projection refit improves dense membership but fails the stronger baseline

**Status: independently audited evidence accepted; fixed quality gate rejected.**
Refitting one membership projection improves dense zero-threshold predictions
from 112 to 122 exact answers out of 222. The accepted eight-branch predictor,
using the same parent checkpoint, already gets 201 exact answers. The refit
therefore fails its declared quality screen despite lower training loss and
higher macro F1. The trained derivative is retained as a distinct checkpoint.

The [frozen plan](../evidence/updates/2026-09-25-projection-only-refit/experiments/2026-09-25-projection-only-refit-plan.md),
[portable results](../evidence/updates/2026-09-25-projection-only-refit/experiments/2026-09-25-projection-only-refit.json)
and [separate decision](../evidence/updates/2026-09-25-projection-only-refit/campaign/decision.json)
preserve the method, measurements and rejection without redefining success.

## What was trained

The parent is `national_dex_continuation_control_s1729_v1`, after 2,000 original
training steps. Every embedding and transformer tensor remains frozen. Only
`symmetric_relation_projection.weight`, a matrix W with shape [256,256], receives
500 full-batch AdamW updates over all 1,637 training queries. These are 500
complete passes over the training partition, not 500 batches of 32. The optimizer
starts fresh, with learning rate 0.0003 and zero weight decay. There is no
scheduler, margin loss, decoder loss, validation early stopping or checkpoint
selection. Only the final trained checkpoint is evaluated.

Let E contain normalized product embeddings scaled by sqrt(256), and U contain
similarly scaled steering embeddings. For subject s, dimension d and product i,

\[
z_{s,d,i}=\frac{(E_s\odot WU_d)^\top E_i}{\sqrt{256}}.
\]

The head uses the embeddings directly, without contextual transformer states.
Detached cached features therefore support a training score tensor of shape
[1637,1025] without running the transformer during fitting. The archived FP32
normalization and matrix-operation order are preserved. All 222 initial head
vectors exactly match both the archived model's head and saved references at
batch eight, with six rows in the final batch.

The loss averages positive and negative softplus penalties separately per
query, then gives each query equal weight. The subject is excluded, and repeated
positive IDs cannot increase their weight. With embeddings fixed, each logit is
linear in W, so this balanced BCE is convex in W. That does not make a finite
AdamW run an established optimum, nor does a lower smooth loss guarantee more
completely correct sets. No capacity-impossibility conclusion follows.

## Two comparisons answer different questions

Both dense policies select every non-subject product with z > 0, exclude exact
zeros, and return IDs in ascending order. They use no teacher cardinality,
threshold search or generated candidate pool. Labels enter training only through
the training partition and enter validation only for reporting.

| Predictor, seed 1729 | Exact /222 | Macro F1 | Exact gains / losses of child versus it |
| --- | ---: | ---: | ---: |
| Unmodified parent, dense threshold | 112 | 0.9903735063 | 13 / 3 |
| Refit child, dense threshold | 122 | 0.9918551978 | — |
| Accepted parent, eight-branch composition | 201 | 0.9799255177 | 2 / 81 |

The first comparison isolates a projection change under the same dense output
rule. The second checks whether the proposed predictor improves on the stronger
existing predictor. These are not interchangeable claims. The screen required
more than 201 exact answers, macro F1 at least 0.9799255177, and no group exact
regression against that stronger comparator.

| Group | Queries | Parent dense exact | Child dense exact | Required comparator exact |
| --- | ---: | ---: | ---: | ---: |
| COLOR | 103 | 75 | 77 | 103 |
| Single-TYPE | 51 | 32 | 37 | 50 |
| Dual-TYPE | 68 | 5 | 8 | 48 |

All 222 child sets satisfy the declared serialization size of 1 through 506
unique non-subject products. Execution and F1 checks pass, but exact-answer and
group requirements fail. The result is not promoted and does not authorize
additional-seed replication under this completed screen.

## Why 99.19% F1 can coexist with 122 exact answers

Macro F1 gives partial credit; exactness requires every membership decision to
be correct. The child has 100 nonexact answers, 39 of which contain exactly one
membership error. For `PKM_APPLIN COLOR SAME`, 129 correct products plus one
false product, ID 1793, produce F1 = 258/259, about 0.996139. Exactness is still
false. Small errors distributed across many queries can yield very high F1 and
much lower complete-answer accuracy.

Across all validation answers, false-positive occurrences decrease from 512
to 389, while false negatives increase from 76 to 109. Precision improves and
recall declines. The loss falls from 0.003822767175734043 before the first update
to 0.001959692919626832 after the final update, a 48.74% reduction. That training
improvement and the validation tradeoff must be reported separately.

Strict positive/negative separation holds for 165 queries before and after.
An unchanged count does not mean unchanged rankings: Hatterene, Keldeo and
Sewaddle TYPE gain strict separation, while Combee, Froslass and Skiploom TYPE
lose it. This diagnostic asks whether every true member outranks every false
member. It is distinct from whether zero happens to split the two classes.

## What the evidence establishes

The independent audit checks saved logits and predictions, query partitions,
metrics, the update trace, checkpoint lineage, optimizer state, and all 93 state
tensors. Only W changes. The final checkpoint reloads exactly, including its
optimizer state. The runner passed 26 focused tests; the auditor passed 70.
The audit reconstructs saved evidence and inspects checkpoint payloads on CPU;
it does not independently repeat training or regenerate neural logits.

The synchronized refit took 3.6429369 seconds on the recorded RTX 5070 Ti;
overall wall time was 11.3313216 seconds. Recorded parent and child standalone
head times, approximately 0.3283 and 0.02468 seconds, include unequal warmup/cache
conditions and are not a paired latency comparison or evidence of a speedup.
There is no serving-throughput, energy or concurrency claim.

The derivative checkpoint hash is
`84a3ba02ab5f43e9073f1f8b8c7afc85ad7a5363e8cb180e1424a77e6a44ab87`.
Its objective is `plm-projection-only-balanced-bce-v1`, and its 500 refit updates
are separate from the parent's 2,000 steps. The standard serving loader does not
support this new objective contract. Dense selection generates neither a token
sequence nor EOS; frozen decoder weights also do not establish unchanged guided
generation, because guidance can consume the changed head.

The [additive model inventory](../evidence/updates/2026-09-25-projection-only-refit/experiments/2026-09-25-projection-only-model-version-inventory.json)
records this as the 28th final checkpoint and freshly hashes all 28 local final
files. That check does not revalidate every older training payload or establish
a durable backup. The [teaching chapter](../evidence/updates/2026-09-25-projection-only-refit/learning/39-refitting-a-frozen-feature-head.md)
develops the tensor arithmetic and the distinction between smooth loss, ranking
and exact-set prediction in more detail.

This is one adaptively chosen screen on repeatedly used validation queries.
It neither tests protected queries nor isolates joint-training gradient
interference as a cause. The graph/compiler remains a perfect oracle for the
scoped relation; this result falls short of oracle parity. The
[dated evidence bundle](../evidence/updates/2026-09-25-projection-only-refit/README.md)
retains code, recipes, receipts and hashes. Weight payloads, full head reports
and training labels remain local dependencies; publication of this bundle is
not a durable weight archive or a self-contained training reproduction.
