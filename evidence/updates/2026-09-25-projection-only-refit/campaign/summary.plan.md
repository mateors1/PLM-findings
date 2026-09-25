# Frozen-feature membership projection refit

Declared 2026-09-25 before new training or validation measurement. Pokemon
TYPE/COLOR SAME only. This is a single-seed training screen, not serving
integration, a protected-test evaluation or a claim of oracle parity.

## Question and comparators

Coverage-seeking generation increased exact candidate availability but reduced
selected exact answers from 603 to 600 across three seeds. Test whether the
existing entity representations support better membership discrimination when
only the symmetric relation projection is optimized. This changes the learned
scorer; it does not tune a threshold on the previous frozen scores.

Use parent national_dex_continuation_control_s1729_v1, checkpoint SHA256
e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1,
inherited configuration SHA256
6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148.
It has 2,000 parent training steps. Authenticate its checkpoint sidecar, original
training receipt, corpus, vocabulary and split through the accepted width-eight
provenance chain. Split SHA256 is
b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d.
The split has 1,637 training, 222 validation and 191 protected-test queries.
Do not predict or compute metrics on protected-test queries.

Accepted width-eight summary SHA256 is
1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183,
audit eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95,
decision 70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601.
Its seed-1729 strong comparator has 201/222 exact answers, macro F1
0.9799255176742276, and group exact counts COLOR 103, single-TYPE 50,
dual-TYPE 48. Also evaluate the unmodified parent with the identical dense
prediction rule used for the child. Report both comparisons: an improvement
over the dense parent alone does not satisfy the strong screen gate.

## Fixed training recipe

Use the authenticated archived source/configs from the width-eight runtime:
source ZIP SHA256
1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376,
configs ZIP SHA256
51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970.
Authenticate/extract a fresh isolated runtime and check imported module origins.
Do not silently substitute live code. Freeze the runner, recipe and focused test
receipt before the first model execution. Preflight must be stdlib-only.

Initialize from the parent. Freeze every parameter except
symmetric_relation_projection.weight, W with shape [256,256]. Cache detached
normalized entity and steering embeddings using the exact archived FP32
normalization/scaling. For subjects s and dimensions d, compute

    E = sqrt(256) * normalize(entity_embeddings)
    U = sqrt(256) * normalize(dimension_embeddings)
    Z = linear(E[s] * linear(U[d], W), E) / sqrt(256)

using the archived F.linear operation order. Full-batch training logits have
shape [1637,1025]. The frozen embeddings can be reused without transformer
forwards. Reuse the archived _prompt_set_loss and its masking semantics:
deduplicated positive products, subject excluded from negatives, equal weight
to mean positive and mean negative softplus loss within each query, then equal
weight to each query. Reject empty positive or negative groups. The coefficient
of this loss is 1. No causal, contextual, margin or auxiliary objective is added.

All 1,637 training queries participate in each of exactly 500 optimizer updates,
in their authenticated corpus order after split filtering. This is 500 complete
training passes, not 500 batches of 32 and not 500 extra parent trainer steps.
Use fresh AdamW state through the archived create_optimizer: learning rate
0.0003, betas (0.9,0.999), epsilon 1e-8, weight decay 0, non-fused optimizer.
Use FP32 parameters and arithmetic, disabled autocast, CUDA matmul TF32=false
and cuDNN TF32=true, matching the archived replay environment. Use deterministic
seed 1729 and archived deterministic attention settings for replay. No scheduler,
gradient clipping, gradient accumulation, random sampling or validation early
stopping. Store the recipe as a versioned JSON configuration and bind its hash.

Record every update's pre-update training loss and the final post-update loss.
Require finite loss, gradients and parameters; retain an immutable failure
record and stop on a nonfinite value. Do not adjust hyperparameters after seeing
losses. Save exactly the final full checkpoint, including fresh optimizer state;
no intermediate checkpoint selection. A failed quality screen still preserves
its final trained checkpoint. No automatic repetition or sweep is part of this
screen. Future seeds 1730/1731 require their own declared replication campaign.

## Replay and prediction

Before fitting, recompute all 222 parent head vectors at the historical ordered
batch shapes B=8, final B=6, requiring exact equality with authenticated saved
width-eight logits. Also require the standalone head computation to match the
archived model's head at those same shapes. Do not relax tolerances on failure.
Preserve parent dense predictions and all logits before any update.

After the 500 updates, save and reload the child, validate its new training
contract, and evaluate once using exactly the same validation order, batch
shapes and FP32 head arithmetic. Reloaded child state must equal saved state.
Hash every state tensor before and after: every tensor except W must remain
byte-identical. Record changed W explicitly. Store all 222 child logit vectors.

For both parent and child, select every non-subject product whose score is
strictly greater than zero; exclude exact zeros and return IDs in ascending
order. No true cardinality, labels, group routing, threshold search or candidate
generation enters prediction. Keep empty and oversized predicted sets in raw
evidence and metrics. A compatible set has 1 through 506 products, allowing
five prompt tokens plus products plus EOS within 512. No fallback or truncation.
This screen measures dense set selection and serialization compatibility;
it generates neither an autoregressive sequence nor EOS. Do not fabricate a
GenerationResult or claim sequence/serving equivalence.

## Evidence, measures and fixed decision gate

Bind hashes of all input artifacts, executed source archives, runner, recipe,
plan and test receipt. Record environment/dependency/device/numerical settings,
source commit plus dirty-state identity, ordered train/validation query identities,
parent model identity and inherited architecture separately from the new recipe.
Record synchronized refit and head-evaluation time, wall time and peak memory
as descriptive measurements; no concurrency, energy or serving claim.
Coordinate GPU use with other tasks and preserve the resident LM Studio service.

Report parent/child dense exact answers, macro precision/recall/F1, false
positives/negatives, gains/losses and COLOR/single-TYPE/dual-TYPE groups. Compare
the child to the accepted width-eight seed-1729 predictions as well. Report
strict positive/negative score separation and zero-threshold correctness as
diagnostics, not a tuned decision rule or additional post-hoc gate. Store all
expected sets separately from prediction inputs. No validation labels enter fit.

The fixed screen gate requires all of:

1. Authentic complete inputs, exact parent replay, finite complete 500-update
   execution, train-only membership labels, exact child reload and frozen-tensor
   invariants, and an independently accepted evidence audit.
2. Every child dense set has 1 through 506 unique non-subject product IDs.
3. Child exact count strictly exceeds 201/222 and macro F1 is at least
   0.9799255176742276.
4. Child group exact counts are at least COLOR 103, single-TYPE 50, dual-TYPE 48.

No additional requirement that every parent-dense metric improves is imposed;
those paired changes must still be reported. A pass authorizes proposing fixed
replication, not promotion. A failed gate remains failed without budget extension.
Neither convexity nor a lower training loss guarantees convergence, exact-set
quality, a calibrated probability or validation generalization.

## Checkpoint and publication contracts

This is a new trained derivative, not an inference-only revision. Declare
objective plm-projection-only-balanced-bce-v1 and evaluator
plm-projection-refit-screen-v1. Reuse archived save_checkpoint/load_checkpoint
with an explicit standalone child validator. Record parent steps=2000 and
projection updates=500 separately; child global_step records refit updates.
Do not falsely label the child causal-next-token-v1/plm-train-causal-v2 merely
to pass the ordinary serving loader. The current serving runtime is expected
to reject this new training contract until a separately reviewed extension.

Keep full weights local under ignored run storage; preserve checkpoint and
sidecar hashes and available archive status. Current inventory is 27 final
checkpoints; register any completed child as the next derivative, even if its
quality gate fails. Frozen decoder weights do not imply unchanged guided
generation, because guidance can use the changed head.

Focused synthetic tests must verify exact loss/subject masking, frozen-parameter
optimization, train/validation separation, update counting, strict-zero/subject
prediction handling, output bounds, metrics and fixed gates, child identity and
reload checks, immutable outputs and failed-input rejection. Independently audit
saved logits, selected IDs, labels, metrics, loss/update trace, tensor hashes and
checkpoint lineage without importing primary prediction/aggregation arithmetic.
This is a saved-evidence audit, not an independent repeat of neural training.
Keep evidence acceptance separate from the owner's quality decision.

Update the model register, teaching lesson, registry, research/dev logs and
PLM-findings after the audited outcome. Deliver an immutable completed-experiment
packet to the existing Luna Max publication task, including a failed outcome.
Git publication, trained checkpoint identity and durable weight archival remain
distinct. The experiment cannot establish that earlier failures were caused by
joint-training gradient interference: it is a fixed-parent refit intervention.
