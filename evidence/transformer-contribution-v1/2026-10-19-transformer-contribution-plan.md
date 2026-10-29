# Transformer contribution ablation v1

Declared 2026-10-19 before any new fits. Campaign: `transformer-contribution-v1`.
User asks whether transformer-dependent training helped the current dense scorer.
This is a development experiment; the 191 protected queries remain closed.

## Question and fixed arms

For each seed 1729, 1730, 1731:

1. **Joint reference:** authenticate and replay the existing 32000-update
   entity-offset treatment. Its parent had 2000 updates of token, prompt-set,
   and symmetric-relation objectives. Do not retrain or select a reference.
2. **Relation-only:** reproduce the original seeded model initialization using
   the parent's archived factory, retain only the embedding table and relation
   projection, and train only the symmetric balanced membership objective for
   2000 updates. No transformer forward/backward. Match the parent's ordered
   training minibatches (32, final partial batch, cycling without shuffle),
   AdamW settings, cosine schedule, warmup and clipping. Then freeze that parent
   and apply the same zero-initialized A/DeltaE adaptation for 32000 updates.
3. **Random:** the identical seeded untrained embeddings/projection, frozen
   immediately, followed by the same 32000-update A/DeltaE adaptation. This
   distinguishes pretraining benefit from the adaptation's capacity.

Primary comparison is joint versus relation-only. It removes BOTH
transformer-dependent objectives and their gradients, not attention alone.
Parent update/query exposure is matched, not FLOPs or wall time. Clipping uses
the active gradients in each recipe; identical optimizer settings do not imply
identical effective updates. Random has zero pretraining exposure by design.
Joint controls are historical authenticated artifacts, not contemporaneous
retraining. A failure of either simpler arm is not proof transformers are
necessary; a success shows a transformer-free training recipe can match this
development result. No causal attribution of individual embedding coordinates.

## Mathematics and fixed settings

N=1025, D=256, TYPE/COLOR SAME, existing stable token IDs and query split.
E = normalize(product embeddings)*16, steering likewise; r_d=W e_d.
z0(s,i,d) = (E_s * r_d).dot(E_i)/16.
F=E+DeltaE (no renormalization), S_d=(A_d+A_d.T)/2;
z=z0+F_s.T S_d F_i/16.
Adaptation trains A:[2,256,256], DeltaE:[1025,256], 393472 scalars.
Use authenticated historical bilinear arithmetic and archived balanced BCE.
The original parent source has byte-identical model/layers, architecture and
optimizer implementations to the accepted adaptation runtime fixture.

Relation parent: FP32 relation computation as in the parent's BF16 run;
AdamW fused, lr .0003, betas(.9,.95), eps1e-8; projection decay .1,
embedding decay0; cosine warmup200/min ratio.1, global grad norm clip1.
Adaptation: full batch1637, AdamW lr .0003, betas(.9,.999), eps1e-8,
decay0, fused false, foreach null, FP32, no schedule/clipping/early stopping.
TF32 matmul false, highest matmul precision, cudnn TF32 true, nondeterministic
algorithms as the reference. No budget, threshold or hyperparameter search.

## Inputs, evidence and evaluation firewall

Only authenticated saved train-membership (1637 rows), existing development
responses (222 rows), parent/run metadata, checkpoints and source archives are
admitted. No corpus records file, graph, protected labels or test evaluation.
An open audit hook rejects undeclared data/run reads. Historical metadata paths
are identities and are not followed. Hashes are sealed before fitting.
Training uses train membership only. The scorer never accepts expected labels.
Validation occurs after the final checkpoint and exact fresh-state reload, in
batch8/final6. No validation-based stopping or preferred seed selection.
Save scores and per-query selections, exactness, macro F1, FP/FN and COLOR,
single-TYPE/dual-TYPE groups inherited from authenticated development evidence.
Select every positive nonself score, ascending ID, no output truncation or
per-query repair. Report every seed; match requires222/222 for all three.
666 outcomes repeat the same222 queries and are not666 independent examples.

Training diagnostics at fixed update1000 intervals contain training loss only;
report final training exactness and parent-only training loss. Save intermediate
relation-only parent and final adaptation with optimizer/RNG, source/config/input
identities, attempted/completed steps, tensor hashes and exact reload checks.
Preserve failed fits; no automatic scientific retries or overwrite. Any repair
requires a separately recorded attempt/version before observing new results.
Synthetic gates check loss/masking equivalence, both dimension routes, symmetry,
zero offset, gradient ownership, original-initialization equivalence, no decoder
calls, final partial minibatch and checkpoint reload. Independent saved-score
audit must reconstruct results; distinguish it from an independent code review.

## Interpretation and publication

This tests a transductive fixed-catalog system, with known symmetric pair
overlap across query partitions. Development has guided prior research.
It cannot establish unseen-entity generalization, protected parity, attention's
necessity in other tasks, speed/energy superiority, or model promotion.
Interactive transformer service is separately labeled and never used as a
scientific evaluator. User chat may contend for GPU; timings are descriptive.
Document all outcomes, register checkpoint/recipe/evaluator identities, update
the canonical synthesis only after audit, retain the old paper verbatim, and
hand the scoped completed evidence to the existing publication owner.
