# Transformer contribution: completed development ablation

Date: 2026-10-20 (America/Bogota). Campaign `transformer-contribution-v1`,
successful execution `transformer-contribution-attempt2-v1`.

## Question and outcome

Does the transformer-dependent part of the original joint training recipe help
the final entity-offset relational scorer reach its existing development result?
The relation-only recipe matches the historical joint reference at 222/222 exact
sets in each of three seeds. Random frozen parent features followed by learned
adaptation reach 221/222, 222/222 and 219/222. Retain the relation-only recipe as
a development-validated alternative, not a promoted replacement or serving winner.

| Seed | Joint reference | Relation-only | Random parent |
| --- | ---: | ---: | ---: |
| 1729 | 222/222 | 222/222 | 221/222 |
| 1730 | 222/222 | 222/222 | 222/222 |
| 1731 | 222/222 | 222/222 | 219/222 |
| Repeated seed-query outcomes | 666/666 | 666/666 | 662/666 |

There are 222 distinct, adaptively reused development queries. This is not a
666-example independent test. All six new final fits report 1637/1637 training
sets. The 191 protected queries remain closed. No new training was needed for
the closure audit.

## Fixed intervention

The [predeclared plan](../evidence/transformer-contribution-v1/2026-10-19-transformer-contribution-plan.md) and
[configuration](../evidence/transformer-contribution-v1/transformer_contribution_v1.json)
fix seeds 1729/1730/1731 and the following arms:

1. Joint reference replays existing authenticated entity-offset checkpoints.
   Their parents had 2000 updates of token, prompt-set and relation objectives;
   adaptation then used 32000 updates. These controls were not retrained.
2. Relation-only recreates the original seeded initialization, retains the
   embedding table and relation projection, and trains only balanced symmetric
   membership loss for 2000 updates. Transformer blocks do no forward or backward
   computation in this arm. The parent is then frozen for 32000 adaptation updates.
3. Random freezes the same initialized table/projection immediately, then learns
   the same adaptation for 32000 updates. The final model is trained: only its
   parent features are random and frozen.

Parent update/query exposure is matched between joint and relation-only, not
FLOPs, elapsed time or effective optimizer updates. Removing both token and
prompt-set objectives is a recipe intervention, not an attention-only ablation.
Factory construction reproduces RNG initialization; constructing a transformer
does not mean executing it during training.

For normalized product rows E and dimension vectors of norm 16, r_d = W e_d:

\[
z(s,i,d)=\frac{(E_s\odot r_d)^T E_i}{16}
 +\frac{(E_s+\Delta E_s)^T S_d(E_i+\Delta E_i)}{16},
\qquad S_d=(A_d+A_d^T)/2.
\]

E and offsets each have shape [1025,256]; W is [256,256], A is [2,256,256].
Adaptation has 393472 trainable scalars: 131072 in A and 262400 in offsets.
The compact parent stores a [2049,256] table plus W: 590080 scalars. Including
adaptation gives 983552 stored scalars, not a claim that every reserved embedding
row participates in scoring. Checkpoint storage also includes optimizer/metadata.

Relation pretraining uses FP32 relation arithmetic, AdamW lr 0.0003,
betas (0.9,0.95), epsilon 1e-8, projection decay 0.1, embedding decay zero,
200-step warmup/cosine minimum ratio 0.1 and norm clipping at 1. Ordered batches
have size 32 with the final partial batch retained. Adaptation is FP32 full-batch
1637, AdamW lr 0.0003, betas (0.9,0.999), zero decay, no schedule/clipping or
early stopping. All positive nonself scores are selected in ascending ID order;
threshold zero and evaluation batches eight/final six were fixed in advance.

## Audit and provenance

Successful seal: `d58ee5315a4d99c2e56c905e439e2eb3b679b9c8bc4c58a754f98c040ee2be4c`.
The [portable evidence](../evidence/transformer-contribution-v1/2026-10-20-transformer-contribution-evidence.json) indexes
the source/config identities, summary, saved-score audit, CPU replay, checkpoint
lineage and preserved failure. It contains no checkpoint payload or raw score rows.

The saved-score auditor independently reconstructs selections and metrics,
checks declared query identities and checkpoint hashes/state, frozen parent
identities and optimizer/reload records. On 2026-10-20 rerunning that auditor
produced exactly the saved audit object. The CPU replay independently implements
the score equation using query-side matrix multiplication: all 1998 selections
(three arms times three seeds times 222 queries) match, with maximum absolute
CPU/GPU score difference 0.00008392333984375, below its 0.001 verification bound.
These are arithmetic checks of existing artifacts, not new scientific samples.
Neither replay retrains the model; training history still depends on execution
receipts and the authenticated historical reference.

The separate [final scientific review](../evidence/transformer-contribution-v1/2026-10-20-transformer-contribution-review.json)
and [research-owner decision](../evidence/transformer-contribution-v1/2026-10-20-transformer-contribution-decision.json)
distinguish evidence acceptance from model promotion. The historical summary's
`audit_pending=true` is superseded by the audit and closure decision; its original
bytes are preserved rather than rewritten.

Two clarifications from the earlier bounded source review are retained:
the evaluation wrapper receives labels to validate/materialize its batch, but
score computation uses prompt IDs and labels are used for post-score metrics;
the literal predeclared wording that the scorer never accepts labels was too
strong. Initialization equivalence and absence of decoder calls have static
source/identity support, not dedicated assertions for both in the original
synthetic test file. Do not present those as separately passed executable gates.

Attempt 1 completed the first relation-only fit, then failed optimizer reload
verification before new-arm development evaluation: an Adam step tensor was on
CPU before saving and CUDA after reload. The [repair record](../evidence/transformer-contribution-v1/2026-10-19-transformer-contribution-reload-repair.md)
preserves the failed attempt and compares tensors on the same device without
relaxing equality. Attempt 2 repeated the fixed campaign in a new directory;
the failed compute is additional cost, not erased or counted in a per-fit budget.
Closure also wraps one overlong replay-script string for lint; executed bytes
are preserved and the string's value and arithmetic remain unchanged.

## What sticks and what does not

Retain the protocol, stable identifiers, deterministic oracle, protected split
boundary, symmetric relation scoring and learned entity corrections. Retain the
relation-only recipe as a simpler development-validated alternative because it
matches every seed without transformer-dependent training. Retain the original
joint checkpoints and transformer implementation as reference and interactive
decoder paths; this audit does not replace any serving default.

Archive the random-parent arm as an informative near-match that fails the fixed
all-seeds-exact criterion. Do not discard its evidence: 662/666 shows that learned
adaptation can fit much of this fixed-catalog task from random frozen features.
It does not show that no learning is required. Retire the unsupported assumption
that transformer-dependent pretraining is necessary for this development target.

Known-entity transduction, reverse-query supervision and development reuse limit
the inference. There is no unseen-entity result, protected-test oracle parity,
attention-only causal estimate, production validation or measured speed/energy
advantage. The stronger SQLite W1 result from earlier work remains relevant;
none of its timing evidence transfers to these new checkpoints.

## Publication boundary

Publication includes source, compact evidence, report, decision, model register,
paper/history and teaching documents. Local weights, raw runs and private records
are excluded. Hashes identify local artifacts; they do not establish remote backup.
Git publication identities and the private lesson-site deployment remain separate.
