# Bilinear residual: fixed 32000-update, three-parent budget campaign

Campaign `bilinear-budget32000-v1`; evaluator
`plm-bilinear-budget32000-screen-v1`. Declared before candidate execution.
Pokemon TYPE/COLOR SAME only. This is adaptive development on the existing
validation queries, not a fresh held-out replication or a serving promotion.

## Question and rationale

Does more optimization improve the unchanged bilinear scorer across all three
existing parent seeds? The accepted 2000-to-8000 comparison gained 9, 8 and 6
exact answers for seeds 1729, 1730 and 1731, respectively, without paired exact
losses. Seed 1729's saved training BCE fell from 2.395098135821172e-6 before
update 7000 to 1.6234706663453835e-6 after update 8000. This motivates a further
budget test; it does not prove undertraining or predict validation improvement.

The audited affine8000-v3 intervention produced the same 222 selected sets as
its bilinear8000 control, despite a slightly lower training loss. It does not
establish a ceiling on frozen representations. Test the next fourfold budget
in the existing 500/2000/8000 sequence: exactly 32000 updates. Do not select an
endpoint after observing results or add another architecture to this campaign.

Historical evidence:

- `docs/experiments/2026-09-25-bilinear-budget8000-replication.json`, SHA256
  `2d96c13e9064431e5f8f74c13a572184e948bcf24e67801dc2a9803c0b85ee61`.
- Accepted replication aggregate `77cc2ee2ebf4dc1f8671e71833712302160190c85477eb6e0dd0b5494bb3cde6`,
  audit `49fc5d7edc7763870ac0e9310295385d8e6a3af6e67d3cc6b5a9f0e44bc2ae77`,
  decision `01a9faefb90ee3d3e7525c0d5191d41741131843dc79ea2b44d860c5c8e21d5e`.
- Seed-1729 control summary `e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8`,
  audit `bdf142f48229fbcdb197bce3a019e38c3a65570ae3f8551a9c9a1f14b2198872`,
  decision `2cadf9dce7ee36cc4095a10a746cc1608d5ce4437724872c1eec1c88e915c195`.
- Affine v3 decision `d3cf1ebfb0e0ba74ca783c5ac86350a842e06dfa1a1f13f82be43cf2a93751aa`.

Resolve each historical child, score report, initial loss and training trace
through these authenticated accepted chains. Preserve failed earlier evidence
versions. Do not recursively open every historical manifest dependency.

## Fixed treatment and tensor contract

Architecture `plm-frozen-symmetric-bilinear-residual-v1`, objective
`plm-bilinear-residual-balanced-bce-v1`, inference
`dense-positive-bilinear-residual-subject-excluded-ascending-fp32-b8-last6-v1`.

The frozen entity table E has shape [1025,256], with rows equal to 16 times the
normalized original embeddings. Train only a fresh zero A[2,256,256]:

    z(s,i,d) = z_parent(s,i,d) + E_s^T sym(A_d) E_i / 16
    sym(A_d) = (A_d + A_d^T) / 2.

There are 131072 stored trainable scalars; symmetry means these are not all
independent functional degrees of freedom. Full-training logits have shape
[1637,1025]. Preserve all 93 original state tensors exactly; final state has 94
entries. No affine terms, entity-offset learning, changed parent representation,
new loss, graph lookup, validation supervision or decoder integration.

Use the unchanged scorer/fit helper
`scripts/refit_bilinear_residual.py`, SHA256
`5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee`,
and archived balanced-BCE objective. Reuse compatible authenticated helpers;
own new campaign, seed and 32000-step lifecycle/checkpoint validators. Never
monkeypatch a frozen module or invoke an incompatible old execute/validator.

## Parents, inputs and optimizer

Execute seeds 1729, 1730, 1731 in that order, each in a separate sequential
process, from its original parent with exact-zero A and fresh AdamW. Never
resume a residual derivative. Original parent checkpoint hashes:

- 1729: `e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1`.
- 1730: `5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2`.
- 1731: `ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d`.

Use input adapter `plm-authenticated-partition-membership-v1`, as specified in
`2026-09-26-symmetric-affine-input-amendment.md` (SHA256
`bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f`).
Reconstruct only authenticated ordered 1637 train and 222 validation records;
prove seed-matched prompts, labels, product columns and query order agree.
Carry split `b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d`
and corpus identity from authenticated evidence. Never open corpus records,
graph databases or protected partitions, even to hash bytes. Explicit allowlist
admission applies equally to primary and independent auditor.

Freeze each original parent's 93 tensors, authenticate checkpoint/config/run/
sidecar identities and recreate its archived model separately from the old
whole-corpus loader. Require exact original-parent/zero-A validation score
replay (B8, final B6), and exact initial training loss against the matching
historical seed. Do not apply seed 1729's scalar to the other parents.

Fit exactly 32000 full-batch updates in training split order. AdamW lr 0.0003,
betas (0.9,0.999), eps 1e-8, weight decay 0, nonfused, fresh optimizer; no
schedule, clipping or accumulation. FP32, no autocast, highest matmul precision,
CUDA TF32 false, cuDNN TF32 true, deterministic algorithms false. Seed RNG with
the original parent seed. Parent pretraining remains 2000 steps, recorded
separately from residual optimization. LM Studio stays offline; one GPU campaign.

## Descriptive training diagnostics and endpoint discipline

Save every pre-update loss and finite score/gradient/parameter check, plus final
post-update loss. Compare the first 8000 losses with the matching historical
trace and compare historical post-8000 to new pre-8001. Record mismatch count
and first mismatch; scalar equality does not establish intermediate weight
equality. A mismatch alone neither changes the gate nor authorizes a retry.

At post-update 8000 and post-update 32000, record training exact-set accuracy
and signed margins. Post-8000 scores may be captured from pre-update 8001;
label this alignment explicitly. For nonself candidate i with training target
y in {0,1}, signed margin is (2y-1)z. Report each row's minimum signed margin,
FP/FN and exact-set status, plus aggregate count and global minimum. Exactness
uses z>0; a negative label at exactly zero is correctly excluded despite zero
margin. Do not replace this rule with margin>0. Preserve local FP32 score
matrices and ordering metadata so the independent auditor can recompute these
diagnostics from train membership. Large matrices remain local; publish bounded
summaries/row chunks and source hashes. They are score artifacts, not checkpoints.

Training diagnostics may use TYPE/COLOR groups only: accepted train membership
does not encode the single/dual-TYPE classification. Do not infer it by opening
the graph. Validation keeps its authenticated COLOR/single-TYPE/dual-TYPE groups.
Report min-positive and max-negative scores and deterministic lowest-ID tie
witnesses, as well as their separation gap, for independent boundary checking.

These diagnostics are read-only detached observations of training logits. If
implemented through an explicit loss callback, retain the unchanged objective's
return value, graph and numerical order; test identical loss/gradients/updates
with and without observation. No backward pass, RNG draw or optimizer step may
be added. Diagnostics never affect learning, stopping, seed choice or acceptance.

Validation endpoints are zero replay and final 32000 only. Save one full final
checkpoint per seed, verify exact model and optimizer reload, then evaluate
the reloaded child once. Require the sole optimizer parameter at step 32000,
all original tensors unchanged and A changed. No intermediate fitted validation.

## Fixed quality gates and interpretation

Select nonself products with z strictly greater than zero and sort IDs ascending.
Preserve raw empty/oversized outputs as failures. No threshold selection,
cardinality oracle, fallback, truncation, per-query exception or seed ensemble.

| Seed | Exact answers must exceed | Minimum macro F1 | COLOR / single TYPE / dual TYPE floors |
| --- | ---: | ---: | --- |
| 1729 | 216 | 0.9998843626799785 | 103 / 51 / 62 |
| 1730 | 220 | 0.9999717006519728 | 103 / 51 / 66 |
| 1731 | 214 | 0.9998584743646364 | 103 / 51 / 60 |

All three seeds must pass their own exact, F1 and group requirements, all
execution/input/frozen-state/replay/reload invariants, and every output must
contain 1..506 distinct nonself products. A tie fails strict improvement. Pooled
gains cannot rescue a failed seed or metric. Report paired gains/losses, FP/FN,
strict separation and per-group results with authenticated query alignment.

Run all declared seeds even if a completed seed fails quality. Stop on execution
or invariant failure, preserve partial evidence and do not silently retry or
substitute a seed. An incomplete campaign cannot satisfy the three-seed gate.
The 666 observations reuse 222 queries and three development-selected parents;
they are not independent examples. No significance or unseen-data claim follows.
A pass permits a separately declared next step; it proves neither protected
oracle parity nor serving/energy advantage. A failure does not prove that the
bilinear hypothesis class is mathematically incapable of solving the task.

## Readiness, audit and publication

Before real preflight or neural execution, finalize and hash-bind this plan,
recipe, runner, aggregator/auditor contracts and actual synthetic-test receipts.
Require independent methodological review. Exercise all three seed admissions,
real evaluator/attach/fit-interface/save/reload/validator/gate stage bodies in a
complete synthetic lifecycle. A bounded simulated fit is acceptable for wiring
tests only and must never be described as 32000 real updates. Include the
previous missing-argument regression, train-config versus root-config admission,
malformed inputs, forbidden loader calls, observation equivalence, boundary
gates, failure preservation, immutable outputs and exact dependency inventories.

Root observes each process's terminal exit and writes a receipt binding seed,
summary and stdout. Poll the same live handle; no restart on observation timeout.
Independent CPU audit recomputes train/validation metrics from saved evidence,
checks all 32000 trace rows, tensors/optimizer/reload/metadata and gates without
importing primary metric or gate arithmetic or repeating CUDA training. Audit
each seed before the next launch. Aggregate only completed audited seeds, verify
receipt chronology as timezone-aware instants with fractional precision, audit
the aggregate and record a separate owner decision. Repairs need new identities.

Register every completed final child after the current 38-checkpoint inventory,
including gate-rejected children; verify final checkpoint bytes and preserve
older inventory entries. Publish each 32000-row trace as sixteen exact 2000-row
chunks, each <=512 KiB, with indices, source hash and exact concatenation check.
Keep model/checkpoint, inference, evaluator and publication identities separate.
The ordinary decoder ignores A; standard_serving_supported remains false.

Update teaching notes, registry, append-only research/dev logs, model register
and PLM-findings. Assess canonical-paper conclusions; archive unpromoted work
through its history index, preserving prior paper bytes if revised. Send the
completed audited immutable handoff to the existing Luna Max publisher. Hashes
identify local weights; they do not prove durable backup or remote availability.
