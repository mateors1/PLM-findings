# Canonical-paper impact assessment: bilinear budget 32,000

**Status:** accepted evidence; publication pending; paper revision deferred. This
assessment does not revise `PAPER.md` or `paper-history/README.md`.

## Claims that remain unchanged

- The retained application-reference decoder, Pokémon `TYPE`/`COLOR SAME`
  protocol, ordinary generation path, and opt-in pair-composition policy do not
  change.
- The bilinear residual remains an experiment-local candidate. The ordinary
  decoder does not apply it, `standard_serving_supported` remains false, and no
  checkpoint is promoted.
- The experiment reuses the fixed 222-query validation split. It does not use
  the protected test partition and does not establish generalization, oracle
  parity beyond this workload, serving advantage, latency, or energy benefit.
- Checkpoint hashes identify local artifacts; they do not establish durable
  backup or public weight availability.

## Finding that changes

Campaign `bilinear-budget32000-v1`, evidence revision
`bilinear-budget32000-evidence-v2`, and evaluator
`plm-bilinear-budget32000-screen-v1` pass all three predeclared per-seed
improvement gates after independent saved-evidence audits. At 32,000 residual
updates, seeds 1729, 1730, and 1731 score 218/222, 222/222, and 220/222 exact,
respectively, versus 216/222, 220/222, and 214/222 at 8,000 updates. The paired
changes are +2, +2, and +6 exact answers, with no losses. The pooled count is
660/666 versus 650/666; these are three evaluations of the same 222 development
queries, not 666 independent examples. All six remaining inexact seed-query
answers are dual-`TYPE`. Seed 1730 is exact only on this validation workload.

This supersedes the 8,000-update candidate as the strongest accepted
validation finding. It does not change the retained model or serving policy.
The aggregate, independent aggregate audit, and owner decision are identified
by SHA256 `8f4539583080275efe53409ce6881af3b50f2b89a092e0a21801ceafe6fd0708`,
`bacc8614e11afceee2080a3faf9850c1b70de92cf5cbd3ac912025da4edb0b3c`, and
`4a508abaf3fac7e2e678cd5604f73636241aabcb18c876076c134468257e44af`. The
41-checkpoint inventory is `eaf8080f8ee8cc9cd7d2c55a4ddad233767943a8e2fe2e5c291e556d009c079c`.
The publication receipt is still required before describing this evidence as
published.

## Deferred paper and history handoff

After this evidence packet is published and the in-flight PAPER-v0002 graph-CPU
packet completes, the canonical-paper owner should prepare `PAPER-v0003` with
predecessor `PAPER-v0002`. Before replacing the root paper, preserve its exact
current bytes as `paper-history/PAPER-v0002-2026-09-26.md` and verify the
archived SHA256 equals
`27b22b90e4c90b3b284bc773025dc9443fddbf56c8ade914ec272e18fd10d29d`.

The new synthesis should update the abstract, best-candidate discussion,
evaluation summary, limitations, and identity ledger to report 660/666 and the
six dual-`TYPE` misses while retaining the fixed-split, non-promotion, and
non-serving caveats. The accompanying history-index handoff should mark v0002
superseded, index the published 32,000-update plan, report, audits, decision,
checkpoint inventory and receipt, and record relevant post-v0001 published
evidence. It must not revise the in-flight v0002 packet or imply that this
assessment itself is a publication receipt.
