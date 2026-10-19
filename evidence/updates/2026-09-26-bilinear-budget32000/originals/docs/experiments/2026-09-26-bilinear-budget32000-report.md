# More optimization of the unchanged bilinear scorer

Campaign `bilinear-budget32000-v1`; evidence revision
`bilinear-budget32000-evidence-v2`; report dated 2026-09-26.
The three fits and their independent audits are complete. The independent
aggregate audit passes, and the separate owner decision accepts the evidence
and fixed quality-gate result without promoting a model. Publication is pending.

Increasing the fixed optimization budget from 8,000 to 32,000 updates improved
exact validation answers for each of the three original parent seeds. The
unchanged bilinear method meets every predeclared per-seed quality gate.
Across the three evaluations, exact answers increase from **650/666 to 660/666**.
These are the **same 222 development queries evaluated under three parents**,
not 666 independent or newly held-out examples.

| Parent seed | Historical 8,000 exact | New 32,000 exact | Net increase | New macro F1 | New FN / FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1729 | 216/222 | 218/222 | 2 | 0.999923235341 | 6 / 0 |
| 1730 | 220/222 | 222/222 | 2 | 1.0 | 0 / 0 |
| 1731 | 214/222 | 220/222 | 6 | 0.999968151730 | 1 / 1 |

The pooled macro F1 is 0.9999637956903659. All six remaining inexact
seed-query answers are dual-TYPE: COLOR is 309/309 exact, single-TYPE 153/153,
and dual-TYPE 198/204. There are seven false-negative and one false-positive
membership occurrences. Strict score separation holds for 665/666 answers.
Seed 1730 matches every target set on this reused validation slice; it does not
establish protected-test oracle parity or generalization. The owner's paired
comparison records 2, 2 and 6 gains, respectively, with no losses; all gains
are dual-TYPE. This is stronger than merely subtracting aggregate exact counts.

## What changed, and what was fixed

The [predeclared plan](2026-09-26-bilinear-budget32000-plan.md) changes only
the optimization budget. Each seed starts again from its original parent,
with a fresh zero residual and fresh AdamW; these are not resumed 8,000-update
children. Parent pretraining steps and residual optimization steps are distinct.
Each run executes exactly 32,000 full-batch updates with unchanged balanced
binary cross-entropy, learning rate 0.0003 and zero weight decay. There is no
validation-based stopping, threshold adjustment, new architecture or ensemble.

Intuitively, a small trainable correction learns which entity pairs should
receive higher or lower membership scores while the parent model stays fixed:

\[
z(s,i,d)=z_{\mathrm{parent}}(s,i,d)
  +\frac{E_s^\top\operatorname{sym}(A_d)E_i}{16},
\qquad \operatorname{sym}(A_d)=\frac{A_d+A_d^\top}{2}.
\]

The frozen entity table has shape `E[1025,256]`; each row is 16 times its
normalized original embedding. Only `A[2,256,256]` is trained, one matrix for
each dimension. That is 131,072 stored scalars, with fewer independent
functional degrees of freedom because scores use the symmetric part. All
93 original state tensors remain unchanged; the residual adds one state entry.
The full training score tensor has shape `[1637,1025]`.

A nonself candidate is selected when its score is strictly greater than zero;
selected IDs are sorted ascending. No true-set cardinality, truncation,
fallback or per-query exception is used. Every final validation answer passes
the declared serialization-size and distinct-ID constraints. The usual decoder
does not use this residual, so `standard_serving_supported` remains false.

## What the audit establishes

The fixed gate requires each seed to beat its own historical exact count,
retain at least its own historical macro F1 and group exact counts, and pass
input, frozen-state, replay, optimizer and reload invariants. A pooled gain
cannot rescue a failed seed. All three gates pass. Each seed audit independently
recomputes the saved evidence on CPU; it is not a second CUDA training run.

Detached training-score observations at post-update 8,000 and 32,000 are
descriptive and do not influence optimization, stopping or the quality gate.
The post-8,000 observation occurs before update 8,001. Each final training
endpoint is 1,637/1,637 exact, but this is fit to training labels, not unseen
performance. For a nonself candidate, signed margin is `(2*y-1)*z`; a negative
label at score zero is correctly excluded even though its signed margin is zero.
Training groups are TYPE/COLOR only; validation's authenticated single/dual
TYPE grouping is not inferred from the graph.

The first seed-1730 audit failed because its predecessor receipt expressed the
same instant with a different UTC offset. The original auditor compared the
timestamp text exactly. Its failed audit, stderr and terminal receipt remain
immutable. The [versioned evidence repair](2026-09-26-bilinear-budget32000-evidence-repair-v2.md)
compares timezone-aware instants with full fractional precision, retaining
exact key, seed and hash checks. It rejects a 100-nanosecond difference.
Seed 1730 was reaudited; its training was not rerun. Seed 1731 uses an explicitly
versioned continuation wrapper. Aggregation preserves the mixed routes:
original seed-1729 audit, repaired seed-1730 and seed-1731 audits.

Both saved aggregate records deliberately retain `acceptance=false`; the
independent aggregate record separately reports `audit_passed=true`.
Neither field should be silently rewritten to represent an owner decision.

## Evidence and publication boundary

- [Aggregate](../../runs/learning/bilinear-budget32000-v1/aggregate-v2.json):
  SHA256 `8f4539583080275efe53409ce6881af3b50f2b89a092e0a21801ceafe6fd0708`.
- [Independent aggregate audit](../../runs/learning/bilinear-budget32000-v1/aggregate-audit-v2.json):
  SHA256 `bacc8614e11afceee2080a3faf9850c1b70de92cf5cbd3ac912025da4edb0b3c`.
- [Owner decision](../../runs/learning/bilinear-budget32000-v1/decision-v2.json):
  SHA256 `4a508abaf3fac7e2e678cd5604f73636241aabcb18c876076c134468257e44af`.
- [Repaired readiness](../../runs/learning/bilinear-budget32000-readiness-v2/readiness.json):
  SHA256 `9df87cdafc02a3449b3b78d9bf48be93f0840da4ba64b35c21c09c286c96a69a`.
- The plan SHA256 is `5f1e0243b373ad1287c31dcef8a3c1cfa310cac9701795469a68677506e6f444`;
  the evidence amendment SHA256 is
  `90aa30088980aeffd999fe6fe4e7349220d163b196ad4eb28f06193b9ce2991c`.

The completed [portable export](2026-09-26-bilinear-budget32000.json) preserves
all 96,000 update records in 48 exact trace chunks and 9,822 diagnostic rows in
102 bounded chunks. The [41-checkpoint inventory](2026-09-26-bilinear-budget32000-model-version-inventory.json)
adds the three new checkpoint identities while retaining the preceding 38
entries unchanged. Only the three new checkpoint files were freshly byte-hashed
by the exporter; prior identities are inherited, not newly reverified.
The [observed export and reconstruction review](../../runs/learning/bilinear-budget32000-planning-v1/export-execution-review-v2.json)
verifies all 152 output hashes, exact row reconstruction and the 512 KiB file
bound. The largest export is 470,615 bytes. Reduced diagnostic rows preserve
audited summaries and score hashes; they do not replace the full matrices for
an independent numerical replay.

The affine-v3 publication packet is an explicit predecessor dependency; its
historical result must not be confused with a promoted retained model. Weights
and large raw training-score matrices remain local dependencies. Hashes
establish identity, not durable backup. Canonical-paper assessment belongs to
the dedicated paper owner: accepted findings may change the synthesis without
promoting this scorer. This handoff does not replace root PAPER/history files.

This result supports further development of the fixed bilinear method under
the tested budget. No model is promoted here, no protected partition is used,
and no serving, latency, energy or generalization advantage is measured.
