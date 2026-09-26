# 48. Does more training help different starting models?

**Status: completed, independently audited; both fresh seeds passed.** The previous experiment
improved seed 1729 from 207 to 216 exact answers out of 222. We tested the same
8000-update recipe on parents 1730 and 1731. This asks whether the improvement
depends on one particular starting model.

## What does the seed change?

A random seed controls random choices in a training procedure. Here the three
parents were already trained with different seeds. Their learned parameters,
including embeddings, differ. We keep each parent frozen and learn a fresh
residual on top of it. We are not changing the Pokemon labels or corpus split.

The residual always starts at zero, so these are not three random initializations
of the residual matrix. They are three different frozen feature spaces and
parent scoring functions. For parent seed r, the score is

\[
z^{(r)}_{sdi}=z^{(r),\mathrm{parent}}_{sdi}
+\frac{(E_s^{(r)})^\top S_d^{(r)}E_i^{(r)}}{16},\qquad
S_d^{(r)}=\frac{A_d^{(r)}+(A_d^{(r)})^\top}{2}.
\]

Each parent supplies E[1025,256]. Each fresh fit trains A[2,256,256], one matrix
for TYPE and one for COLOR. Training scores have shape [1637,1025]; validation
scores cover 222 queries using batches of 8 and a final batch of 6. The original
93 tensors stay unchanged. The only learned addition is A.

Different feature spaces may make the same relation easier or harder to learn.
Testing several parents helps reveal that sensitivity. Two additional parents
still provide limited evidence; they do not establish robustness to every seed.

## Compare each model against its own earlier budget

Both new fits restart from their original parent, A=0 and fresh optimizer state.
Each runs exactly 8000 updates. Its historical 2000-update sibling is the control.
Neither fit resumes that sibling's learned residual or optimizer.

| Parent seed | Historical 2000-update exact count | F1 floor | COLOR / single TYPE / dual TYPE floors |
| --- | ---: | ---: | --- |
| 1730 | 212/222 | 0.9997931269673187 | 103 / 51 / 58 |
| 1731 | 208/222 | 0.999668812076336 | 103 / 51 / 54 |

Each new child must strictly improve its own exact count and meet its own other
floors. It would be misleading to compare every child only to seed 1729's lower
2000-update exact count. A seed that already scored 212 should not get credit for
a supposed improvement merely by exceeding 207.

For query q and parent r, define the paired change

\[
\Delta_{rq}=\mathbf1[\widehat Y^{8000}_{rq}=Y_q]
-\mathbf1[\widehat Y^{2000}_{rq}=Y_q].
\]

Its value is +1 for a newly exact answer, -1 for an answer that became wrong,
and 0 otherwise. Summing over queries gives gains minus losses, exactly equal
to the change in exact count. This pairing controls for which questions were
asked; it does not eliminate all sources of experimental variation.

## Why not average away a regression?

Suppose one seed gains ten exact answers while the other loses one. The pooled
count improves, but the claim that both seeds improved is false. Our declared
gate therefore requires both fresh seeds to pass their own criteria. Gains in
one group or metric cannot compensate for failure in another.

Execution success, evidence validity and quality are also separate. A run can
finish correctly and fail its quality gate. That remains a valid negative result.
An incomplete or unaudited run cannot establish a quality result either way.

## More model-query observations are not more independent questions

We reuse the same 222 validation questions for both fresh seeds. Conceptually,
the results form a [2,222] table of exactness indicators. It contains 444
model-query observations but only 222 distinct questions. Adding the historical
seed 1729 gives 666 observations, still over those same questions.

A difficult dual-TYPE query can fail for multiple models. Treating those outcomes
as unrelated test examples would overstate how much evidence we have. We report
the two fresh seeds separately and label seed 1729 as historical development
evidence. None of these measurements uses the protected test partition.

## What do the execution checks establish?

Before training, each zero residual must reproduce its own parent scores and
training loss. The initial scalar is allowed to differ between parents. We also
compare each first-2000 loss prefix with that seed's historical trace, aligning
old post-update 2000 with new pre-update 2001 as explained in
[Lesson 47](47-more-updates-with-the-same-objective.md).

Prefix equality is descriptive; unequal values cannot trigger tuning or a retry.
Equal scalar losses do not prove identical intermediate tensors. Each final
checkpoint receives an independent audit of saved tensors, metadata, predictions
and metrics. A separate aggregate audit checks the two reports and chronology.

The seeds run sequentially on the GPU. An observed process completion receipt
binds each run's summary and output to its place in that sequence. An audit
receipt binds the correct seed, summary, auditor, audit output and test receipt.
This prevents an unrelated successful audit from being mistaken for verification
of the current model.

The [frozen plan](../experiments/2026-09-25-bilinear-budget8000-replication-plan.md)
defines the complete experiment. The [audited result](../experiments/2026-09-25-bilinear-budget8000-replication.json)
and [model register](../model-versions.md) retain the outcome and checkpoint
lineage. Passing this validation screen does not establish oracle parity or
automatically promote the scorer into serving.

## A real verification failure

The first fit, seed 1730, completed once and provisionally reports 220/222 exact
answers versus its historical 212. Its independent audit then failed before
loading the checkpoint. The auditor passed Python sets to a helper that compares
JSON encodings. JSON has arrays and objects, but no native set type, so this
raised a serialization error. The 131 synthetic tests had missed that production
dependency-inventory path.

This is a verifier defect, not evidence that the model is correct or incorrect.
We retain the failed source, stdout and exit-1 receipt. A separately versioned
auditor compares dependency names without trying to JSON-serialize sets.
Regression tests exercise the actual production check with the authenticated
receipt schemas, including missing and extra dependencies. Passing a test suite
only establishes what the suite actually covers.

The [repair declaration](../experiments/2026-09-25-bilinear-budget8000-replication-evidence-repair.md)
keeps the training recipe and quality gate fixed. The saved seed-1730 model is
audited again with versioned verification code; it was not retrained. That audit
passed, then seed 1731 ran once and passed its audit. The versioned aggregate
reader and independent aggregate audit also passed. The original aggregator
never executed; it must not be described as a failed aggregation. The original
auditor failure remains preserved alongside its repair and successful checks.

## What the completed experiment teaches us

| Parent | 2000 updates | 8000 updates | Newly exact / lost | Macro F1 at 8000 |
| --- | ---: | ---: | ---: | ---: |
| 1729, historical development screen | 207/222 | 216/222 | 9 / 0 | 0.999884363 |
| 1730, fresh replication | 212/222 | 220/222 | 8 / 0 | 0.999971701 |
| 1731, fresh replication | 208/222 | 214/222 | 6 / 0 | 0.999858474 |

Both fresh seeds exceeded their own exact-count floor and preserved their F1
and group floors. The fresh total is 434/444 exact, macro F1 0.999915088; adding
the historical seed gives 650/666 and 0.999904846. Each total still covers the
same 222 questions. This supports the narrow claim that the budget improvement
repeated on both additional frozen parents under this protocol. It does not
prove an improvement on unseen data or for arbitrary initializations.

All remaining errors are dual-TYPE questions. Fresh seed 1730 has no false
positives and two false negatives; seed 1731 has seven false positives and two
false negatives. COLOR and single-TYPE are exact for all three seeds.

For seed 1730 every query has strict score separation:

\[
\min_{i\in Y_q}z_{qi} > \max_{j\notin Y_q,\ j\ne s}z_{qj}.
\]

But the inference rule uses a fixed zero boundary. Separation alone does not
ensure all true members lie above that boundary and all false members below it.
That is why perfect ranking separation can coexist with two wrong answer sets.
No threshold was tuned to these validation labels. Seed 1731 has separation
for 221/222 questions, so a threshold change alone cannot solve every case.

The new derivatives preserve all 93 parent tensors, train only A, and record
8000 optimizer updates separately from the parents' 2000 pretraining steps.
The local inventory now covers 37 final checkpoint files. Eight portable chunks
preserve all 16000 new training records, with exact concatenation checks. The
v2 verifier passed 148 CPU tests, the v2 aggregator 46, and the primary suite
221. Actual per-seed and aggregate audits also passed; those are distinct from
synthetic test coverage. Checkpoint hashes establish identity, not remote backup.

The next research question is what prevents the remaining dual-TYPE decisions
from becoming exact. The ordinary serving decoder still does not apply this
residual. Oracle parity, protected evaluation and serving integration remain open.
