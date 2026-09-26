# 44. Replicating across trained parent seeds

**Status: completed; both fresh seeds pass independent audit and their fixed gates.** The
[replication plan](../experiments/2026-09-25-bilinear-seed-replication-plan.md)
tests the fixed 2000-update bilinear recipe on two other existing parent models.
Lesson 43's passing seed-1729 result stays a historical development result.

The independent audits confirm 212/222 exact answers for seed 1730 and 208/222
for seed 1731. A separate owner decision accepts the evidence and the passed
replication screen. Neither checkpoint is promoted to the default predictor.

## What happened

| Parent seed | Role | Matching selector exact | Bilinear exact | Macro F1 | COLOR / single TYPE / dual TYPE exact |
| --- | --- | ---: | ---: | ---: | --- |
| 1729 | Historical development result | 201/222 | 207/222 | 0.99974564 | 103 / 50 / 54 |
| 1730 | Fresh replication | 205/222 | 212/222 | 0.99979313 | 103 / 51 / 58 |
| 1731 | Fresh replication | 197/222 | 208/222 | 0.99966881 | 103 / 51 / 54 |

Both fresh models beat their own selector's exact count and F1, meet every group
floor and produce 222 serialization-compatible answers. Their combined exact
count rises from 402/444 to **420/444**, or about **94.59%**. Counting the historical
model as well gives 627/666, about 94.14%, compared with the selectors' 603/666.
The historical model helped select the recipe; it is not a third fresh replication.

The fresh comparison contains 30 gains and 12 losses. A gain means a question
previously answered incorrectly becomes exact; a loss means an exact answer
becomes incorrect. The net improvement of 18 does not mean every query improves.
All 24 non-exact fresh answers concern dual-TYPE queries. Each fresh model gets
all COLOR and single-TYPE answers exact on this validation split.

F1 and exactness measure different things. For a query's predicted set P and
correct set T,

\[
F_1=\frac{2|P\cap T|}{|P|+|T|},\qquad
\mathrm{exact}=\mathbf{1}[P=T].
\]

One missing member in a large set can leave F1 close to one while exactness is
zero. Fresh macro F1 is 0.99973097, but 24 of 444 model-query observations still
contain an error. That is why this result is progress toward oracle parity,
not oracle parity itself.

The [portable result](../experiments/2026-09-25-bilinear-seed-replication.json)
binds the decision, comparisons, hashes and separate full training histories.
The [updated inventory](../experiments/2026-09-25-bilinear-seed-replication-model-version-inventory.json)
registers checkpoints 32 and 33 and freshly verifies all 33 final weight files.
Local hashes identify those weights; publication of documents does not archive them.

## Why one good result is not enough

We developed the longer fit using one parent's results. Its 207 exact answers
could reflect a method that transfers well across trained models, or a method
that happens to suit that parent's learned embeddings. Testing two more parents
helps distinguish these possibilities without changing the recipe again.

A **random seed** initializes a random-number generator. It can influence model
initialization and stochastic operations during training. Different trained
parents therefore need not learn identical embeddings even when they use the
same data and architecture. Here we reuse existing parents trained with seeds
1730 and 1731; we are not repeating their entire original training run.

The added residual A always starts at zero. Changing only its RNG seed would
not create a different random initialization. The meaningful variation here
is the already trained parent: its embeddings and original relation weights.

## Same shapes, different learned coordinates

For parent seed k, the full token embedding table has shape [2049,256]. The
scorer takes its 1025 product rows and normalizes each row to length 16:

\[
E_i^{(k)}=16\frac{w_i^{(k)}}{\lVert w_i^{(k)}\rVert_2}.
\]

Thus E^(k)[1025,256] is the derived product representation used by the scorer,
not the full token table. Denote the original scores by z_parent^(k). Each new
fit learns its own A^(k)[2,256,256]:

\[
M_d^{(k)}=\frac{A_d^{(k)}+A_d^{(k)T}}{2},\qquad
z_{sdi}^{(k)}=z_{sdi,\mathrm{parent}}^{(k)}+
\frac{E_s^{(k)T}M_d^{(k)}E_i^{(k)}}{16}.
\]

All dimensions and operations are the same across seeds. The actual numbers
in E and the parent scores differ. Each parent freezes its own 93 tensors;
each child adds one trainable tensor and saves a full 94-entry state.

Each fit uses logits [1637,1025] for all training queries, a scalar balanced
BCE loss, and a gradient [2,256,256]. AdamW settings and 2000-update budget stay
fixed. There is no extra capacity or extra data in this replication.

The initial loss can legitimately differ between parents. We check

\[
Z_{A=0}^{(k)}=Z_{\mathrm{parent}}^{(k)},\qquad
L_{A=0}^{(k)}=L_{\mathrm{parent}}^{(k)}
\]

exactly for each seed's FP32 execution. We do not require every parent to have
seed 1729's initial loss. That would confuse repeatability of the implementation
with equality of different trained models.

## Model seed and split seed are different controls

The two parent configurations change their model/training seed but retain
`data.split_seed=1729`. They use the same 1637 training, 222 validation and
191 protected queries. This keeps the evaluation questions fixed while changing
the learned model.

For each validation query we can therefore ask whether the new head changes
an incorrect answer to an exact answer, or the reverse. Those are paired gains
and losses. Changing the questions at the same time would answer a different
question about variation across data splits.

## Compare each replica with its own baseline

The stronger eight-branch selector behaves differently on each parent's weights:

| Parent seed | Selector exact / 222 | Required child exact | COLOR / single TYPE / dual TYPE floors |
| --- | ---: | --- | --- |
| 1729, historical development seed | 201 | >201; previously passed at 207 | 103 / 50 / 48 |
| 1730, fresh replication | 205 | >205 | 103 / 45 / 57 |
| 1731, fresh replication | 197 | >197 | 103 / 44 / 50 |

Each child must also meet its matching baseline's F1 and all serialization and
execution requirements. We require both fresh seeds to pass. A pooled gain
cannot compensate for a failed seed or group.

Both planned fits run even if the first completes with a failed quality gate.
Stopping after a convenient result would change which evidence gets reported.
An execution failure is different: it stops the campaign and remains recorded
as incomplete, rather than being disguised as a scientific comparison.

## What the averages do and do not say

Let e_k be a seed's exact-answer count. With 222 shared queries per seed,
the three-seed mean exact rate is

\[
\overline p=\frac13\sum_k\frac{e_k}{222}
=\frac{e_{1729}+e_{1730}+e_{1731}}{666}.
\]

That denominator counts model-query observations. It does not mean we acquired
666 independent test questions. The same 222 questions are evaluated with three
models, and seed 1729 helped guide development. We report the fresh two seeds
separately and label the historical contribution.

Two fresh replicas are useful evidence about stability across these parents.
They are a small sample and do not prove generalization, oracle parity, or a
serving advantage. The protected test remains untouched. New weights receive
new checkpoint identities whether the quality gate passes or fails; reused
seed-1729 evidence creates no additional checkpoint.

## A lesson from the evidence checks

Both fixed-budget training processes completed successfully. Audit and
aggregation encountered two evidence-processing failures. The independent
auditor expected metadata in the wrong
parent JSON record. Separately, the aggregate report compared the spelling of
two timestamps: one used UTC and the other used the local -05:00 offset.
They represented the same instant.

This illustrates two different kinds of equality. A SHA256 hash checks exact
file bytes: changing even the timestamp spelling produces a different artifact.
A chronology check asks whether two timestamps identify the same instant, after
accounting for their offsets. Neither kind of equality replaces the other.

The [repair declaration](../experiments/2026-09-25-bilinear-replication-evidence-repair.md)
requires preserving the original source, failed outputs and receipts. A new
aggregate-only script and separately versioned auditor correct the schema and
time comparisons. The trained weights, predictions and fixed quality gate stay
unchanged. The corrected audit must still reconstruct the metrics and lineage;
a plausible explanation of a failed check is not itself a successful audit.
