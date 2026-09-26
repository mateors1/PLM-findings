# Worst-member bilinear training improves one count but fails the quality gate

**Status: independently audited evidence accepted; fixed quality gate rejected.**
The new worst-member objective reaches 208/222 exact validation sets, one more
than its balanced-BCE sibling. Macro F1 falls from 0.9997456354 to 0.9996521512,
and single-TYPE exactness falls from 50 to 49. The declared gate therefore
rejects the child despite its higher total exact count. Both failed requirements
remain part of the result; success on another metric cannot compensate. The
[independent audit](../evidence/updates/2026-09-25-bilinear-worst-boundary/campaign/independent-audit.json)
and [separate owner decision](../evidence/updates/2026-09-25-bilinear-worst-boundary/campaign/decision.json)
accept the completed evidence while rejecting promotion under this quality gate.

The [frozen plan](../evidence/updates/2026-09-25-bilinear-worst-boundary/experiments/2026-09-25-bilinear-worst-boundary-plan.md)
changes the training objective while retaining the bilinear parameterization,
original seed-1729 parent, optimizer and 2,000-update budget. The accepted
[balanced-BCE sibling](22-bilinear-training-budget.md) supplies the historical
comparator. It is neither rerun nor resumed. This single-seed adaptive screen
does not alter its prior acceptance or the subsequent replication record.

## Changing which members drive the gradient

The [saved-score diagnosis](24-bilinear-error-geometry.md) found mistakes with
both incorrect ordering and incorrect placement around zero. It motivated
studying the weakest true member and strongest false member, without proving
that average loss caused the errors. An earlier
[projection-only worst-member screen](19-worst-boundary-projection-refit.md)
failed its gate and reduced strict separation relative to mean fitting. That
negative precedent is retained rather than treated as evidence of success.

For each training query q, deduplicate true products T_q and form false products
N_q, excluding the SAME subject. Scores Z[1637,1025] cover 1,637 training queries
and 1,025 products. Define

\[
a_q=\min_{i\in T_q}Z_{qi},\qquad b_q=\max_{i\in N_q}Z_{qi},
\]

\[
L_{\mathrm{worst}}=\frac{1}{1637}\sum_q
\frac{\operatorname{softplus}(-a_q)+\operatorname{softplus}(b_q)}{2}.
\]

Softplus(x)=log(1+exp(x)). Lowering this loss pushes a weak true score upward
and a strong false score downward. Unlike class-average BCE, only extremal
members supply gradients within each class at a particular update. Masked
amin/amax distributes gradients across exact ties. Control/padding masks,
label deduplication and nonself exclusion remain unchanged, and empty classes
are rejected. Labels use vocabulary IDs; excluded subjects use product-column
offsets, vocabulary ID minus 1024.

There is no mean-loss mixture, temperature or added margin. The frozen
worst-loss implementation is imported by checked hash and passed as an explicit
callback to the unchanged bilinear fitting loop. New orchestration owns its
objective-specific metadata and gate; it does not patch older runners.

## Fixed representation and fit

All 93 original parent state tensors remain frozen. The only trainable tensor
is A[2,256,256], initialized to zero. The full token embedding table has shape
[2049,256]. Its selected product rows are normalized and rescaled into features
E[1025,256], with E_i=16*w_i/||w_i||. The scorer remains

\[
M_d=\frac{A_d+A_d^\top}{2},\qquad
z_{sdi}=z^{\mathrm{parent}}_{sdi}+\frac{E_s^\top M_dE_i}{16}.
\]

The fit uses fresh non-fused AdamW for exactly 2,000 full-batch updates at learning
rate 0.0003 and zero weight decay. FP32 operation order, numerical settings,
subject exclusion and the fixed z>0 selection rule stay unchanged. No scheduler,
clipping, accumulation, intermediate validation or checkpoint selection is added.
Zero A exactly replays the parent validation scores at the historical batch shapes.

With fixed features, each score is affine in A and the worst-member softplus
objective is convex in A, though potentially nondifferentiable at ties.
Convexity does not establish finite-budget convergence or generalization.
Equal update counts and AdamW settings also do not imply equal optimization
difficulty for two different losses.

## Training loss and the separate BCE diagnostic

| Measurement | Initial | Final |
|---|---:|---:|
| New worst-member training loss | 0.44132092595100403 | 0.003092526225373149 |
| New child's ordinary mean BCE diagnostic | 0.003822767175734043 | 0.00016030404367484152 |
| Historical balanced-BCE sibling's training loss | 0.003822767175734043 | 0.00003491543247946538 |

The worst-member objective decreases substantially. The new child's ordinary
BCE also decreases from its own starting value, but finishes higher than the
balanced-BCE sibling's endpoint. This differs from the earlier projection-only
worst-member screen, where ordinary BCE rose during fitting. Distinct objectives
have distinct scales: comparing the numerical size of worst loss directly with
BCE would not measure which model is better. The historical sibling's final
worst-member loss was not measured in this screen, so no such comparison is made.

Mean BCE is measured only before fitting and after update 2,000 under no_grad.
It supplies no update gradient, stopping decision or acceptance criterion.
The exact initial BCE replay is an invariant; the initial worst loss is newly
measured. The full [training trajectory](../evidence/updates/2026-09-25-bilinear-worst-boundary/experiments/2026-09-25-bilinear-worst-boundary-training.json)
retains every worst-loss pre-update observation and the final post-update value.

## The stricter sibling comparison rejects this result

| Predictor | Exact / 222 | Macro F1 | Extra members | Missing members | Strict separation / 222 |
|---|---:|---:|---:|---:|---:|
| Original dense parent | 112 | 0.9903735063 | 512 | 76 | 165 |
| Historical balanced-BCE bilinear sibling | 207 | 0.9997456354 | 8 | 10 | 217 |
| New worst-member bilinear child | 208 | 0.9996521512 | 5 | 15 | 215 |

Relative to the historical sibling, the child gains four exact answers and
loses three. Dual TYPE gains three and loses one; single TYPE gains one and
loses two. COLOR is unchanged. Against the original dense parent it gains 96
and loses none. Against the historical width-eight predictor it gains 14 and
loses seven; that is descriptive context, not the current acceptance threshold.

| Group | Child exact | Required floor | Check |
|---|---:|---:|---|
| COLOR | 103 / 103 | 103 | Pass |
| Single TYPE | 49 / 51 | 50 | Fail |
| Dual TYPE | 56 / 68 | 54 | Pass |

All 222 outputs satisfy serialization bounds and all primary execution invariants
pass. Exact count strictly exceeds 207. However, both the macro-F1 floor and
the single-TYPE group floor fail. There are fewer extra members but more missing
members, and strict separation decreases by two queries. A one-answer net gain
does not establish a uniformly better relational policy.

## Lineage, verification and limits

The final child is a full 94-state checkpoint, with the original 93 tensors
unchanged and trained A added. Its new objective is
plm-bilinear-worst-boundary-softplus-v1 and its evaluator is
plm-bilinear-worst-boundary-screen-v1; the scorer and architecture identities
remain unchanged. Parent pretraining steps and residual updates remain separate
fields. The audited final checkpoint is additively registered as checkpoint 34;
rejected quality is not a reason to discard trained evidence. The inventory
freshly hashes all 34 final weight files, without revalidating every older
payload or establishing durable remote storage.

The primary passed 144 focused CPU tests, including 32 new tests and 112 reused
objective, arithmetic and lifecycle tests. The independent auditor passed 68
synthetic tests before execution. Both actual processes completed successfully
on their first attempt. Its saved-evidence and CPU checkpoint checks
do not independently replay CUDA fitting or regenerate neural scores. Explicit
loss-helper and scorer hashes distinguish what computed the objective from what
computed predictions.

The approximately 5.23-second fit and 12.82-second primary wall time are descriptive
local durations, not matched timing controls or serving/energy claims. Full
weights and head vectors remain hash-bound local dependencies. The
[evidence inventory](../evidence/updates/2026-09-25-bilinear-worst-boundary/README.md)
preserves the exact source, tests, receipts, lineage and archival boundary.
No threshold, learning rate, budget or gate is retuned after this outcome;
there is no additional-seed replication, protected-test evaluation or promotion.
The ordinary inherited decoder still ignores A and does not serve this scorer.
