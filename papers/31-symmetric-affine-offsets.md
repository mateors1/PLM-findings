# Symmetric affine offsets: lower training loss, unchanged validation answers

The completed `symmetric-affine8000-v3` experiment added learned symmetric linear
terms and a bias to the bilinear scorer. After 8000 real optimization updates,
it returned **216 exact answers out of 222**, with macro F1
**0.9998843626799785**. Both equal the accepted 8000-update bilinear control.
The predeclared gate required **strictly more than 216 exact answers**;
therefore this is an audited negative quality screen. The evidence is accepted,
but the candidate is not promoted. [Summary](../evidence/updates/2026-09-26-symmetric-affine8000-v3/artifacts/runs/learning/symmetric-affine8000-v3/summary.json),
[independent audit](../evidence/updates/2026-09-26-symmetric-affine8000-v3/artifacts/runs/learning/symmetric-affine8000-v3/independent-audit.json),
[owner decision](../evidence/updates/2026-09-26-symmetric-affine8000-v3/artifacts/runs/learning/symmetric-affine8000-v3/decision.json).

## Why add offsets?

Earlier saved-score geometry found that one shared scalar threshold could not
make every dual-TYPE answer exact for any of the inspected fixed seed/budget
scores. That result motivates testing a richer learned scoring function; it
does not prove that offsets will help. The affine hypothesis lets training move
both the decision boundary and candidate rankings while retaining pair symmetry.
The [frozen scientific plan](../evidence/updates/2026-09-26-symmetric-affine8000-v3/artifacts/runs/learning/symmetric-affine8000-v3/summary.plan.md)
records this rationale before candidate quality was observed.

For subject $s$, candidate $i$ and steering dimension $d$, the score is

$$
z(s,i,d)=z_{\mathrm{parent}}(s,i,d)
+\frac{E_s^\top\operatorname{sym}(A_d)E_i}{16}
+\frac{u_d^\top E_s+u_d^\top E_i}{16}+b_d,
\qquad \operatorname{sym}(A)=\frac{A+A^\top}{2}.
$$

The frozen embedding table $E$ has shape $[1025,256]$; its rows are 16 times
normalized parent embeddings. The trainable tensors are $A:[2,256,256]$,
$u:[2,256]$ and $b:[2]$. The two dimension rows serve TYPE and COLOR. The affine
terms add 514 stored scalars beyond the bilinear matrix, giving 131586 stored
trainable scalars in total. Only the symmetric part of $A$ affects scores, so
stored size is not the number of independent functional degrees of freedom.

For one query, $u_d^\top E_s/16+b_d$ shifts all candidate scores equally and can
move items across zero. The candidate term $u_d^\top E_i/16$ varies by candidate
and can change their order. Swapping subject and candidate leaves the mathematical
score unchanged. Floating-point paths are tested with declared tolerances.
Candidate projections can be computed as $Q=uE^\top/16:[2,1025]$, then gathered
into $[B,1025]$ without building a $[B,1025,256]$ intermediate.

## What was held fixed

The fit restarted all three residual tensors at exact zero from the original
seed-1729 parent. It did not continue the already trained bilinear child.
All 93 original state tensors remained byte-identical; the resulting checkpoint
contains those 93 states plus three trained residual tensors. Parent pretraining
was 2000 steps; this experiment's residual fit was a separate 8000 updates.

The fixed data boundary was 1637 training queries and 222 validation queries
over 1025 products. The restricted adapter reconstructed only authenticated
training/validation membership records. Corpus and split identities were inherited
from authenticated evidence; the whole corpus was not reopened or revalidated,
and protected examples were not used. The [input amendment](../evidence/updates/2026-09-26-symmetric-affine8000-v3/artifacts/runs/learning/symmetric-affine8000-v3/summary.input-amendment.md)
explains this stronger input boundary.

Optimization used full-batch balanced binary cross entropy, fresh AdamW with
learning rate 0.0003, betas $(0.9,0.999)$, epsilon $10^{-8}$ and zero weight decay.
The fit used FP32 without autocast, scheduling, clipping or accumulation.
Validation occurred at zero initialization and after the final update; there
was no validation threshold search. Inference selected nonself products with
$z>0$ and serialized them in ascending entity-ID order, with no truncation,
fallback, graph lookup or oracle answer count. The [recipe](../evidence/updates/2026-09-26-symmetric-affine8000-v3/artifacts/runs/learning/symmetric-affine8000-v3/summary.recipe.json)
binds the remaining numerical settings and all fixed gates.

## Observed result

| Measure | Historical bilinear, 8000 updates | Affine, 8000 updates |
|---|---:|---:|
| Exact answers | 216/222 | 216/222 |
| Macro F1 | 0.9998843626799785 | 0.9998843626799785 |
| COLOR exact | 103/103 | 103/103 |
| Single-TYPE exact | 51/51 | 51/51 |
| Dual-TYPE exact | 62/68 | 62/68 |
| Strictly separated queries | 220/222 | 220/222 |
| False negatives / false positives | 8 / 1 | 8 / 1 |
| Final training BCE | 1.6234706663453835e-6 | 1.5944932556521962e-6 |

The [historical audit](../evidence/updates/2026-09-26-symmetric-affine8000-v3/artifacts/runs/learning/bilinear-budget8000-v1/independent-audit.json)
and affine audit support this comparison. Direct saved-output comparison also
confirms that **all 222 selected sets are identical**, with aligned query identities,
prompts and expected sets. The [portable comparison record](../evidence/updates/2026-09-26-symmetric-affine8000-v3/artifacts/docs/experiments/2026-09-26-symmetric-affine8000-v3.json)
binds both child-report hashes. The candidate has zero paired exact answer gains
and zero losses. The F1 floor, group floors and serialization
checks pass, but they do not compensate for the failed strict-improvement gate.
All six incomplete answers remain in dual-TYPE.

Training loss decreased slightly more with the added parameters. This does not
establish a validation gain: BCE responds continuously to score changes, whereas
the selected set changes only when a score crosses zero. A lower training loss
can therefore coexist with unchanged answer quality. This result does not prove
that affine terms are useless across other seeds, objectives or training budgets;
it rejects their promotion under this declared screen.

The audited refit took approximately 70.41 seconds on the recorded RTX 5070 Ti
setup. This is the reported optimization duration, not serving latency,
throughput, energy per request or an independently matched speed comparison.

## Execution evidence and predecessor failures

V1 failed configuration admission before training. V2 passed admission but a
missing evaluator argument stopped execution before predictions or optimization.
Those remain separate [runtime](29-affine-runtime-admission.md) and
[lifecycle](30-affine-lifecycle-failure.md) implementation failures, not negative
quality experiments. V3 changed the implementation wiring and evaluator identity;
it retained the scientific plan, scorer, data boundary and quality gate.

The V3 primary CPU suite passed 207 tests, and the independent auditor suite
passed 132, with zero skips. The orchestration integration test used a bounded
fitting double: one real synthetic AdamW step followed by a simulated history.
It tested wiring and checkpoint round trips, not 8000-update numerical behavior.
The production execution and later independent audit provide the separate evidence
that the actual candidate completed all 8000 updates. The audit checked the saved
checkpoint on CPU, original-state preservation, three optimizer states at step
8000, initialization replay, metrics and the fixed gate. Terminal receipts bind
observed exit zero to the summary and audit. Loss/update receipts and reload
chronology are authenticated; the auditor did not regenerate training, replay
the optimizer trajectory or recompute saved logits with a neural forward pass.

## Provenance and limits

The new checkpoint is
`17d59e7aeae81a88914a523c76fd27e8fd33909a5694d28d3913a8240bc82220`.
The architecture is `plm-frozen-symmetric-affine-residual-v1`, inference is
`plm-symmetric-affine-positive-set-v1`, objective is
`plm-affine-residual-balanced-bce-v1`, and evaluator is
`plm-symmetric-affine8000-screen-v3`. These identities are distinct from the
publication commit and from the historical bilinear checkpoint.

The [evidence bundle](../evidence/updates/2026-09-26-symmetric-affine8000-v3/README.md)
contains exact bounded copies, test receipts, declarations and source archives.
Four exact 2000-row trace chunks retain all 8000 history rows; the separately
versioned inventory contains 38 checkpoints, preserving the previous 37 entries
and registering the newly trained negative-result candidate.
Weight/optimizer payloads and large raw score reports are excluded. Their recorded
hashes identify local dependencies; they do not establish durable remote backup.

This is one adaptively chosen seed-1729 validation screen. The same validation
questions already informed model development, so this is not an unbiased
protected-test estimate or evidence of generalization. It does not establish
oracle parity. Standard decoder serving does not execute this standalone affine
scorer (`standard_serving_supported=false`); no serving policy is promoted.
