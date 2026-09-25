# Early continuation-set supervision — declared next experiment

2026-09-24. Selected after the frozen union-coverage diagnosis; not yet
implemented or run. Preserve the existing protocol and data snapshot.

## Hypothesis and intervention

Dual-TYPE generation returns exactly one branch in 75/175 failed observations.
The static symmetric head recognizes 99% of omitted dual-TYPE targets, while
local teacher-prefix accuracy is weakest near the start. Test whether supervising
the remaining set at early continuation states improves the complete union.

Add a default-zero `model.continuation_set_loss_weight`. When positive, reuse
the existing contextual prompt-set projection on hidden states after one, two
and three supplied teacher targets (positions 5, 6 and 7). Require the existing
prompt-set objective to be enabled; no new learned parameters are needed for
the selected recipe. Predict the still-unemitted teacher target set, using only
product columns. Past targets and the subject are negatives. Skip positions
with no remaining product targets; the ordinary causal loss still teaches EOS.

Use class-balanced binary loss per valid position, average positions within
each query, then average participating queries. Coefficient 1 is the sole
initial candidate. Keep the original prompt-set and symmetric coefficients at
1. Keep logits causal: later targets may construct training labels but must not
enter earlier hidden states. Labels never enter inference. Record this auxiliary
loss separately from token cross-entropy and other objectives.

Default zero must preserve the old computation and parameter initialization.
Check target shifts, causal future-token invariance, query weighting, short and
padded sequences, gradients, checkpoint identity and exact deterministic resume
on a meaningful toy fixture. Maintain Torch-free core imports.

## Comparison

After implementation and focused/full checks, freeze the source and execute
fresh control and candidate runs at training seeds 1729, 1730 and 1731. Both
arms use the existing pinned National Dex data, split seed 1729, 2,000 optimizer
updates, batch 32 and all other symmetric-relation recipe settings unchanged.
Controls use coefficient zero; candidates use one. Use unique new run paths.
Record actual source/config/checkpoint/runtime identities. CUDA training remains
subject to the existing nondeterministic setting; seeds are not a promise of
bitwise reproducibility. GPU jobs run sequentially.

Use all 222 validation queries per checkpoint with the already verified
alpha-16 first-target guidance, unique-target masking, KV cache, batch size eight
and completion bound 507. Hold inference policy fixed. Preserve full raw tokens,
processed targets, validity, EOS, repeated-ID counts and per-query metrics.
Report paired raw/processed F1 and exact answers, including TYPE single/dual,
COLOR, and each query gained or lost. No classifier-only success criterion.

Accept for further integration consideration only if all candidate outputs are
valid, terminated and repeat-free; no seed regresses in overall processed F1 or
exact accuracy; mean overall exact accuracy strictly improves; and pooled dual-
TYPE exact answers strictly improve. Report COLOR regressions and other subgroup
tradeoffs even when the overall gate passes. This is not an oracle-quality gate.
Failed runs or tradeoffs remain evidence; do not rewrite this comparison after
observing outcomes. Further weight changes require a separately declared trial.

No protected final-test evaluation, HTTP promotion, source-model optimization,
protocol ordering change or serving-performance claim is included in this trial.
