# Continuation objective with stronger first-target loss

Declared 2026-09-24 after the matched-prefix diagnosis, before training this trial.

## Hypothesis and intervention

Among 36 answers that lost exactness under continuation-set training, ordered
raw first choices fell from 34 to 11, while correct-member first choices rose
from 34 to 35. Test whether modestly increasing ordered first-target supervision
improves full generated answers while retaining the learned remaining-set head.

Use the existing `model.first_target_loss_weight=8.0` on `continuation_set_lab`.
Prompt-set, symmetric and continuation coefficients all remain one. Weight eight
is the sole declared exploratory candidate, not a claimed optimum. The old
weight-32 experiment used `national_dex_v1` with no prompt-set objective, so this
is a different interaction. Do not treat a failed earlier recipe as proof of
success or failure here.

The implementation weights first-target token losses within the normalized
autoregressive mean: `(sum(all token losses) + 7*sum(first losses)) /
(supervised token count + 7*query count)`. Auxiliary losses remain added with
their existing coefficients. No new parameters, token vocabulary, teacher
permutation, graph input or inference rule is introduced.

## Fixed comparison

Run three new checkpoints, seeds 1729/1730/1731, at 2,000 optimizer updates,
batch32 and the same pinned dataset/split, optimizer, learning-rate schedule,
precision and hardware environment. Use unique run names
`national_dex_continuation_order_w8_s{seed}_v1`. GPU jobs run sequentially.

The authoritative reference is the complete six-run comparison under
`runs/learning/continuation-set-v1/`, summary SHA256
`558d18391131333461efee09c3045848a8ffbca7a6a5e4b3d436ac7d49fbf819`.
Reuse its frozen controls only if source/runtime/data and every unrelated
resolved setting match. Existing source archive is
`101d661a16e3f17ecc924c1afed3a188cea25fe8e70bb1df2a19b523f363ecd1`.
If implementation changes become necessary, stop this comparison before training
and declare fresh matched controls rather than mixing changed source silently.

Compare each new run with (a) continuation1/first1 to isolate the changed weight
and (b) continuation0/first1 as the stronger acceptance reference. The former
has 463/666 exact answers, the latter 466/666. CUDA runs are nondeterministic;
record seeds and identities without claiming bitwise repetition. Do not choose
only a favorable seed or substitute historical weaker checkpoints.

Evaluate all 222 validation queries with alpha16, uniqueness, KV cache, offline
batch8 and bound507. Preserve complete raw IDs, processed answers, raw/processed
first-choice accuracy, membership versus ordered first choice, F1 and exact
counts, TYPE single/dual and COLOR, and paired gains/losses against both references.
Record token/first-target/auxiliary losses and observed training time separately.

## Acceptance and limitations

Against the original continuation0/first1 controls, require every candidate
answer to be valid, terminated and unique; no seed to regress in overall
processed F1 or exact accuracy; mean overall exact accuracy to strictly improve;
and pooled dual-TYPE exact answers to strictly improve beyond 34/204. Report all
regressions against continuation1/first1 and COLOR tradeoffs even if this gate
passes. No coefficient search or post hoc threshold change belongs to this trial.

This tests one substantial failure mode. BUDEW/EKANS can choose the wrong second
product even after a correct first product, so stronger first-target loss is not
a proposed complete solution to continuation. Lower loss or stronger head F1
cannot replace the complete-generation gate. No protected-test use, HTTP
promotion, oracle repair or serving-performance claim is included.
