# Lesson 21: giving the first choice more training weight

**2026-09-24. Three-seed experiment completed; acceptance gate failed.**

[Lesson 20](20-generated-versus-teacher-prefixes.md) separated two problems:
the remaining-set head works better with teacher prefixes than generated ones,
and strong set predictions do not always produce the right ordered next token.
Among 36 answers that lost exactness, correct ordered first choices fell from
34 to 11 even though correct-member first choices changed from 34 to 35.

We tested one focused intervention: increase `model.first_target_loss_weight`
from one to eight while keeping the continuation, prompt-set and symmetric
objectives enabled. The [comparison was declared before training](../experiments/2026-09-24-continuation-order-plan.md).
Eight is a chosen exploratory value, not an estimated optimum.

## Why the first decision can be underrepresented in average loss

Each answer provides one first-target decision but potentially hundreds of later
target decisions. Ordinary token-averaged cross-entropy weights every supervised
token equally. Most of its terms therefore describe continuation along a correct
teacher prefix. A low average can coexist with a weak first choice.

This is a statement about the objective's weighting, not proof that the first
token is always more important. BUDEW in the previous lesson starts correctly
and fails at its second choice. First-target weighting tests one failure mode;
it does not directly supervise every later branching decision more strongly.

## The exact weighted objective

Let \(\ell_{i,t}\) be the next-token cross-entropy for query \(i\) at supervised
position \(t\). Let \(N\) be the number of supervised tokens across the batch,
including EOS, and \(B\) the number of queries with a supervised first target.
For first-target weight \(w\), the implementation computes:

\[
L_{\mathrm{AR}}(w)=
\frac{\sum_{i,t}\ell_{i,t}+(w-1)\sum_i\ell_{i,\mathrm{first}}}
{N+(w-1)B}.
\]

At \(w=1\), this is the ordinary token mean. At \(w=8\), each first-target
term receives weight eight and every other supervised term weight one. The
denominator grows too. The loss is still a normalized weighted mean, not the
old mean plus eight times a separate first-target mean.

For a single example with 128 supervised tokens, the first target has weight
\(1/128\approx0.78\%\) at one and \(8/135\approx5.93\%\) at eight. For
variable-length batches, use their actual \(N\) and \(B\), not padded length.
The relative weight of ordinary token gradients also changes through that
normalization. The total parameter update depends on the other objectives too.

The full recipe is:

\[
L_{\mathrm{total}}=L_{\mathrm{AR}}(w)
+L_{\mathrm{prompt\ set}}+L_{\mathrm{symmetric}}
+L_{\mathrm{continuation}}.
\]

Those three auxiliary coefficients remain one. Their numerical losses and
gradients are not guaranteed to have equal scales merely because the coefficients
match. Increasing first-target weight is therefore an interaction experiment,
not an isolated first-token classifier trained separately from the transformer.

## Where the tensors enter

The model returns logits `[B, T, V]`, with `V=2049` here. Causal alignment uses
`logits[:, :-1]` to predict `labels[:, 1:]`, yielding `[B, T-1, V]` scores and
`[B, T-1]` targets. Labels equal to `-100` are excluded from loss and token count.

For each row, the first nonmasked target identifies its first prediction position.
In this protocol it is the ANSWER hidden state, which predicts the first product.
The gathered first-target logits have shape `[B, 2049]`, and the corresponding
target IDs have shape `[B]`. Their mean cross-entropy is converted back to a
sum by multiplying by the participating query count in the formula above.

This reuses existing logits and adds no parameters. It changes training gradients,
not the five-token prompt or the vocabulary. Canonical teacher order is unchanged.
At inference there are no labels, so this loss weight does not directly bias
the output logits. Our separate alpha-16 first-target guidance remains fixed.
The checkpoint still records weight eight as part of its training identity.

## Two references answer two different questions

| Recipe | Continuation-set weight | First-target weight | Purpose |
| --- | ---: | ---: | --- |
| Original matched control | 0 | 1 | Stronger acceptance reference: 466/666 exact |
| Previous continuation candidate | 1 | 1 | Isolate the effect of changing first-target weight: 463/666 exact |
| This trial | 1 | 8 | New candidate |

We reuse the six frozen reference checkpoints only after checking their bytes,
data, source/runtime and resolved settings. The new runs differ from the previous
continuation recipe only in first-target weight and output paths. Each uses the
same training seed, 2,000 updates, batch32, pinned split and decoding settings.
GPU jobs run sequentially. Nondeterministic CUDA training still means identical
seeds are not a promise of bitwise repeated weights.

The primary gate compares against the stronger original controls: every candidate
must terminate validly without repeated IDs; no seed may regress in processed F1
or exact accuracy; mean exact accuracy must strictly improve; and pooled dual-TYPE
exact answers must exceed 34/204. All changes against the previous continuation
candidate and COLOR regressions remain visible. We do not choose a weaker
reference after seeing the outcome.

## Read the logs without confusing the objectives

`train_loss` includes weighted token loss and the auxiliary objectives. It is
not directly comparable across weights as a model-quality score.
`validation_loss` remains unweighted token cross-entropy, and
`validation_first_target_loss` remains the unweighted first-target mean. The
separate continuation diagnostic also remains unchanged. Complete generated
answers, not any one training loss, determine the acceptance result.

The earlier weight-32 experiment used a recipe without these membership
objectives. Its failure is part of the research record, but it does not settle
whether weight eight works with the current combination. Conversely, reusing
the idea in a different combination is not evidence that it will now succeed.

## Results: stronger fitting did not improve held-out answers

Each cell below counts completely correct processed answer sets. Every seed
evaluates the same 222 validation queries; these are 666 query-seed observations,
not 666 distinct queries. F1 is the mean per-query set F1 across those observations.

| Recipe | Seed 1729 | Seed 1730 | Seed 1731 | Total exact | Mean F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original control, continuation0 / first1 | 156 | 158 | 152 | 466/666 | 90.12% |
| Continuation1 / first1 | 159 | 147 | 157 | 463/666 | 91.52% |
| Continuation1 / first8 | 152 | 153 | 159 | 464/666 | 89.42% |

Against the original controls, 34 previously inexact answers become exact while
36 previously exact answers fail. Against continuation1/first1, 35 improve and
34 fail. That net gain of one against the secondary reference would hide the
decline in F1 and the failure against our declared primary reference.

| Exact sets by query group | Original control | Continuation1 / first1 | First8 trial |
| --- | ---: | ---: | ---: |
| Single TYPE | 129/153 | 132/153 | 131/153 |
| Dual TYPE | 34/204 | 32/204 | 31/204 |
| COLOR | 303/309 | 299/309 | 302/309 |

All 666 candidate answers terminate, obey the protocol, and contain no repeated
IDs. Every quality condition in the declared gate fails: seeds 1729 and 1730
regress against their original controls, overall exactness does not improve,
and pooled dual-TYPE exactness declines. Weight eight is not promoted. The
continuation objective stays experimental and default zero.

## Training fit and generalization are different measurements

**Generalization** means doing well on held-out examples, beyond fitting the
examples used to update the weights. Here is the unweighted first-target CE:

| Seed | Final training batch, first1 → first8 | Validation, first1 → first8 |
| --- | ---: | ---: |
| 1729 | 0.13678 → 0.00740 | 1.46815 → 1.57855 |
| 1730 | 0.09705 → 0.00763 | 1.57759 → 1.93973 |
| 1731 | 0.10953 → 0.00658 | 1.38918 → 1.70212 |

The training column describes the final batch, not a full training-set evaluation.
Still, its improvement alongside worsening validation CE in every seed is
consistent with stronger fitting without the intended generalization gain.
Unweighted validation token CE also worsens in all three seeds. This trial does
not identify whether optimization, limited data, or the objective interaction is
the dominant cause; three seeds do not establish that every larger weight fails.

## Picking a member is easier than starting the correct enumeration

Raw first choices belonging to the correct answer set increase from 646/666
under continuation1/first1 to 657/666 under first8. But correct ordered first
choices fall from 502 to 490. After fixed postprocessing, ordered first choices
fall from 514 to 494. The original controls score 494 raw and 505 processed.

The teacher trains one canonical ordering. A different member may be valid as
an individual answer, yet lead the decoder into a continuation that omits or
adds a whole branch. This is a diagnostic hypothesis about sequence behavior;
our actual acceptance metric remains exact **sets**, so harmless reordering is
not counted as failure.

For ARMALDO / COLOR at seed1730, the original exact answer begins with AGGRON.
The new model begins with ALTARIA and finishes with zero correct products,
170 extra products, and 106 missing products. Its set F1 is zero. A few failures
this large explain why nearly unchanged exact counts can coexist with worse
average F1. Aggregates and concrete failure cases answer different questions.

## What changed in the project

The model code, parameter count, data, and inference settings stayed fixed;
the existing loss-weight override produced three new checkpoints. A new CPU
campaign auditor reconstructs raw and processed metrics for all nine runs and
checks checkpoint bytes, resolved settings, source archives, and paired outcomes.
A separate audit agrees on all 1,998 responses and the gate. Neither CPU audit
reruns the network; the recorded CLI evaluations supply the generated outputs.

Parallel work was useful because training, audit implementation, and independent
review had separate responsibilities. GPU runs remained sequential to avoid
resource contention. The existing reference artifacts retain their original
analysis scripts; the new trial freezes its updated helpers separately.

Observed mean training time was 89.82 seconds for first8, versus 75.98 seconds
for continuation1/first1 and 68.55 seconds for the original controls. These runs
were not an isolated timing benchmark, so the differences cannot be attributed
entirely to the loss weighting. They are recorded costs of these actual runs.

Verification passes: 444 tests, Ruff lint and formatting, strict typing for
59 source files, and documentation build. Six existing documentation links to
source files still produce build warnings. No protected-test evaluation or
serving promotion was performed.

The next research question is whether the decoder can recover complete sets
when its likely early prefixes differ, and whether it can select among those
paths using only learned scores. That calls for a declared prefix/decoding
diagnostic before another coefficient trial. Any oracle-assisted comparison
must remain diagnostic, never a deployable quality result. No new decoder or
improvement is claimed here. Oracle parity, final-test evaluation, and serving
performance at matched quality remain open.

## Evidence to revisit

- [Declared comparison](../experiments/2026-09-24-continuation-order-plan.md).
- [Portable results and receipts](../experiments/2026-09-24-continuation-order-w8.json).
- Full local evidence: `runs/learning/continuation-order-w8-v1/`.
- Official summary SHA256: `0d224d5b7008ef61be27cd26023ae0468222c9201698681f1c04877b491ca602`.
- Independent audit SHA256: `52e8ddb732124398c64920d0061994935306c07015cefd01ca8343b6bcf1297a`.

Self-check: why can a model improve the probability of its training first tokens,
choose a valid first member more often, and still become worse at complete
validation answers? Explain using the weighted loss, held-out CE, canonical
ordering, and the difference between F1 and exact-set accuracy.
