# Lesson 9: knowing how to continue is not knowing how to start

**2026-09-24.** The native service preserves the model's predictions. We now
return to the quality question: why are many complete answers still wrong?
This experiment holds learned weights fixed and changes the information supplied
at inference. The [declared plan](../experiments/2026-09-24-sequence-diagnosis-plan.md)
specifies the validation-only intervention before running it.

**Result:** the first decision is a major bottleneck, and sparse TYPE answer
families remain difficult even after a correct opening hint. This changed our
diagnosis, not the trained model or the production API's quality.

## What the experiment found

All 222 zero-hint responses exactly reproduced the archive. The same checkpoint
then received one correct first target per query. The
[result and identities](../experiments/2026-09-24-sequence-diagnosis.json) record:

| Measurement | No oracle hint | Correct first target supplied |
| --- | ---: | ---: |
| Raw generated F1 | 62.90% | 92.97% |
| Processed F1 | 63.19% | 93.48% |
| Processed exact answers | 105/222 (47.30%) | 174/222 (78.38%) |
| TYPE processed exact | 32/119 | 71/119 |
| COLOR processed exact | 73/103 | 103/103 |
| Valid EOS termination | 222/222 | 221/222 |

The intervention rescued 69 previously incorrect queries and lost none of the
105 prior successes. **All 48 remaining failures are TYPE queries.** One,
INKAY/TYPE, failed to terminate within the 507-token completion bound. It remains
an invalid, truncated response; the evaluator does not manufacture EOS or repair
it with post-processing. Even an oracle hint can move decoding into a bad path.

Teacher-forced accuracy separates the opening decision from continuation:

| Predicted position | Correct / total | Accuracy |
| --- | ---: | ---: |
| First target | 108/222 | 48.65% |
| Remaining target tokens | 27,814/28,221 | 98.56% |
| EOS | 221/222 | 99.55% |

These are different denominators: first-target accuracy counts queries; the
middle row counts tokens and consequently gives long answers more weight.
98.56% does not mean 98.56% of complete answers are correct. Nor should those
conditional token accuracies be multiplied as if errors were independent and
all queries had the same length.

## Familiar answer families versus sparse ones

| Matching training queries for an answer family | Validation queries | Exact without hint | Exact with hint |
| --- | ---: | ---: | ---: |
| None | 3 | 0 | 0 |
| 1–4 | 51 | 0 | 12 |
| 5 or more | 168 | 105 | 162 |

This association is striking: all unassisted exact answers occur in the most
supported group. After a hint, 162/168 of those queries are exact, but only
12/54 in the two sparse groups are. The groups also differ in dimension and
answer structure; these counts do not isolate training frequency as a cause.
The three-query group is particularly small. Increasing examples by duplicating
the same rows would not establish new relational information.

Some residual errors are small omissions; others miss large parts of a union.
For example, the hinted AEGISLASH/TYPE answer misses 61 correct products with no
extra products, while ABSOL/TYPE misses one. The per-query failure list preserves
both patterns instead of reducing all 48 failures to one assumed cause.

## A sequence is a chain of conditional predictions

Let `x` be the five-token prompt and `y_1 ... y_n` the correct answer, including
EOS. Autoregressive probability factorizes as:

```text
P(y_1, ..., y_n | x) = product_t P(y_t | x, y_1, ..., y_(t-1))
```

During training, each term receives the correct preceding tokens. This is
**teacher forcing**. During greedy generation, the previous tokens are the
model's own choices. If its first choice starts the wrong familiar list, later
choices can be locally plausible while the complete relationship is wrong.

The compiler's order is meaningful here: shared attribute count descending,
confidence descending, then product key ascending. A dual-type subject's first
target can reveal more about the intended list than an arbitrary member would.
The model sees only product IDs; these ordering rules still create structure
in the training sequences.

## Three measurements answer different questions

1. **Teacher-forced token accuracy:** with all earlier correct tokens supplied,
   does the model predict the next one? Count the first target, remaining targets
   and EOS separately. Otherwise thousands of easy continuation tokens can hide
   a weak opening decision.
2. **Unassisted generation:** supply only the real prompt and let every answer
   token be predicted. This is the model-quality path we already report.
3. **One-token oracle intervention:** supply the correct first target, then
   generate the rest. This diagnoses a bottleneck; it is not deployable quality.

The diagnostic also applies exactly the existing deterministic post-policy to
each complete output. It cannot use graph facts to remove wrong relations or
fill missing products. It retains raw output alongside the processed version.

## Tensor coordinates and the off-by-one trap

For a padded diagnostic batch:

```text
input_ids:                 [B, T]        integer IDs
decoder hidden states:     [B, T, 256]
next-token logits:         [B, T, 2049]
logits[:, 4, :]:            [B, 2049]    predicts first target after ANSWER
logits[:, 5, :]:            [B, 2049]    predicts second target
```

For a record with `m` product targets, the correct labels including EOS are
`input_ids[5:]`. Their matching predictions are `logits[4:4+m+1]`. We take argmax
over the vocabulary for teacher-forced accuracy. This accuracy is unmasked;
the free-generation paths use the usual protocol syntax mask.

For one oracle token, cache prefill receives `[1, 6]` instead of `[1, 5]`:

```text
[BOS, SUBJECT, DIMENSION, SAME, ANSWER, correct_first_target]
```

The resulting final-position logits choose the second target. Each later call
uses a `[1, 1]` token chunk and the existing request-local KV cache. We supply
no additional oracle tokens, set labels, graph attributes or predicted length.

The injected token consumes one position in the original completion budget.
If `M` is the total completion bound and `k` is the number supplied, the loop
may predict at most `M-k` tokens, including EOS. It never manufactures EOS on
truncation. A small scripted-model test guards this accounting.

## How much information is in one hint?

One entity ID is not one bit. With 1,025 products there are up to
`log2(1025) ≈ 10` bits in choosing an arbitrary ID, although the actual entropy
depends on which IDs occur and what the prompt already reveals. The particular
first ID can act as a cue for an entire answer family. Therefore a large gain
from the intervention would not prove the model internally knew all correct
relations before the hint.

We group queries using the set fingerprint `targets ∪ {subject}` and count how
many training queries share that fingerprint and dimension. Adding the subject
undoes SAME's self-exclusion for this diagnostic comparison. This measures
training support for a whole answer family. It does not recover hidden attribute
labels or prove that the model represents those attributes internally.

## Keeping the experiment honest

The diagnostic first verifies checkpoint, corpus, split and archived report
identity. Its zero-hint decoding loop must reproduce every archived raw answer
before the intervention summary is accepted. That controls for accidental changes
in masking, cache handling, EOS or token budget.

Only the same 222 validation queries are evaluated. Final test is untouched.
The source archive, analysis script, input hashes and complete responses are
retained. This is a single-checkpoint diagnosis, not a new training candidate,
checkpoint selection rule, cross-seed replication or serving benchmark.

## What this changes about our next experiment

The evidence supports improving prompt-to-answer conditioning and rare TYPE
continuation, rather than assuming the service or cache caused the quality gap.
It does not justify deploying oracle hints. Lesson 5 already showed that simply
upweighting the first target by 32 made unassisted generation worse; identifying
a bottleneck does not prove a particular remedy will fix it.

The training log also shows validation set loss worsening late in training while
training loss continues falling. Before adding capacity, compare the saved
checkpoints using complete unassisted answers and a declared selection rule.
An earlier checkpoint might generalize better; the loss trend alone cannot
establish that. Keep the final test reserved until the procedure is selected.
