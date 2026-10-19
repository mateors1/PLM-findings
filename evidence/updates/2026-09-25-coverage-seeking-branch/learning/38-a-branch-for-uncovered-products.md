# 38. Starting one path outside existing coverage

**Status: independently audited evidence accepted; fixed quality gate rejected.**
The [plan](../experiments/2026-09-25-coverage-seeking-branch-plan.md) tests one
extra generation path with unchanged weights. The accepted baseline has 603/666
exact validation answers. Full oracle parity remains the objective.

## From an observation to an intervention

[Lesson 37](37-scores-for-missing-members.md) found that most correct products
missing from all eight generated paths still have positive membership scores.
That tells us where to investigate. It does not supply an answer: at inference
time we do not know which absent products are correct.

We can see the union U of products already generated and the head logits z.
Define a set of eligible starting products, excluding subject s:

\[
A=\{i:i\notin U,\ i\ne s,\ z_i>0\}.
\]

If A is nonempty, choose an anchor

\[
a=\underset{i\in A}{\arg\max}\ z_i,
\]

breaking exact ties by smaller ID. If A is empty, keep the old prediction pool.
The rule sees no teacher labels, true answer size or oracle ranks. It runs for
every validation query, including those the baseline already gets right.

An anchor is simply the first product we force the decoder to emit. From there
the model continues greedily using its normal cache, protocol and uniqueness
constraints. We do not forbid previously covered products later in this new
path: those products may provide the familiar continuation needed to generate
a useful complete set. The source union restricts the starting choice only.

## What changes in the tensors

For a batch, head scores Z have shape [B,1025]. A Boolean eligibility mask has
the same shape. Masking ineligible scores to negative infinity and selecting the
largest eligible value yields one anchor per active row, conceptually [B].
An all-ineligible row needs explicit handling; plain argmax would still return
an index even when no valid choice exists.

The decoder still prefills a [B,5] prompt tensor. At its first step we substitute
the anchor for the ordinary token choice. Then the next input has shape [B,1],
containing that emitted ID, while the KV cache retains the prompt history.
Each following step chooses the largest allowed decoder logit.

For this checkpoint family, hidden states have width 256, with eight query
heads and two key/value heads. Each head has 256/8 = 32 channels. Each of the
eight transformer layers retains K and V tensors shaped [B,2,T,32], where T
counts tokens already processed. After the five-token prefill, T=5; processing
the emitted anchor extends it to six. The anchor is first an output choice,
then becomes the next input that extends the cache.

The two scoring spaces also differ. Membership scores have 1,025 product
columns, while decoder logits have 2,049 vocabulary columns, including reserved
protocol positions. Product column c maps to token ID 1024+c. The decoder's
allowed-token and seen-token masks operate in the full vocabulary space; the
per-row seen mask is [B,2049]. Mixing these index spaces would silently select
the wrong product or mask the wrong token.

The forced product counts as one of the 507 allowed completion tokens. It is
marked seen immediately, so the path cannot repeat it. EOS becomes available
under the same protocol rule as before. Prefilling [B,6] or allowing 507 further
tokens would change the execution contract and would not be this experiment.

We retain the original groups of eight, with six in the final group. Rows with
no anchor run a historical starting product as numerical padding but add no
candidate. Finished rows also remain in the tensor batch. This keeps numerical
shapes stable, at a measurable compute cost; it is not a claim of efficient
serving. Padding traces and useful new paths are counted separately.

## One new source gives up to nine new candidates

Let S9 be the new subject-excluded set. Retain the old 36 candidates and append

\[
S_9,\quad S_9\cup S_1,\ldots,S_9\cup S_8.
\]

Active queries therefore have 45 slots. Inactive queries retain 36. The learned
set score remains

\[
F(S)=\sum_{i\in S}z_i.
\]

Old candidates win exact score ties. Adding a candidate cannot lower the best
eligible score, but can lower answer quality: the head can reward a wrong set.
For example,

\[
F(S_j\cup S_9)-F(S_j)=\sum_{i\in S_9\setminus S_j}z_i.
\]

The sum can favor a union containing extra false products. Positive anchor
scores guarantee neither positive scores for the rest of the path nor correct
membership. This is why we measure false positives and false negatives as well
as newly covered truth, and why the quality gate protects every seed.

Every valid active new source must contain its anchor. Because the anchor was
outside U, source coverage must expand by at least that product. This is an
implementation invariant. Whether the expansion is correct is a separate
question answered only during evaluation.

## Controls and a fair failure condition

Before measuring the intervention, the new explicit-first helper must reproduce
all historical rank-one paths when given their historical first products.
Fresh head vectors must also match exactly at the original batch shapes.
The other old paths are authenticated saved evidence; we do not describe them
as newly generated. This separates fresh controls from reused baseline data.

The fixed quality gate requires more than 603 pooled exact answers, more than
155 dual-TYPE exact answers, no overall exact/F1 regression in any seed, no
pooled group exact regression, and valid complete active paths. An improved
score or larger source union cannot substitute for those criteria.

Even if this works, the two existing available-exact selection misses cannot
be repaired just by appending candidates with unchanged scores and old-first
ties. The experiment targets one known coverage limitation; other work remains
necessary for full oracle parity. The audited results below show both
improved coverage and newly introduced selection errors.

The runner passed 28 focused tests, including synthetic CPU decoding controls,
and the independent auditor passed 38 tests, for 66 focused tests in total.
The Torch-free preflight authenticated 47 inputs. The experiment then began
with fresh historical controls. Passing these implementation checks is not a
measurement of improved answer quality.

One timing field keeps the name `set_scoring_seconds` but covers postprocessing,
candidate scoring and per-query diagnostics together. Interpret it as combined
analysis time, not isolated scorer latency. New-path GPU time and padding work
are recorded separately; no serving-performance claim follows from this run.

## The result: higher F1, fewer completely correct answers

All 666 fresh head vectors and forced-first control paths matched their saved
references exactly. Every active new source was valid, terminated and unique,
and every selected answer had eligible sources. Those execution checks passed,
but the fixed quality gate failed:

| Seed | Exact answers, baseline → new | Macro F1, baseline → new | Exact gains / losses | Active anchors, true / false |
| --- | --- | --- | --- | --- |
| 1729 | 201 → 200 | 0.9799255176742276 → 0.9835928374761095 | 1 / 2 | 71, 14 / 57 |
| 1730 | 205 → 202 | 0.9906969338820507 → 0.9906599556215241 | 0 / 3 | 63, 6 / 57 |
| 1731 | 197 → 198 | 0.9798558083418223 → 0.9821032650201781 | 2 / 1 | 71, 10 / 61 |
| Pooled | **603 → 600** | **0.9834927532993669 → 0.985452019372604** | **3 / 6** | **205, 30 / 175** |

Each seed has 222 observations. Pooled group exact counts change from 309 to
306 for COLOR, 139 to 137 for single-TYPE, and 155 to 157 for dual-TYPE. The
dual-TYPE improvement meets one requirement, but pooled exactness, group
nonregression and per-seed exact/F1 nonregression do not all pass. We retain
the failed result; the higher pooled F1 does not override the declared gate.

F1 gives partial credit. Repairing a large incomplete answer can improve its
F1 substantially, while adding one wrong member to a formerly exact large set
reduces F1 only slightly. Exact-answer accuracy counts the latter as a complete
failure. Here precision decreases from 0.99138927159738 to 0.9907678888221955,
while recall rises from 0.9817685798026508 to 0.9853098343640763. This makes the
F1/exactness disagreement understandable without treating either metric as wrong.

## Why the earlier 99.08% did not predict reliable anchors

The previous diagnosis started with products already known, from labels, to be
correct and absent from every source. It measured the empirical fraction

\[
\widehat P(z_i>0\mid i\text{ is an omitted true-member occurrence})
=\frac{1394}{1407}\approx99.08\%.
\]

The new rule must choose without those labels. Among its 205 chosen anchors,
only 30 are true members:

\[
\widehat P(i\in T\mid i\text{ is the chosen anchor})
=\frac{30}{205}\approx14.63\%.
\]

These fractions condition on different events and different populations. The
first pools omitted true-member occurrences in 30 focused observations; the
second measures one selected positive uncovered maximum in each of 205 active
observations drawn from all 666. Neither fraction is the inverse of the other,
and they cannot be plugged together as a direct Bayes estimate. Many false
products can also have positive scores, and selecting a maximum adds another
selection condition. Positive support for known truth does not establish the
precision of a truth-free retrieval rule.

The remaining 461 observations had no eligible anchor and retained their old
candidate pools. All 43 COLOR anchors and 23 single-TYPE anchors were false;
the 30 correct anchors occurred among 139 dual-TYPE anchors. These are
post-selection diagnostics, not permission to introduce subgroup routing into
this frozen experiment.

## More available answers can coexist with worse selection

The expanded pool contains an exact candidate in 608 observations, up from
605. Yet available-exact selection misses increase from 2 to 8, giving
608−8 = 600 selected exact answers. The original two misses remain unresolved,
and six previously exact answers are displaced by wrong higher-scoring sets.
The selector chooses an added slot in 16 observations altogether.

Across active source pools, the new paths add **488 true-member occurrences**
and **13,281 false-member occurrences** outside previous coverage. These are
changes to the union of candidate sources, not false positives in final answers.
Only one candidate is selected per query. The final selected answers contain
1,089 false-positive and 1,632 false-negative occurrences. Full truth containment
in the source union improves from 636 to 642 observations, which still does not
guarantee an exact candidate or correct learned selection.

For a useful gain, return to `PKM_AEGISLASH TYPE SAME`, seed 1729, from Lesson 37.
The anchor is its missing true product ID 1040. The selected answer grows from
124 correct members to the exact 125 through slot 39, the union of old source
2 and the new path. This realizes the motivating possibility in one case.

For a loss, consider `PKM_ABSOL TYPE SAME` under the same seed. The old selected
set was exactly correct with 68 members. Anchor ID 1896 is false, and the new
original source at slot 37 wins with 69 members: all 68 correct products plus
that one extra. No true member was lost, but the answer is no longer exact.
The two examples show why increasing coverage and rewarding positive additions
need a separate test of complete-answer quality.

## What the additional computation actually cost

Times below are measured sums, rounded to milliseconds. Each phase preserves
the original batch shapes, including inactive-anchor rows:

| Seed | Fresh control generation | New-anchor batch generation | Active / inactive rows |
| --- | ---: | ---: | --- |
| 1729 | 41.880 s | 34.848 s | 71 / 151 |
| 1730 | 36.685 s | 36.312 s | 63 / 159 |
| 1731 | 35.056 s | 39.809 s | 71 / 151 |
| Pooled | 113.621 s | 110.970 s | 205 / 461 |

The anchor batches process 129,656 padded decode positions: 39,381 belong to
active rows and 90,275 to inactive-anchor padding rows. Their retained
completion traces contain 25,712 active emitted tokens and 55,949 padding
tokens, including EOS where present. Finished-row padding inside the tensors
is distinct from these inactive-anchor traces. We cannot split the measured
110.970 seconds into active versus padding seconds from these counts alone.

Fresh controls produce 666 paths and 80,422 emitted completion tokens; these
are verification work, not extra prediction candidates. Separate prompt-head
passes total 1.463 seconds. The combined `set_scoring_seconds` field totals
0.272 seconds for postprocessing, scoring and diagnostics. Overall wall time
is 236.494 seconds, including loading and other overhead. Each generation batch
also retains the archived first-step guidance forward, although the explicit
first ID bypasses its ranking; no later guidance is added.

The eight old source paths are reused, so this is not an end-to-end timing of
a nine-source deployment. LM Studio was preserved; timings remain descriptive,
with no throughput, energy, concurrency or serving-default claim.

The [portable result](../experiments/2026-09-25-coverage-seeking-branch.json)
records the accepted evidence and rejected quality gate. The primary summary is
`runs/learning/coverage-seeking-branch-v1/summary.json`, SHA256
`15a4d408fe59c53ed76bcdac8588403677342bd7af5df80691a2461a0667676c`.
Its three seed reports were hash-checked for this lesson. The independent audit
passed, and the owner decision accepted the evidence while rejecting the fixed
quality gate. Auditing confirms the recorded measurements and reconstruction;
it does not turn a failed quality result into an accepted prediction policy.
No weights, protected-test predictions or production defaults changed.
