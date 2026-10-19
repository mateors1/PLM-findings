# 36. Missing members, extra members and what set operations can recover

**Status: completed; independent audit passes, but the fixed quality gate fails.**
The [experiment plan](../experiments/2026-09-25-eight-source-set-algebra-plan.md)
separates a prediction experiment from a diagnosis that uses the correct answers.
Both reuse saved Pokémon outputs. No weights or generated paths change.

## Why different errors need different operations

Suppose the correct answer is {Pikachu, Raichu, Pichu}. One path gives
{Pikachu, Raichu}, while another gives {Raichu, Pichu}. Their union is exact:

\[
A\cup B=\{x:x\in A\;\text{or}\;x\in B\}.
\]

Now suppose each path contains all three correct Pokémon plus a different
unwanted one. Their union retains both extras; their intersection removes both:

\[
A\cap B=\{x:x\in A\;\text{and}\;x\in B\}.
\]

These are illustrative sets, not measured model answers. The lesson is that
union can repair missing coverage, while intersection can remove disagreement.
Neither operation knows which members are correct. Agreement on the same wrong
member survives intersection; a correct member produced by only one path is lost.

If a correct Pokémon is absent from every generated path, neither unions nor
intersections can invent it. With source sets S_1 through S_8, every result
constructed using just these operations stays inside their total union U:

\[
U=\bigcup_{r=1}^{8}S_r,\qquad C\subseteq U.
\]

Thus a teacher set T with T not contained in U requires some new source of
membership. Passing the coverage check T contained in U is only necessary:
unwanted members or restricted combinations can still prevent an exact answer.

The audited eight-branch baseline already places a ceiling on this approach.
Thirty of its 666 observations lack at least one true member from U. Two more
have an exact candidate that loses under the existing scorer. Appending another
copy of that exact set gives the same score and later tie position, so it cannot
rescue either miss. Any expansion made solely from these sources, retaining
the same scorer and old-first ties, therefore reaches at most 666−30−2=634 exact
answers. This is an upper bound, not a measured result or an achievable promise.
Reaching overall oracle parity will require changing more than these operations.

## What the predictor is allowed to know

The fixed experiment retains the current 36 candidates, then appends the 28
pair intersections in rank order. It selects from 64 slots using the same
learned head. It never looks at the teacher set to choose an operation or a pair.
Existing candidates stay first so an equally scored addition cannot change a tie.

Let M have shape [B,8,1025], with a Boolean entry indicating whether a source
contains a product. For a pair (r,s), intersection membership is

\[
I_{brsj}=M_{brj}\land M_{bsj}.
\]

The new pair masks have conceptual shape [B,28,1025]. Concatenating them with
the old masks produces [B,64,1025]. The unchanged head Z remains [B,1025],
and candidate scores have shape [B,64]:

\[
s_{bk}=\sum_j C_{bkj}Z_{bj}.
\]

The saved-output implementation operates on canonical sorted ID sets and
`math.fsum`; it does not allocate these conceptual dense tensors or run another
neural forward pass. Duplicates remain distinct slots with the same membership.
Empty intersections remain eligible empty sets with score zero. They are
postprocessed sets, not fabricated raw sequences with an EOS token.

## Why intersection might win the unchanged score

For any additive set score s, exact arithmetic gives

\[
s(A\cup B)+s(A\cap B)=s(A)+s(B).
\]

Also,

\[
s(A\cap B)-s(A)=-\sum_{x\in A\setminus B}z_x.
\]

Intersection beats A when the removed members have a negative total logit.
Selection requires the maximum score and the earliest slot attaining it. A new
intersection must strictly beat every existing slot. If the head assigns
positive scores to unwanted members, keeping those extras can still win.
The score rule therefore determines whether a newly available correct answer
turns into a better prediction. Floating-point evaluation uses the declared
canonical summation; the algebra describes the mathematical scoring function.

## An oracle diagnosis asks a different question

Eight sources have 2^8−1=255 nonempty subsets. For every subset, the diagnostic
forms its union and its intersection: 510 operation/subset entries, often with
duplicate memberships. It checks whether any equals the teacher set and records
the smallest number of sources needed, with a fixed tie order.

For example, three disjoint partial sets may require a three-way union even
though no original or pair is exact. Conversely, sources that all share the
same unwanted member may cover the truth yet yield no exact pure union or
intersection. These bounds concern one operation applied to a subset; they do
not enumerate arbitrary nested expressions such as (A union B) intersect C.

An oracle witness uses labels to identify a possible answer. A prediction uses
the learned scorer without labels. We report those separately: representability
is an opportunity for a policy, not proof that an implementable policy finds it.

## How this experiment can fail usefully

The fixed gate requires a strict pooled exact-answer improvement above 603/666,
no exact-count or F1 regression within any seed, and no pooled group exact
regression. A no-change result fails the improvement requirement. Added correct
candidates with worse selected answers would locate a selection problem;
no new exact availability would limit what this candidate expansion can achieve.

The baseline must first reconstruct exactly. A separate auditor uses integer
bitsets for set operations and exact rational sums rounded to float for score
checks. Agreement audits the saved evidence; it does not independently replay
the neural network. This remains development validation over 222 repeated
queries under three seeds, with the protected test untouched.

## Result: two more complete answers, but a failed gate

The [audited report](../experiments/2026-09-25-eight-source-set-algebra.json)
records the following results. Each seed evaluates the same 222 queries.

| Seed | Exact: old → intersections | Macro F1: old → intersections |
| --- | ---: | ---: |
| 1729 | 201 → 201 | 0.979926 → 0.979542 |
| 1730 | 205 → 206 | 0.990697 → 0.991794 |
| 1731 | 197 → 198 | 0.979856 → 0.982673 |
| Pooled | 603 → 605 /666 | 0.983493 → 0.984670 |

The two newly exact answers are Lycanroc TYPE under seed 1730 and Ekans TYPE
under seed 1731. No previously exact answer becomes inexact. Fourteen selections
change, all to intersections; none selects an empty set. COLOR and dual-TYPE
exact counts stay 309 and 155; single-TYPE increases from 139 to 141.

However, the declared requirement that every seed's F1 must not regress fails
for seed 1729. We retain the result and reject this policy variant. An audit pass
means the evidence is reconstructed faithfully; it does not mean the scientific
success criterion passed. The earlier eight-branch result remains separate.

## Precision can improve while F1 gets worse

Archaludon TYPE under seed 1729 is the clearest example. Its teacher set has
131 members. The old selected answer contains all 131 plus 65 wrong members.
The new intersection contains 68 correct members and no wrong members:

| Measure | Old selection | Intersection selection |
| --- | ---: | ---: |
| True positives | 131 | 68 |
| False positives | 65 | 0 |
| False negatives | 0 | 63 |
| Precision | 66.84% | 100% |
| Recall | 100% | 51.91% |
| F1 | 80.12% | 68.34% |

For a nonempty teacher set,

\[
P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad
F_1=\frac{2TP}{2TP+FP+FN}.
\]

For an empty prediction, define precision as zero; recall and F1 are also zero.

Removing every false positive looks attractive, but dropping almost half the
correct answer hurts more here. Exactness is zero for both answers, so the
exact-count metric alone hides this deterioration. Macro F1 averages query F1
values, which is why a change in already-inexact answers can fail the gate.
Across all seeds, precision rises 99.14%→99.52% while recall falls 98.18%→98.06%.

## What the exhaustive diagnosis rules out

| Candidate family | Observations with an exact candidate |
| --- | ---: |
| Existing 36 originals/pair unions | 605/666 |
| Fixed 64-slot pool with pair intersections | 607/666 |
| All 255 pure unions | 605/666 |
| All 255 pure intersections | 515/666 |
| Either exhaustive pure family | 610/666 |

Larger pure unions add no exact candidates beyond the old pool. With unchanged
scoring and old-first ties, they cannot create an exact-answer gain; they could
still introduce wrong winners. Pure intersections add five exact opportunities
overall: two require two sources and three require three sources. We did not
run a new three-way-intersection prediction policy after seeing this result.

Even the exhaustive combined pure families have at most 608 selected exact
answers with the unchanged scorer: 610 available minus the two existing
available-exact misses that appended duplicates cannot rescue. This tighter
bound concerns the pure families only. The broader 634 bound still applies to
arbitrary union/intersection constructions from the same sources. Neither is
a forecast of achievable quality, and neither reaches oracle parity.

The runner's 38 focused tests and auditor's 39 pass. Independent reconstruction
checks all 5,328 raw sources, 42,624 candidate slots and 339,660 operation/subset
entries. No model weights, defaults or protected-test measurements changed.
The next useful direction is to change what membership the model proposes or
how it scores that membership, with a new declared experiment. More pure set
operations over these fixed outputs cannot close the remaining gap.
