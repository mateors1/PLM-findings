# 3. Complete-set prediction and remaining errors

**Current reference:** three frozen checkpoints evaluated on the same 222 validation queries, for 666 query-seed observations. The strongest offline set-composition experiment has 569 exact sets and 97 inexact observations. Its application replay has a separate incomplete gate; see [Paper 4](04-serving-and-reproducibility.md).

## The structural difficulty

`SAME × TYPE` for a dual-type subject requires the union of two attribute groups, excluding the subject. That is harder than recognizing a single shared group. In the pre-composition guided study, single-TYPE answers were exact in **124/153** observations, dual-TYPE in only **29/204**, and COLOR in **301/309** ([Lesson 18](../evidence/learning/18-union-coverage.md)). A model could recognize many individual members while omitting a whole branch of the union during ordered generation. Giving an oracle-correct first target improved one checkpoint from **105/222 to 174/222** exact answers, but that was diagnostic privileged input, not a deployable policy ([Lesson 9](../evidence/learning/09-sequence-diagnosis.md)).

## Candidate construction and selection

Four first-target branches produce four complete source paths. An auxiliary symmetric head gives one learned membership logit per product. The set scorer sums those logits over a candidate's members, with a declared stable tie rule. It can choose a coherent output set without using oracle membership at inference. Four-path reranking passed its own offline and application replay gates at **501/666 exact sets** ([Lessons 23–24](../evidence/learning/24-integrating-set-selection.md)).

Pair composition keeps those four paths and adds their six pairwise unions, for ten candidate slots. The union is a derived *set* with two source paths; it is not falsely described as one autoregressively generated sequence or given invented EOS evidence. The frozen offline selector reached **569/666 exact sets** from **501/666**, with **68 gains and zero exact losses** under the declared comparison ([Lesson 25](../evidence/learning/25-composing-candidate-sets.md), [portable report](../evidence/experiments/2026-09-24-pair-unions.json)). The gain is on validation; it does not establish oracle parity or protected-test performance.

## Why the remaining 97 errors persist

An independently recomputed saved-evidence diagnosis partitions the 97 failures:

| Failure source | Observations | Implication |
| --- | ---: | --- |
| At least one true member absent from all four source paths | 53 | Pair unions cannot recover the missing member. |
| Every true member appears in some source, but no ten-slot candidate is exact | 43 | Choosing another whole slot cannot fix an extra or missing member. |
| An exact candidate exists but is not selected | 1 | Better ranking alone can add at most one exact answer in this fixed pool. |

Thus 96/97 failures lack an exact ten-slot candidate ([Lesson 27](../evidence/learning/27-remaining-errors-and-score-signs.md), [diagnostic](../evidence/experiments/2026-09-24-pair-error-diagnosis.json)). Coverage and selection are different failure modes. The diagnosis localizes them; it does not prove why the decoder left products out.

## Why a simple score cleanup fails

The additive scorer assigns score `sum(z_i for i in S)`. Over unrestricted subsets, its fixed mathematical optimum includes every non-subject product with a strictly positive membership logit. That is a property of the learned *score*, not a proof that the resulting set is correct. A direct-score ablation tested exactly that rule on saved scores. It recovered **3,810** missing true-member occurrences and removed **2,003** wrong-member occurrences, yet introduced **1,556** new wrong members and lost **206** previously selected true members. Exact answers fell from **569/666 to 339/666**, including **233** formerly exact answers spoiled; macro F1 rose from **96.69% to 98.96%**. Dual-TYPE exact fell from **125/204 to 13/204**. Its declared exact-answer gate failed ([Lesson 28](../evidence/learning/28-direct-membership-ablation.md), [report](../evidence/experiments/2026-09-24-direct-membership.json)).

The less aggressive idea of deleting every negative-scored member also has a measured logical obstacle: **62** currently exact sets contain negative-scored true members. Given the observed 17 failures with only extra members, that deletion-only rule could reach **at most 524 exact answers**, even if all 17 became correct. This is a bound, not a measured alternative policy ([Lesson 27](../evidence/learning/27-remaining-errors-and-score-signs.md)).

**Lesson:** a candidate restriction can preserve coherent correct sets even when its scorer is imperfect. Future work should expand coverage while controlling false additions and protecting existing exact answers. A larger search space by itself is not guaranteed to improve exact-set accuracy.
