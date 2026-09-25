# 2. Learning and quality results

**Evidence class:** measured validation comparisons on a fixed National Dex snapshot. See the linked local learning notes and portable reports for exact run identities and gates.

## First make the training signal correct

The initial loss alignment rewarded reconstructing a token already visible at the same position. The corrected causal objective makes `ANSWER` predict the first target and each target predict the following token. Prompt and padding positions are excluded. Resume also had to restore the corpus batch offset, not just weights, optimizer and RNG state. A small overfit test now requires prompt-only generated completion as well as low loss. These repairs are in [Lesson 1](../evidence/learning/01-training-correctness.md). They establish mechanics on tested cases; they are not evidence of real-data generalization.

## Quality progression

| Intervention | Validation observation | Lesson |
| --- | --- | --- |
| Dense token-only control, three seeds | 23.13% mean raw F1; 15.62% mean processed exact accuracy | [6](../evidence/learning/06-seed-replication.md) |
| Prompt-set auxiliary supervision, same three seeds | 60.62% mean raw F1; 45.80% mean processed exact accuracy | [5](../evidence/learning/05-prompt-supervision.md), [6](../evidence/learning/06-seed-replication.md) |
| Symmetric relation objective, paired three-seed study | 79.87% mean raw F1; 61.11% mean processed exact accuracy; occasional unfinished responses prevent unconditional promotion | [11](../evidence/learning/11-symmetric-relations.md), [12](../evidence/learning/12-symmetry-replication.md) |
| Guided first target plus uniqueness, frozen weights | Improves the recorded candidate gate; application replay verifies the tested inference policy | [16](../evidence/learning/16-guidance-and-uniqueness.md), [17](../evidence/learning/17-guided-integration.md) |
| Four generated starts, selected using learned membership scores | 501/666 exact sets versus 466/666 greedy; no training change | [22](../evidence/learning/22-first-choice-paths.md), [23](../evidence/learning/23-ranking-candidate-sets.md) |
| Four starts plus six pair unions, same learned set scorer | 569/666 exact sets, 96.69% macro F1; offline experiment passes declared gate | [25](../evidence/learning/25-composing-candidate-sets.md) |

These rows are a development trajectory, not one randomized head-to-head trial. Controls, scoring policies and application paths change across rows. Compare a row to its own declared paired control and report, especially when interpreting gains. The [quality campaign](../evidence/experiments/2026-09-24-quality-campaign.json), [seed replication](../evidence/experiments/2026-09-24-seed-replication.json), [symmetric replication](../evidence/experiments/2026-09-24-symmetric-replication.json) and [pair unions](../evidence/experiments/2026-09-24-pair-unions.json) preserve those comparisons.

## What failed, and what that taught us

The first real-data dense/MoE campaign reached low training loss but **zero exact raw sets** at validation and roughly 28% raw F1 for its best dense run. Equal optimizer steps were not equal compute. This exposed the gap between teacher-forced next-token fit and complete answers to held-out queries ([Lesson 4](../evidence/learning/04-first-quality-campaign.md)).

Weighting only the first target at an exploratory setting did not solve that gap; neither did choosing an earlier saved checkpoint. A later first-target-weight-eight trial lowered final-batch first-token loss yet changed exact sets from **466/666 to 464/666** and mean F1 from **90.12% to 89.42%**, failing its gate ([Lessons 10 and 21](../evidence/learning/21-weighting-the-first-choice.md)). Additional continuation-set supervision improved some token metrics but worsened exact-answer acceptance and dual-TYPE coverage ([Lesson 19](../evidence/learning/19-continuation-set-training.md)). These failures show why a local surrogate metric cannot substitute for complete-set evaluation.

The symmetric head is useful as an auxiliary representation and as a selector, but an isolated score or loss does not guarantee a correct output set. Four first-choice paths made **501/666** exact answers *available*. Selecting by sequence log-probability yielded only **462 or 465**, below greedy **466**; selecting by learned membership scores found all **501** in the fixed study ([Lessons 22–23](../evidence/learning/23-ranking-candidate-sets.md)). This distinction between generating a good candidate and recognizing it motivated pair-union composition.

**Lesson:** train and select for the behavior the application needs, then verify the whole generated and processed answer. Keep failed interventions in the record rather than treating improved proxy metrics as acceptance.
