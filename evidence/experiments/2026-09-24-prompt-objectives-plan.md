# Prompt-supervision campaign: pre-run plan

The first quality campaign's dense/MoE runs both had zero exact sets despite
very low continuation loss. The next change targets supervision at ANSWER.
This plan is recorded before launching either candidate; all comparison is on
validation, with the protected test untouched.

## Candidates

1. `first_target_lab`: dense model, first supervised token weight 32, remaining
   supervised tokens weight 1; normalize by the sum of weights. No new parameters.
2. `prompt_set_lab`: dense model, ordinary token CE plus weight-1 balanced set
   loss. A bias-free 256x256 projection transforms the final ANSWER hidden state;
   dot products with existing entity embedding rows produce 1,025 membership
   logits. Add 65,536 parameters, initialized after the common stack so the seed
   preserves identical common initialization. Entity embeddings are shared, so
   products receive this relation signal as targets as well as as subjects.

For each query, set loss is half the mean positive binary loss plus half the
mean negative binary loss; then average queries. Positives come only from that
training record's product labels. EOS, controls, prompt and padding are excluded.
The head sees only the causal ANSWER representation, never answer history.
The independent projection prevents directly forcing the next-token softmax
to distribute mass equally across an unordered set.

## Fixed controls and evidence

- Retain the previous dense run/results as the reference. Default inference
  behavior stays unchanged; regressions check reference initialization/logits,
  exact loss weighting, label construction, future-token isolation and resume.
- Same snapshot, split seed 1729, initialization seed 1729, dense stack, optimizer,
  2,000 updates, batch 32 and sequential order as `national_dex_v1`.
- Evaluate final checkpoints with the same FP32 greedy syntax-constrained native
  generator and 507-token bound. Do not use auxiliary scores to alter generation,
  inject oracle labels or filter wrong products.
- Primary: full-answer exact sets and macro F1 on all 222 validation queries.
  Also termination, duplicates, self-return, first-target CE, ordinary token CE
  and set loss. Training total loss combines different objectives and is not a
  comparable quality metric across candidates.
- One seed per candidate is exploratory, not an estimate of robustness. Equal
  updates do not establish equal compute. Preserve all checkpoints/reports.
- A better set classifier without better generated answers is an incomplete
  result. Do not claim the generation objective succeeded on a surrogate alone.

Hypotheses: first-target reweighting may improve prompt use but can also overfit;
set supervision may teach shared relational geometry because every target yields
direct prompt-conditioned feedback. Neither outcome is assumed. If unsuccessful,
inspect first-target behavior and continue from evidence rather than adding capacity.
