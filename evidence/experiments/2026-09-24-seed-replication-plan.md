# Fixed-split seed replication: pre-run plan

Recorded before launching seeds 1730 and 1731 on 2026-09-24.

Question: does prompt-set supervision improve complete-answer generation across
initializations, or was seed 1729 unusually favorable?

- Compare dense token-only and dense prompt-set (coefficient 1) models at seeds
  1729, 1730 and 1731. Retain the already completed seed-1729 runs unchanged.
- Add two seeds for **both** objectives so the comparison is paired by seed.
  Common model initialization is identical within each pair; the optional head
  initializes after the common stack. Training is sequential, without shuffling.
- Freeze the canonical National Dex corpus and split seed 1729: 1,637 training,
  222 validation and 191 protected-test queries. Do not evaluate final test.
- Keep 2,000 optimizer updates, batch 32, optimizer/scheduler and decoder settings
  unchanged. Use final checkpoints, without choosing earlier checkpoints based
  on these results. Do not tune the loss coefficient during this replication.
- Run jobs sequentially on the same GPU. Training remains BF16 with
  `deterministic=false`; seed variation does not imply bitwise reproducibility.
- Primary endpoints: raw generated macro F1 and exact-set accuracy on validation.
  Also retain precision/recall, EOS, syntax, duplicates and self-return counts.
- Secondary endpoints: the same predeclared SAME self-exclusion and stable
  deduplication policy for every run, with no IGNORE keys and no return limit.
  Preserve raw responses, and never repair invalid or unfinished output.
- Report every seed, mean, sample standard deviation and paired differences.
  Three seeds are descriptive replication, not a precise population estimate or
  evidence of robustness to different datasets/query splits.
- Validate corpus/split/config/checkpoint provenance before aggregation. Source
  archives differ because the policy component was added after the first runs;
  inspect and report relevant model/training source differences rather than
  claiming identical full source trees.

Training uses the existing `national_dex_v1` (dense) and `prompt_set_lab` recipes,
with `seed` and `run_name` overridden. New run names end in `s1730_v1` and
`s1731_v1`; reports go under `runs/learning/seed-replication-v1/`.

Regardless of outcome, publish the failures and a learning note explaining seed
variance, paired comparisons and the boundary between validation and final test.
