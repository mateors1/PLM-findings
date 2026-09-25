# First National Dex quality campaign

Plan recorded before training/evaluation, 2026-09-24. This is exploratory
validation work; no protected test evaluation or oracle-quality-win claim.

- Dataset: `pokemon_v1_f1541479_20260924`, receipt verified; SAME/TYPE/COLOR.
- Partition: query split seed 1729; 1,637 training, 222 validation, 191 test.
  Repetition seeds must not change this partition.
- Dense: default eight-layer, width-256, tied-embedding model; corrected causal
  objective/RoPE; 2,000 AdamW updates, batch 32, BF16 autocast, learning rate
  0.0003, cosine schedule, warmup 200. Sequential batches; zero loader workers.
- MoE: same recipe with four experts, top-1, auxiliary weight 0.01, no shared
  experts. One seed (1729) per PLM variant initially. No variance claim from these.
- Dense/MoE comparison holds updates and data order fixed, not parameter count,
  FLOPs or elapsed time. Run timings include validation/checkpoint overhead and
  are not isolated serving benchmarks. Later comparisons need repeated seeds
  and compute/capacity controls.
- Baselines: oracle and training-only popularity; ComplEx dimension 32,
  100 epochs, batch 512, learning rate 0.01, one negative, no L2 penalty.
  Repetition seeds 1729/1730/1731. Larger ComplEx batches are an explicit recipe
  choice to keep the initial CPU campaign practical; do not equate its epochs
  or compute budget with transformer updates.
- Primary generated-answer measures: exact-set accuracy and macro F1. Also
  precision/recall, exact sequence, syntax/EOS, duplication and self-return.
  Full autoregressive decoding starts from five prompt tokens, uses the syntax
  mask only and at most 507 new tokens, with FP32 evaluation weights/operations.
- Ranking uses the same unfiltered multi-positive MRR/Hits/MAP implementation
  for all methods. First-token PLM ranking is not complete-answer quality.
- Evaluate the final 2,000-step checkpoint for the first comparison. Intermediate
  validation CE is diagnostic. Any later choice of checkpoint or changed budget
  must be documented as a separate validation-driven decision.
- Save resolved config, source archive, checkpoint identity, validation response
  lists and raw metrics. Failed/poor runs remain evidence; do not overwrite them.

Questions: Does lower teacher-forced loss transfer to complete answer generation?
Does the router use its experts? How do accuracy and runtime change with MoE?
Failures should guide the next experiment, not be hidden by aggregate ranking.
