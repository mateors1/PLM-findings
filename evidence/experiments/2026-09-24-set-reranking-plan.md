# Rerank the frozen four paths with learned set membership

Declared 2026-09-24 after the completed first-choice branching experiment.
No head scoring or new evaluation under this rule has been run yet.

## Question and fixed inputs

Four first-choice paths contain 501/666 exact answers, but the sum and mean
token-score selectors return 462 and 465, versus the original greedy 466.
Test whether the existing symmetric relation head can recognize better complete
candidate sets. Its learned logits already guide the first-token decision; this
trial uses them to score entire generated sets. Do not retrain or widen search.

Freeze the complete first-choice campaign:
`runs/learning/first-choice-paths-v1/summary.json`, SHA256
`7b9db0b137d53ce2c350932aec61b1a8e94295f9affbae16aa55cc52f659fa8a`.
Its independent audit SHA256 is
`e45dcfc7c2709ffa07bf8a719038bcabce812b8daefb62191b7caf0f01e91e48`.
Recheck all three reports, original checkpoints, source/runtime, data/split,
plan/script/helper receipts, raw-token rank1 equality and frozen candidate sets.
All models are original continuation0/first1 controls, seeds1729/1730/1731.
Preserve all222 validation queries per seed and all four candidates per query.
The source archive remains
`101d661a16e3f17ecc924c1afed3a188cea25fe8e70bb1df2a19b523f363ecd1`.

## One declared learned score

For each five-token prompt, obtain the existing `symmetric_relation_logits`
with shape `[B, 1025]` using the frozen model in FP32 evaluation mode, batches
of eight in the same query order. No teacher target, generated prefix, hidden
attribute, expected length, or oracle relation is an input to this forward pass.
Verify all logits are finite and product columns have the pinned vocabulary order.

For a candidate processed set S, define:

`set_score(S) = sum(z_v for v in S)`

where z_v is the model's symmetric relation logit for product v. Use the fixed
postprocessed set: subject exclusion for SAME and deduplication are identity
rules already applied and verified in the frozen evidence. No relation-based
filtering, filling, target count, or token reordering is introduced.

Compute each score over distinct product IDs in canonical ascending ID order,
using `math.fsum` on saved FP32 logit values. This scoring-only ordering does
not reorder emitted tokens. Identical sets reached in different sequence orders
must receive identical scores and obey the same lowest-rank tie rule.

The score is equivalent, up to a query-specific constant, to an independent
Bernoulli set log-likelihood:

`sum(v in S, log(sigmoid(z_v))) + sum(v not in S, log(sigmoid(-z_v)))`

`= sum(all v, log(sigmoid(-z_v))) + sum(v in S, z_v)`.

This derivation does not establish calibrated membership probabilities: the
head was trained with balanced positive/negative losses, and independence is
a scoring assumption. The rule is a fixed learned ranking heuristic to test.
Omitting a positively scored product or adding a negatively scored product can
make a candidate score worse, but incorrect head signs can prefer a wrong set.

Select the highest set_score among the same valid EOS-terminated unique paths.
Tie-break by lowest first-choice rank. With no eligible path, retain rank1 and
record failure. Do not combine this score with token likelihood, tune a weight,
temperature, threshold or expected count, or pick between multiple heads after
seeing outcomes. Primary candidate is this single symmetric-head selector.

## Measurement and acceptance

Choose before reading truth-derived candidate metrics. Then measure complete
raw/processed outputs and paired changes against each query's original rank1.
Also report the frozen sum/mean selectors and oracle availability descriptively.
The oracle ceiling remains 501/666: selection cannot invent a missing candidate.
Show per-seed F1/exact, single/dual-TYPE/COLOR, gains/losses, selected-rank
histograms and oracle-available exact answers missed. Preserve raw logits, each
candidate score, selected rank and full selected IDs.

Acceptance requires all selected outputs valid/terminated/unique, no seed-level
processed F1 or exact regression versus original rank1, strict pooled exact
improvement beyond466/666, and strict pooled dual-TYPE exact improvement beyond
34/204. Keep unfavorable groups and comparisons visible. Passing establishes
eligibility for further integration and performance work; it is not oracle
parity, final-test evidence or serving promotion.

## Reproducibility and costs

Record the new script, helper, plan, source, checkpoint, runtime, vocabulary,
split and input hashes. Freeze the executed script and all imported custom
helpers. Refuse overwrites and preserve partial evidence if interrupted.
Unit tests should verify logit-sum/full-Bernoulli ranking equivalence, token
column mapping, duplicate/SAME handling, tie and no-eligible behavior, and
selection independence from labels. Independently reconstruct all scores and
metrics from saved logits and paths without reusing selection implementation.

GPU head scoring runs sequentially across the three checkpoints. Record its
incremental observed time and memory separately from the approximately8.1-minute
four-path generation campaign. Reusing cached candidates makes this experiment
cheap; it does not remove their generation cost from any future serving system.
No model/config/lock changes, new training, wider search or protected-test access.
