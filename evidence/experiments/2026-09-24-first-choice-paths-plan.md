# First-choice branching: availability versus learned selection

Declared 2026-09-24 after the completed weight8 trial, before new GPU evaluation.

## Question

Can a frozen model recover more complete answers when we retain four plausible
first products, and can its own sequence scores select those answers? Weight8
failed to improve held-out ordered starts; changing another loss coefficient
does not directly test whether useful alternative continuations already exist.

This is first-choice branching followed by greedy continuation, not beam search
at every step. It cannot recover a path requiring a different second decision
after the same first product. It is a diagnostic of the search/selection boundary.

## Frozen references and workload

Use only the three original continuation0/first1 control checkpoints for seeds
1729, 1730 and 1731 from runs/learning/continuation-set-v1. Its summary SHA256 is
`558d18391131333461efee09c3045848a8ffbca7a6a5e4b3d436ac7d49fbf819`.
The reference has466/666 processed exact answers and34/204 dual-TYPE exact.
Source archive remains
`101d661a16e3f17ecc924c1afed3a188cea25fe8e70bb1df2a19b523f363ecd1`.
Validate actual checkpoints, saved settings, source, runtime, corpus and split
identities before reuse. No training, source/config/lock changes or final test.

Evaluate all222 validation queries for each seed, including every COLOR query.
Use FP32 inference, alpha16 learned first-target guidance, protocol constraints,
unique emitted products, context512 and507 new-token bound. The bound includes
the selected first product and any EOS. The prompt subject remains eligible;
only fixed downstream SAME/IGNORE processing excludes it.

## Four paths and their scores

For each prompt, rank legal product logits after learned guidance. Select four
distinct first products by descending score, breaking exact ties by ascending
token ID. EOS is not allowed at the first step. A branch's first product counts
as emitted for subsequent uniqueness. Greedily continue each branch, breaking
logit ties as the existing decoder does. No attributes, expected targets or
oracle membership enter candidate generation or learned selection.

Keep original query order in groups of eight. For each group, run four separate
rank batches with fresh caches, one batch for each first-product rank. Thus
rank1 preserves the original eight-row numerical path; ranks2–4 also use that
batch size. Repeated prefill is an intentional experimental cost. Each branch
applies guidance only at its first decision. Finished rows contribute no later
tokens or scores; dummy cache inputs never become evidence.

Record the chosen token's log probability after class and uniqueness masking
at every step, including the guided first decision and EOS. These are
probabilities under the constrained guided decoding policy, not unmodified
language-model likelihoods. For path y with L emitted tokens including EOS:

The first log probability is normalized over all legal products after guidance,
not just the four selected candidates. Later probabilities are normalized over
all currently legal unconsumed products plus EOS.

`sum_score = sum(log p(y_t | prompt, y_<t))`

`mean_score = sum_score / L`.

Primary learned selector: highest sum_score among valid, EOS-terminated paths.
Fixed secondary diagnostic: highest mean_score among the same eligible paths.
Tie-break by lowest first-product rank. Neither uses truth or expected length.
If no path is eligible, retain rank1 as an explicit failed fallback; do not
manufacture EOS, silently drop the query, or count a truncated path as valid.
Length normalization may change preferences; both selectors are declared now,
and the better one will not retroactively replace the primary selector.

After candidates and learned choices are fixed, compute the oracle ceiling:
maximum processed F1 among eligible candidates and whether any is exact. This
uses validation answers and is diagnostic only. With no eligible candidate,
count ceiling F1/exact as zero and report the case. Rank1 is always included,
so exact availability cannot fall when its replay is valid. That monotonicity
does not apply to a learned selector, which may abandon a correct rank1 answer.

## Evidence and acceptance

Require full raw token equality between rank1 and each archived reference on
all666 query-seed observations. Abort quality interpretation on any mismatch;
retain discrepancy evidence, never overwrite reference output or relax parity.
Record all2,664 paths, token-level scores, lengths, termination/validity, raw and
processed metrics, raw/processed first choices, candidate first rank, selector
choices, paired gains/losses, and single/dual-TYPE/COLOR subgroup metrics.
Record teacher-first-in-top4 and oracle ceiling only as labeled diagnostics.

Compare each fixed learned selector against its matching rank1 reference.
For the primary selector, require valid terminated unique output on every query,
no seed-level F1 or exact regression, strict pooled exact improvement beyond466,
and strict dual-TYPE exact improvement beyond34. Report the secondary selector
under the same conditions without substituting it after results. Passing is
eligibility for a later declared integration/performance comparison, not serving
promotion or oracle parity. Do not mix oracle-selected answers into these gates.

Freeze the executed script, imported custom helper scripts, source archive,
plan and all input hashes. Preserve partial outputs on failure and refuse
overwrites. Independent CPU checking must reconstruct token-score sums, selector
rules, full-output metrics, paired changes and receipt identities. That audit
does not independently recompute the neural network.

Record observed evaluation time and GPU allocation, but do not claim matched
latency, throughput or energy from a four-path offline diagnostic. The candidates
share queries and model weights; they are not independent statistical samples.
If oracle availability rises but learned selection fails, investigate the
selection signal. If availability stays low, this particular first-choice
branching is insufficient; do not infer that wider or deeper search must work.
