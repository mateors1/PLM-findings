# Saved bilinear error geometry

Declared after the completed parent-seed replication, before running this new
diagnostic. Campaign `bilinear-error-geometry-v1` is descriptive posthoc analysis
of already observed validation results. It is not a new quality screen.

## Question and evidence scope

Distinguish zero-threshold mistakes from failures to rank every true member
above every false member. Compare which validation queries fail across trained
parents. This will inform a future, separately declared intervention; it does
not change the completed model results or select a threshold.

Use only saved child reports from historical selection seed 1729 and fresh
replication seeds 1730/1731. Each contains the same 222 validation queries,
ordered product columns 1024..2048 and saved FP32 relation scores. Authenticate
the accepted replication aggregate, independent audit and owner decision:

- Aggregate: `338fb6b7ea08128952ce9abc02b3506639e712a9cff6ce63468ed2e7980f79a0`.
- Audit: `ddbecaceabdacc0808f9b4f122e0309aa3e233fe76b0d75694e31833a40083a8`.
- Decision: `62a8dcbb7cbe5d038984f9d57aecacec4b0ba612ba55c2a9f6fe335977304978`.

Resolve each child report through that chain, including the historical accepted
1729 budget summary/audit/decision. Verify saved per-seed report hashes and the
same query order, subject, dimension, group, prompt and target IDs. Reconstruct
the original z>0, nonself, ascending-ID predictions and require exact equality
with the saved selected IDs, errors, exactness and strict-separation flags.
Read no checkpoint, graph database, training examples or protected-test data.
No Torch, model forward, optimizer, GPU work or new checkpoint is needed.

## Fixed reductions

For each query exclude the subject. Let T be its true product IDs, and N the
remaining false product IDs. Define:

\[
a=\min_{i\in T}z_i,\quad b=\max_{i\in N}z_i,\quad g=a-b.
\]

The fixed rule is exact iff a>0 and b<=0. Strict separation means a>b.
For a hypothetical rule z>t, exact membership is possible iff b<a, with
t in [b,a). This interval is a label-informed diagnostic, not a deployable
prediction rule. Do not choose a value from it or generate alternative answers.

Classify each query once, in this order:

1. Exact at zero.
2. Non-exact but strictly separated, missing members only (a<=0).
3. Non-exact but strictly separated, extra members only (b>0).
4. Non-exact with overlapping or tied true/false score ranges (a<=b).

Report all 222 per-query reductions per seed, extrema and deterministic
lowest-ID witnesses for tied extrema, plus incorrect member IDs/scores only.
Reconcile every FP/FN and exact count with accepted results. Publish per-seed
and group category counts, fresh-two totals and all-three contextual totals.
Record query failure seed lists and overlap counts over aligned query identities;
do not treat repeated questions as independent examples.

For each seed/group also report A=min_query(a), B=max_query(b), and whether
B<A. This asks whether any one constant threshold could separate all saved
labels within that group. It is an interval-existence calculation, not threshold
fitting, threshold selection, a performance estimate or a new quality gate.

Use ordinary finite saved numeric values for reporting a,b,g, and direct
comparisons of a and b for classification. Reject nonfinite or non-FP32 inputs,
bad column order, inconsistent labels/self exclusion, query drift or incorrect
saved predictions. No tolerance, score rounding or near-zero bucket is allowed.

## Verification and recording

Freeze a stdlib-only runner and synthetic CPU tests before the reductions.
Cover exact zero on false/true sides, separation versus overlap/ties, FP/FN
reconstruction, extrema ties, shared-query alignment, nonfinite inputs and
immutable outputs. A separate stdlib auditor must independently reconstruct
the reductions from authenticated saved reports without importing the primary
diagnostic's classification or aggregation functions. It may reuse hash/file
utilities; its scientific arithmetic must be independent.

Write immutable source/plan/test/input snapshots, results, observed execution
receipt, independent audit and separate evidence-acceptance decision. Preserve
any failed attempts. Update the logs, registry, Lesson 45 and PLM-findings, then
send a completed-evidence handoff to the existing Luna Max publisher.

The registered final-checkpoint count remains 33. No inference policy, evaluator
quality gate, standard serving path or default changes. Findings may motivate a
new training-only intervention, but this plan authorizes no tuning, additional
training budget, prediction routing, oracle-count repair or protected evaluation.
