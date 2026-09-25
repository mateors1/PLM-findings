# Continuation and TYPE coverage diagnosis

Declared before execution, 2026-09-24. The integrated alpha-16 guidance plus
uniqueness policy has 454/666 exact validation observations; TYPE contributes
204 of the 212 failures. Investigate membership knowledge versus ordered
continuation without changing weights, protocol, corpus, serving or defaults.

## CPU structural analysis

Verify the three integrated alpha-16 reports and data identities. Recompute
processed metrics and group TYPE outcomes by one versus two subject attributes,
shared-count tiers, branch omissions and training signature support. COLOR is
the comparison group. Graph attributes are labels for analysis only; they do
not enter model inference. Only train/validation records are analyzed.

## Frozen-model teacher-prefix probes

Use all 222 validation queries at each of the same three checkpoints. Supply
known ordered teacher targets up to these positions: first token, second token,
third token, halfway through the answer, EOS, and the first observed divergence
between processed generation and the teacher sequence. Merge equal positions.
The prefix is deliberately oracle-assisted diagnostic input, not a model answer.

At each position, compare the original prompt with (a) a deterministic half-
catalog rotation of the subject and (b) the previous supplied teacher target as
subject, when available. Hold the dimension and full target prefix fixed. Fresh
full-prefix cached prefill prevents stale-cache contamination. Apply the current
legal-token/uniqueness masks; apply alpha-16 guidance only at position zero.
Do not mask the subject or add missing products to a generated answer.

Record next-token choice, original-teacher rank/probability/logit margin and
total-variation distance between next-token distributions. The counterfactual
subject can contradict the fixed prefix. Its match to the original teacher is
a sensitivity measurement, not accuracy against that counterfactual's oracle.
Record prompt-only symmetric membership at the fixed zero-logit threshold,
including how many omitted targets it recognizes. Classifier sets remain
separate from emitted sequences and EOS behavior.

Check exact report/checkpoint/training/data/split/source/runtime identities;
archive the executed diagnostic script and source; preserve complete per-query
probes. First-position agreement with archived generation is a recorded sanity
check, not a claim of full free-generation replay. Refuse existing output paths.
Core/model code remains frozen and GPU work runs sequentially.

## Interpretation

Weak sensitivity after a known prefix is consistent with dependence on the
prefix, but does not prove an internal forgetting mechanism. Strong membership
on omitted targets points to a gap between recognizing and serializing the set.
Structural differences may suggest training or architecture experiments; they
are not causal proof. Choose a next intervention from the observed evidence and
declare its comparison before running it. No protected test, threshold tuning,
training run, serving promotion or performance claim belongs to this diagnosis.
