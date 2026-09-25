# Membership rank separability diagnostic — 2026-09-25

Status: declared before measurement. CPU-only analysis of frozen validation
logits; no training, neural forward, threshold fitting or application change.

## Question and motivation

The fixed zero-threshold membership rule produced 339 exact sets, compared with
569 for the pair selector, on 666 observations (222 validation queries repeated
across three training seeds). Is the membership head unable to rank all true
members above all nonmembers, or does its zero threshold cut an otherwise
separable ranking at the wrong position?

This diagnosis uses expected labels and answer cardinalities. Exact-answer
attainability is an oracle-assisted bound within the stated rule family, never
deployable quality or evidence that a learned predictor will reach that bound.
Top-K precision/recall/F1 are privileged-input diagnostics, not upper bounds on
those metrics over all threshold rules. We do not
select a threshold, fit a calibrator, tune hyperparameters or touch protected test.

## Frozen inputs and identities

- Pair summary SHA256:
  `4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992`.
- Pair independent audit SHA256:
  `13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057`.
- Direct-membership summary SHA256:
  `2cbe82aad296b6ff4a920a951de38a5999a23e0eb8507d40444254e5d404ceab`.
- Direct-membership independent audit SHA256:
  `8111692647153787e095da94c64ec497ebc41537b6f6c3ea8ad1d5b046b8df42`.
- Archived direct-membership helper `summary.script.py` SHA256:
  `4f9baef52a7f365bfafdb46041ede98cf4d04ff27c130e27c053bb4726f4e4c6`.
- Its archived plan SHA256:
  `4004639897b88d54090ce94bfecf26adf5b2a75686be4b8cdf97da2155d19da0`.

Reuse that authenticated helper's input authentication and row preparation,
including its recursive historical archive bindings. Load it only after its
bytes match the pin. Authenticate the direct summary, audit, and per-seed reports
and independently reproduce their zero-threshold sets and exact counts. Preserve
all checkpoint, corpus, vocabulary, split, config, source and batch-shape
identities inherited from the saved inputs. The diagnostic uses the original
batch-eight FP32 logits, not the concurrently generated serial references.

Seeds are 1729, 1730 and 1731; the same ordered 222 validation queries must occur
in each. Product columns are IDs 1024 through 2048 (1025 entities); exclude the
query subject before all calculations. Each row has 1024 eligible products.
These are three replicas of the same queries, not 666 independent examples.

## Fixed calculations

For allowed products A, expected set T, negatives N=A\\T and saved logits z:

1. Reconstruct the unmodified direct rule D={i in A: z_i>0} and pair baseline.
   Exact counts must remain 339 and 569 respectively, with direct gains 3 and
   losses 233 relative to pair. Any discrepancy fails the diagnostic.
2. For nonempty T and N compute a=min(z_i for i in T),
   b=max(z_j for j in N), and gap a-b. An exact strict-threshold rule z>t
   exists iff b<a. Its feasible interval is [b,a). Gap zero is a boundary tie,
   not strict separation. Retain extremal product IDs as witnesses.
3. Empty T is separable via t>=max(z_i for i in A); full T via
   t<min(z_i for i in A). Represent absent
   extrema/bounds as null plus explicit bound semantics, never JSON infinity.
   The empty allowed-universe case is vacuously exact. Test these edge cases
   even if the real corpus does not contain them.
4. Sort allowed products by descending score then ascending product token ID.
   Select the first K=|T| products using the oracle answer size. Report exact,
   TP/FP/FN, precision/recall/F1 and boundary tie details. This is a fixed
   diagnostic, not a cardinality estimate. Strict separation implies top-K
   exactness; a zero-gap boundary may instead depend on the ID tie rule.
5. Record per-row pair/direct exactness, separable-with-wrong-zero-threshold,
   boundary tie versus strict rank overlap, oracle-top-K exactness and its
   gains/losses relative to pair. Cross-tabulate these for all rows and the
   pair's 97 failures, including whether the four-source union lacks truth.

Report totals, each seed, and each existing group (COLOR, TYPE_single,
TYPE_dual). Preserve per-row query identity, expected IDs, selected diagnostic
IDs, extrema, gap, tie witnesses and relevant baseline evidence. Do not invent
new buckets or search additional rules after observing results in this version.

## Evidence and acceptance

The script must remain stdlib-only and refuse overwrite of an existing run.
Snapshot executing script, this plan and helper; bind input, output and snapshot
SHA256 values; rehash inputs and executing files before final completion. No
runtime source/config/model files may change during the live integration job.
Save under `runs/learning/membership-separability-v1/`; keep a compact portable
report under docs only after independent audit.

Before measurement, run focused tests for strict versus zero-gap separation,
misplaced zero threshold, rank inversion, deterministic tie handling, subject
exclusion, empty/full truth, malformed IDs, nonfinite logits, alignment and
immutability. Do not call shared diagnostic arithmetic from the independent
auditor: reconstruct from authenticated raw saved rows and compare all row
and aggregate outputs. An audit failure remains a failed artifact; do not edit
the run in place to make it pass.

Acceptance means that this diagnostic is reproducible and internally correct.
There is no required favorable count and no model/system promotion. Follow-up
training or inference experiments require a separately declared hypothesis.

Pre-measurement editorial review clarified that empty/full bounds concern
scores, not product IDs, and that top-K partial-credit metrics are not ceilings.
No real separability measurements were made before these clarifications.
