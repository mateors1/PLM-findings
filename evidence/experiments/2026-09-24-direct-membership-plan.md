# Direct membership-head ablation at a fixed zero threshold

Declared on 2026-09-24 before executing this alternative selection rule.

## Question

The accepted pair experiment selects 569 exact sets across 666 validation
query-seed observations. Saved-error diagnosis finds 96 errors without an exact
candidate, and 3,810 of 3,850 missing true-member occurrences have positive head
logits. However, 62 already-exact selected sets contain negative-scored true
members. Does using the head directly recover enough missing members to offset
the mistakes that autoregressive candidates currently prevent?

It necessarily loses at least the 62 currently exact observations with negative
true-member logits: adding other products cannot restore an excluded true member.
Overall exact improvement remains logically possible. This known tradeoff makes
the experiment an ablation, not a monotonic repair. It tests the value of the
candidate restriction without isolating why generation succeeds or fails.

This is a fixed ablation of the candidate restriction, not a new transformer
architecture, protocol decoder, runtime option or accepted replacement.

## Fixed inputs and identity

Use only the accepted pair experiment's saved FP32 symmetric-head logits, product
column mapping, query identities and expected validation sets: all 222 queries
for each of the three existing checkpoints, seeds 1729, 1730 and 1731.
Do not execute a model forward, train, read protected test rows, consult graph
relations during selection, try extra thresholds, or change the live application
campaign. Authenticated graph-derived group labels are analysis metadata only.

Pair summary SHA256:
`4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992`.
Pair independent audit SHA256:
`13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057`.
Error diagnosis summary SHA256:
`f5c826295bfc30466dc435c5a995134fe198686f69f2890cac2e3f2435053523`.
Error independent audit SHA256:
`1639c4b6fd2a45600436fbbd4d9a566e43be9016ed4e618750604c3315b7f895`.

Authenticate the pair summary/audit and their bound scripts, plan, archived
source, helper, complete input manifest, and all three per-seed reports before
measurement. Authenticate the diagnosis summary/audit as the design motivation.
Historical source/config identities belong to their archives: do not require
the new inference checkout to masquerade as historical source. Record each input
hash, the new script/plan hashes, Python version and immutable output hashes.
Archive the new script and plan alongside the output; refuse existing outputs.
Verify inputs and executing script/plan again before completing the summary.

## Exactly one truth-independent prediction rule

For each query q, use all 1,025 saved product logits z(q,i):

    predicted(q) = sorted(i for product i if i != subject(q) and z(q,i) > 0)

Zero-scored products are excluded, with no top-k, minimum-size repair, maximum
size, fallback, dimension-specific threshold, expected-size input, graph lookup,
candidate-union restriction or mixture with the baseline. Reject nonfinite or
non-FP32 saved values and misaligned product columns. The selector takes only
logits, product IDs and subject ID. Expected targets and baseline sets enter
measurement afterward. An empty predicted set stays empty and is measured.

This equals maximizing the additive score over unrestricted subject-excluded
product subsets, with zero-score ties resolved toward exclusion. It emits a set
representation, not an autoregressive sequence: no synthetic EOS, termination,
source-path eligibility, sequence probability or protocol-validity claim.
All frozen pair baseline sources must retain their authenticated eligibility;
the direct head has no generation/source eligibility gate.

## Measurements and gate

First reconstruct the accepted pair baseline per query and verify all stored
precision, recall, F1, exactness and pooled/per-seed totals. Validate unique,
aligned query identities and identical query partitions across seeds. Retain
every direct predicted ID set, expected set, baseline selection, metric and
paired exact gain/loss. Report macro precision/recall/F1, exact count and mean
set size per seed, pooled and by COLOR, TYPE_single and TYPE_dual. Count empty
predictions and positive/zero/negative true versus nonmember logits, excluding
the subject. Report source-union-missing true members recovered by direct
prediction, and damage to previously exact answers, as retrospective diagnostics.
Partition membership changes into recovered baseline false negatives, newly
introduced false positives, removed false positives and newly lost true members.
Separate previously exact observations from previously inexact observations.

Acceptance for further research requires no per-seed overall macro-F1 or exact
count regression, strict pooled exact improvement, and strict pooled dual-TYPE
exact improvement over the pair baseline. These criteria are fixed before the
measurement. Report each criterion and all paired losses even if totals improve.
Do not select a threshold or hybrid policy from these outcomes. Failure is a
valid result; it does not authorize silently changing this experiment.

Count precision as zero when prediction is empty and truth is nonempty; empty
truth has recall one, and an empty/empty pair has precision/recall/F1 one. All
queries remain in the macro denominator. Report threshold/set-construction time
separately from authentication and reporting; it excludes the already-paid head
forward and establishes no end-to-end speed, throughput, energy or serving claim.

## Verification and interpretation

Use CPU-only focused tests for threshold boundaries, subject exclusion, empty
outputs, invalid values/column mapping, macro denominators and paired gates.
An independent stdlib audit must derive predictions, metrics and gates from
the original authenticated evidence without importing this evaluator.

The three seeds share the same validation query partition. They are repeated
training seeds, not three independent held-out datasets. Prior validation-guided
development makes this exploratory evidence, not unbiased final-test quality.
The deterministic oracle remains perfect. Even a passing ablation requires a
separate implementation and fresh inference verification before application use.
The active composition GPU/HTTP replay remains frozen and continues unchanged.
