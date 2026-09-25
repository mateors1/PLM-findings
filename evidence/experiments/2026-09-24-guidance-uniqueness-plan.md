# First guidance x uniqueness — declared interaction experiment

2026-09-24. Continue the frozen-checkpoint investigation with two factors:
first-target learned guidance (alpha 0, 4, 16), and target uniqueness (off/on).
The four cells for each nonzero strength are original, guidance only, uniqueness
only, and both. Alpha 4 retained all baseline exact answers in the preceding
campaign; alpha 16 had the highest mean quality but lost one. Examine both
rather than assuming the stronger setting is uniformly preferable.

Reuse the completed no-uniqueness reports from first-guidance-v1 after checking
all report hashes, checkpoint/data/split/runtime/source identities and unchanged
wrapper/helper source. Run all 222 validation queries at each of three seeds
for unique=true with alpha 0/4/16: 1,998 new executions. The alpha-zero unique
cell must exactly replay the saved batched uniqueness output/metrics before
that seed's guided combinations proceed. All paths use cached FP32, batch8,
507 completion-token bound, identical post-processing, and no subject mask.

Require the uniqueness intervention to preserve every first emitted ID relative
to guidance alone at the same strength. For every changed full response, audit
that its first divergence replaces an entity that was already emitted under
the no-uniqueness policy. Save full outputs, raw/processed metrics and individual
exact gains/losses. No artificial EOS or graph-based repair.

Report paired marginal changes relative to guidance alone and to the original
model, as well as per-seed interaction:

  interaction = metric(both) - metric(guidance only)
                - metric(uniqueness only) + metric(original)

Candidate eligibility for subsequent integration: all 666 combined responses
valid/terminated and repeat-free; no per-seed processed F1 or exact-set regression
versus the ORIGINAL decoder; strict mean exact-set improvement versus original.
This gate evaluates the combined intervention against the baseline it would
replace. It is not the previous first-guidance-alone gate, and does not assert
that uniqueness causes no regression relative to guidance alone. Report those
marginal regressions explicitly. Select eligible candidates by mean exact-set
accuracy, then mean processed F1, then smaller alpha. Passing only identifies
an integration candidate; it does not promote a serving checkpoint or prove
oracle parity. No final-test use, training, production config change, commit,
push or performance claim in this campaign.
