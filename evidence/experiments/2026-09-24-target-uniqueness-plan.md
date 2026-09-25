# Target uniqueness decoding experiment — declared before evaluation

2026-09-24. Test an opt-in `eval.prevent_repeated_targets` flag on the existing
symmetric seeds 1729/1730/1731. Do not retrain, alter weights, change the corpus,
change the 507-token budget, supply relation labels to decoding, exclude the
subject before generation, or force EOS. Mask only entity IDs emitted after
ANSWER. Keep defaults off and preserve the original repeated-output evidence.

For all 222 validation queries in every seed, execute cached greedy decoding
with the flag off and require exact output/validity/error parity with the
archived uncached report from the preceding replication. Then execute the
flag-on policy on the same query/checkpoint. Both paths use the same current
source, cache mode and model. Audit identities and ensure model source files
are unchanged. Save all raw outputs and source/checkpoint/report hashes.

Report per-seed raw and processed F1/exactness, validity/termination, duplicate
rates, changed queries, newly exact and lost-exact queries. Report paired mean
and sample SD. Count failed-to-complete queries separately from partial-credit
quality. This measures a model-plus-decoder change, not newly learned knowledge.

A candidate qualifies as the new research reference only if all 666 flag-on
responses are valid and terminated, and processed F1 and exact accuracy do not
regress in any seed. Retain the pre-existing seed-1729 checkpoint rather than
selecting the best new result. If qualified, verify all 222 seed-1729 responses
through the native HTTP API under the explicit policy, preserve metadata and
identity checks, and stop the owned verification server. Only then update the
quickstart. Final test, oracle parity, energy and concurrency remain separate.
