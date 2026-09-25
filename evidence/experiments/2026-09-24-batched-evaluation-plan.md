# Batched validation and decoder timing — declared before evaluation

2026-09-24. Implement opt-in generation batches, size eight for this campaign,
while retaining the serial default. Support the existing constrained KV decoder
and its optional unique-target rule. Validate every prompt before model work,
keep per-row seen/finished state, preserve input ordering and duplicate queries,
and never append dummy tokens or invented EOS to returned answers. Finished
rows may occupy cache slots until the group ends; no continuous batching claim.

Compare all 222 validation queries at each frozen symmetric seed 1729/1730/1731,
for both original and unique-target policies, against all six serial reports
from the preceding campaign. Require identical complete token sequences,
termination/validity/errors and metrics on all 1,332 responses. Check checkpoint,
corpus, split, model-source and runtime identities. Save failures as failures;
do not relax the parity gate or silently fall back to the serial backend.

Only after full parity passes, time the fixed first eight seed-1729 validation
queries under the original policy. Warm up serial and batch paths, then measure
three paired repetitions with alternating execution order, CUDA synchronization
at boundaries, and output checks on every repetition. Report total decoder time
for the eight requests, throughput and peak allocated memory. Timing runs alone,
not alongside training/evaluation work. This is an offline decoder benchmark,
not HTTP latency, concurrent-user SLO, energy, oracle advantage or new model quality.

Default batch size remains one even if the experiment succeeds. No native API
scheduler change, checkpoint selection, retraining or final-test evaluation.
