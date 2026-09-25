# Saved-checkpoint comparison: rule fixed before generation

Scope: the existing prompt-set seed-1729 run, steps 500, 1000, 1500 and 2000
(the final checkpoint represents step 2000). This is an exploratory within-run
validation comparison. No new training, oracle hints, corpus changes or final
test scoring. Existing three-seed final-checkpoint results remain historical facts.

- Validate all checkpoint payload/sidecar steps, hashes and training identities.
- Use identical cached FP32 greedy protocol-masked generation, the same 507-token
  completion bound and all 222 validation queries. Keep all raw responses.
- Apply the existing SAME self-exclusion/deduplication policy identically; no
  IGNORE list or return limit. Invalid/truncated responses stay invalid.
- Eligibility: every query must terminate with valid protocol syntax. Among
  eligible checkpoints, maximize processed exact-set accuracy; break ties by
  processed macro F1, then prefer the earlier step. If none is eligible, report
  no selection. Report raw metrics and per-dimension metrics regardless of eligibility.
- Separately measure FP32 teacher-forced token CE, first-target CE and auxiliary
  set loss over the validation split. These contextual losses do not select the
  checkpoint. Do not compare them as numerically identical to historical BF16 loss.
- Require step-2000 raw response parity with the archived reference; verify report
  identity and full query coverage before interpreting the sweep.
- Save a source archive, exact analysis script, run/reference/plan hashes and
  immutable per-checkpoint reports. Do not change serving defaults automatically.

This rule was chosen after seeing earlier validation findings, so the winning
validation score is selection-biased. It is neither an untouched estimate of
generalization nor a completed across-seed selection procedure. A better earlier
checkpoint would motivate replication; failure to improve is also a useful result.
