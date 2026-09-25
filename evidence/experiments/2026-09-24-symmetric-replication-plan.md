# Symmetric auxiliary objective: paired replication plan

Retain the completed seed-1729 candidate and train the unchanged symmetric
configuration at seeds 1730 and 1731. Keep split seed 1729, data/version, batch,
optimizer/schedule, coefficients and 2,000-step final-checkpoint rule fixed.
No selection of intermediate checkpoints or lucky seeds; no final-test scoring.

Compare each against its archived prompt-set-only run at the same seed. Evaluate
all three symmetric checkpoints using uncached greedy protocol-masked decoding
to match the archived controls exactly. For seed 1729, also require all 222 raw
responses to match the previous cached candidate report before accepting it into
this replication. Cache parity for that candidate is a control, not a speed claim.

Audit data/split/config/checkpoint/source/run identities and completed training
budgets. Normalize only initialization/run name and the declared symmetric-loss
coefficient; prompt-set weight remains one in both variants. Inventory source
archive differences so code-history differences remain visible. All controls use
the same environment and the same raw metric implementation/decoding policy.

Replay the same deterministic SAME policy on every archived output. Report per
seed raw F1 and processed exact/F1, sample standard deviation, paired differences,
TYPE/COLOR breakdown and queries never exact across improved-model seeds.
Count invalid/truncated responses; require all candidates to terminate validly
for promotion. A repeatable candidate should improve processed exact and F1 in
every pair. Three seeds on one validation split establish limited replication,
not a population guarantee or oracle parity.

If the improvement repeats, retain the pre-existing symmetric seed-1729 final
checkpoint as the next research/serving reference (not the best seed selected
afterward). Validate its native HTTP behavior before updating the quickstart.
Use all 222 validation queries, the same post-policy, provenance-bound snapshot,
metadata/error checks and clean server shutdown. No permanent server or global
config change is needed. Linux/vLLM and matched-quality performance remain separate.

Implementation audit before interpreting results: replay the seed-1729 prompt-set
control under the current source with symmetric weight zero, in a fresh run.
Compare every model tensor against the archived control. This checks one full
training trajectory for unintended behavior changes with the feature disabled;
it does not claim bitwise replay for untested seeds. Preserve either result and
investigate any mismatch before relying on historical controls.

## Amendment: refresh the controls on the current source

The audit found matching normalized configurations but 0/92 identical final
model tensors. The training setting is `deterministic: false`; the cause of the
trajectory difference is unresolved, so the historical runs cannot serve as a
clean source-controlled primary comparison. Preserve that failed replay receipt.

Before examining any refreshed control's generation quality, declare fresh
prompt-set controls for all three seeds on the same source as the symmetric
candidates. Reuse the completed seed-1729 bridge run, train 1730 and 1731 with
the same 2,000-step budget, and evaluate all three uncached. The primary six-run
comparison must now have identical source file contents across all training
archives, in addition to the existing identity/configuration checks. Historical
results remain historical context. Do not change the improvement/promotion rule.

Separately compare a fixed batch's logits, losses and gradients using the old
and current implementations to check for a direct mathematical regression. This
bounded check cannot establish bitwise equivalence of complete CUDA training.

Exploratory follow-up after observing the two termination failures: inspect the
symmetric membership head at its existing zero-logit threshold for seeds 1730
and 1731, then report its predictions on the two failed queries with the same
subject exclusion. This asks whether relation classification also fails there.
It does not modify the primary comparison, tune a threshold, repair generated
answers or satisfy the failed promotion gate.
