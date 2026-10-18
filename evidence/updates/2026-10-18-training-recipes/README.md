# Training-recipe synthesis and later bilinear evidence

Publication date: **2026-10-18**. This addition publishes curated aggregate results and a teaching-oriented [technique guide](../../../../papers/14-training-recipes-and-techniques.md). It does not rerun training, product tests, or the historical independent audits.

## Provenance

Source snapshot: `mateors1/Plm-protocolized-language-model-pika-edition` at **de8338f0dbd644bda1e9746d84a36fe9fb53dc22**. It is private. Source paths below identify provenance, not publicly accessible reproduction instructions. Its `PAPER.md` v0008 is dated 2026-10-02 with evidence cutoff 16:53 UTC that day; the residual campaign itself is dated 2026-09-26. Publication date is not experiment date.

- `docs/experiments/2026-09-26-bilinear-budget32000.json`: source of [bilinear-summary.json](bilinear-summary.json). Selected aggregate/per-seed measurements, checkpoint identities, final decision, pre-audit gate state and preserved failed-audit provenance are copied into a curated object. Raw score/training matrices and full report rows are not included.
- `configs/experiments/bilinear_budget32000_v1.json`: selected fields in [recipe-summary.json](recipe-summary.json); this is not a standalone executable configuration.
- `docs/learning/42-learning-cross-coordinate-relations.md`: full versus diagonal bilinear mathematics, frozen-tensor/parameter counts and the failed 500-update pilot.
- `docs/learning/48-testing-a-budget-across-parent-seeds.md`: 2k/8k paired counts, fresh versus historical evidence, and preserved 8k verifier defect/repair.
- `docs/learning/50-learning-symmetric-offsets.md`: affine v3 unchanged-output negative result and its two zero-update implementation failures.
- `PAPER.md`: later dense CPU integration, feature-cache and worker-control interpretation. `SHOWCASE.md`: private seed-1730 demo bundle; not a public weight release or evidence of all-checkpoint archival.

The public historical lessons and papers remain unchanged. They describe their original cutoff; the new guide supplements rather than overwrites them. Existing evidence checksum inventories remain valid.

## Acceptance layers must remain distinct

The source's primary `gate.accepted=false` is an **awaiting-independent-audit field**, not the final decision. This addition retains it as `source_pre_audit_gate`. The separately identified final decision records `evidence_accepted=true`, `fixed_quality_gate_passed=true`, and `model_promoted=false`. These are different stages, not contradictory final statuses.

Seed 1730's original 32k audit failed on text comparison of equivalent timezone instants. A versioned audit compared exact instants; original failed evidence remained unchanged, and there was no training rerun. The source report identifies both failed and successful audit hashes. The 8k campaign had a different earlier verifier serialization failure, repaired without retraining. Scientific failed gates (including the 500-update pilot) are not repaired into successes by these evidence-processing changes.

## What can be reproduced from this public addition?

Readers can inspect arithmetic, recipe settings, per-seed counts and the scope of claims. The summary has 660/666 exact observations, macro F1 0.9999637956903659, seven false negatives and one false positive. Each seed uses the same 222 development queries; 191 protected queries remain unused for these results. Six failed seed-query outcomes cover five distinct queries, all dual-TYPE. The fitting stage changes only the residual, with all 93 parent tensors preserved.

These selected summaries are **not sufficient to replay training or independently recompute predictions**. Weights, raw matrices, protected data, full training/serving code and complete original run payloads are not published here. Historical source hashes authenticate identities, not present public access. No new audit execution or deployment is implied. The CPU/cached adapter remains opt-in, with no promotion or matched-quality serving superiority.

[SHA256SUMS.txt](SHA256SUMS.txt) fingerprints this addition's README and JSON summaries. It is separate from the historical inventories.
