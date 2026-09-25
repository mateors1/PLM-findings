# Academic model version register

Academic traceability is a project requirement. A result must identify the
trained artifact, inference procedure and evaluation procedure separately.
The package version (`0.1.0`) does not identify a trained model.

## What receives a version

| Object | Identity and change rule |
| --- | --- |
| Model recipe | Architecture, objectives, training schedule and resolved configuration. A changed recipe receives a new named experiment revision. |
| Trained artifact | Existing run ID, seed, global step and checkpoint SHA256. Further training produces a new artifact; replicas keep distinct seed identities. |
| Inference policy | Decoder constraints, guidance, candidate construction, selection and serving configuration/source hashes. A changed policy is a new system variant even with unchanged weights. |
| Evaluation | Declared plan, evaluator source/version, data/split hashes, runtime, numerical settings, batch shape and report hashes. A revised verification contract receives a new campaign identity. |

Retain parent checkpoint/run references for continued training and state the
change and hypothesis. A new random-seed replica is not an improvement claim.
Unknown historical parentage must remain explicitly unknown until verified.
Do not infer parentage from names such as `continuation` or `control`.

Preserve old checkpoints, configs, source archives, predictions and failed
reports. Corrections are new records referencing the superseded evidence.
Human-readable labels are aliases for content identities, never substitutes
for hashes. A Git commit alone is insufficient when a run uses a dirty tree;
retain the executed source archive and its manifest too.

## Current checkpoint family used in composition experiments

Registered on 2026-09-25 using existing run IDs, without renaming artifacts.
All three checkpoint files were rehashed and matched their sidecars. Each is
at global step 2000. This table identifies the current family. The historical
inventory below covers available final checkpoints in a declared local scope.

| Run ID | Seed | Checkpoint SHA256 |
| --- | ---: | --- |
| `national_dex_continuation_control_s1729_v1` | 1729 | `e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1` |
| `national_dex_continuation_control_s1730_v1` | 1730 | `5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2` |
| `national_dex_continuation_control_s1731_v1` | 1731 | `ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d` |

Local artifact paths follow `runs/<run ID>/checkpoint-final.pt`, with a
`checkpoint-final.pt.json` sidecar and `run.json` receipt. Those receipts retain
resolved configuration and experiment identity. Their recorded training source
commit is `8afff361447b1357a041aa65f5b588af5b7ba5b5`; use the bound source archive
as well when reconstructing the executed code.

| Seed | Recorded training configuration hash |
| ---: | --- |
| 1729 | `6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148` |
| 1730 | `0eae5252ae9de2d1463992a319c43e695ae8bb4f24bfb2ae94acdf4da6033b31` |
| 1731 | `9e029d5fa280492a382b3b138eb5933a8ef8580c6a95a90deb0c87420c3601b3` |

These weights are reused by the four-path selector, pair-composition experiment
and direct-membership ablation. Those are different inference variants, not
three newly trained models. Likewise, shape-aware verification v2 changes the
verification contract; it does not create new weights or erase v1's failure.
See [lesson 29](learning/29-shape-aware-verification.md) for that distinction.

## Historical final-checkpoint inventory

The [2026-09-25 inventory](experiments/2026-09-25-model-version-inventory.json)
discovers **23 runs** through immediate `runs/*/run.json` files. All 23 final
checkpoint byte hashes agree with their training results and sidecars. Recorded
training/model settings, seeds, steps and data/split identities agree; archived
source, source-tree and dependency-lock hashes also pass. The inventory retains
full historical configurations rather than filling in today's new defaults.

This checks bytes and receipt consistency without loading Torch or deserializing
checkpoint payloads. It does not rerun models, validate all tensor contents,
rehash corpus files or reassess quality. Periodic checkpoints and nested scratch
runs are outside this inventory. Parent checkpoint provenance is explicitly
unestablished by these receipts; a run name is not evidence of continuation.

The producing script is `scripts/inventory_model_versions.py`; its hash is bound
in the report. The immutable local report is
`runs/learning/model-version-inventory-20260925/summary.json`, SHA256
`0c9480d6dd542fc5ea69e60ebb66c945612bbcd81964d1c27abd6d366b76751f`.
Sixteen focused CPU tests cover contradictions, missing metadata, empty inventory
and overwrite refusal. Independent review corrected a training-subsection schema
assumption before the inventory was produced.

## Minimum record for a reported result

Every future reported variant should link its recipe/run IDs and parent lineage,
checkpoint hash, training data/protocol/vocabulary/split identities, executed
source/configuration, seeds, environment and inference policy. Bind the exact
evaluation plan and reports, including failed gates and development versus
protected-test status. State whether evidence is fresh or reused.

Use a descriptive citation tuple such as
`run ID + checkpoint hash + inference policy + evaluation report hash`.
Do not cite an unqualified "latest model". Selection or publication requires
an explicit evidence status; presence in this register is not promotion.

Checkpoint and raw run files currently live in Git-ignored local directories.
Hashes establish identity, not availability or backup. Before academic release,
archive the referenced artifact bundle durably and record its retrieval location,
license and checksums. No public archive, DOI or backup is established by this
register. Periodic checkpoints, verified parent lineage and links from every
historical variant to its evaluations still need cataloguing before a complete
version-history claim.
