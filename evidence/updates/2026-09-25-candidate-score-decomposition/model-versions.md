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

The [v2 integration receipt](experiments/2026-09-25-pair-composition-shape-aware.json)
now records acceptance for these three checkpoints under that revised contract.
The [membership separability diagnosis](experiments/2026-09-25-membership-separability.json)
is a separate evaluator result on the same weights, not a new checkpoint or a
promoted inference policy.

## Hardest-boundary margin family: completed, quality gate failed

The [declared margin experiment](experiments/2026-09-25-symmetric-margin-plan.md)
has completed both fresh initialization runs at seed 1729. Stage one pairs
`national_dex_rank_margin_control_s1729_v1` with
`national_dex_rank_margin_m1_w01_s1729_v1`: the existing recipe with margin
coefficient 0.0 versus 0.1, margin 1.0, and final step 2000. Seeds 1730/1731 are
conditional on the declared first-screen gate and are not run after its failure.

Both final checkpoints reached step 2000 before generated-answer evaluation
began. The frozen campaign harness calls fresh training without a resume
checkpoint and refuses existing run directories; its training seal binds both
completed receipts. This establishes fresh initialization for these two runs
separately from the older inventory's unestablished lineages. Quality evaluation
is complete and independently audited: exact answers fall 192/222 to 6/222,
macro F1 falls 96.77% to 61.56%, and strictly separable rankings fall 167 to zero.
The declared gate fails, with one exact gain and 187 losses. Neither checkpoint
is promoted; the existing composition family remains the application reference.

| Run ID | Seed | Checkpoint SHA256 |
| --- | ---: | --- |
| `national_dex_rank_margin_control_s1729_v1` | 1729 | `8771fce8a76cf7fab9a67913a30d85490f8edce979ec353928cf4c500479c487` |
| `national_dex_rank_margin_m1_w01_s1729_v1` | 1729 | `0bb697a4a0c8a05441fb2c825ffcfc47023f91596be8f48cda639d4e57c1e8ef` |

| Arm | Recorded training configuration hash | Campaign training receipt SHA256 |
| --- | --- | --- |
| Control | `d55cd61326279169d2207790d4bbdcd9a8b85754fd9590193bb2e0075368e332` | `8fa5ce8c4029719013757b65a3d5c78a4fbdb52686e2da2b256a70d87aef64f3` |
| Treatment | `13acb9380d3f5edffee21d8d15c72c92e76a8572569402e31a9707742f3830ce` | `450427afc01338d7789c36fc9362928d6979494ecd1bfe8eadfdffe1f8a2b8f5` |

The receipts are `runs/learning/symmetric-margin-screen-v1/training-control.json`
and `training-treatment.json`. Both bind source archive SHA256
`e806549e1c778b5fc1529423e346ccf915b1484e3a5b322072fff5316c35ab6b`;
`training-seal.json` records completion before evaluation. The executed harness
SHA256 is `a5d50dc4b57277bde1cefb1cf533b4958b28b04eb515182f855b3a4aa81311b8`.
[Lesson 31](learning/31-hardest-boundary-margin.md) explains the objective and
its implementation checks and negative result. The campaign's summary SHA256 is
`39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701`;
its independent audit SHA256 is
`17afc9ca0fedf184ac9e9f2794947306883d4288860b22124cd9eb0c7751bcd9`.
The audit verifies the evidence, while the treatment fails its scientific gate.
The [portable result](experiments/2026-09-25-symmetric-margin-screen.json)
binds the independent audit and separate owner decision to these artifacts.
Historical raw configuration hashes remain historical identities: paired missing
margin fields may acquire disabled defaults only after authentication, and the
effective configuration receives a separate hash. This loading compatibility
does not permit silent historical training resume.

## Weak-margin family: completed, separation gate failed

The separately [declared weak-margin screen](experiments/2026-09-25-weak-margin-plan.md)
has completed fresh control and treatment training at seed 1729, both at final
step 2000. The control uses margin coefficient 0.0; the treatment uses 0.001
with margin 1.0. These are new training runs, not continued training from the
failed 0.1 treatment or the existing composition checkpoints.

| Run ID | Seed | Checkpoint SHA256 |
| --- | ---: | --- |
| `national_dex_rank_margin_weak_control_s1729_v1` | 1729 | `1024045780668d961e993881dbfa89edac0bebc99a8f6e9c048092607a804f63` |
| `national_dex_rank_margin_m1_w0001_s1729_v1` | 1729 | `75dab72aa7f4d6ae3d5d10deda2a34fec473b43d47b702ecbce5c899d7054e16` |

| Arm | Recorded training configuration hash | Campaign training receipt SHA256 |
| --- | --- | --- |
| Control | `d55cd61326279169d2207790d4bbdcd9a8b85754fd9590193bb2e0075368e332` | `98bb3e7845bceeb2c3f35b43e99c04b698885f5d00db6bb231970b83a4329a0b` |
| Treatment | `faf03894c030ecb25c045f31bec7855aa617acb9965c60f6b4f9b5a278784a67` | `0e8178cc74fe6f57da3a84a40dff89e82d6d818b462050c88247e3b0fabb4b3a` |

The receipts are `runs/learning/weak-margin-screen-v1/training-control.json`
and `training-treatment.json`. The campaign calls training without a resume
checkpoint; its `training-seal.json` binds both completed receipts before
generated-answer evaluation. This fresh-initialization evidence is separate
from the older families' unestablished lineage. The control's configuration
hash matches the earlier margin control because run names are excluded from
that identity; its different checkpoint hash identifies a distinct trained
artifact, not a reused checkpoint.

The seal binds executed runner SHA256
`df4b41f30d4a399cda88e62cc8334ad58522da7de191741ced31786caef8dff5`
and unchanged neural source archive SHA256
`e806549e1c778b5fc1529423e346ccf915b1484e3a5b322072fff5316c35ab6b`.
Quality evaluation and independent audit are complete. Exact selected sets
improve 191/222 to 194/222, with 11 gains and 8 losses, and macro F1 rises
.967657 to .974821. Strict head separation declines 168 to 164, failing the
unchanged gate despite nondecreasing exact counts in all three groups.
The [portable result and owner decision](experiments/2026-09-25-weak-margin-screen.json)
accept the evidence and reject the variant. No further-seed replication,
default change or checkpoint promotion is authorized; the existing composition
family remains the application reference.

The completed summary SHA256 is
`3d535d1e298497609169dda8791aef3920d3d8993afab9189df359644b8c7a93`;
the independent audit SHA256 is
`5cf9d5b556f06f068bcfda4709d737f351260f21144bc70ec42c305feef9caab`.
[Lesson 33](learning/33-loss-weight-is-not-gradient-strength.md) explains why
candidate selection can improve while global score separation worsens, and why
this adaptive single-seed result is not an accepted training recipe.

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

### 2026-09-25 follow-up: 25 completed final checkpoints

The [margin follow-up inventory](experiments/2026-09-25-margin-model-version-inventory.json)
adds the two completed runs above: **25/25** discovered final checkpoints pass
the byte-and-receipt consistency checks, with zero failed runs. Its SHA256 is
`58e5951903e64801d94d8f4c161a60418930cbf3f8e2b0d5ac73839c1ce24be6`.
The earlier 23-run inventory remains an unchanged historical snapshot.

The same immediate-directory scope and limitations apply: no payload tensor
inspection, model replay, independent corpus/evaluation revalidation or quality
ranking is established by this inventory. It does not infer training parents;
the fresh-initialization evidence for the two new runs comes from the separate
campaign harness and seal described above. This inventory records completed
artifacts; the failed quality comparison is established separately by the
campaign evaluation and audit, not by the inventory checks.

### 2026-09-25 weak-margin follow-up: 27 completed final checkpoints

The [weak-margin follow-up inventory](experiments/2026-09-25-weak-margin-model-version-inventory.json)
adds the two newly completed weak-margin runs: **27/27** discovered final
checkpoints pass the byte-and-receipt consistency checks, with zero failed
runs. Its SHA256 is
`837d930dbfc75144cd1bc038278ad2ceeddae488de62ad14bed9f4891950b9f8`.
The earlier 23-run and 25-run inventories remain unchanged historical snapshots.

The same immediate-directory scope and limitations apply. This inventory does
not deserialize checkpoint tensors, replay models, establish training parentage
or measure quality. The new campaign's training seal supplies the separate
fresh-training evidence; the separately audited scientific comparison rejects
this variant. Both completed checkpoints remain registered, keeping the count
at 27 rather than deleting an unsuccessful experiment.

## Gradient diagnosis: new evaluator, existing parameter states

The [gradient lesson](learning/32-measuring-gradient-interference.md) and
[portable result](experiments/2026-09-25-margin-gradient-diagnosis.json) report a
completed, independently audited diagnostic under its
[fixed plan](experiments/2026-09-25-margin-gradient-diagnosis-plan.md). It measures
96 training queries at five states: regenerated shared initialization, each
margin-screen arm's step-500 checkpoint and each final step-2000 checkpoint.
No optimizer or parameter update is performed; the final-checkpoint inventory
remains 25. The regenerated initialization has a recorded content identity but
is not a historically saved training checkpoint.

The two step-500 artifacts receive new explicit verification here, separately
from the earlier final-checkpoint seal:

| Existing periodic checkpoint | Checkpoint SHA256 |
| --- | --- |
| `national_dex_rank_margin_control_s1729_v1/checkpoint-step-500.pt` | `ba31f3d3e7e3aa56cd5ee7020a7458c93f16767c018b72ff8b4fdc4ca7337fac` |
| `national_dex_rank_margin_m1_w01_s1729_v1/checkpoint-step-500.pt` | `f74cf4774905d70bc8d8fc9643373c1ae1c53778d9bca2ade5c7b1fb6b45efd8` |

The plan pins their sidecars as well; the runner validates payload/config/run
identity. Its full-model state hashes and empty gradient buffers attest no
mutation, while the independent auditor checks the saved parameter blocks and
gradient arithmetic. This does not independently prove autograd or replay the
historical optimizer. The local summary SHA256 is
`5a0927acf533a2a170d829a781e52e45686961e909e5a7f697ad193de3dcc607`;
the audit SHA256 is
`c2fe33de149f1ade6d28475b952833a304739c9fd32aafd0064b5ea080b0cbd5`.
This diagnostic adds no new answer-quality result or promoted checkpoint.

## Saved candidate/score diagnosis, 2026-09-25

The [candidate-score report](experiments/2026-09-25-candidate-score-decomposition.json)
reuses the two retained weak-margin checkpoints through their authenticated
evaluation reports. It adds no recipe, weights or inference-policy version.
The final-checkpoint inventory remains 27. Four offline pool/score combinations
are diagnostic cells, not four newly trained or deployed models.

Primary SHA256 `c325c76d0137da15ca73a157b1289e1743063489397690a8b6976276e2beef2d`,
independent audit `7d65ffca77cf4a34f991880c9c0ebcf76d47a40e57f4e4bdefd49f2dac9ab23e`,
and separate decision `b291793026ca7f65ee0c18b4d851041896e491403f6a30032a591acf685329da`
bind this analysis. Acceptance concerns reconstruction of saved evidence;
the weak-margin gate remains failed and the accepted application family unchanged.
Raw neural reports and checkpoint payloads remain local, hash-bound artifacts.

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
