# Margin-gradient diagnosis v1 — declared before gradient measurement

Date: 2026-09-25. Status: diagnostic plan; no new training or acceptance gate.

## Question and boundaries

The rejected m=1, lambda=0.1 screen compressed membership scores and reduced
exact answers from 192 to 6. Test whether its gradients, on fixed training
batches at fixed snapshots, are stronger than or opposed to existing losses,
and whether the relation projection has a local score-shrinking gradient.
This is a descriptive gradient experiment, not a causal replay of AdamW or a
new training intervention. It cannot rescue the failed gate or select a weight.

Authenticate the completed screen summary SHA256
`39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701`
and its independent audit SHA256
`17afc9ca0fedf184ac9e9f2794947306883d4288860b22124cd9eb0c7751bcd9`.
The diagnostic executes the same source archive SHA256
`e806549e1c778b5fc1529423e346ccf915b1484e3a5b322072fff5316c35ab6b`.
No source/config changes, parameter updates, optimizer construction, answer
generation, validation-query measurement or protected-test measurement occur.

## Fixed snapshots and sampling

Five parameter states: common seed-1729 initialization, control step 500,
treatment step 500, control final step 2000, treatment final step 2000.
Mirror the authenticated effective run recipe: seed before build_model, apply
the recorded vocabulary and sequence sizes. Separately seed/build both arm
configs and require every initial state tensor to match exactly before using
one shared initialization. Record its content identity as regenerated, not a
historically saved initialization artifact.

| Artifact under runs/ | Checkpoint SHA256 | Sidecar SHA256 |
| --- | --- | --- |
| national_dex_rank_margin_control_s1729_v1/checkpoint-step-500.pt | ba31f3d3e7e3aa56cd5ee7020a7458c93f16767c018b72ff8b4fdc4ca7337fac | 503423dddc0ec5560e159e34c8ed296a199cdfdede73827eacf04a6d1e1065e9 |
| national_dex_rank_margin_m1_w01_s1729_v1/checkpoint-step-500.pt | f74cf4774905d70bc8d8fc9643373c1ae1c53778d9bca2ade5c7b1fb6b45efd8 | da4b2d7a63c1b3af5fab22758f30ca55ab9c7aeec397d391a82005c67ea248d7 |
| national_dex_rank_margin_control_s1729_v1/checkpoint-final.pt | 8771fce8a76cf7fab9a67913a30d85490f8edce979ec353928cf4c500479c487 | cff070bab97f2ed9a741d01f32d3f457ad976d74de0cecda26e95ea4f3c714ff |
| national_dex_rank_margin_m1_w01_s1729_v1/checkpoint-final.pt | 0bb697a4a0c8a05441fb2c825ffcfc47023f91596be8f48cda639d4e57c1e8ef | 339ccb185d1bc7cc05349f1f1c7defbf973d21f2f5813cd924d4f8fe91b0ff7c |

Periodic hashes are newly authenticated by this plan; the previous screen's
final-checkpoint seal did not include them. Load through the existing checkpoint
loader with restore_rng=False and verify authoritative payload/sidecar equality,
global step, full run identity, model/TrainConfig, objective descriptor and
corpus/split identities against the authenticated source run.

Use ordered train-split slices [0:32], [800:832], [1600:1632], independently at
every snapshot: 96 distinct training queries, 15 snapshot/batch observations.
No query is selected using a gradient or outcome. Save indices, subject/dimension,
targets, prompt and label masks; record dimension counts. This sparse fixed
sample does not represent all training queries or validation behavior. Do not
change batch size or slices after seeing results.

## Numerical and loss contract

CUDA, eval mode, all parameter tensors FP32, no autocast, TF32 disabled and
float32_matmul_precision=highest. Record deterministic settings rather than
claiming deterministic GPU replay. The authenticated recipe has zero dropout;
this FP32 diagnostic still differs from original BF16 training. GPU jobs are
sequential and start only after confirming prior model jobs are terminal.

Use collate_records and full teacher-forced sequences from the selected train
queries, with masks preserved. Compute the unweighted token CE, prompt-set BCE,
symmetric BCE and raw margin. For disabled-margin states evaluate the existing
margin helper on their symmetric_relation_logits and the same shifted/masked labels, without
changing the model config. For enabled states require agreement with the model's
margin output. Authenticate weights CE=1, prompt=1, symmetric=1, margin either
0 or 0.1, first-target weight=1, continuation/z-loss/MoE disabled. Verify that
the reported total loss equals the applicable component sum within FP32 tolerance
(relative 1e-5, absolute 1e-6).

Use torch.autograd.grad separately for each objective, retain_graph as needed,
without .backward(), optimizer steps or .grad accumulation. Inspect only shared
token_embedding.weight [2049,256] and symmetric_relation_projection.weight
[256,256]. Save raw FP32 gradients and parameter arrays as non-pickle NPZ data,
along with membership logits, labels/masks and scalar losses. A parameter unused
by an objective is recorded as unused; its zero-filled analysis vector must not
be mislabeled as an observed zero derivative. Check all arrays finite, model
parameter/state hashes unchanged before/after, and .grad fields remain None.

Report separate parameter groups: all token embeddings, product rows 1024+,
the unique dimension-token rows actually present in that batch, and the relation
projection. Product/steering groups are subsets, not independent parameter sets.
No result about unmeasured transformer-block gradients is claimed.

## Fixed analysis

Calculate reductions from saved FP32 vectors converted to FP64:

- Each raw objective's gradient L2 norm and projection dot product with its
  parameter vector. Report raw and weighted margin (0.1) distinctly.
- The control-sum vector CE + prompt BCE + symmetric BCE and hypothetical sum
  control + 0.1 margin at every state. These sums describe gradients, not an
  optimizer update, and are not newly trained models.
- Cosines and norm ratios of weighted margin versus symmetric BCE and versus
  control-sum, plus symmetric BCE versus CE and prompt BCE on shared embeddings.
  Zero-norm cosines/ratios are null with an explicit reason, never zero or NaN.
- Retain every batch result and report simple means of defined per-batch
  measurements with defined counts. Never call a mean of cosines the cosine of
  averaged gradients; no averaging of gradient vectors across batches.

The relation projection is absent from CE and prompt BCE. Verify those gradients
are unused and its control-sum gradient equals its symmetric-BCE gradient.
For positive radial scaling W -> cW with fixed embeddings, logits scale as cZ.
At c=1, the raw margin's radial derivative is

    <grad_W L_margin, W> = mean_q 1[hinge_q > 0] * (max_negative_q - min_true_q).

Verify the two independently computed sides within relative 1e-4, absolute 1e-5;
save residuals. Exact ReLU-boundary and inactive rows contribute zero. Positive
dot product means an infinitesimal ordinary negative-gradient step has a radial
shrinking component, not that AdamW shrank W. Do not extrapolate across hinge
crossings or equate the c=0 tied subgradient with a one-sided scaling derivative.

## Evidence and checks

Provide Torch-free preflight for pinned inputs/slices and output refusal; test
norm/cosine/undefined cases, weighting, unused ownership, mask exclusion and
the radial identity on small synthetic CPU fixtures before GPU measurement.
Freeze script, plan, source/config archives, effective configs, input identities,
query slices and numerical settings. Refuse existing results; retain partial
raw arrays on errors, rehash inputs and parameter states after measurement.

A separate auditor independently rebuilds saved-vector reductions, symmetric
BCE/margin loss and mask arithmetic, radial arithmetic and input/artifact
identities. Token CE and prompt-BCE scalars are authenticated and used in checked
component sums; their losses and gradients are not independently regenerated.
Saved-vector audit does
not independently prove autograd or reproduce neural outputs. No claim is
accepted until those checks pass; preserve any failed diagnostics unchanged.
Report this as one seed and 96 training queries under five parameter states,
not 480 independent queries. Include results and limits in the learning notes,
academic version register (new evaluator, no trained weights) and PLM-findings.
