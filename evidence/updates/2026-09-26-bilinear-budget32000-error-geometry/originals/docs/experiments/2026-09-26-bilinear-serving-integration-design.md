# Serving the bilinear candidate: implementation design

Status: source-reviewed design, not an implemented or accepted serving policy.
This document does not authorize a claim of oracle parity, publish weights or
change any default. The accepted scientific reference remains the separately
versioned 32,000-update saved-validation experiment.

## Why this is a separate integration

The candidate's score uses trained product embeddings and relational weights.
It does not run the autoregressive transformer blocks during dense scoring.
The decoder contributed learned representations during training, but direct
set scoring is a different computation at inference. It cannot inherit the
earlier decoder throughput measurements or generated-sequence provenance.

`NativePredictor.from_checkpoint` currently calls `load_inference_runtime`,
which builds the original model and strictly loads its state, then requires
the causal objective and evaluator metadata. A derivative with the additional
`symmetric_bilinear_residual` parameter and a different training objective is
not a supported input to that factory. Registering the parameter would still
not make the original forward method apply its correction.

Existing `predict_set` demonstrates bounded execution and postprocessing of
selected sets. Its result type describes four generated paths and ten original
or union candidates. Those fields would be false provenance for dense scoring.

## Proposed contract

Add a separately named default-disabled dense-set policy, loader and endpoint.
Keep the original causal loader and both existing prediction paths unchanged.
The request supplies only product subject and TYPE/COLOR SAME protocol tokens;
IGNORE and limit remain postprocessing metadata. Return selected ascending
product IDs with checkpoint/scorer/policy identity. Do not fabricate EOS,
termination, token histories or generated-source ranks. Define an empty selected
set as a valid scored-set outcome, distinct from a generation failure.

The candidate's direct computation is

\[
Z=Z_{parent}+E_s^\top\operatorname{sym}(A_d)E^\top/16,
\quad E:[1025,256],\quad A:[2,256,256],\quad Z:[B,1025].
\]

Extract an independently versioned deployment implementation from the frozen
experimental arithmetic; never edit the executed training scripts to integrate
them. Reuse source concepts from `refit_membership_projection._features/_head`,
`refit_bilinear_residual._residual`, and the fixed positive-score selection.
Production scoring must not import label-reading experiment orchestration.

The new loader must validate derivative architecture, scorer, training objective,
parent/seed/update identity, vocabulary/data identity and checkpoint hash, attach
the residual before strict state loading, and reject missing/unexpected tensors.
Preserve the original checkpoint metadata's `standard_serving_supported=false`;
a new deployment receipt establishes adapter support without rewriting history.

Current standard startup reads corpus records to reconstruct dataset splits.
The new deployment loader should instead validate pinned manifest/vocabulary
identities without loading teacher records or protected examples. Graph access
for hydration must remain limited to active product keys/labels, as in
`HydrationIndex.from_snapshot`; relation edges cannot inform scoring.

## Implementation locations and required evidence

The source review identified `src/plm/config.py`,
`src/plm/serving/runtime.py`, `src/plm/serving/native.py`,
`src/plm/serving/set_composition.py` and `src/plm/model/layers.py` as current
boundaries. A new serving scorer/result module is preferable to altering the
meaning of `SetCompositionResult`. Health and deployment receipts must identify
dense scoring separately from the legacy decoding fields.

Before real checkpoint execution, freeze a concrete plan and synthetic tests:

- Arithmetic parity with authenticated original functions; zero-residual replay,
  symmetry, tensor shapes, FP32/finite checks, signs/ties and self exclusion.
- Strict loader acceptance/rejection, including architecture/objective/parent,
  data/vocabulary/hash and tensor mismatch. Preserve original loader rejection.
- No teacher labels or graph relations at startup/request scoring; IGNORE/limit
  cannot change the complete raw selection.
- Distinct response schema, disabled-policy behavior, errors, empty sets,
  admission release and shared execution limits. Existing endpoints regressions.

Then replay all three accepted validation checkpoints through the integration.
The accepted policy uses FP32 batches of eight with a final batch of six. A
native one-query request changes numerical shape; compare both scores and sets
to an explicitly identified reference at that shape. Do not silently treat a
shape change as the original scorer identity or relax parity after a failure.
Protected evaluation remains closed during adapter development.

Only after integration correctness passes should a new benchmark measure this
same scorer's quality, HTTP latency, concurrency and power against the graph
on matched requests and output semantics. Fixed-group throughput, loopback HTTP,
CPU-package power, GPU-board power and wall power remain distinct measurements.
An opt-in deployment adapter is not model promotion or an oracle-parity result.
