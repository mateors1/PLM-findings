# Lesson 17: moving a tested policy into the application

**2026-09-24.** The combined guidance/uniqueness experiment passed its candidate
gate. This iteration connects that policy to the normal evaluator and native
API, with an explicit default-zero guidance setting. The
[declared verification plan](../experiments/2026-09-24-guided-integration-plan.md)
requires real-data token parity before accepting the integration.

## A model and a decoding policy are different artifacts

A checkpoint contains learned weights. A decoding policy describes how we use
its scores to choose outputs. The same checkpoint can run ordinary greedy
decoding, uniqueness-constrained decoding, or first-target guidance plus
uniqueness. Their answers can differ even though no weight changed.

We therefore preserve the old **training identity** and record a new
**inference configuration identity**. A config hash changes when the inference
settings change; that does not mean the model was retrained. The implementation's
source archive is recorded separately because a software change can also alter
outputs without changing either the weights or the requested settings.

The new evaluation setting is:

```yaml
use_kv_cache: true
constrained_decoding: true
prevent_repeated_targets: true
first_target_guidance_alpha: 16.0
```

Guidance defaults to zero. Positive strengths must be finite and nonnegative,
require constrained cached decoding, and require a checkpoint configured with
the symmetric relation head. The number 16 is an inference setting, **not a
model input token**. The protocol still starts with the same five prompt IDs;
IGNORE and return limits remain post-processing metadata.

## One scoring implementation, two generation paths

The shared helper receives the model, prompt tensor and cached prefill result.
It calls the existing prompt-only forward path to obtain relation scores and
adds the same FP32 log-sigmoid penalty used in the experiment:

```text
logits[:, -1, 1024:] += alpha * logsigmoid(relation_scores)
```

For candidate entity $j$, the first-token choice is therefore

$$
j^* = \operatorname{argmax}_{j \in \text{allowed IDs}}
\left[z_j + g_j\right], \qquad
g_j = \begin{cases}\alpha \log \sigma(s_j) & j \text{ is a product},\\
0 & j = \mathrm{EOS}.\end{cases}
$$

Here $z_j$ is the decoder logit (an unnormalized score), $s_j$ is the
symmetric head's membership score, and $\sigma(s)=1/(1+e^{-s})$ maps that score
into $(0,1)$. For product candidates, exponentiating the adjusted score gives
$e^{z_j}\sigma(s_j)^\alpha$: guidance multiplies the decoder's unnormalized
weight by a membership factor. EOS receives no membership penalty. The head's
sigmoid output is a learned score; we have not established probability calibration.

At alpha 16, a membership score of 0.5 adds about -11.09 to a product logit,
whereas 0.9 adds about -1.69. That is strong steering. It explains why alpha is
an experimental choice with measurable regressions, rather than a harmless knob.
Uniqueness supplies a separate hard constraint: an already emitted entity gets
score negative infinity. Guidance changes preferences; uniqueness removes choices.

The serial generator invokes it on a `[1, 5]` prompt; the offline batch path uses
`[B, 5]`. Product scores have shape `[B, 1025]` for the current catalog. The
helper clones the logits and preserves the cache object. Both generators call
it only at the first step. Later `[B, 1]` chunks use ordinary cached decoding.

At zero strength, the extra head call is skipped and the existing decoder label
is retained. Positive guidance adds a canonical strength suffix. Its numeric
format distinguishes nearby fractional strengths, so two settings do not collapse
into the same label merely because a display format rounded them.

This implementation retains the extra prompt forward used by the experiment.
Factoring the membership calculation into a cheaper method could be a later
optimization, but combining that change with integration would make failures
harder to localize. There is no new speed claim in this lesson.

## What token parity protects

A neural implementation can look mathematically equivalent yet choose different
tokens because of floating-point details, masks or cache state. A single changed
token can redirect the rest of an autoregressive answer. Comparing only average
F1 can conceal such differences.

The offline check uses all 222 validation queries at each of the three frozen
checkpoints for alpha zero and alpha 16, with uniqueness enabled. It compares
1,332 complete sequences against saved experimental outputs. Query ordering,
expected data, checkpoint hashes and model-source bytes are checked before
comparing tokens, parsed fields, errors, termination and metrics.

This is a **regression check**: the new application path should preserve the
specific behavior already selected. It does not ask the model to learn something
new and does not evaluate the protected final test.

## Why the HTTP check is stronger now

The API uses serial model execution. Therefore the second check sends all 222
queries per checkpoint through real loopback HTTP and compares their outputs
with the validated offline references. It verifies:

- Full token IDs, targets, termination, validity and errors.
- Guidance strength, bound, cache/constraint/uniqueness settings and native batch size one.
- Canonical decoder identity in the deployment, readiness response and each answer.
- Training/checkpoint/data identities and inference configuration hash.
- Each processed target list and the aggregate raw/processed metrics.
- IGNORE/limit behavior, invalid-subject rejection and owned-server shutdown.

The only normalized label component is the expected offline batch marker. The
HTTP server still runs the same policy and weights, but processes one query at
a time. The normalization must not hide a different guidance strength or mask.

Older reports sometimes lack token IDs. The verifier now labels those as
partial parsed-response parity. Positive-guidance verification requires complete
token evidence. This makes the strength of the claim explicit instead of
implying that every historical report contains the same evidence.

## Tests that caught useful mistakes

The integration fixture originally put every product in the same relation group.
That gave the symmetric auxiliary loss no negative products once the subject was
excluded. The fixture now has two groups, providing both positive and negative
examples. The real corpus was not changed.

Known-probability tests check the exact guidance arithmetic. Other tests cover
zero bypass, unsupported configurations, missing/malformed heads, per-row state,
cache preservation, serial/batch parity and request metadata. The real trained
fixture exercises both successful and incomplete responses through the API;
an incomplete model answer must remain an explicit failure with its raw tokens.

Parallel workers owned the inference helper and the HTTP verifier separately,
while configuration, application wiring and integration fixtures were handled
alongside them. Source changes finish before the GPU evidence is collected.

## Measured integration results

The full automated suite passed **315 tests** before the campaign. Ruff,
formatting and strict typing passed; the core import path also worked with
Torch imports blocked. A separate static review found no additional actionable
correctness issues.

All **1,332 offline executions matched** the archived experimental answers:
222 validation queries x three frozen checkpoints x two strengths (0 and 16),
with uniqueness enabled in both. Complete token IDs, parsed fields, errors,
termination and raw/processed aggregate metrics matched. The model source and
checkpoint identities remained unchanged. The new inference source archive is
`9d65344bb13ee3844bb69de531c55503357b64364f94bd37deb8a6cfafbbefdf`.

The [portable offline receipt](../experiments/2026-09-24-guided-integration.json)
records the six report hashes and runtime identities. Full sequences and the
executed source/script copies are under `runs/learning/guided-integration-v1/`.
The subsequent real HTTP campaign passed all **666 comparisons**: 222 per
checkpoint, using serial model execution. All complete raw tokens, parsed
fields, per-query processed targets and aggregate metrics matched. Deployment
and health identities, metadata handling and invalid-input rejection passed.
All three owned servers stopped cleanly.

The [portable HTTP receipt](../experiments/2026-09-24-guided-http.json) records
report and deployment hashes, policy settings and parity evidence. Complete
responses and executed script copies are under `runs/learning/guided-http-v1/`.
A separate CPU pass rechecked report/script hashes, full tokens, processed exact
counts, training identity and source/runtime identity.

| Training seed | Exact processed answers | Processed F1 | HTTP token parity |
| --- | ---: | ---: | --- |
| 1729 | 151/222 | 0.885711 | 222/222 |
| 1730 | 150/222 | 0.882990 | 222/222 |
| 1731 | 153/222 | 0.883270 | 222/222 |

These are repeated observations of the same 222 validation queries across
three checkpoints, not 666 distinct queries. Their mean processed F1 is
0.883990 and their combined exact count is 454/666 (68.17%). TYPE accounts
for 153/357 exact observations; COLOR accounts for 301/309.

## What this enables, and what remains open

The selected policy is now an explicitly configured, verified application
path. Default guidance remains zero; the older prompt-set serving reference
remains available for historical comparison. Nothing was retrained.

Next, investigate why TYPE continuations omit correct members after a useful
first choice. A completion constraint can stop loops without solving coverage.
Oracle parity, protected final-test finalization and matched-quality serving
measurements remain open. These serial loopback checks establish no concurrency,
latency-SLO or energy advantage. No permanent service was left running.
