# Lesson 7: remember earlier attention work

**2026-09-24.** [Lesson 6](06-seed-replication.md) repeated a learning improvement.
This experiment changes inference execution while retaining the learned weights.
The [pre-run plan](../experiments/2026-09-24-kv-cache-plan.md) fixes the checkpoint,
correctness checks and timing procedure before measurement.

**Result:** cached execution reproduced all 222 validation responses exactly.
In the local paired microbenchmark it reduced total generation time by about
4%, with a small decrease in peak allocated memory. The reduction in arithmetic
did not translate into a large latency win for this small workload.

## Measured result: correct, but only modestly faster

The [archived report](../experiments/2026-09-24-kv-cache.json) contains all timings,
input hashes, checkpoint identity and the source/runtime used. There are eight
prompts, three repetitions, and thus 24 measured requests per execution path.

| Measurement | Uncached | Cached |
| --- | ---: | ---: |
| Sum of 24 request times | 20.352 s | 19.527 s |
| Median request time in this mixed-length workload | 0.853 s | 0.812 s |
| Maximum peak allocated GPU memory | 37.96 MiB | 35.63 MiB |
| Validation responses matching the archived reference | Reference | 222/222 |

The ratio of summed times is 1.0423, or about 4.06% less time with caching.
Per-prompt ratios of summed times range approximately from 0.99 to 1.055; one
prompt is slightly slower. Three repetitions in one session do not establish
a broad or precisely estimated speed advantage. The measurements exclude
training, startup and model loading, and measure allocated tensors rather than
the GPU allocator's reserved memory or the board's total memory consumption.

Raw validation F1 remains 0.629018, with zero raw exact sets and valid EOS on
222/222 queries. This optimization preserves the model's correct answers **and**
its errors. It does not improve learned relation quality.

## Why generating one token costs more than one token's work

The reference generator passes the entire growing prefix through the model each
time it needs another token. After the five prompt tokens, it processes lengths
5, 6, 7, ... . Earlier hidden states are repeatedly reconstructed, although causal
attention prevents later tokens from changing them during inference.

Each attention layer projects hidden vectors into **queries**, **keys** and
**values**. A query asks which earlier positions are relevant; keys supply the
matching vectors; values supply the vectors to mix. For one attention head:

```text
attention(Q, K, V) = softmax(Q @ K.T / sqrt(head_dim) + causal_mask) @ V
```

Once an earlier token's key and value have been computed, they can be reused for
the next token. We do not need to cache old queries: their outputs were already
calculated, and only the new position needs an attention output now.

This argument depends on causal attention and inference mode. Dropout must be
disabled, the model weights unchanged, and the cache must describe the same
prefix. The implementation rejects training/autograd use and keeps cache state
out of model checkpoints.

## Prefill and decode: the tensor shapes

**Prefill** processes the prompt as one chunk. **Decode** appends new tokens,
usually one at a time. Our small model has width 256, eight query heads, two
key/value heads and head width 32. For batch B, existing prefix P and new chunk S:

```text
New token IDs:           [B, S]
New hidden states:       [B, S, 256]
New queries:             [B, 8, S, 32]
New keys and values:     [B, 2, S, 32] each
Previous K/V cache:      [B, 2, P, 32] each, per layer
Appended K/V cache:      [B, 2, P+S, 32] each, per layer
Attention scores:       [B, 8, S, P+S]
Token logits:           [B, S, 2049]
```

For one request the initial prefill uses S=5, P=0. The next call uses S=1, P=5.
The caller supplies only the new token while the cache supplies earlier keys and
values. We store K/V **before** repeating them to the eight query heads for GQA.
The current attention implementation still creates that repetition temporarily.

The cached path uses the same projections, attention math, feed-forward blocks,
final normalization and token head as the reference. It skips training losses
and the auxiliary prompt-set classifier: neither chooses the next generated
token. That also avoids incorrectly looking for an ANSWER position inside a
one-token decode chunk.

## Absolute positions matter

A one-token input chunk does not mean position zero. If P tokens are cached, its
first token is at position P. RoPE applies that absolute positional rotation to
new queries and keys. We cache keys after normalization and rotation, so old keys
are not rotated a second time.

The causal mask also needs absolute positions. For a two-token chunk appended to
three cached tokens, the allowed key positions are:

```text
                  key position
                0  1  2  3  4
query at 3      1  1  1  1  0
query at 4      1  1  1  1  1
```

Formally, new chunk row i may attend key j when `j <= P+i`. A sliding window of
width W additionally requires `P+i-j < W`. Treating chunk row zero as absolute
position zero would incorrectly hide most of the prefix. For a single appended
token with no sliding window, all supplied keys are legal; none lies in its future.

Tests compare full-sequence logits with a five-token prefill, three-token chunk,
and final single token. This exercises both rectangular masks and the one-token
case, on CPU and CUDA, including sliding windows, soft caps, MoE and prompt-set
models. FP32 comparison tolerances are absolute 2e-6 and relative 2e-5.

## What is stored, and what it costs

Persistent cache payload, excluding allocator overhead and temporary tensors:

```text
bytes = 2 * layers * batch * KV_heads * cached_tokens * head_dim * bytes_per_value
      = 2 * 8      * 1     * 2        * 289           * 32       * 4
      = 1,183,744 bytes = 1.12890625 MiB
```

The leading factor two is for keys **and** values. FP32 uses four bytes per value.
This is not total device memory: weights, temporary attention/FFN tensors and
allocator reservations also matter. Larger batches, contexts and concurrent
requests multiply cache storage.

This implementation concatenates cache tensors when appending. It is deliberately
simple to inspect; copying and allocation still cost time. It retains the full
prefix even when attention has a sliding window. A preallocated or paged cache
would be a separate optimization with additional correctness requirements.

Caching reduces repeated computation. Across N generated steps, uncached
attention repeatedly builds square prefix attention matrices, giving a sum of
squared lengths; cached single-token attention instead has one row per step,
giving a sum of lengths. Feed-forward blocks likewise process each new position
once instead of repeatedly processing its prefix. Kernel launch, memory and
Python overhead prevent us from inferring a speedup from those counts alone.

## Request-local state and unchanged training

`PLMDecoder.decode(input_ids, cache=...)` returns `DecodeOutput(logits, cache)`.
The cache records layer K/V pairs, the prefix position and a model ownership
marker. It validates layer counts, lengths, batch/head shapes, device, dtype and
context bounds. It does not add learned parameters or checkpoint entries.

`generate_response(..., use_cache=True)` starts with a fresh cache for every
request. No cache is stored globally or on the model. Tests also check that
appending leaves earlier cache tensors unchanged, a cache from another model is
rejected, repeated requests do not inherit state, and EOS/length limits are real.
Direct API callers must discard state after changing weights or the prefix.

The default remains uncached. Enable the experiment explicitly:

```powershell
uv run --no-sync plm evaluate --checkpoint runs/national_dex_promptset_s1729_v1/checkpoint-final.pt --override +experiment=prompt_set_lab --override eval.use_kv_cache=true --out runs/learning/my-cached-validation.json
```

The evaluation report records the cache mode and the evaluator's new source
identity separately from the original training identity. The checkpoint is not
retrained or relabeled as having been trained with caching.

## Why logits are not the only correctness check

Two execution shapes can accumulate floating-point arithmetic slightly
differently. Close logits usually pick the same argmax, but a near tie can change
the next token. That change can then alter the entire continuation. Therefore,
unit-level numerical comparisons are followed by complete generated-output
comparison on all 222 validation prompts from the fixed seed-1729 checkpoint.

Latency is measured only after that output gate. The local microbenchmark chooses
four target-length ranks from each dimension, warms both execution paths, and
runs three repetitions in alternating paired order. GPU synchronization brackets
the timer. It includes the generator's Python/tokenization-mask/tensor work, but
excludes model loading, HTTP, hydration and concurrent request scheduling. It
does not establish energy savings or an advantage over the deterministic oracle.

## Following the evidence when a speedup is small

A separate instrumented run of the cached TOXAPEX/COLOR query generated 171
tokens and recorded 715,638 calls to `Vocabulary.class_of` and 21,033 PyTorch
module-dispatch calls. The [profile summary](../experiments/2026-09-24-kv-overhead.json)
retains the raw-report and diagnostic-script hashes. The underlying profiler
output and script remain with the local experiment artifacts.

Inspection explains a concrete source of avoidable work: `allowed_token_ids`
enumerates dimension and mode token classes on every call, even while the
generator is already in the answer portion. It also validates the growing
prefix. Precomputing vocabulary classes or maintaining validated incremental
grammar state is a useful next experiment. It must preserve the public parser's
handling of malformed input.

The profile recorded roughly 0.191 s inside `allowed_token_ids` and 0.239 s
inside its caller `constrain_logits`, within a 1.441 s instrumented request.
Those cumulative times overlap and must not be added. Profiling adds overhead;
these times are not substituted for the uninstrumented benchmark. They are
wall-clock attribution through Python calls, not GPU utilization or isolated
kernel execution time. Module dispatch, small operations and synchronization
remain plausible contributors to the limited gain, not separately proven causes.

The cache stays opt-in, with an uncached reference for future comparisons. A
larger batch, longer context, different precision, compiled execution or fused
kernels would be a different workload and needs new evidence. This experiment
illustrates why "less arithmetic" and "much faster application" are distinct
claims.

Self-checks: Why don't we cache old queries? Why must new RoPE positions start
at P rather than zero? Why can a one-token causal mask be wrong when many keys
are cached? What could dominate latency after repeated attention work is removed?
