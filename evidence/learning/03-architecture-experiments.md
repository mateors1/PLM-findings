# Lesson 3: more machinery, measured tradeoffs

**2026-09-24.** We added activation checkpointing and sparse mixture-of-experts
(MoE), and corrected the existing positional rotations. The dense model remains
the default. This lesson records what the additions do and what we measured;
neither addition has established a National Dex quality improvement.

## 1. Activation checkpointing: remember less, recompute more

During the forward pass, a block computes `H_next = block(H)`. Backpropagation
normally keeps many intermediate **activations** to compute derivatives later.
Checkpointing keeps block inputs and recomputes intermediate values during the
backward pass. This exchanges GPU computation for lower peak memory.

This is different from a **disk checkpoint**, which saves a training run so we
can resume it after stopping. The shared word describes two different mechanisms.

For batch size B, sequence length T and hidden width D:

```text
block input and output: [B, T, D]
our profile:           [8, 289, 256]
```

Memory also holds parameters, gradients, optimizer states, logits and temporary
buffers; those do not all disappear. Therefore saving 63% here does not predict
63% savings for every model or batch.

We use PyTorch's non-reentrant checkpoint implementation, with RNG preservation.
**Dropout** randomly removes activations during training. Recomputing with a
different dropout mask would calculate gradients through a different function.
Our tests compare logits, loss and gradients with dropout enabled, on CPU and
CUDA, for dense and MoE models. They also verify that the block runs again during
backpropagation. Evaluation skips activation checkpointing.

Try it with a new run name:

```powershell
uv run --no-sync plm train --override +experiment=checkpointing_lab --override run_name=checkpointing_lab_01
```

The expected benefit is fitting a larger workload in memory. At a fixed workload
that already fits, extra recomputation usually has no reason to improve learning.
See [PyTorch 2.11 checkpoint documentation](https://docs.pytorch.org/docs/2.11/checkpoint.html).

## 2. MoE: choose which feed-forward networks process each token

The dense block uses one SwiGLU feed-forward network:

```text
FFN(x) = W_down( SiLU(W_gate x) * (W_up x) )
```

Here `*` is elementwise multiplication. Our width is D=256 and rounded inner
width F=768. Each expert has three projection matrices, totaling `3*D*F`
parameters, or 589,824. In MoE we keep E such networks and choose k for each token.
An expert is a learned network, not a predefined category. TYPE/COLOR expert
specialization is a hypothesis we can inspect, not a property we programmed.

Ignore padding and flatten the remaining hidden vectors:

```text
hidden:             [B, T, D]
active vectors X:   [N, D]       N <= B*T, excluding padding
router weight:     [E, D]       PyTorch Linear storage convention
router scores:     [N, E]       X @ W_router.T
router probability:[N, E]       softmax across experts, computed in FP32
selected experts:  [N, k]       top-k integer indices
selected weights:  [N, k]
combined output:   [N, D]       then restored to [B, T, D]
```

For token x, let p_i(x) be the router probability for expert i and S(x) the
selected set. Our implementation computes:

```text
p(x) = softmax(W_router x)
y(x) = sum(p_i(x) * Expert_i(x), i in S(x))
```

Optional shared experts process every non-padding token and add their outputs.
The measured variants use no shared experts. Routing happens separately in each
of eight decoder blocks. There is no token-dropping capacity limit, distributed
expert placement or fused dispatch kernel in this educational implementation.

### Why we do not renormalize the selected top-1 weight

If we divided each selected weight by the sum of selected weights, top-1 would
become `p_selected / p_selected = 1`. The selected index is discrete, so the task
loss would lose its ordinary gradient path to the router through that weight.
We retain the probability from the full softmax. The top-k selection itself is
still discrete; we do not claim to differentiate its integer indices.

A test backpropagates task loss without the balancing penalty and verifies that
the top-1 router receives a gradient. It also checks that inactive experts receive
no task gradients and that padding cannot affect routing statistics.

### Preventing everyone from choosing the same expert

An **auxiliary loss** adds a second training objective. If most tokens choose one
expert, unused experts learn little and the conditional capacity is wasted.
Our Switch-style balancing penalty uses:

```text
f_i = selected routes to expert i / (N*k)  # observed route fraction
P_i = mean_x p_i(x)                       # average soft probability
L_balance = E * sum(f_i * P_i, over experts)
L_total = L_token + alpha * mean_layers(L_balance) + optional_z_loss
```

The discrete `f_i` values are treated as constants for gradients. At uniform
usage/probability the penalty equals 1, not 0. With alpha=0.01 it contributes
0.01 in that case. We report it separately from token cross-entropy.
Balanced traffic does not prove useful specialization or good answers.
The design is informed by [Switch Transformers](https://arxiv.org/abs/2101.03961);
this small implementation is not a reproduction of its scale or performance.

Try top-1 routing; a repeatable `--override` changes one configuration field:

```powershell
uv run --no-sync plm train --override +experiment=moe_lab --override run_name=moe_top1_01
uv run --no-sync plm train --override +experiment=moe_lab --override model.moe.experts_per_token=2 --override run_name=moe_top2_01
```

## 3. Our first GPU measurements

RTX 5070 Ti; Torch 2.11.0+cu128; BF16 autocast with FP32 weights; fused AdamW.
Each variant used the same eight longest training records, padded to `[8,289]`,
with two warmup updates and five measured updates. The timer covers forward,
backward and optimizer update with CUDA synchronization. It excludes data loading.

| Variant | Total parameters | Median step | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| Dense | 6,558,720 | 60.44 ms | 498.56 MiB |
| Activation checkpointing | 6,558,720 | 98.03 ms | 184.26 MiB |
| Four experts, top-1 | 20,722,688 | 127.08 ms | 737.75 MiB |
| Four experts, top-2 | 20,722,688 | 127.99 ms | 894.88 MiB |

Checkpointing reduced peak allocated memory by about 63% while increasing step
time by about 62%. MoE has about 3.16 times as many total parameters and took about
2.1 times as long per step. Sparse activation does not remove inactive experts'
parameter storage. Small expert batches and dispatch overhead are plausible
contributors to the measured slowdown; this timing alone does not isolate them.

Top-1 and top-2 had nearly the same timing in this short run, with different
memory usage. Five samples in one fixed-order run are insufficient to conclude
that top-2 computation is free. Hardware state, sample count and execution layout
matter. We need repeated, order-balanced profiling to compare close timings.

This is a **microbenchmark**: a narrow workload chosen to understand a mechanism.
It is not a training-quality comparison, production throughput measurement,
maximum-memory estimate across all batches, or energy measurement. Allocated
memory is PyTorch tensor allocation, not total board memory shown by a GPU monitor.

The [raw measurement report](../experiments/2026-09-24-feature-profile.json)
contains individual times, full configs, router loads, source hashes and dataset
identity. Its matching source archive is retained locally beside the original
report under `runs/learning/20260924-feature-profile.source.zip`.

```powershell
uv run --no-sync python scripts/profile_features.py --out runs/learning/feature-profile-02.json
```

## 4. The RoPE repair: test the mathematical property

RoPE rotates coordinate pairs of queries and keys. For one pair (a,b):

```text
R(theta) [a,b] = [a*cos(theta) - b*sin(theta),
                  a*sin(theta) + b*cos(theta)]
||R(theta)x|| = ||x||
dot(R(m*w)q, R(n*w)k) = dot(q, R((n-m)*w)k)
```

The last identity makes the score depend on relative position n-m. Our code
pairs first-half and second-half coordinates. Its old frequency expansion
repeated adjacent angles, which mismatched those pairs. We now concatenate the
frequency vector with itself so both coordinates in each pair share one angle.
Tests check length preservation and simultaneous position-shift invariance.
This fixes a mathematical requirement of [RoPE](https://arxiv.org/abs/2104.09864),
not a hyperparameter to select by whichever validation score happens to win.

The corrected implementation changes model behavior; older results cannot be
silently pooled with new runs. Keep source identities with the measurements.

## 5. What we should experiment with next

First establish dense and MoE validation behavior on the same frozen query
split. Compare token loss, full-answer accuracy, termination, router usage and
training cost. Equal-step comparisons spend different compute; equal-time and
similar-parameter-budget comparisons answer additional questions.

RoPE scaling remains unsupported. It is a useful next lesson about position
interpolation, but the real records currently fit within 289 tokens, below the
512-token limit. We need an explicit longer-sequence test to learn anything
about context extension. muP, a KV cache and optimized serving also remain future
work; the `overkill` config is still a proposal, not a fully supported recipe.

Self-checks: Why can top-1 MoE still cost more memory than dense? Why should
checkpointing preserve gradients despite rerunning a block? Why can perfectly
balanced experts still produce wrong relationships?
