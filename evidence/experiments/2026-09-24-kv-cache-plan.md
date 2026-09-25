# Native KV cache: pre-run plan

Implement an opt-in, inference-only, request-local cache without new learned
parameters. Keep the existing uncached path as the default reference. Cache
rotated keys and projected values at KV-head count, before GQA repetition.
Support unpadded batches and chunks, absolute positions, sliding-window masks,
soft caps, dense/MoE and the existing token head of prompt-set models. Reject
training/autograd use, cross-model caches, malformed shapes and context overflow.
Do not persist cache state in checkpoints or share it between requests.

Correctness gates: cached chunk/token logits versus full-prefix logits on CPU
and CUDA with declared tolerances; mathematical attention/position masking;
request isolation; real generation/EOS/bounds and unchanged training tests.
Use the retained prompt-set seed-1729 final checkpoint as a fixed anchor, not a
new model selection. Compare all 222 cached validation outputs with its archived
uncached outputs and preserve any disagreements. No final-test use or retraining.

After correctness, measure uncached/cached end-to-end generation on eight fixed
validation prompts spanning target lengths, one request at a time. Warm both
paths and use three paired repetitions with alternating order and CUDA
synchronization. Retain individual timings, output equality and peak allocated
memory. This is a local generation microbenchmark, not HTTP/concurrency/energy
evidence, and no oracle throughput advantage can be inferred from it.
