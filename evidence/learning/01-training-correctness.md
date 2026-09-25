# Lesson 1: what is the model actually learning?

This project is both a research experiment and a practical ML learning project.
We will explain changes through intuition, equations, tensor shapes, and evidence.
This lesson accompanies the training-correctness repairs of 2026-09-24.

## 1. Turn a sequence into prediction problems

Our example sequence is:

```text
BOS PKM_A TYPE SAME ANSWER PKM_B EOS
```

A **token** is one symbol represented by an integer ID. The model gets a prefix
and predicts the next token. It should predict PKM_B after ANSWER, and EOS after
PKM_B. EOS means "the answer is finished" and must also be learned.

The transformer returns **logits**, raw scores before conversion to probabilities.
Their shape is `[B, T, V]`: batch size, sequence length, vocabulary size. Input
IDs and labels both have shape `[B, T]`. Labels are the correct answers used
to score predictions; they are not extra inputs to the forward computation.

For a logit vector z, softmax gives `p_i = exp(z_i) / sum_j exp(z_j)`.
The cross-entropy loss for the correct token y is `-log(p_y)`. Assigning the
correct token probability 0.9 costs about 0.105; probability 0.1 costs about
2.303. Lower is better, provided we score the intended prediction task.

The objective is the mean of `-log P(x[t+1] | x[0], ..., x[t])` over supervised
positions. In PyTorch, the alignment is:

```python
prediction_logits = logits[:, :-1, :]  # [B, T-1, V]
next_labels = labels[:, 1:]            # [B, T-1]
```

The corpus stores unshifted labels; the decoder performs this shift exactly
once. Inference still receives all `[B, T, V]` logits and reads the last position.

| Position seen | Desired next token | Scored? |
| --- | --- | --- |
| BOS | PKM_A | No |
| PKM_A | TYPE | No |
| TYPE | SAME | No |
| SAME | ANSWER | No |
| ANSWER | PKM_B | Yes |
| PKM_B | EOS | Yes |
| EOS | No further target | No |

The **loss mask** sets prompt labels to `-100`, PyTorch's ignored-label value.
It does not hide the prompt from the model. The **causal attention mask** has a
different job: prevent each position from seeing future tokens. **Padding** makes
examples equally long within a batch; padded targets do not contribute to loss.

Before the fix, the loss scored logits and labels at the same position. The
model could learn to reconstruct the token already visible there. We now test
the actual ANSWER-to-target and target-to-EOS positions, and verify that changing
future tokens cannot change the first-answer prediction.

## 2. Loss reduction is not sufficient evidence

During **teacher forcing**, the correct previous answer tokens are present in
the input. Causal attention still prevents a prediction from seeing its target.
During **autoregressive generation**, the model receives only the prompt, emits
a token, appends that token to its input, and repeats. Its own mistakes can
therefore affect later predictions.

An **overfit test** deliberately asks a tiny model to memorize a tiny example.
This checks training mechanics, not generalization. Our repaired test requires
both low loss and the correct generated completion from the prompt alone.
**Generalization** means success on examples excluded from training; that needs
the separate validation/test campaign.

## 3. Resume means continuing the same computation

A **checkpoint** saves weights and training state. The **optimizer** updates
weights using loss gradients; AdamW also maintains running statistics. The
**scheduler** changes the learning rate over time. An **RNG state** records the
position in a pseudorandom sequence; a seed only specifies its starting point.

A **batch** groups examples for a forward/backward computation. With **gradient
accumulation**, A batches contribute gradients before one optimizer update.
An **epoch** is one pass through the dataset.

For fixed order, K batches per epoch, and s completed optimizer steps:

```python
next_batch_position = (s * gradient_accumulation_steps) % K
```

The repaired trainer resumes at this position instead of repeating the first
batch. Loader iterator creation also consumes randomness, even without worker
processes. Dedicated loader generators prevent that bookkeeping from changing
the model's **dropout** sequence. Dropout randomly disables some activations
during training; it is disabled during evaluation.

The resume tests compare final weights, losses, optimizer state, scheduler state,
and model RNG state. They include distinct records, incomplete final batches,
accumulation, epoch boundaries, dropout, CPU, CUDA when available, and a Windows
worker process. This contract assumes the current deterministic dataset and
`shuffle=False`; shuffled sampling or random data augmentation would require
additional checkpointed state.

New trainer checkpoints record the objective and batch-order contract. Legacy
checkpoints without it are rejected for resume: successful deserialization does
not establish that they were trained with the correct objective.

## Evidence and next lesson

The new objective tests failed before the fix. The new multi-batch resume test
also failed before its fix. The regression tests now exercise the behaviors
that the earlier suite missed. See `tests/unit/model/test_reference_model.py`
and `tests/unit/training/test_training_runtime.py`.

Next: dataset provenance and experimental identity. A hash is a fingerprint of
content; it cannot by itself tell us whether the content is scientifically valid.
We must preserve the old dataset, build a traceable new version, and establish
comparable baselines before interpreting a model score.
