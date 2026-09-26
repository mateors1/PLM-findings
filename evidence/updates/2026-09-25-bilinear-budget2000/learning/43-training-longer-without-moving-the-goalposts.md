# 43. Training longer without moving the goalposts

**Status: completed; independently audited single-seed screen passed.** The
[frozen plan](../experiments/2026-09-25-bilinear-budget2000-plan.md)
tests 2000 updates of the same bilinear head from Lesson 42. The completed
500-update experiment stays frozen, including its failed quality gate.

## Why more updates is a separate question

Imagine adjusting a camera's focus. Changing the lens and spending longer
adjusting the same lens are different interventions. Lesson 42 changed what
the membership head could express. This experiment gives that same head more
optimization steps.

The saved training loss was still falling at update 500:

| Point | Balanced training BCE |
| --- | ---: |
| Before update 400 | 0.0003255420 |
| Before update 450 | 0.0002822331 |
| Before update 500 | 0.0002477969 |
| After update 500 | 0.0002471808 |

That makes a longer fit worth testing. It does not show that the validation
answers will improve. The optimizer sees the training loss, while the research
gate asks whether entire unseen validation answers are correct.

## What an update changes

The trainable parameter remains A[2,256,256]. The leading axis selects TYPE or
COLOR. Frozen product embeddings are E[1025,256]. For each query q, subject s,
dimension d and candidate i, the score is

\[
M_d=\frac{A_d+A_d^T}{2},\qquad
z_{qi}=z^{\mathrm{parent}}_{qi}+\frac{E_s^T M_d E_i}{16}.
\]

One full batch contains all 1637 training queries, producing logits
Z[1637,1025]. The loss is a scalar; differentiation produces a gradient tensor
G[2,256,256]. Each update changes A using that gradient and Adam's accumulated
statistics. None of the original 93 state tensors changes.

Ignoring weight decay, which is zero in this experiment, Adam maintains
elementwise first and second moments:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

\[
\widehat m_t=\frac{m_t}{1-\beta_1^t},\quad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\quad
A_t=A_{t-1}-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\]

Here eta=0.0003, beta1=0.9, beta2=0.999 and epsilon=1e-8. All tensor arithmetic
is FP32. More updates means more successive parameter adjustments; it does
not increase the number of trainable parameters or add training examples.

## Restarting and resuming answer different implementation questions

We restart from the original parent with A exactly zero and fresh optimizer
state. The 2000-update checkpoint will be a sibling of the 500-update checkpoint.
It will not be a child resumed from that checkpoint.

A true resume would restore A, both Adam moments, the update counter and other
execution state. Restoring only A and resetting Adam is a different trajectory.
Restarting gives us one declared uninterrupted 2000-update recipe, with the
same initial scores and loss verified before fitting. We do not promise bitwise
identity of the first 500 steps across separate GPU executions.

The original parent's 2000 pretraining steps and this head's 2000 updates are
separate facts. They train different parameters under different recipes; calling
the result simply a "4000-step model" would hide that distinction.

## Why the evaluation happens only at the end

The new budget is fixed before running. We do not inspect validation at update
500, 1000 and 1500 and choose whichever looks best. That would turn the experiment
into a checkpoint-selection procedure and would need its own declared protocol.

Initial validation replay only checks that the new orchestration reproduces the
already known parent. Final validation measures the 2000-update child. Protected
test predictions remain untouched. The historical 500-update results are reused
for a paired comparison: which exact answers were gained and which were lost?

The quality gate stays >201 exact sets, the existing F1 and TYPE/COLOR floors,
valid output sizes and all execution invariants. Lower training loss cannot
override a failed gate. A pass would justify a separately declared replication,
not immediate deployment or a claim of oracle parity.

## What this can teach us

Better validation answers would support this longer recipe on this development
screen. Worse answers could show that lower training loss does not track the
metric we care about. Either result alone would not prove convergence, global
optimality or generalization to new datasets.

We have already used this validation split to guide multiple experiments. That
is adaptive development. Independent seeds and a protected final evaluation
remain necessary before broader claims.

## What actually happened

The one declared run completed all 2000 updates. An independent saved-evidence
audit passed, followed by a separate owner decision accepting the evidence and
the fixed quality screen. The [portable result](../experiments/2026-09-25-bilinear-budget2000.json)
preserves the measurements and unchanged gate.

| Predictor | Exact sets / 222 | Macro F1 | False positives / negatives |
| --- | ---: | ---: | ---: |
| Original dense parent, replayed | 112 | 0.990374 | 512 / 76 |
| Historical 500-update residual | 198 | 0.999406 | 25 / 11 |
| New 2000-update residual | 207 | 0.999746 | 8 / 10 |

The stronger historical eight-branch selector has 201 exact answers and F1
0.9799255. The new result exceeds that exact-count requirement and meets every
group floor: COLOR 103/103, single TYPE 50/51 and dual TYPE 54/68. All 222
outputs meet the declared serialization bounds.

Against the 500-update residual, the longer fit gains 10 exact answers and loses
one: net +9. Against the stronger selector it gains 13 and loses seven: net +6.
Against the dense parent it gains 95 without losing an exact answer. A higher
total therefore does not mean every query improved.

Training loss falls from 0.0038227672 to 0.00003491543, a 99.09% decrease from
the shared zero-residual start. This time the longer budget improves both loss
and validation exactness. It does not establish that further training would help.

Fifteen validation answers are still wrong. Descriptive posthoc counts find
12 with one wrong member and three with two. Ten incorrect sets nevertheless
have strictly separated true and false scores; the fixed zero threshold still
does not recover them exactly. No thresholds or outputs were adjusted after
seeing this diagnostic. Passing the screen is not oracle parity.

The child is registered as final checkpoint 31. All 93 original tensors remain
unchanged, and only A differs from zero. The final checkpoint reload reproduces
all 94 state tensors and the optimizer state exactly. Parent pretraining and
residual fitting retain their separate 2000-step identities. Local weight hashes
identify the files; they do not establish a durable remote backup.

Verification comprised 91 runner CPU tests, 37 independent auditor synthetic
tests, successful primary execution and a successful actual audit. The audit
reconstructs metrics and inspects saved checkpoint evidence; it does not rerun
CUDA training. Fitting took about 4.88 seconds and the primary run about 12.47
seconds locally. These are descriptive timings, not a controlled speedup claim.

This passing screen permits a separately declared replication campaign. It
does not promote the default predictor, integrate ordinary serving, use the
protected test or establish the concurrent-user and energy goals. The frozen
result and teaching material go to the existing Luna Max publication task.
