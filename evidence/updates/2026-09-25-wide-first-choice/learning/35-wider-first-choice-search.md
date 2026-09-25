# 35. Give the selector more useful choices

**Status: completed; independent audit and the declared offline quality gate pass.**
The [experiment plan](../experiments/2026-09-25-wide-first-choice-plan.md)
fixes eight first-choice branches on the accepted three checkpoints. It changes
inference, not learned weights. The earlier weak-margin models remain rejected.

## Search can change an answer without changing a model

A model supplies scores; a decoding algorithm decides how to turn those scores
into a sequence. Ordinary greedy decoding chooses the highest-scoring allowed
token at every position. Our existing four-path procedure instead tries the
first, second, third and fourth ranked products at the first position, then
continues greedily from each prefix.

Different starts can lead to different sets. A dual-TYPE subject may have two
overlapping neighborhoods, and one greedy path may favor one neighborhood.
That is an intuition for branching, not proof that a branch corresponds to an
actual type or that additional paths recover the missing neighborhood.

We tried ranks one through eight. This is a fixed first-position branching
rule. It is not beam search: we do not maintain, expand and prune several
prefixes at every later step. It also is not the earlier experiment that gave
the first target a larger training-loss weight. The model parameters stay fixed.

## What gets ranked at the first position?

Let l_i be the decoder logit for entity i and z_i its prompt-only relation logit.
The existing guidance rule is

\[
g_i=l_i+16\log\sigma(z_i).
\]

The sigmoid maps a relation logit to a number between zero and one. Its logarithm
is nonpositive, so a low relation score strongly penalizes a candidate start;
a high one receives little penalty. These are learned scores, not calibrated
proofs of membership. We retain the existing coefficient 16 and change only
how many ranked starts we explore. After the first choice, generation reverts
to the existing greedy rule with protocol and uniqueness masks.

For a batch of B queries, the five-token prompt has shape [B,5]. The decoder's
prefill logits have shape [B,5,2049], and the relation head gives [B,1025].
Product columns occupy vocabulary indices 1024 through 2048. Guidance modifies
those product columns at the prompt's last position. The generator selects one
rank per query, then feeds one new token at a time with a private KV cache.

The experiment keeps B=8, with B=6 in the last group. Changing that shape can
slightly change floating-point head scores, so the historical comparison must
use the same shape. The first four fresh paths and scores must reproduce their
saved references exactly before a seed's new branches are generated.

## Eight paths become thirty-six candidate sets

With K source sets, retaining originals and every pair union gives

\[
N(K)=K+\binom{K}{2}=\frac{K(K+1)}{2}.
\]

Thus N(4)=10 and N(8)=36. Pair unions add no model forward passes. They do add
set-scoring work, and their membership includes each entity only once.

Conceptually, the candidate masks change from [B,10,1025] to [B,36,1025], while
the learned head remains [B,1025]. The score matrix changes from [B,10] to
[B,36]:

\[
s_{bk}=\sum_j M_{bkj}z_{bj}.
\]

The implementation uses sorted ID sets and canonical `math.fsum`, not a newly
trained network. It retains all old ten slots first, then appends the new
originals and pairs. Equal-scoring additions cannot displace the old winner.

## More opportunities can also create new mistakes

Because the old pool is retained, exact-answer availability cannot decrease
if source replay is exact. Selected exactness can decrease. For example,
suppose the old pool contains truth {a,b}, scored 5, and the new pool adds
{a,b,x}, scored 6 because the head incorrectly gives x a positive score.
The extra candidate makes the selected answer worse even though the exact
answer remains available.

That is why we measure two quantities separately:

- Availability: does any eligible candidate equal the complete teacher set?
- Selection: does the learned scoring rule actually choose an exact answer?

An oracle availability gain is an opportunity, not a model-quality gain. The
gate requires actual selected exactness and macro F1 not to regress per seed,
strict pooled exact and dual-TYPE improvements, and no pooled group losing
exact answers. Labels never enter the candidate generator or selector.

## Extra search has a cost

The eight rank batches execute sequentially. We do not need eight simultaneous
KV caches, but we pay for four additional generated paths per query and their
first-step guidance forwards. Longer paths and batch padding affect actual work;
doubling the number of branches does not prove exactly twice the runtime.

For the current eight-layer model, each layer's key and value caches have
conceptual shape [B,2,T,32]: two KV heads, T cached positions and 32 values per
head. At B=8 and T=512, FP32 keys and values across eight layers contain
2 × 8 × 8 × 2 × 512 × 32 × 4 bytes = 16 MiB. This is cache payload arithmetic,
not a prediction of total GPU allocation: weights, activations, temporary
buffers and allocator behavior also matter. Sequential branches reuse the
capacity instead of requiring eight such branch caches at once.

We record generation time, padded decode work, emitted tokens and peak allocated
memory. This is an offline quality/work comparison. Even a quality pass would
still need serving integration and a matched-quality latency/throughput study.

## What is being versioned?

The same three checkpoint hashes identify the learned models. The new plan,
runner, archived runtime and output receipts identify a different inference
experiment. The checkpoint inventory stays at 27. A change in the algorithm
used to query weights is not a new set of trained weights.

The independent auditor reconstructs slots, scores, selections and metrics from
saved evidence. It does not rerun the neural network or independently establish
that every next token was a model argmax. The archived runtime records that
procedure, while exact replay checks its agreement with the accepted baseline.

## What happened?

The [audited portable report](../experiments/2026-09-25-wide-first-choice.json)
records 222 validation queries evaluated under each of three training seeds:
666 query-seed observations, not 666 independent queries.

| Measure | Four branches | Eight branches |
| --- | ---: | ---: |
| Selected exact answers | 569/666 (85.44%) | 603/666 (90.54%) |
| Macro F1 | 96.69% | 98.35% |
| Exact answer available in the pool | 570 | 605 |
| Available exact answer missed by selection | 1 | 2 |
| COLOR exact | 308/309 | 309/309 |
| Single-TYPE exact | 136/153 | 139/153 |
| Dual-TYPE exact | 125/204 | 155/204 |

Each seed improved: 191→201, 192→205 and 186→197 exact answers out of 222.
There were 35 gained answers and one lost answer. The lost answer was Cetitan's
TYPE query under seed 1730. Its old exact candidate remains available; a newly
available wrong answer wins the score comparison. This is the concrete version
of the earlier toy example: increasing the search space can expose a scoring
mistake even while improving average quality.

The result supports a useful inference: these frozen weights can produce more
useful candidates than the four-branch procedure exposes. It does not show that
the weights learned anything new during this experiment. We spent more compute
querying the same learned function.

Across the three seeds, generating ranks 1–4 took 479.66 seconds and the
additional ranks 5–8 took 486.34 seconds. Combined generation time was about
2.01 times the four-branch time. These sequential observations are not a
randomized serving benchmark. They reveal the cost paid for the quality gain,
without establishing concurrent throughput, latency targets or energy savings.

## What still prevents oracle parity?

Of the 63 remaining nonexact observations, 61 have no exact candidate among
the 36 slots and two have an exact candidate that the scorer misses. Among
those 61, 30 lack at least one true member across all eight sources; the other
31 contain the truth collectively but no allowed original/pair equals it.
Containment is weaker than exactness because extra entities can remain in unions.
We have not tested all larger unions of these eight sources.

COLOR is exact for all 103 validation queries under all three seeds. That is
a scoped validation result, not overall oracle parity or a protected-test result.
The 191-query protected test remains untouched. Repeated validation-guided
experimentation also means this result needs a later, frozen final assessment.

The runner passed 29 focused tests; the separate auditor passed 48. The actual
audit reconstructed all 5,328 source paths and 23,976 candidate slots. A separate
decision accepts the evidence and quality gate, while leaving serving defaults
unchanged. The next research work remains within Pokémon: understand the missing
candidates and verify any application integration under its own contract.
