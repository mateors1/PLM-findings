# 34. Did the answer exist among the candidates?

The weaker-margin screen gained three exact selected answers but failed its
ranking-separation requirement. Before changing another training setting, we
can ask where its saved candidate-and-selector pipeline succeeded or failed.
This [declared diagnostic](../experiments/2026-09-25-candidate-score-decomposition-plan.md)
uses saved outputs only. Both runs selected every exact answer available among
their ten candidates. Their remaining exact-answer failures are therefore
candidate-availability failures. This does not reopen the rejected screen.

## A selector cannot choose an answer it was never offered

Our inference procedure produces four original answer sets and six pair unions.
For one query, call this ordered list P. It contains ten slots, even when two
slots contain identical sets. Keeping the order matters because the first slot
wins a score tie.

Let T be the teacher's complete answer set. Two indicators separate availability
from selection:

\[
A=\mathbf{1}_{\{\exists S\in P_{eligible}:S=T\}},
\qquad E=\mathbf{1}_{\{selected\ eligible\ \land\ S_{selected}=T\}}.
\]

An all-ineligible fallback receives no exact-answer credit, even if its raw
membership happens to equal T. Every query then falls into one of three states:

| State | A | E | Meaning |
| --- | ---: | ---: | --- |
| Selected exact | 1 | 1 | The answer was available and the selector found it. |
| Available but missed | 1 | 0 | The selector preferred a wrong candidate. |
| Unavailable | 0 | 0 | No eligible candidate was exactly correct. |

The combination A=0,E=1 is impossible. The count of available-but-missed queries
is sum(A-E). Availability uses the labels, so it is an oracle diagnostic ceiling,
not a model we can serve. It tells us how much a perfect selector over this
fixed pool could recover; it does not tell us how to build that selector.

Even containing every true member somewhere in the four sources is insufficient.
If truth is {a,b}, sources {a,x} and {b,x} contain all true members collectively,
but their union is {a,b,x}. Union cannot delete x. Conversely, if a true member
is absent from every source, no union of those sources can recover it.

## Four combinations from two saved runs

Each checkpoint supplies two objects: its generated candidate pool and a
prompt-only vector scoring the 1025 entities. We retain both separately and
calculate the following fixed table:

| Candidate pool | Control score vector | Treatment score vector |
| --- | --- | --- |
| Control | Q_cc: original control | Q_ct: offline rescoring |
| Treatment | Q_tc: offline rescoring | Q_tt: original treatment |

Q can mean exact-answer count or macro F1; we report which metric is used.
The diagonal cells must reproduce the saved original choices exactly before
we interpret either off-diagonal cell. No model or generator runs again.

Conceptually, saved scores form z_b[222,1025]. A pool can be represented by
binary membership masks M_a[222,10,1025] and an eligibility mask[222,10]:

\[
s_{ab}[q,k]=\sum_j M_a[q,k,j]z_b[q,j],
\qquad k^*_{ab}(q)=\operatorname{firstargmax}_{k\ eligible}s_{ab}[q,k].
\]

The diagnostic need not allocate that dense mask tensor. Sorted entity-ID lists
encode its nonzero entries; Python's `math.fsum` reproduces the existing set
score. It neither averages scores by set size nor changes score thresholds.
The expected set enters only the subsequent metric calculation.

Raw paths and processed sets also differ. Under the existing SAME rule, the
postprocessor removes the query subject from a terminated, protocol-valid
answer. The raw protocol may legally emit it. We reconstruct that rule explicitly,
preserve the raw path and record subject-removal flags. Uniqueness is tested
before removal; this transformation must not turn an invalid path into valid
evidence. Failed paths retain their original membership under the old contract.

## Why there may be two different explanations of the same total

The observed diagonal change is Q_tt-Q_cc. We can walk through the table in
two orders:

\[
(Q_{tc}-Q_{cc})+(Q_{tt}-Q_{tc})=Q_{tt}-Q_{cc},
\]

\[
(Q_{ct}-Q_{cc})+(Q_{tt}-Q_{ct})=Q_{tt}-Q_{cc}.
\]

The first path changes the pool before changing scores; the second changes
scores first. A score vector may work well with one pool and poorly with the
other. The interaction term

\[
I=Q_{tt}-Q_{tc}-Q_{ct}+Q_{cc}
\]

describes that dependence on which pool is used. Exact-count arithmetic is
integer arithmetic; floating F1 summaries are subject to rounding. We do not
force a unique allocation of the total gain to two supposedly independent modules.

In the real pipeline the head also steers first-choice generation, and the
trained components share parameters. Substituting a saved vector after generation
holds the pool fixed artificially. It is not the same intervention as swapping
heads and regenerating answers. These conditional comparisons can guide the next
question but cannot establish the causal mechanism of the training trajectory.

## What happened on the saved runs

All counts below concern the same 222 validation queries at seed 1729.

| Availability or outcome | Control | Treatment |
| --- | ---: | ---: |
| Exact answer among four original paths | 160 | 166 |
| Exact answer among all ten slots | 191 | 194 |
| Exact answer selected | 191 | 194 |
| Exact answer available but missed | 0 | 0 |
| Exact answer unavailable | 31 | 28 |
| All true entities present somewhere in the sources | 202 | 208 |

Pair unions add exact availability for 31 control queries and 28 treatment
queries. A perfect reranker restricted to these pools cannot improve their
exact counts: it already achieves the available ceiling. This does not mean
the scores are perfect for every objective or every possible candidate pool.

The last row exposes two obstacles. In control, 20 queries lack a true member
from every source; in treatment, 14 do. No union can recover those missing
members. Another 11 control and 14 treatment queries collectively contain all
true members but have no exact candidate in the ten slots. Extra members or
the restricted combinations can prevent exactness; this diagnostic does not
separate those explanations.

| Fixed candidate pool | Control scores: exact / macro F1 | Treatment scores: exact / macro F1 |
| --- | ---: | ---: |
| Control | 191 / 0.967657 | 191 / 0.965878 |
| Treatment | 194 / 0.974821 | 194 / 0.974821 |

For exact counts, both paths through the table assign +3 to changing pools and
0 to changing saved scores; interaction is 0. For F1, changing scores on the
control pool loses 0.001779, while changing scores on the treatment pool changes
the aggregate by 0. The F1 interaction is therefore +0.001779. Equal exact
counts do not imply equal partial-answer quality.

This is more informative than saying the treatment gained three answers:
183 queries remain exact, 20 remain unavailable, 11 become available and exact,
and 8 lose availability and exactness. None moves through an available-but-missed
state. In particular, COLOR stays at 101/103 exact despite two gains and two
losses. Aggregate stability can hide changed errors.

| Group | Queries | Control exact / unavailable | Treatment exact / unavailable |
| --- | ---: | ---: | ---: |
| COLOR | 103 | 101 / 2 | 101 / 2 |
| Single TYPE | 51 | 45 / 6 | 46 / 5 |
| Dual TYPE | 68 | 45 / 23 | 47 / 21 |

The useful next question concerns generating better candidates, especially for
dual-TYPE queries. Reranking the same candidates cannot solve these exactness
failures. That is an observed limit, not evidence that any particular generation
change will work. The head helped generate these pools, so the table also does
not prove the head was irrelevant to the improvement.

## What this adds to the research record

The [portable report](../experiments/2026-09-25-candidate-score-decomposition.json)
binds the primary result, independent audit and separate acceptance decision.
The audit reconstructs the saved arithmetic without importing the primary
diagnosis helpers; exact rational sums independently check the stored scores.
It authenticates neural outputs already measured, without rerunning the network.
This is a post hoc, single-seed development diagnosis, with no new checkpoint,
policy promotion or protected-test use. The weak-margin gate remains failed.

For a self-check: if a reranker has perfect accuracy whenever an exact candidate
exists, can a better reranker improve exact answers? What changes if the metric
is F1 rather than exactness? Use the off-diagonal control-pool cell above.
