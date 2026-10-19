# 37. Does the membership head support the products generation misses?

**Status: independent audit passed; diagnostic evidence accepted.**
The [plan](../experiments/2026-09-25-missing-source-membership-plan.md) examines
saved Pokémon validation outputs. It changes no weights or decoding rule.

## Two learned components can disagree

The decoder generates a sequence of product IDs. A separate prompt membership
head assigns each product a relation score. These components share model
parameters, but their outputs answer different questions: what should come next
in the sequence, and how strongly does this product fit the requested relation?

Our previous experiment found 30 dual-TYPE query-seed observations where at least
one correct product is absent from every generated path. Set operations cannot
recover an entity that none of their input sets contains. Before changing the
generator, we want to know how the membership head scores those absent products.

Let T be the correct set and U the union of all eight generated source sets:

\[
M=T\setminus U,\qquad C=T\cap U.
\]

M contains omitted correct products, while C contains correct products covered
somewhere in the source pool. We select the 30 cases using only nonempty M,
before examining head values. We do not pick cases with particularly favorable
scores or confuse errors in the final selected set with omissions from all sources.

## What a positive logit means

The head emits a logit z_i for product i. Its sigmoid is

\[
\sigma(z_i)=\frac{1}{1+e^{-z_i}}.
\]

A positive logit maps above one half; zero maps to one half; a negative logit
maps below one half. This is a statement about the transformation, not proof
that these numbers are calibrated probabilities or that the model knows the
correct relation. Earlier experiments already showed that simply taking every
positive-scoring product can introduce many extra members.

If an omitted true product has a strong positive head score, the head supplies
some support that generation has not converted into a candidate member. If its
score is nonpositive or many false products rank above it, the membership head
also gives weak or misleading support. These observations locate disagreement;
they do not prove why the generator missed the product.

In particular, the first-token guidance score is

\[
g_i=\ell_i+16\log\sigma(z_i),
\]

where \(\ell_i\) is a decoder logit. Ranking z alone does not reconstruct g.
We do not have the decoder logits needed for that reconstruction in this
diagnostic, and later positions follow the existing greedy decoder. Do not read
"head rank" as "the rank generation should have selected."

## Ranks, ties and an oracle-only comparison

For a prompt, the head has 1025 product columns. Removing the subject leaves
1024 possible answers. We rank them by decreasing z, breaking exact ties by
smaller product ID. Ranks start at one:

\[
r_i=1+\#\{j:z_j>z_i\}+\#\{j:z_j=z_i,\ j<i\}.
\]

For every omitted true product we also count false products with strictly higher
scores and false products tied with it. This separates a low rank caused by other
correct products from one caused by confusing incorrect products with correct ones.

We record whether each omitted product falls within the first K ranks for
K=|T|. K is supplied by the teacher, so this is an oracle-cardinality diagnostic.
It is not a proposed serving rule: the real request does not reveal the correct
answer size to the model, and the protocol contains no numeric count token.

Conceptually, head values have shape [B,1025]. Truth and source-union membership
are Boolean masks of the same shape; the omitted mask is truth AND NOT union.
The subject column is excluded. The implementation uses saved ID sets and Python
arithmetic, without allocating new model tensors or running the network.

## Two percentages with different denominators

Let m_q be the number of omitted true members in focused query q, and p_q the
number of those with positive scores. We report both

\[
P_{\mathrm{members}}=\frac{\sum_q p_q}{\sum_q m_q},\qquad
P_{\mathrm{queries}}=\frac{1}{30}\sum_q\frac{p_q}{m_q}.
\]

The first weights queries with many omissions more heavily. The second gives
each focused query equal weight. They need not agree. Products and queries recur
across seeds, so neither denominator creates independent statistical samples.

We preserve exact zeros and null summaries for empty comparison groups. No
covered-positive per-query fraction is defined when C is empty; exclude that
query from that comparison mean and report its nonempty-C denominator. The
missing-member fraction remains defined for all 30 focused observations. No
threshold fitting, statistical significance claim, new policy or quality gain
is part of this diagnosis. Its outcome will guide a separately declared next
experiment while preserving the protected final test.

## Measured result: strong head support for most omitted members

The primary run reconstructs all 666 query-seed observations. In 636, the source
union contains every true member. The remaining 30 observations represent
**22 distinct queries**, all dual-TYPE. Across all 666 observations, 83,922 of
85,329 true-member occurrences appear somewhere in the source pool; 1,407 do
not. These are occurrences, so the same product can contribute more than once.

Within that preselected missing-member subset, the exact counts are:

| Seed | Focused observations | Omitted members | Positive / zero / negative | Omitted members in oracle top K | Queries with all / some / no omitted scores positive |
| --- | ---: | ---: | --- | ---: | --- |
| 1729 | 14 | 776 | 770 / 0 / 6 | 757 | 11 / 3 / 0 |
| 1730 | 6 | 183 | 180 / 0 / 3 | 174 | 4 / 2 / 0 |
| 1731 | 10 | 448 | 444 / 0 / 4 | 436 | 7 / 3 / 0 |
| Pooled | 30 | 1,407 | 1,394 / 0 / 13 | 1,367 | 22 / 8 / 0 |

Thus 1,394/1,407 omitted occurrences have positive scores, approximately 99.08%,
and 1,367/1,407 fall inside the oracle-cardinality top K, approximately 97.16%.
The per-query positive fraction also stays high, but it is a different average:

| Seed | Omitted positive fraction, member-weighted | Mean per-query omitted positive fraction | Covered positive fraction, member-weighted | Mean per-query covered positive fraction |
| --- | --- | --- | --- | --- |
| 1729 | 770/776 | 0.9917032967032967 | 1,711/1,731 | 0.9860660777417959 |
| 1730 | 180/183 | 0.99063891070961 | 865/878 | 0.9829551298224946 |
| 1731 | 444/448 | 0.9896389524535978 | 1,221/1,235 | 0.9894380660714173 |
| Pooled | 1,394/1,407 | 0.9908023047546597 | 3,797/3,844 | 0.9865678842678095 |

The covered comparison here uses only the same 30 focused observations, not all
666. Their 3,844 covered true-member occurrences contain 3,797 positive scores,
47 negative scores and no zeros. None of these 30 observations has an empty C:
the mean-per-query denominators are therefore 14, 6 and 10, or 30 pooled, for
both omitted and covered groups. The 22 “all-positive” observations in the first
table are not the same counting concept as the 22 distinct query identities.

## One concrete disagreement

For `PKM_AEGISLASH TYPE SAME` under seed 1729, the correct set has 125 members.
Only one true member is absent from all eight source paths: product ID **1040**.
Its saved head logit is **8.01483154296875**, and its subject-excluded head rank
is **18**. No false product scores higher or ties with it, and 18 is inside the
oracle top 125.

This is a useful distinction: the head gives that product a strong score, yet
none of the eight generated sequences includes it. The recorded rank still
does not tell us its decoder-plus-guidance rank, because the decoder logit
\(\ell_i\) in the guidance equation is absent from this diagnosis. We cannot
conclude that choosing the head's top products would repair the full answer,
or explain why this product disappeared from generation.

The audited evidence supports investigating how candidate generation uses
membership information. It also preserves counterexamples: 13 omitted
occurrences have negative scores, and 40 lie outside the oracle top K. No new
decoder, prediction policy, weights or protected-test results were produced.
Evidence acceptance is separate from prediction quality; the earlier
intersection policy's failed quality gate remains failed.

These results come from `runs/learning/missing-source-membership-v1/summary.json`,
SHA256 `47bf9d54967009a2be6e46bedc28f7949231be9bc14e18798fd66ad776c59805`,
and its three hash-bound seed reports. The diagnostic uses saved FP32 vectors
and Python arithmetic; it does not rerun the network or independently recreate
the upstream training/runtime provenance.

The [portable report](../experiments/2026-09-25-missing-source-membership.json)
binds the independent audit and separate owner decision. The primary runner's
26 focused tests and the auditor's 37 tests passed before their actual runs.
No quality gate applies to this diagnosis because it generates no new predictions.
