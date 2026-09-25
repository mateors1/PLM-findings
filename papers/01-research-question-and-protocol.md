# 1. Research question and protocol

**Snapshot:** 2026-09-24 research, copied 2026-09-25. **Status:** protocol and National Dex snapshot implemented; systems thesis unproven.

## The question

PLM emits identifiers from a constrained relational query: given a product, a dimension and the `SAME` mode, generate every other product that shares at least one value of that dimension. Version one uses `TYPE` and `COLOR` over Pokémon names treated as opaque SKUs. Names are output identifiers, not natural-language content. The surrounding deterministic pipeline parses requests, forms tokens, validates and post-processes IDs, then hydrates labels.

The research program has two distinct experiments. **Experiment A** asks whether a trained model can match a deterministic graph oracle while improving serving properties at a specified quality, concurrency and latency. It cannot beat the oracle on correctness because the oracle defines the exact answer. **Experiment B**, deferred, would test learned policy on relationships that a simple compiler cannot deterministically derive, such as complementary products or compositional held-out relations. Current TYPE/COLOR SAME results are Experiment A development findings; they do not validate Experiment B. This scope and its rationale come from the [source research log](../evidence/source/research_log.md) and the [project overview](../evidence/learning/00-project-so-far.md).

## Why the protocol has this shape

The model-visible grammar is:

```text
BOS SUBJECT DIMENSION SAME ANSWER TARGET... EOS
```

`ANSWER` separates the prompt from the supervised completion. `TYPE` and `COLOR` make the steering token load-bearing: with only one dimension, the model could ignore it. Attribute values and graph predicates remain hidden; the model neither receives nor emits them. `IGNORE`, counts and pagination belong to the parser or post-processor, not the token stream. Only product identifiers are legal targets. The direct vocabulary reserves IDs `0–31` for controls, `32–127` for dimension/mode tokens and `1024+` for append-only product IDs. A frozen vocabulary must grow without renumbering existing product embeddings. See [protocol details in the source learning note](../evidence/learning/00-project-so-far.md) and its [dataset contract](../evidence/learning/02-data-and-evaluation.md).

## The data and experimental unit

The graph is the factual source of truth in SQLite; versioned config defines relation derivation and answer order; content-addressed corpus files are derived teacher output. The pinned National Dex import has **1,025 product nodes**, **28 hidden attribute nodes**, **2,576 edges**, **2,050 subject/dimension records** and **2,049 vocabulary rows**. The longest record is 289 tokens within a 512-token context. The copied [dataset manifest](../evidence/pokemon_v1_f1541479_20260924.json) and [data lesson](../evidence/learning/02-data-and-evaluation.md) provide provenance and snapshot details.

The split unit is a complete `(subject, dimension)` query: **1,637 train**, **222 validation**, **191 protected test**. Entities can occur in other training queries and answer lists, so this is *transductive query generalization*. It does not test cold-start products or unseen token IDs. Three seeded model runs reuse the same 222 validation queries; pooled counts across them are paired observations, not an independent 666-query sample.

## What should be measured

Token loss, first-token ranking, complete generated sequences, processed product sets and serving behavior answer different questions. A response counts as exact only if it has the correct unique members, legal syntax and observed termination; F1 gives partial credit. Post-processing can remove self-returns and duplicates but cannot invent missing correct members. The oracle must remain a reference at every quality comparison. A claimed serving advantage requires quality-matched latency distributions, concurrency/throughput and energy per request against a suitable oracle or lookup implementation. [Evaluation methodology](../evidence/learning/02-data-and-evaluation.md) and [serving scope](04-serving-and-reproducibility.md) set those boundaries.

**Lesson:** precise scope and identity make a negative result informative. The model's quality is measurable against a perfect reference, while any systems benefit remains an empirical question.
