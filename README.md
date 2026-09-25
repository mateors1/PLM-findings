# PLM findings

Research synthesis for **PLM / Pika Edition**, a decoder-only model that emits product identifiers for structured relation queries. Pokémon identifiers are the first opaque product catalog. This repository is a dated research snapshot of the source project's working tree, not a software release.

## Read the findings

1. [Research question and protocol](papers/01-research-question-and-protocol.md) — what the experiment can establish.
2. [Learning and quality results](papers/02-learning-and-quality.md) — the measured progression and failed interventions.
3. [Set prediction and failure analysis](papers/03-set-prediction.md) — why complete answers remain difficult.
4. [Serving and numerical reproducibility](papers/04-serving-and-reproducibility.md) — what application and performance tests established.
5. [Evidence ledger and next experiments](papers/05-evidence-ledger.md) — claim status, source map, and acceptance work still open.
6. [Academic versioning and the next hypothesis](papers/06-academic-versioning-and-next-hypothesis.md) — 23 final-checkpoint identities, remaining archival work, and learned relational retrieval.

The [`evidence`](evidence/) directory preserves the 2026-09-24 learning notes, experiment plans and portable reports, figures, and the National Dex manifest from the source checkout. The evidence files are copied without editorial changes. Research papers cite those local copies; consult them for methods, exact run identities and detailed measurements. The evidence notes' links into `src/` refer to the separate [PLM source repository](https://github.com/mateors1/Plm-protocolized-language-model-pika-edition) and may not resolve here.

## Reading the numbers

The main quality comparisons reuse **222 validation queries** under each of three training seeds. Thus **666 query-seed observations** represent repeated model evaluations on the same queries, not 666 independent held-out queries. The protected final-test partition has **191 queries** and has not been used to support the findings here. Validation has guided many choices, so its results are developmental evidence. The graph/compiler oracle is correct by construction for the scoped task; a model result below 100% exactness is below oracle parity.

The latest offline composition result is **569/666 exact sets** with **96.69% macro F1**. A distinct, newer application replay did not pass its exact HTTP score-parity gate because FP32 scores changed with batch shape; the selected answers in the tested HTTP checkpoint matched. The failed gate remains failed. No matched-quality concurrent throughput or energy advantage is established.

## Snapshot and reuse

The [2026-09-25 update](evidence/updates/2026-09-25/README.md) adds a historical
checkpoint inventory, the revised verification contract, a completed native
concurrency probe and the future learned-retrieval hypothesis. Original evidence
copies remain unchanged; the addition has its own checksum inventory.

Source checkout: `mateors1/Plm-protocolized-language-model-pika-edition`, `HEAD` `8afff361447b1357a041aa65f5b588af5b7ba5b5`; the copied source material also contained uncommitted changes. The copy was made on 2026-09-25 from the 2026-09-24 research state. [`evidence/SHA256SUMS.txt`](evidence/SHA256SUMS.txt) records the copied file hashes, allowing exact verification without pretending that the source `HEAD` alone identifies this snapshot. The papers are interpretive syntheses; the evidence copies remain the detailed record.

This repository is private by default when published because it contains an uncommitted research snapshot. Review its visibility before changing that setting.
