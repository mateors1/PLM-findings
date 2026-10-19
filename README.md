# PLM findings

**Can a small model answer a structured relation query with the complete, correct set of product IDs?** PLM (Protocolized Language Model) / Pika Edition studies that question using 1,025 Pokémon identifiers as an opaque product catalog. Given a subject and `TYPE` or `COLOR`, the task is to return every other product sharing at least one value of that attribute.

**Latest published synthesis (2026-10-18):** the separate **~6.82M-parameter frozen-parent bilinear scorer** reaches **660/666 exact development answers (99.10%)** after 32,000 residual updates per parent seed, versus 650/666 at 8,000. These are **the same 222 queries evaluated under three seeds**, not 666 independent queries. It remains an **unpromoted candidate**, with a separately verified opt-in CPU/cached adapter that bypasses the transformer decoder. [Training techniques, ranked and explained](papers/14-training-recipes-and-techniques.md) · [curated evidence and limits](evidence/updates/2026-10-18-training-recipes/README.md).

The **~6.69M parent with autoregressive candidate generation and learned set composition** retains its distinct **569/666 exact answers (85.44%)**, **96.69% macro F1** result under the [accepted application verification](evidence/updates/2026-09-25-composition-verification/audit/acceptance.json). Comparing 660 with 569 changes the predictor as well as training; it is not an isolated training ablation.

**The research has not established oracle parity or a quality-matched serving advantage.** Exact graph lookup already defines the correct answers for this task. Personalized recommendation, cold-start products and edge-device deployment have not been validated here. This is a dated research snapshot, with results and failed experiments to inspect; it is not a runnable model release.

## Start here

1. [The question and protocol](papers/01-research-question-and-protocol.md): what the model receives, what it returns, and what the experiment can establish.
2. [Learning and quality](papers/02-learning-and-quality.md): the measured progression, including interventions that failed.
3. [Architecture and possible applications](papers/13-decoder-and-applications.md): why decoder-only, where it might fit, and which claims remain hypotheses.

For the latest methods and measured outcomes, read [14: training recipes and techniques](papers/14-training-recipes-and-techniques.md). For deeper reading, see the [full 14-paper catalog](papers/README.md) or the [reader guide to model details and archived evidence](docs/reader-guide.md).

## A recorded example

The saved composition diagnosis includes this validation query for seed **1729**:

```text
Subject: PKM_BERGMITE
Dimension: TYPE
Relation: SAME
```

In the [documented grammar](evidence/source/protocol-spec.md), its model prompt is:

```text
BOS PKM_BERGMITE TYPE SAME ANSWER
```

**Recorded selected result:** 47 product IDs, including `PKM_ARCTIBAX`; all 47 expected members recovered, with no extra members. The following is an abbreviated excerpt of the saved report, not the full generated sequence:

```json
{
  "subject": "PKM_BERGMITE",
  "dimension": "TYPE",
  "seed": 1729,
  "selected_size": 47,
  "truth_size": 47,
  "tp": 47,
  "fp": 0,
  "fn": 0,
  "exact": true,
  "negative_true_members": [
    {"id": 1052, "key": "PKM_ARCTIBAX", "logit": -0.5478532910346985}
  ]
}
```

Source: [`negative_member_counterexample` in the saved diagnosis](evidence/experiments/2026-09-24-pair-error-diagnosis.json). The full 47-ID response is not included in that public report. The prompt above renders its recorded subject and dimension using the protocol; it is not a newly run prediction.

This example also exposes a limitation: ARCTIBAX belongs in the exact answer despite its negative auxiliary score. Removing every negative-scored member would damage already-correct answers. [Lesson 27](evidence/learning/27-remaining-errors-and-score-signs.md) explains the diagnosis; the aggregate composition result still contains **97 inexact answers**. Composition can select a generated candidate or a union of two candidates, so its output set need not be a single autoregressive sequence.

## What is available

- **Public here:** 14 synthesis papers; copied research notes, plans, reports and figures; a pinned [dataset manifest](evidence/pokemon_v1_f1541479_20260924.json); selected audit scripts and tests; and dated checksum inventories.
- **Not included here:** model weights/checkpoints, the complete training and serving source tree, raw SQLite databases, protected-test outputs, or the full raw responses behind the composition verification.
- **Reproduction boundary:** the separate source repository referenced by archival links is not publicly accessible. The copied methods may depend on its layout and locally held artifacts. This repository supports reading and auditing the published evidence, but is not a standalone end-to-end replication bundle.

The [reader guide](docs/reader-guide.md#navigating-archived-links) maps the architecture note's available lessons and identifies source-only links. Archived files keep their original text and paths so their checksums remain meaningful.

## Reading the numbers and status

The main quality comparisons reuse **222 validation queries** under each of three training seeds. Thus **666 query-seed observations** represent repeated model evaluations on the same queries, not 666 independent held-out queries. The protected final-test partition has **191 queries** and has not been used to support the findings here. Validation has guided many choices, so its results are developmental evidence. The graph/compiler oracle is correct by construction for the scoped task; a model result below 100% exactness is below oracle parity.

The composition result is **569/666 exact sets** with **96.69% macro F1**, now preserved by the accepted shape-aware application verification. The first application campaign failed exact HTTP score parity because FP32 scores changed with batch shape; that failed gate remains failed. The revised contract uses exact same-shape score references and unchanged cross-shape decisions, with explicit mixed fresh/reused evidence. No matched-quality concurrent throughput or energy advantage is established.

The **~6.82M bilinear derivative** is a separate research variant with a frozen parent and 131,072 added parameters. Its 660/666 validation result and opt-in CPU integration do not replace the composition application's 569/666 result or establish model promotion. The 191 protected queries remain unused, and six seed-query failures across five distinct queries remain. See the [parameter breakdown and model identities](docs/reader-guide.md#model-size-and-retained-techniques).

Earlier archival notes still say composition acceptance was pending. That was their status when copied. The later [completed verification](evidence/updates/2026-09-25-composition-verification/README.md) records acceptance of the revised contract, while preserving the original failure. The [snapshot guide](docs/reader-guide.md#snapshot-history-and-reuse) keeps these stages separate.

## Pokémon names and IP

Pokémon and individual Pokémon names appear here as reference labels for a research corpus. The model treats names as opaque `PKM_<NAME>` identifiers to study relational retrieval and serving, rather than as characters in a game or story. This is independent research, with no affiliation with or endorsement by the owners of Pokémon or related marks. No rights in those marks are claimed.

The intended rationale is limited, referential use of names to identify the benchmark entities and make the experiments reproducible. Under U.S. guidance, individual names are not protected by copyright, although names may be protected as trademarks ([U.S. Copyright Office](https://www.copyright.gov/help/faq/faq-protect.html); [USPTO](https://www.uspto.gov/trademarks/basics/what-trademark)). Copyright fair use and trademark defenses depend on the specific use and jurisdiction; this notice is not a legal determination or permission to reuse other Pokémon material ([U.S. Copyright Office fair-use guidance](https://copyright.gov/fair-use/)).

## Support

To support the maintainer, [sponsor @mateors1](https://github.com/sponsors/mateors1).
The repository's GitHub Sponsor button is configured in [`.github/FUNDING.yml`](.github/FUNDING.yml).

## Additional audited updates (2026-10-18)

The [canonical paper, revision v0011](PAPER.md) and [immutable paper history](paper-history/README.md) preserve the current synthesis and preceding revisions. These updates do not promote a model or open the protected test partition.

- [A stronger matched HTTP baseline](papers/2026-10-18-cached-http-w1.md): serializing SQLite execution removes the earlier measured cached-model serving advantage in the tested setting.
- [Held-out queries and supervised pairs](papers/2026-10-18-bilinear-pair-coverage.md): reverse training queries directly supervise 79.84% of the eligible validation candidate pairs; all eight remaining membership mistakes lack that coverage. This is a descriptive limitation, not a causal explanation or a new trained model.
- [32,000-update screen](papers/bilinear-thirty-two-thousand-update-screen.md) and [remaining error geometry](papers/bilinear-thirty-two-thousand-update-error-geometry.md): accepted development evidence and its limitations.
- [Symmetric affine offsets](papers/31-symmetric-affine-offsets.md): the fixed gate rejects this intervention; execution failures and negative results remain preserved.

Historical experiment papers and their dated evidence are retained alongside the 14 synthesis papers linked above. The source webapp, local uploads, model weights and private records are not included in this public findings update.


[Shared entity offsets](papers/2026-10-18-bilinear-entity-offsets.md) records the accepted matched 660/666 to666/666 development result, with repeated-query and serving limitations.
