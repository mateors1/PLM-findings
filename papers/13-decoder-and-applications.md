# 13. Why decoder-only, and where might the protocol apply?

**Status:** architectural interpretation and application hypotheses. This paper introduces no new model result.

## What “decoder-only” means here

The model receives a short structured prefix such as `BOS PKM_PIKACHU TYPE SAME ANSWER`. Token embeddings and causal self-attention form its internal representation of that prefix. At inference it predicts one permitted product ID at a time from the prompt and the IDs already emitted, stopping at `EOS` or a serving bound. It has no separate encoder stack. “Decoder-only” describes the computation; it does not imply free-form language output. Only product identifiers are model-visible targets, while attribute values and graph predicates remain hidden ([protocol](../evidence/source/protocol-spec.md), [architecture](../evidence/source/architecture.md)).

A separate encoder is not obviously needed to read this five-token prompt. That is a reasonable simplicity argument for the current reference design, **not a proof of architectural optimality**. The requested answer is a set, but an autoregressive decoder must serialize its members into an order. The source project therefore measures complete-set accuracy and has tested candidate branching, learned set selection and pair unions. The fixed offline composition policy reaches 569/666 exact validation observations; its revised application integration passed separately, while oracle parity remains open ([set study](03-set-prediction.md), [integration study](04-serving-and-reproducibility.md)).

An encoder with a set-prediction head, a smaller relation scorer or a graph lookup may be better for some tasks. The direct `z > 0` membership ablation fell from 569 to 339 exact observations despite higher macro F1 ([Lesson 28](../evidence/learning/28-direct-membership-ablation.md)). That rejects **that fixed rule on those saved scores**. It does not rule out other set objectives, calibrations or architectures. Recent saved-pool analysis also found that two tested selectors chose every available exact answer; missing candidates, rather than a selection miss, explained their exact-count difference ([paper 11](11-candidate-availability-and-selection.md)).

## Application map

| Candidate use | Fit to the protocol | Evidence boundary |
| --- | --- | --- |
| Exact same-attribute catalog query | Query-to-ID grammar and deterministic post-processing fit directly. | The graph oracle already answers perfectly. PLM must match quality and prove a serving benefit before it can replace lookup. |
| Candidate discovery for human review | A constrained model can propose ID sets while an operator reviews them. | Current results are on a computable relation; the model's missed members and false additions limit exhaustive use. No user study or workflow advantage has been measured. |
| Learned complementary or preference relation | A relation that cannot be completely compiled from existing graph rules is the stronger hypothesis for model value. | Independent labels, meaningful contexts, explicit unknowns, leakage-resistant splits and simple ranking baselines are prerequisites. No Experiment B result exists ([paper 6](06-academic-versioning-and-next-hypothesis.md)). |
| High-volume ID-only serving | Batching can amortize model work for many requests. | The current native scheduler serialized callers; an offline fixed-group sweep is not an HTTP/SLO or energy result ([paper 4](04-serving-and-reproducibility.md)). |

The immediate defensible application is as a **research testbed for protocolized relational retrieval**. A deployed exact-query service should use the oracle until a quality-matched model advantage is measured. A future learned-relation service needs a new data contract and fair comparisons against lookup, neighborhood retrieval, direct membership and learned ranking before the decoder earns its extra cost. The [data feasibility review](../evidence/updates/2026-09-25-candidate-score-decomposition/experiments/2026-09-25-learned-retrieval-data-review.md) is a lead, not an adopted dataset or a measured result.
