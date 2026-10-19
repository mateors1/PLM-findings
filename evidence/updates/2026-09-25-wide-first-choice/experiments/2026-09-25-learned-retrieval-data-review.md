# Learned retrieval: three concrete data leads

**Parked by user instruction, 2026-09-25:** active work stays within Pokémon
while pursuing oracle parity and the user-directed evolutions. External datasets
require an explicit user request before acquisition, preparation or experiments.
The recommendations below are historical leads, not the current work plan.

Reviewed: 2026-09-25. This is a source review and feasibility recommendation,
not dataset adoption, a downloaded corpus or a measured PLM application.
It follows the [feasibility question](2026-09-25-learned-retrieval-feasibility.md).

The central issue is what the labels actually mean. None of the three sources
below establishes the fully judged complementary relation proposed earlier.
A behavioral prediction task could still test genuinely learned retrieval,
provided it receives its own explicit task contract.

## Evidence from maintainers and authors

| Source | What the source supplies | Consequence for this project |
| --- | --- | --- |
| Polyvore Outfits | User-curated outfits; the paper forms negatives using randomly sampled same-category alternatives. | Co-outfit evidence and constructed negatives do not equal independent incompatible-pair judgments. Candidate categories alone may not make steering load-bearing. |
| Retailrocket | Timestamped visitor/item events with view, add-to-cart and transaction types. | A plausible source for predicting observed subsequent behavior under distinct event contexts, not product compatibility. |
| Amazon Reviews 2023 | Reviews and product metadata; the schema defines `bought_together` as website-recommended bundles. | Do not label that field observed baskets. Review timestamps do not establish when recommendation links arose. |

The corresponding primary sources are the
[Polyvore author paper](https://arxiv.org/html/1803.09196),
[Retailrocket publisher card](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset/home)
and [Amazon maintainer schema](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

The Polyvore author's current card describes approval-based academic/nonprofit
access while also stating CC BY4.0 commercial permissions. Those conflicting
statements remain unresolved here; the card also notes third-party content
rights. Its outfit-disjoint and item-disjoint variants imply different leakage
and cold-start questions. [Author dataset card](https://huggingface.co/datasets/mvasil/polyvore-outfits/blob/main/README.md)

Retailrocket's publisher card lists CC BY-NC-SA4.0 and a version2 file bundle.
It includes behavior events and time-varying item properties. Direct page
opening returned no extracted text in this review; indexed publisher-card
content supplied these facts. File access, exact release bytes and checksums
have not been tested. [Publisher card](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset/home)

The Amazon maintainer says it cannot assign a dataset license, describes
research availability and later approves a questioner's thesis use. This review
does not turn those responses into a blanket redistribution permission.
[Maintainer license discussion](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1)

These are recorded source statements, not a new determination of rights or a
change to the current project's settled research-use stance. No access requests,
accounts, dataset downloads or third-party messages were made in this review.

## Recommended next audit: observed behavior after an item view

My inference from the documented schema is that Retailrocket provides the
clearest first feasibility audit among these leads. The proposed question is:

> Given a known item that was viewed, do subsequent known-item targets differ
> meaningfully between add-to-cart and transaction contexts?

This would be a directional, context-steered behavioral relation. It would not
establish that two products complement each other, that an event caused another
event, or that recommendation improves sales. It explicitly changes the proposed
target from judged compatibility; it does not relabel the earlier proposal.

Before training, the audit must produce these concrete outputs:

1. A release manifest with publisher URL/version, retrieval details, source-file
   hashes, declared usage terms and a reproducible parser. User/visitor IDs are
   grouping metadata, never PLM output tokens.
2. A frozen session/window rule and chronological cutoffs. Count usable subject/
   target pairs and known-item coverage for both contexts; disclose events dropped
   by window boundaries. Group duplicate events and transactions consistently.
3. Training-only catalog and feature construction. Future events and future item
   properties cannot contribute to a training lookup, neighborhood or popularity
   table. Separate recurring transitions from genuinely unseen pairs.
4. A support and steering report. Check whether both contexts have enough data
   and distinct conditional targets, including comparison against context-specific
   popularity. Mere differences in total event counts are insufficient.
5. A declared evaluation convention for implicit feedback. Unobserved transitions
   remain unknown; sampled candidates are evaluation choices, not verified
   negative judgments. No complete-answer exactness claim follows from sparse logs.
6. Baselines using the same permitted information: training-only lookup, event-
   specific popularity, transition counts and a small directional scorer. Report
   their costs and validation tuning budgets before adding a decoder comparison.

The present symmetric relation head cannot silently represent this directional
task. The v1 compiler's complete SAME sets, class-balanced BCE targets and
subject/dimension split also cannot be transferred unchanged. Any successful
data audit therefore leads to a separate protocol/corpus/evaluation proposal;
it does not mutate the existing Pokémon vocabulary or its protected test.

The current recommendation is to audit feasibility, not to claim that this
dataset will support a useful application or that PLM will beat a simple ranker.
