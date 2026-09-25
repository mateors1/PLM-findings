# Symmetric relation supervision: controlled candidate

The previous diagnosis finds a weak first decision and sparse TYPE errors;
checkpoint selection retains the final prompt-set anchor. Test whether directly
teaching SAME symmetry through its shared embeddings improves generation.

Candidate: add `symmetric_relation_loss_weight=1` to prompt-set seed 1729, with
the same corpus/split, dense architecture, optimizer/schedule, batch size 32 and
2,000 optimizer updates. Select the final checkpoint before training; no checkpoint
sweep for this candidate. Keep prior prompt-set and token coefficients unchanged.

Score: normalize shared product embeddings to norm sqrt(D); map the normalized
dimension-token embedding through a learned bias-free D-by-D projection to a
signed diagonal relation vector. Compute z(s,d,t) = sum_k e_s[k] r_d[k] e_t[k] / sqrt(D)
in FP32. Swapping subject/target preserves the score. Add 65,536 parameters at D=256.
This is a DistMult-style score with project-specific normalization and conditioning,
not a reproduction of the original DistMult training protocol.

Labels come only from the current training record's answer set. Collapse duplicates,
exclude the subject from the auxiliary loss, and balance remaining positive/negative
classes per query using the existing binary loss helper. Self-exclusion is a known
application rule, so the auxiliary relation scorer need not model a negative diagonal.
No attribute labels, extra queries, validation labels or oracle predictions enter training.

Keep the symmetric head out of autoregressive decoding: the possible benefit flows
through training gradients into shared embeddings. Report total/token/first/set/symmetric
loss separately. Verify symmetry, future invariance, gradients/diagonal masking,
common initialization, cached token logits, toy overfit and deterministic resume.

After all gates, train once and evaluate all 222 validation queries using ordinary
unassisted cached greedy generation and the existing SAME post-policy. Compare
against the retained prompt-set anchor: eligible only with all responses valid and
terminated; primary processed exact-set accuracy, then processed F1. Report raw
F1 and TYPE/COLOR separately regardless of outcome. Keep deployment unchanged;
replicate a promising result before stronger claims. Final test stays untouched.

Reference: [Yang et al., ICLR 2015](https://arxiv.org/abs/1412.6575), diagonal
bilinear relation models. The symmetry follows directly from our scalar product;
its suitability for learning and transfer to generation is the experiment.
