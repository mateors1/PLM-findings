# Learned first-target guidance — declared before evaluation

2026-09-24. Test a decoding intervention on the three frozen symmetric checkpoints
(seeds 1729/1730/1731). Training, vocabulary, corpus, validation partition and
post-processing stay fixed. This is exploratory validation, not final-test evidence.

For the first generated position only, modify each product's token score:

    guided_logit(v) = token_logit(v) + alpha * log(sigmoid(relation_logit(v)))

The relation head uses the five prompt IDs and learned weights only. Obtain its
scores through the existing forward path, without labels; no graph facts or
expected targets enter the wrapper. EOS/control/reserved scores stay unchanged.
Do not exclude the subject or enable target uniqueness in this experiment.
The subject's relation score was excluded from auxiliary training loss, a known
limitation to inspect if self-selection matters. Subsequent token scoring is
unchanged; guidance can affect later tokens only through the chosen history.

Run alpha = 0, 1, 4, 16 at all three seeds, batch size eight, constrained cached
FP32 generation, completion bound 507, all 222 validation queries. Alpha zero
must exactly replay every saved original-policy output and metric before that
seed's nonzero variants proceed. Check checkpoint/data/split/runtime identity
and unchanged model source. Save complete outputs, raw and processed metrics,
first-choice changes, first-target membership and exact-first-target accuracy.

Report every setting, including regressions. Predeclare further-integration
eligibility: all 666 responses valid and terminated; processed F1 and exact-set
accuracy no lower in any seed; strict mean exact-set improvement over alpha zero.
If multiple settings pass, select highest mean exact-set accuracy, then mean
processed F1, then smaller alpha. This selects a candidate for subsequent
integration and independent verification, not serving promotion or oracle parity.
If none pass, preserve the failure and use its diagnosis to choose the next step.

No retraining, default changes, HTTP promotion, final-test access, latency claim,
commit or push. The wrapper is an offline experiment, with extra prompt-forward
work that has not been optimized or benchmarked.
