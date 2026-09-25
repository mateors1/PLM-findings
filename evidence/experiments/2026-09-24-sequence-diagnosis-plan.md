# Sequence failure diagnosis: declared before execution

Use the fixed seed-1729 prompt-set final checkpoint and its 222 validation
queries. Do not train, select another checkpoint, tune a threshold or score test.

1. Validate checkpoint/corpus/split/report identities through the shared loader.
2. Measure next-token accuracy under the correct preceding tokens (teacher
   forcing), separately for the first target, remaining targets and EOS.
3. Reproduce every archived raw response with the diagnostic's zero-hint cached
   loop before interpreting its one-hint results.
4. Inject exactly the first correct target ID after ANSWER, then decode greedily
   under the same syntax mask and total answer-token budget. The injected token
   consumes one budget position. Record raw and identical post-policy metrics,
   every response, successes rescued and previous successes lost.
5. Group validation queries by dimension and count of training queries with the
   same target-set-plus-subject fingerprint. This is an answer-family diagnostic
   derived from train/validation corpus records, not an input to the model.

The one-hint intervention is deliberately oracle-assisted diagnostic evidence.
It is not a deployable candidate, unassisted quality score or throughput result.
Higher exact accuracy would implicate the first decision and list continuation;
remaining failures would show that supplying the first token is insufficient.
Do not infer that a single oracle token contains only one bit of information.
Preserve source, script, config and input hashes plus per-query observations.
