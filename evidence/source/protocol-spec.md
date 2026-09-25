# Protocol specification — Pokémon v1

**Status: implemented.** The load-bearing versioned config is
`configs/protocol/pokemon_v1.yaml`.

## Grammar

```text
sequence = BOS SUBJECT DIMENSION SAME ANSWER TARGET+ EOS
DIMENSION = TYPE | COLOR
```

`SUBJECT` and every `TARGET` are `PKM_*` product tokens. `ANSWER` is the
prompt/completion delimiter: labels through it are `-100` (ignored by causal-LM
loss); every target and `EOS` is supervised. `IGNORE`, counts, pagination,
predicates, and attribute values are parser/serving concerns and never model
tokens.

## Optional decoder constraints

The optional decoding setting `eval.prevent_repeated_targets=true` masks entity
IDs already emitted after ANSWER. It requires protocol-constrained decoding and
is recorded as `+unique-v1`; its default is false. The prompt subject remains
eligible until emitted, and EOS is not fabricated when the length bound is hit.
This restricts decoder choices without consulting relations or changing the
grammar/parser. Legacy repeated outputs can still be parsed and measured.
The seen-target set is local to one request and independent of IGNORE/count metadata.

## Optional first-target guidance

`eval.first_target_guidance_alpha` is a finite nonnegative inference setting,
defaulting to zero. Positive strengths require constrained cached decoding and
a checkpoint with the symmetric relation head. At the first generated position,
product logits receive `alpha * logsigmoid(relation_logits)` from the model's
existing prompt-only head. Later positions receive no direct guidance penalty.
The helper introduces no expected targets or graph relation lookup.

The grammar and five-token prompt stay unchanged: alpha is configuration, not a
numeric model token. Guidance does not exclude the subject, force EOS, or imply
target uniqueness; the optional uniqueness setting remains separate. Positive
guidance has a canonical numeric decoding suffix; zero retains the original
descriptor and skips the head call. Native HTTP runs one query at a time;
offline evaluation may group eight independent prompts. See
[Lesson 17](learning/17-guided-integration.md) for the completed integration
verification: 1,332 offline replays and 666 real HTTP responses match their
full-token references. This verifies the tested application path, not oracle parity.

## Deterministic post-processing

After parsing a completed response, `postprocess_response` excludes the subject
for SAME, applies caller-provided IGNORE keys, removes duplicates in their first
occurrence order, then applies an optional nonnegative return limit. These are
request/identity rules, not graph-relation lookups. They cannot fill missing
products or decide whether a different product actually shares TYPE/COLOR.

The raw generated sequence still obeys TARGET+ EOS. Its processed result may be
empty after exclusions or a zero return limit. No EOS is manufactured, and raw
generation metrics remain separate from post-processed pipeline metrics.

## Graph derivation

`TYPE` maps to `HAS_TYPE`; `COLOR` maps to `HAS_COLOR`. Those predicates and
their `ATTR_*` targets are graph-internal. For a subject/dimension pair, targets
are products sharing one or more attribute pivots, excluding the subject. They
are ordered deterministically by shared-attribute count descending, confidence
descending, then product key ascending.

## Vocabulary and identity

The direct tokenizer uses control IDs `0–31`, steering IDs `32–127`, reserved
IDs `128–1023`, and append-only product IDs from `1024`. The compiler creates a
vocabulary from `kind='product'` nodes only. A manifest records the protocol
version, tokenizer hash, canonical graph hash, records hash, record count, and
vocabulary size.

The current National Dex import yields 1,025 products and 2,050 non-empty
`TYPE`/`COLOR` queries. Its longest sequence is 289 tokens, so the 512-token
tiny-model context is sufficient.

## Query partitions

Phase C splits complete `(subject, dimension)` queries into train, validation, and
test partitions, not nodes or individual targets. `validation_query_fraction` and
`test_query_fraction` are applied in that order; train receives the remainder.
Assignment is deterministic from a seeded SHA-256 digest, while the SQLite graph
remains unchanged. `split_hash` includes the ordered assignment of every query to
all three partitions. Evaluation uses all product entities as candidates and
treats every target in a query as positive; macro MRR, Hits@K, and MAP are
computed by the shared ranking evaluator with score-descending/product-key tie
breaking. The full-corpus oracle is expected to be perfect by construction.
