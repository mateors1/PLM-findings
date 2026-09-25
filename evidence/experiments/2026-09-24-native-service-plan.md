# Native service boundary: pre-run plan

Build a local, cross-platform PyTorch/FastAPI reference service before the
separate Linux/vLLM milestone. Do not claim Phase D performance completion.

- Share checkpoint/config/corpus/split validation between evaluation and serving.
- Back up the source graph into a dedicated deployment snapshot and validate it
  against checkpoint/corpus identity. Hydrate from a frozen product lookup loaded
  from that copy, never from the live curation database.
- Accept subject, TYPE/COLOR, SAME and metadata IGNORE/limit. Metadata never
  reaches model tokens. Preserve raw model evidence alongside processed results.
- Reject invalid requests before inference. Return explicit errors for invalid
  or truncated generation; never hydrate it as a successful complete answer.
- Serialize model execution with bounded admission, fresh request-local caches,
  queue/generation/total timing and deployment provenance. Default to loopback.
- Remove repeated steering-class scans from the grammar helper, retaining full
  prefix checks; fix malformed-prefix acceptance discovered during inspection.
- Test request validation, snapshot isolation, metadata behavior, generation
  failure, concurrency/admission and checkpoint identity. Exercise actual HTTP
  on all 222 fixed-anchor validation queries and compare raw outputs and the
  existing deterministic post-policy. Do not access final-test metrics.
- Retain the seed-1729 prompt-set checkpoint and opt in to its verified KV path
  for this integration experiment. This is API/correctness evidence, not model
  selection, oracle parity, energy or concurrency-at-SLO evidence.
