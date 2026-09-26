# Native PLM versus SQLite graph oracle — declared 2026-09-25

## Question

On the same Pokémon graph and query workload, how does the current native PLM
HTTP endpoint compare with a deterministic graph-oracle endpoint for exact-set
quality and serial serving latency/throughput?

## Frozen identities and scope

- Use the existing seed-1729 native deployment receipt at
  `runs/learning/native-saturation-20260925/deployments/native-a2hhcdo7/serving.json`.
  Its checkpoint SHA256 is
  `3af8aa4a7bfae722aa27b04a8b2cc9982dedbf718915ed81ea088e31558a98d1`.
- Use the immutable deployment graph snapshot, SHA256
  `28aa79a1a888b52110fe329684dd958306f6db04081f3077ca9bb750cca1e747`, with
  canonical corpus graph hash
  `bcd363451618f1a83f33213098dff803507b8c02966c7bce08ea0c82ecab2a0d`.
  The loaded PLM runtime must reproduce the receipt's checkpoint, graph,
  vocabulary, records and split identities before serving.
- Use all 222 `(subject, dimension)` validation queries from the bound split.
  `TYPE` and `COLOR` are both included. Training labels may initialize the model
  as recorded in its checkpoint, but no training queries are benchmarked and
  the protected final-test split is not read for evaluation.
- The PLM arm is the existing constrained, unique, KV-cached native
  `POST /v1/predict` path with first-target guidance alpha 16. The graph arm is
  a benchmark-only loopback HTTP app that runs the corpus compiler's graph
  target query and ordering against the same read-only snapshot. Both use the
  same Uvicorn/HTTP client, request bodies, query order and product hydration.
  The graph response has no model-token trace; record response bytes so this
  payload difference remains visible.

## Measurement

- Before timing, verify graph-oracle targets against the validation records and
  verify one PLM HTTP warmup response. Then issue exactly one serial timed pass
  over the 222 validation queries to each arm. No retries or concurrent load.
- Measure loopback client p50/p95/p99 latency, elapsed time and completed RPS.
  Report response bytes and PLM-generated output tokens separately. Timing starts
  after service startup and warmup; it includes request transport, prediction,
  hydration and response serialization.
- Compare order-insensitive postprocessed prediction sets to graph targets.
  Report exact-set accuracy, macro F1, micro precision/recall, and PLM protocol
  validity and termination. Invalid or incomplete PLM responses count as empty
  predicted sets. The SQL oracle must match the graph-derived validation labels
  exactly or the campaign stops before serving measurements are accepted.
- Save the exact executed script, source archive, immutable raw summary and a
  portable result bound to the plan, checkpoint, graph snapshot, corpus split,
  source tree and runtime identities. Do not change checkpoints, serving
  defaults or graph data.

## Interpretation limits

This is an initial single-host, loopback, serial comparison on one validation
split. It is not a concurrency/SLO, energy/request, vLLM or production-capacity
result. The graph oracle is expected to be exact by construction; the relevant
quality result is how closely this PLM serving policy matches it.
