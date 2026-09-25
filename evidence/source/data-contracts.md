# Data contracts

## Graph schema (SQLite, v1)

See [`plm.graph.schema`](../src/plm/graph/schema.py). Protocol-agnostic:
structure and integrity only; relation *semantics* live in the protocol.

- `provenance(id, source_type, citation, license, notes)` — lineage for source facts.
- `nodes(id, key, namespace, kind, label, status, provenance_id)` — product keys
  are `PKM_*`; hidden pivots are `ATTR_*`.
- `relation_types(id, name, category)`
- `edges(id, src, dst, rel, weight, confidence, is_canonical, provenance_id)` —
  `UNIQUE(src, dst, rel)`, FKs enforced.
- `aliases(id, node_id, alias, locale, priority)` — deferred in v1.
- `snapshots(id, name, protocol_version, tokenizer_version, node_count, edge_count, hash, created_at)` — versioned snapshots.
- `meta(key, value)` — `schema_version`, `protocol_version`.

## Versioning invariants

1. A regenerated corpus is a **new dataset version** even from the same graph.
2. Every run records: resolved config, git SHA, `uv.lock` checksum,
   Python/Torch/CUDA versions, GPU name, seed, corpus manifest, tokenizer hash.
3. Serving reads an **immutable snapshot** copy — never a live-written `.db`
   (SQLite locking is unsafe across the WSL/Windows boundary).

## What is committed vs ignored

Native serving uses SQLite backup to create a dedicated `graph.snapshot.db` in
each deployment directory. Startup validates that copy against corpus/checkpoint
identities and loads a frozen product-label lookup. Later edits to the original
database or snapshot file do not alter that running lookup. `serving.json` and
`source.zip` record the deployment's configuration, identities and implementation.
The snapshot's relation facts do not repair generated predictions.

- **Committed:** manifests, corpus recipes, protocol spec, test fixtures, configs.
- **Ignored:** `data/raw`, `data/processed`, `data/sqlite/*.db`, `runs/`, wandb,
  TensorBoard events, checkpoints, `.env`.

## National Dex v1 recipe

`plm graph import-national-dex` imports all PokeAPI National-Dex species using
the maintained PokeAPI CSV dump. It persists the `TYPE` and `COLOR` derivation
facts only, with provenance in SQLite. `plm corpus compile` consumes that graph
plus `configs/protocol/pokemon_v1.yaml` and writes ignored corpus artifacts:
`records.jsonl`, `vocabulary.json`, and `manifest.json`.

## Query partitions and labeling

`validation_query_fraction`, `test_query_fraction`, and `split_seed` define a deterministic
train/validation/test partition of complete corpus queries. The graph is never
mutated by evaluation. Baselines and the future PLM consume the same candidate
vocabulary and multi-positive ranking metrics; `split_hash` records all three
ordered assignments.

The local labeling app is the curation path for TYPE/COLOR facts. A save validates
known attributes, replaces both outgoing relations atomically, and records a
`curated` provenance row shared by the replacement edges. It is local-only in v1;
remote authentication and multi-user collaboration are not part of this contract.
