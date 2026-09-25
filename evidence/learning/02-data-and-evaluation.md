# Lesson 2: knowing what an experiment actually measured

**2026-09-24.** Read after [training correctness](01-training-correctness.md).

We can now create a traceable dataset, train against it, and measure complete
generated answers. These are prerequisites for comparing architecture choices.
An exciting loss curve is hard to interpret if the data, code, or evaluator changed.

## A snapshot is a frozen teacher

The compiler is our teacher: graph facts plus protocol rules determine the target
sequences. We created `pokemon_v1_f1541479_20260924` from PokeAPI revision
`f15414790832c88d784d1537658b957fd73cbbbd`, preserving the older dataset.
It contains 1,025 products, 2,576 edges and 2,050 records. Records and vocabulary
match the earlier corpus; the new version repairs graph identity and provenance.

```powershell
uv run --no-sync plm graph snapshot --name pokemon_v1_f1541479_20260924
```

The command records hashes for five source CSVs, the SQLite file, and three
corpus files. It checks the graph, validates the corpus against it, and compiles
again into a second directory to check byte-for-byte reproducibility. The JSON
receipt lives in `data/manifests/`; large data artifacts remain ignored by Git.

Running it again verifies existing files without fetching. A clean checkout
containing only the receipt can rebuild from the pinned sources: the original
snapshot timestamp is preserved, and all nine file hashes must match before
publication. A different SQLite runtime could produce different database bytes;
the command rejects that difference rather than weakening the recorded identity.
Partial or modified snapshots are rejected. Use a separate mutable database
for labeling, then publish a new dataset version.

**Semantic identity** means the facts are the same; **byte identity** means the
file representation is identical. The corpus binds to a semantic graph hash;
the receipt additionally checks the exact files. These answer different questions.

## A run is more than a checkpoint

A checkpoint saves weights, optimizer/scheduler state, RNG state and training
position. The runner also saves:

| File | Why it matters |
| --- | --- |
| `run.json` | Effective config, data/split identity, seed and environment |
| `source.zip` | Exact Python source and dependency definitions, including uncommitted changes |
| `metrics.jsonl` | Progress at evaluation intervals |
| `training-result.json` | Final step, losses, history and checkpoint fingerprint |

A Git commit alone cannot identify an experiment run with uncommitted edits.
The source hash and archive address that gap. Checkpoint metadata inside the
saved payload is authoritative; a conflicting JSON sidecar now fails validation.

## Ranking one token does not evaluate a whole answer

Suppose the oracle answer is `[B, C, D]`. The model's first prediction might rank
B first, giving excellent first-token ranking, yet generate `[B, B, EOS]`.
That response has duplicates and misses C and D. We now measure both behaviors.

Native generation starts from only the five-token prompt. Each step appends the
predicted ID and repeats until EOS or a configured bound. It does not append EOS
to disguise a truncated response. The protocol mask allows legal token classes;
it does not look up correct targets or remove repeated/subject IDs for the model.

For generated list P and expected set G, our per-query metrics use:

```text
matches   = |set(P) intersect G|
precision = matches / len(P)     # duplicates consume output slots
recall    = matches / |G|
F1        = 2 * precision * recall / (precision + recall)
```

Empty predictions score zero. Metrics are averaged over queries (**macro
averaging**), so a query with 200 targets does not outweigh one with two.
Exact-set accuracy requires the right unique members, valid syntax and observed
termination. Exact-sequence accuracy additionally requires the teacher's order.
Termination, protocol validity, duplicates, self-return and output length are
reported separately. A valid protocol response can still be factually wrong.

First-token `PLMScorer` now reuses one forward pass per query instead of repeating
the same prompt computation for every candidate. Its standalone result leaves
generation validity unknown; the CLI measures validity through actual generation.

## Comparable losses

With MoE, training optimizes token loss plus a router-balancing penalty. Comparing
that sum directly with dense token loss mixes two objectives. `task_loss` exposes
cross-entropy alone. Validation reports token-weighted cross-entropy:

```text
validation_CE = sum(batch_CE * supervised_token_count) / total_supervised_tokens
```

This differs intentionally from macro query accuracy. It estimates prediction
loss per supervised token, excluding prompt and padding positions. Router loss
and per-layer expert loads have separate telemetry; last-batch telemetry is not
a full-dataset specialization analysis.

## What this establishes

An integration test constructs a tiny graph, trains, saves its source/config and
checkpoint, and invokes complete validation evaluation. It also checks identity
mismatches, sidecar tampering and the protected test path. This demonstrates the
pipeline mechanics. National Dex validation quality and deployment performance
still require recorded experiments.

Next: [Lesson 3: architecture experiments](03-architecture-experiments.md).
