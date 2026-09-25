# AGENTS.md

Standing guidance for Codex (and humans) working in this repository. The
rules here are project policy — follow them every session.

## What this is

**PLM — Protocolized Language Model ("Pika Edition").** A deliberately
over-engineered, SOTA decoder-only transformer that learns a *steered relational
policy* over a product graph and emits **entity identifiers only — never text,
JSON, or images**. The model is one component inside a deterministic pipeline
(parse → tokens → PLM → IDs → hydrate). Pokémon names are the first dataset,
treated as opaque SKUs; the research payload is the **protocol** and the
**serving/throughput** story, not language.

**Read these first**, in order:
- `research_log.md` — *why* the project is shaped the way it is (decisions, dead ends).
- `devlog.md` — *what* happened in recent sessions.
- `registry.md` — the symbol index. **Check it before writing new code (DRY).**
- `docs/` — `architecture.md`, `protocol-spec.md`, `data-contracts.md`, `decisions/`.

## Standing practices — keep these current every session

### 1. `research_log.md` — the "why" (append-only narrative)
Append a **dated** entry whenever you: make an architectural decision, hit a dead
end or failed experiment, run a benchmark, or overturn a settled assumption.
Chronological and append-only — never rewrite history; add a new entry that
supersedes the old one and say so. This is the source material for the eventual
paper / blog / portfolio writeup, so capture rationale, not just the outcome.

### 2. `devlog.md` — the "what" (session summaries)
At the **end of each working session**, append one short dated entry: what was
done, what changed, and what's next. Keep it skimmable — this is the fast catch-up
for the next session, not a transcript.

### 3. `registry.md` — the "where" (DRY index)
`registry.md` is the human-navigable index of every public symbol, DB table,
config field, CLI command, and protocol token.
- **Before** adding a function/class/constant/table/token, search `registry.md`
  to avoid duplicating something that already exists.
- **After** adding, renaming, or removing a public symbol, update `registry.md`
  in the *same* change. A stale registry is a bug.

## Settled decisions — do not relitigate (rationale in `research_log.md`)

**Active research boundary (user instruction, 2026-09-25):** keep work within
the Pokémon realm while pursuing oracle parity and the user-directed evolutions.
External datasets may remain parked references, but do not acquire, prepare or
experiment on them unless the user explicitly asks. Existing Retailrocket
metadata/access notes are a parked lead, not an active data or application track.

**Scope (v1):**
- Identifiers: Pokémon names (opaque SKUs, `PKM_<NAME>`).
- Mode (main classifier): **`SAME` only.** `COMPLEMENTARY` is deferred; `RELATED` is out.
- Dimensions (secondary classifier): **`TYPE` and `COLOR`.** `BIOME`, `PREDATOR/PREY`
  are project-scope but **not** this-scope. Two dimensions is the minimum that makes
  the steering token load-bearing (with one dimension the protocol is degenerate).
- `IGNORE` is a **parser/post-process directive, never a model token.**
- **No numeric tokens.** Count / `RETURN` / `SIZE` / `PAGE` are parser metadata; the
  model never sees numbers. Generation stops on `EOS` or a serving max-length bound.

**Grammar & vocabulary:**
- Sequence: `BOS SUBJECT DIMENSION MODE ANSWER TARGET… EOS`.
- `ANSWER` is the required prompt/completion delimiter and the loss-mask boundary.
- Reserved ID ranges: specials/control `0–31`, dimensions/modes `32–127`,
  entities `1024+`. Entity IDs are **append-only** — never renumber.
- Attribute values (electric, yellow, …) are **hidden graph nodes** (`kind='attribute'`),
  **never tokenized and never emitted.** Only `kind='product'` nodes get a `PKM_` token.
  The model never learns *what* a type is — only that two names share one.

**Data model & stores (three layers, three stores):**
- **Graph = source of truth = SQLite.** Integer PK + stable string `key`; **no UUIDs**
  (they break the "corpus is a pure function of the snapshot" guarantee).
- **Protocol = the language = versioned config files** (dimensions, modes, derivation
  rules, complement policy). Not DB rows.
- **Corpus = the teacher's output = content-addressed files** (vocabulary JSON,
  corpus manifest, training records). Not the source-of-truth DB.

**Labeling:** bucket-style ("what type/color is this Pokémon?"), not pairwise
("are these two the same?"). Bucketing is O(N) clicks and yields a dense, consistent,
complete oracle-labeled corpus; pairwise is O(N²) and starves the model.

**Experimental framing (two experiments):**
- **Exp A (v1 — systems).** `TYPE/COLOR SAME` is *deterministically computable*, so the
  compiler is a perfect **oracle**. v1 is a controlled **systems/serving** proof: match
  the oracle on quality, win on concurrent-users@latency and energy/request. **Do not
  claim a quality win over the baseline in v1** — the baseline is by definition correct.
- **Exp B (later — learning/policy).** `COMPLEMENTARY`, compositional held-out relations,
  world-evolution — the things the compiler *cannot* compute. This is where "the model
  only learns what cannot be computed deterministically" applies.

## Working conventions

- Run everything through **uv**: `uv run ruff check`, `uv run ruff format`,
  `uv run mypy` (`--strict`), `uv run pytest`. All must be green before committing.
- Core env has **no torch** (Phase A–B). Torch/serving live behind dependency groups
  and lockfiles; keep imports torch-free at module import time in core packages.
- Windows-native for Phases A–C; WSL2/Linux only at the Phase D serving milestone.
- Two lockfiles: `serving/` (vLLM) resolves separately; the main uv project never
  resolves it.
- If `uv sync` / `uv python install` fails on Windows with *"Missing expected target
  directory for Python minor version link"*, pass `--python <path-to-cpython>` explicitly
  (the interpreter extracts fine; only the optional minor-version junction fails).
- Naming: `PKM_<NAME>` for product/entity node keys; `ATTR_<VALUE>` for hidden
  attribute node keys; protocol tokens are `UPPERCASE_ASCII`.
- Don't relitigate the fair-use stance (non-profit, research, transformative — settled)
  or the settled decisions above.

## Git

- Commit or push only when asked. Branch off `main` first if asked to commit.
- Keep `registry.md`, `research_log.md`, and `devlog.md` updated in the same change
  that warrants them.
