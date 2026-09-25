# Lesson 8: turn the model into a reproducible service

**2026-09-24.** A checkpoint is a file of learned parameters. A service must also
define accepted requests, data identity, output handling, concurrency and failure
behavior. This iteration connects those boundaries without changing the learned
weights. The [plan](../experiments/2026-09-24-native-service-plan.md) fixes the
anchor checkpoint and validation comparison before the live HTTP run.

**Result:** a real loopback server reproduced all 222 archived raw validation
responses exactly. Filtered/hydrated results reproduced the prior offline
post-policy metrics: F1 63.19%, exact answers 105/222 (47.30%). This verifies the
application wiring; it does not make the remaining model errors correct.

## What was verified over actual HTTP

The [portable result](../experiments/2026-09-24-native-service.json) binds the
checkpoint, data, source archive and deployment receipt. The raw local report
retains every request's response and timing. The verification used an ephemeral
loopback port, a real Uvicorn server and ordinary HTTP requests; it stopped its
own server cleanly afterward.

- All 222 raw responses matched, including target order, validity, EOS and errors.
- The processed metrics matched the archived offline policy evaluation exactly.
- IGNORE/limit requests preserved raw generation; a zero limit returned an empty
  processed result, and a hidden attribute subject returned HTTP 422.
- The test suite checks snapshot isolation, malformed inputs, checkpoint/data
  mismatches, failed-generation handling, serial execution and admission bounds.

The serial HTTP run took 167.55 seconds across 222 requests, with median client
latency 0.690 seconds. These are observations from a correctness run: no warmup,
first request included, no concurrency sweep or independent repetitions. Do not
compare this median with Lesson 7's different eight-prompt workload as a speedup.
It is not a p95/p99/SLO, energy or oracle-performance result.

## The model is one stage inside the application

```text
HTTP JSON request
    -> strict request validation and product lookup
    -> five-token protocol prompt
    -> PLM generates entity IDs and EOS
    -> parse and validate the complete raw sequence
    -> SAME self-exclusion, IGNORE, deduplication, limit
    -> hydrate remaining IDs with display labels
    -> HTTP JSON response with raw evidence and processed result
```

The API uses JSON, but the model still emits **identifiers only**. Python builds
the JSON response around those identifiers. Hydration supplies display labels;
the decoder does not generate their spelling as free text.

For this request:

```json
{"subject": "PKM_PIKACHU", "dimension": "TYPE", "ignore": ["PKM_RAICHU"], "limit": 5}
```

the model receives:

```text
[BOS, PKM_PIKACHU, TYPE, SAME, ANSWER]   shape [1, 5], integer token IDs
```

IGNORE and the number five remain Python request metadata. They are not tokens,
embedding inputs or hidden attribute labels. Changing these metadata fields
must leave raw generation unchanged; tests and the live check verify that.

The model still determines which relationships it predicts. The output rules
can remove a subject or duplicate but cannot identify a wrong TYPE relationship
or add a missing correct product. Serving therefore preserves the distinction
between raw model quality and the application's filtered result.

## One loading path prevents two definitions of the model

Evaluation and serving now share `load_inference_runtime`. It verifies the
checkpoint's objective/model configuration, authoritative payload metadata,
vocabulary, corpus, graph and train/validation/test split identities. It then
loads the model on the selected device and enters evaluation mode.

An identity is a collection of hashes and settings that says which experiment
an artifact belongs to. A successful tensor load alone is insufficient: a model
can have compatible tensor shapes while using the wrong vocabulary or graph.
Wrong IDs would then be attached to otherwise plausible-looking scores.

Training identity remains attached to the checkpoint. A deployment separately
records inference settings, current source archive, device, snapshot identity
and scheduling policy. Turning on KV caching changes execution identity, not
the history of how the weights were trained.

## A running deployment gets its own graph snapshot

The labeling database is editable. A deployed model is tied to a particular graph
version, so hydration must not silently follow later edits to that live database.
Startup uses SQLite backup to make a consistent copy in a new deployment directory,
then validates that copy against corpus/checkpoint identity.

The service loads active product keys and labels into a frozen lookup from the
copy. It does not pass relation edges or oracle target lists into generation.
Later edits to the original database, or even to the snapshot file, cannot change
this already-loaded lookup. Deploying updated data requires a newly validated
deployment. Tests exercise isolation using temporary fixtures.

Each deployment directory contains `graph.snapshot.db`, `serving.json` and
`source.zip`. The receipt records the exact checkpoint, graph, vocabulary,
configuration and source identities. It is useful research evidence, not a claim
that the model is correct on every request.

## Concurrency: accepting requests is different from running a batch

The native reference uses one model and a lock around generation. Accepted
requests wait their turn, and a semaphore limits running-plus-waiting requests
to eight by default. A **semaphore** is an admission counter; a **lock** prevents
two threads from entering the model execution region simultaneously.

This is serial scheduling with bounded admission, not GPU batching or continuous
batching. Increasing the request limit permits a longer queue; it does not make
the GPU complete more requests per second. When admission is full, the service
returns HTTP 503 with a retry hint instead of silently accepting unbounded work.

```text
client latency ~= validation + queue wait + generation + postprocess/hydration
                  + HTTP transport/serialization overhead
```

The response exposes queue, generation and total predictor times. External
client timing also includes HTTP work. Queue time can grow even when model
generation time stays constant, which is why future serving benchmarks need
concurrency levels, tail latency, error rate and quality together.

KV cache state stays local to each generation call. It is not reused across
users. The queue limit and execution lock are independent of the cache.

## Errors are part of the contract

| Situation | HTTP behavior |
| --- | --- |
| Valid request and complete valid generation | 200 with raw and processed results |
| Unknown/non-product entity, wrong field type, unsupported dimension/mode or extra field | 422; no inference |
| Generation ends without EOS or fails protocol validation | 502 with raw failure evidence; no successful hydrated result |
| Admission capacity exhausted | 503 with retry hint; no inference |

Unexpected internal runtime errors remain server errors; they are not
misclassified as bad user requests. A failure releases its admission slot.
The endpoint checks the parsed subject/dimension/mode against the request and
never turns malformed or truncated output into a successful answer.

A zero return limit legitimately produces an empty processed result. In this
reference it still runs normal generation first, so metadata does not change
the model's output path. It is not an early-stopping instruction.

## Removing avoidable grammar work

Lesson 7's profile found hundreds of thousands of token-class lookups in one
request. `allowed_token_ids` was constructing dimension and mode lists even
after the prompt was complete. Those scans now happen only at the relevant
prompt stage; the answer stage still validates the prefix and permits only
entity tokens followed by EOS.

Inspection also found that some malformed earlier prompt positions could be
overlooked when the helper checked a later position. It now validates the whole
prefix and rejects invalid/negative/out-of-range IDs. The v1 mode remains SAME
only. Tests check failures at each position and verify that answer-stage class
lookups scale with prefix length rather than scanning steering classes across
the whole vocabulary. Entity-list enumeration and full-prefix checks still
remain; this is not yet an incremental parser.

Repeating the same instrumented 171-token TOXAPEX/COLOR diagnostic reduced
`class_of` calls from 715,638 to 15,393. The [new profile](../experiments/2026-09-24-native-overhead.json)
records 0.044 s cumulatively in `allowed_token_ids`, compared with 0.191 s before.
These are instrumented observations rather than a paired application benchmark.
The call-count reduction confirms that the redundant work was removed; it does
not isolate a precise production latency improvement.

## Run it locally

```powershell
uv sync --group training --group serving
uv run --no-sync plm serve --checkpoint runs/national_dex_promptset_s1729_v1/checkpoint-final.pt --override +experiment=prompt_set_lab --override eval.use_kv_cache=true
```

The default address is `http://127.0.0.1:8000`. Open `/docs` for the interactive
API schema, `/health` for readiness/deployment identity, or send a request:

```powershell
$query = @{ subject = "PKM_PIKACHU"; dimension = "TYPE"; limit = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/predict -Method Post -ContentType application/json -Body $query
```

The checkpoint and ignored local data artifacts must already exist. The
`--runtime-dir` option chooses where deployment snapshots/receipts are written;
`--max-pending` changes the admission bound. The CLI uses one server worker so
multiple workers do not silently load multiple copies of the model.

The native API is cross-platform. The vLLM adapter still belongs to the separate
Linux/WSL2 project and is not implemented by this change. A working HTTP boundary
does not establish oracle parity, a serving speed advantage, or an energy win.
