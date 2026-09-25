"""Independent stdlib audit of the declared fresh-control margin screen.

No runner, model, or shared measurement helper imports. Protocol/set arithmetic
is independently reconstructed from raw saved paths and same-shape FP32 vectors.
Run only after the complete primary campaign and owner authorization.
"""

import argparse
import copy
import hashlib
import json
import math
import sqlite3
import struct
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUTS = {}
PLAN_SHA = "845663b5f739c43b84f372aaf098e097c6380719c2882bd64c56f0661b203803"
SOURCE_SHA = "e806549e1c778b5fc1529423e346ccf915b1484e3a5b322072fff5316c35ab6b"
CPU_SHA = "240764f7686a47a057941c5151cf6ac070303c1958065aa523aed0bbcb0995ef"
GPU_SHA = "2bb3716e49d25600181e34a427b6cbc21cf176f02d11e83179e55b325f8a9412"
RUNNER_SHA = "a5d50dc4b57277bde1cefb1cf533b4958b28b04eb515182f855b3a4aa81311b8"
ORIGINAL_SHA = "ec6c8e358586efa5808591ca0112ea13b4e52cae4b9891ccac82c299332b870d"
ORIGINAL_CONFIG_SHA = "6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148"
PAIR_SHA = "4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992"
PAIR_AUDIT_SHA = "13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057"
OBJECTIVE = "hard-boundary-hinge-fp32-amin-amax-query-mean-v1"
BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"
POLICY = BASE + "+first4-pair6-symmetric-set-logit-sum-v1"
ORDER = ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
ARMS = ("control", "treatment")
RUN_NAMES = (
    "national_dex_rank_margin_control_s1729_v1",
    "national_dex_rank_margin_m1_w01_s1729_v1",
)
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, digest=None):
    path = Path(path).resolve()
    actual = sha(path)
    check(digest is None or actual == digest, f"hash mismatch: {path}")
    check(str(path) not in INPUTS or INPUTS[str(path)] == actual, f"input drift: {path}")
    INPUTS[str(path)] = actual
    return json.loads(path.read_text(encoding="utf-8"))


def bind(path, digest):
    path = Path(path).resolve()
    check(sha(path) == digest, f"hash mismatch: {path}")
    check(str(path) not in INPUTS or INPUTS[str(path)] == digest, f"input drift: {path}")
    INPUTS[str(path)] = digest


def canonical(config):
    content = {k: v for k, v in config.items() if k != "run_name"}
    return hashlib.sha256(
        json.dumps(
            content, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def archive(path, environment):
    bind(path, environment["source_archive_sha256"])
    with zipfile.ZipFile(path) as source:
        names = source.namelist()
        check(len(names) == len(set(names)), "duplicate source archive path")
        contents = {name: source.read(name) for name in names}
    digest = hashlib.sha256()
    for name, payload in contents.items():
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(8, "big") + encoded)
        digest.update(len(payload).to_bytes(8, "big") + payload)
    check(digest.hexdigest() == environment["source_tree_sha256"], "source tree hash mismatch")
    check(
        hashlib.sha256(contents["uv.lock"]).hexdigest() == environment["uv_lock_sha256"],
        "lock hash mismatch",
    )
    return contents


def diagnostic_groups(path):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as database:
        counts = database.execute(
            "SELECT n.key,r.name,COUNT(DISTINCT e.dst) FROM nodes n "
            "JOIN edges e ON e.src=n.id JOIN relation_types r ON r.id=e.rel "
            "WHERE n.kind='product' AND r.name IN ('HAS_TYPE','HAS_COLOR') "
            "GROUP BY n.key,r.name"
        ).fetchall()
    result = {}
    for subject, relation, count in counts:
        dimension = relation.removeprefix("HAS_")
        check(
            (dimension == "TYPE" and count in (1, 2)) or (dimension == "COLOR" and count == 1),
            "unexpected diagnostic attribute cardinality",
        )
        result[subject, dimension] = (
            "COLOR" if dimension == "COLOR" else ("TYPE_single" if count == 1 else "TYPE_dual")
        )
    return result


def zip_contents(path):
    with zipfile.ZipFile(path) as source:
        names = source.namelist()
        check(len(names) == len(set(names)), "duplicate archive members")
        return {name: source.read(name) for name in names}


def vectors(values):
    check(isinstance(values, list) and len(values) == 1025, "head column count")
    for value in values:
        check(type(value) in (int, float) and math.isfinite(value), "nonfinite/nonnumeric head")
        check(struct.unpack("!f", struct.pack("!f", value))[0] == value, "head not exact FP32")


def parse_path(ids, subject, dimension, tokens, descriptor, bound=507):
    """Reconstruct all GenerationResult fields from the full token sequence."""
    lookup = {name: index for index, name in enumerate(tokens)}
    prompt = [lookup[name] for name in ("BOS", subject, dimension, "SAME", "ANSWER")]
    check(isinstance(ids, list) and all(type(t) is int for t in ids), "noninteger raw token")
    check(ids[:5] == prompt and 6 <= len(ids) <= 5 + bound, "raw prompt/budget")
    tail = ids[5:]
    ended = tail[-1] == lookup["EOS"]
    products = tail[:-1] if ended else tail
    check(products and all(1024 <= t < len(tokens) for t in products), "raw product grammar")
    check(ended or len(tail) == bound, "unfinished path stopped before bound")
    valid = ended and len(ids) >= 7
    error = None
    if not valid:
        error = (
            "generated response is shorter than the protocol grammar"
            if len(ids) < 7
            else "response must terminate with EOS"
        )
    return {
        "token_ids": ids,
        "targets": [tokens[t] for t in products],
        "terminated": ended,
        "protocol_valid": valid,
        "decoding": descriptor,
        "error": error,
    }


def build_composition(raw_paths, head, subject, dimension, tokens, bound=507):
    """No labels: independently form the fixed pool and choose its legal maximum."""
    vectors(head)
    check(len(raw_paths) == 4, "four raw paths required")
    source_sets, eligibility = [], []
    for rank, raw in enumerate(raw_paths, 1):
        descriptor = BASE if rank == 1 else BASE + f"+first-rank{rank}-v1"
        parsed = parse_path(raw["token_ids"], subject, dimension, tokens, descriptor, bound)
        check(raw == parsed, "source full sequence/flags/policy mismatch")
        ids = raw["token_ids"][5:-1] if parsed["terminated"] else raw["token_ids"][5:]
        unique = len(ids) == len(set(ids))
        eligible = parsed["terminated"] and parsed["protocol_valid"] and unique
        # Failed sources retain raw membership; removing SUBJECT here fabricates evidence.
        source_sets.append(
            set(ids) - {tokens.index(subject)}
            if parsed["terminated"] and parsed["protocol_valid"]
            else set(ids)
        )
        eligibility.append(eligible)
    pool = []
    for slot, ranks in enumerate(ORDER, 1):
        union = set()
        for rank in ranks:
            union.update(source_sets[rank - 1])
        ids = sorted(union)
        pool.append(
            {
                "slot": slot,
                "kind": "original" if slot <= 4 else "pair_composition",
                "source_ranks": list(ranks),
                "set_ids": ids,
                "source_eligible": all(eligibility[r - 1] for r in ranks),
                "score": math.fsum(head[token - 1024] for token in ids),
            }
        )
    valid = [slot for slot in pool if slot["source_eligible"]]
    chosen = sorted(valid, key=lambda s: (-s["score"], s["slot"]))[0] if valid else pool[0]
    return {
        "source_paths": raw_paths,
        "slots": pool,
        "selected_slot": chosen["slot"],
        "selected_kind": chosen["kind"],
        "selected_source_ranks": chosen["source_ranks"],
        "selected_set_ids": chosen["set_ids"],
        "selected_source_eligible": chosen["source_eligible"],
        "fallback_no_valid_source": not valid,
        "policy": POLICY,
    }


def exact_json(left, right, label):
    check(
        json.dumps(left, sort_keys=True, allow_nan=False)
        == json.dumps(right, sort_keys=True, allow_nan=False),
        label,
    )


def effective_recipe(original, arm):
    """Authenticate the raw old hash before the explicitly named default insertion."""
    check(
        canonical(original["config"])
        == ORIGINAL_CONFIG_SHA
        == original["identity"]["resolved_config_hash"],
        "original raw config identity",
    )
    check(arm in ARMS, "unknown arm")
    expected = copy.deepcopy(original["config"])
    check(
        not {"symmetric_margin_loss_weight", "symmetric_margin"} & expected["model"].keys(),
        "original recipe unexpectedly has margin fields",
    )
    check(
        not {"symmetric_set_reranking", "pair_set_composition"} & expected["eval"].keys(),
        "original recipe unexpectedly has later selection fields",
    )
    expected["eval"].update(symmetric_set_reranking=False, pair_set_composition=False)
    expected["model"].update(
        symmetric_margin_loss_weight=0.0 if arm == "control" else 0.1, symmetric_margin=1.0
    )
    expected["run_name"] = RUN_NAMES[ARMS.index(arm)]
    check(
        expected["seed"] == 1729
        and expected["train"]["max_steps"] == 2000
        and expected["train"]["batch_size"] == 32
        and expected["train"]["grad_accum_steps"] == 1
        and expected["train"]["precision"] == "bf16"
        and expected["train"]["optimizer"]["lr"] == 0.0003,
        "declared training recipe",
    )
    check(
        expected["model"]["first_target_loss_weight"] == 1.0
        and expected["model"]["prompt_set_loss_weight"] == 1.0
        and expected["model"]["symmetric_relation_loss_weight"] == 1.0
        and expected["model"]["continuation_set_loss_weight"] == 0.0,
        "declared objectives",
    )
    return expected


def evaluation_recipe(training):
    result = copy.deepcopy(training)
    result["eval"].update(
        constrained_decoding=True,
        use_kv_cache=True,
        prevent_repeated_targets=True,
        first_target_guidance_alpha=16.0,
        symmetric_set_reranking=True,
        pair_set_composition=True,
        generation_batch_size=8,
        max_new_tokens=507,
    )
    return result


def membership(values, subject_id, expected_ids):
    vectors(values)
    check(type(subject_id) is int and 1024 <= subject_id <= 2048, "invalid subject column")
    check(
        isinstance(expected_ids, list)
        and expected_ids == sorted(set(expected_ids))
        and all(type(i) is int and 1024 <= i <= 2048 and i != subject_id for i in expected_ids),
        "invalid expected IDs",
    )
    true = set(expected_ids)
    allowed = {i for i in range(1024, 2049) if i != subject_id}
    negative = allowed - true
    check(true and negative, "margin objective requires both classes")
    weakest = min(values[i - 1024] for i in true)
    strongest = max(values[i - 1024] for i in negative)
    predicted = sorted(i for i in allowed if values[i - 1024] > 0)
    return {
        "min_true_logit": weakest,
        "max_negative_logit": strongest,
        "gap": weakest - strongest,
        "strict_separable": weakest > strongest,
        "min_true_ids": sorted(i for i in true if values[i - 1024] == weakest),
        "max_negative_ids": sorted(i for i in negative if values[i - 1024] == strongest),
        "direct_set_ids": predicted,
        "direct_metrics": set_metrics(predicted, true),
    }


def set_metrics(predicted, truth):
    found, wanted = set(predicted), set(truth)
    tp = len(found & wanted)
    precision = tp / len(found) if found else float(not wanted)
    recall = tp / len(wanted) if wanted else 1.0
    return {
        "exact": found == wanted,
        "tp": tp,
        "fp": len(found - wanted),
        "fn": len(wanted - found),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "set_size": len(found),
    }


def paired_gate(control, treatment):
    """Pure predefined predicates; the historical 191 count never enters this gate."""
    return {
        "all_sources_and_selections_eligible": control["all_valid"] and treatment["all_valid"],
        "strict_exact_improvement": treatment["exact_count"] > control["exact_count"],
        "macro_f1_nonregression": treatment["f1"] >= control["f1"],
        "group_exact_nonregression": all(
            treatment["groups"][g] >= control["groups"][g] for g in GROUPS
        ),
        "strict_separation_improvement": treatment["separation_count"]
        > control["separation_count"],
    }


def training_evidence(run_dir, expected, environment, checkpoint_hash, original):
    """Authenticate final bytes and all saved training diagnostics without unpickling Torch."""
    run_dir = Path(run_dir)
    run = read(run_dir / "run.json")
    exact_json(run["config"], expected, "effective fresh training recipe differs")
    identity = run["identity"]
    check(identity["resolved_config_hash"] == canonical(expected), "fresh training config hash")
    check(
        identity["environment"] == environment and identity["seeds"] == [1729], "fresh runtime/seed"
    )
    for name in (
        "protocol_version",
        "graph_hash",
        "records_hash",
        "vocabulary_hash",
        "split_hash",
        "evaluator_version",
        "repetitions",
    ):
        check(
            identity[name] == original["identity"][name],
            "historical data/training semantics: " + name,
        )
    archive(run_dir / "source.zip", environment)
    final_path = run_dir / "checkpoint-final.pt"
    bind(final_path, checkpoint_hash)
    sidecar = read(final_path.with_suffix(".pt.json"))
    result = read(run_dir / "training-result.json")
    check(sidecar["global_step"] == result["global_step"] == 2000, "not fixed final checkpoint")
    check(
        sidecar["checkpoint_hash"] == result["checkpoint_hash"] == checkpoint_hash,
        "final byte identity",
    )
    check(Path(result["checkpoint"]).resolve() == final_path.resolve(), "training checkpoint path")
    exact_json(sidecar["experiment_identity"], identity, "checkpoint experiment identity")
    exact_json(result["identity"], identity, "training result identity")
    exact_json(sidecar["config"], expected["train"], "checkpoint train config")
    check(sidecar["split_hash"] == identity["split_hash"], "checkpoint split")
    exact_json(
        sidecar["corpus_identity"],
        {name: identity[name] for name in ("graph_hash", "records_hash", "vocabulary_hash")},
        "checkpoint corpus",
    )
    metadata = sidecar["training_metadata"]
    exact_json(metadata["model_config"], expected["model"], "checkpoint raw model config")
    check(
        metadata["objective"] == "causal-next-token-v1"
        and metadata["data_order"] == "sequential-v1"
        and metadata["batch_size"] == 32
        and metadata["grad_accum_steps"] == 1,
        "training contract",
    )
    enabled = expected["model"]["symmetric_margin_loss_weight"] > 0.0
    check(("symmetric_margin_objective" in metadata) == enabled, "margin descriptor presence")
    if enabled:
        check(metadata["symmetric_margin_objective"] == OBJECTIVE, "margin objective version")
    metrics_path = run_dir / "metrics.jsonl"
    bind(metrics_path, sha(metrics_path))
    history = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line
    ]
    exact_json(history, result["history"], "history/jsonl mismatch")
    cadence = expected["train"]["eval_every"]
    check(
        [row["step"] for row in history] == list(range(cadence, 2001, cadence)),
        "training diagnostic cadence",
    )
    check(
        history
        and history[-1]["train_loss"] == result["train_loss"] == metadata["train_loss"]
        and history[-1]["validation_loss"] == result["validation_loss"],
        "final training losses",
    )
    check(
        type(result["training_wall_seconds"]) in (int, float)
        and math.isfinite(result["training_wall_seconds"])
        and result["training_wall_seconds"] > 0,
        "training timing",
    )
    margin_keys = {
        "last_batch_symmetric_margin_loss",
        "last_batch_symmetric_margin_query_count",
        "last_batch_symmetric_margin_active_count",
        "validation_symmetric_margin_loss",
        "validation_symmetric_margin_query_count",
        "validation_symmetric_margin_active_count",
        "validation_symmetric_margin_active_fraction",
    }
    for row in history:
        check(
            all(type(value) in (int, float) and math.isfinite(value) for value in row.values()),
            "nonfinite training diagnostic",
        )
        check("validation_loss" in row and "last_batch_token_loss" in row, "missing token CE")
        if enabled:
            check(margin_keys <= row.keys(), "missing raw margin diagnostics")
            count = row["validation_symmetric_margin_query_count"]
            active = row["validation_symmetric_margin_active_count"]
            check(
                count == 222 and 0 <= active <= count and active == int(active),
                "validation margin counts",
            )
            check(
                row["validation_symmetric_margin_active_fraction"] == active / count
                and row["validation_symmetric_margin_loss"] >= 0,
                "validation margin aggregation receipt",
            )
            query_count = row["last_batch_symmetric_margin_query_count"]
            check(
                0 < query_count <= 32
                and query_count == int(query_count)
                and 0 <= row["last_batch_symmetric_margin_active_count"] <= query_count
                and row["last_batch_symmetric_margin_loss"] >= 0,
                "last microbatch margin diagnostics",
            )
        else:
            check(
                not any("symmetric_margin" in key for key in row),
                "disabled margin diagnostics present",
            )
    return run, sidecar, result


def pinned_input(path, manifests):
    resolved = Path(path).resolve()
    digests = {
        digest
        for manifest in manifests
        for name, digest in manifest.get("input_sha256", {}).items()
        if Path(name).resolve() == resolved
    }
    check(len(digests) == 1, f"missing or ambiguous historical input: {resolved}")
    bind(resolved, digests.pop())


def reference_evidence():
    old_dir = ROOT / "runs/national_dex_continuation_control_s1729_v1"
    original = read(old_dir / "run.json", ORIGINAL_SHA)
    check(canonical(original["config"]) == ORIGINAL_CONFIG_SHA, "original config pin")
    pair_dir = ROOT / "runs/learning/pair-unions-v1"
    pair = read(pair_dir / "summary.json", PAIR_SHA)
    audit = read(pair_dir / "independent-audit.json", PAIR_AUDIT_SHA)
    check(
        audit["summary_sha256"] == PAIR_SHA
        and audit["all_arithmetic_and_provenance_equal"] is True,
        "historical pair independent audit",
    )
    bind(pair_dir / "independent-audit.py", audit["script_sha256"])
    ref = read(pair_dir / "seed-1729.json", pair["report_sha256"]["seed-1729.json"])
    check(
        ref["seed"] == 1729 and ref["query_count"] == len(ref["responses"]) == 222,
        "reference coverage",
    )
    checkpoint = old_dir / "checkpoint-final.pt"
    bind(checkpoint, ref["checkpoint_hash"])
    sidecar, result = (
        read(checkpoint.with_suffix(".pt.json")),
        read(old_dir / "training-result.json"),
    )
    check(
        sidecar["checkpoint_hash"] == result["checkpoint_hash"] == ref["checkpoint_hash"]
        and sidecar["global_step"] == result["global_step"] == 2000,
        "original checkpoint receipts",
    )
    exact_json(original["identity"], ref["training_identity"], "original/pair training identity")
    exact_json(original["identity"], sidecar["experiment_identity"], "original checkpoint identity")
    exact_json(original["identity"], result["identity"], "original result identity")
    exact_json(
        sidecar["training_metadata"]["model_config"],
        original["config"]["model"],
        "original raw model",
    )
    archive(old_dir / "source.zip", original["identity"]["environment"])
    corpus = Path(original["config"]["data"]["corpus_manifest"]).parent
    for filename in ("manifest.json", "vocabulary.json", "records.jsonl"):
        pinned_input(corpus / filename, (pair, audit))
    graph = Path(original["config"]["data"]["graph_db"])
    pinned_input(graph, (pair, audit))
    manifest, vocabulary = read(corpus / "manifest.json"), read(corpus / "vocabulary.json")
    tokens = vocabulary["tokens"]
    check(
        len(tokens) == len(set(tokens)) == 2049
        and all(t.startswith("PKM_") for t in tokens[1024:]),
        "vocabulary/product alignment",
    )
    vocab_hash = hashlib.sha256(
        json.dumps(vocabulary, ensure_ascii=True, sort_keys=True).encode()
    ).hexdigest()
    check(
        vocab_hash == manifest["tokenizer_hash"] == original["identity"]["vocabulary_hash"],
        "vocab logical hash",
    )
    check(
        manifest["records_hash"]
        == original["identity"]["records_hash"]
        == sha(corpus / "records.jsonl"),
        "canonical corpus bytes/hash",
    )
    check(
        manifest["graph_hash"] == original["identity"]["graph_hash"]
        and manifest["protocol_version"] == original["identity"]["protocol_version"],
        "corpus identity",
    )
    data = original["config"]["data"]
    groups = diagnostic_groups(graph)
    assignments = {"train": [], "validation": [], "test": []}
    validation = []
    # Other partitions contribute only query identities to split authentication.
    # Their target, input_ids and labels fields are never accessed or measured.
    with (corpus / "records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            query = [record["subject"], record["dimension"]]
            key = f"{data['split_seed']}:{query[0]}:{query[1]}".encode()
            fraction = int.from_bytes(hashlib.sha256(key).digest(), "big") / float(1 << 256)
            if fraction < data["validation_query_fraction"]:
                partition = "validation"
            elif fraction < data["validation_query_fraction"] + data["test_query_fraction"]:
                partition = "test"
            else:
                partition = "train"
            assignments[partition].append(query)
            if partition == "validation":
                validation.append(
                    {
                        "subject": query[0],
                        "dimension": query[1],
                        "expected": record["targets"],
                        "group": groups[tuple(query)],
                    }
                )
    split_payload = {
        "algorithm": "sha256-threshold-v2",
        "seed": data["split_seed"],
        "validation_fraction": data["validation_query_fraction"],
        "test_fraction": data["test_query_fraction"],
        **assignments,
    }
    split_hash = hashlib.sha256(
        json.dumps(split_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    check(
        split_hash == ref["split_hash"] == original["identity"]["split_hash"],
        "split assignment hash",
    )
    check(
        len(validation) == 222 and len({(r["subject"], r["dimension"]) for r in validation}) == 222,
        "validation query uniqueness",
    )
    for observed, frozen in zip(validation, ref["responses"], strict=True):
        exact_json(
            observed,
            {key: frozen[key] for key in observed},
            "actual validation/reference alignment",
        )
    check(
        sidecar["training_metadata"]["record_count"] == len(assignments["train"]),
        "old train record count",
    )
    return original, ref, tokens, validation, len(assignments["train"])


def aggregate_compositions(rows):
    check(rows, "empty evaluation partition")
    values = [row["selected_metrics"] for row in rows]
    outputs = [row["composition"] for row in rows]
    sources = [source for answer in outputs for source in answer["source_paths"]]
    success = [
        answer["selected_source_eligible"] and not answer["fallback_no_valid_source"]
        for answer in outputs
    ]
    exact = sum(value["exact"] for value in values)
    return {
        "query_count": len(rows),
        **{
            key: math.fsum(value[key] for value in values) / len(rows)
            for key in ("precision", "recall", "f1")
        },
        "exact_set_accuracy": exact / len(rows),
        "exact_set_count": exact,
        "mean_set_size": math.fsum(value["set_size"] for value in values) / len(rows),
        "selected_source_eligibility_rate": sum(success) / len(rows),
        "failure_count": len(rows) - sum(success),
        "original_selection_count": sum(
            ok and answer["selected_kind"] == "original"
            for ok, answer in zip(success, outputs, strict=True)
        ),
        "pair_selection_count": sum(
            ok and answer["selected_kind"] == "pair_composition"
            for ok, answer in zip(success, outputs, strict=True)
        ),
        "source_path_count": len(sources),
        "source_path_protocol_valid_rate": sum(p["protocol_valid"] for p in sources) / len(sources),
        "source_path_termination_rate": sum(p["terminated"] for p in sources) / len(sources),
        "source_path_error_count": sum(p["error"] is not None for p in sources),
        "source_path_duplicate_count": sum(
            len(p["targets"]) != len(set(p["targets"])) for p in sources
        ),
    }


def head_totals(rows):
    check(rows, "empty head diagnostic partition")
    return {
        "query_count": len(rows),
        "strict_separation_count": sum(r["strict_separable"] for r in rows),
        "direct_exact_count": sum(r["direct_metrics"]["exact"] for r in rows),
        **{
            f"direct_{key}": math.fsum(r["direct_metrics"][key] for r in rows) / len(rows)
            for key in ("precision", "recall", "f1", "set_size")
        },
    }


def evaluation_evidence(report, arm, training, original, tokens, validation):
    check(
        report["complete"] is True
        and report["arm"] == arm
        and report["query_count"] == len(report["responses"]) == 222
        and not report["cleanup_errors"]
        and not report.get("error"),
        "incomplete evaluation",
    )
    expected = evaluation_recipe(effective_recipe(original, arm))
    exact_json(report["config"], expected, "evaluation recipe")
    check(report["config_hash"] == canonical(expected), "evaluation exact config hash")
    check(report["checkpoint_hash"] == training["checkpoint_hash"], "evaluation checkpoint hash")
    exact_json(report["training_identity"], training["identity"], "evaluation training identity")
    exact_json(
        report["corpus_identity"],
        {
            key: original["identity"][key]
            for key in ("graph_hash", "records_hash", "vocabulary_hash")
        },
        "eval corpus",
    )
    check(report["split_hash"] == original["identity"]["split_hash"], "eval split")
    exact_json(report["product_token_ids"], list(range(1024, 2049)), "head product mapping")
    check(report["head_reference_batch_count"] == 28, "reference batch count")
    numerical = report["numerical_settings"]
    check(
        numerical["parameter_dtype"] == "torch.float32"
        and numerical["autocast_cuda_enabled"] is False,
        "FP32 inference scope",
    )
    check(
        type(report["evaluation_wall_seconds"]) in (int, float)
        and math.isfinite(report["evaluation_wall_seconds"])
        and report["evaluation_wall_seconds"] > 0,
        "evaluation timing",
    )
    mapping = {token: i for i, token in enumerate(tokens)}
    rows = []
    for index, (record, observed) in enumerate(zip(validation, report["responses"], strict=True)):
        subject, dimension = record["subject"], record["dimension"]
        prompt = [mapping[token] for token in ("BOS", subject, dimension, "SAME", "ANSWER")]
        head = observed["symmetric_relation_logits"]
        answer = build_composition(
            observed["composition"]["source_paths"], head, subject, dimension, tokens
        )
        # Oracle labels enter only after the independent learned-score selection is fixed.
        truth = sorted(mapping[token] for token in record["expected"])
        diagnostics = membership(head, prompt[1], truth)
        success = answer["selected_source_eligible"] and not answer["fallback_no_valid_source"]
        selected = set_metrics(answer["selected_set_ids"] if success else [], truth)
        selected["exact"] = success and selected["exact"]
        row = {
            "index": index,
            "subject": subject,
            "dimension": dimension,
            "group": record["group"],
            "expected_set_ids": truth,
            "prompt_ids": prompt,
            "batch_index": index // 8,
            "batch_shape": [8 if index < 216 else 6, 5],
            "composition": answer,
            "symmetric_relation_logits": head,
            "selected_metrics": selected,
            **{
                key: diagnostics[key]
                for key in (
                    "direct_set_ids",
                    "direct_metrics",
                    "strict_separable",
                    "min_true_logit",
                    "max_negative_logit",
                    "gap",
                )
            },
        }
        exact_json(observed, row, f"{arm} independent row {index}")
        rows.append(row)
    rebuilt = {
        "metrics": aggregate_compositions(rows),
        "head_diagnostics": head_totals(rows),
        "groups": {
            group: {
                "metrics": aggregate_compositions([r for r in rows if r["group"] == group]),
                "head_diagnostics": head_totals([r for r in rows if r["group"] == group]),
            }
            for group in GROUPS
        },
    }
    for key, value in rebuilt.items():
        exact_json(report[key], value, arm + " " + key)
    return {**rebuilt, "responses": rows}


def final_gate(control, treatment):
    condensed = []
    for arm in (control, treatment):
        valid = all(
            all(
                p["terminated"]
                and p["protocol_valid"]
                and p["error"] is None
                and len(p["targets"]) == len(set(p["targets"]))
                for p in r["composition"]["source_paths"]
            )
            and all(s["source_eligible"] for s in r["composition"]["slots"])
            and r["composition"]["selected_source_eligible"]
            and not r["composition"]["fallback_no_valid_source"]
            for r in arm["responses"]
        )
        condensed.append(
            {
                "all_valid": valid,
                "exact_count": arm["metrics"]["exact_set_count"],
                "f1": arm["metrics"]["f1"],
                "separation_count": arm["head_diagnostics"]["strict_separation_count"],
                "groups": {g: arm["groups"][g]["metrics"]["exact_set_count"] for g in GROUPS},
            }
        )
    tests = paired_gate(*condensed)
    checks = dict(
        zip(
            (
                "all_sources_and_selected_eligible",
                "exact_count_strictly_improved",
                "macro_f1_nondecreased",
                "groups_exact_nondecreased",
                "strict_separation_strictly_improved",
            ),
            tests.values(),
            strict=True,
        )
    )
    gained, lost = [], []
    for a, b in zip(control["responses"], treatment["responses"], strict=True):
        keys = ("index", "subject", "dimension", "group", "expected_set_ids")
        exact_json([a[k] for k in keys], [b[k] for k in keys], "paired validation identity")
        query = {k: a[k] for k in keys[:-1]}
        if b["selected_metrics"]["exact"] and not a["selected_metrics"]["exact"]:
            gained.append(query)
        if a["selected_metrics"]["exact"] and not b["selected_metrics"]["exact"]:
            lost.append(query)
    return {
        "checks": checks,
        "numerical_gates_passed": all(checks.values()),
        "gained_exact_count": len(gained),
        "lost_exact_count": len(lost),
        "gained_exact": gained,
        "lost_exact": lost,
        "independent_audit_status": "required",
        "stage_one_accepted": False,
        "replication_authorized": False,
    }


def frozen_inventory(summary, path):
    expected = summary["live_source_config_sha256"]
    source_names = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")}
    source_names.update(("pyproject.toml", "uv.lock"))
    config_names = {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").rglob("*.yaml")}
    check(set(expected) == source_names | config_names, "live source/config inventory")
    for name, digest in expected.items():
        bind(ROOT / name, digest)
    for suffix, names in (("source.zip", source_names), ("configs.zip", config_names)):
        contents = zip_contents(path.with_suffix("." + suffix))
        check(set(contents) == names, "archive source/config inventory")
        check(
            all(hashlib.sha256(contents[n]).hexdigest() == expected[n] for n in names),
            "archive/live bytes",
        )
    check(summary["environment"]["source_archive_sha256"] == SOURCE_SHA, "prerequisite source pin")
    archive(path.with_suffix(".source.zip"), summary["environment"])


def prerequisite_evidence(summary, path):
    cpu = read(path.with_suffix(".cpu-receipt.json"), CPU_SHA)
    gpu = read(path.with_suffix(".gpu-smoke.json"), GPU_SHA)
    for receipt in (cpu, gpu):
        check(
            receipt["passed"] is True and receipt["source_archive_sha256"] == SOURCE_SHA,
            "prerequisite pass/source binding",
        )
        exact_json(receipt["environment"], summary["environment"], "prerequisite runtime")
        check(receipt["source_commit"] == summary["source_commit"], "prerequisite revision")
        for category in ("implementation_source_sha256", "config_files_sha256"):
            for name, digest in receipt[category].items():
                bind(ROOT / name, digest)
                check(
                    summary["live_source_config_sha256"][name] == digest,
                    "prerequisite live inventory",
                )
    check(
        cpu["gpu_used"] is False and cpu["experiment_training_executed"] is False,
        "CPU receipt scope",
    )
    check(
        all(r["exit_code"] == 0 for r in cpu["checks"]) and len(cpu["checks"]) == 4,
        "CPU prerequisite outcomes",
    )
    check(
        cpu["checks"][0]["passed"] == 818 and cpu["checks"][0]["skipped"] == 9,
        "pinned CPU suite receipt",
    )
    for name, digest in cpu["tests_sha256"].items():
        bind(ROOT / name, digest)
    comparison = read(
        ROOT / cpu["archived_source_comparison"]["path"],
        cpu["archived_source_comparison"]["sha256"],
    )
    bind(path.with_suffix(".cpu-comparison.json"), cpu["archived_source_comparison"]["sha256"])
    check(
        comparison["complete"] is True and comparison["all_bit_exact"] is True,
        "disabled archived-source comparison",
    )
    compare_dir = ROOT / "runs/learning/margin-default-contract-v1"
    bind(compare_dir / "capture.py", comparison["script_sha256"])
    bind(compare_dir / "original.pt", comparison["original_snapshot_sha256"])
    bind(compare_dir / "current.pt", comparison["current_snapshot_sha256"])
    bind(compare_dir / "seal-prerequisites.py", cpu["seal_script_sha256"])
    check(cpu["seal_script_sha256"] == gpu["seal_script_sha256"], "prerequisite seal script")
    check(
        gpu["synthetic"] is True and gpu["steps"] == 3 and gpu["real_corpus_used"] is False,
        "GPU smoke scope",
    )
    smoke = read(ROOT / gpu["raw_receipt"]["path"], gpu["raw_receipt"]["sha256"])
    bind(path.with_suffix(".gpu-smoke-raw.json"), gpu["raw_receipt"]["sha256"])
    check(
        smoke["complete"] is True and [r["step"] for r in smoke["steps"]] == [1, 2, 3],
        "GPU smoke completion",
    )
    check(
        all(
            all(
                type(r[k]) in (float, int) and math.isfinite(r[k])
                for k in ("loss", "margin", "active_queries")
            )
            for r in smoke["steps"]
        ),
        "GPU smoke finite outputs",
    )
    bind(compare_dir / "gpu-smoke.py", smoke["script_sha256"])
    for name, digest in smoke["input_sha256"].items():
        bind(ROOT / name, digest)
    return cpu["evidence_kind"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    path = args.summary.resolve()
    output = path.parent / "independent-audit.json"
    check(not output.exists(), "refusing existing independent audit")
    check(path.is_file(), "complete primary summary missing; no audit output")
    check("torch" not in sys.modules, "auditor must remain stdlib-only")
    own_sha = sha(__file__)
    summary = read(path)
    check(
        summary["complete"] is True and summary["phase"] == "complete" and not summary.get("error"),
        "incomplete primary campaign",
    )
    check(
        summary["campaign_version"] == "plm-symmetric-margin-stage1-v1"
        and summary["seed"] == 1729
        and summary["stage_one_accepted"] is False
        and summary["independent_audit_status"] == "required",
        "declared stage/acceptance boundary",
    )
    for key, digest in {
        "plan_sha256": PLAN_SHA,
        "original_recipe_sha256": ORIGINAL_SHA,
        "historical_training_config_hash": ORIGINAL_CONFIG_SHA,
        "historical_pair_summary_sha256": PAIR_SHA,
    }.items():
        check(summary[key] == digest, "pinned campaign input: " + key)
    for name, digest in summary["input_sha256"].items():
        bind(name, digest)
    snapshot_names = {
        "script.py",
        "plan.md",
        "source.zip",
        "configs.zip",
        "original-recipe.json",
        "cpu-receipt.json",
        "gpu-smoke.json",
        "cpu-comparison.json",
        "gpu-smoke-raw.json",
    }
    check(set(summary["snapshot_sha256"]) == snapshot_names, "campaign snapshot inventory")
    for suffix, digest in summary["snapshot_sha256"].items():
        bind(path.with_suffix("." + suffix), digest)
    check(
        summary["snapshot_sha256"]["original-recipe.json"] == ORIGINAL_SHA
        and summary["snapshot_sha256"]["plan.md"] == PLAN_SHA
        and summary["snapshot_sha256"]["source.zip"] == SOURCE_SHA
        and summary["snapshot_sha256"]["cpu-receipt.json"] == CPU_SHA
        and summary["snapshot_sha256"]["gpu-smoke.json"] == GPU_SHA,
        "snapshot hash pins",
    )
    check(summary["snapshot_sha256"]["script.py"] == RUNNER_SHA, "frozen runner pin")
    bind(ROOT / "scripts/run_symmetric_margin_screen.py", summary["snapshot_sha256"]["script.py"])
    bind(ROOT / "docs/experiments/2026-09-25-symmetric-margin-plan.md", PLAN_SHA)
    for filename, digest in summary["report_sha256"].items():
        check(
            Path(filename).name == filename and filename != path.name and filename != output.name,
            "unsafe/circular report manifest",
        )
        bind(path.parent / filename, digest)
    required_reports = {
        f"{phase}-{arm}.json"
        for arm in ARMS
        for phase in ("training-config", "training", "evaluation")
    }
    required_reports.add("training-seal.json")
    check(required_reports <= summary["report_sha256"].keys(), "missing arm artifacts")
    seal_path = path.parent / "training-seal.json"
    seal = read(seal_path)
    exact_json(
        seal,
        {
            "both_trainings_completed_before_evaluation": True,
            "source_archive_sha256": SOURCE_SHA,
            "script_sha256": RUNNER_SHA,
            "training_receipt_sha256": {
                arm: sha(path.parent / f"training-{arm}.json") for arm in ARMS
            },
        },
        "both fresh trainings sealed before evaluation",
    )
    check(
        summary["input_sha256"].get(str(seal_path)) == sha(seal_path),
        "training seal absent from bound input manifest",
    )
    check(not summary["cleanup_errors"], "campaign cleanup failures")
    frozen_inventory(summary, path)
    cpu_origin = prerequisite_evidence(summary, path)
    original, historical, tokens, validation, train_count = reference_evidence()
    check(
        summary["historical_checkpoint_sha256"] == historical["checkpoint_hash"],
        "historical checkpoint pin",
    )
    check(
        summary["historical_pair_seed_sha256"]
        == sha(ROOT / "runs/learning/pair-unions-v1/seed-1729.json")
        and summary["historical_pair_exact_context"]
        == historical["comparison"]["pair_selector"]["exact_count"]
        == 191,
        "historical contextual result",
    )
    check(
        [r["arm"] for r in summary["training"]] == list(ARMS)
        and [r["arm"] for r in summary["evaluation"]] == list(ARMS),
        "arm order",
    )
    rebuilt, checkpoints = [], []
    settings = None
    for index, arm in enumerate(ARMS):
        expected = effective_recipe(original, arm)
        exact_json(
            read(path.parent / f"training-config-{arm}.json"), expected, "saved effective recipe"
        )
        receipt = read(path.parent / f"training-{arm}.json")
        exact_json(receipt, summary["training"][index], "training receipt/summary")
        exact_json(receipt["config"], expected, "training receipt recipe")
        check(
            receipt["config_hash"] == canonical(expected)
            and receipt["global_step"] == 2000
            and receipt["arm"] == arm
            and receipt["run_name"] == RUN_NAMES[index],
            "training receipt identity",
        )
        directory = ROOT / expected["paths"]["run_root"] / expected["run_name"]
        check(
            Path(receipt["checkpoint"]).resolve() == (directory / "checkpoint-final.pt").resolve(),
            "fresh checkpoint directory",
        )
        for name, digest in receipt["artifact_sha256"].items():
            bind(name, digest)
        check(
            {Path(name).resolve() for name in receipt["artifact_sha256"]}
            == {
                directory / name
                for name in (
                    "run.json",
                    "training-result.json",
                    "checkpoint-final.pt",
                    "checkpoint-final.pt.json",
                    "metrics.jsonl",
                    "source.zip",
                )
            },
            "training artifact inventory",
        )
        run, sidecar, training = training_evidence(
            directory, expected, summary["environment"], receipt["checkpoint_hash"], original
        )
        check(
            sidecar["training_metadata"]["record_count"] == train_count, "new training record count"
        )
        check(run["identity"]["source_commit"] == summary["source_commit"], "training revision")
        exact_json(receipt["identity"], run["identity"], "training receipt identity")
        exact_json(
            receipt["training_history"], training["history"], "training receipt full history"
        )
        check(
            receipt["training_wall_seconds"] == training["training_wall_seconds"],
            "training timing receipt",
        )
        checkpoints.append(receipt["checkpoint_hash"])
        report = read(path.parent / f"evaluation-{arm}.json")
        exact_json(
            {k: v for k, v in report.items() if k != "responses"},
            summary["evaluation"][index],
            "evaluation report/summary",
        )
        rebuilt.append(evaluation_evidence(report, arm, receipt, original, tokens, validation))
        if settings is None:
            settings = report["numerical_settings"]
        exact_json(report["numerical_settings"], settings, "matched inference numerical settings")
        print(
            f"{arm}: final receipts, 222 queries, 888 sources and 2220 slots verified",
            flush=True,
        )
    check(
        len(set(checkpoints)) == 2 and historical["checkpoint_hash"] not in checkpoints,
        "fresh checkpoint byte identity",
    )
    gate = final_gate(*rebuilt)
    exact_json(summary["gate"], gate, "predeclared paired gate")
    frozen_inventory(summary, path)
    check(all(sha(name) == digest for name, digest in INPUTS.items()), "input drift during audit")
    check(sha(__file__) == own_sha, "auditor changed during execution")
    result = {
        "complete": True,
        "all_recomputed_outputs_equal": True,
        "audit_passed": True,
        "stage_one_accepted": False,
        "replication_authorized": False,
        "acceptance_authority": "owner reviews the independent audit and predeclared gate",
        "summary_sha256": sha(path),
        "script_sha256": own_sha,
        "plan_sha256": PLAN_SHA,
        "query_count": 444,
        "distinct_validation_queries": 222,
        "seed": 1729,
        "source_path_count": 1776,
        "slot_count": 4440,
        "gate": gate,
        "arms": [
            {"arm": arm, **{k: value[k] for k in ("metrics", "head_diagnostics", "groups")}}
            for arm, value in zip(ARMS, rebuilt, strict=True)
        ],
        "input_sha256": dict(sorted(INPUTS.items())),
        "limitations": [
            "Only seed 1729 fresh control/treatment is audited; "
            "no untested-seed or deployment claim.",
            "Checkpoint bytes and authenticated receipts are checked "
            "without unpickling Torch payloads.",
            "Saved FP32 logits and training diagnostics are authenticated, "
            "not neurally regenerated.",
            "CPU prerequisite provenance: " + cpu_origin,
            "Protected partition query identities only authenticate the split; "
            "its target fields are not measured.",
            "Historical 191 is contextual only; the gate compares the fresh matched control.",
        ],
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "stage_one_accepted": False,
                "numerical_gates_passed": gate["numerical_gates_passed"],
                "audit_sha256": sha(output),
            }
        )
    )


if __name__ == "__main__":
    main()
