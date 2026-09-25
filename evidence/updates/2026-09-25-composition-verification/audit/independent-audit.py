"""Independent, read-only stdlib audit of complete shape-aware v2 evidence.

No evaluator/model/helper imports. Authentication routines retain the prior audit
contract; protocol parsing, selection, metrics and transport checks are independent.
Overall integration acceptance is deliberately left to the campaign owner.
"""

import argparse
import copy
import hashlib
import json
import math
import re
import sqlite3
import struct
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = (1729, 1730, 1731)
PRIOR_SHA = "780f2e219d55164acefaa3b29a6b25e803154baa057d64a5002f05733a4d99de"
PRIOR_AUDIT_SHA = "96cce95af7cfc5001a1b94ed5df34d30eeb729ae04bd7c2141cba53b47e1c3ad"
PAIR_SHA = "4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992"
PAIR_AUDIT_SHA = "13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057"
SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"
LEGACY = BASE + "+first4-symmetric-set-logit-sum-v1"
POLICY = BASE + "+first4-pair6-symmetric-set-logit-sum-v1"
ORDER = ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
INPUTS = {}


V1_DIR = ROOT / "runs/learning/pair-composition-integration-v1"


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


def catalog(path):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as database:
        return {
            key: {"key": key, "label": label}
            for key, label in database.execute(
                "SELECT key,label FROM nodes WHERE kind='product' AND status='active'"
            )
        }


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


def config(run, batch, composition=True, reranking=True, bound=507):
    raw = run["config"]
    check(canonical(raw) == run["identity"]["resolved_config_hash"], "raw historical hash")
    check(
        not {"pair_set_composition", "symmetric_set_reranking"} & raw["eval"].keys(), "old schema"
    )
    result = copy.deepcopy(raw)
    result["eval"].update(
        pair_set_composition=composition,
        symmetric_set_reranking=reranking,
        first_target_guidance_alpha=16.0,
        constrained_decoding=True,
        use_kv_cache=True,
        prevent_repeated_targets=True,
        generation_batch_size=batch,
        max_new_tokens=bound,
    )
    return result


def remove_default(payload, field):
    lines = payload.splitlines(keepends=True)
    matches = [line for line in lines if line.startswith(field.encode() + b":")]
    check(
        len(matches) == 1
        and re.fullmatch(
            re.escape(field.encode()) + rb":[ \t]+false[ \t]*(?:#[^\r\n]*)?\r?\n?", matches[0]
        ),
        "nondefault YAML migration",
    )
    return b"".join(line for line in lines if line != matches[0])


def zip_contents(path):
    with zipfile.ZipFile(path) as source:
        names = source.namelist()
        check(len(names) == len(set(names)), "duplicate archive members")
        return {name: source.read(name) for name in names}


def provenance(summary):
    prior_dir = ROOT / "runs/learning/set-reranking-integration-v1"
    pair_dir = ROOT / "runs/learning/pair-unions-v1"
    prior = read(prior_dir / "summary.json", PRIOR_SHA)
    audit = read(prior_dir / "independent-audit.json", PRIOR_AUDIT_SHA)
    pair = read(pair_dir / "summary.json", PAIR_SHA)
    pair_audit = read(pair_dir / "independent-audit.json", PAIR_AUDIT_SHA)
    check(
        prior["all_outputs_equal"] and audit["complete"] and audit["all_exact_parity"],
        "prior acceptance",
    )
    check(
        audit["summary_sha256"] == PRIOR_SHA
        and pair_audit["summary_sha256"] == PAIR_SHA
        and pair_audit["all_arithmetic_and_provenance_equal"],
        "audit binding",
    )
    for key, value in {
        "prior_summary_sha256": PRIOR_SHA,
        "prior_audit_sha256": PRIOR_AUDIT_SHA,
        "historical_pair_sha256": PAIR_SHA,
        "historical_pair_audit_sha256": PAIR_AUDIT_SHA,
    }.items():
        check(summary[key] == value, "reference pin differs")
    before = archive(prior_dir / "summary.source.zip", prior["evaluator_environment"])
    check(
        summary["evaluator_environment"]["source_archive_sha256"] == SOURCE_SHA, "launch source pin"
    )
    after = archive(V1_DIR / "summary.source.zip", summary["evaluator_environment"])
    check(
        summary["prior_inference_environment"] == prior["evaluator_environment"],
        "prior environment",
    )
    check(
        summary["historical_training_environment"] == pair["historical_model_environment"],
        "training environment",
    )
    check(summary["source_commit"] == prior["source_commit"], "commit changed")
    for key, value in prior["evaluator_environment"].items():
        if key not in {"source_tree_sha256", "source_archive_sha256"}:
            check(summary["evaluator_environment"][key] == value, "runtime changed")
    allowed = {
        "src/plm/" + name
        for name in (
            "cli.py",
            "config.py",
            "configuration.py",
            "evaluation/set_composition.py",
            "serving/native.py",
            "serving/set_composition.py",
            "serving/set_reranking.py",
            "serving/web.py",
        )
    }
    changes = [
        {
            "path": name,
            "historical_sha256": hashlib.sha256(before[name]).hexdigest()
            if name in before
            else None,
            "inference_sha256": hashlib.sha256(after[name]).hexdigest() if name in after else None,
        }
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    ]
    check(
        changes == summary["source_boundary"] and {c["path"] for c in changes} == allowed,
        "source boundary",
    )
    for name, payload in after.items():
        check((ROOT / name).read_bytes() == payload, "live source changed during campaign")
    for suffix, key in (
        ("script.py", "script_sha256"),
        ("plan.md", "plan_sha256"),
        ("helper.py", "helper_sha256"),
        ("configs.zip", "configs_archive_sha256"),
    ):
        bind(V1_DIR / f"summary.{suffix}", summary[key])
    check(
        summary["helper_sha256"] == prior["script_sha256"],
        "custom helper not accepted prior harness",
    )
    bind(prior_dir / "summary.configs.zip", prior["configs_archive_sha256"])
    old_configs = zip_contents(prior_dir / "summary.configs.zip")
    new_configs = zip_contents(V1_DIR / "summary.configs.zip")
    check(old_configs.keys() == new_configs.keys(), "config inventory changed")
    expected_boundary = {}
    for name, payload in new_configs.items():
        check((ROOT / name).read_bytes() == payload, "live config changed")
        if name == "configs/eval/ranking.yaml":
            check(
                remove_default(payload, "pair_set_composition") == old_configs[name],
                "config changed beyond default",
            )
            expected_boundary[name] = {
                "historical_sha256": hashlib.sha256(old_configs[name]).hexdigest(),
                "current_sha256": hashlib.sha256(payload).hexdigest(),
                "inserted_default": "pair_set_composition: false",
            }
        else:
            check(payload == old_configs[name], "unrelated config changed")
    check(summary["config_boundary"] == expected_boundary, "config boundary receipt")
    for path, digest in summary["input_sha256"].items():
        bind(path, digest)
    archives = {}
    for path, environment in (
        (prior_dir / "summary.source.zip", prior["evaluator_environment"]),
        (pair_dir / "summary.source.zip", prior["evaluator_environment"]),
        (
            ROOT / "runs/learning/set-reranking-v1/summary.source.zip",
            pair["historical_model_environment"],
        ),
    ):
        archives[str(path)] = archive(path, environment)
    expected_archived = []
    for manifest in (prior, pair, audit, pair_audit):
        for name, digest in manifest["input_sha256"].items():
            path = (ROOT / name).resolve()
            try:
                relative = path.relative_to(ROOT).as_posix()
            except ValueError:
                relative = ""
            archive_path = None
            if relative.startswith("src/") or relative in {"pyproject.toml", "uv.lock"}:
                archive_path = next(
                    (
                        p
                        for p, members in archives.items()
                        if relative in members
                        and hashlib.sha256(members[relative]).hexdigest() == digest
                    ),
                    None,
                )
                check(archive_path is not None, "historical source unavailable")
            elif relative.startswith("configs/"):
                payload = old_configs[relative]
                if (
                    hashlib.sha256(payload).hexdigest() != digest
                    and relative == "configs/eval/ranking.yaml"
                ):
                    payload = remove_default(payload, "symmetric_set_reranking")
                check(
                    hashlib.sha256(payload).hexdigest() == digest, "historical config bytes differ"
                )
                archive_path = str(prior_dir / "summary.configs.zip")
            if archive_path:
                expected_archived.append({"path": name, "sha256": digest, "archive": archive_path})
            else:
                bind(path, digest)
    check(
        summary["historical_archived_inputs"] == expected_archived,
        "historical input bridge differs",
    )
    for directory, manifest in ((prior_dir, prior), (pair_dir, pair)):
        for name, digest in manifest["report_sha256"].items():
            check(Path(name).name == name, "unsafe reference report path")
            bind(directory / name, digest)
        for suffix, key in (("script.py", "script_sha256"), ("plan.md", "plan_sha256")):
            bind(directory / f"summary.{suffix}", manifest[key])
    bind(prior_dir / "independent-audit.py", audit["script_sha256"])
    bind(pair_dir / "independent-audit.py", pair_audit["script_sha256"])
    bind(pair_dir / "summary.helper.py", pair["helper_sha256"])
    return prior, pair


PLAN_SHA = "0e406e2515c582ba5f4e8f3897b58e7daa6c27c96711913ef5d335f7eef2861f"
V1_SHA = "7abc3a99ea8aa0e95749da2b379893572f7f76643a53ab98f8242c2c77da7042"
FAILURE_SHA = "65ffee3f6035943cf07210f371de061e460f0a554619cb9fbd56783aab1c0253"
DIAG_SHA = "d263db1e956064b1f5c597369f5c82b95a7edc2fca55901d98a2f72e4a01557b"
DIAG_AUDIT_SHA = "6bc483713c5dd4511a3bdbc455649435f6da85a63ef8756bb6998573e90c44ee"
CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
HTTP_SHA = "e6759659c0eec1130088c1324c8457cc2c0bf814af7773c05a8f6b251c421e87"
OFFLINE_SHA = (
    "c7ad1db44d3c7e0387369b1ffc982d5e7b96afb046c4f8a68111b235ae5a3f67",
    "932ad33ad4d0b00e29cd7d64966b3f0d5820df91764d9ba1188a22cdd84e1659",
    "cd713d451a1235f7ad73d8b524acbf3c250b482f78e84c2cfd08f778e3c5ae73",
)
IDENTITY = ("checkpoint_hash", "training_identity", "corpus_identity", "split_hash")


def value_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def inventory(source_path, configs_path):
    """Reject both changed bytes and unarchived added source/config files."""
    inventories = (
        (
            source_path,
            {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")}
            | {"pyproject.toml", "uv.lock"},
        ),
        (
            configs_path,
            {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").rglob("*.yaml")},
        ),
    )
    for path, names in inventories:
        members = zip_contents(path)
        check(set(members) == names, "live/archive inventory differs")
        for name, payload in members.items():
            check((ROOT / name).read_bytes() == payload, f"live/archive bytes differ: {name}")


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
        check(unique, "unique decoding emitted a repeated target")
        eligible = parsed["terminated"] and parsed["protocol_valid"] and unique
        # Failed sources retain raw membership; removing SUBJECT here fabricates evidence.
        source_sets.append(set(ids) - {tokens.index(subject)} if eligible else set(ids))
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


def scoreless(value):
    result = copy.deepcopy(value)
    for slot in result["slots"]:
        slot.pop("score")
    return result


def filtering(targets, subject, directives=None):
    directives = directives or {}
    counts = dict.fromkeys(
        ("removed_subject", "removed_ignored", "removed_duplicates", "removed_by_limit"), 0
    )
    chosen = []
    for token in targets:
        if token == subject:
            counts["removed_subject"] += 1
        elif token in directives.get("ignore", []):
            counts["removed_ignored"] += 1
        elif token in chosen:
            counts["removed_duplicates"] += 1
        else:
            chosen.append(token)
    limit = directives.get("limit")
    if limit is not None:
        check(type(limit) is int and limit >= 0, "invalid filter limit")
        counts["removed_by_limit"] = max(len(chosen) - limit, 0)
        chosen = chosen[:limit]
    return chosen, counts


def metrics(rows):
    check(rows, "empty metric partition")
    return {
        "query_count": len(rows),
        "exact_count": sum(r["exact_count"] for r in rows),
        **{
            k: math.fsum(r[k] for r in rows) / len(rows)
            for k in ("precision", "recall", "f1", "set_size", "selected_source_eligible")
        },
    }


def measure(ref, answer, tokens):
    successful = answer["selected_source_eligible"] and not answer["fallback_no_valid_source"]
    prediction = {tokens[i] for i in answer["selected_set_ids"]} if successful else set()
    truth = set(ref["expected"])
    check(truth, "unexpected empty teacher set")
    overlap = len(prediction.intersection(truth))
    return {
        "precision": overlap / len(prediction) if prediction else 0.0,
        "recall": overlap / len(truth),
        "f1": 2 * overlap / (len(prediction) + len(truth)),
        "exact_count": int(successful and prediction == truth),
        "set_size": len(prediction),
        "selected_source_eligible": int(successful),
    }


def finite_timing(body):
    check(
        body["timing"]
        and all(
            type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in body["timing"].values()
        ),
        "invalid timing",
    )


def hydration(body, targets, ref, products, directives=None):
    targets, counts = filtering(targets, ref["subject"], directives)
    expected = {
        "subject": products[ref["subject"]],
        "dimension": ref["dimension"],
        "mode": "SAME",
        "targets": [products[token] for token in targets],
    }
    check(body["result"] == expected and body["postprocess"] == counts, "hydration/filter counts")
    finite_timing(body)


def composition_body(body, answer, ref, item, products, directives=None):
    extras = {
        "result",
        "postprocess",
        "checkpoint_hash",
        "corpus_identity",
        "split_hash",
        "graph_hash",
        "inference_config_hash",
        "timing",
    }
    check(body.keys() == answer.keys() | extras, "HTTP fabricated/missing top-level fields")
    check({key: body[key] for key in answer} == answer, "HTTP complete composition mismatch")
    for key in ("checkpoint_hash", "corpus_identity", "split_hash"):
        check(body[key] == item["reference"][key], "HTTP identity mismatch")
    check(body["graph_hash"] == item["reference"]["corpus_identity"]["graph_hash"], "HTTP graph")
    check(body["inference_config_hash"] == canonical(config(item["run"], 1)), "HTTP config")
    hydration(
        body, [item["tokens"][i] for i in answer["selected_set_ids"]], ref, products, directives
    )


def deployment_receipt(report, item, environment, enabled=True, bound=507):
    saved = read(report["deployment_receipt"])
    effective = config(item["run"], 1, enabled, bound=bound)
    check(saved["config"] == effective, "deployment exact config")
    check(saved["inference_config_hash"] == canonical(effective), "deployment config hash")
    check(
        saved["training_identity"] == item["reference"]["training_identity"], "deployment training"
    )
    check(saved["source_commit"] == item["run"]["identity"]["source_commit"], "deployment commit")
    expected = {
        "pair_set_composition": enabled,
        "set_composition_policy": POLICY if enabled else None,
        "decoding": LEGACY,
        "first_target_guidance_alpha": 16.0,
        "symmetric_set_reranking": True,
        "constrained_decoding": True,
        "use_kv_cache": True,
        "prevent_repeated_targets": True,
        "generation_batch_size": 1,
        "max_new_tokens": bound,
        "environment": environment,
        **{k: item["reference"][k] for k in ("checkpoint_hash", "corpus_identity", "split_hash")},
    }
    check(
        report["server_stopped"] is True and report["health"]["ready"] is True, "server lifecycle"
    )
    check(report["health"] == {"ready": True, **report["deployment"]}, "health receipt")
    for key, value in report["deployment"].items():
        check(saved[key] == value, "deployment/file disagreement")
    for key, value in expected.items():
        check(saved[key] == report["health"][key] == value, f"deployment policy: {key}")
    bind(saved["checkpoint"], item["reference"]["checkpoint_hash"])
    bind(saved["snapshot"], saved["snapshot_sha256"])
    archive(Path(report["deployment_receipt"]).parent / "source.zip", environment)
    products = catalog(saved["snapshot"])
    check(products == catalog(item["run"]["config"]["data"]["graph_db"]), "snapshot catalog")
    return products


def offline_audit(seed, prior, pair, v1):
    prior_dir = ROOT / "runs/learning/set-reranking-integration-v1"
    reference = read(
        ROOT / f"runs/learning/pair-unions-v1/seed-{seed}.json",
        pair["report_sha256"][f"seed-{seed}.json"],
    )
    offline = read(V1_DIR / f"offline-{seed}.json", OFFLINE_SHA[SEEDS.index(seed)])
    old = read(prior_dir / f"offline-{seed}.json", prior["report_sha256"][f"offline-{seed}.json"])
    old_http = read(prior_dir / f"http-{seed}.json", prior["report_sha256"][f"http-{seed}.json"])
    checkpoint = ROOT / f"runs/national_dex_continuation_control_s{seed}_v1/checkpoint-final.pt"
    bind(checkpoint, reference["checkpoint_hash"])
    run, sidecar, training = (
        read(checkpoint.parent / "run.json"),
        read(checkpoint.with_suffix(".pt.json")),
        read(checkpoint.parent / "training-result.json"),
    )
    check(
        run["identity"]
        == sidecar["experiment_identity"]
        == training["identity"]
        == reference["training_identity"],
        "training identity chain",
    )
    check(
        sidecar["checkpoint_hash"] == training["checkpoint_hash"] == reference["checkpoint_hash"],
        "checkpoint identity chain",
    )
    check(sidecar["training_metadata"]["model_config"] == run["config"]["model"], "trained heads")
    check(offline["config"] == config(run, 8), "offline config recipe")
    check(offline["inference_config_hash"] == canonical(config(run, 8)), "offline config hash")
    check(
        offline["historical_training_config_hash"] == run["identity"]["resolved_config_hash"]
        and offline["inserted_training_defaults"]
        == ["eval.symmetric_set_reranking=false", "eval.pair_set_composition=false"],
        "migration",
    )
    check(
        offline["disabled_composition_config_hash"] == canonical(config(run, 8, False))
        and offline["disabled_greedy_config_hash"] == canonical(config(run, 8, False, False)),
        "disabled config hashes",
    )
    check(
        offline["query_count"] == 222
        and offline["candidate_count"] == 888
        and offline["slot_count"] == 2220
        and offline["score_tolerance"] == 0
        and offline["exact_parity"] is True
        and not offline["mismatches"]
        and not offline["cleanup_errors"],
        "offline completeness",
    )
    for report in (reference, offline, old, old_http):
        check(report["seed"] == seed and len(report["responses"]) == 222, "seed/coverage")
        check(all(report[key] == reference[key] for key in IDENTITY), "historical identity")
    manifest_path = Path(run["config"]["data"]["corpus_manifest"])
    manifest = read(manifest_path)
    tokens = read(manifest_path.parent / "vocabulary.json")["tokens"]
    check(
        len(tokens) == 2049
        and len(set(tokens)) == 2049
        and all(t.startswith("PKM_") for t in tokens[1024:]),
        "vocabulary mapping",
    )
    # Corpus/split file hashes are authenticated through the immutable input manifests.
    check(manifest is not None, "missing corpus manifest")
    labels = diagnostic_groups(run["config"]["data"]["graph_db"])
    measured, groups, seen = [], {}, set()
    for ref, row, previous in zip(
        reference["responses"], offline["responses"], old["responses"], strict=True
    ):
        key = (ref["subject"], ref["dimension"])
        check(key not in seen, "duplicate validation query")
        seen.add(key)
        check(
            (row["subject"], row["dimension"], row["expected"]) == (*key, ref["expected"]),
            "offline partition/order/labels",
        )
        check(
            (previous["subject"], previous["dimension"], previous["expected"])
            == (*key, ref["expected"]),
            "legacy partition/order",
        )
        check(row["group"] == ref["group"] == labels[key], "graph-derived subgroup")
        paths = [
            parse_path(
                p["token_ids"], *key, tokens, BASE if rank == 1 else BASE + f"+first-rank{rank}-v1"
            )
            for rank, p in enumerate(ref["source_paths"], 1)
        ]
        answer = build_composition(paths, ref["symmetric_relation_logits"], *key, tokens)
        check(
            {k: answer[k] for k in ref["selection"]} == ref["selection"], "frozen pool arithmetic"
        )
        check(row["composition"] == answer and not row["mismatches"], "offline full composition")
        check(
            row["selected_set_keys"]
            == ref["selected_set_keys"]
            == [tokens[t] for t in answer["selected_set_ids"]],
            "offline set keys",
        )
        for name, descriptor, saved in (
            ("disabled_composition", LEGACY, previous["reranked"]["selected"]),
            ("disabled_greedy", BASE, previous["disabled_greedy"]),
        ):
            check(
                row[name] == saved == parse_path(saved["token_ids"], *key, tokens, descriptor),
                "legacy raw parity/grammar",
            )
        metric = measure(ref, answer, tokens)
        measured.append(metric)
        groups.setdefault(labels[key], []).append(metric)
    totals = metrics(measured)
    grouped = {name: metrics(rows) for name, rows in groups.items()}
    check(totals == offline["metrics"] == reference["comparison"]["pair_selector"], "macro metrics")
    check(grouped == offline["groups"], "offline grouped metrics")
    check(
        all(grouped[g] == reference["groups"][g]["pair_selector"] for g in grouped),
        "reference groups",
    )
    check(totals["exact_count"] == (191, 192, 186)[SEEDS.index(seed)], "declared exact counts")
    reuse = {
        "seed": seed,
        "evidence_origin": "reused",
        "query_count": 222,
        "source_paths": 888,
        "slots": 2220,
        "metrics": totals,
        "groups": grouped,
        "report": str(V1_DIR / f"offline-{seed}.json"),
        "report_sha256": OFFLINE_SHA[SEEDS.index(seed)],
    }
    return {
        "seed": seed,
        "reference": reference,
        "offline": offline,
        "old_http": old_http,
        "run": run,
        "tokens": tokens,
        "metrics": measured,
        "reuse": reuse,
    }


def serial_audit(path, item, environment, diagnosis):
    serial = read(path)
    seed = item["seed"]
    config_value = config(item["run"], 1)
    config_digest = canonical(config_value)
    check(
        serial["complete"] is True
        and serial["evidence_origin"] == "fresh"
        and serial["seed"] == seed
        and serial["query_count"] == 222
        and serial["reference_forward_count"] == 222
        and serial["repeat_forward_count"] == 8
        and len(serial["responses"]) == 222,
        "serial incomplete",
    )
    check(not serial.get("error") and not serial.get("cleanup_error"), "serial error receipt")
    check(
        serial["config"] == config_value and serial["config_hash"] == config_digest,
        "serial historical/effective config",
    )
    check(
        serial["environment"] == environment
        and serial["source_archive_sha256"] == SOURCE_SHA
        and serial["runtime_environment_sha256"] == value_sha(environment),
        "serial runtime",
    )
    check(
        serial["numerical_settings"]
        == {
            **diagnosis["torch_settings"],
            "autocast_cpu_enabled": False,
            "autocast_cuda_enabled": False,
        },
        "serial numerical settings",
    )
    check(
        serial["product_token_ids"] == list(range(1024, 2049)) and serial["batch_shape"] == [1, 5],
        "serial layout",
    )
    check(all(serial[k] == item["reference"][k] for k in IDENTITY), "serial provenance")
    check(math.isfinite(serial["wall_seconds"]) and serial["wall_seconds"] >= 0, "serial timing")
    for index, (row, ref, offline) in enumerate(
        zip(
            serial["responses"],
            item["reference"]["responses"],
            item["offline"]["responses"],
            strict=True,
        )
    ):
        query = (ref["subject"], ref["dimension"])
        check((row["subject"], row["dimension"]) == query, "serial partition/order")
        prompt = [item["tokens"].index(t) for t in ("BOS", *query, "SAME", "ANSWER")]
        check(
            row["prompt_ids"] == prompt and row["seed"] == seed and row["batch_shape"] == [1, 5],
            "serial observation shape/prompt",
        )
        check(
            row["checkpoint_hash"] == item["reference"]["checkpoint_hash"]
            and row["config_hash"] == config_digest
            and row["source_archive_sha256"] == SOURCE_SHA
            and row["runtime_environment_sha256"] == value_sha(environment),
            "serial row identity",
        )
        answer = build_composition(
            offline["composition"]["source_paths"], row["logits"], *query, item["tokens"]
        )
        check(row["composition"] == answer, "serial independently derived sums/ties")
        check(scoreless(answer) == scoreless(offline["composition"]), "cross-shape nonscore output")
        differences = [
            s["score"] - b["score"]
            for s, b in zip(answer["slots"], offline["composition"]["slots"], strict=True)
        ]
        check(row["cross_shape_score_differences"] == differences, "score difference diagnostic")
        if seed == 1729:
            check(
                row["logits"] == diagnosis["responses"][index]["serial_logits"],
                "seed 1729 diagnosed B1 vector",
            )
    check(
        serial["repeat_logits"] == [r["logits"] for r in serial["responses"][:8]],
        "serial repeatability",
    )
    return serial


def network_audit(report, item, serial, environment):
    seed = item["seed"]
    check(
        report["seed"] == seed and report["query_count"] == len(report["responses"]) == 222,
        "HTTP coverage",
    )
    check(all(report[key] == item["reference"][key] for key in IDENTITY), "HTTP provenance")
    check(report["exact_parity"] is (seed != 1729), "old/fresh HTTP gate")
    if seed != 1729:
        check(not report["mismatches"], "fresh HTTP mismatches")
    products = deployment_receipt(report, item, environment)
    by_query = {}
    for index, (ref, record, serial_row) in enumerate(
        zip(item["reference"]["responses"], report["responses"], serial["responses"], strict=True)
    ):
        query = (ref["subject"], ref["dimension"])
        by_query[query] = index
        check(
            record["request"] == {"subject": query[0], "dimension": query[1]}
            and record["status"] == 200,
            "HTTP request order/status",
        )
        if seed != 1729:
            check(not record["mismatches"], "fresh per-query mismatches")
        composition_body(record["response"], serial_row["composition"], ref, item, products)
    metadata = report["metadata_checks"]
    check(
        [r["kind"] for r in metadata]
        == ["filtered"] * 2 + ["legacy"] + ["invalid"] * 5 + ["admission"] * 2,
        "HTTP auxiliary scenario inventory",
    )
    check(
        [r["status"] for r in metadata] == [200] * 3 + [422] * 5 + [503] * 2, "auxiliary statuses"
    )
    for record in metadata[:2]:
        query = (record["request"]["subject"], record["request"]["dimension"])
        index = by_query[query]
        ref = item["reference"]["responses"][index]
        answer = serial["responses"][index]["composition"]
        check(answer["selected_kind"] == "pair_composition", "filter exercise is not composed pair")
        composition_body(record["response"], answer, ref, item, products, record["request"])
    check(
        metadata[0]["request"]["limit"] == 0
        and metadata[1]["request"]["limit"] == 1
        and len(metadata[1]["request"]["ignore"]) == 1,
        "filter scenario details",
    )
    legacy = metadata[2]
    index = by_query[legacy["request"]["subject"], legacy["request"]["dimension"]]
    check(legacy["reference_index"] == index, "legacy query index")
    ref = item["reference"]["responses"][index]
    raw = item["old_http"]["responses"][index]["response"]["raw"]
    check(
        raw
        == parse_path(raw["token_ids"], ref["subject"], ref["dimension"], item["tokens"], LEGACY),
        "legacy independently parsed raw",
    )
    body = legacy["response"]
    check(
        set(body) == {"raw", "result", "postprocess", "checkpoint_hash", "graph_hash", "timing"}
        and body["raw"] == raw,
        "legacy raw/schema parity",
    )
    check(
        body["checkpoint_hash"] == item["reference"]["checkpoint_hash"]
        and body["graph_hash"] == item["reference"]["corpus_identity"]["graph_hash"],
        "legacy identity",
    )
    hydration(body, raw["targets"], ref, products)
    for record, mutation in zip(
        metadata[3:8],
        (
            {"subject": "ATTR_RED"},
            {"ignore": ["ATTR_RED"]},
            {"limit": -1},
            {"dimension": "BIOME"},
            {"max_new_tokens": 1},
        ),
        strict=True,
    ):
        check(
            all(record["request"].get(k) == v for k, v in mutation.items())
            and set(record["response"]) == {"detail"},
            "invalid request receipt",
        )
    check(
        [r["endpoint"] for r in metadata[8:]] == ["/v1/predict-set", "/v1/predict"],
        "admission routes",
    )
    check(
        all(
            r["response"] == {"detail": "in-flight request limit reached; retry later"}
            for r in metadata[8:]
        ),
        "admission error payloads",
    )
    disabled, failed = report["disabled_check"], report["failure_check"]
    deployment_receipt(disabled, item, environment, False)
    deployment_receipt(failed, item, environment, True, 1)
    check(
        disabled["status"] == 409
        and disabled["direct_composition"] is None
        and disabled["response"] == {"detail": {"code": "set_composition_disabled"}},
        "disabled contract",
    )
    first = item["reference"]["responses"][0]
    check(
        disabled["request"]
        == failed["request"]
        == {"subject": first["subject"], "dimension": first["dimension"]},
        "failure scenario query",
    )
    raw = failed["direct_composition"]["source_paths"]
    check(
        raw == item["old_http"]["failure_check"]["direct_candidates"]["candidates"]
        and len({r["token_ids"][5] for r in raw}) == 4,
        "bound-one raw source parity",
    )
    answer = build_composition(
        raw,
        serial["responses"][0]["logits"],
        first["subject"],
        first["dimension"],
        item["tokens"],
        1,
    )
    check(
        answer == failed["direct_composition"]
        and answer["fallback_no_valid_source"]
        and answer["selected_slot"] == 1
        and not any(s["source_eligible"] for s in answer["slots"]),
        "bound-one raw-set sums/fallback",
    )
    check(
        failed["status"] == 502
        and failed["response"] == {"detail": {"code": "set_composition_no_valid_source", **answer}},
        "failed-source HTTP evidence",
    )
    return {
        "seed": seed,
        "evidence_origin": "reused" if seed == 1729 else "fresh",
        "complete": True,
        "query_count": 222,
        "source_paths": 888,
        "slot_scores_checked": 2220,
        "metadata_checks": 10,
        "filtered_checks": 2,
        "legacy_checks": 1,
        "invalid_422_checks": 5,
        "admission_503_checks": 2,
        "disabled_409_checks": 1,
        "failure_502_checks": 1,
        "owned_servers_stopped": 3,
    }


def campaign_authentication(summary, path):
    check(
        summary["verification_complete"] is True and not summary.get("error"), "incomplete campaign"
    )
    check(
        summary["integration_accepted"] is False
        and summary["independent_audit_status"] == "required",
        "verification improperly self-accepted",
    )
    check(
        summary["evaluator_version"] == "plm-pair-composition-shape-aware-v2"
        and summary["plan_sha256"] == PLAN_SHA
        and summary["original_failed_campaign_sha256"] == V1_SHA
        and summary["original_campaign_accepted"] is False,
        "campaign declaration",
    )
    check(summary["absolute_tolerance"] == summary["relative_tolerance"] == 0, "nonzero tolerance")
    for filename, digest in summary["report_sha256"].items():
        check(
            Path(filename).name == filename
            and filename != path.name
            and filename != "independent-audit.json",
            "invalid/circular report manifest",
        )
        bind(path.parent / filename, digest)
    required = {"references-seal.json"}
    required.update(f"serial-{seed}.json" for seed in SEEDS)
    required.update(f"http-validation-{seed}.json" for seed in SEEDS)
    required.update(f"http-{seed}.json" for seed in SEEDS[1:])
    required.update(f"http-prelaunch-{seed}.json" for seed in SEEDS[1:])
    check(required <= summary["report_sha256"].keys(), "missing final report phases")
    check(not (path.parent / "http-1729.json").exists(), "reused evidence copied as fresh HTTP")
    for filename, digest in summary["snapshot_sha256"].items():
        check(Path(filename).name == filename, "invalid snapshot filename")
        bind(path.with_suffix("." + filename), digest)
    check(
        set(summary["snapshot_sha256"])
        == {
            "script.py",
            "plan.md",
            "source.zip",
            "configs.zip",
            "helper-arithmetic.py",
            "helper-transport.py",
            "helper-server.py",
            "tests.py",
            "cpu-tests.json",
            "gpu-release.json",
        },
        "custom helper snapshot inventory",
    )
    check(
        summary["snapshot_sha256"]["plan.md"] == PLAN_SHA
        and summary["snapshot_sha256"]["source.zip"] == SOURCE_SHA
        and summary["snapshot_sha256"]["configs.zip"] == CONFIGS_SHA,
        "snapshot declarations",
    )
    verifier = ROOT / "scripts/verify_pair_composition_shapes.py"
    bind(verifier, summary["snapshot_sha256"]["script.py"])
    bind(ROOT / "docs/experiments/2026-09-25-pair-composition-shape-aware-plan.md", PLAN_SHA)
    cpu_receipt = read(path.with_suffix(".cpu-tests.json"))
    check(
        cpu_receipt.get("passed") is True
        and cpu_receipt.get("model_execution") is False
        and cpu_receipt.get("verifier_sha256") == summary["snapshot_sha256"]["script.py"]
        and cpu_receipt.get("tests_sha256") == summary["snapshot_sha256"]["tests.py"],
        "focused CPU test receipt",
    )
    bind(
        ROOT / "tests/unit/test_pair_composition_shapes.py",
        summary["snapshot_sha256"]["tests.py"],
    )
    gpu_receipt = read(path.with_suffix(".gpu-release.json"))
    check(
        gpu_receipt.get("gpu_released") is True
        and gpu_receipt.get("owned_servers_stopped") is True,
        "GPU release snapshot",
    )
    for name, filename in (
        ("arithmetic", "independent-audit.py"),
        ("transport", "summary.script.py"),
        ("server", "summary.helper.py"),
    ):
        bind(V1_DIR / filename, summary["snapshot_sha256"][f"helper-{name}.py"])
    for name, digest in summary["input_sha256"].items():
        bind(name, digest)
    # Locate the explicitly pinned release receipt; do not infer release from idle hardware.
    releases = []
    for name in summary["input_sha256"]:
        if Path(name).suffix == ".json":
            payload = json.loads(Path(name).read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("gpu_released") is True:
                check(
                    payload.get("owned_servers_stopped") is True, "GPU release lacks stopped server"
                )
                releases.append(name)
    check(len(releases) == 1, "missing/ambiguous pinned GPU release receipt")
    check(
        sha(releases[0]) == summary["snapshot_sha256"]["gpu-release.json"],
        "GPU release snapshot/original mismatch",
    )
    check(
        summary["snapshot_sha256"]["cpu-tests.json"] in summary["input_sha256"].values()
        and summary["snapshot_sha256"]["tests.py"] in summary["input_sha256"].values(),
        "CPU test snapshot/original mismatch",
    )
    inventory(path.with_suffix(".source.zip"), path.with_suffix(".configs.zip"))
    v1 = read(V1_DIR / "summary.json", V1_SHA)
    check(v1["all_outputs_equal"] is False, "historical failure retroactively accepted")
    prior, pair = provenance(v1)
    failure = read(V1_DIR / "failure-audit.json", FAILURE_SHA)
    bind(V1_DIR / "failure-audit.py", failure["script_sha256"])
    diagnosis_dir = ROOT / "runs/learning/pair-composition-batch-shape-v1"
    diagnosis = read(diagnosis_dir / "summary.json", DIAG_SHA)
    audit = read(diagnosis_dir / "independent-audit.json", DIAG_AUDIT_SHA)
    check(audit["summary_sha256"] == DIAG_SHA, "diagnostic independent binding")
    bind(diagnosis_dir / "independent-audit.py", audit["script_sha256"])
    # Prior immutable audits authenticate diagnostic input archives and their exact bytes.
    for document in (failure, diagnosis, audit):
        for name, digest in document.get("input_sha256", {}).items():
            bind(name, digest)
    check(diagnosis["environment"] == v1["evaluator_environment"], "diagnostic runtime")
    return prior, pair, v1, diagnosis, releases[0]


def sealed_references(summary, path):
    descriptors = summary["references"]
    check([r["seed"] for r in descriptors] == list(SEEDS), "reference seeds/order")
    for entry in descriptors:
        target = path.parent / f"serial-{entry['seed']}.json"
        check(
            Path(entry["path"]).resolve() == target.resolve()
            and entry["complete"] is True
            and entry["query_count"] == 222
            and entry["repeat_forward_count"] == 8,
            "sealed reference descriptor",
        )
        bind(target, entry["sha256"])
        check(
            summary["report_sha256"][target.name] == entry["sha256"], "reference hash disagreement"
        )
    seal_path = path.parent / "references-seal.json"
    seal = read(seal_path, summary["reference_seal_sha256"])
    check(
        seal
        == {
            "all_references_sealed": True,
            "references": descriptors,
            "verifier_sha256": summary["snapshot_sha256"]["script.py"],
            "plan_sha256": PLAN_SHA,
            "normal_reference_forwards": 666,
            "repeat_forwards": 24,
        },
        "reference seal content",
    )
    return descriptors


def validate_receipt(summary, path, item, descriptor, reconstructed):
    seed = item["seed"]
    receipt = read(path.parent / f"http-validation-{seed}.json")
    saved = summary["http_validations"][SEEDS.index(seed)]
    check(receipt == saved, "HTTP validation receipt/summary disagreement")
    for name, value in reconstructed.items():
        check(receipt[name] == value, f"HTTP validation count/origin mismatch: {name}")
    target = V1_DIR / "http-1729.json" if seed == 1729 else path.parent / f"http-{seed}.json"
    check(Path(receipt["report_path"]).resolve() == target.resolve(), "HTTP evidence origin path")
    bind(target, receipt["report_sha256"])
    check(
        receipt["serial_reference_sha256"] == descriptor["sha256"]
        and receipt["reference_seal_sha256"] == summary["reference_seal_sha256"],
        "HTTP reference binding",
    )
    if seed == 1729:
        check(
            receipt["report_sha256"] == HTTP_SHA
            and receipt["original_campaign_sha256"] == V1_SHA
            and receipt["original_exact_parity"] is False
            and receipt["original_campaign_accepted"] is False,
            "original failed gate erased",
        )
        check("prelaunch_receipt_path" not in receipt, "reused HTTP falsely claims new launch")
    else:
        prelaunch_path = path.parent / f"http-prelaunch-{seed}.json"
        check(
            Path(receipt["prelaunch_receipt_path"]).resolve() == prelaunch_path.resolve(),
            "launch path",
        )
        launch = read(prelaunch_path, receipt["prelaunch_receipt_sha256"])
        check(
            launch
            == {
                "seed": seed,
                "evidence_origin": "fresh",
                "action": "launch_archived_http_campaign",
                "reference_seal_path": str(path.parent / "references-seal.json"),
                "reference_seal_sha256": summary["reference_seal_sha256"],
                "serial_reference_path": descriptor["path"],
                "serial_reference_sha256": descriptor["sha256"],
                "verifier_sha256": summary["snapshot_sha256"]["script.py"],
                "checkpoint_hash": item["reference"]["checkpoint_hash"],
            },
            "prelaunch seal/checkpoint/verifier binding",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    path = args.summary.resolve()
    output = path.parent / "independent-audit.json"
    check(not output.exists(), "refusing existing independent audit")
    check(path.is_file(), "final summary missing; no audit receipt written")
    check("torch" not in sys.modules, "independent auditor must remain stdlib-only")
    script_path = Path(__file__).resolve()
    script_digest = sha(script_path)
    summary = read(path)
    prior, pair, v1, diagnosis, release = campaign_authentication(summary, path)
    descriptors = sealed_references(summary, path)
    check(len(summary["offline_reuse"]) == len(summary["http_validations"]) == 3, "phase count")
    rows, all_metrics = [], []
    for seed, descriptor in zip(SEEDS, descriptors, strict=True):
        item = offline_audit(seed, prior, pair, v1)
        check(
            item["reuse"] == summary["offline_reuse"][SEEDS.index(seed)], "offline reuse accounting"
        )
        serial = serial_audit(
            Path(descriptor["path"]), item, v1["evaluator_environment"], diagnosis
        )
        http_path = V1_DIR / "http-1729.json" if seed == 1729 else path.parent / f"http-{seed}.json"
        report = read(
            http_path, HTTP_SHA if seed == 1729 else summary["report_sha256"][http_path.name]
        )
        reconstructed = network_audit(report, item, serial, v1["evaluator_environment"])
        validate_receipt(summary, path, item, descriptor, reconstructed)
        all_metrics.extend(item["metrics"])
        rows.append(
            {
                **reconstructed,
                "offline_metrics": item["reuse"]["metrics"],
                "offline_groups": item["reuse"]["groups"],
                "serial_reference_sha256": descriptor["sha256"],
                "http_sha256": sha(http_path),
            }
        )
        print(
            f"seed {seed}: 888 sources, 2220 slots, 222 HTTP and all auxiliaries verified",
            flush=True,
        )
    accounting = {
        "offline_queries": {"reused": 666, "fresh": 0},
        "offline_sources": {"reused": 2664, "fresh": 0},
        "offline_slots": {"reused": 6660, "fresh": 0},
        "four_path_compatibility_queries": {"reused": 666, "fresh": 0},
        "disabled_greedy_queries": {"reused": 666, "fresh": 0},
        "serial_reference_queries": {"reused": 0, "fresh": 666},
        "serial_repeat_forwards": 24,
        "http_queries": {"reused": 222, "fresh": 444},
        "http_auxiliary_suites": {"reused": 1, "fresh": 2},
        "normal_http_sources": 2664,
        "normal_http_slots": 6660,
        "normal_http_slot_scores_checked": 6660,
        "distinct_reference_slot_scores": 6660,
        "same_shape_query_selection_checks": 666,
        "independent_audit_passed": False,
        "integration_accepted": False,
    }
    check(summary["accounting"] == accounting, "mixed-origin evidence accounting")
    pooled = metrics(all_metrics)
    check(pooled["exact_count"] == 569 and pooled["query_count"] == 666, "pooled exact quality")
    # Re-read every bound artifact at completion; never report success after input drift.
    inventory(path.with_suffix(".source.zip"), path.with_suffix(".configs.zip"))
    check(all(sha(name) == digest for name, digest in INPUTS.items()), "input drift during audit")
    check(sha(script_path) == script_digest, "auditor changed during execution")
    result = {
        "audit_version": "independent-pair-composition-shape-aware-v2",
        "complete": True,
        "audit_passed": True,
        "integration_accepted": False,
        "overall_acceptance_authority": "campaign owner after this independent receipt",
        "summary_sha256": sha(path),
        "script_sha256": script_digest,
        "plan_sha256": PLAN_SHA,
        "input_sha256": INPUTS,
        "all_exact_arithmetic_and_transport_contracts": True,
        "original_campaign_accepted": False,
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
        "metrics": pooled,
        "runs": rows,
        "accounting": {**accounting, "independent_audit_passed": True},
        "owned_server_shutdown_receipts": 9,
        "gpu_release_receipt": release,
        "limitations": [
            "Saved FP32 vectors are authenticated evidence, "
            "not independently regenerated neural outputs.",
            "HTTP source paths/selected slot/sets must match across shapes; "
            "only slot score references are shape-specific.",
            "Seed 1729 HTTP and all offline executions are reused; "
            "HTTP seeds 1730/1731 and 690 head forwards are fresh.",
            "Shutdown is checked from archived owned-server join/stop receipts, "
            "not independent socket probes.",
            "Seal/prelaunch receipts bind phase order via the archived verifier; "
            "filesystem timestamps are not execution proof.",
            "No protected test data, training, neural inference, throughput, "
            "energy, SLO, or oracle-quality claim.",
        ],
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "integration_accepted": False,
                "summary_sha256": sha(path),
                "audit_sha256": sha(output),
            }
        )
    )


if __name__ == "__main__":
    main()
