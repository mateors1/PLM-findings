"""Verify unchanged composition with exact references at each declared batch shape."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import struct
import sys
import time
import zipfile
from pathlib import Path

_PLAN_SHA = "0e406e2515c582ba5f4e8f3897b58e7daa6c27c96711913ef5d335f7eef2861f"
_V1_SHA = "7abc3a99ea8aa0e95749da2b379893572f7f76643a53ab98f8242c2c77da7042"
_FAILURE_SHA = "65ffee3f6035943cf07210f371de061e460f0a554619cb9fbd56783aab1c0253"
_DIAG_SHA = "d263db1e956064b1f5c597369f5c82b95a7edc2fca55901d98a2f72e4a01557b"
_DIAG_AUDIT_SHA = "6bc483713c5dd4511a3bdbc455649435f6da85a63ef8756bb6998573e90c44ee"
_SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
_CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
_OFFLINE_SHA = (
    "c7ad1db44d3c7e0387369b1ffc982d5e7b96afb046c4f8a68111b235ae5a3f67",
    "932ad33ad4d0b00e29cd7d64966b3f0d5820df91764d9ba1188a22cdd84e1659",
    "cd713d451a1235f7ad73d8b524acbf3c250b482f78e84c2cfd08f778e3c5ae73",
)
_HTTP_SHA = "e6759659c0eec1130088c1324c8457cc2c0bf814af7773c05a8f6b251c421e87"
_SEEDS = (1729, 1730, 1731)
_COLUMNS = list(range(1024, 2049))
_ORDER = ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
_IDENTITY = ("checkpoint_hash", "training_identity", "corpus_identity", "split_hash")


def _check(condition, code, message):
    if not condition:
        raise ValueError(f"{code}: {message}")


def _call(code, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError(f"{code}: {exc}") from exc


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _value_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _bind(path, digest, inputs):
    path = Path(path).resolve()
    _check(path.is_file() and _sha(path) == digest, "identity_mismatch", str(path))
    inputs[str(path)] = digest
    return path


def _load(path, digest, inputs, name):
    _bind(path, digest, inputs)
    spec = importlib.util.spec_from_file_location(name, path)
    _check(spec is not None and spec.loader is not None, "identity_mismatch", "helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inventory(root, source, configs):
    for path, names in (
        (
            source,
            {p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")}
            | {"pyproject.toml", "uv.lock"},
        ),
        (configs, {p.relative_to(root).as_posix() for p in (root / "configs").rglob("*.yaml")}),
    ):
        with zipfile.ZipFile(path) as archive:
            _check(
                len(archive.namelist()) == len(set(archive.namelist()))
                and set(archive.namelist()) == names,
                "identity_mismatch",
                "archive inventory",
            )
            _check(
                all((root / n).read_bytes() == archive.read(n) for n in names),
                "identity_mismatch",
                "live source/config bytes",
            )


def _coverage(rows, reference, seed, expected_count=222):
    _check(
        seed in _SEEDS and len(rows) == len(reference) == expected_count,
        "coverage_mismatch",
        "seed or query count",
    )
    keys = [(r["subject"], r["dimension"]) for r in rows]
    wanted = [(r["subject"], r["dimension"]) for r in reference]
    _check(
        len(set(keys)) == len(keys) and keys == wanted, "coverage_mismatch", "query partition/order"
    )


def _logits(values, columns):
    _check(columns == _COLUMNS, "coverage_mismatch", "product columns")
    _check(len(values) == len(columns), "invalid_serial_reference", "logit cardinality")
    for value in values:
        _check(
            type(value) in (int, float) and math.isfinite(value),
            "invalid_serial_reference",
            "nonfinite/nonnumeric logit",
        )
        try:
            rounded = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError("invalid_serial_reference: FP32 overflow") from exc
        _check(rounded == value, "invalid_serial_reference", "not exact saved FP32")


def _without_scores(composed):
    result = copy.deepcopy(composed)
    for slot in result["slots"]:
        del slot["score"]
    return result


def _serial_composition(offline, logits, columns):
    """Pure canonical sums and earliest-slot selection; no truth or HTTP input."""
    _logits(logits, columns)
    result = copy.deepcopy(offline)
    slots = result["slots"]
    _check(len(slots) == 10, "selection_arithmetic_mismatch", "slot count")
    for index, (slot, ranks) in enumerate(zip(slots, _ORDER, strict=True), 1):
        ids = slot["set_ids"]
        _check(
            slot["slot"] == index
            and slot["source_ranks"] == list(ranks)
            and slot["kind"] == ("original" if len(ranks) == 1 else "pair_composition")
            and type(slot["source_eligible"]) is bool
            and all(type(i) is int and i in columns for i in ids)
            and ids == sorted(set(ids)),
            "selection_arithmetic_mismatch",
            "canonical slot",
        )
        if len(ranks) == 2:
            first, second = (slots[i - 1] for i in ranks)
            _check(
                ids == sorted(set(first["set_ids"]) | set(second["set_ids"]))
                and slot["source_eligible"]
                == (first["source_eligible"] and second["source_eligible"]),
                "selection_arithmetic_mismatch",
                "pair union/eligibility",
            )
        slot["score"] = math.fsum(logits[i - 1024] for i in ids)
    eligible = [s for s in slots if s["source_eligible"]]
    chosen = min(eligible, key=lambda s: (-s["score"], s["slot"])) if eligible else slots[0]
    result.update(
        selected_slot=chosen["slot"],
        selected_kind=chosen["kind"],
        selected_source_ranks=chosen["source_ranks"],
        selected_set_ids=chosen["set_ids"],
        selected_source_eligible=chosen["source_eligible"],
        fallback_no_valid_source=not eligible,
    )
    return result


def _same_shape(actual, offline, serial):
    _check(
        _without_scores(actual) == _without_scores(offline) == _without_scores(serial),
        "cross_shape_output_mismatch",
        "non-score sources/slots/selection/policy",
    )
    _check(
        [s["score"] for s in actual["slots"]] == [s["score"] for s in serial["slots"]],
        "same_shape_score_mismatch",
        "exact serial canonical sums",
    )


def _settings(actual, diagnostic):
    expected = {**diagnostic, "autocast_cuda_enabled": False, "autocast_cpu_enabled": False}
    _check(actual == expected, "identity_mismatch", "numerical settings")


def _runtime_settings(torch, model):
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "model_training": model.training,
        "autocast_cuda_enabled": torch.is_autocast_enabled("cuda"),
        "autocast_cpu_enabled": torch.is_autocast_enabled("cpu"),
    }


def _repeated(first, repeat, diagnostic=None):
    _check(
        first == repeat and (diagnostic is None or first == diagnostic),
        "invalid_serial_reference",
        "repeat or seed1729 diagnosis drift",
    )


def _stopped(report):
    _check(
        report.get("server_stopped") is True
        and all(
            report.get(k, {}).get("server_stopped") is True
            for k in ("disabled_check", "failure_check")
        ),
        "owned_server_shutdown_failure",
        "normal/auxiliary shutdown not confirmed",
    )


def _accounting(receipts, references, audit_passed=False):
    _check(type(audit_passed) is bool, "evidence_accounting_failure", "audit flag must be boolean")
    _check(
        len(receipts) == len(references) == 3
        and {r["seed"] for r in receipts} == {r["seed"] for r in references} == set(_SEEDS),
        "evidence_accounting_failure",
        "missing or duplicate seed evidence",
    )
    for row in receipts:
        _check(
            row["complete"] is True
            and row["query_count"] == 222
            and row["evidence_origin"] == ("reused" if row["seed"] == 1729 else "fresh"),
            "evidence_accounting_failure",
            "HTTP completeness/origin",
        )
    _check(
        all(
            r["complete"] is True and r["query_count"] == 222 and r["repeat_forward_count"] == 8
            for r in references
        ),
        "evidence_accounting_failure",
        "reference completeness",
    )
    return {
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
        "independent_audit_passed": audit_passed,
        "integration_accepted": audit_passed is True,
    }


def _preflight(root, plan):
    inputs = {}
    v1dir = root / "runs/learning/pair-composition-integration-v1"
    diagdir = root / "runs/learning/pair-composition-batch-shape-v1"
    _bind(plan, _PLAN_SHA, inputs)
    v1 = _read(_bind(v1dir / "summary.json", _V1_SHA, inputs))
    failure = _read(_bind(v1dir / "failure-audit.json", _FAILURE_SHA, inputs))
    diagnosis = _read(_bind(diagdir / "summary.json", _DIAG_SHA, inputs))
    diag_audit = _read(_bind(diagdir / "independent-audit.json", _DIAG_AUDIT_SHA, inputs))
    _check(
        v1["all_outputs_equal"] is False
        and failure["campaign_accepted"] is False
        and failure["failure_audit_complete"] is True
        and diag_audit["complete"] is True
        and diag_audit["all_saved_arithmetic_and_bindings_equal"] is True
        and diag_audit["summary_sha256"] == _DIAG_SHA
        and diagnosis["all_declared_expectations_passed"] is True,
        "identity_mismatch",
        "failed-v1/diagnostic audit prerequisites",
    )
    _bind(v1dir / "summary.source.zip", _SOURCE_SHA, inputs)
    _bind(v1dir / "summary.configs.zip", _CONFIGS_SHA, inputs)
    _inventory(root, v1dir / "summary.source.zip", v1dir / "summary.configs.zip")
    arithmetic_path = v1dir / "independent-audit.py"
    arithmetic = _load(
        arithmetic_path,
        failure["independent_arithmetic_helper_sha256"],
        inputs,
        "_shape_arithmetic",
    )
    prior, pair = _call("identity_mismatch", arithmetic.provenance, v1)
    helpers = {
        "arithmetic": arithmetic_path,
        "transport": v1dir / "summary.script.py",
        "server": v1dir / "summary.helper.py",
    }
    transport = _load(helpers["transport"], v1["script_sha256"], inputs, "_shape_transport")
    server = _load(helpers["server"], v1["helper_sha256"], inputs, "_shape_server")
    for path, digest in (
        (v1dir / "failure-audit.py", failure["script_sha256"]),
        (diagdir / "independent-audit.py", diag_audit["script_sha256"]),
        (diagdir / "summary.script.py", diagnosis["script_sha256"]),
        (diagdir / "summary.source.zip", _SOURCE_SHA),
        (diagdir / "summary.declaration.md", diagnosis["declaration_sha256"]),
    ):
        _bind(path, digest, inputs)
    for manifest in (failure, diagnosis, diag_audit):
        for path, digest in manifest["input_sha256"].items():
            _bind(path, digest, inputs)
    _read(_bind(v1dir / "http-1729.json", _HTTP_SHA, inputs))
    items, reused, partition = [], [], None
    for seed, digest in zip(_SEEDS, _OFFLINE_SHA, strict=True):
        offline = _read(_bind(v1dir / f"offline-{seed}.json", digest, inputs))
        reference = arithmetic.read(
            root / f"runs/learning/pair-unions-v1/seed-{seed}.json",
            pair["report_sha256"][f"seed-{seed}.json"],
        )
        olddir = root / "runs/learning/set-reranking-integration-v1"
        old_offline = arithmetic.read(
            olddir / f"offline-{seed}.json", prior["report_sha256"][f"offline-{seed}.json"]
        )
        old_http = arithmetic.read(
            olddir / f"http-{seed}.json", prior["report_sha256"][f"http-{seed}.json"]
        )
        checkpoint = _bind(
            root / f"runs/national_dex_continuation_control_s{seed}_v1/checkpoint-final.pt",
            reference["checkpoint_hash"],
            inputs,
        )
        run = arithmetic.read(checkpoint.parent / "run.json")
        sidecar = arithmetic.read(checkpoint.with_suffix(".pt.json"))
        training = arithmetic.read(checkpoint.parent / "training-result.json")
        _check(
            run["identity"]
            == sidecar["experiment_identity"]
            == training["identity"]
            == reference["training_identity"]
            and sidecar["checkpoint_hash"]
            == training["checkpoint_hash"]
            == reference["checkpoint_hash"]
            and sidecar["training_metadata"]["model_config"] == run["config"]["model"],
            "identity_mismatch",
            "training/checkpoint receipts",
        )
        for report in (offline, old_offline, old_http):
            _check(
                report["seed"] == seed and all(report[k] == reference[k] for k in _IDENTITY),
                "identity_mismatch",
                "seed/checkpoint/data/split",
            )
        expected_config = arithmetic.config(run, 8)
        _check(
            offline["config"] == expected_config
            and offline["inference_config_hash"] == arithmetic.canonical(expected_config)
            and offline["historical_training_config_hash"]
            == run["identity"]["resolved_config_hash"]
            and offline["inserted_training_defaults"]
            == ["eval.symmetric_set_reranking=false", "eval.pair_set_composition=false"]
            and offline["disabled_composition_config_hash"]
            == arithmetic.canonical(arithmetic.config(run, 8, False))
            and offline["disabled_greedy_config_hash"]
            == arithmetic.canonical(arithmetic.config(run, 8, False, False)),
            "identity_mismatch",
            "historical/effective config and named migrations",
        )
        _check(
            offline["exact_parity"] is True
            and not offline["mismatches"]
            and not offline["cleanup_errors"]
            and offline["candidate_count"] == 888
            and offline["slot_count"] == 2220
            and offline["score_tolerance"] == 0,
            "coverage_mismatch",
            "offline report incomplete",
        )
        _coverage(offline["responses"], reference["responses"], seed)
        _coverage(old_offline["responses"], reference["responses"], seed)
        identities = [
            (r["subject"], r["dimension"], r["expected"], r["group"])
            for r in reference["responses"]
        ]
        _check(
            partition is None or identities == partition,
            "coverage_mismatch",
            "seed partition drift",
        )
        partition = identities
        tokens = arithmetic.read(
            Path(run["config"]["data"]["corpus_manifest"]).parent / "vocabulary.json"
        )["tokens"]
        head = arithmetic.read(root / f"runs/learning/set-reranking-v1/seed-{seed}.json")
        _check(head["product_token_ids"] == _COLUMNS, "coverage_mismatch", "historical columns")
        labels = arithmetic.diagnostic_groups(run["config"]["data"]["graph_db"])
        metrics = []
        for ref, actual, previous in zip(
            reference["responses"], offline["responses"], old_offline["responses"], strict=True
        ):
            _logits(ref["symmetric_relation_logits"], _COLUMNS)
            composed = _call(
                "selection_arithmetic_mismatch", arithmetic.reference_composition, ref, tokens
            )
            _check(
                actual["composition"] == composed
                and not actual["mismatches"]
                and actual["selected_set_keys"] == ref["selected_set_keys"]
                and actual["expected"] == previous["expected"] == ref["expected"]
                and actual["group"] == ref["group"] == labels[ref["subject"], ref["dimension"]]
                and actual["disabled_composition"] == previous["reranked"]["selected"]
                and actual["disabled_greedy"] == previous["disabled_greedy"],
                "cross_shape_output_mismatch",
                "offline source/selection/compatibility",
            )
            for name, descriptor in (
                ("disabled_composition", arithmetic.LEGACY),
                ("disabled_greedy", arithmetic.BASE),
            ):
                raw = actual[name]
                _check(
                    raw
                    == arithmetic.raw_evidence(
                        raw["token_ids"], ref["subject"], ref["dimension"], tokens, descriptor
                    ),
                    "cross_shape_output_mismatch",
                    "compatibility full raw evidence",
                )
            metrics.append(arithmetic.metric(ref, composed, tokens))
        totals = arithmetic.aggregate(metrics)
        groups = {
            group: arithmetic.aggregate(
                [
                    m
                    for m, r in zip(metrics, reference["responses"], strict=True)
                    if r["group"] == group
                ]
            )
            for group in ("TYPE_single", "TYPE_dual", "COLOR")
        }
        _check(
            totals == offline["metrics"] == reference["comparison"]["pair_selector"]
            and groups == offline["groups"]
            and all(groups[g] == reference["groups"][g]["pair_selector"] for g in groups),
            "selection_arithmetic_mismatch",
            "offline/group metrics",
        )
        reused.append(
            {
                "seed": seed,
                "evidence_origin": "reused",
                "query_count": 222,
                "source_paths": 888,
                "slots": 2220,
                "metrics": totals,
                "groups": groups,
                "report": str(v1dir / f"offline-{seed}.json"),
                "report_sha256": digest,
            }
        )
        items.append(
            {
                "seed": seed,
                "reference": reference,
                "offline": offline,
                "old_offline": old_offline,
                "old_http": old_http,
                "run": run,
                "checkpoint": checkpoint,
                "tokens": tokens,
            }
        )
    _check(
        [r["metrics"]["exact_count"] for r in reused] == [191, 192, 186],
        "selection_arithmetic_mismatch",
        "reused quality totals",
    )
    inputs.update(arithmetic.INPUTS)
    return {
        "v1": v1,
        "pair": pair,
        "diagnosis": diagnosis,
        "items": items,
        "inputs": inputs,
        "helpers": helpers,
        "arithmetic": arithmetic,
        "transport": transport,
        "server": server,
        "offline_reuse": reused,
        "root": root,
        "v1dir": v1dir,
    }


def _seal_reference(report, item, context):
    _check(
        report["complete"] is True
        and report["seed"] == item["seed"]
        and report["query_count"] == report["reference_forward_count"] == 222
        and report["repeat_forward_count"] == 8
        and len(report["repeat_logits"]) == 8,
        "invalid_serial_reference",
        "reference completeness",
    )
    config = context["arithmetic"].config(item["run"], 1)
    digest = context["arithmetic"].canonical(config)
    environment = context["v1"]["evaluator_environment"]
    _check(
        report["config"] == config
        and report["config_hash"] == digest
        and report["batch_shape"] == [1, 5]
        and report["environment"] == environment
        and report["source_archive_sha256"] == _SOURCE_SHA
        and report["runtime_environment_sha256"] == _value_sha(environment)
        and report["evidence_origin"] == "fresh"
        and all(report[k] == item["reference"][k] for k in _IDENTITY),
        "identity_mismatch",
        "serial config/runtime/checkpoint/source",
    )
    _settings(report["numerical_settings"], context["diagnosis"]["torch_settings"])
    _coverage(report["responses"], item["reference"]["responses"], item["seed"])
    mapping = {name: i for i, name in enumerate(item["tokens"])}
    for index, row in enumerate(report["responses"]):
        prompt = [mapping[k] for k in ("BOS", row["subject"], row["dimension"], "SAME", "ANSWER")]
        _check(
            row["prompt_ids"] == prompt
            and row["batch_shape"] == [1, 5]
            and row["seed"] == item["seed"]
            and row["config_hash"] == digest
            and row["source_archive_sha256"] == _SOURCE_SHA
            and row["checkpoint_hash"] == item["reference"]["checkpoint_hash"]
            and row["runtime_environment_sha256"] == _value_sha(environment),
            "identity_mismatch",
            "serial observation identity/prompt",
        )
        _logits(row["logits"], report["product_token_ids"])
        expected = _serial_composition(
            item["offline"]["responses"][index]["composition"], row["logits"], _COLUMNS
        )
        _same_shape(
            row["composition"], item["offline"]["responses"][index]["composition"], expected
        )
        _check(
            row["cross_shape_score_differences"]
            == [
                a["score"] - b["score"]
                for a, b in zip(
                    expected["slots"],
                    item["offline"]["responses"][index]["composition"]["slots"],
                    strict=True,
                )
            ],
            "selection_arithmetic_mismatch",
            "cross-shape diagnostic arithmetic",
        )
        if item["seed"] == 1729:
            _repeated(row["logits"], context["diagnosis"]["responses"][index]["serial_logits"])
    _repeated([r["logits"] for r in report["responses"][:8]], report["repeat_logits"])


def _reference(item, context, output):
    import torch

    from plm.config import RootConfig
    from plm.configuration import config_hash
    from plm.serving.parser import prompt_ids
    from plm.serving.runtime import load_inference_runtime

    arithmetic = context["arithmetic"]
    config = RootConfig.model_validate(arithmetic.config(item["run"], 1))
    runtime = load_inference_runtime(config, item["checkpoint"])
    _check(
        runtime.device == "cuda"
        and all(
            getattr(runtime, key) == item["reference"][key]
            for key in ("checkpoint_hash", "training_identity", "corpus_identity")
        )
        and runtime.split.split_hash == item["reference"]["split_hash"],
        "identity_mismatch",
        "runtime",
    )
    numerical = _runtime_settings(torch, runtime.model)
    _settings(numerical, context["diagnosis"]["torch_settings"])
    _check(
        config_hash(runtime.config) == arithmetic.canonical(arithmetic.config(item["run"], 1)),
        "identity_mismatch",
        "serial reference config",
    )
    queries = [{"subject": r.subject, "dimension": r.dimension} for r in runtime.split.validation]
    _coverage(queries, item["reference"]["responses"], item["seed"])
    report = {
        "seed": item["seed"],
        "evidence_origin": "fresh",
        "complete": False,
        **{k: item["reference"][k] for k in _IDENTITY},
        "environment": context["v1"]["evaluator_environment"],
        "config": runtime.config.model_dump(mode="json"),
        "config_hash": config_hash(runtime.config),
        "numerical_settings": numerical,
        "product_token_ids": _COLUMNS,
        "batch_shape": [1, 5],
        "source_archive_sha256": _SOURCE_SHA,
        "runtime_environment_sha256": _value_sha(context["v1"]["evaluator_environment"]),
        "query_count": 0,
        "reference_forward_count": 0,
        "repeat_forward_count": 0,
        "responses": [],
        "repeat_logits": [],
    }

    def forward(query):
        prompt = list(prompt_ids(query["subject"], query["dimension"], runtime.vocabulary))
        with torch.inference_mode():
            tensor = torch.tensor([prompt], dtype=torch.long, device=runtime.device)
            logits = runtime.model(tensor).symmetric_relation_logits
            _check(
                logits.dtype == torch.float32
                and tuple(logits.shape) == (1, 1025)
                and bool(torch.isfinite(logits).all()),
                "invalid_serial_reference",
                "head tensor",
            )
            values = logits.cpu().tolist()[0]
        _logits(values, _COLUMNS)
        return prompt, values

    def cleanup():
        nonlocal runtime
        runtime = None
        gc.collect()
        torch.cuda.empty_cache()

    return _collect_reference(report, queries, forward, item, context, output, cleanup)


def _collect_reference(report, queries, forward, item, context, output, cleanup):
    """Persist every completed observation even if a later forward or cleanup fails."""
    started = time.perf_counter()
    try:
        for index, query in enumerate(queries):
            prompt, values = forward(query)
            report["reference_forward_count"] += 1
            offline = item["offline"]["responses"][index]["composition"]
            composed = _serial_composition(offline, values, _COLUMNS)
            row = {
                **query,
                "prompt_ids": prompt,
                "logits": values,
                "composition": composed,
                "batch_shape": [1, 5],
                "seed": item["seed"],
                "checkpoint_hash": report["checkpoint_hash"],
                "config_hash": report["config_hash"],
                "source_archive_sha256": _SOURCE_SHA,
                "runtime_environment_sha256": report["runtime_environment_sha256"],
                "cross_shape_score_differences": [
                    a["score"] - b["score"]
                    for a, b in zip(composed["slots"], offline["slots"], strict=True)
                ],
            }
            report["responses"].append(row)
            report["query_count"] = len(report["responses"])
            _same_shape(composed, offline, composed)
        for query in queries[:8]:
            _, logits = forward(query)
            report["repeat_logits"].append(logits)
            report["repeat_forward_count"] += 1
        report["complete"] = True
        _seal_reference(report, item, context)
    except BaseException as exc:
        report.update(complete=False, error=f"incomplete_execution: {exc!r}")
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        try:
            cleanup()
        except Exception as exc:
            report.update(complete=False, cleanup_error=repr(exc))
        _write(output, report)
    _check(report["complete"], "incomplete_execution", "reference cleanup")
    return report


def _validate_http(report, item, reference, context, origin):
    audit = context["arithmetic"]
    _stopped(report)
    _check(
        origin == ("reused" if item["seed"] == 1729 else "fresh"),
        "evidence_accounting_failure",
        "HTTP origin",
    )
    _check(
        report.get("exact_parity") is (origin == "fresh")
        and (origin == "reused" or not report.get("mismatches")),
        "transport_contract_mismatch",
        "original/fresh HTTP parity gate",
    )
    _check(
        report["seed"] == item["seed"] and report["query_count"] == len(report["responses"]) == 222,
        "coverage_mismatch",
        "HTTP query count/seed",
    )
    for key in _IDENTITY:
        _check(report[key] == item["reference"][key], "identity_mismatch", "HTTP identity")
    products, config_digest = _call(
        "identity_mismatch",
        audit.deployment,
        report,
        item["run"],
        item["reference"],
        context["v1"]["evaluator_environment"],
    )
    tokens, by_query = item["tokens"], {}
    for index, (observed, ref, serial) in enumerate(
        zip(
            report["responses"], item["reference"]["responses"], reference["responses"], strict=True
        )
    ):
        query = (ref["subject"], ref["dimension"])
        by_query[query] = index
        _check(
            observed["request"] == {"subject": query[0], "dimension": query[1]}
            and observed["status"] == 200,
            "transport_contract_mismatch",
            "HTTP request/status",
        )
        offline = item["offline"]["responses"][index]["composition"]
        expected = _serial_composition(offline, serial["logits"], _COLUMNS)
        actual = {k: observed["response"].get(k) for k in expected}
        _same_shape(actual, offline, expected)
        _call(
            "transport_contract_mismatch",
            audit.http_body,
            observed["response"],
            expected,
            ref,
            tokens,
            products,
            item["reference"],
            config_digest,
        )
    metadata = report.get("metadata_checks", [])
    _check(
        [r["kind"] for r in metadata]
        == ["filtered"] * 2 + ["legacy"] + ["invalid"] * 5 + ["admission"] * 2
        and [r["status"] for r in metadata] == [200] * 3 + [422] * 5 + [503] * 2,
        "transport_contract_mismatch",
        "auxiliary inventory/status",
    )
    for row in metadata[:2]:
        index = by_query[row["request"]["subject"], row["request"]["dimension"]]
        ref, serial = item["reference"]["responses"][index], reference["responses"][index]
        _check(
            ref["selection"]["selected_kind"] == "pair_composition",
            "transport_contract_mismatch",
            "pair filter coverage",
        )
        expected = _serial_composition(
            item["offline"]["responses"][index]["composition"], serial["logits"], _COLUMNS
        )
        _call(
            "transport_contract_mismatch",
            audit.http_body,
            row["response"],
            expected,
            ref,
            tokens,
            products,
            item["reference"],
            config_digest,
            row["request"],
        )
    _check(
        metadata[0]["request"]["limit"] == 0
        and metadata[1]["request"]["limit"] == 1
        and len(metadata[1]["request"]["ignore"]) == 1,
        "transport_contract_mismatch",
        "filter scenarios",
    )
    legacy = metadata[2]
    index = by_query[legacy["request"]["subject"], legacy["request"]["dimension"]]
    _check(index == legacy["reference_index"], "transport_contract_mismatch", "legacy query")
    old = item["old_http"]["responses"][index]["response"]
    _call(
        "transport_contract_mismatch",
        audit.verify_http_body,
        legacy["response"],
        old["raw"],
        item["reference"]["responses"][index],
        products,
        item["reference"]["checkpoint_hash"],
        item["reference"]["corpus_identity"]["graph_hash"],
    )
    for row, mutation in zip(
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
        _check(
            all(row["request"].get(k) == v for k, v in mutation.items())
            and "detail" in row["response"],
            "transport_contract_mismatch",
            "invalid-request receipt",
        )
    _check(
        [r["endpoint"] for r in metadata[8:]] == ["/v1/predict-set", "/v1/predict"]
        and all(
            r["response"] == {"detail": "in-flight request limit reached; retry later"}
            for r in metadata[8:]
        ),
        "transport_contract_mismatch",
        "shared admission",
    )
    disabled, failure = report["disabled_check"], report["failure_check"]
    environment = context["v1"]["evaluator_environment"]
    _call(
        "identity_mismatch",
        audit.deployment,
        disabled,
        item["run"],
        item["reference"],
        environment,
        False,
    )
    _call(
        "identity_mismatch",
        audit.deployment,
        failure,
        item["run"],
        item["reference"],
        environment,
        True,
        1,
    )
    _check(
        disabled["status"] == 409
        and disabled["direct_composition"] is None
        and disabled["response"] == {"detail": {"code": "set_composition_disabled"}},
        "transport_contract_mismatch",
        "disabled deployment",
    )
    first = item["reference"]["responses"][0]
    _check(
        disabled["request"]
        == failure["request"]
        == {"subject": first["subject"], "dimension": first["dimension"]},
        "transport_contract_mismatch",
        "auxiliary query",
    )
    raw = failure["direct_composition"]["source_paths"]
    _check(
        raw == item["old_http"]["failure_check"]["direct_candidates"]["candidates"]
        and len({r["token_ids"][5] for r in raw}) == 4,
        "cross_shape_output_mismatch",
        "bound-one sources",
    )
    expected = _call(
        "selection_arithmetic_mismatch",
        audit.composition,
        raw,
        reference["responses"][0]["logits"],
        first["subject"],
        first["dimension"],
        tokens,
        1,
    )
    _check(
        failure["direct_composition"] == expected
        and expected["fallback_no_valid_source"]
        and expected["selected_slot"] == 1
        and not any(s["source_eligible"] for s in expected["slots"]),
        "selection_arithmetic_mismatch",
        "independent bound-one slot sums/fallback",
    )
    _check(
        failure["status"] == 502
        and failure["response"]
        == {"detail": {"code": "set_composition_no_valid_source", **expected}},
        "transport_contract_mismatch",
        "bound-one HTTP evidence",
    )
    return {
        "seed": item["seed"],
        "evidence_origin": origin,
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
        "shutdown_evidence": (
            "Archived owned-server finally receipt; reused shutdown booleans "
            "are not fresh listener probes"
        ),
    }


def _refuse(output):
    _check(
        not output.exists()
        and not list(output.parent.glob(output.stem + ".*"))
        and not list(output.parent.glob("serial-*.json"))
        and not list(output.parent.glob("http-*.json"))
        and not (output.parent / "references-seal.json").exists(),
        "incomplete_execution",
        "immutable output already exists",
    )


def _release(path, digest, inputs):
    _check(
        path is not None and digest is not None,
        "identity_mismatch",
        "explicit pinned GPU release receipt required",
    )
    release = _read(_bind(path, digest, inputs))
    _check(
        release.get("gpu_released") is True and release.get("owned_servers_stopped") is True,
        "identity_mismatch",
        "benchmark GPU/server release unconfirmed",
    )


def _test_receipt(path, digest, script_sha, tests_path, inputs):
    _check(
        path is not None and digest is not None,
        "identity_mismatch",
        "pinned focused CPU test receipt required",
    )
    receipt = _read(_bind(path, digest, inputs))
    _check(
        receipt.get("passed") is True
        and receipt.get("model_execution") is False
        and receipt.get("verifier_sha256") == script_sha
        and receipt.get("tests_sha256") == _sha(tests_path),
        "identity_mismatch",
        "focused CPU test receipt identity/result",
    )
    _bind(tests_path, receipt["tests_sha256"], inputs)


def _launch_receipt(output, item, seal_path, seal_sha, references, inputs, script_path):
    _check(
        _sha(seal_path) == seal_sha
        and [r["seed"] for r in references] == list(_SEEDS)
        and all(_sha(r["path"]) == r["sha256"] for r in references),
        "identity_mismatch",
        "sealed reference changed before HTTP",
    )
    row = references[_SEEDS.index(item["seed"])]
    receipt = {
        "seed": item["seed"],
        "evidence_origin": "fresh",
        "action": "launch_archived_http_campaign",
        "reference_seal_path": str(seal_path),
        "reference_seal_sha256": seal_sha,
        "serial_reference_path": row["path"],
        "serial_reference_sha256": row["sha256"],
        "verifier_sha256": inputs[str(script_path)],
        "checkpoint_hash": item["reference"]["checkpoint_hash"],
    }
    path = output.parent / f"http-prelaunch-{item['seed']}.json"
    _write(path, receipt)
    inputs[str(path)] = _sha(path)
    return path


def _transport_reference(item, reference):
    revised = copy.deepcopy(item["reference"])
    for row, serial in zip(revised["responses"], reference["responses"], strict=True):
        row["selection"] = {k: serial["composition"][k] for k in row["selection"]}
    return revised


def _unchanged(context):
    _check(
        all(_sha(path) == digest for path, digest in context["inputs"].items()),
        "identity_mismatch",
        "input/helper/script changed",
    )
    _inventory(
        context["root"],
        context["v1dir"] / "summary.source.zip",
        context["v1dir"] / "summary.configs.zip",
    )


def main():
    script_path = Path(__file__).resolve()
    script_bytes = script_path.read_bytes()
    root = script_path.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "runs/learning/pair-composition-integration-v2/summary.json",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=root / "docs/experiments/2026-09-25-pair-composition-shape-aware-plan.md",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--gpu-release-receipt", type=Path)
    parser.add_argument("--gpu-release-sha256")
    parser.add_argument("--cpu-test-receipt", type=Path)
    parser.add_argument("--cpu-test-sha256")
    args = parser.parse_args()
    _check(Path.cwd().resolve() == root, "identity_mismatch", "run from repository root")
    _check(
        "torch" not in sys.modules, "identity_mismatch", "standalone CPU preflight imported Torch"
    )
    _refuse(args.out)
    context = _preflight(root, args.plan)
    context["inputs"][str(script_path)] = hashlib.sha256(script_bytes).hexdigest()
    _unchanged(context)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "reused_offline_queries": 666,
                    "source_paths": 2664,
                    "slots": 6660,
                    "torch_imported": "torch" in sys.modules,
                    "model_execution": False,
                }
            )
        )
        return
    _release(args.gpu_release_receipt, args.gpu_release_sha256, context["inputs"])
    tests_path = root / "tests/unit/test_pair_composition_shapes.py"
    _test_receipt(
        args.cpu_test_receipt,
        args.cpu_test_sha256,
        context["inputs"][str(script_path)],
        tests_path,
        context["inputs"],
    )
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": script_bytes,
        "plan.md": args.plan.read_bytes(),
        "source.zip": (context["v1dir"] / "summary.source.zip").read_bytes(),
        "configs.zip": (context["v1dir"] / "summary.configs.zip").read_bytes(),
        "tests.py": tests_path.read_bytes(),
        "cpu-tests.json": args.cpu_test_receipt.read_bytes(),
        "gpu-release.json": args.gpu_release_receipt.read_bytes(),
    }
    snapshots.update(
        {f"helper-{name}.py": path.read_bytes() for name, path in context["helpers"].items()}
    )
    for suffix, payload in snapshots.items():
        with output.with_suffix("." + suffix).open("xb") as stream:
            stream.write(payload)
    result = {
        "evaluator_version": "plm-pair-composition-shape-aware-v2",
        "plan_sha256": _PLAN_SHA,
        "original_failed_campaign_sha256": _V1_SHA,
        "original_campaign_accepted": False,
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
        "verification_complete": False,
        "integration_accepted": False,
        "independent_audit_status": "required",
        "input_sha256": context["inputs"],
        "snapshot_sha256": {s: _sha(output.with_suffix("." + s)) for s in snapshots},
        "offline_reuse": context["offline_reuse"],
        "references": [],
        "http_validations": [],
        "report_sha256": {},
        "limitations": (
            "Mixed reused/fresh validation evidence; no oracle/test/SLO/throughput/energy claim"
        ),
    }
    try:
        from plm.experimentation.provenance import capture_runtime_provenance

        _, environment = capture_runtime_provenance()
        _check(
            environment == context["v1"]["evaluator_environment"],
            "identity_mismatch",
            "runtime before forwards",
        )
        references = []
        for item in context["items"]:
            path = output.parent / f"serial-{item['seed']}.json"
            report = _reference(item, context, path)
            references.append(report)
            result["references"].append(
                {
                    "seed": item["seed"],
                    "complete": report["complete"],
                    "query_count": 222,
                    "repeat_forward_count": 8,
                    "path": str(path),
                    "sha256": _sha(path),
                }
            )
        seal = {
            "all_references_sealed": True,
            "references": result["references"],
            "verifier_sha256": context["inputs"][str(script_path)],
            "plan_sha256": _PLAN_SHA,
            "normal_reference_forwards": 666,
            "repeat_forwards": 24,
        }
        seal_path = output.parent / "references-seal.json"
        _write(seal_path, seal)
        seal_sha = _sha(seal_path)
        result["reference_seal_sha256"] = seal_sha
        for item, reference in zip(context["items"], references, strict=True):
            _check(
                _sha(seal_path) == seal_sha
                and all(_sha(r["path"]) == r["sha256"] for r in result["references"]),
                "identity_mismatch",
                "sealed reference changed before HTTP",
            )
            if item["seed"] == 1729:
                report_path = context["v1dir"] / "http-1729.json"
                report = _read(_bind(report_path, _HTTP_SHA, context["inputs"]))
                origin = "reused"
            else:
                prelaunch = _launch_receipt(
                    output,
                    item,
                    seal_path,
                    seal_sha,
                    result["references"],
                    context["inputs"],
                    script_path,
                )
                revised = _transport_reference(item, reference)
                transport_item = (
                    item["seed"],
                    revised,
                    item["old_offline"],
                    item["old_http"],
                    item["run"],
                    item["checkpoint"],
                )
                context["transport"]._http_campaign(
                    transport_item, output.parent, context["server"], environment
                )
                report_path = output.parent / f"http-{item['seed']}.json"
                report = _read(report_path)
                origin = "fresh"
            validation = _validate_http(report, item, reference, context, origin)
            validation.update(
                report_path=str(report_path),
                report_sha256=_sha(report_path),
                serial_reference_sha256=result["references"][_SEEDS.index(item["seed"])]["sha256"],
                reference_seal_sha256=seal_sha,
            )
            if origin == "reused":
                validation.update(
                    original_campaign_sha256=_V1_SHA,
                    original_exact_parity=report["exact_parity"],
                    original_campaign_accepted=False,
                )
            else:
                validation.update(
                    prelaunch_receipt_path=str(prelaunch), prelaunch_receipt_sha256=_sha(prelaunch)
                )
            _write(output.parent / f"http-validation-{item['seed']}.json", validation)
            result["http_validations"].append(validation)
        result["accounting"] = _accounting(result["http_validations"], result["references"])
        context["inputs"].update(context["arithmetic"].INPUTS)
        _, final_environment = capture_runtime_provenance()
        _check(final_environment == environment, "identity_mismatch", "final runtime")
        _unchanged(context)
        _check(
            _sha(seal_path) == seal_sha
            and all(_sha(r["path"]) == r["sha256"] for r in result["references"]),
            "identity_mismatch",
            "final reference seal",
        )
        result["verification_complete"] = True
    except BaseException as exc:
        result["error"] = f"incomplete_execution: {exc!r}"
        raise
    finally:
        result["report_sha256"] = {
            p.name: _sha(p) for p in output.parent.glob("*.json") if p != output
        }
        _write(output, result)
    print(
        json.dumps(
            {
                "verification_complete": True,
                "integration_accepted": False,
                "independent_audit_status": "required",
                "summary_sha256": _sha(output),
            }
        )
    )


if __name__ == "__main__":
    main()
