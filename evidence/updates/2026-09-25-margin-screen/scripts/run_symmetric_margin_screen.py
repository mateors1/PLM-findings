"""Run the declared fresh-control/fresh-treatment seed-1729 margin screen."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import math
import struct
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

_PLAN_SHA = "845663b5f739c43b84f372aaf098e097c6380719c2882bd64c56f0661b203803"
_RUN_SHA = "ec6c8e358586efa5808591ca0112ea13b4e52cae4b9891ccac82c299332b870d"
_CONFIG_SHA = "6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148"
_CHECKPOINT_SHA = "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
_PAIR_SHA = "4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992"
_PAIR_AUDIT_SHA = "13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057"
_PAIR_SEED_SHA = "bc468c517941ba8d0f218eae745ddc9558c683ebceb48d73f50d1b977fef4fb5"
_ARMS = ("control", "treatment")
_NAMES = ("national_dex_rank_margin_control_s1729_v1", "national_dex_rank_margin_m1_w01_s1729_v1")
_GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
_COLUMNS = list(range(1024, 2049))
_ORDER = ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
_BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"
_POLICY = _BASE + "+first4-pair6-symmetric-set-logit-sum-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _jsonable(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _bind(path, digest, inputs):
    path = Path(path).resolve()
    _require(path.is_file() and _sha(path) == digest, f"identity mismatch: {path}")
    _require(str(path) not in inputs or inputs[str(path)] == digest, f"input drift: {path}")
    inputs[str(path)] = digest
    return path


def _inventory(root):
    paths = [
        *(root / "src").rglob("*.py"),
        *(root / "configs").rglob("*.yaml"),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(paths)}


def _archive(root, names):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (root / name).read_bytes())
    return buffer.getvalue()


def _source_bytes(root):
    names = [p.relative_to(root).as_posix() for p in sorted((root / "src").rglob("*.py"))]
    return _archive(root, [*names, "pyproject.toml", "uv.lock"])


def _recipe(original):
    from plm.config import RootConfig
    from plm.configuration import validate_saved_config

    base = validate_saved_config(original["config"], _CONFIG_SHA).model_dump(mode="json")
    arms = []
    for name, weight in zip(_NAMES, (0.0, 0.1), strict=True):
        raw = copy.deepcopy(base)
        raw["run_name"] = name
        raw["model"]["symmetric_margin_loss_weight"] = weight
        raw["model"]["symmetric_margin"] = 1.0
        config = RootConfig.model_validate(raw)
        _require(config.model_dump(mode="json") == raw, "recipe coercion")
        arms.append(raw)
    _comparability(arms, base)
    return arms


def _comparability(arms, migrated):
    _require(len(arms) == 2, "two training arms required")
    for raw, name, weight in zip(arms, _NAMES, (0.0, 0.1), strict=True):
        _require(
            raw["run_name"] == name
            and raw["seed"] == 1729
            and raw["model"]["symmetric_margin_loss_weight"] == weight
            and raw["model"]["symmetric_margin"] == 1.0,
            "fixed arm recipe",
        )
        normalized = copy.deepcopy(raw)
        normalized["run_name"] = migrated["run_name"]
        normalized["model"]["symmetric_margin_loss_weight"] = 0.0
        _require(normalized == migrated, "recipe differs beyond named margin weight/run name")


def _evaluation_config(training):
    from plm.config import RootConfig

    raw = copy.deepcopy(training)
    raw["eval"].update(
        first_target_guidance_alpha=16.0,
        generation_batch_size=8,
        prevent_repeated_targets=True,
        use_kv_cache=True,
        constrained_decoding=True,
        symmetric_set_reranking=True,
        pair_set_composition=True,
        max_new_tokens=507,
    )
    return RootConfig.model_validate(raw)


def _refuse(output, root, configs):
    _require(
        not output.exists()
        and not list(output.parent.glob(output.stem + ".*"))
        and not list(output.parent.glob("evaluation-*.json"))
        and not list(output.parent.glob("training-*.json")),
        "immutable campaign output exists",
    )
    for config in configs:
        _require(
            not (root / config["paths"]["run_root"] / config["run_name"]).exists(),
            "fresh training run directory already exists",
        )


def _unchanged(context):
    _require(_inventory(context["root"]) == context["live_inventory"], "live source/config changed")
    _require(
        all(Path(p).is_file() and _sha(p) == h for p, h in context["inputs"].items()),
        "authenticated input/executing script/plan changed",
    )


def _preflight(root, plan, output):
    from plm.configuration import validate_saved_config
    from plm.corpus import load_corpus_records, require_valid_artifacts
    from plm.evaluation.split import split_queries
    from plm.protocol.tokenizer import Vocabulary

    inputs, live = {}, _inventory(root)
    _bind(plan, _PLAN_SHA, inputs)
    old_dir = root / "runs/national_dex_continuation_control_s1729_v1"
    original = _read(_bind(old_dir / "run.json", _RUN_SHA, inputs))
    _require(
        original["identity"]["resolved_config_hash"] == _CONFIG_SHA, "original config identity"
    )
    pair_dir = root / "runs/learning/pair-unions-v1"
    pair = _read(_bind(pair_dir / "summary.json", _PAIR_SHA, inputs))
    audit = _read(_bind(pair_dir / "independent-audit.json", _PAIR_AUDIT_SHA, inputs))
    reference = _read(_bind(pair_dir / "seed-1729.json", _PAIR_SEED_SHA, inputs))
    _require(
        audit["summary_sha256"] == _PAIR_SHA
        and audit["all_arithmetic_and_provenance_equal"] is True
        and pair["report_sha256"]["seed-1729.json"] == _PAIR_SEED_SHA,
        "pair audit binding",
    )
    _bind(pair_dir / "independent-audit.py", audit["script_sha256"], inputs)
    # The historical manifest independently pins data and original-training receipts.
    for name, digest in pair["input_sha256"].items():
        path = (root / name).resolve()
        relative = path.relative_to(root).as_posix()
        if relative.startswith("data/") or path.parent == old_dir:
            _bind(path, digest, inputs)
    _bind(old_dir / "checkpoint-final.pt", _CHECKPOINT_SHA, inputs)
    for name in ("run.json", "checkpoint-final.pt.json", "training-result.json"):
        _require(str((old_dir / name).resolve()) in inputs, "unbound original training receipt")
    sidecar, training = (
        _read(old_dir / "checkpoint-final.pt.json"),
        _read(old_dir / "training-result.json"),
    )
    _require(
        original["identity"]
        == reference["training_identity"]
        == sidecar["experiment_identity"]
        == training["identity"]
        and reference["checkpoint_hash"]
        == sidecar["checkpoint_hash"]
        == training["checkpoint_hash"]
        == _CHECKPOINT_SHA
        and sidecar["global_step"] == training["global_step"] == 2000
        and sidecar["training_metadata"]["model_config"] == original["config"]["model"],
        "original checkpoint/training chain",
    )
    migrated = validate_saved_config(original["config"], _CONFIG_SHA)
    configs = _recipe(original)
    _refuse(output, root, configs)
    corpus = root / Path(migrated.data.corpus_manifest).parent
    report = require_valid_artifacts(corpus, graph_db=root / migrated.data.graph_db)
    _require(report.manifest is not None, "validated corpus manifest missing")
    split = split_queries(
        load_corpus_records(corpus),
        migrated.data.validation_query_fraction,
        migrated.data.test_query_fraction,
        migrated.data.split_seed,
    )
    vocabulary = Vocabulary.load(corpus / "vocabulary.json")
    _require(vocabulary.entity_ids() == _COLUMNS and len(vocabulary) == 2049, "product columns")
    corpus_identity = {
        "graph_hash": report.manifest.graph_hash,
        "records_hash": report.manifest.records_hash,
        "vocabulary_hash": report.manifest.tokenizer_hash,
    }
    _require(
        corpus_identity == reference["corpus_identity"]
        and split.split_hash == reference["split_hash"],
        "validation corpus/split identity",
    )
    rows = reference["responses"]
    _require(
        reference["seed"] == 1729
        and reference["query_count"] == len(rows) == len(split.validation) == 222,
        "validation coverage",
    )
    seen = set()
    for record, row in zip(split.validation, rows, strict=True):
        key = (record.subject, record.dimension)
        _require(
            key not in seen
            and key == (row["subject"], row["dimension"])
            and sorted(record.targets) == sorted(row["expected"])
            and row["group"] in _GROUPS
            and (record.dimension == "COLOR") == (row["group"] == "COLOR"),
            "validation order/expected/group metadata",
        )
        seen.add(key)
        truth = vocabulary.encode(record.targets)
        _require(
            0 < len(set(truth)) == len(truth) < 1024
            and vocabulary.id_of(record.subject) not in truth,
            "invalid subject-excluded teacher set",
        )
    context = {
        "root": root,
        "inputs": inputs,
        "live_inventory": live,
        "original": original,
        "reference": reference,
        "configs": configs,
        "corpus_identity": corpus_identity,
        "split_hash": split.split_hash,
    }
    _unchanged(context)
    _require("torch" not in sys.modules, "CPU preflight imported Torch")
    return context


def _metrics(ids, truth, successful=True):
    prediction, expected = set(ids) if successful else set(), set(truth)
    _require(expected, "empty expected set")
    tp = len(prediction & expected)
    precision = tp / len(prediction) if prediction else 0.0
    recall = tp / len(expected)
    return {
        "exact": successful and prediction == expected,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": len(prediction - expected),
        "fn": len(expected - prediction),
        "set_size": len(prediction),
    }


def _head_analysis(logits, truth, subject):
    _require(
        len(logits) == 1025 and all(type(z) in (int, float) and math.isfinite(z) for z in logits),
        "nonfinite or misaligned head logits",
    )
    _require(
        all(struct.unpack("f", struct.pack("f", z))[0] == z for z in logits), "head logits not FP32"
    )
    allowed = set(_COLUMNS) - {subject}
    _require(
        subject in _COLUMNS and set(truth) <= allowed and 0 < len(truth) < len(allowed),
        "invalid head labels",
    )
    direct = [i for i in _COLUMNS if i != subject and logits[i - 1024] > 0]
    a = min(logits[i - 1024] for i in truth)
    b = max(logits[i - 1024] for i in allowed - set(truth))
    return {
        "direct_set_ids": direct,
        "direct_metrics": _metrics(direct, truth),
        "strict_separable": b < a,
        "min_true_logit": a,
        "max_negative_logit": b,
        "gap": a - b,
    }


def _verify_composition(value, logits, prompt, tokens):
    _require(
        value["policy"] == _POLICY
        and len(value["source_paths"]) == 4
        and len(value["slots"]) == 10,
        "composition source/slot/policy inventory",
    )
    sets, flags = [], []
    eos, subject = tokens.index("EOS"), prompt[1]
    for rank, source in enumerate(value["source_paths"], 1):
        ids = source["token_ids"]
        _require(
            all(type(i) is int and 0 <= i < len(tokens) for i in ids)
            and ids[:5] == prompt
            and 6 <= len(ids) <= 512,
            "raw source prompt/budget",
        )
        ended = ids[-1] == eos
        targets = ids[5:-1] if ended else ids[5:]
        _require(
            targets and all(i in _COLUMNS for i in targets) and (ended or len(ids) == 512),
            "raw product grammar",
        )
        valid = ended and len(ids) >= 7
        error = (
            None
            if valid
            else (
                "generated response is shorter than the protocol grammar"
                if len(ids) < 7
                else "response must terminate with EOS"
            )
        )
        _require(
            source
            == {
                "token_ids": ids,
                "targets": [tokens[i] for i in targets],
                "terminated": ended,
                "protocol_valid": valid,
                "error": error,
                "decoding": _BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            },
            "raw source fields",
        )
        flags.append(valid and ended and len(targets) == len(set(targets)))
        sets.append(set(targets) - {subject} if valid and ended else set(targets))
    for index, ranks in enumerate(_ORDER, 1):
        members = sorted(set().union(*(sets[r - 1] for r in ranks)))
        expected = {
            "slot": index,
            "kind": "original" if index <= 4 else "pair_composition",
            "source_ranks": list(ranks),
            "set_ids": members,
            "source_eligible": all(flags[r - 1] for r in ranks),
            "score": math.fsum(logits[i - 1024] for i in members),
        }
        _require(
            value["slots"][index - 1] == expected,
            "canonical slot or exact same-shape score mismatch",
        )
    eligible = [s for s in value["slots"] if s["source_eligible"]]
    chosen = (
        min(eligible, key=lambda s: (-s["score"], s["slot"])) if eligible else value["slots"][0]
    )
    _require(
        all(
            value[k] == chosen[v]
            for k, v in (
                ("selected_slot", "slot"),
                ("selected_kind", "kind"),
                ("selected_source_ranks", "source_ranks"),
                ("selected_set_ids", "set_ids"),
                ("selected_source_eligible", "source_eligible"),
            )
        )
        and value["fallback_no_valid_source"] == (not eligible),
        "slot selection/tie/fallback",
    )


def _row(index, record, metadata, composition, logits, prompt, tokens, batch_index, batch_size):
    _verify_composition(composition, logits, prompt, tokens)
    truth = sorted(tokens.index(t) for t in record.targets)
    success = (
        composition["selected_source_eligible"] and not composition["fallback_no_valid_source"]
    )
    return {
        "index": index,
        "subject": record.subject,
        "dimension": record.dimension,
        "group": metadata["group"],
        "expected_set_ids": truth,
        "prompt_ids": prompt,
        "batch_index": batch_index,
        "batch_shape": [batch_size, 5],
        "composition": composition,
        "symmetric_relation_logits": logits,
        "selected_metrics": _metrics(composition["selected_set_ids"], truth, success),
        **_head_analysis(logits, truth, prompt[1]),
    }


def _head_totals(rows):
    _require(rows, "empty diagnostic partition")
    return {
        "query_count": len(rows),
        "strict_separation_count": sum(r["strict_separable"] for r in rows),
        "direct_exact_count": sum(r["direct_metrics"]["exact"] for r in rows),
        **{
            f"direct_{key}": math.fsum(r["direct_metrics"][key] for r in rows) / len(rows)
            for key in ("precision", "recall", "f1", "set_size")
        },
    }


def _record_observation(
    report, index, record, metadata, composition, logits, prompt, tokens, batch_index, batch_size
):
    # Keep the expensive raw evidence even if exact arithmetic verification fails.
    report["failed_observation"] = {
        "index": index,
        "subject": record.subject,
        "dimension": record.dimension,
        "group": metadata["group"],
        "expected": list(record.targets),
        "composition": composition,
        "symmetric_relation_logits": logits,
        "prompt_ids": prompt,
        "batch_index": batch_index,
        "batch_shape": [batch_size, 5],
    }
    row = _row(
        index, record, metadata, composition, logits, prompt, tokens, batch_index, batch_size
    )
    report["responses"].append(row)
    report["query_count"] = len(report["responses"])
    del report["failed_observation"]


def _gate(control, treatment):
    _require(
        control["complete"] is True and treatment["complete"] is True, "incomplete arm evidence"
    )
    _require(len(control["responses"]) == len(treatment["responses"]) == 222, "paired coverage")
    gained, lost = [], []
    for a, b in zip(control["responses"], treatment["responses"], strict=True):
        identity = (a["index"], a["subject"], a["dimension"], a["group"], a["expected_set_ids"])
        _require(
            identity
            == (b["index"], b["subject"], b["dimension"], b["group"], b["expected_set_ids"]),
            "paired alignment",
        )
        query = {k: a[k] for k in ("index", "subject", "dimension", "group")}
        if b["selected_metrics"]["exact"] and not a["selected_metrics"]["exact"]:
            gained.append(query)
        if a["selected_metrics"]["exact"] and not b["selected_metrics"]["exact"]:
            lost.append(query)
    source_ok = all(
        all(
            p["terminated"] is True and p["protocol_valid"] is True and p["error"] is None
            for p in r["composition"]["source_paths"]
        )
        and all(s["source_eligible"] is True for s in r["composition"]["slots"])
        and r["composition"]["selected_source_eligible"] is True
        and r["composition"]["fallback_no_valid_source"] is False
        for arm in (control, treatment)
        for r in arm["responses"]
    )
    checks = {
        "all_sources_and_selected_eligible": source_ok,
        "exact_count_strictly_improved": treatment["metrics"]["exact_set_count"]
        > control["metrics"]["exact_set_count"],
        "macro_f1_nondecreased": treatment["metrics"]["f1"] >= control["metrics"]["f1"],
        "groups_exact_nondecreased": all(
            treatment["groups"][g]["metrics"]["exact_set_count"]
            >= control["groups"][g]["metrics"]["exact_set_count"]
            for g in _GROUPS
        ),
        "strict_separation_strictly_improved": treatment["head_diagnostics"][
            "strict_separation_count"
        ]
        > control["head_diagnostics"]["strict_separation_count"],
    }
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


def _training_receipt(arm, config, context, environment):
    from plm.config import RootConfig
    from plm.configuration import config_hash
    from plm.model.interfaces import SYMMETRIC_MARGIN_OBJECTIVE

    directory = context["root"] / config["paths"]["run_root"] / config["run_name"]
    paths = {
        name: directory / name
        for name in (
            "run.json",
            "training-result.json",
            "checkpoint-final.pt",
            "checkpoint-final.pt.json",
            "metrics.jsonl",
            "source.zip",
        )
    }
    hashes = {str(path): _sha(path) for path in paths.values()}
    run, result, sidecar = (
        _read(paths["run.json"]),
        _read(paths["training-result.json"]),
        _read(paths["checkpoint-final.pt.json"]),
    )
    _require(
        run["config"] == config
        and run["identity"] == result["identity"] == sidecar["experiment_identity"]
        and run["identity"]["resolved_config_hash"]
        == config_hash(RootConfig.model_validate(config))
        and run["identity"]["environment"] == environment
        and run["identity"]["split_hash"] == context["split_hash"]
        and all(run["identity"][k] == v for k, v in context["corpus_identity"].items())
        and run["identity"]["seeds"] == [1729]
        and result["global_step"] == sidecar["global_step"] == 2000
        and sidecar["checkpoint_hash"]
        == result["checkpoint_hash"]
        == hashes[str(paths["checkpoint-final.pt"])]
        and hashes[str(paths["source.zip"])] == environment["source_archive_sha256"],
        "fresh training identity/final-step evidence",
    )
    metadata = sidecar["training_metadata"]
    active = config["model"]["symmetric_margin_loss_weight"] > 0
    _require(
        metadata["objective"] == "causal-next-token-v1"
        and metadata["model_config"] == config["model"]
        and ("symmetric_margin_objective" in metadata) == active
        and metadata.get("symmetric_margin_objective")
        == (SYMMETRIC_MARGIN_OBJECTIVE if active else None),
        "training objective contract",
    )
    _require(
        result["history"] and result["history"][-1]["step"] == 2000, "final diagnostics absent"
    )
    history = [
        json.loads(line) for line in paths["metrics.jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    _require(history == result["history"], "training history differs from metrics journal")
    if active:
        _require(
            all(
                k in result["history"][-1]
                for k in (
                    "validation_symmetric_margin_loss",
                    "validation_symmetric_margin_active_fraction",
                )
            ),
            "margin training diagnostics absent",
        )
    for path, digest in hashes.items():
        _bind(path, digest, context["inputs"])
    return {
        "arm": arm,
        "run_name": config["run_name"],
        "config": config,
        "config_hash": run["identity"]["resolved_config_hash"],
        "identity": run["identity"],
        "checkpoint": str(paths["checkpoint-final.pt"]),
        "checkpoint_hash": sidecar["checkpoint_hash"],
        "artifact_sha256": hashes,
        "training_history": result["history"],
        "training_wall_seconds": result["training_wall_seconds"],
        "global_step": 2000,
    }


def _evaluate(arm, training, context, output):
    import torch

    from plm.configuration import config_hash
    from plm.evaluation.set_composition import evaluate_set_compositions
    from plm.serving.parser import prompt_ids
    from plm.serving.runtime import load_inference_runtime
    from plm.serving.set_composition import generate_set_compositions

    runtime = None
    report = {
        "arm": arm,
        "complete": False,
        "query_count": 0,
        "responses": [],
        "cleanup_errors": [],
    }
    rows, compositions = report["responses"], []
    started = time.perf_counter()
    try:
        config = _evaluation_config(training["config"])
        runtime = load_inference_runtime(config, training["checkpoint"])
        _require(
            runtime.device == "cuda"
            and next(runtime.model.parameters()).dtype == torch.float32
            and not runtime.model.training
            and runtime.checkpoint_hash == training["checkpoint_hash"]
            and runtime.training_identity == training["identity"]
            and runtime.corpus_identity == context["corpus_identity"]
            and runtime.split.split_hash == context["split_hash"],
            "evaluation runtime identity/FP32",
        )
        tokens = [runtime.vocabulary.token_of(i) for i in range(len(runtime.vocabulary))]
        records = runtime.split.validation
        _require(len(records) == 222, "evaluation validation coverage")
        report.update(
            config=config.model_dump(mode="json"),
            config_hash=config_hash(config),
            checkpoint_hash=runtime.checkpoint_hash,
            training_identity=runtime.training_identity,
            corpus_identity=runtime.corpus_identity,
            split_hash=runtime.split.split_hash,
            product_token_ids=_COLUMNS,
            head_reference_batch_count=0,
            numerical_settings={
                "parameter_dtype": str(next(runtime.model.parameters()).dtype),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "autocast_cuda_enabled": torch.is_autocast_enabled("cuda"),
            },
        )
        for offset in range(0, len(records), 8):
            batch = records[offset : offset + 8]
            queries = [(r.subject, r.dimension) for r in batch]
            prompts = [list(prompt_ids(*query, runtime.vocabulary)) for query in queries]
            generated = generate_set_compositions(
                runtime.model,
                runtime.vocabulary,
                queries,
                max_new_tokens=507,
                device=runtime.device,
                first_target_guidance_alpha=16.0,
            )
            with torch.inference_mode():
                head = runtime.model(
                    torch.tensor(prompts, device=runtime.device, dtype=torch.long)
                ).symmetric_relation_logits
                _require(
                    head.dtype == torch.float32
                    and tuple(head.shape) == (len(batch), 1025)
                    and bool(torch.isfinite(head).all()),
                    "same-shape reference tensor",
                )
                vectors = head.cpu().tolist()
            report["head_reference_batch_count"] += 1
            for index, (record, result, logits, prompt) in enumerate(
                zip(batch, generated, vectors, prompts, strict=True), offset
            ):
                metadata = context["reference"]["responses"][index]
                _require(
                    (record.subject, record.dimension, sorted(record.targets))
                    == (metadata["subject"], metadata["dimension"], sorted(metadata["expected"])),
                    "evaluation query alignment",
                )
                _record_observation(
                    report,
                    index,
                    record,
                    metadata,
                    _jsonable(asdict(result)),
                    logits,
                    prompt,
                    tokens,
                    offset // 8,
                    len(batch),
                )
                compositions.append(result)
                report["query_count"] = len(rows)
            print(f"margin {arm}: {len(rows)}/222 validation queries", flush=True)
        report["metrics"] = evaluate_set_compositions(
            records, compositions, runtime.vocabulary
        ).to_dict()
        report["head_diagnostics"] = _head_totals(rows)
        report["groups"] = {}
        for group in _GROUPS:
            indices = [i for i, row in enumerate(rows) if row["group"] == group]
            report["groups"][group] = {
                "metrics": evaluate_set_compositions(
                    [records[i] for i in indices],
                    [compositions[i] for i in indices],
                    runtime.vocabulary,
                ).to_dict(),
                "head_diagnostics": _head_totals([rows[i] for i in indices]),
            }
        report["complete"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["evaluation_wall_seconds"] = time.perf_counter() - started
        runtime = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
            report["complete"] = False
        _write(output, report)
    _require(report["complete"], "evaluation cleanup failed")
    return report


def _receipt(path, digest, context, kind, source_sha):
    _require(path is not None and digest is not None, f"pinned {kind} receipt required")
    value = _read(_bind(path, digest, context["inputs"]))
    _require(
        value.get("passed") is True and value.get("source_archive_sha256") == source_sha,
        f"{kind} verification/source identity",
    )
    _require(value["implementation_source_sha256"], "missing implementation hashes")
    for name, expected in value["implementation_source_sha256"].items():
        _require(
            context["live_inventory"].get(name) == expected, "prerequisite implementation drift"
        )
    _require(
        value["config_files_sha256"]
        == {k: v for k, v in context["live_inventory"].items() if k.startswith("configs/")},
        "prerequisite config drift",
    )
    if kind == "CPU":
        _require(
            value.get("gpu_used") is False and value.get("experiment_training_executed") is False,
            "CPU receipt scope",
        )
        _require(
            value["checks"] and all(c["exit_code"] == 0 for c in value["checks"]),
            "CPU prerequisite failed",
        )
        for name, expected in value["tests_sha256"].items():
            _bind(context["root"] / name, expected, context["inputs"])
        receipt = value["archived_source_comparison"]
    else:
        _require(
            value.get("synthetic") is True
            and value.get("real_corpus_used") is False
            and type(value.get("steps")) is int
            and value["steps"] == 3,
            "GPU smoke scope",
        )
        receipt = value["raw_receipt"]
    linked = _bind(context["root"] / receipt["path"], receipt["sha256"], context["inputs"])
    if kind != "CPU":
        raw = _read(linked)
        _require(
            raw["complete"] is True
            and len(raw["steps"]) == 3
            and [r["step"] for r in raw["steps"]] == [1, 2, 3]
            and all(math.isfinite(r[k]) for r in raw["steps"] for k in ("loss", "margin")),
            "raw synthetic smoke incomplete/nonfinite",
        )
    return value


def _train_once(train, config, cleanup, errors):
    """Retain the training exception if allocator cleanup also fails."""
    try:
        train(config)
    except BaseException:
        try:
            cleanup()
        except Exception as exc:
            errors.append(repr(exc))
        raise
    cleanup()


def main():
    script = Path(__file__).resolve()
    script_bytes = script.read_bytes()
    root = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=root / "runs/learning/symmetric-margin-screen-v1/summary.json"
    )
    parser.add_argument(
        "--plan", type=Path, default=root / "docs/experiments/2026-09-25-symmetric-margin-plan.md"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--cpu-sha256")
    parser.add_argument("--gpu-smoke-receipt", type=Path)
    parser.add_argument("--gpu-smoke-sha256")
    args = parser.parse_args()
    _require(
        Path.cwd().resolve() == root and "torch" not in sys.modules,
        "standalone repository-root launch required",
    )
    output = args.out.resolve()
    context = _preflight(root, args.plan, output)
    context["inputs"][str(script)] = hashlib.sha256(script_bytes).hexdigest()
    _unchanged(context)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "training_executed": False,
                    "evaluation_executed": False,
                    "torch_imported": False,
                    "validation_query_count": 222,
                    "arms": [c["run_name"] for c in context["configs"]],
                }
            )
        )
        return
    source = _source_bytes(root)
    source_sha = hashlib.sha256(source).hexdigest()
    cpu_receipt = _receipt(args.cpu_receipt, args.cpu_sha256, context, "CPU", source_sha)
    gpu_receipt = _receipt(
        args.gpu_smoke_receipt, args.gpu_smoke_sha256, context, "GPU smoke", source_sha
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": script_bytes,
        "plan.md": args.plan.read_bytes(),
        "source.zip": source,
        "configs.zip": _archive(
            root, sorted(n for n in context["live_inventory"] if n.startswith("configs/"))
        ),
        "original-recipe.json": (
            root / "runs/national_dex_continuation_control_s1729_v1/run.json"
        ).read_bytes(),
        "cpu-receipt.json": args.cpu_receipt.read_bytes(),
        "gpu-smoke.json": args.gpu_smoke_receipt.read_bytes(),
    }
    for kind, receipt in (
        ("cpu-comparison", cpu_receipt["archived_source_comparison"]),
        ("gpu-smoke-raw", gpu_receipt["raw_receipt"]),
    ):
        snapshots[kind + ".json"] = (root / receipt["path"]).read_bytes()
    for suffix, payload in snapshots.items():
        with output.with_suffix("." + suffix).open("xb") as stream:
            stream.write(payload)
    result = {
        "campaign_version": "plm-symmetric-margin-stage1-v1",
        "plan_sha256": _PLAN_SHA,
        "complete": False,
        "stage_one_accepted": False,
        "independent_audit_status": "required",
        "seed": 1729,
        "original_recipe_sha256": _RUN_SHA,
        "historical_training_config_hash": _CONFIG_SHA,
        "historical_checkpoint_sha256": _CHECKPOINT_SHA,
        "historical_pair_summary_sha256": _PAIR_SHA,
        "historical_pair_seed_sha256": _PAIR_SEED_SHA,
        "historical_pair_exact_context": 191,
        "input_sha256": context["inputs"],
        "live_source_config_sha256": context["live_inventory"],
        "snapshot_sha256": {k: _sha(output.with_suffix("." + k)) for k in snapshots},
        "training": [],
        "evaluation": [],
        "report_sha256": {},
        "cleanup_errors": [],
        "limitations": (
            "Single-seed validation screen; no held-out test, "
            "serving-performance or default-promotion claim"
        ),
    }
    try:
        import torch

        from plm.config import RootConfig
        from plm.experimentation.provenance import capture_runtime_provenance
        from plm.training.runner import run_training

        _require(torch.cuda.is_available(), "declared GPU unavailable")
        revision, environment = capture_runtime_provenance()
        _require(
            environment["source_archive_sha256"] == source_sha
            and environment == cpu_receipt["environment"] == gpu_receipt["environment"],
            "runtime/prerequisite environment changed",
        )
        result.update(source_commit=revision, environment=environment)
        # Finish both fresh trainings before observing either arm's generated answers.
        for arm, config in zip(_ARMS, context["configs"], strict=True):
            result["phase"] = f"training_{arm}"
            _unchanged(context)
            _write(output.parent / f"training-config-{arm}.json", config)

            def cleanup():
                gc.collect()
                torch.cuda.empty_cache()

            _train_once(
                run_training, RootConfig.model_validate(config), cleanup, result["cleanup_errors"]
            )
            receipt = _training_receipt(arm, config, context, environment)
            _write(output.parent / f"training-{arm}.json", receipt)
            result["training"].append(receipt)
        _unchanged(context)
        seal = {
            "both_trainings_completed_before_evaluation": True,
            "source_archive_sha256": source_sha,
            "script_sha256": context["inputs"][str(script)],
            "training_receipt_sha256": {
                arm: _sha(output.parent / f"training-{arm}.json") for arm in _ARMS
            },
        }
        _write(output.parent / "training-seal.json", seal)
        _bind(
            output.parent / "training-seal.json",
            _sha(output.parent / "training-seal.json"),
            context["inputs"],
        )
        for arm, training in zip(_ARMS, result["training"], strict=True):
            result["phase"] = f"evaluation_{arm}"
            _unchanged(context)
            report = _evaluate(arm, training, context, output.parent / f"evaluation-{arm}.json")
            result["evaluation"].append({k: v for k, v in report.items() if k != "responses"})
        reports = [_read(output.parent / f"evaluation-{arm}.json") for arm in _ARMS]
        _require(
            reports[0]["numerical_settings"] == reports[1]["numerical_settings"],
            "cross-arm numerical settings",
        )
        result["gate"] = _gate(*reports)
        _unchanged(context)
        final_revision, final_environment = capture_runtime_provenance()
        _require(
            (revision, environment) == (final_revision, final_environment),
            "runtime changed during screen",
        )
        _require(
            all(
                _sha(output.with_suffix("." + suffix)) == digest
                for suffix, digest in result["snapshot_sha256"].items()
            ),
            "snapshot changed",
        )
        result.update(complete=True, phase="complete")
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["report_sha256"] = {
            p.name: _sha(p) for p in output.parent.glob("*.json") if p != output
        }
        _write(output, result)
    print(
        json.dumps(
            {
                "complete": True,
                "stage_one_accepted": False,
                "gate": result["gate"],
                "summary_sha256": _sha(output),
            }
        )
    )


if __name__ == "__main__":
    main()
