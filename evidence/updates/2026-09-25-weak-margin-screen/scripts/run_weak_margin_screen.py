"""Run the separately declared seed-1729 weak-margin paired screen, lambda=.001."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_PLAN = "3c0449a3154492ab4c053674b6cf76ad3a00e18974e07f05bfbb638ada0973d2"
_HELPER = "a5d50dc4b57277bde1cefb1cf533b4958b28b04eb515182f855b3a4aa81311b8"
_FAILED = "39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701"
_FAILED_AUDIT = "17afc9ca0fedf184ac9e9f2794947306883d4288860b22124cd9eb0c7751bcd9"
_PORTABLE = "d44e1c656a09f53f1b32f1495b9a39ebc35a03085b3fb0639dc5bdaa0b4c87ad"
_ACCEPTANCE = "a66863691b2573e5fa8e1a44549b561ab2b46aaece9033e2dba64e8161700aed"
_SOURCE = "e806549e1c778b5fc1529423e346ccf915b1484e3a5b322072fff5316c35ab6b"
_CPU = "240764f7686a47a057941c5151cf6ac070303c1958065aa523aed0bbcb0995ef"
_GPU = "2bb3716e49d25600181e34a427b6cbc21cf176f02d11e83179e55b325f8a9412"
_CONFIG = "6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148"
_ORIGINAL = "ec6c8e358586efa5808591ca0112ea13b4e52cae4b9891ccac82c299332b870d"
_NAMES = (
    "national_dex_rank_margin_weak_control_s1729_v1",
    "national_dex_rank_margin_m1_w0001_s1729_v1",
)
_ARMS = ("control", "treatment")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load_helper(path):
    # Authenticate bytes before importing; never replace the archived module's globals.
    _require(
        path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == _HELPER,
        "archived helper identity mismatch",
    )
    spec = importlib.util.spec_from_file_location("weak_margin_authenticated_helper", path)
    _require(spec is not None and spec.loader is not None, "helper import unavailable")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    _require(helper._sha(path) == _HELPER, "archived helper changed during import")
    return helper


def _comparability(configs, migrated):
    _require(len(configs) == 2, "two fixed arms required")
    for raw, name, weight in zip(configs, _NAMES, (0.0, 0.001), strict=True):
        _require(
            raw["run_name"] == name
            and type(raw["seed"]) is int
            and raw["seed"] == 1729
            and type(raw["model"]["symmetric_margin_loss_weight"]) is float
            and raw["model"]["symmetric_margin_loss_weight"] == weight
            and type(raw["model"]["symmetric_margin"]) is float
            and raw["model"]["symmetric_margin"] == 1.0,
            "fixed weak-margin recipe",
        )
        normalized = copy.deepcopy(raw)
        normalized["run_name"] = migrated["run_name"]
        normalized["model"]["symmetric_margin_loss_weight"] = 0.0
        _require(normalized == migrated, "recipe drift beyond run name and margin coefficient")


def _recipe(original):
    from plm.config import RootConfig
    from plm.configuration import validate_saved_config

    base = validate_saved_config(original["config"], _CONFIG).model_dump(mode="json")
    configs = []
    for name, weight in zip(_NAMES, (0.0, 0.001), strict=True):
        raw = copy.deepcopy(base)
        raw["run_name"] = name
        raw["model"]["symmetric_margin_loss_weight"] = weight
        raw["model"]["symmetric_margin"] = 1.0
        _require(RootConfig.model_validate(raw).model_dump(mode="json") == raw, "recipe coercion")
        configs.append(raw)
    _comparability(configs, base)
    return configs


def _preflight(root, plan, output, helper):
    from plm.configuration import validate_saved_config
    from plm.corpus import load_corpus_records, require_valid_artifacts
    from plm.evaluation.split import split_queries
    from plm.protocol.tokenizer import Vocabulary

    inputs = {}
    bind, read = helper._bind, helper._read
    bind(Path(helper.__file__), _HELPER, inputs)
    bind(plan, _PLAN, inputs)
    prior_dir = root / "runs/learning/symmetric-margin-screen-v1"
    failed = read(bind(prior_dir / "summary.json", _FAILED, inputs))
    audit = read(bind(prior_dir / "independent-audit.json", _FAILED_AUDIT, inputs))
    _require(
        failed["complete"] is True
        and failed["gate"]["numerical_gates_passed"] is False
        and audit["audit_passed"] is True
        and audit["all_recomputed_outputs_equal"] is True
        and audit["summary_sha256"] == _FAILED
        and failed["snapshot_sha256"]["script.py"] == _HELPER,
        "failed screen provenance",
    )
    bind(prior_dir / "independent-audit.py", audit["script_sha256"], inputs)
    portable_path = root / "docs/experiments/2026-09-25-margin-gradient-diagnosis.json"
    acceptance_path = root / "runs/learning/margin-gradient-diagnosis-v1/acceptance.json"
    portable = read(bind(portable_path, _PORTABLE, inputs))
    acceptance = read(bind(acceptance_path, _ACCEPTANCE, inputs))
    _require(
        acceptance["diagnostic_accepted"] is True
        and acceptance["model_promoted"] is False
        and portable["acceptance_sha256"] == _ACCEPTANCE
        and portable["audit"]["audit_passed"] is True,
        "gradient diagnosis acceptance binding",
    )
    live = helper._inventory(root)
    _require(
        live == failed["live_source_config_sha256"]
        and hashlib.sha256(helper._source_bytes(root)).hexdigest() == _SOURCE,
        "source/config must remain the verified margin implementation",
    )
    old_dir = root / "runs/national_dex_continuation_control_s1729_v1"
    original = read(bind(old_dir / "run.json", _ORIGINAL, inputs))
    _require(original["identity"]["resolved_config_hash"] == _CONFIG, "original config identity")
    # The previously authenticated manifest pins corpus and original training bytes.
    for name, digest in failed["input_sha256"].items():
        path = Path(name).resolve()
        if path.is_relative_to(root / "data") or path.parent == old_dir:
            bind(path, digest, inputs)
    pair_dir = root / "runs/learning/pair-unions-v1"
    pair = read(bind(pair_dir / "summary.json", helper._PAIR_SHA, inputs))
    pair_audit = read(bind(pair_dir / "independent-audit.json", helper._PAIR_AUDIT_SHA, inputs))
    reference = read(bind(pair_dir / "seed-1729.json", helper._PAIR_SEED_SHA, inputs))
    _require(
        pair["report_sha256"]["seed-1729.json"] == helper._PAIR_SEED_SHA
        and pair_audit["summary_sha256"] == helper._PAIR_SHA
        and pair_audit["all_arithmetic_and_provenance_equal"] is True,
        "pair metadata provenance",
    )
    bind(pair_dir / "independent-audit.py", pair_audit["script_sha256"], inputs)
    sidecar = read(old_dir / "checkpoint-final.pt.json")
    training = read(old_dir / "training-result.json")
    for name in (
        "run.json",
        "checkpoint-final.pt",
        "checkpoint-final.pt.json",
        "training-result.json",
    ):
        _require(str((old_dir / name).resolve()) in inputs, "unbound original artifact")
    _require(
        original["identity"]
        == reference["training_identity"]
        == sidecar["experiment_identity"]
        == training["identity"]
        and reference["checkpoint_hash"]
        == sidecar["checkpoint_hash"]
        == training["checkpoint_hash"]
        == helper._CHECKPOINT_SHA
        and sidecar["global_step"] == training["global_step"] == 2000
        and sidecar["training_metadata"]["model_config"] == original["config"]["model"],
        "original training identity chain",
    )
    configs = _recipe(original)
    helper._refuse(output, root, configs)
    migrated = validate_saved_config(original["config"], _CONFIG)
    corpus = root / Path(migrated.data.corpus_manifest).parent
    for path in (
        corpus / "manifest.json",
        corpus / "records.jsonl",
        corpus / "vocabulary.json",
        root / migrated.data.graph_db,
    ):
        _require(str(path.resolve()) in inputs, "unbound corpus input")
    report = require_valid_artifacts(corpus, graph_db=root / migrated.data.graph_db)
    _require(report.manifest is not None, "corpus manifest missing")
    split = split_queries(
        load_corpus_records(corpus),
        migrated.data.validation_query_fraction,
        migrated.data.test_query_fraction,
        migrated.data.split_seed,
    )
    vocabulary = Vocabulary.load(corpus / "vocabulary.json")
    corpus_identity = {
        "graph_hash": report.manifest.graph_hash,
        "records_hash": report.manifest.records_hash,
        "vocabulary_hash": report.manifest.tokenizer_hash,
    }
    _require(
        vocabulary.entity_ids() == list(range(1024, 2049))
        and len(vocabulary) == 2049
        and corpus_identity == reference["corpus_identity"]
        and split.split_hash == reference["split_hash"],
        "validation vocabulary/corpus/split identity",
    )
    _require(
        reference["seed"] == 1729
        and reference["query_count"] == len(reference["responses"]) == len(split.validation) == 222,
        "validation coverage",
    )
    seen = set()
    for record, row in zip(split.validation, reference["responses"], strict=True):
        key = (record.subject, record.dimension)
        truth = vocabulary.encode(record.targets)
        _require(
            key not in seen
            and key == (row["subject"], row["dimension"])
            and sorted(record.targets) == sorted(row["expected"])
            and row["group"] in helper._GROUPS
            and (record.dimension == "COLOR") == (row["group"] == "COLOR")
            and 0 < len(set(truth)) == len(truth) < 1024
            and vocabulary.id_of(record.subject) not in truth,
            "validation order/labels/group alignment",
        )
        seen.add(key)
    context = {
        "root": root,
        "helper": helper,
        "inputs": inputs,
        "live_inventory": live,
        "original": original,
        "reference": reference,
        "configs": configs,
        "corpus_identity": corpus_identity,
        "split_hash": split.split_hash,
        "failed": failed,
        "portable_path": portable_path,
        "acceptance_path": acceptance_path,
    }
    helper._unchanged(context)
    _require("torch" not in sys.modules, "CPU preflight imported Torch")
    return context


def _execute(context, output, result, train, make_config, cleanup, capture):
    """Sequential lifecycle with immutable failure receipts, also testable without Torch."""
    helper = context["helper"]
    try:
        revision, environment = capture()
        _require(
            environment
            == context["cpu_receipt"]["environment"]
            == context["gpu_receipt"]["environment"]
            and environment == context["failed"]["environment"]
            and environment["source_archive_sha256"] == _SOURCE,
            "prerequisite/runtime environment changed",
        )
        result.update(source_commit=revision, environment=environment)
        for arm, config in zip(_ARMS, context["configs"], strict=True):
            result["phase"] = f"training_{arm}"
            helper._unchanged(context)
            helper._write(output.parent / f"training-config-{arm}.json", config)
            helper._train_once(train, make_config(config), cleanup, result["cleanup_errors"])
            receipt = helper._training_receipt(arm, config, context, environment)
            helper._write(output.parent / f"training-{arm}.json", receipt)
            result["training"].append(receipt)
        helper._unchanged(context)
        seal = {
            "both_trainings_completed_before_evaluation": True,
            "source_archive_sha256": _SOURCE,
            "script_sha256": result["script_sha256"],
            "helper_sha256": _HELPER,
            "training_receipt_sha256": {
                arm: helper._sha(output.parent / f"training-{arm}.json") for arm in _ARMS
            },
        }
        helper._write(output.parent / "training-seal.json", seal)
        helper._bind(
            output.parent / "training-seal.json",
            helper._sha(output.parent / "training-seal.json"),
            context["inputs"],
        )
        for arm, training in zip(_ARMS, result["training"], strict=True):
            result["phase"] = f"evaluation_{arm}"
            helper._unchanged(context)
            report = helper._evaluate(
                arm, training, context, output.parent / f"evaluation-{arm}.json"
            )
            result["evaluation"].append({k: v for k, v in report.items() if k != "responses"})
        reports = [helper._read(output.parent / f"evaluation-{arm}.json") for arm in _ARMS]
        _require(
            reports[0]["numerical_settings"] == reports[1]["numerical_settings"],
            "cross-arm numerical settings",
        )
        result["gate"] = helper._gate(*reports)
        helper._unchanged(context)
        _require(capture() == (revision, environment), "runtime changed during screen")
        _require(
            all(
                helper._sha(output.with_suffix("." + k)) == v
                for k, v in result["snapshot_sha256"].items()
            ),
            "snapshot changed",
        )
        result.update(complete=True, phase="complete")
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["report_sha256"] = {
            p.name: helper._sha(p) for p in output.parent.glob("*.json") if p != output
        }
        helper._write(output, result)


def main():
    script = Path(__file__).resolve()
    script_bytes = script.read_bytes()
    root = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=root / "runs/learning/weak-margin-screen-v1/summary.json"
    )
    parser.add_argument(
        "--plan", type=Path, default=root / "docs/experiments/2026-09-25-weak-margin-plan.md"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--cpu-receipt",
        type=Path,
        default=root / "runs/learning/margin-default-contract-v1/cpu-prerequisite.json",
    )
    parser.add_argument("--cpu-sha256", default=_CPU)
    parser.add_argument(
        "--gpu-smoke-receipt",
        type=Path,
        default=root / "runs/learning/margin-default-contract-v1/gpu-prerequisite.json",
    )
    parser.add_argument("--gpu-smoke-sha256", default=_GPU)
    args = parser.parse_args()
    _require(
        Path.cwd().resolve() == root and "torch" not in sys.modules,
        "standalone repository-root launch required",
    )
    helper = _load_helper(root / "runs/learning/symmetric-margin-screen-v1/summary.script.py")
    output = args.out.resolve()
    context = _preflight(root, args.plan, output, helper)
    context["inputs"][str(script)] = hashlib.sha256(script_bytes).hexdigest()
    _require(
        args.cpu_sha256 == _CPU and args.gpu_smoke_sha256 == _GPU,
        "fixed prerequisite receipt identities",
    )
    cpu = helper._receipt(args.cpu_receipt, _CPU, context, "CPU", _SOURCE)
    gpu = helper._receipt(args.gpu_smoke_receipt, _GPU, context, "GPU smoke", _SOURCE)
    _require(
        cpu["environment"] == gpu["environment"] == context["failed"]["environment"],
        "prerequisite environments disagree",
    )
    context.update(cpu_receipt=cpu, gpu_receipt=gpu)
    helper._unchanged(context)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "torch_imported": False,
                    "training_executed": False,
                    "evaluation_executed": False,
                    "validation_query_count": 222,
                    "arms": list(_NAMES),
                    "treatment_coefficient": 0.001,
                    "helper_sha256": _HELPER,
                }
            )
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": script_bytes,
        "helper.py": Path(helper.__file__).read_bytes(),
        "plan.md": args.plan.read_bytes(),
        "source.zip": helper._source_bytes(root),
        "configs.zip": helper._archive(
            root, sorted(n for n in context["live_inventory"] if n.startswith("configs/"))
        ),
        "original-recipe.json": (
            root / "runs/national_dex_continuation_control_s1729_v1/run.json"
        ).read_bytes(),
        "cpu-receipt.json": args.cpu_receipt.read_bytes(),
        "gpu-smoke.json": args.gpu_smoke_receipt.read_bytes(),
        "cpu-comparison.json": (root / cpu["archived_source_comparison"]["path"]).read_bytes(),
        "gpu-smoke-raw.json": (root / gpu["raw_receipt"]["path"]).read_bytes(),
        "failed-screen.json": (
            root / "runs/learning/symmetric-margin-screen-v1/summary.json"
        ).read_bytes(),
        "gradient-portable.json": context["portable_path"].read_bytes(),
        "gradient-acceptance.json": context["acceptance_path"].read_bytes(),
    }
    for name, content in snapshots.items():
        with output.with_suffix("." + name).open("xb") as stream:
            stream.write(content)
    result = {
        "campaign_version": "plm-weak-margin-stage1-v1",
        "plan_sha256": _PLAN,
        "script_sha256": context["inputs"][str(script)],
        "helper_sha256": _HELPER,
        "failed_screen_summary_sha256": _FAILED,
        "gradient_portable_sha256": _PORTABLE,
        "gradient_acceptance_sha256": _ACCEPTANCE,
        "complete": False,
        "stage_one_accepted": False,
        "independent_audit_status": "required",
        "seed": 1729,
        "original_recipe_sha256": _ORIGINAL,
        "historical_training_config_hash": _CONFIG,
        "historical_checkpoint_sha256": helper._CHECKPOINT_SHA,
        "historical_pair_summary_sha256": helper._PAIR_SHA,
        "historical_pair_seed_sha256": helper._PAIR_SEED_SHA,
        "historical_pair_exact_context": 191,
        "input_sha256": context["inputs"],
        "live_source_config_sha256": context["live_inventory"],
        "snapshot_sha256": {k: helper._sha(output.with_suffix("." + k)) for k in snapshots},
        "training": [],
        "evaluation": [],
        "report_sha256": {},
        "cleanup_errors": [],
        "limitations": (
            "Adaptive single-seed development screen; no unbiased generalization, "
            "test, throughput or default-promotion claim."
        ),
    }

    # Imports happen only after immutable snapshots and every CPU prerequisite check.
    def capture():
        import torch

        from plm.experimentation.provenance import capture_runtime_provenance

        _require(torch.cuda.is_available(), "declared GPU unavailable")
        return capture_runtime_provenance()

    def train(config):
        from plm.training.runner import run_training

        return run_training(config)

    def make_config(raw):
        from plm.config import RootConfig

        return RootConfig.model_validate(raw)

    def cleanup():
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    _execute(context, output, result, train, make_config, cleanup, capture)
    print(
        json.dumps(
            {
                "complete": True,
                "stage_one_accepted": False,
                "gate": result["gate"],
                "summary_sha256": helper._sha(output),
            }
        )
    )


if __name__ == "__main__":
    main()
