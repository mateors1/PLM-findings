"""Independent failure-only audit. Reads metadata/source; never imports Torch."""

import argparse
import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/learning/symmetric-affine8000-v1"
HERE = Path(__file__).resolve().parent
DIAGNOSIS_SHA = "6fafc78ad3bb24c4f9762e7516639d6ff73adb70044816bbf8c50df3d721d696"
DIAGNOSIS_SOURCE_SHA = "6886cc37c5317dcadaecfbaffcd2ef0f08b423717501a080fa48320c6a5da935"
SUMMARY_SHA = "4f1471ec7880d1718ed08d1fabc6bb414f7e55f48b52cbe9ad604e57193425ec"
TRAINING_SHA = "40e783b5380236f56cc21e85563e99d82418a35c45a14ed42cf214d285f1b591"
STDOUT_SHA = "c2e1f5ce1e9ac4e135c110310c78b733a34fcf1f6587468d25ada1491f3d4dd9"
RUNNER_SHA = "a9899d4c494198afcf35346903f93bc39d5771d60a6711dc164d12a42ad11568"
SCORER_SHA = "8c665818bbfb4306f954ce8b0d9cf67e64e1de2b6f883ee087f3fbe8b3037fbe"
PLAN_SHA = "d102c6d00ab536b89c0a0021097976fb997831184a4e4c2db82d172a4c61a19a"
AMENDMENT_SHA = "bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f"
RECIPE_SHA = "5c86e154ce18078a2a12a9ff8d5ab619cfc02227b61ee09dbfadaa1cd3563035"
SUCCESS_AUDITOR_SHA = "00551f6d2fe05e193f7ed67481b9decc65f7c1264c78e3edac7b9a9454f2d14f"
SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
ERROR = "ValueError('parent checkpoint identity')"
SNAPSHOTS = {
    "script.py",
    "affine-helper.py",
    "bilinear-helper.py",
    "budget-helper.py",
    "mean-helper.py",
    "helper.py",
    "source.zip",
    "configs.zip",
    "plan.md",
    "input-amendment.md",
    "recipe.json",
    "inputs.json",
    "test-receipt.json",
    "test-stdout.txt",
}
ABSENT = {
    "parent.json",
    "zero-replay.json",
    "child.json",
    "comparisons.json",
    "train-membership.json",
    "checkpoint-final.pt",
    "checkpoint-final.pt.json",
    "run.json",
    "failure-state.pt",
    "failure-state.json",
    "independent-audit.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(summary, training, receipt, stdout):
    require(summary["campaign_version"] == "symmetric-affine8000-v1", "campaign identity")
    require(summary["complete"] is False and training["complete"] is False, "failed lifecycle")
    require(
        summary["acceptance"] is False and summary["standard_serving_supported"] is False,
        "no acceptance or serving promotion",
    )
    require(summary["error"] == training["error"] == ERROR, "same admission failure")
    require(
        type(training["completed_updates"]) is int and training["completed_updates"] == 0,
        "zero completed updates",
    )
    require(training["history"] == [], "empty optimization history")
    require(summary["artifact_sha256"] == {"training.json": TRAINING_SHA}, "only training receipt")
    require(
        summary["failure_state_availability"]
        == {"model_available": False, "optimizer_available": False},
        "outer frame availability",
    )
    require(
        not ({"gate", "child", "parent", "child_training_identity", "queries"} & set(summary)),
        "no completed prediction or child result",
    )
    require(
        type(receipt["exit_code"]) is int and receipt["exit_code"] == 1,
        "observed primary failure exit",
    )
    require(
        receipt["terminal_completion_observed_by_primary"] is True
        and receipt["terminal_chunk_id"] == "50ba53",
        "terminal observation",
    )
    require(
        receipt["summary_sha256"] == SUMMARY_SHA
        and receipt["training_sha256"] == TRAINING_SHA
        and receipt["stdout_sha256"] == STDOUT_SHA,
        "execution artifact hashes",
    )
    require(receipt["candidate_quality_result"] is False, "failure is not quality result")
    require(
        "line 965, in _execute" in stdout
        and "line 466, in _load_selected_runtime" in stdout
        and stdout.rstrip().endswith("ValueError: parent checkpoint identity"),
        "failure stack",
    )


def source_contract(source, checkpoint_source, trainer_source):
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    loader = ast.get_source_segment(source, functions["_load_selected_runtime"])
    execute = ast.get_source_segment(source, functions["_execute"])
    stages = [
        "model = build_model(model_config)",
        'state = load_checkpoint(context["parent"], model, map_location="cpu", restore_rng=False)',
        "validate_checkpoint_identity(",
        "raw = state.metadata",
        'and raw["config"] == context["parent_run"]["config"]',
        '"parent checkpoint identity",',
        "model.to(device).eval()",
        "return SimpleNamespace(",
    ]
    positions = [loader.index(stage) for stage in stages]
    require(positions == sorted(positions), "loader stage ordering")
    require(
        execute.index("runtime = _load_selected_runtime(context)")
        < execute.index("model = runtime.model.float().eval()")
        < execute.index("parent = _evaluate("),
        "outer binding and prediction order",
    )
    require(
        '"model_available": "model" in locals()' in execute and 'if "model" in locals()' in execute,
        "outer-frame failure semantics",
    )
    require(
        'raw["training_metadata"]["objective"] == "causal-next-token-v1"' in loader,
        "later objective predicate retained",
    )
    require(
        'identity["evaluator_version"] == "plm-train-causal-v2"' in loader,
        "later evaluator predicate retained",
    )
    require(
        "if metadata_raw.get(key) != expected:" in checkpoint_source
        and 'raise ValueError(f"checkpoint sidecar {key} does not match checkpoint payload")'
        in checkpoint_source,
        "payload-sidecar identity check",
    )
    require(
        checkpoint_source.index("if metadata_raw.get(key) != expected:")
        < checkpoint_source.index('model.load_state_dict(payload["model"])')
        < checkpoint_source.index(
            "return CheckpointState(target, raw_step, digest, authoritative)"
        ),
        "authoritative metadata is payload matched to sidecar",
    )
    require(
        "config=self.config" in trainer_source and "TrainConfig" in trainer_source,
        "archived trainer checkpoints training configuration",
    )
    return {
        "loader_function_line": functions["_load_selected_runtime"].lineno,
        "outer_execute_line": functions["_execute"].lineno,
        "failing_assertion_line": 466,
        "outer_call_line": 965,
    }


def config_diagnosis(sidecar, run):
    require(sidecar["config"] != run["config"], "whole RootConfig comparison fails")
    require(sidecar["config"] == run["config"]["train"], "TrainConfig identity matches")
    return {
        "sidecar_config_equals_run_root_config": False,
        "sidecar_config_equals_run_train_config": True,
        "payload_config_matches_sidecar": (
            "Enforced by pinned loader before returning; inferred from observed "
            "continuation, not reloaded by this audit."
        ),
        "later_short_circuited_predicates_independently_executed": False,
    }


def audit():
    inputs = {}
    allowed = {
        RUN / name
        for name in (
            "summary.json",
            "training.json",
            "execution-receipt.json",
            "primary-stdout.txt",
            "independent-audit.py",
        )
    }
    allowed.update(RUN / ("summary." + name) for name in SNAPSHOTS)
    diagnostic_dir = ROOT / "runs/learning/symmetric-affine8000-failure-diagnosis-v1"
    allowed.update(
        diagnostic_dir / name
        for name in ("diagnosis.json", "diagnose.py", "execution-receipt.json")
    )
    parent_dir = ROOT / "runs/national_dex_continuation_control_s1729_v1"
    allowed.update(parent_dir / name for name in ("run.json", "checkpoint-final.pt.json"))

    def bind(path, expected):
        path = path.resolve()
        require(path in allowed, "unapproved failure audit input")
        actual = sha(path)
        require(actual == expected, "failure evidence hash: " + str(path))
        inputs[str(path)] = actual
        return path

    def read(path, expected):
        return json.loads(bind(path, expected).read_text(encoding="utf-8"))

    summary = read(RUN / "summary.json", SUMMARY_SHA)
    training = read(RUN / "training.json", TRAINING_SHA)
    receipt_path = RUN / "execution-receipt.json"
    receipt = read(receipt_path, sha(receipt_path))
    stdout = bind(RUN / "primary-stdout.txt", STDOUT_SHA).read_text(encoding="utf-8")
    contract(summary, training, receipt, stdout)
    require(
        {str(RUN / ("summary." + name)) for name in SNAPSHOTS} == set(summary["snapshot_sha256"]),
        "complete fixed snapshot inventory",
    )
    for path in summary["snapshot_sha256"]:
        require(Path(path).resolve() in allowed, "snapshot path admission before read")
    for path, expected in summary["snapshot_sha256"].items():
        bind(Path(path), expected)
    for name, expected in {
        "script.py": RUNNER_SHA,
        "affine-helper.py": SCORER_SHA,
        "plan.md": PLAN_SHA,
        "input-amendment.md": AMENDMENT_SHA,
        "recipe.json": RECIPE_SHA,
        "source.zip": SOURCE_SHA,
        "configs.zip": CONFIGS_SHA,
    }.items():
        bind(RUN / ("summary." + name), expected)
    frozen_inputs = json.loads((RUN / "summary.inputs.json").read_text(encoding="utf-8"))
    require(frozen_inputs == summary["input_sha256"], "input manifest snapshot identity")
    # These manifests are historical data, not instructions to reopen dependencies.
    test_receipt = json.loads((RUN / "summary.test-receipt.json").read_text(encoding="utf-8"))
    require(
        test_receipt["passed"] is True
        and type(test_receipt["exit_code"]) is int
        and test_receipt["exit_code"] == 0
        and test_receipt["tests_passed"] == 192
        and test_receipt["tests_skipped"] == 0
        and test_receipt["terminal_completion_observed"] is True
        and test_receipt["terminal_chunk_id"] == "528ffe"
        and test_receipt["tested_script_sha256"] == RUNNER_SHA
        and test_receipt["tested_recipe_sha256"] == RECIPE_SHA,
        "frozen primary test evidence",
    )
    bind(RUN / "summary.test-stdout.txt", test_receipt["stdout"]["sha256"])
    bind(RUN / "independent-audit.py", SUCCESS_AUDITOR_SHA)
    runtime = RUN / "runtime"
    archive_files = {}
    for name in ("source.zip", "configs.zip"):
        with zipfile.ZipFile(RUN / ("summary." + name)) as archive:
            for entry in archive.infolist():
                relative = entry.filename
                require(
                    not entry.is_dir()
                    and not relative.startswith("/")
                    and ".." not in relative.split("/")
                    and "\\" not in relative
                    and ":" not in relative
                    and relative not in archive_files,
                    "safe disjoint archive entries",
                )
                require(
                    relative.startswith(("src/", "configs/"))
                    or relative in ("pyproject.toml", "uv.lock"),
                    "source/config archive scope",
                )
                archive_files[relative] = hashlib.sha256(archive.read(relative)).hexdigest()
    expected_runtime = {str(runtime / name): value for name, value in archive_files.items()}
    require(summary["runtime_files_sha256"] == expected_runtime, "authenticated runtime inventory")
    for path in expected_runtime:
        allowed.add(Path(path).resolve())
    for path, digest in expected_runtime.items():
        bind(Path(path), digest)
    existing_runtime = {
        str(p) for p in runtime.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    require(existing_runtime == set(expected_runtime), "no extra runtime source files")
    lines = source_contract(
        (RUN / "summary.script.py").read_text(encoding="utf-8"),
        (runtime / "src/plm/training/checkpoint.py").read_text(encoding="utf-8"),
        (runtime / "src/plm/training/trainer.py").read_text(encoding="utf-8"),
    )
    metadata = {
        name: read(parent_dir / name, frozen_inputs[str(parent_dir / name)])
        for name in ("run.json", "checkpoint-final.pt.json")
    }
    diagnosis = config_diagnosis(metadata["checkpoint-final.pt.json"], metadata["run.json"])
    separate_diagnosis = read(diagnostic_dir / "diagnosis.json", DIAGNOSIS_SHA)
    bind(diagnostic_dir / "diagnose.py", DIAGNOSIS_SOURCE_SHA)
    diagnostic_receipt = read(
        diagnostic_dir / "execution-receipt.json", sha(diagnostic_dir / "execution-receipt.json")
    )
    require(
        diagnostic_receipt["diagnosis_sha256"] == DIAGNOSIS_SHA
        and diagnostic_receipt["script_sha256"] == DIAGNOSIS_SOURCE_SHA
        and type(diagnostic_receipt["exit_code"]) is int
        and diagnostic_receipt["exit_code"] == 0
        and diagnostic_receipt["terminal_completion_observed"] is True,
        "separate CPU diagnosis receipt",
    )
    predicates = separate_diagnosis["admission_predicates"]
    require(
        len(predicates) == 9
        and predicates["checkpoint_config_equals_run_root_config"] is False
        and all(
            v is True
            for k, v in predicates.items()
            if k != "checkpoint_config_equals_run_root_config"
        ),
        "sole config failure in later CPU diagnosis",
    )
    require(
        separate_diagnosis["verified_repair_equivalence"][
            "checkpoint_config_equals_run_train_config"
        ]
        is True
        and all(
            v is True
            for v in separate_diagnosis["cpu_checkpoint_load"][
                "sidecar_payload_metadata_equal"
            ].values()
        ),
        "separate payload metadata equality",
    )
    require(
        separate_diagnosis["neural_forward_executed"] is False
        and separate_diagnosis["training_executed"] is False
        and separate_diagnosis["cuda_initialized_after_cpu_load"] is False,
        "CPU-only diagnostic scope",
    )
    missing = sorted(name for name in ABSENT if not (RUN / name).exists())
    require(set(missing) == ABSENT, "unexpected result or child weight artifact")
    require(not list(RUN.glob("*.pt")), "no candidate or failure checkpoint")
    require("torch" not in sys.modules, "failure auditor remains Torch-free")
    for path, digest in inputs.items():
        require(sha(Path(path)) == digest, "evidence changed during failure audit")
    return {
        "schema_version": 1,
        "audit_id": "symmetric-affine8000-failure-audit-v1",
        "campaign": "symmetric-affine8000-v1",
        "complete": True,
        "audit_passed": True,
        "failure_evidence_verified": True,
        "experiment_completed": False,
        "candidate_quality_result": False,
        "hypothesis_supported": None,
        "hypothesis_rejected": None,
        "quality_gate_evaluated": False,
        "source_sha256": sha(Path(__file__)),
        "summary_sha256": SUMMARY_SHA,
        "execution_receipt_sha256": inputs[str(receipt_path)],
        "training_sha256": TRAINING_SHA,
        "stdout_sha256": STDOUT_SHA,
        "frozen_success_auditor_sha256": SUCCESS_AUDITOR_SHA,
        "separate_cpu_predicate_diagnosis": {
            "sha256": DIAGNOSIS_SHA,
            "script_sha256": DIAGNOSIS_SOURCE_SHA,
            "execution_receipt_sha256": inputs[str(diagnostic_dir / "execution-receipt.json")],
            "nine_predicates_reported": predicates,
            "payload_not_reloaded_by_this_auditor": True,
        },
        "failure": {
            "exception": ERROR,
            "stage": "parent-runtime-admission",
            "diagnosis": diagnosis,
            "source_locations": lines,
        },
        "stage_evidence": {
            "runtime_source_extracted_and_verified": True,
            "runtime_source_file_count": len(archive_files),
            "parent_constructed_on_cpu": (
                "Reached before observed failing assertion in pinned source."
            ),
            "parent_checkpoint_loaded_on_cpu": (
                "Reached before observed failing assertion; this audit did not reload it."
            ),
            "payload_sidecar_matching_and_original_identity_validation": (
                "Returned before later admission assertion."
            ),
            "model_assigned_to_outer_execute_frame": False,
            "model_transferred_to_cuda": False,
            "model_forward_reached": False,
            "affine_parameters_attached": False,
            "optimizer_created": False,
            "training_updates": 0,
            "candidate_predictions": False,
            "parent_replay": False,
            "zero_replay": False,
            "final_checkpoint": False,
            "reload": False,
            "quality_gate": False,
        },
        "absent_stage_artifacts": missing,
        "bound_input_sha256": inputs,
        "frozen_primary_input_manifest_entries": len(frozen_inputs),
        "historical_input_dependencies_reopened": False,
        "checkpoint_payload_loaded_by_failure_auditor": False,
        "torch_imported_by_failure_auditor": False,
        "protected_examples_opened_by_failure_auditor": False,
        "limitations": [
            (
                "Failure evidence is accepted for a software lifecycle diagnosis, "
                "not a model-quality result."
            ),
            (
                "Stage reachability follows the pinned executed source and observed "
                "exception, not a new execution."
            ),
            (
                "Original run queried/seeds Torch/CUDA environment; no model "
                "transfer or neural forward was reached."
            ),
            (
                "The compound assertion short-circuited at config comparison; later "
                "predicates need separate verification."
            ),
            (
                "No success-audit relaxation, candidate retry, weights backup, "
                "oracle parity or serving claim."
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "report.json")
    args = parser.parse_args()
    require(not args.out.exists(), "immutable failure audit output")
    report = audit()
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "report_sha256": sha(args.out),
                "candidate_quality_result": False,
                "training_updates": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
