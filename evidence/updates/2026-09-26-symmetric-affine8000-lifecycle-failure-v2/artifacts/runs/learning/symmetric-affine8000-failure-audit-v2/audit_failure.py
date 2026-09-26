"""Failure-only V2 audit: fixed metadata, source and CPU diagnostic state."""

import ast
import hashlib
import inspect
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/learning/symmetric-affine8000-v2"
HERE = Path(__file__).resolve().parent
SUMMARY = "3939855979f2418f2ef2faefae2823d5a0c90b32b30ae106f963ce2c1bcab6b1"
TRAINING = "2ae347a9a4d38ee94c14c41d0168ae4227ce9ba2f8b6f48e8c3d2dbd6534c33f"
STDOUT = "86dd0916468310ad9649ba8a7d2e9b4a0eef9f2f9790af045054ecaff7098143"
RUNNER = "aab0e064a9c9729af4926626c671f28905d90aa6c29eff0cd10737222d911314"
FAILURE_STATE = "546e4d5274cb8f581260d690e0ae8e0bc0a898f33fe50a2510ff5a7d95c9e746"
HISTORICAL = "e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8"
SUCCESS_AUDITOR = "6b0eb73eb4985c4482221bb7be1f385a0df58a4232d61855e316e117ac8b40f1"
ERROR = "TypeError(\"_evaluate() missing 1 required positional argument: 'bilinear'\")"
SNAPSHOTS = {
    "script.py",
    "affine-helper.py",
    "bilinear-helper.py",
    "budget-helper.py",
    "mean-helper.py",
    "helper.py",
    "plan.md",
    "input-amendment.md",
    "runtime-repair.md",
    "recipe.json",
    "inputs.json",
    "source.zip",
    "configs.zip",
    "test-receipt.json",
    "test-stdout.txt",
}
ABSENT = {
    "parent.json",
    "zero-replay.json",
    "child.json",
    "comparisons.json",
    "run.json",
    "checkpoint-final.pt",
    "checkpoint-final.pt.json",
    "independent-audit.json",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call_diagnosis(source):
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    definition = functions["_evaluate"]
    args = definition.args
    required = len(args.args) - len(args.defaults)
    signature = inspect.Signature(
        [
            inspect.Parameter(
                arg.arg,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty if i < required else None,
            )
            for i, arg in enumerate(args.args)
        ]
    )
    calls = [
        n
        for n in ast.walk(functions["_execute"])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_evaluate"
    ]
    parent_calls = [
        n
        for n in calls
        if any(
            k.arg == "phase" and isinstance(k.value, ast.Constant) and k.value.value == "parent"
            for k in n.keywords
        )
    ]
    require(len(parent_calls) == 1, "one parent evaluator call")
    call = parent_calls[0]
    names = [ast.unparse(n) for n in call.args]
    require(
        [a.arg for a in args.args[:8]]
        == ["model", "records", "wide_rows", "mean", "helper", "path", "affine", "bilinear"],
        "frozen callee schema",
    )
    require(names[3] == "helper" and len(names) == 7, "missing mean argument before helper")
    try:
        signature.bind(*[object() for _ in call.args], **{k.arg: object() for k in call.keywords})
    except TypeError as exc:
        require("'bilinear'" in str(exc), "same binding failure")
        reason = str(exc)
    else:
        raise ValueError("call unexpectedly binds")
    return {
        "callee_line": definition.lineno,
        "call_line": call.lineno,
        "signature": str(signature),
        "positional_arguments": names,
        "independent_binding_error": reason,
        "helper_body_entered": False,
    }


def failure_contract(summary, training, receipt, sidecar):
    require(summary["campaign_version"] == "symmetric-affine8000-v2", "campaign")
    require(
        summary["complete"] is False
        and training["complete"] is False
        and summary["acceptance"] is False,
        "failed nonaccepted attempt",
    )
    require(summary["error"] == training["error"] == sidecar["error"] == ERROR, "same error")
    require(
        type(training["completed_updates"]) is int
        and training["completed_updates"] == 0
        and training["history"] == [],
        "no updates or losses",
    )
    require(
        summary["failure_state_availability"]
        == {"model_available": True, "optimizer_available": False},
        "outer state",
    )
    require(
        type(receipt["exit_code"]) is int
        and receipt["exit_code"] == 1
        and receipt["terminal_completion_observed_by_primary"] is True
        and receipt["terminal_chunk_id"] == "47269c",
        "observed failed terminal",
    )
    require(
        receipt["summary_sha256"] == SUMMARY
        and receipt["training_sha256"] == TRAINING
        and receipt["stdout_sha256"] == STDOUT,
        "terminal artifact identity",
    )
    require(
        receipt["candidate_quality_result"] is False
        and "gate" not in summary
        and "child" not in summary,
        "no quality result",
    )
    require(
        sidecar["path"] == "failure-state.pt"
        and sidecar["sha256"] == FAILURE_STATE
        and sidecar["diagnostic_only"] is True
        and sidecar["resume_allowed"] is False
        and sidecar["completed_updates"] == sidecar["optimizer_steps_attempted"] == 0,
        "diagnostic checkpoint only",
    )


def inspect_failure_payload(payload, expected):
    import torch

    require(
        set(payload)
        == {
            "model_state",
            "optimizer_state",
            "completed_updates",
            "optimizer_steps_attempted",
            "diagnostic_only",
            "resume_allowed",
        },
        "failure payload schema",
    )
    require(
        payload["optimizer_state"] is None
        and payload["diagnostic_only"] is True
        and payload["resume_allowed"] is False
        and payload["completed_updates"] == 0
        and payload["optimizer_steps_attempted"] == 0,
        "untrained diagnostic payload",
    )
    state = payload["model_state"]
    require(
        len(state) == len(expected) == 93 and set(state) == set(expected), "original93 inventory"
    )
    require(
        not {"symmetric_bilinear_residual", "symmetric_affine_linear", "symmetric_affine_bias"}
        & set(state),
        "residual not attached",
    )
    for name, value in state.items():
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and torch.isfinite(value).all().item(),
            "finite CPU state",
        )
        manifest = {
            "sha256": hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        require(manifest == expected[name], "original parent tensor bytes: " + name)
    require(not torch.cuda.is_initialized(), "CPU audit did not initialize CUDA")
    return {
        "tensor_count": 93,
        "unchanged_original_tensors": 93,
        "residual_tensors": 0,
        "optimizer_state": None,
        "kind": "diagnostic-original-parent-state",
        "new_trained_model_checkpoint": False,
    }


def audit():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment")
    allowed = {
        RUN / name
        for name in (
            "summary.json",
            "training.json",
            "execution-receipt.json",
            "primary-stdout.txt",
            "failure-state.json",
            "failure-state.pt",
            "train-membership.json",
            "independent-audit.py",
        )
    }
    allowed.update(RUN / ("summary." + name) for name in SNAPSHOTS)
    historical = ROOT / "runs/learning/bilinear-budget8000-v1"
    allowed.update(historical / name for name in ("summary.json", "training.json"))
    inputs = {}

    def bind(path, expected):
        path = path.resolve()
        require(path in allowed, "explicit failure input allowlist")
        require(sha(path) == expected, "failure evidence hash: " + str(path))
        inputs[str(path)] = expected
        return path

    def read(path, expected):
        return json.loads(bind(path, expected).read_text(encoding="utf-8"))

    summary = read(RUN / "summary.json", SUMMARY)
    training = read(RUN / "training.json", TRAINING)
    receipt_path = RUN / "execution-receipt.json"
    receipt = read(receipt_path, sha(receipt_path))
    sidecar = read(RUN / "failure-state.json", summary["artifact_sha256"]["failure-state.json"])
    failure_contract(summary, training, receipt, sidecar)
    require(
        set(summary["artifact_sha256"])
        == {"training.json", "train-membership.json", "failure-state.json", "failure-state.pt"},
        "artifacts",
    )
    require(
        set(summary["snapshot_sha256"]) == {str(RUN / ("summary." + n)) for n in SNAPSHOTS},
        "exact15 snapshot paths",
    )
    for path in summary["snapshot_sha256"]:
        require(Path(path).resolve() in allowed, "snapshot admitted before opens")
    for path, digest in summary["snapshot_sha256"].items():
        bind(Path(path), digest)
    for name, digest in summary["artifact_sha256"].items():
        bind(RUN / name, digest)
    bind(RUN / "summary.script.py", RUNNER)
    bind(RUN / "independent-audit.py", SUCCESS_AUDITOR)
    require(
        read(
            RUN / "summary.inputs.json",
            summary["snapshot_sha256"][str(RUN / "summary.inputs.json")],
        )
        == summary["input_sha256"],
        "historical input snapshot identity only",
    )
    stdout = bind(RUN / "primary-stdout.txt", STDOUT).read_text(encoding="utf-8")
    require(
        "line 1018, in _execute" in stdout
        and stdout.rstrip().endswith(
            "TypeError: _evaluate() missing 1 required positional argument: 'bilinear'"
        ),
        "failure stack",
    )
    source = (RUN / "summary.script.py").read_text(encoding="utf-8")
    require(
        source.index("runtime = _load_selected_runtime(context)")
        < source.index("parent = _evaluate(")
        < source.index("affine.attach(model)"),
        "pre-residual failure ordering",
    )
    require(
        "model.to(device).eval()" in source
        and '_require(device == "cuda", "historical CUDA device")' in source,
        "returned parent moved to CUDA",
    )
    primary_tests = json.loads((RUN / "summary.test-receipt.json").read_text(encoding="utf-8"))
    require(
        primary_tests["passed"] is True
        and primary_tests["exit_code"] == 0
        and primary_tests["terminal_completion_observed"] is True
        and primary_tests["tests_passed"] == 204
        and primary_tests["tests_skipped"] == 0,
        "frozen primary tests",
    )
    diagnosis = call_diagnosis(source)
    require(diagnosis["call_line"] == 1018, "actual call location")
    runtime = RUN / "runtime"
    archive_files = {}
    for name in ("source.zip", "configs.zip"):
        with zipfile.ZipFile(RUN / ("summary." + name)) as archive:
            for entry in archive.infolist():
                relative = entry.filename
                require(
                    not entry.is_dir()
                    and not relative.startswith("/")
                    and "\\" not in relative
                    and ":" not in relative
                    and ".." not in relative.split("/")
                    and relative not in archive_files,
                    "safe source entries",
                )
                require(
                    relative.startswith(("src/", "configs/"))
                    or relative in ("pyproject.toml", "uv.lock"),
                    "source/config entry before read",
                )
                archive_files[relative] = hashlib.sha256(archive.read(relative)).hexdigest()
    runtime_files = {str(runtime / name): digest for name, digest in archive_files.items()}
    require(summary["runtime_files_sha256"] == runtime_files, "runtime source identity")
    allowed.update(Path(p).resolve() for p in runtime_files)
    for path, digest in runtime_files.items():
        bind(Path(path), digest)
    old_summary = read(historical / "summary.json", HISTORICAL)
    old_training = read(
        historical / "training.json", old_summary["artifact_sha256"]["training.json"]
    )
    require(
        training["parent_state_before"] == old_training["parent_state_before"],
        "accepted original state",
    )
    import torch

    payload = torch.load(
        bind(RUN / "failure-state.pt", FAILURE_STATE), map_location="cpu", weights_only=False
    )
    state = inspect_failure_payload(payload, old_training["parent_state_before"])
    absent = sorted(n for n in ABSENT if not (RUN / n).exists())
    require(set(absent) == ABSENT, "no result artifacts")
    require(
        {p.name for p in RUN.glob("*.pt")} == {"failure-state.pt"}, "only diagnostic checkpoint"
    )
    for path, digest in inputs.items():
        require(sha(Path(path)) == digest, "failure evidence stable")
    return {
        "audit_id": "symmetric-affine8000-failure-audit-v2",
        "complete": True,
        "audit_passed": True,
        "failure_evidence_verified": True,
        "campaign": "symmetric-affine8000-v2",
        "experiment_completed": False,
        "candidate_quality_result": False,
        "hypothesis_supported": None,
        "hypothesis_rejected": None,
        "quality_gate_evaluated": False,
        "source_sha256": sha(Path(__file__)),
        "summary_sha256": SUMMARY,
        "execution_receipt_sha256": inputs[str(receipt_path)],
        "failure_checkpoint_sha256": FAILURE_STATE,
        "diagnosis": diagnosis,
        "diagnostic_checkpoint": state,
        "stages": {
            "parent_admission_returned": True,
            "original93_state_check_passed": True,
            "parent_model_transferred_to_cuda_in_failed_run": True,
            "outer_model_available": True,
            "evaluation_body_entered": False,
            "neural_forward_reached": False,
            "residual_attached": False,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "predictions": False,
            "candidate_checkpoint": False,
            "reload": False,
            "quality_gate": False,
        },
        "absent_stage_artifacts": absent,
        "bound_input_sha256": inputs,
        "runtime_source_files_verified": len(runtime_files),
        "snapshots_verified": len(SNAPSHOTS),
        "cpu_payload_inspection_only": True,
        "cuda_initialized_by_auditor": torch.cuda.is_initialized(),
        "protected_corpus_or_graph_opened_by_auditor": False,
        "limitations": [
            "Argument binding failed before evaluator body; no hypothesis result.",
            "Original run transferred parent to CUDA; this audit inspected state on CPU.",
            "Diagnostic parent state is not a newly trained model or new model lineage.",
            "Historical manifests were not recursively reopened; success gates remain unchanged.",
        ],
    }


if __name__ == "__main__":
    out = HERE / "report.json"
    require(not out.exists(), "immutable failure audit")
    result = audit()
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "report_sha256": sha(out),
                "candidate_quality_result": False,
                "original_tensors_verified": 93,
            }
        )
    )
