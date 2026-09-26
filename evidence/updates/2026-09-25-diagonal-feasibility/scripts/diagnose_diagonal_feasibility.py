"""Authenticated frozen-feature extraction and bounded isolated feasibility workers.

This creates diagnostic coefficients/certificates, never a neural checkpoint.
Preflight is standard-library only; real extraction/solving needs a frozen receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_PLAN = "7b2e575dfd651d10edbf2d8c2a5f4a9f3268d6ad350248fafd35f1e09efe694b"
_MEAN = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
_RECIPE = "05948f589834699de0f1f99538c739eee33242a04ee1e1c402d41295a13f74a4"
_SUMMARY = "0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61"
_AUDIT = "6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc"
_DECISION = "da0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6"
_MEMBERSHIP = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_bytes(value))


def _bind(path, digest, inputs):
    path = Path(path).resolve()
    _require(_sha(path) == digest, f"identity mismatch: {path}")
    inputs[str(path)] = digest
    return path


def _load_mean(root):
    path = root / "scripts/refit_membership_projection.py"
    _require(_sha(path) == _MEAN, "frozen feature helper identity")
    spec = importlib.util.spec_from_file_location("diagonal_mean_helper", path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _refuse(out):
    _require(not out.exists() and not list(out.parent.glob(out.stem + ".*")), "immutable summary")
    for name in (
        "features.npz",
        "extraction.json",
        "train-membership.json",
        "runtime",
        "TYPE.json",
        "COLOR.json",
    ):
        _require(not (out.parent / name).exists(), f"immutable output: {name}")
    _require(
        not any(out.parent.glob("TYPE.*")) and not any(out.parent.glob("COLOR.*")),
        "immutable worker evidence",
    )


def _preflight(root, plan, recipe_path):
    inputs = {}
    _bind(plan, _PLAN, inputs)
    _bind(recipe_path, _RECIPE, inputs)
    mean, mean_path = _load_mean(root)
    _bind(mean_path, _MEAN, inputs)
    helper, helper_path = mean._load_helper(root)
    _bind(helper_path, mean._HELPER, inputs)
    directory = root / "runs/learning/projection-only-refit-v1"
    summary = _read(_bind(directory / "summary.json", _SUMMARY, inputs))
    audit = _read(_bind(directory / "independent-audit.json", _AUDIT, inputs))
    decision = _read(_bind(directory / "decision.json", _DECISION, inputs))
    _require(
        summary["complete"] is True
        and audit["audit_passed"] is True
        and audit["complete"] is True
        and decision["evidence_accepted"] is True
        and audit["summary_sha256"] == decision["summary_sha256"] == _SUMMARY
        and decision["audit_sha256"] == _AUDIT,
        "accepted parent feature provenance",
    )
    _bind(directory / "independent-audit.py", audit["script_sha256"], inputs)
    for path, digest in summary["input_sha256"].items():
        _bind(path, digest, inputs)
    for name in ("summary.source.zip", "summary.configs.zip"):
        path = directory / name
        _bind(path, summary["snapshot_sha256"][str(path.resolve())], inputs)
    membership_path = _bind(directory / "train-membership.json", _MEMBERSHIP, inputs)
    membership = _read(membership_path)
    _require(
        membership["partition"] == "train"
        and membership["query_count"] == len(membership["queries"]) == 1637,
        "train-only membership",
    )
    project = root / "experiments/diagonal-feasibility"
    for path in (project / "solver.py", project / "pyproject.toml", project / "uv.lock"):
        _bind(path, _sha(path), inputs)
    python = (
        project / ".venv/Scripts/python.exe" if os.name == "nt" else project / ".venv/bin/python"
    )
    probe = (
        "import sys,json,importlib.metadata as m; "
        "print(json.dumps({'python':list(sys.version_info[:3]),"
        "'versions':{p:m.version(p) for p in ('numpy','scipy','python-flint')}}))"
    )
    environment = json.loads(subprocess.check_output([str(python), "-I", "-c", probe], text=True))
    recipe = _read(recipe_path)
    _require(
        environment["python"][:2] == [3, 12]
        and all(environment["versions"][p] == recipe[p] for p in environment["versions"]),
        "isolated solver environment",
    )
    config_path = root / "runs/learning/pair-composition-integration-v1/offline-1729.json"
    _require(str(config_path.resolve()) in inputs, "authenticated runtime config")
    parent = root / "runs/national_dex_continuation_control_s1729_v1/checkpoint-final.pt"
    _bind(parent, mean._PARENT, inputs)
    return {
        "inputs": inputs,
        "mean": mean,
        "mean_path": mean_path,
        "helper": helper,
        "helper_path": helper_path,
        "source": directory / "summary.source.zip",
        "configs": directory / "summary.configs.zip",
        "membership_path": membership_path,
        "membership": membership,
        "recipe": recipe,
        "project": project,
        "python": python,
        "solver_environment": environment,
        "config": _read(config_path)["config"],
        "parent": parent,
        "old_summary": summary,
    }


def _outcome(reports):
    statuses = [r["status"] for r in reports]
    if "certified_infeasible" in statuses:
        return "certified_infeasible"
    if len(statuses) == 2 and all(s == "certified_feasible" for s in statuses):
        return "certified_feasible"
    return "inconclusive"


def _worker(command, out, timeout, environment):
    started = time.perf_counter()
    receipt = {
        "command": command,
        "hard_limit_seconds": timeout,
        "timed_out": False,
        "status": "inconclusive",
    }
    process = None
    pending = None
    try:
        with out.with_suffix(".stdout.txt").open("xb") as stream:
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, env=environment
            )
            receipt["pid"] = process.pid
            try:
                remaining = max(0.0, timeout - (time.perf_counter() - started))
                receipt["wait_budget_seconds"] = remaining
                receipt["exit_code"] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
    except BaseException as exc:
        receipt["parent_error"] = repr(exc)
        pending = exc
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                receipt["exit_code"] = process.wait()
            except BaseException as exc:
                receipt["cleanup_error"] = repr(exc)
                if pending is None:
                    pending = exc
            receipt["reaped"] = process.poll() is not None
            receipt["process_running"] = process.poll() is None
        else:
            receipt.update(reaped=True, process_running=False, launch_failed=True)
        receipt["elapsed_seconds"] = time.perf_counter() - started
        if not receipt["timed_out"] and receipt.get("exit_code") == 0:
            try:
                report = _read(out)
                _require(
                    report["status"]
                    in {"certified_feasible", "certified_infeasible", "inconclusive"},
                    "worker status",
                )
                _require(
                    type(report["complete"]) is bool and report["final_identity_check"] is True,
                    "worker completion/identity",
                )
                _require(
                    report["status"] == "inconclusive" or report["complete"] is True,
                    "incomplete certification",
                )
                receipt["status"] = report["status"]
                receipt["report_complete"] = report["complete"]
                receipt["report_sha256"] = _sha(out)
            except Exception as exc:
                receipt["report_error"] = repr(exc)
        _write(out.with_suffix(".process.json"), receipt)
    if pending is not None:
        raise pending
    return receipt


def _extract(root, folder, context):
    mean, helper = context["mean"], context["helper"]
    runtime_root = folder / "runtime"
    helper._modules()
    hashes = helper._extract(
        helper._archive_members((context["source"], context["configs"])), runtime_root
    )
    sys.path.insert(0, str(runtime_root / "src"))
    import numpy as np
    import torch

    from plm.config import RootConfig
    from plm.protocol import load_protocol
    from plm.reproducibility import seed_everything
    from plm.serving.runtime import load_inference_runtime

    seed_everything(1729, deterministic=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    environment = helper._environment(torch)
    config = RootConfig.model_validate(context["config"])
    runtime = load_inference_runtime(
        config,
        context["parent"],
        corpus=(root / config.data.corpus_manifest).parent,
        graph_db=root / config.data.graph_db,
        protocol=load_protocol(runtime_root / "configs/protocol/pokemon_v1.yaml"),
    )
    _require(
        runtime.checkpoint_hash == mean._PARENT and runtime.split.split_hash == mean._SPLIT,
        "loaded original parent",
    )
    model = runtime.model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = mean._state_hashes(model)
    _require(
        mean._training_membership(runtime.split.train) == context["membership"],
        "ordered train labels replay",
    )
    prompts = [
        next(
            row["prompt_ids"]
            for row in context["membership"]["queries"]
            if row["dimension"] == dimension
        )
        for dimension in ("TYPE", "COLOR")
    ]
    with torch.no_grad():
        entities, steering, _, _ = mean._features(
            model, torch.tensor(prompts, dtype=torch.long, device=runtime.device)
        )
        _require(
            bool(torch.isfinite(entities).all()) and bool(torch.isfinite(steering).all()),
            "finite normalized features",
        )
        entity_array, steering_array = entities.cpu().numpy(), steering.cpu().numpy()
    _require(
        list(entity_array.shape) == [1025, 256]
        and list(steering_array.shape) == [2, 256]
        and entity_array.dtype == steering_array.dtype == np.float32,
        "feature shapes/dtypes",
    )
    after = mean._state_hashes(model)
    _require(before == after, "no tensor changes")
    path = folder / "features.npz"
    with path.open("xb") as stream:
        np.savez(stream, entities=entity_array, steering=steering_array)
    receipt = {
        "file_sha256": _sha(path),
        "arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
            for name, array in (("entities", entity_array), ("steering", steering_array))
        },
        "product_token_ids": list(range(1024, 2049)),
        "dimensions": ["TYPE", "COLOR"],
        "dimension_token_ids": [p[2] for p in prompts],
        "parent_checkpoint_sha256": mean._PARENT,
        "parent_embedding_tensor": before["token_embedding.weight"],
        "state_before": before,
        "state_after": after,
        "environment": environment,
        "numerical_settings": helper._settings(torch),
        "runtime_files_sha256": hashes,
        "module_origins": helper._modules(runtime_root),
        "transformer_forwards": 0,
        "optimizer_updates": 0,
        "training_membership_replayed": True,
    }
    helper._runtime_unchanged(runtime_root, hashes)
    _write(folder / "extraction.json", receipt)
    del model, runtime, entities, steering
    torch.cuda.empty_cache()
    return receipt


def _execute(root, out, context, summary):
    started = time.perf_counter()
    helper = context["helper"]
    try:
        summary["extraction"] = _extract(root, out.parent, context)
        generated = {
            str(out.parent / name): _sha(out.parent / name)
            for name in ("features.npz", "extraction.json", "train-membership.json")
        }
        summary["generated_input_sha256"] = generated
        _require(
            generated[str(out.parent / "train-membership.json")] == _MEMBERSHIP,
            "copied training labels",
        )
        reports = []
        for dimension in ("TYPE", "COLOR"):
            helper._unchanged(context["inputs"])
            helper._unchanged(generated)
            command = [
                str(context["python"]),
                "-I",
                str(out.with_suffix(".solver.py")),
                "--features",
                str(out.parent / "features.npz"),
                "--membership",
                str(out.parent / "train-membership.json"),
                "--recipe",
                str(out.with_suffix(".recipe.json")),
                "--dimension",
                dimension,
                "--out",
                str(out.parent / f"{dimension}.json"),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
            receipt = _worker(
                command,
                out.parent / f"{dimension}.json",
                context["recipe"]["worker_seconds"],
                environment,
            )
            receipt["dimension"] = dimension
            reports.append(receipt)
            helper._unchanged(generated)
            summary["workers"] = reports
        summary["outcome"] = _outcome(reports)
        _require(
            _sha(out.parent / "features.npz") == summary["extraction"]["file_sha256"],
            "extracted features changed",
        )
        _require(
            _sha(out.parent / "train-membership.json") == _MEMBERSHIP,
            "training membership copy changed",
        )
        helper._unchanged(context["inputs"])
        helper._unchanged(summary["snapshot_sha256"])
        helper._runtime_unchanged(
            out.parent / "runtime", summary["extraction"]["runtime_files_sha256"]
        )
        summary["final_identity_check"] = True
        summary["complete"] = True
    except BaseException as exc:
        summary["error"] = repr(exc)
        summary["outcome"] = "inconclusive"
        raise
    finally:
        summary["artifact_sha256"] = {
            p.name: _sha(p)
            for p in out.parent.iterdir()
            if p.is_file()
            and (
                p.name.startswith(("TYPE.", "COLOR."))
                or p.name in {"features.npz", "extraction.json", "train-membership.json"}
            )
        }
        summary["wall_seconds"] = time.perf_counter() - started
        _write(out, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/experiments/2026-09-25-diagonal-feasibility-plan.md"),
    )
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/experiments/diagonal_feasibility_v1.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/diagonal-feasibility-v1/summary.json")
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    content = script.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    _require(
        Path.cwd().resolve() == root and "torch" not in sys.modules, "Torch-free root preflight"
    )
    out = args.out.resolve()
    _refuse(out)
    context = _preflight(root, args.plan.resolve(), args.recipe.resolve())
    _bind(script, digest, context["inputs"])
    if args.preflight_only:
        _require("torch" not in sys.modules, "preflight imported Torch")
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "torch_imported": False,
                    "input_count": len(context["inputs"]),
                    "solver_environment": context["solver_environment"],
                }
            )
        )
        return
    _require(
        args.test_receipt is not None and args.test_receipt_sha256,
        "frozen CPU test receipt required",
    )
    receipt_path = _bind(args.test_receipt, args.test_receipt_sha256, context["inputs"])
    receipt = _read(receipt_path)
    _require(
        receipt["passed"] is True
        and receipt["real_data_executed"] is False
        and receipt["gpu_used"] is False
        and receipt["tested_script_sha256"] == digest,
        "test receipt flags/script",
    )
    required = [
        script,
        context["project"] / "solver.py",
        context["project"] / "pyproject.toml",
        context["project"] / "uv.lock",
        args.recipe.resolve(),
        root / "tests/unit/test_diagonal_feasibility.py",
        context["project"] / "test_solver.py",
    ]
    _require(
        set(receipt["tested_files_sha256"]) == {str(p.resolve()) for p in required},
        "tested dependency set",
    )
    for path, pinned in receipt["tested_files_sha256"].items():
        _bind(path, pinned, context["inputs"])
    stdout = _bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"], context["inputs"])
    context["helper"]._unchanged(context["inputs"])
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "script.py": script,
        "solver.py": context["project"] / "solver.py",
        "solver-pyproject.toml": context["project"] / "pyproject.toml",
        "solver-uv.lock": context["project"] / "uv.lock",
        "plan.md": args.plan,
        "recipe.json": args.recipe,
        "mean-helper.py": context["mean_path"],
        "helper.py": context["helper_path"],
        "source.zip": context["source"],
        "configs.zip": context["configs"],
        "test-receipt.json": receipt_path,
        "test-stdout.txt": stdout,
        "test-main.py": root / "tests/unit/test_diagonal_feasibility.py",
        "test-solver.py": context["project"] / "test_solver.py",
    }
    snapshots = {}
    for suffix, path in paths.items():
        target = out.with_suffix("." + suffix)
        with target.open("xb") as stream:
            stream.write(content if suffix == "script.py" else path.read_bytes())
        snapshots[str(target)] = _sha(target)
    _write(out.with_suffix(".inputs.json"), context["inputs"])
    snapshots[str(out.with_suffix(".inputs.json"))] = _sha(out.with_suffix(".inputs.json"))
    with (out.parent / "train-membership.json").open("xb") as stream:
        stream.write(context["membership_path"].read_bytes())
    summary = {
        "campaign_version": "diagonal-feasibility-v1",
        "evaluator": "plm-diagonal-feasibility-v1",
        "complete": False,
        "acceptance": False,
        "outcome": "inconclusive",
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "recipe_sha256": _RECIPE,
        "input_sha256": context["inputs"],
        "snapshot_sha256": snapshots,
        "solver_environment": context["solver_environment"],
        "validation_predictions": 0,
        "protected_test_predictions": 0,
        "neural_checkpoints_created": 0,
        "scope": (
            "exact-real diagonal family over fixed FP32 training features; "
            "not rounding-dependent FP32 head behavior"
        ),
        "limitations": [
            "Numerical LP status alone is not a proof.",
            "Training feasibility does not establish validation quality.",
            "The independent audit does not regenerate CUDA normalization.",
            "Local weight-derived feature bytes are not durable remote archives.",
        ],
    }
    _execute(root, out, context, summary)


if __name__ == "__main__":
    main()
