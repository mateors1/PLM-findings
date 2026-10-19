"""Synthetic continuation admission and unchanged scientific interface checks."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wrapper():
    return load("scripts/refit_bilinear_budget32000_continuation_v2.py")


@pytest.fixture
def base():
    return load("scripts/refit_bilinear_budget32000.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def helper(wrapper):
    def bind(path, digest, inputs):
        assert wrapper.sha(path) == digest
        inputs[str(path)] = digest
        return path

    def unchanged(inputs):
        assert all(wrapper.sha(path) == digest for path, digest in inputs.items())

    return SimpleNamespace(
        _sha=wrapper.sha,
        _bind=bind,
        _read=lambda p: json.loads(Path(p).read_text()),
        _unchanged=unchanged,
        _modules=lambda: None,
    )


def ready_fixture(wrapper, tmp_path, monkeypatch):
    files = {}
    for name in wrapper.FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = wrapper.sha(path)
    monkeypatch.setattr(wrapper, "AMENDMENT", files[wrapper.AMENDMENT_PATH])
    monkeypatch.setattr(wrapper, "BASE_READY_SHA", files[wrapper.BASE_READY])
    for role, directory in wrapper.DIRECTORIES.items():
        receipt = {
            "passed": True,
            "gpu_used": False,
            "experiment_training_executed": False,
            "exit_code": 0,
            "terminal_completion_observed": True,
            "tests_passed": 1,
            "tests_skipped": 0,
            "evidence_revision": wrapper.REVISION,
            "amendment_sha256": wrapper.AMENDMENT,
            "tested_script_sha256": files[wrapper.SOURCES[role]],
            "test_dependencies": {n: files[n] for n in wrapper.DEPENDENCIES[role]},
            "stdout": {
                "path": directory + "/stdout.txt",
                "sha256": files[directory + "/stdout.txt"],
            },
        }
        path = tmp_path / directory / "test-receipt.json"
        write(path, receipt)
        files[directory + "/test-receipt.json"] = wrapper.sha(path)
    value = {
        "schema_version": 1,
        "campaign_version": wrapper.CAMPAIGN,
        "evidence_revision": wrapper.REVISION,
        "amendment_sha256": wrapper.AMENDMENT,
        "base_readiness": {"path": wrapper.BASE_READY, "sha256": wrapper.BASE_READY_SHA},
        "ready": True,
        "files": files,
    }
    path = tmp_path / wrapper.READINESS
    write(path, value)
    return path, value


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "extra",
        "missing",
        "false",
        "boolschema",
        "wrong_revision",
        "exit_bool",
        "unobserved",
        "wrong_source",
        "extra_dependency",
    ],
)
def test_v2_readiness_exact_before_open(wrapper, base, tmp_path, monkeypatch, mutation):
    path, value = ready_fixture(wrapper, tmp_path, monkeypatch)
    calls = []

    def old_readiness(root, oldpath, pin, helper, inputs):
        calls.append("base")
        assert oldpath == root / wrapper.BASE_READY and pin == wrapper.BASE_READY_SHA
        return {"files": {}}

    synthetic_base = SimpleNamespace(_readiness=old_readiness, _resolve=base._resolve)
    if mutation == "extra":
        value["files"]["data/records.jsonl"] = "0" * 64
    if mutation == "missing":
        value["files"].pop(next(iter(value["files"])))
    if mutation == "false":
        value["ready"] = False
    if mutation == "boolschema":
        value["schema_version"] = True
    if mutation == "wrong_revision":
        value["evidence_revision"] = "old"
    if mutation in {"exit_bool", "unobserved", "wrong_source", "extra_dependency"}:
        relative = wrapper.DIRECTORIES["runner"] + "/test-receipt.json"
        receipt = json.loads((tmp_path / relative).read_text())
        if mutation == "exit_bool":
            receipt["exit_code"] = False
        if mutation == "unobserved":
            receipt["terminal_completion_observed"] = False
        if mutation == "wrong_source":
            receipt["tested_script_sha256"] = "wrong"
        if mutation == "extra_dependency":
            receipt["test_dependencies"]["data/records.jsonl"] = "0" * 64
        write(tmp_path / relative, receipt)
        value["files"][relative] = wrapper.sha(tmp_path / relative)
    write(path, value)
    h = helper(wrapper)
    original_bind = h._bind
    opened = []

    def guarded_bind(target, digest, inputs):
        opened.append(target)
        assert "data" not in target.parts
        return original_bind(target, digest, inputs)

    h._bind = guarded_bind
    if mutation:
        with pytest.raises(ValueError):
            wrapper.readiness(tmp_path, path, wrapper.sha(path), synthetic_base, h, {})
        if mutation in {"extra", "missing", "false", "boolschema", "wrong_revision"}:
            assert opened == [path]
    else:
        old, result = wrapper.readiness(tmp_path, path, wrapper.sha(path), synthetic_base, h, {})
        assert len(result["files"]) == 14 and old == {"files": {}}
    assert calls == ["base"]


@pytest.mark.parametrize(
    "mutation", [None, "100ns", "missing", "extra", "seed", "boolseed", "hash", "naive"]
)
def test_exact_equivalent_predecessor_instant(wrapper, base, mutation):
    expected = {
        "seed": 1729,
        "summary_sha256": "hash",
        "completed_utc": "2026-09-26T14:56:49.5359481+00:00",
    }
    actual = {**expected, "completed_utc": "2026-09-26T09:56:49.5359481-05:00"}
    if mutation == "100ns":
        actual["completed_utc"] = "2026-09-26T09:56:49.5359482-05:00"
    if mutation == "missing":
        actual.pop("seed")
    if mutation == "extra":
        actual["unexpected"] = True
    if mutation in {"seed", "boolseed"}:
        actual["seed"] = 1730 if mutation == "seed" else True
    if mutation == "hash":
        actual["summary_sha256"] = "wrong"
    if mutation == "naive":
        actual["completed_utc"] = "2026-09-26T14:56:49"
    if mutation:
        with pytest.raises(ValueError):
            wrapper.same_predecessor(actual, expected, base)
    else:
        wrapper.same_predecessor(actual, expected, base)


def predecessor_fixture(wrapper, base, tmp_path, monkeypatch):
    folder = tmp_path / f"runs/learning/{wrapper.CAMPAIGN}/seed-1730"
    files = {
        wrapper.SOURCES["auditor"]: "auditor",
        wrapper.DIRECTORIES["auditor"] + "/test-receipt.json": "tests",
    }
    prior_path = tmp_path / f"runs/learning/{wrapper.CAMPAIGN}/seed-1729/execution-receipt.json"
    write(
        prior_path,
        {"summary_sha256": "prior", "completed_utc": "2026-09-26T14:56:49.5359481+00:00"},
    )
    for name in ("primary-stdout.txt", "audit-stdout-v2.txt"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic")
    summary = {
        "seed": 1730,
        "complete": True,
        "final_identity_check": True,
        "gate": {"checks": {"execution_invariants": True}, "primary_checks_passed": False},
        "script_sha256": wrapper.BASE_SHA,
        "shared_readiness": {"path": wrapper.BASE_READY, "sha256": wrapper.BASE_READY_SHA},
        "child": {"synthetic": True},
    }
    write(folder / "summary.json", summary)
    execution = {
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "seed": 1730,
        "summary_sha256": wrapper.sha(folder / "summary.json"),
        "started_utc": "2026-09-26T15:00:00Z",
        "completed_utc": "2026-09-26T15:01:00Z",
        "prior_execution_receipt_sha256": wrapper.sha(prior_path),
        "prior_seed": {
            "seed": 1729,
            "summary_sha256": "prior",
            "completed_utc": "2026-09-26T09:56:49.5359481-05:00",
        },
        "stdout_sha256": wrapper.sha(folder / "primary-stdout.txt"),
    }
    write(folder / "execution-receipt.json", execution)
    write(folder / "independent-audit.json", {"audit_passed": False})
    write(folder / "audit-execution-receipt.json", {"exit_code": 1})
    pins = {n: wrapper.sha(folder / n) for n in wrapper.PREVIOUS_PINS}
    monkeypatch.setattr(wrapper, "PREVIOUS_PINS", pins)
    checked = {
        "complete": True,
        "audit_passed": True,
        "seed": 1730,
        "summary_sha256": pins["summary.json"],
        "execution_receipt_sha256": pins["execution-receipt.json"],
        "script_sha256": "auditor",
        "test_receipt_sha256": "tests",
        "audit_revision": wrapper.REVISION,
        "amendment_sha256": wrapper.AMENDMENT,
        "plan_sha256": base._PLAN,
        "base_readiness": {"path": wrapper.BASE_READY, "sha256": wrapper.BASE_READY_SHA},
        "shared_readiness": {"path": wrapper.READINESS, "sha256": "ready"},
        "child": summary["child"],
        "gate": summary["gate"],
    }
    write(folder / "independent-audit-v2.json", checked)
    terminal = {
        **{
            k: execution[k]
            for k in (
                "exit_code",
                "terminal_completion_observed_by_primary",
                "seed",
                "summary_sha256",
            )
        },
        "auditor_sha256": "auditor",
        "test_receipt_sha256": "tests",
        "cuda_visible_devices": "-1",
        "audit_sha256": wrapper.sha(folder / "independent-audit-v2.json"),
        "execution_receipt_sha256": pins["execution-receipt.json"],
        "started_utc": "2026-09-26T15:02:00Z",
        "completed_utc": "2026-09-26T15:03:00Z",
        "stdout_sha256": wrapper.sha(folder / "audit-stdout-v2.txt"),
    }
    write(folder / "audit-execution-receipt-v2.json", terminal)
    return folder, {"files": files}, checked, terminal


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "failed_audit",
        "stale_ready",
        "wrong_source",
        "wrong_test",
        "terminal_failed",
        "terminal_bool",
        "unobserved",
        "overlap",
        "reverse",
        "future",
        "cuda",
    ],
)
def test_repaired_predecessor_admission(wrapper, base, tmp_path, monkeypatch, mutation):
    folder, ready, checked, terminal = predecessor_fixture(wrapper, base, tmp_path, monkeypatch)
    if mutation == "failed_audit":
        checked["audit_passed"] = False
    if mutation == "stale_ready":
        checked["shared_readiness"]["sha256"] = "old"
    if mutation == "wrong_source":
        checked["script_sha256"] = "old"
    if mutation == "wrong_test":
        checked["test_receipt_sha256"] = "old"
    if mutation == "terminal_failed":
        terminal["exit_code"] = 1
    if mutation == "terminal_bool":
        terminal["exit_code"] = False
    if mutation == "unobserved":
        terminal["terminal_completion_observed_by_primary"] = False
    if mutation == "overlap":
        terminal["started_utc"] = "2026-09-26T15:00:00Z"
    if mutation == "reverse":
        terminal["completed_utc"] = "2026-09-26T15:01:00Z"
    if mutation == "future":
        terminal["completed_utc"] = "2026-09-26T15:05:00Z"
    if mutation == "cuda":
        terminal["cuda_visible_devices"] = "0"
    write(folder / "independent-audit-v2.json", checked)
    terminal["audit_sha256"] = wrapper.sha(folder / "independent-audit-v2.json")
    write(folder / "audit-execution-receipt-v2.json", terminal)
    args = (tmp_path, ready, "ready", base, helper(wrapper), {}, "2026-09-26T15:04:00Z")
    if mutation:
        with pytest.raises(ValueError):
            wrapper.predecessor(*args)
    else:
        result = wrapper.predecessor(*args)
        assert result["seed"] == 1730 and result["audit_execution_receipt_sha256"] == wrapper.sha(
            folder / "audit-execution-receipt-v2.json"
        )


@pytest.mark.parametrize("mode", ["success", "preflight", "fit_failure", "post_drift", "immutable"])
def test_real_cli_snapshot_and_scientific_stage_interface(
    wrapper, base, tmp_path, monkeypatch, mode
):
    path, ready = ready_fixture(wrapper, tmp_path, monkeypatch)
    h = helper(wrapper)
    mean = load("scripts/refit_membership_projection.py")
    mean_double = SimpleNamespace(
        _load_helper=lambda root: (h, tmp_path / "helper.py"),
        _bytes=mean._bytes,
        _refuse=mean._refuse,
        _HELPER="helper",
    )
    bilinear = SimpleNamespace(_load_mean=lambda root: (mean_double, tmp_path / "mean.py"))
    context = {
        "inputs": {},
        "source": tmp_path / "source.zip",
        "configs": tmp_path / "configs.zip",
        **{f"historical8000_{n}_sha256": n for n in ("summary", "audit", "decision", "child")},
    }
    extra = [
        wrapper.BASE_SOURCE,
        "helper.py",
        "mean.py",
        "bilinear.py",
        "source.zip",
        "configs.zip",
        "scripts/refit_symmetric_affine_v3.py",
        "docs/experiments/2026-09-26-bilinear-budget32000-plan.md",
        "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md",
        "configs/experiments/bilinear_budget32000_v1.json",
    ]
    for name in extra:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name)
    out = tmp_path / f"runs/learning/{wrapper.CAMPAIGN}/seed-1731/summary.json"
    events = []
    synthetic_base = SimpleNamespace(**{k: v for k, v in vars(base).items() if k.startswith("_")})
    synthetic_base._load_bilinear = lambda root: (bilinear, tmp_path / "bilinear.py")
    synthetic_base._adapter = lambda root: object()

    def preflight(root, seed, plan, recipe_path, bilinear_arg, mean_arg, helper_arg, adapter):
        inspect.signature(base._preflight).bind(
            root, seed, plan, recipe_path, bilinear_arg, mean_arg, helper_arg, adapter
        )
        assert seed == 1731
        events.append("preflight")
        return context

    synthetic_base._preflight = preflight

    def stage(root, destination, passed, bilinear_arg, mean_arg, helper_arg, summary):
        inspect.signature(base._execute).bind(
            root, destination, passed, bilinear_arg, mean_arg, helper_arg, summary
        )
        assert passed is context and bilinear_arg is bilinear and helper_arg is h
        assert summary["script_sha256"] == wrapper.BASE_SHA
        assert summary["execution_wrapper_sha256"] == ready["files"][wrapper.SOURCES["runner"]]
        assert len(summary["continuation_snapshots"]) == 15
        assert (
            destination.with_suffix(".script.py").read_bytes()
            == (tmp_path / wrapper.BASE_SOURCE).read_bytes()
        )
        assert (
            destination.with_suffix(".readiness.json").read_bytes()
            == (tmp_path / wrapper.BASE_READY).read_bytes()
        )
        assert all(
            (root / p).is_file() and wrapper.sha(root / p) == digest
            for p, digest in summary["continuation_snapshots"].items()
        )
        events.append("execute")
        if mode == "fit_failure":
            raise RuntimeError("synthetic fit failure")
        summary.update(complete=True, final_identity_check=True)
        write(destination, summary)
        if mode == "post_drift":
            (root / wrapper.SOURCES["runner"]).write_text("changed")

    synthetic_base._execute = stage
    monkeypatch.setattr(wrapper, "load_base", lambda root: synthetic_base)
    monkeypatch.setattr(
        wrapper, "readiness", lambda root, path, pin, base, helper, inputs: ({"files": {}}, ready)
    )
    monkeypatch.setattr(
        wrapper, "predecessor", lambda root, ready, pin, base, helper, inputs, now: {"seed": 1730}
    )
    monkeypatch.setattr(wrapper, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(wrapper, "__file__", str(tmp_path / wrapper.SOURCES["runner"]))
    monkeypatch.chdir(tmp_path)
    args = [
        str(wrapper.__file__),
        "--seed",
        "1731",
        "--readiness",
        str(path),
        "--readiness-sha256",
        wrapper.sha(path),
    ]
    if mode == "preflight":
        args.append("--preflight-only")
    monkeypatch.setattr(sys, "argv", args)
    if mode == "immutable":
        (out.parent / "continuation").mkdir(parents=True)
    if mode in {"fit_failure", "post_drift", "immutable"}:
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            wrapper.main()
        assert not (out.parent / "continuation-complete.json").exists()
    else:
        wrapper.main()
        if mode == "preflight":
            assert events == ["preflight"] and not out.parent.exists()
        else:
            completion = h._read(out.parent / "continuation-complete.json")
            assert events == ["preflight", "execute"]
            assert completion["summary_sha256"] == wrapper.sha(out)
            assert completion["root_observed_terminal_required"] is True


def test_actual_unchanged_scientific_stage_with_bounded_synthetic_fit(base, tmp_path, monkeypatch):
    original = load("tests/unit/test_bilinear_budget32000.py")
    mean = load("scripts/refit_membership_projection.py")
    original.test_full_execute_lifecycle_with_bounded_synthetic_fit(
        base, mean, tmp_path, monkeypatch, False, 1731
    )


def test_wrapper_cli_calls_real_scientific_stage_end_to_end(wrapper, base, tmp_path, monkeypatch):
    """Route the frozen synthetic lifecycle fixture through the actual continuation CLI.

    Runtime is synthetic CPU and fit simulates 32000 after one real optimizer step.
    The wrapper, evaluator, save/reload and scientific stage bodies execute together.
    """
    original = load("tests/unit/test_bilinear_budget32000.py")
    mean = load("scripts/refit_membership_projection.py")
    real_stage = base._execute
    ready_path, ready = ready_fixture(wrapper, tmp_path, monkeypatch)
    run_dir = tmp_path / f"runs/learning/{wrapper.CAMPAIGN}/seed-1731"
    run_dir.mkdir(parents=True)

    def routed(root, out, context, bilinear, mean, helper_arg, initial_summary):
        assert out == run_dir / "summary.json"
        inspect.signature(real_stage).bind(
            root, out, context, bilinear, mean, helper_arg, initial_summary
        )
        for name in (
            wrapper.BASE_SOURCE,
            "helper.py",
            "mean.py",
            "bilinear.py",
            "source.zip",
            "configs.zip",
            "scripts/refit_symmetric_affine_v3.py",
            "docs/experiments/2026-09-26-bilinear-budget32000-plan.md",
            "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md",
            "configs/experiments/bilinear_budget32000_v1.json",
        ):
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        context.update(
            source=tmp_path / "source.zip",
            configs=tmp_path / "configs.zip",
            historical8000_audit_sha256="synthetic",
            historical8000_decision_sha256="synthetic",
        )
        monkeypatch.setattr(helper_arg, "_unchanged", staticmethod(helper(wrapper)._unchanged))
        # This is an isolated fixture module, never the production frozen import.
        composed_base = SimpleNamespace(**vars(base))
        composed_base._execute = real_stage
        composed_base._load_bilinear = lambda root: (bilinear, tmp_path / "bilinear.py")
        composed_base._adapter = lambda root: object()

        def preflight(root, seed, plan, recipe, bilinear_arg, mean_arg, helper_arg, adapter):
            assert seed == 1731
            return context

        composed_base._preflight = preflight
        monkeypatch.setattr(bilinear, "_load_mean", lambda root: (mean, tmp_path / "mean.py"))
        monkeypatch.setattr(mean, "_load_helper", lambda root: (helper_arg, tmp_path / "helper.py"))
        monkeypatch.setattr(wrapper, "load_base", lambda root: composed_base)
        monkeypatch.setattr(
            wrapper,
            "readiness",
            lambda root, path, pin, base, helper, inputs: ({"files": {}}, ready),
        )
        monkeypatch.setattr(
            wrapper,
            "predecessor",
            lambda root, ready, pin, base, helper, inputs, now: {"seed": 1730},
        )
        monkeypatch.setattr(wrapper, "sys", SimpleNamespace(modules={}))
        monkeypatch.setattr(wrapper, "__file__", str(tmp_path / wrapper.SOURCES["runner"]))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "wrapper",
                "--seed",
                "1731",
                "--readiness",
                str(ready_path),
                "--readiness-sha256",
                wrapper.sha(ready_path),
            ],
        )
        wrapper.main()
        completion = json.loads((run_dir / "continuation-complete.json").read_text())
        assert completion["summary_sha256"] == wrapper.sha(out)

    monkeypatch.setattr(base, "_execute", routed)
    original.test_full_execute_lifecycle_with_bounded_synthetic_fit(
        base, mean, run_dir, monkeypatch, False, 1731
    )


def test_no_frozen_module_monkeypatch_or_old_main_in_production(wrapper):
    import ast

    tree = ast.parse(Path(wrapper.__file__).read_text())
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in {"setattr", "exec"}
        for n in ast.walk(tree)
    )
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "base"
    ]
    assert sum(n.func.attr == "_execute" for n in calls) == 1
    assert not any(n.func.attr in {"main", "_previous_seed"} for n in calls)


@pytest.mark.parametrize("seed", [1729, 1730])
def test_cli_rejects_retraining_existing_seed(wrapper, monkeypatch, seed):
    monkeypatch.setattr(
        sys, "argv", ["wrapper", "--seed", str(seed), "--readiness", "x", "--readiness-sha256", "x"]
    )
    with pytest.raises(SystemExit):
        wrapper.main()
