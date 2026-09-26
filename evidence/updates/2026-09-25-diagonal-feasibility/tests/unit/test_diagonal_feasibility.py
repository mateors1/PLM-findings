"""No real data, normalization, model or LP execution in orchestration tests."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/diagnose_diagonal_feasibility.py"
spec = importlib.util.spec_from_file_location("diagonal_main_test", _SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_status_reduction_requires_both_feasible():
    assert runner._outcome([]) == "inconclusive"
    assert runner._outcome([{"status": "certified_feasible"}]) == "inconclusive"
    assert runner._outcome([{"status": "certified_feasible"}] * 2) == "certified_feasible"
    assert (
        runner._outcome([{"status": "inconclusive"}, {"status": "certified_infeasible"}])
        == "certified_infeasible"
    )
    assert (
        runner._outcome([{"status": "certified_feasible"}, {"status": "inconclusive"}])
        == "inconclusive"
    )


@pytest.mark.parametrize("changed", ["plan", "recipe", "helper"])
def test_authentication_drift_precedes_helper_execution_or_data_access(
    tmp_path, monkeypatch, changed
):
    plan = tmp_path / "plan.md"
    recipe = tmp_path / "recipe.json"
    helper = tmp_path / "scripts/refit_membership_projection.py"
    helper.parent.mkdir()
    plan.write_text("synthetic declared plan", encoding="utf-8")
    recipe.write_text("{}", encoding="utf-8")
    helper.write_text("raise AssertionError('untrusted helper executed')", encoding="utf-8")
    monkeypatch.setattr(runner, "_PLAN", hashlib.sha256(plan.read_bytes()).hexdigest())
    monkeypatch.setattr(runner, "_RECIPE", hashlib.sha256(recipe.read_bytes()).hexdigest())
    target = {"plan": plan, "recipe": recipe, "helper": helper}[changed]
    target.write_text("changed unauthenticated bytes", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("authentication must precede imports, data reads and solver probes")

    monkeypatch.setattr(runner.importlib.util, "spec_from_file_location", forbidden)
    monkeypatch.setattr(runner.subprocess, "check_output", forbidden)
    monkeypatch.setattr(runner, "_read", forbidden)
    with pytest.raises(ValueError, match="identity"):
        runner._preflight(tmp_path, plan, recipe)


def test_refuse_partial_but_allow_independent_helpers(tmp_path):
    out = tmp_path / "summary.json"
    (tmp_path / "independent-audit.py").write_text("# prepared independently", encoding="utf-8")
    runner._refuse(out)
    (tmp_path / "TYPE.trace.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        runner._refuse(out)


@pytest.mark.parametrize(
    "existing",
    [
        "summary.json",
        "summary.script.py",
        "features.npz",
        "extraction.json",
        "train-membership.json",
        "runtime",
        "TYPE.json",
        "COLOR.process.json",
    ],
)
def test_each_owned_output_collision_fails_without_changing_evidence(tmp_path, existing):
    path = tmp_path / existing
    if existing == "runtime":
        path.mkdir()
    else:
        path.write_bytes(b"existing immutable evidence")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="immutable"):
        runner._refuse(tmp_path / "summary.json")
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def _report(status="certified_feasible", complete=True, final_identity_check=True):
    return {"status": status, "complete": complete, "final_identity_check": final_identity_check}


def _child_command(out, content=None, exit_code=0):
    code = (
        "import pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8') "
        "if sys.argv[2] != '<missing>' else None; "
        "print('synthetic worker stdout', flush=True); "
        "sys.exit(int(sys.argv[3]))"
    )
    return [
        sys.executable,
        "-c",
        code,
        str(out),
        "<missing>" if content is None else content,
        str(exit_code),
    ]


@pytest.mark.parametrize("status", ["certified_feasible", "certified_infeasible", "inconclusive"])
def test_successful_known_report_is_bound_to_reaped_child(tmp_path, status):
    out = tmp_path / "TYPE.json"
    report = _report(status)
    result = runner._worker(_child_command(out, json.dumps(report)), out, 5, os.environ.copy())
    assert result["exit_code"] == 0 and not result["timed_out"]
    assert result["reaped"] and not result["process_running"]
    assert result["status"] == status and result["report_complete"] is True
    assert result["report_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert json.loads(out.with_suffix(".process.json").read_text(encoding="utf-8")) == result
    assert "synthetic worker stdout" in out.with_suffix(".stdout.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "content",
    [
        None,
        "bad json",
        "[]",
        json.dumps(_report("invented")),
        json.dumps(_report(complete=1)),
        json.dumps(_report(final_identity_check=False)),
        json.dumps(_report(complete=False)),
    ],
)
def test_invalid_worker_report_preserves_receipt(tmp_path, content):
    out = tmp_path / "TYPE.json"
    result = runner._worker(_child_command(out, content), out, 5, os.environ.copy())
    assert result["status"] == "inconclusive"
    assert result["reaped"] and not result["process_running"]
    assert result["exit_code"] == 0
    assert "report_error" in json.loads(
        out.with_suffix(".process.json").read_text(encoding="utf-8")
    )
    if content is not None:
        assert out.read_text(encoding="utf-8") == content


def test_nonzero_exit_cannot_certify_even_with_valid_report(tmp_path):
    out = tmp_path / "TYPE.json"
    content = json.dumps(_report())
    result = runner._worker(_child_command(out, content, 7), out, 5, os.environ.copy())
    assert result["exit_code"] == 7 and result["status"] == "inconclusive"
    assert result["reaped"] and not result["process_running"]
    assert "report_sha256" not in result
    assert out.read_text(encoding="utf-8") == content
    assert json.loads(out.with_suffix(".process.json").read_text(encoding="utf-8")) == result


def test_timeout_kills_and_reaps_actual_child(tmp_path, monkeypatch):
    original_popen = runner.subprocess.Popen
    children = []

    def owned_popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", owned_popen)
    out = tmp_path / "TYPE.json"
    try:
        result = runner._worker(
            [sys.executable, "-c", "import time; time.sleep(20)"], out, 0.1, os.environ.copy()
        )
        assert result["timed_out"] and result["reaped"] and not result["process_running"]
        assert result["exit_code"] != 0 and result["status"] == "inconclusive"
        assert len(children) == 1 and children[0].poll() is not None
        assert result["pid"] == children[0].pid
        assert json.loads(out.with_suffix(".process.json").read_text(encoding="utf-8")) == result
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_failed_launch_preserves_receipt_and_original_error(tmp_path):
    out = tmp_path / "TYPE.json"
    with pytest.raises(FileNotFoundError):
        runner._worker([str(tmp_path / "nonexistent-child")], out, 5, os.environ.copy())
    receipt = json.loads(out.with_suffix(".process.json").read_text(encoding="utf-8"))
    assert receipt["launch_failed"] and receipt["reaped"] and not receipt["process_running"]
    assert receipt["status"] == "inconclusive"
    assert "FileNotFoundError" in receipt["parent_error"]


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, RuntimeError])
def test_parent_exception_kills_reaps_actual_owned_child(tmp_path, monkeypatch, exception_type):
    original_popen = runner.subprocess.Popen
    children = []
    failure = exception_type("synthetic parent interruption")

    def interrupted_popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        original_wait = child.wait
        calls = 0

        def interrupted_wait(timeout=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise failure
            return original_wait(timeout=timeout)

        monkeypatch.setattr(child, "wait", interrupted_wait)
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", interrupted_popen)
    out = tmp_path / "TYPE.json"
    try:
        with pytest.raises(exception_type) as captured:
            runner._worker(
                [sys.executable, "-c", "import time; time.sleep(20)"], out, 5, os.environ.copy()
            )
        assert captured.value is failure
        assert len(children) == 1 and children[0].poll() is not None
    finally:
        # Keep the test safe even when a cleanup regression makes an assertion fail.
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
    receipt = json.loads(out.with_suffix(".process.json").read_text(encoding="utf-8"))
    assert receipt["reaped"] and "parent_error" in receipt
    assert not receipt["process_running"] and receipt["status"] == "inconclusive"
    assert exception_type.__name__ in receipt["parent_error"]


def test_extraction_exception_retains_summary(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic extraction failure")

    monkeypatch.setattr(runner, "_extract", fail)
    out = tmp_path / "summary.json"
    summary = {"complete": False, "acceptance": False}
    with pytest.raises(RuntimeError, match="synthetic"):
        runner._execute(tmp_path, out, {"helper": object()}, summary)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["outcome"] == "inconclusive" and not saved["complete"]
    assert "synthetic" in saved["error"]
