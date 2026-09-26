"""Synthetic aggregate, audit and exact-instant contracts; no campaign reads."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aggregate():
    return load("scripts/aggregate_bilinear_budget8000_replication_v2.py", "aggregate8000_v2_test")


@pytest.fixture(scope="module")
def runner():
    return load("scripts/refit_bilinear_budget8000_replication.py", "seed8000_test")


@pytest.fixture(scope="module")
def oldtests():
    return load("tests/unit/test_bilinear_replication.py", "old_replication_fixtures")


@pytest.fixture(scope="module")
def mean():
    return load("scripts/refit_membership_projection.py", "aggregate_mean")


def test_equivalent_offsets_preserve_exact_instant(aggregate):
    assert aggregate._same_instant(
        "2026-09-25T10:00:00.5251055Z", "2026-09-25T05:00:00.5251055-05:00"
    )
    assert not aggregate._same_instant(
        "2026-09-25T10:00:00.5251055Z", "2026-09-25T10:00:00.5251056Z"
    )
    assert aggregate._same_instant("2026-09-25T10:00:00Z", "2026-09-25T10:00:00.0000000+00:00")


@pytest.mark.parametrize(
    "timestamp", ["2026-09-25T10:00:00", "2026-09-25", "bad", "2026-09-25T25:00:00Z"]
)
def test_reject_naive_or_invalid_instants(aggregate, timestamp):
    with pytest.raises(ValueError):
        aggregate._instant(timestamp)


@pytest.mark.parametrize("seed", [1730, 1731])
def test_terminal_receipt_binds_order_and_completion(aggregate, oldtests, seed):
    receipt = oldtests._terminal_receipt(seed)
    previous = None if seed == 1730 else "synthetic-previous"
    begin, finish = aggregate._terminal(
        receipt, seed, "synthetic-summary", "synthetic-stdout", previous
    )
    assert finish > begin
    receipt["terminal_completion_observed_by_primary"] = False
    with pytest.raises(ValueError):
        aggregate._terminal(receipt, seed, "synthetic-summary", "synthetic-stdout", previous)


@pytest.mark.parametrize(
    "changed", ["chain", "naive", "reverse", "exit", "bool_exit", "summary", "seed", "stdout"]
)
def test_terminal_rejects_invalid_receipts(aggregate, oldtests, changed):
    receipt = oldtests._terminal_receipt(1731)
    if changed == "chain":
        receipt["previous_terminal_receipt_sha256"] = "other"
    elif changed == "naive":
        receipt["started_at_utc"] = "2026-09-25T01:00:00"
    elif changed == "reverse":
        receipt["started_at_utc"] = "2026-09-25T01:01:00.0000001Z"
    elif changed in {"exit", "bool_exit"}:
        receipt["exit_code"] = 1 if changed == "exit" else False
    elif changed == "seed":
        receipt["seed"] = 1730
    else:
        receipt[changed + "_sha256"] = "other"
    with pytest.raises(ValueError):
        aggregate._terminal(
            receipt, 1731, "synthetic-summary", "synthetic-stdout", "synthetic-previous"
        )


def audit_fixture():
    audit = {
        "complete": True,
        "audit_passed": True,
        "seed": 1730,
        "summary_sha256": "summary",
        "script_sha256": "auditor",
        "test_receipt_sha256": "test",
    }
    receipt = {
        "seed": 1730,
        "summary_sha256": "summary",
        "audit_sha256": "audit",
        "auditor_sha256": "auditor",
        "test_receipt_sha256": "test",
        "stdout_sha256": "stdout",
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
    }
    return audit, receipt


def test_independent_audit_receipt_complete_contract(aggregate):
    aggregate._audit_contract(*audit_fixture(), 1730, "summary", "audit", "stdout")


@pytest.mark.parametrize("field", ["complete", "audit_passed", "seed", "summary_sha256"])
def test_reject_missing_or_failed_audit(aggregate, field):
    audit, receipt = audit_fixture()
    audit[field] = False if field in {"complete", "audit_passed"} else "wrong"
    with pytest.raises(ValueError):
        aggregate._audit_contract(audit, receipt, 1730, "summary", "audit", "stdout")


@pytest.mark.parametrize(
    "field",
    [
        "seed",
        "summary_sha256",
        "audit_sha256",
        "auditor_sha256",
        "test_receipt_sha256",
        "stdout_sha256",
        "exit_code",
        "terminal_completion_observed_by_primary",
    ],
)
def test_reject_unbound_audit_execution(aggregate, field):
    audit, receipt = audit_fixture()
    receipt[field] = False if field == "terminal_completion_observed_by_primary" else "wrong"
    with pytest.raises(ValueError):
        aggregate._audit_contract(audit, receipt, 1730, "summary", "audit", "stdout")


def test_missing_fresh_audit_file_fails_closed(aggregate, tmp_path):
    helper = SimpleNamespace(
        _sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
        _read=lambda p: json.loads(p.read_text()),
        _bind=lambda p, h, i: p,
    )
    with pytest.raises(FileNotFoundError):
        aggregate._fresh_audit(tmp_path, 1730, "summary", helper, {}, "auditor", "test")


def test_gate_requires_both_quality_results_and_both_audits(aggregate, runner, oldtests):
    reports = oldtests._campaign_reports(runner)
    audits = {s: {"complete": True, "audit_passed": True} for s in (1730, 1731)}
    assert aggregate._campaign_gate(reports, audits, True, runner)["primary_checks_passed"]
    reports[1730]["gate"]["primary_checks_passed"] = False
    gate = aggregate._campaign_gate(reports, audits, True, runner)
    assert not gate["primary_checks_passed"] and gate["checks"]["fresh_seed1730_audited"]
    reports[1730]["gate"]["primary_checks_passed"] = True
    audits[1731]["audit_passed"] = False
    assert not aggregate._campaign_gate(reports, audits, True, runner)["primary_checks_passed"]
    audits.pop(1731)
    with pytest.raises(ValueError):
        aggregate._campaign_gate(reports, audits, True, runner)


def test_pooling_remains444_and666_over_same222(runner, oldtests, mean):
    historical = oldtests._observations(mean, 6)
    fresh = {1730: oldtests._observations(mean, 7), 1731: oldtests._observations(mean, 8)}
    pooled = runner._aggregate_reports(fresh, historical, mean)
    assert pooled["fresh_two"]["aggregate"]["query_count"] == 444
    assert pooled["fresh_two"]["aggregate"]["exact_count"] == 429
    assert pooled["all_three"]["aggregate"]["query_count"] == 666
    assert pooled["all_three"]["aggregate"]["exact_count"] == 645
    fresh[1730]["responses"][0]["expected_set_ids"] = [2047]
    with pytest.raises(ValueError):
        runner._aggregate_reports(fresh, historical, mean)


def test_missing_seed_preserves_partial_failure(aggregate, runner, mean, tmp_path):
    failure = FileNotFoundError("synthetic missing seed")

    def missing(_):
        raise failure

    helper = SimpleNamespace(
        _read=missing,
        _bind=lambda p, h, i: p,
        _sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing",
        _failure_safe=lambda value: value,
    )
    out = tmp_path / "aggregate.json"
    with pytest.raises(FileNotFoundError) as caught:
        aggregate._aggregate(out, {"inputs": {}}, mean, helper, {"complete": False}, runner)
    assert caught.value is failure
    record = json.loads(out.read_text())
    assert not record["complete"] and record["reports"] == [] and record["error"] == repr(failure)
    assert json.loads(out.with_suffix(".aggregate-inputs.json").read_text()) == {}
    with pytest.raises(ValueError, match="immutable"):
        runner._refuse(out, mean, aggregate=True)


def test_aggregate_has_no_training_route_and_checks_overlap(aggregate):
    source = Path(aggregate.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"_execute", "_fit", "_validate_child"}
        if isinstance(node, ast.Import):
            assert all(name.name != "torch" for name in node.names)
    assert "begin >= previous_finish" in source
    assert "explicit launch receipt required" in source
    assert "args.auditor_sha256" in source


def test_reject_overlap_at_seventh_fractional_digit(aggregate):
    previous = aggregate._instant("2026-09-25T10:00:00.5251055Z")
    equivalent = aggregate._instant("2026-09-25T05:00:00.5251055-05:00")
    aggregate._chronology(equivalent, previous)
    aggregate._chronology(equivalent, None)
    with pytest.raises(ValueError, match="overlapping"):
        aggregate._chronology(aggregate._instant("2026-09-25T10:00:00.5251054Z"), previous)


def test_v2_scientific_and_audit_contract_functions_match_original(aggregate):
    original = ast.parse(
        (ROOT / "scripts/aggregate_bilinear_budget8000_replication.py").read_text()
    )
    revised = ast.parse(Path(aggregate.__file__).read_text())
    old_functions = {n.name: n for n in original.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in revised.body if isinstance(n, ast.FunctionDef)}
    translations = {
        "independent-audit-v2.json": "independent-audit.json",
        "audit-v2-execution-receipt.json": "audit-execution-receipt.json",
        "audit-v2-stdout.txt": "audit-stdout.txt",
    }

    class NormalizePaths(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = translations.get(node.value, node.value)
            return node

    for name in old_functions.keys() - {"main"}:
        revised_function = NormalizePaths().visit(new_functions[name])
        assert ast.dump(revised_function, include_attributes=False) == ast.dump(
            old_functions[name], include_attributes=False
        ), name


def test_v2_uses_distinct_outputs_and_auditor_paths(aggregate):
    source = Path(aggregate.__file__).read_text()
    for required in (
        "replication-v1/aggregate-v2.json",
        'out.parent / "independent-audit-v2.py"',
        'folder / "independent-audit-v2.json"',
        'folder / "audit-v2-execution-receipt.json"',
        'folder / "audit-v2-stdout.txt"',
    ):
        assert required in source
    assert len(aggregate._REPAIR_BINDINGS) == 7
    assert (
        "snapshots.update({suffix: path.read_bytes() for suffix, path in repair_paths.items()})"
        in source
    )


@pytest.mark.parametrize(
    "changed",
    ["exit_code", "auditor_sha256", "test_receipt_sha256", "stdout_sha256", "training_executed"],
)
def test_repair_rejects_changed_failed_attempt(aggregate, tmp_path, changed):
    failed = {
        "exit_code": 1,
        "seed": 1730,
        "audit_sha256": None,
        "training_executed": False,
        "terminal_completion_observed_by_primary": True,
        "stdout_sha256": aggregate._REPAIR_BINDINGS["failed-audit-stdout.txt"][1],
        "auditor_sha256": aggregate._REPAIR_BINDINGS["failed-auditor-script.py"][1],
        "test_receipt_sha256": aggregate._REPAIR_BINDINGS["failed-auditor-test-receipt.json"][1],
    }
    bound = []
    helper = SimpleNamespace(
        _bind=lambda p, h, i: bound.append((p, h)) or p, _read=lambda p: failed
    )
    assert len(aggregate._bind_repair(tmp_path, helper, {})) == 7
    assert len(bound) == 7
    failed[changed] = True if changed == "training_executed" else "wrong"
    with pytest.raises(ValueError, match="preserved failed audit"):
        aggregate._bind_repair(tmp_path, helper, {})


def test_production_inventory_accepts_actual_v1_four_and_v2_six_roles(aggregate):
    receipt = json.loads(
        (
            ROOT
            / "runs/learning/bilinear-budget8000-replication-runner-tests-v1/aggregate-receipt.json"
        ).read_text()
    )
    old = receipt["test_dependencies"]
    assert len(old) == 4
    aggregate._test_dependency_inventory(old, [Path(p) for p in old])
    new = dict(old)
    new.pop(str(ROOT / "tests/unit/test_bilinear_budget8000_replication_aggregate.py"))
    for relative in (
        "tests/unit/test_bilinear_budget8000_replication_aggregate_v2.py",
        "scripts/aggregate_bilinear_budget8000_replication.py",
        "tests/unit/test_bilinear_budget8000_replication_aggregate.py",
    ):
        path = ROOT / relative
        new[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(new) == 6
    aggregate._test_dependency_inventory(new, [Path(p) for p in new])
    with pytest.raises(ValueError):
        aggregate._test_dependency_inventory(old, [Path(p) for p in new])


@pytest.mark.parametrize("changed", ["missing", "extra", "list", "string"])
def test_production_inventory_rejects_inexact_schema(aggregate, tmp_path, changed):
    required = [tmp_path / "a", tmp_path / "b"]
    values = {str(p): "hash" for p in required}
    if changed == "missing":
        values.pop(str(required[0]))
    elif changed == "extra":
        values[str(tmp_path / "c")] = "hash"
    elif changed == "list":
        values = list(values)
    else:
        values = str(values)
    with pytest.raises(ValueError, match="dependency inventory"):
        aggregate._test_dependency_inventory(values, required)
