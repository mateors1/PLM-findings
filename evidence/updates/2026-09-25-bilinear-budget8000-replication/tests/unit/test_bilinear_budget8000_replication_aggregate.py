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
    return load("scripts/aggregate_bilinear_budget8000_replication.py", "aggregate8000_test")


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
