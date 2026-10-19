"""Synthetic paired-budget geometry and real provenance schemas; no score reductions."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "paired_geometry_runner", ROOT / "scripts/diagnose_bilinear_budget_geometry.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def old_tests():
    spec = importlib.util.spec_from_file_location(
        "old_geometry_tests", ROOT / "tests/unit/test_bilinear_error_geometry.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reduced(runner, old_tests, category, index=0, group="TYPE_dual"):
    scores = {
        "exact_zero": [9, 1, 2, 0, -1, -2],
        "separated_missing": [9, 0, 1, -1, -2, -3],
        "separated_extra": [9, 2, 3, 1, 0, -1],
        "overlap_or_tie": [9, 0, 2, 0, -1, -2],
    }[category]
    return runner.helper(ROOT)._reduce(old_tests.row(scores, index, group), list(range(1024, 1030)))


def test_full_four_by_four_transition_matrix(runner, old_tests):
    before, after = [], []
    for left in runner.CATEGORIES:
        for right in runner.CATEGORIES:
            index = len(before)
            before.append(reduced(runner, old_tests, left, index))
            after.append(reduced(runner, old_tests, right, index))
    rows = runner.paired_rows(before, after)
    result = runner.transition(rows)
    assert result["matrix"] == [[1] * 4 for _ in range(4)]
    assert result["query_count"] == 16
    assert result["exact_gains"] == result["exact_losses"] == 3
    assert result["separation_gains"] == result["separation_losses"] == 3
    assert result["category_order"] == list(runner.CATEGORIES)


def test_paired_deltas_are_exact_within_parent(runner, old_tests):
    left = reduced(runner, old_tests, "separated_missing")
    right = reduced(runner, old_tests, "separated_extra")
    row = runner.paired_rows([left], [right])[0]
    assert row["delta"] == {"a": 2, "b": 2, "g": 0}
    assert row["budget2000"]["a"] == 0 and row["budget8000"]["a"] == 2
    assert "threshold" not in row and "selected_set_ids" not in row


@pytest.mark.parametrize(
    "key", ["index", "subject", "dimension", "group", "prompt_ids", "expected_set_sha256"]
)
def test_cross_budget_identity_drift_rejected(runner, old_tests, key):
    left = reduced(runner, old_tests, "exact_zero")
    right = copy.deepcopy(left)
    right[key] = "changed"
    with pytest.raises(ValueError, match="paired query identity"):
        runner.paired_rows([left], [right])


def test_pair_count_drift_and_empty_transition_rejected(runner, old_tests):
    with pytest.raises(ValueError, match="paired query count"):
        runner.paired_rows([reduced(runner, old_tests, "exact_zero")], [])
    with pytest.raises(ValueError, match="nonempty"):
        runner.transition([])


def test_group_reconciliation_and_pooled_counts(runner, old_tests):
    h = runner.helper(ROOT)
    left = [
        reduced(runner, old_tests, "overlap_or_tie", 0, "TYPE_dual"),
        reduced(runner, old_tests, "exact_zero", 1, "COLOR"),
    ]
    right = [
        reduced(runner, old_tests, "exact_zero", 0, "TYPE_dual"),
        reduced(runner, old_tests, "exact_zero", 1, "COLOR"),
    ]
    before = {
        "rows": left,
        "groups": {
            g: h._totals([r for r in left if r["group"] == g]) for g in ("COLOR", "TYPE_dual")
        },
    }
    after = {
        "rows": right,
        "groups": {
            g: h._totals([r for r in right if r["group"] == g]) for g in ("COLOR", "TYPE_dual")
        },
    }
    comparison = {
        "gains": 1,
        "losses": 0,
        "groups": {"COLOR": {"gains": 0, "losses": 0}, "TYPE_dual": {"gains": 1, "losses": 0}},
    }
    report = runner.paired_report(1730, before, after, comparison)
    assert report["aggregate"]["separation_gains"] == 1
    pool = runner.pooled_transitions({1730: report, 1731: report}, [1730, 1731])
    assert pool["aggregate"]["query_count"] == 4 and pool["aggregate"]["exact_gains"] == 2
    assert "shared_threshold" not in pool["aggregate"]
    assert "shared_threshold" not in pool["groups"]["TYPE_dual"]
    comparison["gains"] = 2
    with pytest.raises(ValueError, match="accepted paired"):
        runner.paired_report(1730, before, after, comparison)


@pytest.mark.parametrize("part", ["rows", "aggregate", "groups", "child_sha256"])
def test_historical_reduction_exact_equality(runner, part):
    old = {
        "rows": [{"a": 1}],
        "aggregate": {"exact_count": 1},
        "groups": {"x": 1},
        "child_sha256": "abc",
    }
    runner.historical_equal(copy.deepcopy(old), old)
    changed = copy.deepcopy(old)
    changed[part] = "different"
    with pytest.raises(ValueError, match="historical 2000"):
        runner.historical_equal(changed, old)


@pytest.mark.parametrize(
    "bad",
    [
        "summary_complete",
        "audit_complete",
        "audit_passed",
        "audit_summary",
        "decision_summary",
        "decision_audit",
        "accepted",
    ],
)
def test_incomplete_or_unaccepted_chain_rejected(runner, bad):
    s = {"complete": True}
    a = {"complete": True, "audit_passed": True, "summary_sha256": "summary"}
    d = {"summary_sha256": "summary", "audit_sha256": "audit", "evidence_accepted": True}
    runner.accepted_chain(s, a, d, "summary", "audit")
    target, key = {
        "summary_complete": (s, "complete"),
        "audit_complete": (a, "complete"),
        "audit_passed": (a, "audit_passed"),
        "audit_summary": (a, "summary_sha256"),
        "decision_summary": (d, "summary_sha256"),
        "decision_audit": (d, "audit_sha256"),
        "accepted": (d, "evidence_accepted"),
    }[bad]
    target[key] = False
    with pytest.raises(ValueError, match="accepted complete"):
        runner.accepted_chain(s, a, d, "summary", "audit")


@pytest.mark.parametrize("bad", ["columns", "count", "order", "complete"])
def test_child_schema_drift(runner, bad):
    child = {
        "complete": True,
        "query_count": 222,
        "product_token_ids": list(range(1024, 2049)),
        "responses": [{"index": i} for i in range(222)],
    }
    runner.child_contract(child)
    if bad == "columns":
        child["product_token_ids"].reverse()
    elif bad == "count":
        child["query_count"] = 221
    elif bad == "order":
        child["responses"].reverse()
    else:
        child["complete"] = False
    with pytest.raises(ValueError, match="column contract"):
        runner.child_contract(child)


def receipt(runner, tmp_path):
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("synthetic completed\n", encoding="utf-8")
    return {
        "passed": True,
        "exit_code": 0,
        "terminal_completion_observed": True,
        "gpu_used": False,
        "diagnostic_executed": False,
        "skipped": 0,
        "test_count": 1,
        "tested_script_sha256": runner.sha(ROOT / "scripts/diagnose_bilinear_budget_geometry.py"),
        "tested_helper_sha256": runner.HELPER_SHA,
        "test_sha256": runner.sha(Path(__file__)),
        "test_dependencies": runner.test_dependencies(ROOT),
        "stdout": {"path": str(stdout), "sha256": runner.sha(stdout)},
    }


def test_production_receipt_schema_with_actual_source_dependencies(runner, tmp_path):
    data = receipt(runner, tmp_path)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    inputs = {}
    runner.test_receipt(ROOT, runner.helper(ROOT), path, runner.sha(path), inputs)
    assert len(data["test_dependencies"]) == 4
    assert set(data["test_dependencies"]) <= set(inputs)
    assert inputs[str(ROOT / "scripts/diagnose_bilinear_errors.py")] == runner.HELPER_SHA


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "extra",
        "wrong_hash",
        "wrong_type",
        "helper",
        "passed",
        "gpu",
        "executed",
        "skip",
        "count",
        "stdout",
    ],
)
def test_receipt_and_helper_dependency_rejection(runner, tmp_path, bad):
    data = receipt(runner, tmp_path)
    if bad == "missing":
        data["test_dependencies"].pop(next(iter(data["test_dependencies"])))
    elif bad == "extra":
        data["test_dependencies"]["extra"] = "bad"
    elif bad == "wrong_hash":
        data["test_dependencies"][next(iter(data["test_dependencies"]))] = "bad"
    elif bad == "wrong_type":
        data["test_dependencies"] = list(data["test_dependencies"])
    elif bad == "stdout":
        data["stdout"]["sha256"] = "bad"
    else:
        key, value = {
            "helper": ("tested_helper_sha256", "bad"),
            "passed": ("passed", False),
            "gpu": ("gpu_used", True),
            "executed": ("diagnostic_executed", True),
            "skip": ("skipped", 1),
            "count": ("test_count", True),
        }[bad]
        data[key] = value
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.test_receipt(ROOT, runner.helper(ROOT), path, runner.sha(path), {})


@pytest.mark.parametrize(
    "key,value",
    [
        ("exit_code", None),
        ("exit_code", 1),
        ("exit_code", False),
        ("exit_code", "0"),
        ("exit_code", 0.0),
        ("terminal_completion_observed", None),
        ("terminal_completion_observed", False),
        ("terminal_completion_observed", 1),
        ("skipped", False),
    ],
)
def test_production_receipt_requires_observed_zero_exit(runner, tmp_path, key, value):
    data = receipt(runner, tmp_path)
    if value is None:
        data.pop(key)
    else:
        data[key] = value
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="completed synthetic"):
        runner.test_receipt(ROOT, runner.helper(ROOT), path, runner.sha(path), {})


@pytest.mark.parametrize(
    "key",
    ["parent_checkpoint_sha256", "historical2000_summary_sha256", "historical2000_child_sha256"],
)
def test_matched_historical_identity_rejects_wrong_comparator(runner, key):
    before = {
        "parent_checkpoint_sha256": "parent",
        "summary_sha256": "summary",
        "child_sha256": "child",
    }
    after = {
        "parent_checkpoint_sha256": "parent",
        "summary": {
            "historical2000_summary_sha256": "summary",
            "historical2000_child_sha256": "child",
        },
    }
    runner.matched_historical(before, after)
    (after if key == "parent_checkpoint_sha256" else after["summary"])[key] = "wrong"
    with pytest.raises(ValueError, match="matched historical"):
        runner.matched_historical(before, after)


def test_frozen_helper_rejects_changed_bytes_before_import(runner, tmp_path):
    path = tmp_path / "scripts/diagnose_bilinear_errors.py"
    path.parent.mkdir()
    path.write_text("raise AssertionError('must not execute')", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen scientific helper"):
        runner.helper(tmp_path)


def test_chain_only_preflight_does_not_reduce_or_open_payloads(runner, monkeypatch):
    h = runner.helper(ROOT)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not reduce scores")

    for name in ("_seed", "_reduce", "_totals", "_pool", "_overlap"):
        monkeypatch.setattr(h, name, forbidden)
    original = Path.read_bytes

    def guarded(path):
        assert path.suffix in (".json", ".md"), path
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    chains, old, _, inputs = runner.preflight(
        ROOT, ROOT / "docs/experiments/2026-09-25-bilinear-budget-geometry-plan.md", h
    )
    assert set(chains) == {2000, 8000} and set(old) == {1729, 1730, 1731}
    assert len(inputs) == 34
    assert all(Path(p).suffix in (".json", ".md", ".py") for p in inputs)


def test_synthetic_execute_and_immutable_namespace(runner, old_tests, tmp_path):
    h = runner.helper(ROOT)
    for relative in (
        "scripts/diagnose_bilinear_errors.py",
        "tests/unit/test_bilinear_budget_geometry.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    plan = tmp_path / "plan.md"
    plan.write_text("synthetic only", encoding="utf-8")
    rec = tmp_path / "receipt.json"
    rec.write_text("{}", encoding="utf-8")
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("synthetic", encoding="utf-8")
    source_row = old_tests.row([9, 1, 2, 0, -1, -2])
    reduction = h._reduce(source_row, list(range(1024, 1030)))
    metrics = h._totals([reduction])
    child = {
        "responses": [source_row],
        "product_token_ids": list(range(1024, 1030)),
        "aggregate": metrics,
        "groups": {"TYPE_dual": metrics},
    }
    accepted = {"gains": 0, "losses": 0, "groups": {"TYPE_dual": {"gains": 0, "losses": 0}}}
    chains = {
        b: {
            s: {
                "child": child,
                "child_sha256": "synthetic",
                "checkpoint_sha256": "identity-only",
                "summary": {"child": {"paired_vs_historical2000": accepted}},
            }
            for s in runner.SEEDS
        }
        for b in runner.BUDGETS
    }
    old = {s: h._seed(s, child, "synthetic") for s in runner.SEEDS}
    old_summary = {
        "fresh_two": h._pool(old, [1730, 1731]),
        "all_three": h._pool(old, list(runner.SEEDS)),
        "overlap": h._overlap(old),
    }
    digest = runner.execute(tmp_path, h, chains, old, old_summary, {}, plan, rec, stdout)
    out = tmp_path / "runs/learning/bilinear-budget-geometry-v1"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] and runner.sha(out / "summary.json") == digest
    assert len(summary["per_seed"]) == 3 and len(summary["snapshot_sha256"]) == 7
    assert len(list(out.glob("seed-*.json"))) == 9
    assert summary["transitions"]["fresh_two"]["aggregate"]["query_count"] == 2
    with pytest.raises(ValueError, match="immutable output namespace"):
        runner.execute(tmp_path, h, chains, old, old_summary, {}, plan, rec, stdout)
    assert runner.sha(out / "summary.json") == digest
