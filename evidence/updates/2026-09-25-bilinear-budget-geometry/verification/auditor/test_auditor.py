"""Synthetic paired-geometry checks; never reduce campaign score reports."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "runs/learning/bilinear-budget-geometry-v1/independent-audit.py"
SPEC = importlib.util.spec_from_file_location("paired_geometry_auditor", PATH)
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
h = a.helper(ROOT, {})


def reduced(category="exact_zero", group="TYPE_dual", index=0):
    scores = {
        "exact_zero": [9.0, 1.0, 0.0],
        "separated_missing": [9.0, 0.0, -1.0],
        "separated_extra": [9.0, 2.0, 1.0],
        "overlap_or_tie": [9.0, 0.0, 0.0],
    }[category]
    predicted = [i for i, z in zip((1024, 1025, 1026), scores, strict=True) if i != 1024 and z > 0]
    dimension = "COLOR" if group == "COLOR" else "TYPE"
    row = {
        "index": index,
        "subject": f"PKM_{index}",
        "dimension": dimension,
        "group": group,
        "prompt_ids": [1, 1024, 33 if dimension == "COLOR" else 32, 34, 5],
        "expected_set_ids": [1025],
        "symmetric_relation_logits": scores,
        "selected_set_ids": predicted,
        "metrics": h.metrics(predicted, [1025]),
        "strict_separation": scores[1] > scores[2],
    }
    return h.reduce_row(row, [1024, 1025, 1026])


def test_pair_deltas_are_direct_subtractions():
    old = reduced("overlap_or_tie")
    new = reduced("exact_zero")
    result = a.pair_row(old, new)
    assert result["delta"] == {"a": 1.0, "b": 0.0, "g": 1.0}
    assert result["budget2000"]["category"] == "overlap_or_tie"
    assert result["budget8000"]["category"] == "exact_zero"
    assert "false_positive" not in result["budget8000"]


@pytest.mark.parametrize("key", a.IDENTITY)
def test_paired_alignment_rejects_every_identity_drift(key):
    before, after = reduced(), reduced()
    after[key] = "changed"
    with pytest.raises(ValueError, match="paired query identity"):
        a.pair_row(before, after)


def test_all_sixteen_transitions_have_fixed_orientation_and_counts():
    rows = [a.pair_row(reduced(old), reduced(new)) for old in a.CATEGORIES for new in a.CATEGORIES]
    actual = a.transitions(rows)
    assert actual["matrix"] == [[1] * 4 for _ in range(4)]
    assert actual["query_count"] == 16
    assert actual["exact_gains"] == actual["exact_losses"] == 3
    assert actual["separation_gains"] == actual["separation_losses"] == 3
    assert actual["category_order"] == list(a.CATEGORIES)


def test_transition_gain_is_not_net_change_and_separation_can_regress():
    rows = [
        a.pair_row(reduced("separated_missing"), reduced()),
        a.pair_row(reduced(), reduced("overlap_or_tie")),
        a.pair_row(reduced("separated_extra"), reduced()),
    ]
    actual = a.transitions(rows)
    assert (actual["exact_gains"], actual["exact_losses"]) == (2, 1)
    assert (actual["separation_gains"], actual["separation_losses"]) == (0, 1)
    assert actual["matrix"][1][0] == actual["matrix"][0][3] == 1


def test_pooled_transitions_sum_counts_without_threshold_intervals():
    reports = {}
    for seed in a.SEEDS:
        before = {
            "rows": [reduced("separated_missing", group, i) for i, group in enumerate(a.GROUPS)]
        }
        after = {"rows": [reduced("exact_zero", group, i) for i, group in enumerate(a.GROUPS)]}
        reports[seed] = a.paired_report(seed, before, after)
    fresh = a.pooled_transitions(reports, (1730, 1731))
    assert fresh["aggregate"]["query_count"] == fresh["aggregate"]["exact_gains"] == 6
    assert fresh["groups"]["TYPE_dual"]["exact_gains"] == 2
    assert "shared_threshold" not in fresh["aggregate"]
    expected = {"gains": 3, "losses": 0, "groups": {g: {"gains": 1, "losses": 0} for g in a.GROUPS}}
    a.reconcile_pairs(reports[1729], expected)
    expected["groups"]["COLOR"]["losses"] = 1
    with pytest.raises(ValueError, match="group losses"):
        a.reconcile_pairs(reports[1729], expected)


def test_pair_coverage_drift_rejected():
    with pytest.raises(ValueError, match="coverage"):
        a.paired_report(1729, {"rows": [reduced()]}, {"rows": []})


@pytest.mark.parametrize("mutation", ["missing", "extra", "digest", "list"])
def test_production_inventory_path_rejects_corruption(mutation):
    expected = {"script.py": "a", "helper.py": "b", "new-test.py": "c", "old-test.py": "d"}
    a.dependency_inventory(expected, expected)
    actual = dict(expected)
    if mutation == "missing":
        actual.pop("script.py")
    elif mutation == "extra":
        actual["other.py"] = "e"
    elif mutation == "digest":
        actual["helper.py"] = "wrong"
    else:
        actual = list(actual)
    with pytest.raises(ValueError):
        a.dependency_inventory(actual, expected)


def test_child_binding_checks_saved_checkpoint_identity_without_payload():
    summary = {
        "complete": True,
        "artifact_sha256": {"child.json": "child", "checkpoint-final.pt": "weight"},
    }
    child = {"aggregate": {"exact_count": 1}, "groups": {}}
    summary["child"] = child
    audit = {
        "complete": True,
        "audit_passed": True,
        "checkpoint": {"child_checkpoint_sha256": "weight"},
        "child": child,
    }
    a.child_binding(summary, audit, child, "child")
    bad = copy.deepcopy(audit)
    bad["checkpoint"]["child_checkpoint_sha256"] = "other"
    with pytest.raises(ValueError, match="checkpoint"):
        a.child_binding(summary, bad, child, "child")
    bad = copy.deepcopy(audit)
    bad["audit_passed"] = False
    with pytest.raises(ValueError, match="audited"):
        a.child_binding(summary, bad, child, "child")


@pytest.mark.parametrize("which", ["summary", "audit", "decision", "binding"])
def test_chain_rejects_incomplete_unaccepted_or_wrong_binding(which):
    s = {"complete": True}
    proof = {"complete": True, "audit_passed": True, "summary_sha256": "s"}
    decision = {"evidence_accepted": True, "summary_sha256": "s", "audit_sha256": "a"}
    a.accepted(s, proof, decision, "s", "a")
    if which == "summary":
        s["complete"] = False
    elif which == "audit":
        proof["audit_passed"] = False
    elif which == "decision":
        decision["evidence_accepted"] = False
    else:
        decision["audit_sha256"] = "wrong"
    with pytest.raises(ValueError):
        a.accepted(s, proof, decision, "s", "a")


def test_historical_reduction_equality_rejects_changed_witness():
    historical = reduced()
    rebuilt = copy.deepcopy(historical)
    a.same(historical, rebuilt, "historical reduction equality")
    rebuilt["minimum_true_id"] = 1026
    with pytest.raises(ValueError, match="historical reduction"):
        a.same(historical, rebuilt, "historical reduction equality")


def test_immutable_output_and_helper_identity(tmp_path):
    p = tmp_path / "result.json"
    a.write(p, {"accepted": False})
    with pytest.raises(FileExistsError):
        a.write(p, {})
    with pytest.raises(ValueError, match="input hash"):
        a.bound(p, "wrong", {})
    assert (
        a.sha(ROOT / "runs/learning/bilinear-error-geometry-v1/independent-audit.py")
        == a.OLD_AUDITOR
    )


@pytest.fixture
def production_receipt(tmp_path, monkeypatch):
    names = (
        "scripts/diagnose_bilinear_budget_geometry.py",
        "scripts/diagnose_bilinear_errors.py",
        "tests/unit/test_bilinear_budget_geometry.py",
        "tests/unit/test_bilinear_error_geometry.py",
    )
    for index, name in enumerate(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic source {index}\n")
    stdout = tmp_path / "stdout.txt"
    stdout.write_text("synthetic completed test output\n")
    dependencies = {str((tmp_path / n).resolve()): a.sha(tmp_path / n) for n in names}
    runner, helper, test = [a.sha(tmp_path / n) for n in names[:3]]
    monkeypatch.setattr(a, "RUNNER", runner)
    monkeypatch.setattr(a, "PRIMARY_HELPER", helper)
    return tmp_path, {
        "passed": True,
        "gpu_used": False,
        "diagnostic_executed": False,
        "exit_code": 0,
        "terminal_completion_observed": True,
        "test_count": 2,
        "skipped": 0,
        "tested_script_sha256": runner,
        "tested_helper_sha256": helper,
        "test_sha256": test,
        "test_dependencies": dependencies,
        "stdout": {"path": str(stdout), "sha256": a.sha(stdout)},
    }


def test_exact_production_receipt_path_and_four_inventory_entries(production_receipt):
    root, receipt = production_receipt
    inputs = {}
    stdout = a.validate_test_receipt(root, receipt, inputs)
    assert len(inputs) == 5
    assert set(receipt["test_dependencies"]) <= set(inputs)
    assert str(stdout) in inputs


@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", False),
        ("exit_code", 1),
        ("exit_code", "0"),
        ("exit_code", None),
        ("terminal_completion_observed", False),
        ("terminal_completion_observed", 1),
        ("passed", False),
        ("gpu_used", True),
        ("diagnostic_executed", True),
        ("skipped", 1),
        ("skipped", False),
        ("test_count", 0),
        ("test_count", True),
        ("tested_script_sha256", "wrong"),
        ("tested_helper_sha256", "wrong"),
        ("test_sha256", "wrong"),
    ],
)
def test_production_receipt_rejects_terminal_flags_and_source_drift(
    production_receipt, field, value
):
    root, receipt = production_receipt
    receipt[field] = value
    with pytest.raises(ValueError):
        a.validate_test_receipt(root, receipt, {})


@pytest.mark.parametrize("field", ["exit_code", "terminal_completion_observed"])
def test_production_receipt_missing_terminal_field_rejected(production_receipt, field):
    root, receipt = production_receipt
    del receipt[field]
    with pytest.raises(ValueError):
        a.validate_test_receipt(root, receipt, {})


@pytest.mark.parametrize("mutation", ["missing", "extra", "digest"])
def test_actual_production_inventory_validation_path(production_receipt, mutation):
    root, receipt = production_receipt
    dependencies = receipt["test_dependencies"]
    if mutation == "missing":
        dependencies.pop(next(iter(dependencies)))
    elif mutation == "extra":
        dependencies["another-source.py"] = "a"
    else:
        dependencies[next(iter(dependencies))] = "wrong"
    with pytest.raises(ValueError, match="dependency"):
        a.validate_test_receipt(root, receipt, {})


@pytest.mark.parametrize("field", ["historical2000_summary_sha256", "historical2000_child_sha256"])
def test_matched_comparator_binding_rejects_other_seed_or_budget(field):
    current = {"historical2000_summary_sha256": "s", "historical2000_child_sha256": "c"}
    a.comparator_binding(current, "s", "c")
    current[field] = "other"
    with pytest.raises(ValueError, match="matched historical"):
        a.comparator_binding(current, "s", "c")


def test_historical1729_older_schema_without_child_field():
    a.comparator_binding({"historical2000_summary_sha256": "s"}, "s", "c")
    with pytest.raises(KeyError):
        a.comparator_binding({}, "s", "c")
