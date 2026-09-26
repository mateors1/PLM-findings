"""Synthetic independent-auditor boundary tests; no real score reductions."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "bilinear-error-geometry-v1/independent-audit.py"
SPEC = importlib.util.spec_from_file_location("geometry_auditor", PATH)
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
COLS = [1024, 1025, 1026, 1027, 1028]


def row(values=(100.0, 1.0, 1.0, -1.0, 0.0), index=0):
    selected = [i for i, value in zip(COLS, values, strict=True) if i != 1024 and value > 0]
    truth = [1025, 1026]
    return {
        "index": index,
        "subject": f"PKM_{index}",
        "dimension": "TYPE",
        "group": "TYPE_dual",
        "prompt_ids": [1, 1024, 32, 34, 5],
        "expected_set_ids": truth,
        "symmetric_relation_logits": list(values),
        "selected_set_ids": selected,
        "metrics": a.metrics(selected, truth),
        "strict_separation": min(values[1:3]) > max(values[3:]),
    }


@pytest.mark.parametrize(
    "values,category,fp,fn",
    [
        ((9.0, 1.0, 1.0, 0.0, -1.0), "exact_zero", [], []),
        ((9.0, 0.0, 1.0, -1.0, -2.0), "separated_missing", [], [1025]),
        ((9.0, -1.0, -1.0, -2.0, -3.0), "separated_missing", [], [1025, 1026]),
        ((9.0, 2.0, 3.0, 1.0, 0.0), "separated_extra", [1027], []),
        ((9.0, 0.0, 1.0, 0.0, -1.0), "overlap_or_tie", [], [1025]),
        ((9.0, -1.0, 1.0, 1.0, -1.0), "overlap_or_tie", [1027], [1025]),
    ],
)
def test_categories_and_error_members(values, category, fp, fn):
    actual = a.reduce_row(row(values), COLS)
    assert actual["category"] == category
    assert [p["id"] for p in actual["false_positive"]] == fp
    assert [p["id"] for p in actual["false_negative"]] == fn
    assert actual["g"] == actual["a"] - actual["b"]


def test_tie_witnesses_lowest_id_and_subject_exclusion():
    result = a.reduce_row(row((1000.0, 1.0, 1.0, 0.0, 0.0)), COLS)
    assert result["minimum_true_id"] == 1025
    assert result["maximum_false_id"] == 1027
    assert result["exact"] is True


def test_negative_zero_false_is_legal_true_zero_is_missing():
    assert a.reduce_row(row((1.0, 1.0, 1.0, -0.0, 0.0)), COLS)["exact"] is True
    assert a.reduce_row(row((1.0, -0.0, 1.0, -1.0, -2.0)), COLS)["category"] == "separated_missing"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.1, True, "1", 1e100])
def test_bad_numeric_scores_rejected(bad):
    with pytest.raises(ValueError):
        a.fp32(bad)


def test_smallest_positive_fp32_is_not_rounded_to_zero():
    tiny = 2.0**-149
    result = a.reduce_row(row((1.0, tiny, tiny, 0.0, -tiny)), COLS)
    assert result["exact"] is True
    assert result["g"] == tiny


@pytest.mark.parametrize(
    "key,value",
    [
        ("expected_set_ids", [1024, 1025]),
        ("expected_set_ids", [1025, 1025]),
        ("expected_set_ids", []),
        ("expected_set_ids", [1025, 1026, 1027, 1028]),
        ("selected_set_ids", [1025]),
        ("selected_set_ids", [1024, 1025, 1026]),
        ("prompt_ids", [1, 1024, 33, 34, 5]),
        ("group", "COLOR"),
        ("strict_separation", 1),
    ],
)
def test_corrupt_row_contract(key, value):
    data = row()
    data[key] = value
    with pytest.raises(ValueError):
        a.reduce_row(data, COLS)


def test_metric_tamper_rejected():
    data = row()
    data["metrics"]["f1"] = 0.999
    with pytest.raises(ValueError, match="saved row metrics"):
        a.reduce_row(data, COLS)


def test_bad_columns_and_width_rejected():
    with pytest.raises(ValueError, match="columns"):
        a.reduce_row(row(), list(reversed(COLS)))
    data = row()
    data["symmetric_relation_logits"].pop()
    with pytest.raises(ValueError, match="logit width"):
        a.reduce_row(data, COLS)


def test_shared_interval_can_fail_despite_each_query_separated():
    rows = [
        a.reduce_row(row((5.0, -1.0, -1.0, -2.0, -2.0), 2), COLS),
        a.reduce_row(row((5.0, 2.0, 2.0, 1.0, 1.0), 1), COLS),
    ]
    result = a.totals(rows)
    assert result["strict_separation_count"] == 2
    assert result["shared_threshold"] == {
        "A": -1.0,
        "B": 1.0,
        "exists": False,
        "lower_inclusive": True,
        "upper_exclusive": True,
        "A_query_index": 2,
        "B_query_index": 1,
    }
    assert "shared_threshold" not in a.totals(rows, False)


def test_shared_interval_witnesses_lowest_query_index():
    data = [a.reduce_row(row(index=5), COLS), a.reduce_row(row(index=1), COLS)]
    threshold = a.totals(data)["shared_threshold"]
    assert threshold["exists"] is True
    assert threshold["A_query_index"] == threshold["B_query_index"] == 1


def test_alignment_rejects_query_drift_and_duplicate():
    children = {s: {"responses": [row()]} for s in a.SEEDS}
    a.align(children)
    children[1730]["responses"][0]["subject"] = "WRONG"
    with pytest.raises(ValueError, match="query alignment"):
        a.align(children)
    children = {s: {"responses": [row(), row()]} for s in a.SEEDS}
    with pytest.raises(ValueError, match="unique queries"):
        a.align(children)


def test_overlap_uses_query_identities_not_independent_observations():
    good = a.reduce_row(row(), COLS)
    bad = a.reduce_row(row((9.0, 0.0, 1.0, -1.0, -2.0)), COLS)
    reports = {1729: {"rows": [bad]}, 1730: {"rows": [good]}, 1731: {"rows": [bad]}}
    actual = a.overlap(reports)
    assert actual["shared_queries"] == 1
    assert actual["failure_seed_histogram"] == {"1729,1731": 1}
    assert actual["failed_queries"][0]["seeds"] == [1729, 1731]


@pytest.mark.parametrize(
    "key,value",
    [
        ("exit_code", 1),
        ("exit_code", False),
        ("terminal_completion_observed_by_primary", False),
        ("summary_sha256", "wrong"),
        ("stdout_sha256", "wrong"),
    ],
)
def test_execution_binding_rejects_bad_receipt(key, value):
    receipt = {
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "summary_sha256": "summary",
        "stdout_sha256": "stdout",
    }
    a.execution_receipt(receipt, "summary", "stdout")
    receipt[key] = value
    with pytest.raises(ValueError):
        a.execution_receipt(receipt, "summary", "stdout")


def test_owner_acceptance_and_hash_bindings_required():
    summary = {"complete": True}
    audit = {"complete": True, "audit_passed": True, "summary_sha256": "s"}
    decision = {"summary_sha256": "s", "audit_sha256": "a", "evidence_accepted": True}
    a.accepted(summary, audit, decision, "s", "a")
    bad = copy.deepcopy(decision)
    bad["evidence_accepted"] = False
    with pytest.raises(ValueError, match="owner acceptance"):
        a.accepted(summary, audit, bad, "s", "a")
    with pytest.raises(ValueError, match="audit binding"):
        a.accepted(summary, audit, decision, "changed", "a")


def test_immutable_write_and_corrupt_bytes(tmp_path):
    path = tmp_path / "saved.json"
    a.write(path, {"example": 1})
    digest = a.sha(path)
    assert a.bound(path, digest, {}) == {"example": 1}
    with pytest.raises(FileExistsError):
        a.write(path, {})
    path.write_text("{}")
    with pytest.raises(ValueError, match="input hash"):
        a.bound(path, digest, {})


def test_zero_false_and_empty_prediction_metric_conventions():
    assert a.metrics([], [1025])["f1"] == 0.0
    data = row((100.0, -1.0, -1.0, -2.0, -2.0))
    assert a.reduce_row(data, COLS)["false_positive"] == []
    assert math.isfinite(data["metrics"]["recall"])
