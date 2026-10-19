"""Synthetic score geometry; no real saved reports or neural runtime."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "geometry_test_runner", ROOT / "scripts/diagnose_bilinear_errors.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def row(scores, index=0, group="TYPE_dual"):
    columns = list(range(1024, 1030))
    truth = [1025, 1026]
    selected = [i for i, z in zip(columns, scores, strict=True) if i != 1024 and z > 0]
    fp = set(selected) - set(truth)
    fn = set(truth) - set(selected)
    return {
        "index": index,
        "subject": "PKM_TEST",
        "dimension": "TYPE",
        "group": group,
        "prompt_ids": [1, 1024, 32, 34, 5],
        "expected_set_ids": truth,
        "selected_set_ids": selected,
        "symmetric_relation_logits": scores,
        "metrics": {
            "exact": not fp and not fn,
            "false_positive_count": len(fp),
            "false_negative_count": len(fn),
        },
        "strict_separation": min(scores[1:3]) > max(scores[3:]),
    }


@pytest.mark.parametrize(
    "scores,category",
    [
        ([9, 1, 2, 0, -1, -2], "exact_zero"),
        ([9, 0, 1, -1, -2, -3], "separated_missing"),
        ([9, 2, 3, 1, 0, -1], "separated_extra"),
        ([9, 0, 2, 0, -1, -2], "overlap_or_tie"),
        ([9, -1, 2, 1, -2, -3], "overlap_or_tie"),
    ],
)
def test_four_categories_and_exact_zero_boundaries(runner, scores, category):
    reduced = runner._reduce(row(scores), list(range(1024, 1030)))
    assert reduced["category"] == category
    assert reduced["g"] == reduced["a"] - reduced["b"]
    assert reduced["exact"] == (category == "exact_zero")


def test_extrema_choose_lowest_id_and_subject_is_excluded(runner):
    result = runner._reduce(row([100, 1, 1, 0, 0, -1]), list(range(1024, 1030)))
    assert result["minimum_true_id"] == 1025 and result["maximum_false_id"] == 1027
    assert result["b"] == 0 and result["false_positive"] == result["false_negative"] == []


def test_errors_keep_exact_member_ids_and_scores(runner):
    result = runner._reduce(row([100, 0, 2, 1, -2, -3]), list(range(1024, 1030)))
    assert result["false_positive"] == [{"id": 1027, "score": 1}]
    assert result["false_negative"] == [{"id": 1025, "score": 0}]


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), 0.1, 1e100, True, "1"]
)
def test_nonfinite_nonfp32_and_nonnumeric_rejected(runner, value):
    data = row([9, 1, 2, 0, -1, -2])
    data["symmetric_relation_logits"][0] = value
    with pytest.raises(ValueError):
        runner._reduce(data, list(range(1024, 1030)))


@pytest.mark.parametrize(
    "change",
    [
        "prediction",
        "exact",
        "fp",
        "fn",
        "separation",
        "self",
        "columns",
        "count",
        "truth",
        "empty_truth",
        "empty_false",
        "prompt",
    ],
)
def test_invalid_saved_contract_rejected(runner, change):
    data = row([9, 1, 2, 0, -1, -2])
    columns = list(range(1024, 1030))
    if change == "prediction":
        data["selected_set_ids"] = [1025]
    elif change == "exact":
        data["metrics"]["exact"] = False
    elif change == "fp":
        data["metrics"]["false_positive_count"] = 1
    elif change == "fn":
        data["metrics"]["false_negative_count"] = 1
    elif change == "separation":
        data["strict_separation"] = False
    elif change == "self":
        data["expected_set_ids"] = [1024, 1025]
    elif change == "columns":
        columns = columns[::-1]
    elif change == "count":
        data["symmetric_relation_logits"].pop()
    elif change == "truth":
        data["expected_set_ids"] = [1026, 1025]
    elif change == "empty_truth":
        data["expected_set_ids"] = []
    elif change == "empty_false":
        data["expected_set_ids"] = columns[1:]
    elif change == "prompt":
        data["prompt_ids"] = []
    with pytest.raises(ValueError):
        runner._reduce(data, columns)


def test_shared_threshold_requires_one_interval_for_all_rows(runner):
    rows = [
        runner._reduce(row([9, 0, 1, -1, -2, -3], 0), list(range(1024, 1030))),
        runner._reduce(row([9, 2, 3, 1, 0, -1], 1), list(range(1024, 1030))),
    ]
    shared = runner._totals(rows)["shared_threshold"]
    assert shared == {
        "A": 0,
        "B": 1,
        "exists": False,
        "lower_inclusive": True,
        "upper_exclusive": True,
        "A_query_index": 0,
        "B_query_index": 1,
    }
    assert all(r["strict_separation"] for r in rows)


def test_shared_extrema_query_ties_choose_lowest_index(runner):
    rows = [runner._reduce(row([9, 1, 2, 0, -1, -2], i), list(range(1024, 1030))) for i in [4, 2]]
    interval = runner._totals(rows)["shared_threshold"]
    assert interval["exists"] and interval["A_query_index"] == interval["B_query_index"] == 2


@pytest.mark.parametrize(
    "key", ["index", "subject", "dimension", "group", "prompt_ids", "expected_set_ids"]
)
def test_alignment_rejects_any_identity_drift(runner, key):
    original = {"responses": [row([9, 1, 2, 0, -1, -2])]}
    children = {s: copy.deepcopy(original) for s in [1729, 1730, 1731]}
    runner._alignment(children)
    children[1731]["responses"][0][key] = "changed"
    with pytest.raises(ValueError, match="aligned"):
        runner._alignment(children)


def test_overlap_counts_shared_queries_and_pool_omits_threshold(runner):
    good = runner._reduce(row([9, 1, 2, 0, -1, -2]), list(range(1024, 1030)))
    bad = runner._reduce(row([9, 0, 2, 0, -1, -2]), list(range(1024, 1030)))
    reports = {
        1729: {"rows": [bad, good]},
        1730: {"rows": [good, good]},
        1731: {"rows": [bad, good]},
    }
    overlap = runner._overlap(reports)
    assert overlap["shared_queries"] == 2
    assert overlap["failure_seed_histogram"] == {"1729,1731": 1, "none": 1}
    assert overlap["failed_queries"][0]["seeds"] == [1729, 1731]
    pooled = runner._pool(reports, [1730, 1731])
    assert pooled["aggregate"]["query_count"] == 4
    assert pooled["aggregate"]["exact_count"] == 3
    assert "shared_threshold" not in pooled["aggregate"]


def test_truth_hash_has_documented_serialization(runner):
    import hashlib

    reduced = runner._reduce(row([9, 1, 2, 0, -1, -2]), list(range(1024, 1030)))
    assert reduced["expected_set_sha256"] == hashlib.sha256(b"[\n  1025,\n  1026\n]\n").hexdigest()


def test_immutable_output_and_publication_size(runner, tmp_path):
    target = tmp_path / "evidence.json"
    runner._write(target, {"kept": True})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        runner._write(target, {"kept": False})
    assert target.read_bytes() == before
    with pytest.raises(ValueError, match="512KiB"):
        runner._write(tmp_path / "large", b"x" * (512 * 1024 + 1))
    assert not (tmp_path / "large").exists()


def test_bound_input_rejects_wrong_hash_and_binary_type(runner, tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"ok": True}))
    with pytest.raises(ValueError, match="identity"):
        runner._read_bound(path, "wrong", {})
    assert runner._read_bound(path, runner._sha(path), {}) == {"ok": True}
    binary = tmp_path / "checkpoint.pt"
    binary.write_bytes(b"never-opened")
    with pytest.raises(ValueError, match="only JSON"):
        runner._read_bound(binary, "wrong", {})
