"""Synthetic saved-head membership diagnosis; no real diagnostic measurements."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "missing_source", ROOT / "scripts/diagnose_missing_source_membership.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def raw(truth=None, values=None):
    truth = [1025, 1100, 1101] if truth is None else truth
    logits = [-2.0] * 1025
    for token, value in (values or {}).items():
        logits[token - 1024] = value
    prompt = [1, 1024, 32, 34, 5]
    sets = [[1025 + i] for i in range(8)]
    missing = set(truth) - set().union(*(set(s) for s in sets))
    return {
        "index": 0,
        "subject": "PKM_SUBJECT",
        "dimension": "TYPE",
        "group": "TYPE_dual",
        "prompt_ids": prompt,
        "expected_set_ids": truth,
        "symmetric_relation_logits": logits,
        "source_set_ids": sets,
        "source_subject_removed": [False] * 8,
        "source_paths": [
            {
                "token_ids": [*prompt, *ids, 2],
                "targets": [f"PKM_{ids[0]}"],
                "terminated": True,
                "protocol_valid": True,
                "error": None,
                "decoding": runner._BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            }
            for rank, ids in enumerate(sets, 1)
        ],
        "diagnosis": {
            "source_union_contains_truth": not missing,
            "baseline_state": "unavailable_and_truth_missing_from_all_sources"
            if missing
            else "selected_exact",
        },
    }


def focused(truth=None, values=None):
    return runner._focused(runner._coverage(raw(truth, values)))


@pytest.mark.parametrize(
    "values,category,counts",
    [
        ({1100: 1.0, 1101: 2.0}, "all", (2, 0, 0)),
        ({1100: 1.0, 1101: 0.0}, "some", (1, 1, 0)),
        ({1100: -1.0, 1101: 0.0}, "none", (0, 1, 1)),
    ],
)
def test_all_some_none_exact_zero_signs(values, category, counts):
    value = focused(values=values)
    assert value["positive_omitted_category"] == category
    summary = value["omitted_summary"]
    assert tuple(summary[k] for k in ("positive_count", "zero_count", "negative_count")) == counts


def test_rank_ties_false_comparisons_subject_exclusion_and_oracle_k():
    value = focused(values={1024: 100.0, 1099: 3.0, 1100: 3.0, 1101: 3.0, 1200: 4.0})
    first, second = value["omitted_members"]
    assert first == {
        "token_id": 1100,
        "logit": 3.0,
        "rank": 3,
        "false_strictly_higher": 1,
        "false_tied": 1,
        "oracle_top_k": True,
    }
    assert second["rank"] == 4 and not second["oracle_top_k"]
    assert value["oracle_k"] == 3


def test_all_equal_logits_rank_by_id_and_subject_never_false_tie():
    value = focused(truth=[1100], values={})
    member = value["omitted_members"][0]
    assert member["rank"] == 76
    assert member["false_strictly_higher"] == 0
    assert member["false_tied"] == 1023


def test_empty_covered_group_has_null_spreads_and_denominators():
    value = focused(truth=[1100])
    assert value["covered_members"] == []
    assert value["covered_summary"] == {
        "count": 0,
        "positive_count": 0,
        "zero_count": 0,
        "negative_count": 0,
        "logit": None,
        "rank": None,
    }
    total = runner._totals([value])
    assert total["covered"]["member_occurrences"] == total["covered"]["nonempty_query_count"] == 0
    assert total["covered"]["positive_member_fraction"] is None
    assert total["covered"]["mean_query_positive_fraction"] is None


def test_pooled_member_and_query_fractions_use_distinct_denominators():
    first = focused(truth=[1100], values={1100: 1.0})
    second = focused(values={1100: -1.0, 1101: -1.0, 1025: 1.0})
    total = runner._totals([first, second])
    assert total["focused_queries"] == 2 and total["distinct_focused_queries"] == 1
    assert total["omitted"]["positive_member_fraction"] == 1 / 3
    assert total["omitted"]["mean_query_positive_fraction"] == 0.5
    assert total["omitted"]["member_occurrences"] == 3
    assert total["omitted"]["nonempty_query_count"] == 2
    assert total["covered"]["nonempty_query_count"] == 1
    assert total["covered"]["mean_query_positive_fraction"] == 1.0


def test_median_values_and_rank_even_cardinality():
    value = focused(values={1100: 1.0, 1101: 3.0})
    assert value["omitted_summary"]["logit"] == {"min": 1.0, "median": 2.0, "max": 3.0}
    assert value["omitted_summary"]["rank"] == {"min": 1, "median": 1.5, "max": 2}


def test_focus_selected_by_missing_members_not_scores():
    first = runner._coverage(raw(values={1100: 100.0, 1101: 100.0}))
    second = runner._coverage(raw(values={1100: -100.0, 1101: -100.0}))
    assert first["missing"] == second["missing"] == [1100, 1101]
    covered = runner._coverage(raw(truth=[1025]))
    with pytest.raises(ValueError, match="focus requires"):
        runner._focused(covered)


def test_raw_subject_removal_preserved_but_never_counts_as_coverage():
    value = raw()
    path = value["source_paths"][0]
    path["token_ids"].insert(-1, 1024)
    path["targets"].append("PKM_SUBJECT")
    value["source_subject_removed"][0] = True
    coverage = runner._coverage(value)
    assert 1024 not in coverage["union"]
    assert path["token_ids"][-2] == 1024


@pytest.mark.parametrize(
    "corruption", ["raw", "source", "duplicate", "unfinished", "truth", "prompt", "coverage"]
)
def test_corrupt_source_membership_rejected(corruption):
    value = raw()
    if corruption == "raw":
        value["source_paths"][0]["token_ids"][5] = 1100
    elif corruption == "source":
        value["source_set_ids"][0] = [1100]
    elif corruption == "duplicate":
        value["source_paths"][0]["token_ids"].insert(-1, 1025)
        value["source_paths"][0]["targets"].append("PKM_1025")
    elif corruption == "unfinished":
        value["source_paths"][0]["terminated"] = False
    elif corruption == "truth":
        value["expected_set_ids"] = [1024]
    elif corruption == "prompt":
        value["prompt_ids"][2] = 33
    else:
        value["diagnosis"]["source_union_contains_truth"] = True
    with pytest.raises(ValueError):
        runner._coverage(value)


@pytest.mark.parametrize("bad", [math.inf, math.nan, 0.1, True, 1e100])
def test_nonfinite_or_nonfp32_head_rejected(bad):
    value = raw()
    value["symmetric_relation_logits"][0] = bad
    with pytest.raises(ValueError):
        runner._coverage(value)


def test_cross_seed_query_and_truth_mismatch():
    rows = [runner._coverage(raw())]
    previous = runner._aligned(rows, None, count=1)
    changed = copy.deepcopy(rows)
    changed[0]["raw"]["subject"] = "PKM_CHANGED"
    with pytest.raises(ValueError, match="cross-seed"):
        runner._aligned(changed, previous, count=1)
    changed = copy.deepcopy(rows)
    changed[0]["raw"]["expected_set_ids"] = [1100]
    with pytest.raises(ValueError, match="cross-seed"):
        runner._aligned(changed, previous, count=1)


def test_all_row_coverage_accounting():
    rows = [runner._coverage(raw())["coverage"], runner._coverage(raw(truth=[1025]))["coverage"]]
    assert runner._coverage_totals(rows) == {
        "query_count": 2,
        "focused_queries": 1,
        "fully_covered_queries": 1,
        "true_member_occurrences": 4,
        "covered_member_occurrences": 2,
        "omitted_member_occurrences": 2,
    }


def test_failed_second_diagnostic_preserves_first(tmp_path, monkeypatch):
    original = runner._focused

    def failed(item):
        if item["raw"]["index"] == 1:
            raise RuntimeError("synthetic failure")
        return original(item)

    monkeypatch.setattr(runner, "_focused", failed)
    first, second = runner._coverage(raw()), runner._coverage(raw())
    second["raw"]["index"] = second["coverage"]["index"] = 1
    summary = {
        "complete": False,
        "acceptance": False,
        "input_sha256": {},
        "snapshot_sha256": {},
        "reports": [],
    }
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner._execute(tmp_path / "summary.json", {1729: [first, second]}, summary)
    saved = runner._read(tmp_path / "seed-1729.json")
    assert not saved["complete"] and len(saved["focused_rows"]) == 1
    assert saved["failed_query"]["index"] == 1
    assert not runner._read(tmp_path / "summary.json")["complete"]


def test_bound_input_drift_and_output_refusal(tmp_path):
    path = tmp_path / "summary.json"
    runner._refuse(path)
    runner._write(path, {"complete": False})
    inputs = {}
    runner._bind(path, runner._sha(path), inputs)
    with pytest.raises(ValueError, match="immutable"):
        runner._refuse(path)
    path.write_text("different")
    with pytest.raises(ValueError, match="changed"):
        runner._unchanged(inputs)
