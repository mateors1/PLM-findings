"""Synthetic saved-score decomposition contracts; no real diagnostic measurement."""

from __future__ import annotations

import copy
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "candidate_scores", ROOT / "scripts/diagnose_candidate_scores.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def pool(sets, eligible=None):
    eligible = [True] * 4 if eligible is None else eligible
    return {
        "slots": [
            {
                "slot": i,
                "kind": "original" if i <= 4 else "pair_composition",
                "source_ranks": list(ranks),
                "set_ids": sorted(set().union(*(set(sets[r - 1]) for r in ranks))),
                "source_eligible": all(eligible[r - 1] for r in ranks),
            }
            for i, ranks in enumerate(runner._ORDER, 1)
        ],
        "source_subject_removed": [False] * 4,
    }


def vector(values):
    result = [0.0] * 1025
    for key, value in values.items():
        result[key - 1024] = value
    return result


def row(paths=None):
    paths = [[1025], [1026], [1025], [1026]] if paths is None else paths
    scores = vector({1025: 2.0, 1026: 1.0})
    original = pool(
        [set(ids) - {1024} for ids in paths], [len(ids) == len(set(ids)) for ids in paths]
    )
    cell = runner._cell(original, scores, [1025, 1026], "control", "control")
    composition = {
        k: v
        for k, v in cell.items()
        if k.startswith("selected_") or k == "fallback_no_valid_source"
    }
    composition["slots"] = [
        {**slot, "score": score}
        for slot, score in zip(original["slots"], cell["slot_scores"], strict=True)
    ]
    composition["policy"] = runner._POLICY
    prompt = [1, 1024, 32, 34, 5]
    composition["source_paths"] = [
        {
            "token_ids": [*prompt, *ids, 2],
            "targets": [f"PKM_{i}" for i in ids],
            "terminated": True,
            "protocol_valid": True,
            "error": None,
            "decoding": runner._BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
        }
        for rank, ids in enumerate(paths, 1)
    ]
    return {
        "index": 0,
        "subject": "PKM_1024",
        "dimension": "TYPE",
        "group": "TYPE_single",
        "prompt_ids": prompt,
        "expected_set_ids": [1025, 1026],
        "composition": composition,
        "symmetric_relation_logits": scores,
        "selected_metrics": cell["metrics"],
        "batch_index": 0,
        "batch_shape": [8, 5],
    }


def test_diagonal_reconstructs_canonical_paths_unions_and_tie_order():
    evidence = row()
    rebuilt = runner._pool(evidence, {})
    result = runner._diagonal(evidence, rebuilt, "control")
    assert len(rebuilt["slots"]) == len(result["slot_scores"]) == 10
    assert rebuilt["slots"][4]["set_ids"] == rebuilt["slots"][6]["set_ids"]
    assert result["selected_slot"] == 5
    assert result["metrics"]["exact"] is True


def test_available_but_missed_differs_from_unavailable():
    available = pool([[1025], [1026], [1027], [1027]])
    scores = vector({1025: 1.0, 1026: 1.0, 1027: 10.0})
    cell = runner._cell(available, scores, [1025, 1026], "control", "control")
    state = runner._availability(available, cell, [1025, 1026])
    assert state["state"] == "exact_available_but_missed"
    assert state["recoverable_exact_miss"] is True and state["all_exact_available"] is True
    unavailable = pool([[1025], [1027], [1027], [1027]])
    cell = runner._cell(unavailable, scores, [1025, 1026], "control", "control")
    state = runner._availability(unavailable, cell, [1025, 1026])
    assert state["state"] == "exact_unavailable" and not state["source_union_contains_truth"]


def test_full_union_containment_does_not_imply_exact_pair_availability():
    candidates = pool([[1025], [1026], [1027], [1025]])
    truth = [1025, 1026, 1027]
    selected = runner._cell(candidates, vector({}), truth, "control", "control")
    state = runner._availability(candidates, selected, truth)
    assert state["source_union_contains_truth"] is True
    assert state["all_exact_available"] is False
    assert state["state"] == "exact_unavailable"


def test_ineligible_slots_cannot_win_and_fallback_has_no_metric_credit():
    candidates = pool([[1025], [1026], [1027], [1028]], [False, True, True, True])
    values = vector({1025: 100.0, 1026: 1.0})
    selected = runner._cell(candidates, values, [1025], "control", "treatment")
    assert selected["selected_slot"] == 2 and not selected["metrics"]["exact"]
    empty = pool([[1025]] * 4, [False] * 4)
    selected = runner._cell(empty, values, [1025], "control", "control")
    assert selected["fallback_no_valid_source"] is True
    assert selected["metrics"] == runner._metrics([], [1025], False)


def test_raw_subject_is_allowed_and_removal_is_explicit():
    evidence = row([[1024, 1025], [1026], [1025], [1026]])
    rebuilt = runner._pool(evidence, {})
    assert rebuilt["source_subject_removed"] == [True, False, False, False]
    assert evidence["composition"]["source_paths"][0]["token_ids"][5] == 1024
    assert rebuilt["slots"][0]["set_ids"] == [1025]
    runner._diagonal(evidence, rebuilt, "control")


def test_uniqueness_is_checked_before_subject_removal():
    evidence = row([[1024, 1024, 1025], [1026], [1025], [1026]])
    rebuilt = runner._pool(evidence, {})
    assert rebuilt["source_subject_removed"][0] is True
    assert rebuilt["slots"][0]["source_eligible"] is False


@pytest.mark.parametrize("corruption", ["path", "union", "name", "score", "selected", "truth"])
def test_corrupt_saved_evidence_is_rejected(corruption):
    evidence = row()
    if corruption == "path":
        evidence["composition"]["source_paths"][0]["token_ids"][2] = 33
    elif corruption == "union":
        evidence["composition"]["slots"][4]["set_ids"] = [1025]
    elif corruption == "name":
        evidence["composition"]["source_paths"][2]["targets"][0] = "PKM_OTHER"
    elif corruption == "score":
        evidence["composition"]["slots"][4]["score"] = math.nextafter(3.0, math.inf)
    elif corruption == "selected":
        evidence["composition"]["selected_slot"] = 7
    else:
        evidence["expected_set_ids"] = [1024, 1025]
    with pytest.raises(ValueError):
        rebuilt = runner._pool(evidence, {})
        runner._diagonal(evidence, rebuilt, "control")


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.1, True, 1e50])
def test_invalid_fp32_vectors_fail(bad):
    values = vector({1025: bad})
    with pytest.raises(ValueError):
        runner._fp32(values)


def test_fsum_preserves_small_member_under_cancellation():
    candidates = pool([[1025, 1026, 1027]] * 4)
    values = vector({1025: 2.0**60, 1026: 1.0, 1027: -(2.0**60)})
    runner._fp32(values)
    result = runner._cell(candidates, values, [1025], "control", "control")
    assert result["slot_scores"] == [1.0] * 10


def test_query_alignment_prevents_cross_query_score_swap():
    a, b = row(), copy.deepcopy(row())
    b["subject"] = "PKM_OTHER"
    with pytest.raises(ValueError, match="alignment"):
        runner._align(a, b)


def test_ordered_decompositions_and_interaction():
    result = runner._decomposition({"cc": 4, "ct": 7, "tc": 2, "tt": 8})
    assert result == {
        "diagonal_change": 4,
        "pool_then_score": {"pool_effect": -2, "score_effect": 6, "sum": 4},
        "score_then_pool": {"score_effect": 3, "pool_effect": 1, "sum": 4},
        "interaction": 3,
    }


def test_three_by_three_transition_and_availability_counts():
    rows = []
    for control in runner._STATES:
        for treatment in runner._STATES:
            availability = {}
            cells = {}
            for arm, state in (("control", control), ("treatment", treatment)):
                exact = state == "selected_exact"
                availability[arm] = {
                    "state": state,
                    "original_exact_available": exact,
                    "all_exact_available": state != "exact_unavailable",
                    "selected_exact": exact,
                    "recoverable_exact_miss": state == "exact_available_but_missed",
                    "source_union_contains_truth": True,
                }
                cell = {
                    "metrics": runner._metrics([1025] if exact else [1026], [1025]),
                    "selected_source_eligible": True,
                    "fallback_no_valid_source": False,
                }
                cells[arm[0] * 2] = cell
            cells["ct"], cells["tc"] = cells["cc"], cells["tt"]
            rows.append({"availability": availability, "cells": cells})
    result = runner._summarize(rows)
    assert all(count == 1 for row in result["transitions"].values() for count in row.values())
    assert result["query_count"] == 9
    assert result["arm_availability"]["control"]["all_exact_available_count"] == 6
    assert result["arm_availability"]["treatment"]["recoverable_exact_miss_count"] == 3
    assert result["cell_metrics"]["cc"]["exact_set_count"] == 3


def test_input_and_output_identities_are_immutable(tmp_path):
    path = tmp_path / "input.json"
    runner._write(path, {"complete": True})
    digest = runner._sha(path)
    runner._read_bound(path, digest, {})
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        runner._read_bound(path, digest, {})
    with pytest.raises(FileExistsError):
        runner._write(path, {})
    with pytest.raises(ValueError, match="immutable"):
        runner._refuse(path)


def test_module_is_stdlib_and_torch_free():
    code = (
        "import runpy,sys; runpy.run_path('scripts/diagnose_candidate_scores.py',"
        "run_name='check'); assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, check=True)
