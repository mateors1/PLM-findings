"""Synthetic contracts only; no real intersection screen or oracle diagnosis."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "source_intersections", ROOT / "scripts/evaluate_source_intersections.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def prepared(sets=None, truth=None, logits=None, group="TYPE_dual", index=0):
    sets = sets or [[1025 + i] for i in range(8)]
    truth = truth or [1025, 1026]
    logits = [1.0] * 1025 if logits is None else logits
    slots = runner._slots(sets, logits, intersections=False)
    baseline = runner._select(slots)
    raw = {
        "index": index,
        "subject": "PKM_1024",
        "dimension": "COLOR" if group == "COLOR" else "TYPE",
        "group": group,
        "prompt_ids": [1, 1024, 33 if group == "COLOR" else 32, 34, 5],
        "expected": [f"PKM_{i}" for i in truth],
        "expected_set_ids": truth,
        "symmetric_relation_logits": logits,
        "subject_removed": [False] * 8,
        "composition": {"source_paths": [], "slots": slots, **baseline},
        "metrics": {"wide": runner._metrics(baseline, truth)},
    }
    for rank, ids in enumerate(sets, 1):
        raw["composition"]["source_paths"].append(
            {
                "token_ids": [*raw["prompt_ids"], *ids, 2],
                "targets": [f"PKM_{i}" for i in ids],
                "protocol_valid": True,
                "terminated": True,
                "error": None,
                "decoding": runner._BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            }
        )
    return {
        "raw": raw,
        "source_sets": sets,
        "removed": [False] * 8,
        "slots": slots,
        "baseline": baseline,
        "baseline_metrics": runner._metrics(baseline, truth),
    }


def test_64_order_preserves_all36_and_lexicographic_intersections():
    value = prepared()
    slots = runner._slots(
        value["source_sets"], value["raw"]["symmetric_relation_logits"], intersections=True
    )
    assert len(slots) == 64 and slots[:36] == value["slots"]
    assert [s["source_ranks"] for s in slots[36:]] == [list(p) for p in runner._PAIRS]
    assert all(s["kind"] == "pair_intersection" and s["source_eligible"] for s in slots[36:])


def test_duplicate_union_intersection_identities_and_first_tie():
    sets = [[1025, 1026]] * 8
    slots = runner._slots(sets, [1.0] * 1025, intersections=True)
    assert len(slots) == 64
    assert all(s["set_ids"] == sets[0] for s in slots)
    assert runner._select(slots)["selected_slot"] == 1


def test_union_exactness_requires_three_sources_and_first_witness():
    sets = [[1025], [1026], [1027], *([[1025]] * 5)]
    result = runner._witnesses(sets, [1025, 1026, 1027])
    assert result["unions"] == {"exact_available": True, "min_sources": 3, "witness": [1, 2, 3]}
    assert result["intersections"] == {
        "exact_available": False,
        "min_sources": None,
        "witness": None,
    }


def test_intersection_exactness_requires_three_sources():
    sets = [[1025, 1026, 1027], [1025, 1026, 1028], [1025, 1027, 1028], *([[1025, 1026, 1027]] * 5)]
    result = runner._witnesses(sets, [1025])
    assert result["intersections"] == {
        "exact_available": True,
        "min_sources": 3,
        "witness": [1, 2, 3],
    }
    assert not result["unions"]["exact_available"]


def test_truth_covered_but_shared_extras_prevent_any_pure_exact_set():
    value = prepared([[1025, 1026]] * 8, [1025])
    row = runner._row(value)
    assert row["diagnosis"]["baseline_state"] == "unavailable_with_truth_covered"
    assert not row["availability"]["either_pure_family"]
    assert row["errors"]["sources"][0] == {"false_positive_ids": [1026], "false_negative_ids": []}


def test_missing_true_member_cannot_be_recovered_by_either_operation():
    row = runner._row(prepared(truth=[1100]))
    assert row["diagnosis"]["baseline_state"] == "unavailable_and_truth_missing_from_all_sources"
    assert not row["availability"]["either_pure_family"]
    assert row["errors"]["prediction"]["false_negative_ids"] == [1100]


def test_empty_intersection_zero_beats_negative_nonempty_candidates_and_regresses():
    value = prepared(truth=[1025], logits=[-1.0] * 1025)
    row = runner._row(value)
    assert row["baseline"]["selected_set_ids"] == [1025]
    assert row["prediction"]["selected_slot"] == 37
    assert row["prediction"]["selected_set_ids"] == []
    assert row["metrics"]["prediction"] == {
        "exact": False,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "set_size": 0,
        "success": True,
    }
    assert row["lost_exact"] and row["selection_changed"]


def test_zero_score_intersection_cannot_displace_old_zero_tie():
    value = prepared(logits=[0.0] * 1025)
    row = runner._row(value)
    assert row["prediction"]["selected_slot"] == 1
    assert not row["selection_changed"]


def test_truth_free_predictor_and_available_miss_state():
    first = runner._row(prepared())
    second = runner._row(prepared(truth=[1027, 1028]))
    assert first["prediction"] == second["prediction"] and first["slots"] == second["slots"]
    assert first["diagnosis"]["baseline_state"] == "selected_exact"
    assert second["diagnosis"]["baseline_state"] == "available_exact_but_missed"
    assert second["availability"]["baseline"]


def test_existing_exact_miss_is_not_repaired_by_appended_equal_scoring_set():
    row = runner._row(prepared(truth=[1027, 1028]))
    assert row["availability"]["policy"]
    assert not row["metrics"]["prediction"]["exact"]


def test_subject_removal_keeps_raw_path_and_uniqueness_precedes_removal():
    value = prepared()
    raw = value["raw"]
    path = raw["composition"]["source_paths"][0]
    path["token_ids"].insert(-1, 1024)
    path["targets"].append("PKM_1024")
    raw["subject_removed"][0] = True
    sets, removed = runner._sources(raw, {})
    assert sets[0] == [1025] and removed[0]
    assert path["token_ids"][-2] == 1024
    path["token_ids"].insert(-1, 1024)
    path["targets"].append("PKM_1024")
    with pytest.raises(ValueError, match="uniqueness"):
        runner._sources(raw, {})


@pytest.mark.parametrize(
    "bad", ["control", "unfinished", "repeated_first", "subject_truth", "group", "rank", "names"]
)
def test_invalid_saved_sources_fail_closed(bad):
    raw = prepared()["raw"]
    path = raw["composition"]["source_paths"][0]
    if bad == "control":
        path["token_ids"][5] = 32
    elif bad == "unfinished":
        path["terminated"] = False
    elif bad == "repeated_first":
        other = raw["composition"]["source_paths"][1]
        other["token_ids"][5] = 1025
        other["targets"] = ["PKM_1025"]
    elif bad == "subject_truth":
        raw["expected_set_ids"] = [1024]
    elif bad == "group":
        raw["group"] = "COLOR"
    elif bad == "rank":
        path["decoding"] += "wrong"
    else:
        path["targets"] = ["PKM_1024"]
    with pytest.raises(ValueError):
        runner._sources(raw, {})


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.1, True, 1e100])
def test_invalid_head_rejected(bad):
    values = [0.0] * 1025
    values[0] = bad
    with pytest.raises(ValueError):
        runner._logits(values)


def test_canonical_fsum_and_error_ids():
    logits = [0.0] * 1025
    logits[1:4] = [2.0**60, 1.0, -(2.0**60)]
    slots = runner._slots([[1025, 1026, 1027]] * 8, logits, intersections=True)
    assert all(s["score"] == 1.0 for s in slots)
    assert runner._errors([1025, 1027], [1026, 1027]) == {
        "false_positive_ids": [1025],
        "false_negative_ids": [1026],
    }


def test_aggregate_empty_intersections_and_partition_cover_every_row():
    rows = [
        runner._row(prepared(truth=[1025], logits=[-1.0] * 1025)),
        runner._row(prepared(truth=[1100])),
        runner._row(prepared(truth=[1027, 1028])),
        runner._row(prepared([[1025, 1026]] * 8, [1025])),
    ]
    total = runner._totals(rows)
    assert sum(total["baseline_state_counts"].values()) == 4
    assert total["prediction"]["empty_selections"] == 1
    assert total["prediction"]["intersection_selections"] == 1
    assert total["losses"] == 1


@pytest.mark.parametrize("failure", ["exact", "f1", "pooled", "group", "incomplete"])
def test_fixed_gate_rejects_each_failure(failure):
    totals = runner._aggregate([runner._row(prepared(group=g)) for g in runner._GROUPS])
    totals["overall"]["prediction"]["exact_count"] = 604
    reports = [
        {
            "complete": True,
            "query_count": 222,
            "baseline_reproduction_exact": True,
            "overall": copy.deepcopy(totals["overall"]),
        }
        for _ in range(3)
    ]
    assert runner._gate(reports, totals)["quality_passed"]
    if failure == "exact":
        reports[0]["overall"]["prediction"]["exact_count"] = 0
    elif failure == "f1":
        reports[0]["overall"]["prediction"]["f1"] = 0
    elif failure == "pooled":
        totals["overall"]["prediction"]["exact_count"] = 603
    elif failure == "group":
        totals["groups"]["COLOR"]["prediction"]["exact_count"] = 0
    else:
        reports[0]["complete"] = False
    assert not runner._gate(reports, totals)["quality_passed"]


def test_failed_second_query_preserves_first_row_and_error(tmp_path, monkeypatch):
    original = runner._row

    def injected(item):
        if item["raw"]["index"] == 1:
            raise RuntimeError("injected query failure")
        return original(item)

    monkeypatch.setattr(runner, "_row", injected)
    summary = {
        "complete": False,
        "acceptance": False,
        "input_sha256": {},
        "snapshot_sha256": {},
        "reports": [],
    }
    with pytest.raises(RuntimeError, match="injected query failure"):
        runner._execute(tmp_path / "summary.json", {1729: [prepared(), prepared(index=1)]}, summary)
    report = runner._read(tmp_path / "seed-1729.json")
    assert report["query_count"] == 1 and len(report["responses"]) == 1
    assert not report["complete"] and report["failed_query"]["index"] == 1
    assert not runner._read(tmp_path / "summary.json")["complete"]
    assert summary["reports"][0]["sha256"] == runner._sha(tmp_path / "seed-1729.json")


def test_first_query_failure_writes_empty_partial_report(tmp_path, monkeypatch):
    def injected(item):
        raise ValueError("first query")

    monkeypatch.setattr(runner, "_row", injected)
    summary = {"complete": False, "input_sha256": {}, "snapshot_sha256": {}, "reports": []}
    with pytest.raises(ValueError, match="first query"):
        runner._execute(tmp_path / "summary.json", {1729: [prepared()]}, summary)
    assert runner._read(tmp_path / "seed-1729.json")["responses"] == []


def test_immutable_output_and_input_drift(tmp_path):
    output = tmp_path / "summary.json"
    (tmp_path / "independent-audit.py").write_text("# allowed")
    runner._refuse(output)
    runner._write(output, {"complete": False})
    with pytest.raises(ValueError, match="immutable"):
        runner._refuse(output)
    inputs = {}
    runner._bind(output, runner._sha(output), inputs)
    output.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        runner._unchanged(inputs)


@pytest.mark.parametrize("field", ["score", "choice", "metrics"])
def test_baseline_reference_corruption_rejected_before_new_policy(field):
    raw = prepared()["raw"]
    if field == "score":
        raw["composition"]["slots"][0]["score"] += 0.00000001
    elif field == "choice":
        raw["composition"]["selected_slot"] = 2
    else:
        raw["metrics"]["wide"]["f1"] = 0.0
    with pytest.raises(ValueError, match="exact baseline"):
        runner._prepare({1729: {"responses": [raw]}})


def test_complete_synthetic_campaign_bound_and_no_promotion(tmp_path):
    sample = [
        prepared(truth=[1100], group="COLOR"),
        prepared(truth=[1027, 1028], group="TYPE_single", index=1),
        prepared(group="TYPE_dual", index=2),
    ]
    summary = {
        "complete": False,
        "acceptance": False,
        "input_sha256": {},
        "snapshot_sha256": {},
        "reports": [],
    }
    runner._execute(tmp_path / "summary.json", dict.fromkeys(runner._SEEDS, sample), summary)
    assert summary["complete"] and not summary["acceptance"]
    assert summary["derived_bound"]["missing_union_count"] == 3
    assert summary["derived_bound"]["existing_available_miss_count"] == 3
    assert summary["derived_bound"]["selected_exact_upper_bound"] == 3
    assert len(summary["reports"]) == 3


def test_postwork_input_drift_persists_failed_summary(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_text("before")
    original = runner._row

    def changed(item):
        result = original(item)
        path.write_text("after")
        return result

    monkeypatch.setattr(runner, "_row", changed)
    summary = {
        "complete": False,
        "acceptance": False,
        "input_sha256": {str(path): runner._sha(path)},
        "snapshot_sha256": {},
        "reports": [],
    }
    samples = [prepared(group=g, index=i) for i, g in enumerate(runner._GROUPS)]
    output = tmp_path / "summary.json"
    with pytest.raises(ValueError, match="final identity"):
        runner._execute(output, dict.fromkeys(runner._SEEDS, samples), summary)
    saved = runner._read(output)
    assert not saved["complete"] and not saved["final_identity_check"]
    assert "input changed" in saved["identity_error"]
