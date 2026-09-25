"""Synthetic saved-pool audits without loading any campaign measurements."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "runs/learning/candidate-score-decomposition-v1/independent-audit.py"
)
spec = importlib.util.spec_from_file_location("candidate_score_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def make_row(products=None, truth=None, head=None):
    products = products or [[1025], [1026], [1027], [1028]]
    truth = truth or [1025, 1026]
    head = head or [-1.0, 2.0, 3.0] + [-1.0] * 1022
    prompt = [1, 1024, 32, 34, 5]
    raw = [
        {
            "token_ids": prompt + ids + [2],
            "targets": [f"PKM_{i}" for i in ids],
            "terminated": True,
            "protocol_valid": True,
            "error": None,
            "decoding": audit.BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
        }
        for rank, ids in enumerate(products, 1)
    ]
    row = {
        "index": 0,
        "subject": "PKM_1024",
        "dimension": "TYPE",
        "group": "TYPE_dual",
        "prompt_ids": prompt,
        "expected_set_ids": truth,
        "symmetric_relation_logits": head,
        "composition": {"source_paths": raw},
    }
    pool, _ = audit.pool(row)
    chosen = audit.choose(pool, head)
    row["composition"] = {"source_paths": raw, **chosen, "policy": audit.POLICY}
    row["selected_metrics"] = audit.set_metrics(
        chosen["selected_set_ids"], truth, chosen["selected_source_eligible"]
    )
    return row


def test_canonical_exact_rational_sum_matches_fsum_with_cancellation():
    values = [0.0] * 1025
    values[1:4] = [2.0**80, 1.0, -(2.0**80)]
    assert audit.canonical_sum([1025, 1026, 1027], values) == math.fsum(values[1:4]) == 1.0
    with pytest.raises(ValueError, match="canonical"):
        audit.canonical_sum([1026, 1025], values)
    with pytest.raises(ValueError):
        audit.canonical_sum([1025, 1025], values)


def test_duplicate_slots_tie_order_and_subject_postprocess():
    row = make_row([[1024, 1025], [1026], [1025], [1027]])
    rebuilt, _ = audit.rebuild_row(row, copy.deepcopy(row))
    assert rebuilt["pools"]["control"]["slots"][0]["set_ids"] == [1025]
    assert (
        rebuilt["pools"]["control"]["slots"][4]["set_ids"]
        == rebuilt["pools"]["control"]["slots"][7]["set_ids"]
    )
    assert rebuilt["cells"]["cc"]["selected_slot"] == 5
    assert rebuilt["availability"]["control"]["source_subject_removed"] == [
        True,
        False,
        False,
        False,
    ]


def test_repeated_subject_is_ineligible_before_postprocessing():
    row = make_row([[1024, 1024, 1025], [1026], [1025], [1027]])
    slots, _ = audit.pool(row)
    assert slots[0]["set_ids"] == [1025]
    assert slots[0]["source_eligible"] is False
    assert audit.choose(slots, row["symmetric_relation_logits"])["selected_slot"] == 8


def test_higher_union_containment_does_not_establish_exact_availability():
    row = make_row(truth=[1025, 1026, 1027])
    rebuilt, _ = audit.rebuild_row(row, row)
    status = rebuilt["availability"]["control"]
    assert status["source_union_contains_truth"] is True
    assert status["all_exact_available"] is False
    assert status["state"] == "exact_unavailable"


def test_available_exact_miss_and_selected_exact_are_distinct():
    row = make_row([[1025, 1026], [1027], [1028], [1029]])
    row["symmetric_relation_logits"][3] = 100.0
    slots, _ = audit.pool(row)
    chosen = audit.choose(slots, row["symmetric_relation_logits"])
    status = audit.availability(slots, chosen, row["expected_set_ids"], [])
    assert status["original_exact_available"] is True
    assert status["state"] == "exact_available_but_missed"
    assert status["recoverable_exact_miss"] == 1
    row["symmetric_relation_logits"][3] = -100.0
    chosen = audit.choose(slots, row["symmetric_relation_logits"])
    status = audit.availability(slots, chosen, row["expected_set_ids"], [])
    assert status["state"] == "selected_exact"
    assert status["recoverable_exact_miss"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.1, True])
def test_nonfinite_or_non_fp32_logits_rejected(bad):
    values = [0.0] * 1025
    values[1] = bad
    with pytest.raises(ValueError):
        audit.vector(values)


@pytest.mark.parametrize(
    "change", ["score", "selection", "set", "metrics", "query", "subject_truth", "raw_flags"]
)
def test_diagonal_corruption_and_pair_misalignment_rejected(change):
    control = make_row()
    treatment = copy.deepcopy(control)
    if change == "score":
        control["composition"]["slots"][0]["score"] += 0.001
    elif change == "selection":
        control["composition"]["selected_slot"] = 10
    elif change == "set":
        control["composition"]["slots"][0]["set_ids"] = [1026]
    elif change == "metrics":
        control["selected_metrics"]["f1"] = 0.5
    elif change == "query":
        treatment["subject"] = "PKM_other"
    elif change == "subject_truth":
        control["expected_set_ids"] = treatment["expected_set_ids"] = [1024, 1025]
    else:
        control["composition"]["source_paths"][0]["terminated"] = False
    with pytest.raises(ValueError):
        audit.rebuild_row(control, treatment)


def test_offdiagonal_uses_other_head_without_modifying_pool_or_labels():
    control = make_row()
    treatment_head = [-1.0] * 1025
    treatment_head[3:5] = [5.0, 6.0]
    treatment = make_row(head=treatment_head)
    row, _ = audit.rebuild_row(control, treatment)
    assert row["cells"]["cc"]["selected_set_ids"] == [1025, 1026]
    assert row["cells"]["ct"]["selected_set_ids"] == [1027, 1028]
    assert row["cells"]["ct"]["pool_arm"] == "control"
    assert row["cells"]["ct"]["score_arm"] == "treatment"
    assert row["pools"]["control"] == row["pools"]["treatment"]


def test_incomplete_source_membership_is_preserved_and_excluded():
    row = make_row()
    for rank, raw in enumerate(row["composition"]["source_paths"], 1):
        raw["token_ids"] = row["prompt_ids"] + [1024] * 507
        raw["targets"] = ["PKM_1024"] * 507
        raw.update(terminated=False, protocol_valid=False, error="response must terminate with EOS")
        ids, eligible, _ = audit.source_set(raw, row["prompt_ids"], rank)
        assert ids == [1024]
        assert eligible is False
    slots, _ = audit.pool(row)
    chosen = audit.choose(slots, row["symmetric_relation_logits"])
    assert chosen["selected_slot"] == 1
    assert chosen["fallback_no_valid_source"] is True
    assert audit.set_metrics(chosen["selected_set_ids"], [1025], False)["set_size"] == 0


def test_ordered_effects_interaction_and_macro_rounding():
    result = audit.decomposition({"cc": 10, "ct": 12, "tc": 13, "tt": 18})
    assert result == {
        "diagonal_change": 8,
        "pool_then_score": {"pool_effect": 3, "score_effect": 5, "sum": 8},
        "score_then_pool": {"score_effect": 2, "pool_effect": 6, "sum": 8},
        "interaction": 3,
    }
    values = {"cc": 0.1, "ct": 0.2, "tc": 0.3, "tt": 0.6}
    result = audit.decomposition(values)
    assert result["interaction"] == 0.6 - 0.3 - 0.2 + 0.1
    assert result["pool_then_score"]["sum"] == (0.3 - 0.1) + (0.6 - 0.3)


def test_three_state_table_uses_queries_and_preserves_all_nine_cells():
    exact_row, _ = audit.rebuild_row(make_row(), make_row())
    unavailable, _ = audit.rebuild_row(
        make_row(truth=[1025, 1026, 1027]), make_row(truth=[1025, 1026, 1027])
    )
    old_head = [-1.0, 2.0, 3.0, 100.0] + [-1.0] * 1021
    changed, _ = audit.rebuild_row(
        make_row([[1025, 1026], [1027], [1028], [1029]], head=old_head),
        make_row([[1025, 1026], [1027], [1028], [1029]]),
    )
    result = audit.summarize([exact_row, unavailable, changed])
    assert result["query_count"] == 3
    assert result["transitions"]["selected_exact"]["selected_exact"] == 1
    assert result["transitions"]["exact_available_but_missed"]["selected_exact"] == 1
    assert result["transitions"]["exact_unavailable"]["exact_unavailable"] == 1
    assert sum(sum(row.values()) for row in result["transitions"].values()) == 3
    assert all(len(row) == 3 for row in result["transitions"].values())
    assert result["cell_metrics"]["cc"]["exact_set_count"] == 1
    assert result["cell_metrics"]["tt"]["exact_set_count"] == 2
    assert result["arm_availability"]["control"]["recoverable_exact_miss_count"] == 1
    assert (
        result["cell_metrics"]["tt"]["f1"]
        == math.fsum(
            row["cells"]["tt"]["metrics"]["f1"] for row in (exact_row, unavailable, changed)
        )
        / 3
    )


def test_identifier_mapping_is_bijective_across_saved_sources():
    forward, reverse = {}, {}
    audit.checked_name_map([(1024, "PKM_A"), (1025, "PKM_B"), (1024, "PKM_A")], forward, reverse)
    with pytest.raises(ValueError, match="ID-to-name"):
        audit.checked_name_map([(1024, "PKM_C")], forward, reverse)
    with pytest.raises(ValueError, match="name-to-ID"):
        audit.checked_name_map([(1026, "PKM_A")], forward, reverse)


def test_selector_has_no_teacher_dependency():
    row = make_row()
    slots, _ = audit.pool(row)
    choice = audit.choose(slots, row["symmetric_relation_logits"])
    first = audit.set_metrics(choice["selected_set_ids"], [1025, 1026])
    second = audit.set_metrics(choice["selected_set_ids"], [1027])
    assert first["exact"] is True and second["exact"] is False
    assert audit.choose(slots, row["symmetric_relation_logits"]) == choice
