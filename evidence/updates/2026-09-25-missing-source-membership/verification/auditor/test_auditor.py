"""Synthetic missing-membership audit tests; no real diagnostic measurements."""

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "runs/learning/missing-source-membership-v1/independent-audit.py"
)
spec = importlib.util.spec_from_file_location("missing_membership_audit", SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def row(truth=None):
    prompt = [1, 1024, 32, 34, 5]
    return {
        "index": 0,
        "subject": "PKM_1024",
        "dimension": "TYPE",
        "group": "TYPE_dual",
        "prompt_ids": prompt,
        "expected_set_ids": truth or [1025, 1040, 1041],
        "source_paths": [
            {
                "token_ids": [*prompt, 1025 + i, 2],
                "targets": [f"PKM_{1025 + i}"],
                "terminated": True,
                "protocol_valid": True,
                "error": None,
                "decoding": a.BASE + (f"+first-rank{i + 1}-v1" if i else ""),
            }
            for i in range(8)
        ],
        "source_set_ids": [[1025 + i] for i in range(8)],
        "source_subject_removed": [False] * 8,
        "symmetric_relation_logits": [-1.0] * 1025,
        "diagnosis": {
            "source_union_contains_truth": False,
            "baseline_state": "unavailable_and_truth_missing_from_all_sources",
        },
    }


def focus(raw):
    union, missing, covered = a.membership(raw, 0, {})
    return a.reconstruct(raw, union, missing, covered)


def test_all_source_coverage_selected_before_scores():
    raw = row()
    union, missing, covered = a.membership(raw, 0, {})
    assert union == list(range(1025, 1033)) and missing == [1040, 1041] and covered == [1025]
    assert a.coverage_row(raw, missing, covered) == {
        "index": 0,
        "subject": "PKM_1024",
        "dimension": "TYPE",
        "group": "TYPE_dual",
        "true_member_count": 3,
        "covered_member_count": 1,
        "missing_member_count": 2,
        "missing_ids": [1040, 1041],
    }
    raw["symmetric_relation_logits"] = [2.0] * 1025
    assert a.membership(raw, 0, {}) == (union, missing, covered)


def test_rank_ties_false_products_and_subject_exclusion():
    raw = row()
    head = raw["symmetric_relation_logits"]
    head[0] = 100.0
    head[2] = 2.0
    head[1040 - 1024] = head[1041 - 1024] = 0.0
    result = focus(raw)
    assert result["omitted_members"] == [
        {
            "token_id": 1040,
            "logit": 0.0,
            "rank": 2,
            "false_strictly_higher": 1,
            "false_tied": 0,
            "oracle_top_k": True,
        },
        {
            "token_id": 1041,
            "logit": 0.0,
            "rank": 3,
            "false_strictly_higher": 1,
            "false_tied": 0,
            "oracle_top_k": True,
        },
    ]
    assert result["positive_omitted_category"] == "none"
    assert result["omitted_summary"]["zero_count"] == 2
    assert result["omitted_summary"]["rank"] == {"min": 2, "median": 2.5, "max": 3}


def test_false_ties_count_all_not_just_tie_break_predecessors():
    head = [-1.0] * 1025
    for token in (1026, 1040, 1050):
        head[token - 1024] = 0.0
    value = a.member(1040, {1040}, 1024, head)
    assert value["rank"] == 2
    assert value["false_tied"] == 2 and value["false_strictly_higher"] == 0
    assert not value["oracle_top_k"]


@pytest.mark.parametrize(
    "scores,category,counts",
    [
        ((2.0, 1.0), "all", (2, 0, 0)),
        ((2.0, 0.0), "some", (1, 1, 0)),
        ((0.0, -1.0), "none", (0, 1, 1)),
    ],
)
def test_omitted_sign_categories(scores, category, counts):
    raw = row()
    raw["symmetric_relation_logits"][16:18] = scores
    result = focus(raw)
    assert result["positive_omitted_category"] == category
    assert (
        tuple(
            result["omitted_summary"][k] for k in ("positive_count", "zero_count", "negative_count")
        )
        == counts
    )


def test_empty_covered_null_statistics_and_denominator():
    first = row([1040])
    first["symmetric_relation_logits"][16] = 1.0
    second = row([1025, 1040])
    second["symmetric_relation_logits"][1] = 2.0
    result = focus(first)
    assert result["covered_members"] == []
    assert result["covered_summary"] == {
        "count": 0,
        "positive_count": 0,
        "zero_count": 0,
        "negative_count": 0,
        "logit": None,
        "rank": None,
    }
    total = a.aggregate([result, focus(second)])
    assert total["covered"]["member_occurrences"] == total["covered"]["nonempty_query_count"] == 1
    assert total["covered"]["mean_query_positive_fraction"] == 1
    assert a.aggregate([result])["covered"]["mean_query_positive_fraction"] is None
    assert a.aggregate([result])["covered"]["positive_member_fraction"] is None


def test_member_weighted_and_query_weighted_fractions_differ():
    first = row([1040])
    first["symmetric_relation_logits"][16] = 1.0
    second = row([1040, 1041, 1042])
    total = a.aggregate([focus(first), focus(second)])
    assert total["omitted"]["member_occurrences"] == 4
    assert total["omitted"]["positive_member_fraction"] == 0.25
    assert total["omitted"]["mean_query_positive_fraction"] == 0.5
    assert total["omitted"]["nonempty_query_count"] == 2


def test_exact_coverage_not_focused():
    raw = row([1025, 1026])
    raw["diagnosis"] = {"source_union_contains_truth": True, "baseline_state": "selected_exact"}
    union, missing, covered = a.membership(raw, 0, {})
    assert missing == [] and covered == [1025, 1026]
    with pytest.raises(ValueError, match="focus"):
        a.reconstruct(raw, union, missing, covered)


def test_subject_removal_matches_saved_sets_and_head_subject_does_not_change_ranks():
    raw = row()
    path = raw["source_paths"][0]
    path["token_ids"].insert(-1, 1024)
    path["targets"].append("PKM_1024")
    raw["source_subject_removed"][0] = True
    before = copy.deepcopy(raw)
    result = focus(raw)
    assert raw == before and 1024 not in result["source_union_ids"]
    raw["symmetric_relation_logits"][0] = 1e10
    assert focus(raw)["omitted_members"] == result["omitted_members"]


@pytest.mark.parametrize(
    "mutation",
    [
        "source_set",
        "repeat",
        "early_stop",
        "saved_covered",
        "subject_truth",
        "prompt",
        "target_name",
        "group",
        "dimension",
    ],
)
def test_membership_corruption_rejected(mutation):
    raw = row()
    names = {1025: "PKM_1025"}
    if mutation == "source_set":
        raw["source_set_ids"][0].append(1040)
    if mutation == "repeat":
        raw["source_paths"][0]["token_ids"].insert(-1, 1025)
        raw["source_paths"][0]["targets"].append("PKM_1025")
    if mutation == "early_stop":
        raw["source_paths"][0]["token_ids"][-1] = 1026
    if mutation == "saved_covered":
        raw["diagnosis"]["source_union_contains_truth"] = True
    if mutation == "subject_truth":
        raw["expected_set_ids"] = [1024]
    if mutation == "prompt":
        raw["prompt_ids"][3] = 64
    if mutation == "target_name":
        raw["source_paths"][0]["targets"][0] = "PKM_wrong"
    if mutation == "group":
        raw["group"] = "COLOR"
    if mutation == "dimension":
        raw["dimension"] = "BIOME"
    with pytest.raises(ValueError):
        a.membership(raw, 0, names)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.1, True, 1])
def test_nonfinite_or_nonfp32_rejected(bad):
    raw = row()
    raw["symmetric_relation_logits"][20] = bad
    with pytest.raises(ValueError):
        a.membership(raw, 0, {})


def test_median_even_and_odd_and_empty():
    assert a.extent([]) is None
    assert a.extent([1, 3, 2]) == {"min": 1, "median": 2, "max": 3}
    assert a.extent([-2.0, 1.0, 4.0, 0.0]) == {"min": -2.0, "median": 0.5, "max": 4.0}


def test_cross_seed_order_teacher_and_duplicates():
    first = row()
    second = row()
    second["subject"] = "PKM_1027"
    second["prompt_ids"][1] = 1027
    previous = a.aligned([first, second], None, count=2)
    assert a.aligned([copy.deepcopy(first), copy.deepcopy(second)], previous, count=2) == previous
    with pytest.raises(ValueError, match="cross-seed"):
        a.aligned([second, first], previous, count=2)
    altered = copy.deepcopy(second)
    altered["expected_set_ids"] = [1040]
    with pytest.raises(ValueError, match="cross-seed"):
        a.aligned([first, altered], previous, count=2)
    with pytest.raises(ValueError, match="duplicate"):
        a.aligned([first, first], None, count=2)


def test_unique_focused_identities_separate_from_observations():
    focused = focus(row())
    total = a.aggregate([focused, copy.deepcopy(focused)])
    assert total["focused_queries"] == 2 and total["distinct_focused_queries"] == 1
    second = copy.deepcopy(focused)
    second["subject"] = "PKM_another"
    assert a.aggregate([focused, second])["distinct_focused_queries"] == 2


def report_fixture():
    raw = row()
    memberships = [a.membership(raw, 0, {})]
    coverage = [a.coverage_row(raw, memberships[0][1], memberships[0][2])]
    focused = [a.reconstruct(raw, *memberships[0])]
    report = {
        "seed": 1729,
        "complete": True,
        "coverage": coverage,
        "focused_rows": focused,
        "overall": a.aggregate(focused),
        "coverage_summary": a.coverage_totals(coverage),
    }
    return report, {"seed": 1729, "responses": [raw]}, memberships


def test_full_saved_report_reconstruction_and_accounting():
    report, reference, memberships = report_fixture()
    coverage, focused = a.audit_seed(report, reference, memberships)
    assert a.coverage_totals(coverage) == {
        "query_count": 1,
        "focused_queries": 1,
        "fully_covered_queries": 0,
        "true_member_occurrences": 3,
        "covered_member_occurrences": 1,
        "omitted_member_occurrences": 2,
    }
    assert focused == report["focused_rows"]


@pytest.mark.parametrize(
    "mutation", ["rank", "false_higher", "tie", "oracle", "fraction", "coverage", "omission"]
)
def test_saved_report_mutations_rejected(mutation):
    report, reference, memberships = report_fixture()
    member = report["focused_rows"][0]["omitted_members"][0]
    if mutation == "rank":
        member["rank"] += 1
    if mutation == "false_higher":
        member["false_strictly_higher"] += 1
    if mutation == "tie":
        member["false_tied"] += 1
    if mutation == "oracle":
        member["oracle_top_k"] = not member["oracle_top_k"]
    if mutation == "fraction":
        report["overall"]["omitted"]["positive_member_fraction"] = 1.0
    if mutation == "coverage":
        report["coverage"][0]["missing_member_count"] = 1
    if mutation == "omission":
        report["focused_rows"] = []
    with pytest.raises(ValueError):
        a.audit_seed(report, reference, memberships)


def test_incomplete_primary_stops_before_upstream(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    path.write_text('{"complete":false}')
    monkeypatch.setattr(a.sys, "argv", [str(SCRIPT), "--summary", str(path)])
    monkeypatch.setattr(a, "upstream", lambda: pytest.fail("upstream must not be read"))
    with pytest.raises(ValueError, match="incomplete"):
        a.main()
    assert not (tmp_path / "independent-audit.json").exists()


def test_existing_audit_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    out = tmp_path / "independent-audit.json"
    out.write_text("preserved")
    monkeypatch.setattr(a.sys, "argv", [str(SCRIPT), "--summary", str(path)])
    with pytest.raises(ValueError, match="overwrite"):
        a.main()
    assert out.read_text() == "preserved"
