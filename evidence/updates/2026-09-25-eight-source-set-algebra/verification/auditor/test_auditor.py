"""Synthetic bitset/rational audit checks; no real algebra-screen measurements."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "runs/learning/eight-source-set-algebra-v1/independent-audit.py"
)
spec = importlib.util.spec_from_file_location("set_algebra_audit", SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def masks(sets):
    return [a.bits(sorted(set(s))) for s in sets]


def fixture_sets():
    return masks([[1025 + i] for i in range(8)])


def test_bitset_roundtrip_boundary_and_invalid_ids():
    values = [1024, 1025, 2048]
    assert a.ids(a.bits(values)) == values
    for values in ([1025, 1024], [1025, 1025], [1023], [2049], [True]):
        with pytest.raises(ValueError):
            a.bits(values)
    for value in (-1, True, 1 << 1025):
        with pytest.raises(ValueError):
            a.ids(value)


def test_union_intersection_algebra_and_subset_order():
    source = masks(
        [[1025, 1026], [1026, 1027], [1025, 1027], [1028], [1029], [1030], [1031], [1032]]
    )
    assert a.ids(a.operate(source, [1, 2])) == [1025, 1026, 1027]
    assert a.ids(a.operate(source, [1, 2], True)) == [1026]
    assert a.operate(source, [1, 2, 3], True) == 0
    assert len(a.SUBSETS) == len(set(a.SUBSETS)) == 255
    assert a.SUBSETS[:8] == tuple((i,) for i in range(1, 9))
    assert a.SUBSETS[8] == (1, 2) and a.SUBSETS[-1] == tuple(range(1, 9))


def test_all64_slots_preserve36_before28_intersections():
    source = fixture_sets()
    values = [0.0] * 1025
    slots = a.slots(source, values)
    assert len(slots) == 64
    assert slots[:36] == a.slots(source, values, include_intersections=False)
    assert [(s["source_ranks"], s["kind"]) for s in slots[36:]] == [
        (list(p), "pair_intersection") for p in a.PAIRS
    ]
    assert all(s["set_ids"] == [] and s["source_eligible"] and s["score"] == 0 for s in slots[36:])
    assert a.select(slots)["selected_slot"] == 1


def test_exact_rational_scores_and_float32_guards():
    values = [0.0] * 1025
    values[1:4] = [2.0**80, 1.0, -(2.0**80)]
    assert a.canonical_sum([1025, 1026, 1027], values) == math.fsum(values[1:4]) == 1.0
    for bad in (0.1, float("inf"), float("nan"), 1, True):
        values[0] = bad
        with pytest.raises(ValueError):
            a.vector(values)


def test_empty_intersection_can_win_negative_scores_without_being_invalid():
    slots = a.slots(fixture_sets(), [-1.0] * 1025)
    selection = a.select(slots)
    assert selection["selected_slot"] == 37
    assert selection["selected_set_ids"] == [] and selection["selected_source_eligible"]
    assert a.metric(selection, [1025]) == {
        "exact": False,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "set_size": 0,
        "success": True,
    }
    assert a.errors(0, a.bits([1025])) == {"false_positive_ids": [], "false_negative_ids": [1025]}


def test_duplicate_intersection_and_zero_score_tie_keeps_old_candidate():
    source = masks([[1025, 1026], [1025, 1027], [1025], [1028], [1029], [1030], [1031], [1032]])
    values = [-1.0] * 1025
    values[1] = 0.0
    slots = a.slots(source, values)
    assert slots[2]["set_ids"] == slots[36]["set_ids"] == [1025]
    assert a.select(slots)["selected_slot"] == 3


def test_union_exact_requires_three_first_witness_is_cardinality_lexicographic():
    source = fixture_sets()
    assert a.witness(source, [1025, 1026, 1027]) == {
        "exact_available": True,
        "min_sources": 3,
        "witness": [1, 2, 3],
    }
    assert a.witness(source, [1025, 1026, 1027], True) == {
        "exact_available": False,
        "min_sources": None,
        "witness": None,
    }


def test_truth_covered_but_every_union_polluted_intersection_can_repair():
    source = masks([[1025, 1026], [1025, 1027], [1028], [1029], [1030], [1031], [1032], [1033]])
    assert a.witness(source, [1025])["exact_available"] is False
    assert a.witness(source, [1025], True) == {
        "exact_available": True,
        "min_sources": 2,
        "witness": [1, 2],
    }
    assert a.bits([1025]) & ~a.operate(source, tuple(range(1, 9))) == 0


def test_missing_truth_unattainable_by_either_family():
    source = fixture_sets()
    assert not a.witness(source, [1025, 1040])["exact_available"]
    assert not a.witness(source, [1025, 1040], True)["exact_available"]


def test_intersection_requires_three_sources_and_tie_witness_is_first():
    source = masks(
        [
            [1025, 1026, 1027],
            [1025, 1026, 1028],
            [1025, 1027, 1028],
            [1029],
            [1030],
            [1031],
            [1032],
            [1033],
        ]
    )
    assert a.witness(source, [1025], True) == {
        "exact_available": True,
        "min_sources": 3,
        "witness": [1, 2, 3],
    }


def test_predictor_is_truth_free_and_new_intersection_can_regress():
    source = masks([[1025, 1026], [1025, 1027], [1028], [1029], [1030], [1031], [1032], [1033]])
    values = [-10.0] * 1025
    values[1] = 2.0
    values[2] = -1.0
    values[3] = -2.0
    old = a.select(a.slots(source, values, include_intersections=False))
    wide = a.select(a.slots(source, values))
    assert old["selected_set_ids"] == [1025, 1026]
    assert wide["selected_set_ids"] == [1025]
    assert a.metric(old, [1025, 1026])["exact"]
    assert not a.metric(wide, [1025, 1026])["exact"]
    assert a.metric(wide, [1025])["exact"]
    assert a.select(a.slots(source, values)) == wide


@pytest.mark.parametrize(
    "exact,available,covered,state",
    [
        (True, True, True, "selected_exact"),
        (False, True, True, "available_exact_but_missed"),
        (False, False, False, "unavailable_and_truth_missing_from_all_sources"),
        (False, False, True, "unavailable_with_truth_covered"),
    ],
)
def test_baseline_states(exact, available, covered, state):
    assert a.baseline_state(exact, available, covered) == state


def base_row():
    vocabulary = [f"reserved_{i}" for i in range(1024)] + [f"PKM_{i}" for i in range(1024, 2049)]
    prompt = [1, 1024, 32, 34, 5]
    values = [1.0] * 1025
    paths = [
        {
            "token_ids": [*prompt, 1025 + i, 2],
            "targets": [f"PKM_{1025 + i}"],
            "terminated": True,
            "protocol_valid": True,
            "error": None,
            "decoding": a.BASE + (f"+first-rank{i + 1}-v1" if i else ""),
        }
        for i in range(8)
    ]
    rawslots = a.slots(fixture_sets(), values, include_intersections=False)
    selection = a.select(rawslots)
    comp = {"source_paths": paths, "slots": rawslots, **selection, "policy": a.POLICY8}
    row = {
        "index": 0,
        "subject": "PKM_1024",
        "dimension": "TYPE",
        "group": "TYPE_dual",
        "prompt_ids": prompt,
        "expected": ["PKM_1025", "PKM_1026"],
        "expected_set_ids": [1025, 1026],
        "symmetric_relation_logits": values,
        "composition": comp,
        "metrics": {"wide": a.metric(selection, [1025, 1026])},
        "subject_removed": [False] * 8,
        "availability": {"wide": {"exact_available": True}},
    }
    return row, vocabulary


def test_upstream_replay_passes_and_subject_semantics_preserve_raw():
    row, vocab = base_row()
    before = copy.deepcopy(row)
    source, removed, reconstructed, selection = a.validate_base(row, vocab, 0)
    assert row == before and source == fixture_sets() and removed == [False] * 8
    assert reconstructed == row["composition"]["slots"]
    assert selection["selected_set_ids"] == [1025, 1026]


@pytest.mark.parametrize(
    "mutation",
    [
        "score",
        "head",
        "slot_order",
        "raw_repeat",
        "raw_early",
        "rank",
        "subject_truth",
        "target_key",
        "metric",
    ],
)
def test_upstream_mutations_rejected(mutation):
    row, vocab = base_row()
    if mutation == "score":
        row["composition"]["slots"][35]["score"] += 1
    if mutation == "head":
        row["symmetric_relation_logits"][2] = float("nan")
    if mutation == "slot_order":
        row["composition"]["slots"][34], row["composition"]["slots"][35] = (
            row["composition"]["slots"][35],
            row["composition"]["slots"][34],
        )
    if mutation == "raw_repeat":
        row["composition"]["source_paths"][0]["token_ids"].insert(6, 1025)
        row["composition"]["source_paths"][0]["targets"].append("PKM_1025")
    if mutation == "raw_early":
        row["composition"]["source_paths"][0]["token_ids"][-1] = 1026
    if mutation == "rank":
        row["composition"]["source_paths"][7]["decoding"] = a.BASE
    if mutation == "subject_truth":
        row["expected_set_ids"] = [1024]
    if mutation == "target_key":
        row["composition"]["source_paths"][0]["targets"][0] = "PKM_wrong"
    if mutation == "metric":
        row["metrics"]["wide"]["f1"] = 0.5
    with pytest.raises(ValueError):
        a.validate_base(row, vocab, 0)


def test_complete_synthetic_row_witness_errors_and_aggregate():
    base, vocab = base_row()
    row = a.reconstruct(base, vocab, 0)
    assert len(row["slots"]) == 64
    assert row["source_set_ids"] == [[1025 + i] for i in range(8)]
    assert row["metrics"]["baseline"]["exact"] and row["metrics"]["prediction"]["exact"]
    assert row["diagnosis"]["unions"] == {
        "exact_available": True,
        "min_sources": 2,
        "witness": [1, 2],
    }
    assert row["diagnosis"]["intersections"] == {
        "exact_available": False,
        "min_sources": None,
        "witness": None,
    }
    assert row["errors"]["sources"][0] == {"false_positive_ids": [], "false_negative_ids": [1026]}
    assert row["errors"]["sources"][2] == {
        "false_positive_ids": [1027],
        "false_negative_ids": [1025, 1026],
    }
    assert row["errors"]["prediction"] == {"false_positive_ids": [], "false_negative_ids": []}
    total = a.totals([row, copy.deepcopy(row)])
    assert total["query_count"] == total["prediction"]["exact_count"] == 2
    assert total["baseline_state_counts"]["selected_exact"] == 2
    assert total["pure_family_availability"] == {
        "all_unions": 2,
        "all_intersections": 0,
        "either_pure_family": 2,
    }
    assert total["gains"] == total["losses"] == total["changed_selection_count"] == 0


def test_source_subject_removal_checked_from_raw_before_bitset_construction():
    base, vocab = base_row()
    path = base["composition"]["source_paths"][0]
    path["token_ids"].insert(-1, 1024)
    path["targets"].append("PKM_1024")
    base["subject_removed"][0] = True
    row = a.reconstruct(base, vocab, 0)
    assert row["source_subject_removed"][0]
    assert 1024 in row["source_paths"][0]["token_ids"][5:]
    assert 1024 not in row["source_set_ids"][0]


def test_empty_selection_reported_not_filtered():
    base, vocab = base_row()
    base["symmetric_relation_logits"] = [-1.0] * 1025
    oldslots = a.slots(
        fixture_sets(), base["symmetric_relation_logits"], include_intersections=False
    )
    chosen = a.select(oldslots)
    base["composition"] = {
        "source_paths": base["composition"]["source_paths"],
        "slots": oldslots,
        **chosen,
        "policy": a.POLICY8,
    }
    base["metrics"]["wide"] = a.metric(chosen, base["expected_set_ids"])
    row = a.reconstruct(base, vocab, 0)
    total = a.totals([row])
    assert (
        total["prediction"]["empty_selections"]
        == total["prediction"]["intersection_selections"]
        == 1
    )
    assert total["prediction"]["f1"] == 0
    assert row["metrics"]["prediction"]["success"] is True
    assert row["selection_changed"]


def gate_inputs():
    seeds = [
        {
            "baseline": {"exact_count": 201, "f1": 0.9},
            "prediction": {"exact_count": 202, "f1": 0.91},
        }
        for _ in range(3)
    ]
    overall = {"baseline": {"exact_count": 603}, "prediction": {"exact_count": 606}}
    groups = {
        g: {"baseline": {"exact_count": 201}, "prediction": {"exact_count": 202}} for g in a.GROUPS
    }
    return seeds, overall, groups


def test_fixed_gate_can_pass_without_a_dual_strict_gain():
    seeds, overall, groups = gate_inputs()
    groups["TYPE_dual"]["prediction"]["exact_count"] = 201
    assert a.quality_gate(seeds, overall, groups)["quality_passed"]


@pytest.mark.parametrize(
    "mutation", ["seed_exact", "seed_f1", "strict_gain", "color", "baseline", "invariants"]
)
def test_gate_each_declared_failure(mutation):
    seeds, overall, groups = gate_inputs()
    if mutation == "seed_exact":
        seeds[0]["prediction"]["exact_count"] = 200
    if mutation == "seed_f1":
        seeds[0]["prediction"]["f1"] = 0.89
    if mutation == "strict_gain":
        overall["prediction"]["exact_count"] = 603
    if mutation == "color":
        groups["COLOR"]["prediction"]["exact_count"] = 200
    assert not a.quality_gate(
        seeds,
        overall,
        groups,
        baseline_ok=mutation != "baseline",
        invariants=mutation != "invariants",
    )["quality_passed"]


def test_unchanged_scorer_bound_excludes_disjoint_obstructions():
    overall = {
        "query_count": 666,
        "baseline_state_counts": {
            "selected_exact": 603,
            "available_exact_but_missed": 2,
            "unavailable_and_truth_missing_from_all_sources": 30,
            "unavailable_with_truth_covered": 31,
        },
    }
    assert a.derived_bound(overall) == {
        "missing_union_count": 30,
        "existing_available_miss_count": 2,
        "selected_exact_upper_bound": 634,
        "scope": "appended pure union/intersection sets; unchanged scorer and old-first ties",
    }


def test_hash_mutation_rejected(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("original")
    digest = a.sha(path)
    a.bind(path, digest)
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        a.bind(path, digest)


def test_incomplete_primary_fails_before_upstream_reads(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    path.write_text('{"complete":false}')
    monkeypatch.setattr(a.sys, "argv", [str(SCRIPT), "--summary", str(path)])
    monkeypatch.setattr(a, "upstream", lambda: pytest.fail("must not read upstream"))
    with pytest.raises(ValueError, match="incomplete"):
        a.main()
    assert not (tmp_path / "independent-audit.json").exists()


def test_audit_immutable_output_refusal(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    output = tmp_path / "independent-audit.json"
    output.write_text("original receipt")
    monkeypatch.setattr(a.sys, "argv", [str(SCRIPT), "--summary", str(path)])
    with pytest.raises(ValueError, match="overwrite"):
        a.main()
    assert output.read_text() == "original receipt"
