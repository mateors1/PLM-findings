"""Synthetic coverage-seeking audit tests; no neural or campaign measurements."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "runs/learning/coverage-seeking-branch-v1/independent-audit.py"
)
spec = importlib.util.spec_from_file_location("coverage_audit", SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
PROMPT = [1, 1024, 32, 34, 5]
VOCAB = [f"reserved_{i}" for i in range(1024)] + [f"PKM_{i}" for i in range(1024, 2049)]


def raw(products, descriptor):
    return {
        "token_ids": [*PROMPT, *products, 2],
        "targets": [VOCAB[i] for i in products],
        "terminated": True,
        "protocol_valid": True,
        "error": None,
        "decoding": descriptor,
    }


def base():
    sets = [[1025 + i] for i in range(8)]
    values = [-1.0] * 1025
    for i in range(1, 9):
        values[i] = 1.0
    paths = [
        raw(members, a.BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""))
        for rank, members in enumerate(sets, 1)
    ]
    slots = a.pool(sets, [True] * 8, values)
    return {
        "index": 0,
        "subject": "PKM_1024",
        "dimension": "TYPE",
        "group": "TYPE_dual",
        "prompt_ids": PROMPT,
        "expected": ["PKM_1025", "PKM_1026"],
        "expected_set_ids": [1025, 1026],
        "symmetric_relation_logits": values,
        "composition": {
            "source_paths": paths,
            "slots": slots,
            **a.select(slots),
            "policy": a.POLICY8,
        },
    }


def test_anchor_subject_union_positive_and_token_ties():
    values = [-1.0] * 1025
    values[0] = 100.0
    values[1] = 99.0
    values[9] = values[10] = 2.0
    values[11] = 0.0
    assert a.anchor([1025], 1024, values) == {
        "active": True,
        "token_id": 1033,
        "logit": 2.0,
        "uncovered_positive_count": 2,
    }


@pytest.mark.parametrize(
    "values,union",
    [([0.0] * 1025, []), ([-1.0] * 1025, []), ([1.0] * 1025, list(range(1025, 2049)))],
)
def test_no_eligible_anchor(values, union):
    assert a.anchor(union, 1024, values) == {
        "active": False,
        "token_id": None,
        "logit": None,
        "uncovered_positive_count": 0,
    }


def test_anchor_does_not_receive_truth_and_can_be_false():
    values = [-1.0] * 1025
    values[9] = 2.0
    decision = a.anchor([1025], 1024, values)
    truth1 = {1033}
    truth2 = {1040}
    assert decision["token_id"] in truth1 and decision["token_id"] not in truth2
    assert a.anchor([1025], 1024, values) == decision


def test_45_order_retains36_then_new_original_and_eight_pairs():
    original = base()
    sets, ok, union = a.baseline(original, VOCAB)
    extended = a.pool([*sets, [1033]], [*ok, True], original["symmetric_relation_logits"])
    assert len(extended) == 45 and extended[:36] == original["composition"]["slots"]
    assert extended[36]["source_ranks"] == [9] and extended[36]["kind"] == "original"
    assert [s["source_ranks"] for s in extended[37:]] == [[i, 9] for i in range(1, 9)]
    assert union == list(range(1025, 1033))


def test_old_first_ties_and_fraction_score_cancellation():
    values = [0.0] * 1025
    sets = [[1025 + i] for i in range(8)]
    slots = a.pool([*sets, [1033]], [True] * 9, values)
    assert a.select(slots)["selected_slot"] == 1
    values[1:4] = [2.0**80, 1.0, -(2.0**80)]
    assert a.canonical_sum([1025, 1026, 1027], values) == math.fsum(values[1:4]) == 1.0


def test_invalid_new_source_not_eligible_and_never_improves_coverage():
    original = base()
    sets, ok, union = a.baseline(original, VOCAB)
    values = original["symmetric_relation_logits"]
    values[9] = 100.0
    slots = a.pool([*sets, [1033]], [*ok, False], values)
    assert all(not s["source_eligible"] for s in slots[36:])
    assert a.select(slots)["selected_slot"] <= 36
    assert a.expansion(union, [1033], False, a.anchor(union, 1024, values)) == []


def test_valid_active_expansion_contains_anchor_not_necessarily_truth():
    values = [-1.0] * 1025
    values[9] = 2.0
    decision = a.anchor([1025], 1024, values)
    assert a.expansion([1025], [1025, 1033, 1034], True, decision) == [1033, 1034]
    with pytest.raises(ValueError, match="anchor"):
        a.expansion([1025], [1025, 1034], True, decision)


def test_control_full_path_parity_descriptor_only_exception():
    saved = raw([1025, 1026], a.BASE)
    control = copy.deepcopy(saved)
    control["decoding"] = "forced-v1"
    a.control_parity(control, saved, PROMPT, "forced-v1", VOCAB)
    control["token_ids"][6] = 1027
    control["targets"][1] = VOCAB[1027]
    with pytest.raises(ValueError, match="parity"):
        a.control_parity(control, saved, PROMPT, "forced-v1", VOCAB)


def test_new_forced_first_id_and_subject_removal():
    path = raw([1033, 1024, 1025], "forced-v1")
    assert a.new_path_membership(path, PROMPT, "forced-v1", 1033, VOCAB) == ([1025, 1033], True)
    with pytest.raises(ValueError, match="forced first"):
        a.new_path_membership(path, PROMPT, "forced-v1", 1034, VOCAB)


def test_inactive_padding_trace_must_not_add_to_pool():
    original = base()
    sets, ok, union = a.baseline(original, VOCAB)
    decision = a.anchor(union, 1024, original["symmetric_relation_logits"])
    assert not decision["active"]
    padding = raw([1025, 1033], "forced-v1")
    members, valid = a.new_path_membership(padding, PROMPT, "forced-v1", 1025, VOCAB)
    assert members == [1025, 1033] and valid
    assert a.expansion(union, members, valid, decision) == []
    assert (
        a.pool(sets, ok, original["symmetric_relation_logits"]) == original["composition"]["slots"]
    )


def test_nonterminated_raw_preserves_subject_and_ineligible_status():
    products = [1033, 1024, *range(1040, 1545)]
    path = {
        "token_ids": [*PROMPT, *products],
        "targets": [VOCAB[i] for i in products],
        "terminated": False,
        "protocol_valid": False,
        "error": "response must terminate with EOS",
        "decoding": "forced-v1",
    }
    members, valid = a.new_path_membership(path, PROMPT, "forced-v1", 1033, VOCAB)
    assert 1024 in members and not valid and len(path["token_ids"]) == 512


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.1, True, 1])
def test_nonfp32_or_nonfinite_anchor_inputs_rejected(bad):
    values = [-1.0] * 1025
    values[9] = bad
    with pytest.raises(ValueError):
        a.anchor([], 1024, values)


def complete_row(active=True):
    old = base()
    if active:
        old["symmetric_relation_logits"][9] = 2.0
    control = {
        "index": 0,
        "symmetric_relation_logits": old["symmetric_relation_logits"],
        "control_path": raw([1025], a.FORCED),
    }
    decision = a.anchor(list(range(1025, 1033)), 1024, old["symmetric_relation_logits"])
    first = 1033 if active else 1025
    new = raw([first], a.FORCED)
    source_sets = [[i] for i in range(1025, 1033)] + ([[first]] if active else [])
    slots = a.pool(source_sets, [True] * len(source_sets), old["symmetric_relation_logits"])
    composition = {
        "source_paths": [*old["composition"]["source_paths"], *([new] if active else [])],
        "slots": slots,
        **a.select(slots),
        "policy": a.POLICY,
    }
    result = {k: v for k, v in old.items() if k != "composition"}
    result.update(
        {
            "baseline": old["composition"],
            "control_path": control["control_path"],
            "anchor": decision,
            "forced_first_id": first,
            "new_path": new,
            "new_source_set_ids": [first],
            "new_source_eligible": True,
            "composition": composition,
        }
    )
    truth = old["expected_set_ids"]
    result["metrics"] = {
        "baseline": a.answer_metrics(old["composition"], truth),
        "new": a.answer_metrics(composition, truth),
    }
    result["availability"] = {
        "baseline": a.availability(old["composition"], truth),
        "new": a.availability(composition, truth),
    }
    result.update(a.coverage_changes(list(range(1025, 1033)), [first], True, decision, truth))
    result.update(a.answer_errors(composition, truth))
    result["gained_exact"] = (
        result["metrics"]["new"]["exact"] and not result["metrics"]["baseline"]["exact"]
    )
    result["lost_exact"] = (
        result["metrics"]["baseline"]["exact"] and not result["metrics"]["new"]["exact"]
    )
    return old, result, control


@pytest.mark.parametrize("active", [True, False])
def test_complete_row_and_padding_exclusion(active):
    old, row, control = complete_row(active)
    assert a.rebuild_row(old, row, control, VOCAB, 0) == row
    assert len(row["composition"]["slots"]) == (45 if active else 36)
    assert row["newly_covered_false_ids"] == ([1033] if active else [])


@pytest.mark.parametrize(
    "field",
    [
        "forced_first_id",
        "new_source_set_ids",
        "new_source_eligible",
        "anchor_correct",
        "gained_exact",
    ],
)
def test_complete_row_corruption_rejected(field):
    old, row, control = complete_row()
    row[field] = None
    with pytest.raises(ValueError):
        a.rebuild_row(old, row, control, VOCAB, 0)


def test_work_tracks_inactive_padding_and_inclusive_eos():
    paths = [raw([1025], a.FORCED), raw([1026, 1027, 1028], a.FORCED)]
    work = a.work_record(paths, [True, False], 8, "anchor", 0.5)
    assert work["decode_calls"] == 4
    assert work["padded_decode_positions"] == 16
    assert work["active_decode_positions"] == work["padding_decode_positions"] == 8
    assert work["total_emitted_tokens"] == 6
    assert work["useful_emitted_tokens"] == work["active_emitted_tokens"] == 2
    assert work["padding_emitted_tokens"] == 4
    assert work["guidance_forwards"] == 1


def passing_gate():
    reports = [
        {
            "complete": True,
            "replay_exact": True,
            "query_count": 222,
            "responses": [{"anchor": {"active": True}, "new_source_eligible": True}],
            "overall": {
                "baseline": {"exact_count": 201, "f1": 0.9},
                "new": {"exact_count": 202, "f1": 0.91},
            },
        }
        for _ in range(3)
    ]
    pooled = {
        "overall": {"new": {"failures": 0, "exact_count": 606}},
        "groups": {
            g: {"baseline": {"exact_count": 155}, "new": {"exact_count": 156}} for g in a.GROUPS
        },
    }
    return reports, pooled


def test_strict_gate_and_seed_regression():
    reports, pooled = passing_gate()
    assert a.gate(reports, pooled)["quality_passed"]
    reports[0]["overall"]["new"]["f1"] = 0.89
    assert not a.gate(reports, pooled)["checks"]["per_seed_f1_nonregression"]
    assert not a.gate(reports, pooled)["quality_passed"]


def test_invalid_active_source_fails_but_padding_not_gate_input():
    reports, pooled = passing_gate()
    reports[0]["responses"][0]["new_source_eligible"] = False
    assert not a.gate(reports, pooled)["quality_passed"]
    reports[0]["responses"][0]["anchor"]["active"] = False
    assert a.gate(reports, pooled)["quality_passed"]


def test_no_improvement_or_group_loss_fails_gate():
    reports, pooled = passing_gate()
    pooled["overall"]["new"]["exact_count"] = 603
    assert not a.gate(reports, pooled)["checks"]["pooled_exact_improvement"]
    pooled["groups"]["COLOR"]["new"]["exact_count"] = 154
    assert not a.gate(reports, pooled)["checks"]["pooled_group_nonregression"]


def test_hash_corruption_and_overwrite_refusal(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    digest = a.sha(path)
    assert a.read(path, digest) == {}
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        a.read(path, digest)


def test_totals_include_false_anchors_and_padding_denominator():
    _, active, _ = complete_row(True)
    _, inactive, _ = complete_row(False)
    result = a.totals([active, inactive])
    assert result["query_count"] == 2
    assert result["active_anchors"] == result["inactive_anchors"] == 1
    assert result["false_anchors"] == 1
    assert result["newly_covered_false_occurrences"] == 1
    assert (
        result["new"]["f1"]
        == (active["metrics"]["new"]["f1"] + inactive["metrics"]["new"]["f1"]) / 2
    )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1.0])
def test_timing_rejects_nonfinite_negative_and_boolean(value):
    with pytest.raises(ValueError):
        a.finite_nonnegative(value)


def test_runtime_origin_cannot_escape_archive(tmp_path):
    runtime = tmp_path / "runtime"
    origins = {
        name: {"path": str(tmp_path / (name + ".py")), "sha256": "x"}
        for name in ("plm", "plm.serving.runtime", "plm.serving.set_reranking")
    }
    with pytest.raises(ValueError, match="outside isolation"):
        a.runtime_origins(origins, {}, runtime)


def test_archive_traversal_is_rejected(tmp_path):
    import zipfile

    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escaped.py", "bad")
    with pytest.raises(ValueError, match="unsafe archive"):
        a.safe_archive(path, a.sha(path))


def test_fresh_head_or_control_corruption_rejected():
    old, row, control = complete_row()
    control["symmetric_relation_logits"] = list(control["symmetric_relation_logits"])
    control["symmetric_relation_logits"][9] = 3.0
    with pytest.raises(ValueError, match="fresh head"):
        a.rebuild_row(old, row, control, VOCAB, 0)
    old, row, control = complete_row()
    control["control_path"] = raw([1026], a.FORCED)
    with pytest.raises(ValueError, match="parity"):
        a.rebuild_row(old, row, control, VOCAB, 0)
