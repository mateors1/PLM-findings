"""Synthetic new objective/comparator contracts; no neural forward or experiment."""

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "runs/learning/projection-worst-boundary-v1/independent-audit.py"
)
spec = importlib.util.spec_from_file_location("worst_projection_audit", SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def comparator():
    summary = {
        "complete": True,
        "final_identity_check": True,
        "objective": "plm-projection-only-balanced-bce-v1",
        "artifact_sha256": {
            "child.json": a.MEAN_CHILD_SHA,
            "checkpoint-final.pt": a.MEAN_CHECKPOINT_SHA,
        },
    }
    checked = {
        "complete": True,
        "audit_passed": True,
        "summary_sha256": a.MEAN_SUMMARY_SHA,
        "script_sha256": a.AUDITOR_HELPER_SHA,
    }
    decision = {
        "evidence_accepted": True,
        "fixed_quality_gate_passed": False,
        "summary_sha256": a.MEAN_SUMMARY_SHA,
        "audit_sha256": a.MEAN_AUDIT_SHA,
        "auditor_sha256": a.AUDITOR_HELPER_SHA,
        "child_checkpoint_sha256": a.MEAN_CHECKPOINT_SHA,
        "parent_checkpoint_sha256": a.PARENT_SHA,
    }
    return summary, checked, decision


def test_auth_import_pure_helpers_does_not_mutate_old_globals():
    before = copy.deepcopy(a.BASE_AUDIT.INPUTS)
    assert (
        a.sha(a.ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py")
        == a.AUDITOR_HELPER_SHA
    )
    z = [-1.0] * 1025
    z[0] = 10.0
    z[1] = 0.0
    z[2] = 1.0
    assert a.predict(z, 1024) == [1026]
    assert a.metrics([1026], [1026])["exact"]
    assert before == a.BASE_AUDIT.INPUTS
    assert a.BASE_AUDIT.OBJECTIVE == "plm-projection-only-balanced-bce-v1"
    assert a.OBJECTIVE == "plm-projection-worst-boundary-softplus-v1"


def test_historical_failed_quality_can_have_accepted_evidence():
    a.mean_comparator_binding(*comparator())


@pytest.mark.parametrize(
    "change",
    [
        "summary",
        "audit",
        "decision",
        "child",
        "parent",
        "objective",
        "accepted",
        "quality",
        "auditor",
    ],
)
def test_historical_comparator_binding_mutations(change):
    summary, checked, decision = comparator()
    if change == "summary":
        summary["complete"] = False
    elif change == "audit":
        checked["summary_sha256"] = "bad"
    elif change == "decision":
        decision["audit_sha256"] = "bad"
    elif change == "child":
        summary["artifact_sha256"]["checkpoint-final.pt"] = a.PARENT_SHA
    elif change == "parent":
        decision["parent_checkpoint_sha256"] = a.MEAN_CHECKPOINT_SHA
    elif change == "objective":
        summary["objective"] = a.OBJECTIVE
    elif change == "accepted":
        decision["evidence_accepted"] = False
    elif change == "quality":
        decision["fixed_quality_gate_passed"] = True
    else:
        decision["auditor_sha256"] = "bad"
    with pytest.raises(ValueError):
        a.mean_comparator_binding(summary, checked, decision)


def diagnostic():
    return {
        "initial": a.MEAN_INITIAL_LOSS,
        "final": 0.004,
        "initial_replay_exact": True,
        "used_for_updates": False,
    }


def test_diagnostic_is_not_worst_loss_or_improvement_gate():
    old = {"initial_pre_update_loss": a.MEAN_INITIAL_LOSS}
    assert (
        a.mean_diagnostic_contract(diagnostic(), {"mean_bce_diagnostic": diagnostic()}, old)[
            "final"
        ]
        > 0.003
    )
    history = [
        {"update": i, "pre_update_loss": 0.5, "gradient_finite": True, "parameters_finite": True}
        for i in range(1, 501)
    ]
    a.loss_endpoints(history, 0.5, 0.2)
    assert a.BASE_AUDIT.INPUTS == {}


@pytest.mark.parametrize(
    "change", ["initial", "final_nan", "final_negative", "replay", "gradient", "historical"]
)
def test_mean_diagnostic_endpoint_mutations(change):
    value = diagnostic()
    old = {"initial_pre_update_loss": a.MEAN_INITIAL_LOSS}
    if change == "initial":
        value["initial"] += 1e-8
    elif change == "final_nan":
        value["final"] = float("nan")
    elif change == "final_negative":
        value["final"] = -1.0
    elif change == "replay":
        value["initial_replay_exact"] = False
    elif change == "gradient":
        value["used_for_updates"] = True
    else:
        old["initial_pre_update_loss"] += 1e-8
    with pytest.raises(ValueError):
        a.mean_diagnostic_contract(value, {"mean_bce_diagnostic": value}, old)


def rows():
    return [
        {
            "subject": f"PKM_{i}",
            "dimension": "TYPE",
            "prompt_ids": [1, 1024 + i, 32, 34, 5],
            "expected_set_ids": [2048],
            "group": "TYPE_dual",
            "strict_separation": value,
        }
        for i, value in enumerate((True, False, True))
    ]


def test_separation_pairs_are_distinct_from_exact_pairs():
    current = rows()
    old = copy.deepcopy(current)
    old[0]["strict_separation"] = False
    old[1]["strict_separation"] = True
    result = a.separation_pairs(current, old)
    assert result["gains"] == result["losses"] == 1
    assert result["groups"]["TYPE_dual"] == {"gains": 1, "losses": 1}


@pytest.mark.parametrize("field", ["subject", "dimension", "prompt_ids", "expected_set_ids"])
def test_separation_comparator_identity_mutation(field):
    current = rows()
    old = copy.deepcopy(current)
    old[0][field] = None
    with pytest.raises(ValueError):
        a.separation_pairs(current, old)


def test_new_gate_remains_strong201_not_historical122():
    child = {
        "aggregate": {"exact_count": 123, "f1": 0.99, "serialization_compatible": 222},
        "groups": {
            g: {"exact_count": n}
            for g, n in [("COLOR", 103), ("TYPE_single", 50), ("TYPE_dual", 48)]
        },
    }
    gate = a.screen_gate(child, {"complete": True})
    assert not gate["checks"]["exact_above_201"]
    assert gate["accepted"] is False


def lineage():
    return {
        "objective": a.OBJECTIVE,
        "parent_checkpoint_sha256": a.PARENT_SHA,
        "parent_training_steps": 2000,
        "projection_updates": 500,
        "trainable_parameters": [a.W],
        "record_count": 1637,
    }, {"evaluator_version": a.EVALUATOR}


def test_new_sibling_lineage():
    a.lineage_contract(*lineage())


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective", "plm-projection-only-balanced-bce-v1"),
        ("parent_checkpoint_sha256", a.MEAN_CHECKPOINT_SHA),
        ("parent_training_steps", 2500),
        ("projection_updates", 1000),
        ("trainable_parameters", ["token_embedding.weight"]),
        ("record_count", 1859),
    ],
)
def test_wrong_sibling_lineage(field, value):
    metadata, identity = lineage()
    metadata[field] = value
    with pytest.raises(ValueError):
        a.lineage_contract(metadata, identity)


def test_old_evaluator_is_not_compatible():
    metadata, identity = lineage()
    identity["evaluator_version"] = "plm-projection-refit-screen-v1"
    with pytest.raises(ValueError):
        a.lineage_contract(metadata, identity)


def test_mean_metadata_endpoint_cannot_be_worst_endpoint():
    d = diagnostic()
    metadata = {"mean_bce_diagnostic": {**d, "final": 0.8}}
    with pytest.raises(ValueError, match="metadata endpoints"):
        a.mean_diagnostic_contract(d, metadata, {"initial_pre_update_loss": a.MEAN_INITIAL_LOSS})


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective", "plm-projection-only-balanced-bce-v1"),
        ("projection_updates", 501),
        ("threshold", 0.1),
        ("loss_definition", "query-mean-bce"),
        ("mean_bce_diagnostic", {"used_for_updates": True}),
    ],
)
def test_recipe_has_only_declared_treatment(field, value):
    import json

    recipe = json.loads(
        (a.ROOT / "configs/experiments/projection_worst_boundary_v1.json").read_text()
    )
    a.recipe_contract(recipe)
    recipe[field] = value
    with pytest.raises(ValueError):
        a.recipe_contract(recipe)


def test_comparison_full_rows_and_separation_are_independent():
    current = rows()
    for index, row in enumerate(current):
        row["index"] = index
        row["selected_set_ids"] = [2048] if index == 0 else []
        row["metrics"] = a.metrics(row["selected_set_ids"], row["expected_set_ids"])
    prior = copy.deepcopy(current)
    prior[0]["strict_separation"] = False
    prior[1]["selected_set_ids"] = [2048]
    prior[1]["metrics"] = a.metrics([2048], [2048])
    result = a.comparison_report(
        {"responses": current},
        {"responses": prior},
        {"responses": prior},
        {"responses": [{"composition": {"selected_set_ids": [2048]}} for _ in current]},
    )
    assert result["aggregate"]["parent_dense"]["exact"]["losses"] == 1
    assert result["aggregate"]["parent_dense"]["strict_separation"] == {"gains": 1, "losses": 0}
    assert len(result["per_query"]) == 6
    prior[0]["group"] = "COLOR"
    with pytest.raises(ValueError, match="query identity"):
        a.comparison_report(
            {"responses": current}, {"responses": prior}, {"responses": prior}, {"responses": []}
        )
