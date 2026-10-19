"""Synthetic saved-evidence fixtures only; no real data or neural execution."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "bilinear-budget2000-v1/independent-audit.py"
SPEC = importlib.util.spec_from_file_location("budget2000_auditor", PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def trace(count=2000):
    return [
        {
            "update": i,
            "pre_update_loss": A.INITIAL_LOSS,
            "gradient_finite": True,
            "parameters_finite": True,
        }
        for i in range(1, count + 1)
    ]


def test_complete2000_trace():
    A.loss_contract(trace(), A.INITIAL_LOSS, 0.0001)


@pytest.mark.parametrize("count", [500, 1999, 2001])
def test_old_or_extended_budget_rejected(count):
    with pytest.raises(ValueError):
        A.loss_contract(trace(count), A.INITIAL_LOSS, 0.0001)


@pytest.mark.parametrize(
    "mutation", ["order", "nan", "negative", "gradient", "parameter", "initial", "final"]
)
def test_full_trace_corruption_rejected(mutation):
    records = trace()
    initial, final = A.INITIAL_LOSS, 0.001
    if mutation == "order":
        records[1500]["update"] = 1500
    elif mutation == "nan":
        records[1500]["pre_update_loss"] = float("nan")
    elif mutation == "negative":
        records[1500]["pre_update_loss"] = -1
    elif mutation == "gradient":
        records[1500]["gradient_finite"] = False
    elif mutation == "parameter":
        records[1500]["parameters_finite"] = False
    elif mutation == "initial":
        initial = 0.01
    else:
        final = float("inf")
    with pytest.raises(ValueError):
        A.loss_contract(records, initial, final)


def optimizer():
    group = {
        "lr": 0.0003,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "fused": None,
        "amsgrad": False,
        "maximize": False,
    }
    payload = {
        "state": {
            9: {
                "step": torch.tensor(2000.0),
                "exp_avg": torch.zeros(A.SHAPE),
                "exp_avg_sq": torch.ones(A.SHAPE),
            }
        },
        "param_groups": [{**group, "params": [9]}, {**group, "params": []}],
    }
    implementation = {
        "class": "torch.optim.adamw.AdamW",
        "groups": [{**group, "parameter_names": [A.A]}, {**group, "parameter_names": []}],
        "defaults": group,
    }
    return payload, implementation


def test_optimizer2000_named_state():
    A.optimizer_contract(*optimizer())


@pytest.mark.parametrize(
    "mutation", ["step500", "step2001", "name", "shape", "extra", "lr", "nonfinite"]
)
def test_optimizer_violations(mutation):
    payload, impl = optimizer()
    state = payload["state"][9]
    if mutation == "step500":
        state["step"] = torch.tensor(500)
    elif mutation == "step2001":
        state["step"] = torch.tensor(2001)
    elif mutation == "name":
        impl["groups"][0]["parameter_names"] = ["symmetric_relation_projection.weight"]
    elif mutation == "shape":
        state["exp_avg"] = torch.zeros(256, 256)
    elif mutation == "extra":
        payload["state"][10] = {}
    elif mutation == "lr":
        payload["param_groups"][0]["lr"] = 0.001
    else:
        state["exp_avg"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        A.optimizer_contract(payload, impl)


def lineage():
    return {
        "architecture": A.ARCHITECTURE_DESCRIPTOR,
        "objective": A.OBJECTIVE,
        "evaluator": A.EVALUATOR,
        "parent_checkpoint_sha256": A.PARENT_SHA,
        "parent_training_steps": 2000,
        "residual_updates": 2000,
        "trainable_parameters": [A.A],
        "record_count": 1637,
    }


def test_separate_parent_and_new_update_fields():
    A.lineage_contract(lineage(), {"evaluator_version": A.EVALUATOR})


@pytest.mark.parametrize(
    "key,value",
    [
        ("residual_updates", 500),
        ("residual_updates", 4000),
        ("parent_checkpoint_sha256", A.HISTORICAL_CHECKPOINT_SHA),
        ("parent_training_steps", 500),
        ("trainable_parameters", ["symmetric_relation_projection.weight"]),
    ],
)
def test_new_child_cannot_resume_old_or_add_steps(key, value):
    metadata = lineage()
    metadata[key] = value
    with pytest.raises(ValueError):
        A.lineage_contract(metadata, {"evaluator_version": A.EVALUATOR})


def test_recipe_changes_only_campaign_and_budget():
    old = {
        "experiment": "bilinear-residual-refit-v1",
        "residual_updates": 500,
        "learning_rate": 0.0003,
        "architecture": A.ARCHITECTURE,
    }
    new = {**old, "experiment": "bilinear-budget2000-v1", "residual_updates": 2000}
    A.recipe_contract(new, old)
    new["learning_rate"] = 0.001
    with pytest.raises(ValueError):
        A.recipe_contract(new, old)


def report(missing=()):
    rows = []
    for i in range(222):
        correct = i not in missing
        group = ("COLOR", "TYPE_single", "TYPE_dual")[i % 3]
        values = [0.0] * 1025
        values[-1] = 1.0 if correct else -1.0
        rows.append(
            {
                "index": i,
                "subject": f"PKM_{i}",
                "dimension": "COLOR" if group == "COLOR" else "TYPE",
                "group": group,
                "prompt_ids": [1, 1024 + i, 33 if group == "COLOR" else 32, 34, 5],
                "expected_set_ids": [2048],
                "symmetric_relation_logits": values,
                "selected_set_ids": [2048] if correct else [],
                "metrics": {
                    "exact": correct,
                    "precision": 1.0 if correct else 0.0,
                    "recall": 1.0 if correct else 0.0,
                    "f1": 1.0 if correct else 0.0,
                    "set_size": 1 if correct else 0,
                    "false_positive_count": 0,
                    "false_negative_count": 0 if correct else 1,
                    "serialization_compatible": correct,
                },
            }
        )
    return {
        "complete": True,
        "query_count": 222,
        "product_token_ids": list(range(1024, 2049)),
        "responses": rows,
    }


def test_historical_pairing_independent_gain_loss():
    result = A.historical_pairing(report([2]), report([0, 1]), "a" * 64)
    assert result["paired_vs_historical500"] == {
        "gains": 2,
        "losses": 1,
        "groups": {
            "COLOR": {"gains": 1, "losses": 0},
            "TYPE_single": {"gains": 1, "losses": 0},
            "TYPE_dual": {"gains": 0, "losses": 1},
        },
    }
    assert result["comparison_scope"] == "saved historical output; no rerun"


@pytest.mark.parametrize(
    "mutation", ["order", "truth", "group", "prompt", "columns", "missing", "metric", "selection"]
)
def test_historical_pairing_rejects_misalignment_or_corruption(mutation):
    child, historical = report(), report()
    if mutation == "order":
        historical["responses"].reverse()
    elif mutation == "truth":
        historical["responses"][0]["expected_set_ids"] = [2047]
    elif mutation == "group":
        historical["responses"][0]["group"] = "TYPE_dual"
    elif mutation == "prompt":
        historical["responses"][0]["prompt_ids"][2] = 32
    elif mutation == "columns":
        historical["product_token_ids"].reverse()
    elif mutation == "missing":
        historical["responses"].pop()
    elif mutation == "metric":
        historical["responses"][0]["metrics"]["exact"] = False
    else:
        historical["responses"][0]["selected_set_ids"] = []
    with pytest.raises(ValueError):
        A.historical_pairing(child, historical, "a" * 64)


def test_architecture_and_gate_reused_without_global_mutation():
    A.architecture_contract(copy.deepcopy(A.OLD.ARCHITECTURE_DESCRIPTOR))
    child = {
        "aggregate": {"serialization_compatible": 222, "exact_count": 201, "f1": 0.999},
        "groups": {
            "COLOR": {"exact_count": 103},
            "TYPE_single": {"exact_count": 50},
            "TYPE_dual": {"exact_count": 48},
        },
    }
    assert not A.screen_gate(child, {"checked": True})["primary_checks_passed"]
    assert A.OLD.INPUTS == A.BASE.INPUTS == {}


def test_output_overwrite_refused(tmp_path, monkeypatch):
    output = tmp_path / "independent-audit.json"
    output.write_text("preserve")
    monkeypatch.setattr("sys.argv", ["audit", "--summary", str(tmp_path / "summary.json")])
    monkeypatch.setattr(A, "audit", lambda _: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="immutable"):
        A.main()
    assert output.read_text() == "preserve"
