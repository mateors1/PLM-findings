"""Synthetic saved-evidence fixtures only; no real data or neural execution."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "bilinear-budget8000-v1/independent-audit.py"
SPEC = importlib.util.spec_from_file_location("budget8000_auditor", PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def trace(count=8000):
    return [
        {
            "update": i,
            "pre_update_loss": A.INITIAL_LOSS,
            "gradient_finite": True,
            "parameters_finite": True,
        }
        for i in range(1, count + 1)
    ]


def test_complete8000_trace():
    A.loss_contract(trace(), A.INITIAL_LOSS, 0.0001)


@pytest.mark.parametrize("count", [500, 2000, 7999, 8001])
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
                "step": torch.tensor(8000.0),
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


def test_optimizer8000_named_state():
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
        "residual_updates": 8000,
        "trainable_parameters": [A.A],
        "record_count": 1637,
    }


def test_separate_parent_and_new_update_fields():
    A.lineage_contract(lineage(), {"evaluator_version": A.EVALUATOR})


@pytest.mark.parametrize(
    "key,value",
    [
        ("residual_updates", 2000),
        ("residual_updates", 8001),
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


def test_recipe_changes_only_declared_loss_gate_fields():
    old = {
        "experiment": "bilinear-budget2000-v1",
        "objective": "plm-bilinear-residual-balanced-bce-v1",
        "residual_updates": 2000,
        "learning_rate": 0.0003,
        "architecture": A.ARCHITECTURE,
        "initial_training_loss": A.INITIAL_LOSS,
    }
    new = old | A.recipe_changes()
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
    assert result["paired_vs_historical2000"] == {
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


def test_architecture_reused_and_gate_strengthened_without_global_mutation():
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


def gate_fixture():
    return {
        "aggregate": {
            "serialization_compatible": 222,
            "exact_count": 208,
            "f1": 0.9997456353806234,
        },
        "groups": {
            "COLOR": {"exact_count": 103},
            "TYPE_single": {"exact_count": 50},
            "TYPE_dual": {"exact_count": 55},
        },
    }


def test_fixed_gate_pass_still_does_not_accept_or_promote():
    gate = A.screen_gate(gate_fixture(), {"checked": True})
    assert gate["primary_checks_passed"] is True
    assert gate["accepted"] is False
    assert gate["independent_audit_required"] is True


@pytest.mark.parametrize(
    "mutation", ["tie207", "old202", "f1", "color", "single", "dual", "invalid", "invariant"]
)
def test_every_strengthened_gate_is_required(mutation):
    child = gate_fixture()
    invariants = {"checked": True}
    if mutation in ("tie207", "old202"):
        child["aggregate"]["exact_count"] = 207 if mutation == "tie207" else 202
    elif mutation == "f1":
        child["aggregate"]["f1"] = 0.9997456353806233
    elif mutation in ("color", "single", "dual"):
        group, count = {
            "color": ("COLOR", 102),
            "single": ("TYPE_single", 49),
            "dual": ("TYPE_dual", 53),
        }[mutation]
        child["groups"][group]["exact_count"] = count
    elif mutation == "invalid":
        child["aggregate"]["serialization_compatible"] = 221
    else:
        invariants["checked"] = False
    assert A.screen_gate(child, invariants)["primary_checks_passed"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective", "plm-bilinear-worst-boundary-softplus-v1"),
        ("evaluator", "plm-bilinear-residual-screen-v1"),
    ],
)
def test_old_objective_and_evaluator_rejected(field, value):
    metadata = lineage()
    metadata[field] = value
    with pytest.raises(ValueError):
        A.lineage_contract(metadata, {"evaluator_version": A.EVALUATOR})


def tensor_states():
    parent = {f"frozen_{i}": torch.tensor([float(i)]) for i in range(93)}
    child = {name: tensor.clone() for name, tensor in parent.items()}
    child[A.A] = torch.ones(A.SHAPE, dtype=torch.float32)
    return parent, child


def test_all93_frozen_and_new94th_residual():
    before, initial, after = A.compare_states(*tensor_states())
    assert len(before) == 93
    assert len(initial) == len(after) == 94
    assert initial[A.A]["sha256"] != after[A.A]["sha256"]


@pytest.mark.parametrize("mutation", ["frozen", "extra", "zero", "shape", "dtype", "nonfinite"])
def test_full_state_corruption_rejected(mutation):
    parent, child = tensor_states()
    if mutation == "frozen":
        child["frozen_92"][0] += 1
    elif mutation == "extra":
        child["new_parameter"] = torch.zeros(1)
    elif mutation == "zero":
        child[A.A].zero_()
    elif mutation == "shape":
        child[A.A] = torch.ones(256, 256)
    elif mutation == "dtype":
        child[A.A] = child[A.A].double()
    else:
        child[A.A][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        A.compare_states(parent, child)


def test_inherited_source_pins_unchanged():
    assert A.MEAN_RUNNER_SHA == "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
    assert (
        A.HISTORICAL_RECIPE_SHA
        == "e454c52a315acb389b2bda3c126367e13df63b96230f9f698517705b531d51da"
    )
    assert A.SCORER_SHA == A.BUDGET.SCORER_SHA
    assert A.BUDGET.INPUTS == A.OLD.INPUTS == A.BASE.INPUTS == {}


@pytest.mark.parametrize(
    "key,value",
    [
        ("exit_code", 1),
        ("exit_code", False),
        ("terminal_completion_observed_by_primary", False),
        ("summary_sha256", "different"),
        ("stdout_sha256", "different"),
    ],
)
def test_primary_terminal_receipt_binding(key, value):
    receipt = {
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "summary_sha256": "summary",
        "stdout_sha256": "stdout",
    }
    A.execution_contract(receipt, "summary", "stdout")
    receipt[key] = value
    with pytest.raises(ValueError):
        A.execution_contract(receipt, "summary", "stdout")


def test_exact_scalar_prefix_and_correct_endpoint_alignment():
    historical, current = trace(2000), trace()
    current[2000]["pre_update_loss"] = 0.01
    result = A.prefix_diagnostic(current, historical, 0.01)
    assert result["all_pre_update_equal"] is True
    assert result["mismatch_count"] == 0
    assert result["first_mismatch_update"] is None
    assert result["boundary_loss_equal"] is True
    assert result["current_pre_update_2001_loss"] == 0.01
    assert current[1999]["pre_update_loss"] != 0.01


def test_prefix_mismatch_is_reported_and_never_changes_gate():
    current, historical = trace(), trace(2000)
    current[1]["pre_update_loss"] = 0.25
    current[1999]["pre_update_loss"] = 0.5
    result = A.prefix_diagnostic(current, historical, 0.9)
    assert result["mismatch_count"] == 2
    assert result["first_mismatch_update"] == 2
    assert result["boundary_loss_equal"] is False
    assert result["descriptive_only"] is True
    assert result["used_for_selection_or_updates"] is False
    assert A.screen_gate(gate_fixture(), {"checked": True})["primary_checks_passed"] is True


@pytest.mark.parametrize(
    "mutation", ["short_old", "short_new", "old_order", "new_order", "endpoint_order", "nan"]
)
def test_prefix_malformed_evidence_is_rejected(mutation):
    current, historical = trace(), trace(2000)
    if mutation == "short_old":
        historical.pop()
    elif mutation == "short_new":
        current.pop()
    elif mutation == "old_order":
        historical[0]["update"] = 2
    elif mutation == "new_order":
        current[0]["update"] = 2
    elif mutation == "endpoint_order":
        current[2000]["update"] = 2000
    else:
        current[5]["pre_update_loss"] = float("nan")
    with pytest.raises(ValueError):
        A.prefix_diagnostic(current, historical, 0.1)
