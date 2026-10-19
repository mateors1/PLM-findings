"""Synthetic CPU-only audit fixtures; no real checkpoints or model forwards."""

import copy
import hashlib
import importlib.util
import struct
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "bilinear-residual-refit-v1/independent-audit.py"
SPEC = importlib.util.spec_from_file_location("bilinear_independent_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture
def states():
    parent = {f"frozen_{i}": torch.tensor([float(i)]) for i in range(91)}
    parent["token_embedding.weight"] = torch.ones(3, 2)
    parent["symmetric_relation_projection.weight"] = torch.eye(2)
    child = {name: value.clone() for name, value in parent.items()}
    child[AUDIT.A] = torch.ones(AUDIT.SHAPE, dtype=torch.float32)
    return parent, child


def test_original93_plus_residual94(states):
    original, initial, after = AUDIT.compare_states(*states)
    assert len(original) == 93 and len(initial) == len(after) == 94
    assert initial[AUDIT.A]["sha256"] == AUDIT.tensor_digest(torch.zeros(AUDIT.SHAPE))
    assert initial[AUDIT.A] != after[AUDIT.A]
    assert all(initial[k] == after[k] for k in original)


@pytest.mark.parametrize(
    "name", ["frozen_2", "token_embedding.weight", "symmetric_relation_projection.weight"]
)
def test_any_original_tensor_change_rejected(states, name):
    states[1][name].add_(1)
    with pytest.raises(ValueError, match="changed frozen"):
        AUDIT.compare_states(*states)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "wrong_shape", "wrong_dtype", "zero", "nan", "inf"]
)
def test_bad_residual_state_rejected(states, mutation):
    parent, child = states
    if mutation == "missing":
        child.pop(AUDIT.A)
    elif mutation == "extra":
        child["unplanned_bias"] = torch.zeros(1)
    elif mutation == "wrong_shape":
        child[AUDIT.A] = torch.ones(256, 256)
    elif mutation == "wrong_dtype":
        child[AUDIT.A] = child[AUDIT.A].double()
    elif mutation == "zero":
        child[AUDIT.A].zero_()
    else:
        child[AUDIT.A][0, 0, 0] = float(mutation)
    with pytest.raises(ValueError):
        AUDIT.compare_states(parent, child)


def optimizer_fixture():
    group = {
        "lr": 0.0003,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "fused": None,
        "amsgrad": False,
        "maximize": False,
    }
    groups = [{**group, "params": [7]}, {**group, "params": []}]
    optimizer = {
        "state": {
            7: {
                "step": torch.tensor(500.0),
                "exp_avg": torch.zeros(AUDIT.SHAPE),
                "exp_avg_sq": torch.ones(AUDIT.SHAPE),
            }
        },
        "param_groups": groups,
    }
    implementation = {
        "class": "torch.optim.adamw.AdamW",
        "groups": [{**group, "parameter_names": [AUDIT.A]}, {**group, "parameter_names": []}],
        "defaults": group,
    }
    return optimizer, implementation


def test_named_three_dimensional_optimizer_state():
    AUDIT.optimizer_contract(*optimizer_fixture())


@pytest.mark.parametrize(
    "mutation", ["name", "shape", "step", "negative_moment", "extra_state", "lr", "fused"]
)
def test_optimizer_mutations_rejected(mutation):
    optimizer, implementation = optimizer_fixture()
    if mutation == "name":
        implementation["groups"][0]["parameter_names"] = ["symmetric_relation_projection.weight"]
    elif mutation == "shape":
        optimizer["state"][7]["exp_avg"] = torch.zeros(256, 256)
    elif mutation == "step":
        optimizer["state"][7]["step"] = torch.tensor(499)
    elif mutation == "negative_moment":
        optimizer["state"][7]["exp_avg_sq"][0, 0, 0] = -1
    elif mutation == "extra_state":
        optimizer["state"][8] = {}
    elif mutation == "lr":
        optimizer["param_groups"][0]["lr"] = 0.001
    else:
        optimizer["param_groups"][0]["fused"] = True
    with pytest.raises(ValueError):
        AUDIT.optimizer_contract(optimizer, implementation)


@pytest.mark.parametrize(
    "key,value",
    [
        ("dimensions", ["COLOR", "TYPE"]),
        ("dimension_token_ids", [33, 32]),
        ("shape", [2, 256]),
        ("score_order", "folded-diagonal"),
        ("initialization", "random"),
        ("id", "causal-next-token-v1"),
    ],
)
def test_architecture_cannot_be_relabeled(key, value):
    descriptor = copy.deepcopy(AUDIT.ARCHITECTURE_DESCRIPTOR)
    descriptor[key] = value
    with pytest.raises(ValueError):
        AUDIT.architecture_contract(descriptor)


def trace_fixture():
    return [
        {
            "update": i,
            "pre_update_loss": AUDIT.INITIAL_LOSS,
            "gradient_finite": True,
            "parameters_finite": True,
        }
        for i in range(1, 501)
    ]


def test_initial_endpoint_and_complete_trace():
    AUDIT.loss_contract(trace_fixture(), AUDIT.INITIAL_LOSS, 0.001)


@pytest.mark.parametrize(
    "mutation",
    ["short", "bad_order", "bad_initial", "nonfinite_final", "negative_final", "bad_gradient"],
)
def test_loss_trace_corruption_rejected(mutation):
    trace = trace_fixture()
    initial, final = AUDIT.INITIAL_LOSS, 0.001
    if mutation == "short":
        trace.pop()
    elif mutation == "bad_order":
        trace[2]["update"] = 2
    elif mutation == "bad_initial":
        initial = 0.01
    elif mutation == "nonfinite_final":
        final = float("nan")
    elif mutation == "negative_final":
        final = -0.1
    else:
        trace[0]["gradient_finite"] = False
    with pytest.raises(ValueError):
        AUDIT.loss_contract(trace, initial, final)


def test_reused_independent_helpers_have_no_global_mutation():
    assert AUDIT.BASE.INPUTS == {}
    values = [0.0] * 1025
    values[1] = 1.0
    assert AUDIT.BASE.predict(values, 1024) == [1025]
    assert AUDIT.BASE.INPUTS == {}


def zero_fixture():
    parent = {"responses": [{"symmetric_relation_logits": [float(i)] * 1025} for i in range(222)]}
    data = b"".join(
        struct.pack("<1025f", *r["symmetric_relation_logits"]) for r in parent["responses"]
    )
    report = {
        "complete": True,
        "query_count": 222,
        "dtype": "float32",
        "logits_shape": [222, 1025],
        "logits_sha256": hashlib.sha256(data).hexdigest(),
        "parent_logits_sha256": hashlib.sha256(data).hexdigest(),
        "residual_sha256": hashlib.sha256(bytes(len(data))).hexdigest(),
        "residual_nonzero_count": 0,
        "batch_sizes": [8] * 27 + [6],
        "batches": [],
    }
    for offset in range(0, 222, 8):
        length = min(8, 222 - offset)
        part = data[offset * 4100 : (offset + length) * 4100]
        report["batches"].append(
            {
                "indices": list(range(offset, offset + length)),
                "shape": [length, 1025],
                "logits_sha256": hashlib.sha256(part).hexdigest(),
                "parent_logits_sha256": hashlib.sha256(part).hexdigest(),
                "residual_sha256": hashlib.sha256(bytes(len(part))).hexdigest(),
                "residual_nonzero_count": 0,
            }
        )
    return report, parent


def test_complete_zero_A_replay_hashes():
    result = AUDIT.zero_replay_contract(*zero_fixture())
    assert result["query_count"] == 222 and result["batch_sizes"][-1] == 6


@pytest.mark.parametrize(
    "mutation",
    ["global_hash", "batch_hash", "residual", "query_order", "batch_shape", "dtype", "short"],
)
def test_zero_replay_corruption_rejected(mutation):
    report, parent = zero_fixture()
    if mutation == "global_hash":
        report["logits_sha256"] = "0" * 64
    elif mutation == "batch_hash":
        report["batches"][0]["parent_logits_sha256"] = "0" * 64
    elif mutation == "residual":
        report["batches"][0]["residual_nonzero_count"] = 1
    elif mutation == "query_order":
        parent["responses"][0], parent["responses"][1] = (
            parent["responses"][1],
            parent["responses"][0],
        )
    elif mutation == "batch_shape":
        report["batches"][-1]["shape"] = [8, 1025]
    elif mutation == "dtype":
        report["dtype"] = "float64"
    else:
        report["batches"].pop()
    with pytest.raises(ValueError):
        AUDIT.zero_replay_contract(report, parent)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.1])
def test_head_values_must_be_exact_finite_fp32(bad):
    with pytest.raises(ValueError):
        AUDIT.fp32_bytes([{"symmetric_relation_logits": [bad] * 1025}])


def gate_fixture():
    return {
        "aggregate": {"serialization_compatible": 222, "exact_count": 202, "f1": 0.99},
        "groups": {
            "COLOR": {"exact_count": 103},
            "TYPE_single": {"exact_count": 50},
            "TYPE_dual": {"exact_count": 49},
        },
    }


def test_gate_requires_strict_improvement_over201():
    child = gate_fixture()
    assert AUDIT.screen_gate(child, {"checked": True})["primary_checks_passed"]
    child["aggregate"]["exact_count"] = 201
    assert not AUDIT.screen_gate(child, {"checked": True})["primary_checks_passed"]


@pytest.mark.parametrize("mutation", ["group", "f1", "serialized", "execution"])
def test_gate_other_constraints_not_hidden_by_total_exact(mutation):
    child = gate_fixture()
    invariants = {"checked": True}
    if mutation == "group":
        child["groups"]["COLOR"]["exact_count"] = 102
    elif mutation == "f1":
        child["aggregate"]["f1"] = 0.97
    elif mutation == "serialized":
        child["aggregate"]["serialization_compatible"] = 221
    else:
        invariants["checked"] = False
    result = AUDIT.screen_gate(child, invariants)
    assert not result["primary_checks_passed"] and not result["accepted"]


@pytest.mark.parametrize(
    "key", ["parent_state_before", "state_before", "state_after", "state_reloaded"]
)
def test_state_receipt_byte_corruption_rejected(states, key):
    original, initial, final = AUDIT.compare_states(*states)
    training = {
        "parent_state_before": original,
        "state_before": initial,
        "state_after": final,
        "state_reloaded": final,
        "residual_changed": True,
        "frozen_tensors_unchanged": True,
        "reload_exact": True,
        "optimizer_reload_exact": True,
    }
    AUDIT.state_receipts_contract(training, original, initial, final)
    training = copy.deepcopy(training)
    training[key]["token_embedding.weight"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        AUDIT.state_receipts_contract(training, original, initial, final)


@pytest.mark.parametrize(
    "key,value",
    [
        ("objective", "causal-next-token-v1"),
        ("evaluator", "plm-projection-refit-screen-v1"),
        ("parent_checkpoint_sha256", "0" * 64),
        ("parent_training_steps", 500),
        ("residual_updates", 2000),
        ("trainable_parameters", ["symmetric_relation_projection.weight"]),
    ],
)
def test_new_lineage_rejects_old_objective_or_wrong_parent(key, value):
    metadata = {
        "architecture": AUDIT.ARCHITECTURE_DESCRIPTOR,
        "objective": AUDIT.OBJECTIVE,
        "evaluator": AUDIT.EVALUATOR,
        "parent_checkpoint_sha256": AUDIT.PARENT_SHA,
        "parent_training_steps": 2000,
        "residual_updates": 500,
        "trainable_parameters": [AUDIT.A],
        "record_count": 1637,
    }
    identity = {"evaluator_version": AUDIT.EVALUATOR}
    AUDIT.lineage_contract(metadata, identity)
    metadata[key] = value
    with pytest.raises(ValueError):
        AUDIT.lineage_contract(metadata, identity)


def test_main_refuses_existing_receipt_before_reading_data(tmp_path, monkeypatch):
    output = tmp_path / "independent-audit.json"
    output.write_text("preserved")
    monkeypatch.setattr("sys.argv", ["audit", "--summary", str(tmp_path / "summary.json")])
    monkeypatch.setattr(
        AUDIT, "audit", lambda _: pytest.fail("must not audit before overwrite guard")
    )
    with pytest.raises(ValueError, match="immutable"):
        AUDIT.main()
    assert output.read_text() == "preserved"
