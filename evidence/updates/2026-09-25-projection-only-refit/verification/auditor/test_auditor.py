"""Synthetic audit failures only; no real data, models, or optimization."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "runs/learning/projection-only-refit-v1/independent-audit.py"
)
spec = importlib.util.spec_from_file_location("projection_audit", SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def test_threshold_subject_and_zero_are_exact():
    z = [-1.0] * 1025
    z[0] = 100.0
    z[1] = 0.0
    z[2] = 0.25
    z[3] = -0.25
    assert a.predict(z, 1024) == [1026]
    assert a.predict([0.0] * 1025, 1024) == []
    assert len(a.predict([1.0] * 1025, 1024)) == 1024


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, 1, 0.1])
def test_bad_head_rejected(value):
    z = [0.0] * 1025
    z[5] = value
    with pytest.raises(ValueError):
        a.predict(z, 1024)


def test_prediction_cannot_use_labels():
    z = [-1.0] * 1025
    z[8] = 3.0
    predicted = a.predict(z, 1024)
    assert predicted == [1032]
    assert a.metrics(predicted, [1032])["exact"]
    assert not a.metrics(predicted, [1033])["exact"]
    assert a.predict(z, 1024) == predicted


@pytest.mark.parametrize(
    "size,valid", [(0, False), (1, True), (506, True), (507, False), (1024, False)]
)
def test_serialization_bounds_without_truncation(size, valid):
    ids = list(range(1025, 1025 + size))
    result = a.metrics(ids, [1025])
    assert result["set_size"] == size
    assert result["serialization_compatible"] is valid


def test_precision_recall_and_false_members():
    m = a.metrics([1025, 1026, 1027], [1025, 1028])
    assert m["precision"] == 1 / 3 and m["recall"] == 0.5 and m["f1"] == 0.4
    assert m["false_positive_count"] == 2 and m["false_negative_count"] == 1
    assert a.metrics([], [1025])["f1"] == 0


def test_separation_is_distinct_from_zero_threshold():
    z = [-3.0] * 1025
    z[0] = 999.0
    z[1] = -1.0
    d = a.separation(z, [1025], 1024)
    assert d["strict_separable"] and not d["zero_threshold_correct"]
    z[1] = 1.0
    z[2] = 0.0
    d = a.separation(z, [1025], 1024)
    assert d["strict_separable"] and d["zero_threshold_correct"]
    z[2] = 1.0
    assert not a.separation(z, [1025], 1024)["strict_separable"]


def trace():
    return [
        {
            "update": i,
            "pre_update_loss": 1.0 / i,
            "gradient_finite": True,
            "parameters_finite": True,
        }
        for i in range(1, 501)
    ]


def test_complete_trace():
    a.trace_contract(trace())


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate",
        "nan",
        "negative",
        "bad_gradient",
        "bad_parameter",
        "bool_update",
    ],
)
def test_trace_tampering(change):
    rows = trace()
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(rows[-1])
    elif change == "duplicate":
        rows[10]["update"] = 10
    elif change == "nan":
        rows[100]["pre_update_loss"] = math.nan
    elif change == "negative":
        rows[100]["pre_update_loss"] = -1.0
    elif change == "bad_gradient":
        rows[100]["gradient_finite"] = False
    elif change == "bad_parameter":
        rows[100]["parameters_finite"] = 1
    else:
        rows[0]["update"] = True
    with pytest.raises(ValueError):
        a.trace_contract(rows)


def states():
    import torch

    parent = {a.W: torch.zeros((256, 256)), "token_embedding.weight": torch.ones((10, 4))}
    child = {k: v.clone() for k, v in parent.items()}
    child[a.W][0, 0] = 0.25
    return parent, child


def test_state_hashes_prove_only_W_changed():
    parent, child = states()
    before, after = a.compare_states(parent, child)
    assert before[a.W] != after[a.W]
    assert before["token_embedding.weight"] == after["token_embedding.weight"]


@pytest.mark.parametrize("change", ["frozen", "same_W", "shape", "nan", "dtype", "extra"])
def test_state_tampering_rejected(change):
    import torch

    parent, child = states()
    if change == "frozen":
        child["token_embedding.weight"][0, 0] = 2
    elif change == "same_W":
        child[a.W] = parent[a.W].clone()
    elif change == "shape":
        child[a.W] = torch.zeros((4, 4))
    elif change == "nan":
        child[a.W][0, 0] = math.nan
    elif change == "dtype":
        child[a.W] = child[a.W].double()
    else:
        child["extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        a.compare_states(parent, child)


def optimizer():
    import torch

    group = {
        "params": [0],
        "lr": 0.0003,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
        "fused": None,
        "amsgrad": False,
    }
    return {
        "param_groups": [group, {**group, "params": []}],
        "state": {
            0: {
                "step": torch.tensor(500.0),
                "exp_avg": torch.zeros((256, 256)),
                "exp_avg_sq": torch.zeros((256, 256)),
            }
        },
    }


def test_optimizer_exact_final_state():
    a.optimizer_contract(optimizer())


@pytest.mark.parametrize(
    "change", ["step", "weight_decay", "lr", "fused", "extra_parameter", "bad_moment"]
)
def test_optimizer_tampering(change):
    opt = optimizer()
    if change == "step":
        opt["state"][0]["step"] -= 1
    elif change == "extra_parameter":
        opt["param_groups"][1]["params"] = [1]
    elif change == "bad_moment":
        opt["state"][0]["exp_avg_sq"][0, 0] = -1
    else:
        opt["param_groups"][0][change] = True if change == "fused" else 0.1
    with pytest.raises(ValueError):
        a.optimizer_contract(opt)


def test_split_is_deterministic_disjoint_and_preserves_order():
    records = [{"subject": f"PKM_{i}", "dimension": "TYPE"} for i in range(100)]
    split, digest = a.split_membership(records)
    assert all(split.values())
    assert len({r["subject"] for rows in split.values() for r in rows}) == 100
    assert all(
        [records.index(r) for r in rows] == sorted(records.index(r) for r in rows)
        for rows in split.values()
    )
    assert a.split_membership(copy.deepcopy(records))[1] == digest
    with pytest.raises(ValueError):
        a.split_membership([*records, records[0]])


@pytest.mark.parametrize(
    "initial,final", [(0.5, 0.1), (1.0, float("nan")), (1.0, -0.1), (True, 0.1)]
)
def test_bad_loss_endpoint(initial, final):
    with pytest.raises(ValueError):
        a.loss_endpoints(trace(), initial, final)


def test_loss_endpoints_need_not_monotonically_improve():
    a.loss_endpoints(trace(), 1.0, 2.0)


@pytest.mark.parametrize(
    "names", [[], ["token_embedding.weight"], [a.W, "token_embedding.weight"], [a.W, a.W]]
)
def test_named_trainable_manifest(names):
    with pytest.raises(ValueError):
        a.trainable_manifest(names)


def test_strong_gate_not_weaker_parent_dense_gate():
    aggregate = {
        "query_count": 222,
        "exact_count": 202,
        "f1": 0.98,
        "serialization_compatible": 222,
    }
    groups = {
        g: {"exact_count": n} for g, n in [("COLOR", 103), ("TYPE_single", 50), ("TYPE_dual", 49)]
    }
    child = {"aggregate": aggregate, "groups": groups}
    assert a.screen_gate(child, {"execution": True})["primary_checks_passed"]
    aggregate["exact_count"] = 201
    assert not a.screen_gate(child, {"execution": True})["primary_checks_passed"]
    aggregate["exact_count"] = 210
    groups["COLOR"]["exact_count"] = 102
    assert not a.screen_gate(child, {"execution": True})["primary_checks_passed"]
    groups["COLOR"]["exact_count"] = 103
    aggregate["serialization_compatible"] = 221
    assert not a.screen_gate(child, {"execution": True})["primary_checks_passed"]


def test_sidecar_cannot_substitute_metadata():
    payload = {
        "global_step": 500,
        "config": {"x": 1},
        "experiment_identity": {},
        "corpus_identity": {},
        "split_hash": "split",
        "training_metadata": {"objective": a.OBJECTIVE},
    }
    sidecar = {"schema_version": 1, "checkpoint_hash": "abc", **copy.deepcopy(payload)}
    a.sidecar_matches(payload, sidecar, "abc")
    sidecar["training_metadata"]["objective"] = "causal-next-token-v1"
    with pytest.raises(ValueError):
        a.sidecar_matches(payload, sidecar, "abc")


def evaluation_fixture():
    vocabulary = [f"R_{i}" for i in range(1024)] + [f"PKM_{i}" for i in range(1024, 2049)]
    records = []
    rows = []
    oldrows = []
    for i in range(222):
        group = "COLOR" if i < 103 else "TYPE_single" if i < 153 else "TYPE_dual"
        dimension = "COLOR" if group == "COLOR" else "TYPE"
        prompt = [1, 1024 + i, 33 if dimension == "COLOR" else 32, 34, 5]
        truth = [2048]
        z = [-1.0] * 1025
        z[-1] = 1.0
        z[i] = 5.0
        record = {
            "subject": vocabulary[1024 + i],
            "dimension": dimension,
            "input_ids": [*prompt, 2048, 2],
            "labels": [-100] * 5 + [2048, 2],
            "targets": [vocabulary[2048]],
        }
        row = {
            "index": i,
            **a.query_descriptor(record),
            "group": group,
            "expected_set_ids": truth,
            "symmetric_relation_logits": z,
            "selected_set_ids": truth,
            "metrics": a.metrics(truth, truth),
            "strict_separation": True,
        }
        records.append(record)
        rows.append(row)
        oldrows.append({**row, "composition": {"selected_set_ids": truth}})
    report = {
        "phase": "parent",
        "complete": True,
        "query_count": 222,
        "product_token_ids": list(range(1024, 2049)),
        "responses": rows,
        "head_seconds": 1.0,
        "aggregate": a.totals(rows),
        "groups": {g: a.totals([r for r in rows if r["group"] == g]) for g in a.GROUPS},
        "paired_vs_width8": {
            "gains": 0,
            "losses": 0,
            "groups": {g: {"gains": 0, "losses": 0} for g in a.GROUPS},
        },
        "exact_parent_replay": True,
    }
    return report, {"responses": oldrows}, records, vocabulary


def test_full_parent_dense_reconstruction():
    report, reference, records, vocabulary = evaluation_fixture()
    _, aggregate, groups = a.evaluation(report, "parent", reference, records, vocabulary)
    assert aggregate["exact_count"] == 222 and aggregate["f1"] == 1
    assert groups["COLOR"]["exact_count"] == 103


@pytest.mark.parametrize(
    "change",
    [
        "ids",
        "truth",
        "metrics",
        "group",
        "head",
        "row_order",
        "aggregate",
        "pair_counts",
        "missing_row",
    ],
)
def test_saved_evaluation_tampering(change):
    report, reference, records, vocabulary = evaluation_fixture()
    report = copy.deepcopy(report)
    if change == "ids":
        report["responses"][0]["selected_set_ids"] = [1025]
    elif change == "truth":
        report["responses"][0]["expected_set_ids"] = [1025]
    elif change == "metrics":
        report["responses"][0]["metrics"]["precision"] = 0.5
    elif change == "group":
        report["responses"][0]["group"] = "TYPE_dual"
    elif change == "head":
        report["responses"][0]["symmetric_relation_logits"][-1] = 0.5
    elif change == "row_order":
        report["responses"][0], report["responses"][1] = (
            report["responses"][1],
            report["responses"][0],
        )
    elif change == "aggregate":
        report["aggregate"]["exact_count"] = 221
    elif change == "pair_counts":
        report["paired_vs_width8"]["gains"] = 1
    else:
        report["responses"].pop()
    with pytest.raises(ValueError):
        a.evaluation(report, "parent", reference, records, vocabulary)


def test_child_empty_predictions_are_measured_and_gate_fails():
    parent, reference, records, vocabulary = evaluation_fixture()
    child = copy.deepcopy(parent)
    child["phase"] = "child"
    child["exact_parent_replay"] = False
    child["responses"][0]["symmetric_relation_logits"][-1] = -1.0
    child["responses"][0]["selected_set_ids"] = []
    child["responses"][0]["metrics"] = a.metrics([], [2048])
    child["responses"][0]["strict_separation"] = False
    child["aggregate"] = a.totals(child["responses"])
    child["groups"] = {
        g: a.totals([r for r in child["responses"] if r["group"] == g]) for g in a.GROUPS
    }
    child["paired_vs_width8"] = child["paired_vs_parent_dense"] = {
        "gains": 0,
        "losses": 1,
        "groups": {g: {"gains": 0, "losses": int(g == "COLOR")} for g in a.GROUPS},
    }
    a.evaluation(child, "child", reference, records, vocabulary, parent)
    gate = a.screen_gate(child, {"evidence": True})
    assert child["aggregate"]["exact_count"] == 221
    assert child["aggregate"]["false_negative_count"] == 1
    assert not gate["checks"]["serialization_compatible"]
    assert gate["accepted"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("threshold", 0.1),
        ("projection_updates", 501),
        ("loss_coefficient", 0.5),
        ("validation_batch_size", 1),
        ("gradient_clipping", 1.0),
        ("parent_training_steps", 2500),
    ],
)
def test_recipe_changes_rejected(key, value):
    import json

    recipe = json.loads(
        (SCRIPT.parents[3] / "configs/experiments/projection_only_refit_v1.json").read_text()
    )
    a.recipe_contract(recipe)
    recipe[key] = value
    with pytest.raises(ValueError):
        a.recipe_contract(recipe)


def test_cpu_payload_hash_binding(tmp_path, monkeypatch):
    import torch

    path = tmp_path / "synthetic.pt"
    torch.save({"model": {"weight": torch.ones((2, 2))}}, path)
    digest = a.sha(path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    payload = a.load_payload(path, digest)
    assert payload["model"]["weight"].device.type == "cpu"
    with pytest.raises(ValueError, match="hash mismatch"):
        a.load_payload(path, "0" * 64)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        a.load_payload(path, digest)


def test_train_membership_masks_and_deduplication():
    vocabulary = [f"R_{i}" for i in range(1024)] + [f"PKM_{i}" for i in range(1024, 2049)]
    record = {
        "subject": "PKM_1024",
        "dimension": "TYPE",
        "input_ids": [1, 1024, 32, 34, 5, 1025, 1025, 2],
        "labels": [-100] * 5 + [1025, 1025, 2],
        "targets": ["PKM_1025", "PKM_1025"],
    }
    assert a.query_labels(record, vocabulary)[1] == [1025]
    record["labels"][1] = 1024
    with pytest.raises(ValueError, match="loss mask"):
        a.query_labels(record, vocabulary)


def test_mismatched_train_target_mapping_rejected():
    vocabulary = [f"R_{i}" for i in range(1024)] + [f"PKM_{i}" for i in range(1024, 2049)]
    record = {
        "subject": "PKM_1024",
        "dimension": "TYPE",
        "input_ids": [1, 1024, 32, 34, 5, 1025, 2],
        "labels": [-100] * 5 + [1025, 2],
        "targets": ["PKM_1026"],
    }
    with pytest.raises(ValueError, match="target mapping"):
        a.query_labels(record, vocabulary)
