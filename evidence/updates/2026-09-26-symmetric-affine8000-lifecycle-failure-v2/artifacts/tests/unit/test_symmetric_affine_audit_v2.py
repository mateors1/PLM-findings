"""Independent affine audit contracts on self-contained synthetic CPU evidence."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/learning/symmetric-affine8000-v2/independent-audit.py"
SPEC = importlib.util.spec_from_file_location("independent_affine_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture(autouse=True)
def isolate_inputs():
    AUDIT.INPUTS.clear()
    AUDIT.RUNTIME_ALLOWED.clear()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    AUDIT.INPUTS.clear()
    AUDIT.RUNTIME_ALLOWED.clear()


def trace():
    return [
        {
            "update": i,
            "pre_update_loss": AUDIT.INITIAL_LOSS,
            "gradient_finite": True,
            "parameters_finite": True,
        }
        for i in range(1, 8001)
    ]


def state():
    parent = {f"original_{i}": torch.tensor([float(i)]) for i in range(93)}
    child = {k: v.clone() for k, v in parent.items()}
    child.update({n: torch.ones(shape, dtype=torch.float32) for n, shape in AUDIT.SHAPES.items()})
    return parent, child


def optimizer():
    options = {
        "lr": 0.0003,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "fused": None,
        "foreach": None,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
    }
    payload = {
        "state": {},
        "param_groups": [options | {"params": [0, 1]}, options | {"params": [2]}],
    }
    for index, name in enumerate(AUDIT.PARAMETERS):
        payload["state"][index] = {
            "step": torch.tensor(8000.0),
            "exp_avg": torch.zeros(AUDIT.SHAPES[name]),
            "exp_avg_sq": torch.ones(AUDIT.SHAPES[name]),
        }
    implementation = {
        "class": "torch.optim.adamw.AdamW",
        "defaults": options | {"weight_decay": 0.01},
        "groups": [
            options | {"parameter_names": list(AUDIT.PARAMETERS[:2])},
            options | {"parameter_names": [AUDIT.PARAMETERS[2]]},
        ],
    }
    return payload, implementation


def test_full_trace():
    AUDIT.loss_contract(trace(), AUDIT.INITIAL_LOSS, 0.001)


@pytest.mark.parametrize(
    "kind",
    [
        "short",
        "long",
        "order",
        "bool",
        "nan",
        "negative",
        "gradient",
        "parameters",
        "initial",
        "final",
    ],
)
def test_loss_corruption(kind):
    rows = trace()
    initial, final = AUDIT.INITIAL_LOSS, 0.001
    if kind == "short":
        rows.pop()
    elif kind == "long":
        rows.append(rows[-1])
    elif kind in ("order", "bool"):
        rows[0]["update"] = 2 if kind == "order" else True
    elif kind in ("nan", "negative"):
        rows[7777]["pre_update_loss"] = float("nan") if kind == "nan" else -1
    elif kind in ("gradient", "parameters"):
        rows[4000]["gradient_finite" if kind == "gradient" else "parameters_finite"] = False
    elif kind == "initial":
        initial = 0.2
    else:
        final = float("inf")
    with pytest.raises(ValueError):
        AUDIT.loss_contract(rows, initial, final)


def test_exact_96_tensor_contract():
    parent, child = state()
    original, initial, final = AUDIT.compare_states(parent, child)
    assert len(original) == 93 and len(initial) == len(final) == 96
    assert all(initial[n]["sha256"] != final[n]["sha256"] for n in AUDIT.PARAMETERS)


@pytest.mark.parametrize(
    "kind",
    ["parent_count", "missing", "extra", "frozen", "signed_zero", "dtype", "shape", "nan", "zero"],
)
def test_state_corruption(kind):
    parent, child = state()
    if kind == "parent_count":
        parent.pop("original_92")
    elif kind == "missing":
        child.pop(AUDIT.PARAMETERS[2])
    elif kind == "extra":
        child["extra"] = torch.zeros(1)
    elif kind == "frozen":
        child["original_0"][0] = 1
    elif kind == "signed_zero":
        child["original_0"][0] = -0.0
    elif kind == "dtype":
        child[AUDIT.PARAMETERS[1]] = child[AUDIT.PARAMETERS[1]].double()
    elif kind == "shape":
        child[AUDIT.PARAMETERS[2]] = torch.ones(3)
    elif kind == "nan":
        child[AUDIT.PARAMETERS[0]][0, 0, 0] = float("nan")
    else:
        for name in AUDIT.PARAMETERS:
            child[name].zero_()
    with pytest.raises(ValueError):
        AUDIT.compare_states(parent, child)


def test_three_optimizer_states():
    payload, implementation = optimizer()
    result = AUDIT.optimizer_contract(payload, implementation)
    assert result["steps"] == dict.fromkeys(AUDIT.PARAMETERS, 8000.0)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "duplicate",
        "swapped_names",
        "step",
        "moment_shape",
        "moment_dtype",
        "nan",
        "negative_second",
        "lr",
        "fused",
        "decay",
        "class",
    ],
)
def test_optimizer_corruption(kind):
    payload, implementation = optimizer()
    if kind == "missing":
        payload["state"].pop(2)
    elif kind == "extra":
        payload["state"][5] = payload["state"][0]
    elif kind == "duplicate":
        payload["param_groups"][0]["params"] = [0, 0]
    elif kind == "swapped_names":
        implementation["groups"][0]["parameter_names"].reverse()
    elif kind == "step":
        payload["state"][2]["step"] = torch.tensor(7999.0)
    elif kind == "moment_shape":
        payload["state"][1]["exp_avg"] = torch.zeros(2)
    elif kind == "moment_dtype":
        payload["state"][2]["exp_avg"] = torch.zeros(2, dtype=torch.float64)
    elif kind == "nan":
        payload["state"][0]["exp_avg"][0, 0, 0] = float("nan")
    elif kind == "negative_second":
        payload["state"][2]["exp_avg_sq"][0] = -1
    elif kind == "class":
        implementation["class"] = "Adam"
    else:
        key, value = {"lr": ("lr", 0.1), "fused": ("fused", True), "decay": ("weight_decay", 0.1)}[
            kind
        ]
        payload["param_groups"][0][key] = value
        implementation["groups"][0][key] = value
    with pytest.raises(ValueError):
        AUDIT.optimizer_contract(payload, implementation)


@pytest.mark.parametrize("value", [True, False, "missing"])
@pytest.mark.parametrize("location", ["group", "defaults"])
def test_optimizer_foreach_fixed_null(value, location):
    payload, implementation = optimizer()
    mappings = (
        [implementation["defaults"]]
        if location == "defaults"
        else [payload["param_groups"][0], implementation["groups"][0]]
    )
    for mapping in mappings:
        if value == "missing":
            mapping.pop("foreach")
        else:
            mapping["foreach"] = value
    with pytest.raises(ValueError):
        AUDIT.optimizer_contract(payload, implementation)


def gate_child():
    return {
        "aggregate": {
            "exact_count": 217,
            "f1": 0.9998843626799785,
            "serialization_compatible": 222,
        },
        "groups": {
            g: {"exact_count": n}
            for g, n in (("COLOR", 103), ("TYPE_single", 51), ("TYPE_dual", 63))
        },
    }


def test_gate_pass_keeps_acceptance_false():
    gate = AUDIT.screen_gate(gate_child(), dict.fromkeys(AUDIT.INVARIANTS, True))
    assert gate["primary_checks_passed"] and not gate["accepted"]


@pytest.mark.parametrize(
    "kind", ["exact", "f1", "serial", "color", "single", "dual", "invariant", "missing", "extra"]
)
def test_gate_noncompensatory(kind):
    child = gate_child()
    invariants = dict.fromkeys(AUDIT.INVARIANTS, True)
    if kind == "exact":
        child["aggregate"]["exact_count"] = 216
    elif kind == "f1":
        child["aggregate"]["f1"] -= 1e-9
    elif kind == "serial":
        child["aggregate"]["serialization_compatible"] = 221
    elif kind in ("color", "single", "dual"):
        group, count = {
            "color": ("COLOR", 102),
            "single": ("TYPE_single", 50),
            "dual": ("TYPE_dual", 61),
        }[kind]
        child["groups"][group]["exact_count"] = count
    elif kind == "invariant":
        invariants["reload_exact"] = False
    elif kind == "missing":
        invariants.pop("reload_exact")
    else:
        invariants["fabricated"] = True
    assert not AUDIT.screen_gate(child, invariants)["primary_checks_passed"]


def test_self_mask_strict_zero_and_raw_serialization():
    values = [-1.0] * 1025
    values[0], values[1], values[2] = 100.0, 0.0, 1.0
    assert AUDIT.BASE.predict(values, 1024) == [1026]
    assert not AUDIT.BASE.metrics([], [1026])["serialization_compatible"]
    assert not AUDIT.BASE.metrics(list(range(1025, 1532)), [1026])["serialization_compatible"]
    assert AUDIT.BASE.metrics([1026], [1026])["exact"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 1, 0.1])
def test_invalid_score_scalar(value):
    values = [-1.0] * 1025
    values[500] = value
    with pytest.raises(ValueError):
        AUDIT.BASE.predict(values, 1024)


def receipt():
    return {
        "passed": True,
        "gpu_used": False,
        "experiment_training_executed": False,
        "exit_code": 0,
        "terminal_completion_observed": True,
        "tests_passed": 2,
        "tests_skipped": 0,
        "test_dependencies": {"a.py": "a" * 64},
        "input_adapter_identity": AUDIT.INPUT_ADAPTER,
        "input_amendment_sha256": AUDIT.AMENDMENT_SHA,
        "repair_declaration_sha256": AUDIT.REPAIR_SHA,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("gpu_used", True),
        ("experiment_training_executed", True),
        ("exit_code", False),
        ("exit_code", 1),
        ("terminal_completion_observed", False),
        ("tests_passed", True),
        ("tests_passed", 0),
        ("tests_skipped", False),
        ("tests_skipped", 1),
        ("test_dependencies", {}),
    ],
)
def test_receipt_rejections(field, value):
    data = receipt()
    data[field] = value
    with pytest.raises(ValueError):
        AUDIT.receipt_contract(data)


def test_valid_receipt():
    AUDIT.receipt_contract(receipt())


def test_hash_mutation_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    monkeypatch.setattr(AUDIT, "dependency_allowlist", lambda: {"data.json"})
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")
    digest = AUDIT.sha(path)
    AUDIT.bind(path, digest)
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError):
        AUDIT.bind(path, digest)


def test_temporary_cpu_payload_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    monkeypatch.setattr(AUDIT, "dependency_allowlist", lambda: {"synthetic.pt"})
    parent, child = state()
    path = tmp_path / "synthetic.pt"
    torch.save({"model": child, "optimizer": optimizer()[0]}, path)
    loaded = AUDIT.load_payload(path, AUDIT.sha(path))
    assert AUDIT.compare_states(parent, loaded["model"])[2] == AUDIT.state_manifest(child)


def test_torch_free_metadata_preflight():
    result = subprocess.run(
        [sys.executable, str(PATH), "--preflight-only"],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1"},
    )
    data = json.loads(result.stdout)
    assert data["preflight_passed"] and not data["torch_imported"]
    assert not data["candidate_audited"] and data["required_state_tensors"] == 96


def evidence_rows():
    vocabulary = [f"token_{i}" for i in range(2049)]
    rows = []
    for index in range(222):
        subject = 1024 + index
        truth = [1024 + (index + 1) % 1025]
        values = [-1.0] * 1025
        values[truth[0] - 1024] = 1.0
        rows.append(
            {
                "index": index,
                "subject": vocabulary[subject],
                "dimension": "COLOR",
                "group": "COLOR",
                "prompt_ids": [1, subject, 33, 34, 5],
                "expected_set_ids": truth,
                "selected_set_ids": truth,
                "symmetric_relation_logits": values,
                "metrics": AUDIT.BASE.metrics(truth, truth),
                "strict_separation": True,
            }
        )
    return rows, vocabulary


def test_pairing_alignment_and_metrics():
    rows, _ = evidence_rows()
    report = {
        "complete": True,
        "query_count": 222,
        "responses": rows,
        "product_token_ids": list(range(1024, 2049)),
    }
    result = AUDIT.historical_pairing(report, copy.deepcopy(report), "a" * 64)
    assert result["paired_vs_historical8000"]["gains"] == 0


@pytest.mark.parametrize("kind", ["columns", "order", "prompt", "labels", "prediction", "metric"])
def test_pair_corruption(kind):
    rows, _ = evidence_rows()
    report = {
        "complete": True,
        "query_count": 222,
        "responses": rows,
        "product_token_ids": list(range(1024, 2049)),
    }
    old = copy.deepcopy(report)
    if kind == "columns":
        old["product_token_ids"].reverse()
    elif kind == "order":
        old["responses"].reverse()
    elif kind == "prompt":
        old["responses"][0]["prompt_ids"][2] = 32
    elif kind == "labels":
        old["responses"][0]["expected_set_ids"] = [1040]
    elif kind == "prediction":
        old["responses"][0]["selected_set_ids"] = []
    else:
        old["responses"][0]["metrics"]["exact"] = False
    with pytest.raises(ValueError):
        AUDIT.historical_pairing(report, old, "a" * 64)


def test_membership_reconstruction_and_duplicates():
    rows, vocabulary = evidence_rows()
    records = AUDIT.records_from_evidence(rows, vocabulary)
    assert len(records) == 222
    assert AUDIT.query_labels(records[0], vocabulary)[1] == rows[0]["expected_set_ids"]
    rows[1].update(subject=rows[0]["subject"], prompt_ids=rows[0]["prompt_ids"])
    with pytest.raises(ValueError):
        AUDIT.records_from_evidence(rows, vocabulary)


def test_zero_replay_full_coverage():
    rows, _ = evidence_rows()
    parent = {"responses": rows}
    digest = hashlib.sha256(AUDIT.fp32_bytes(rows)).hexdigest()
    report = {
        "complete": True,
        "batches": [],
        "query_count": 222,
        "dtype": "float32",
        "logits_shape": [222, 1025],
        "logits_sha256": digest,
        "parent_logits_sha256": digest,
        "residual_nonzero_count": 0,
        "residual_sha256": hashlib.sha256(bytes(222 * 1025 * 4)).hexdigest(),
        "batch_sizes": [8] * 27 + [6],
    }
    for offset in range(0, 222, 8):
        part = rows[offset : offset + 8]
        digest = hashlib.sha256(AUDIT.fp32_bytes(part)).hexdigest()
        report["batches"].append(
            {
                "indices": list(range(offset, offset + len(part))),
                "shape": [len(part), 1025],
                "logits_sha256": digest,
                "parent_logits_sha256": digest,
                "residual_nonzero_count": 0,
                "residual_sha256": hashlib.sha256(bytes(len(part) * 1025 * 4)).hexdigest(),
            }
        )
    AUDIT.zero_replay_contract(report, parent)
    report["batches"][-1]["shape"] = [8, 1025]
    with pytest.raises(ValueError):
        AUDIT.zero_replay_contract(report, parent)


def test_terminal_receipt_exact_bool_and_hashes():
    data = {
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "summary_sha256": "a",
        "stdout_sha256": "b",
    }
    AUDIT.execution_contract(data, "a", "b")
    data["exit_code"] = False
    with pytest.raises(ValueError):
        AUDIT.execution_contract(data, "a", "b")


def test_immutable_output_refusal(tmp_path):
    (tmp_path / "independent-audit.json").write_text("preserve", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(PATH), "--summary", str(tmp_path / "summary.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "immutable independent audit" in result.stderr
    assert (tmp_path / "independent-audit.json").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "relative",
    [
        "data/records.jsonl",
        "data/graph.sqlite",
        "data/processed/pokemon_v1_f1541479_20260924/records.jsonl",
        "runs/learning/symmetric-affine8000-v2/protected.json",
        "runs/learning/bilinear-budget8000-v1/unlisted.json",
    ],
)
def test_forbidden_input_rejected_before_hash(relative, monkeypatch):
    def forbidden(_):
        raise AssertionError("bytes opened before scope validation")

    monkeypatch.setattr(AUDIT, "sha", forbidden)
    with pytest.raises(ValueError, match="allowlist"):
        AUDIT.bind(ROOT / relative, "0" * 64)


def test_manifest_validates_all_paths_before_any_hash(monkeypatch):
    def forbidden(_):
        raise AssertionError("manifest started hashing before complete scope check")

    monkeypatch.setattr(AUDIT, "sha", forbidden)
    with pytest.raises(ValueError, match="allowlist"):
        AUDIT.hash_manifest(
            {
                str(ROOT / "scripts/refit_symmetric_affine.py"): "a" * 64,
                str(ROOT / "data/records.jsonl"): "b" * 64,
            }
        )


def test_partition_membership_rejects_swaps_overlap_and_protected():
    found = {"train": [], "validation": [], "test": []}
    for i in range(100):
        subject = f"PKM_SYNTHETIC_{i}"
        value = int.from_bytes(
            hashlib.sha256(f"1729:{subject}:TYPE".encode()).digest(), "big"
        ) / float(1 << 256)
        partition = "validation" if value < 0.1 else "test" if value < 0.2 else "train"
        found[partition].append({"subject": subject, "dimension": "TYPE"})
    selected = {"train": found["train"], "validation": found["validation"]}
    AUDIT.partition_contract(selected)
    for changed in (
        {"train": selected["validation"], "validation": selected["train"]},
        {"train": selected["train"] + selected["validation"], "validation": selected["validation"]},
        {"train": selected["train"] + found["test"], "validation": selected["validation"]},
        found,
    ):
        with pytest.raises(ValueError):
            AUDIT.partition_contract(changed)


def test_full_synthetic_training_artifact_contract(tmp_path, monkeypatch):
    """Exercise metadata, adapter, states and optimizer together without real data."""
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    monkeypatch.setattr(AUDIT, "dependency_allowlist", lambda: {"train-membership.json"})
    vocabulary = [f"token_{i}" for i in range(2049)]
    vocabulary[32:34] = ["TYPE", "COLOR"]
    partitions = {"train": [], "validation": []}
    serial = 0
    for entity in range(930):
        partition = "train" if entity < 819 else "validation"
        while True:
            subject = f"PKM_FIXTURE_{serial}"
            serial += 1
            values = [
                int.from_bytes(hashlib.sha256(f"1729:{subject}:{d}".encode()).digest(), "big")
                / float(1 << 256)
                for d in ("TYPE", "COLOR")
            ]
            if (
                all(v >= 0.2 for v in values)
                if partition == "train"
                else all(v < 0.1 for v in values)
            ):
                break
        vocabulary[1024 + entity] = subject
        for dimension, token in (("TYPE", 32), ("COLOR", 33)):
            ids = [1, 1024 + entity, token, 34, 5, 2048, 2]
            partitions[partition].append(
                {
                    "subject": subject,
                    "dimension": dimension,
                    "input_ids": ids,
                    "labels": [-100] * 5 + ids[5:],
                    "targets": [vocabulary[2048]],
                }
            )
    partitions["train"].pop()
    assert len(partitions["train"]) == 1637 and len(partitions["validation"]) == 222
    queries = {n: [AUDIT.query_descriptor(r) for r in rows] for n, rows in partitions.items()}
    membership = {
        "partition": "train",
        "query_count": 1637,
        "queries": [
            {"index": i, **AUDIT.query_descriptor(r), "expected_set_ids": [2048]}
            for i, r in enumerate(partitions["train"])
        ],
    }
    membership_path = tmp_path / "train-membership.json"
    membership_path.write_text(json.dumps(membership), encoding="utf-8")
    monkeypatch.setattr(AUDIT, "MEMBERSHIP_SHA", AUDIT.sha(membership_path))
    parent_state, child_state = state()
    original, initial, final = AUDIT.compare_states(parent_state, child_state)
    parent_identity = {
        "source_commit": "c" * 40,
        "resolved_config_hash": "c" * 64,
        "evaluator_version": "plm-train-causal-v2",
        "environment": {},
        "checkpoint_hash": None,
        "seeds": [1729],
        "repetitions": 1,
    }
    parent = {
        "global_step": 2000,
        "model": parent_state,
        "experiment_identity": parent_identity,
        "corpus_identity": {"fixture": "only"},
        "training_metadata": {"model_config": {"dim": 256}},
    }
    parent["training_metadata"]["objective"] = "causal-next-token-v1"
    parent["training_metadata"]["model_config"] = {
        "dim": 256,
        "vocab_size": 2049,
        "max_seq_len": 512,
    }
    parent["experiment_identity"]["fixture"] = "only"
    parent["split_hash"] = AUDIT.SPLIT_SHA
    parent["config"] = {"seq_len": 512, "optimizer": {"lr": 0.001}}
    old_run = {
        "config": {
            "model": {"dim": 256, "vocab_size": 0, "max_seq_len": 128},
            "train": copy.deepcopy(parent["config"]),
        },
        "identity": copy.deepcopy(parent_identity),
    }
    parent_sidecar = {
        "checkpoint_hash": AUDIT.PARENT_SHA,
        "training_metadata": copy.deepcopy(parent["training_metadata"]),
        "split_hash": AUDIT.SPLIT_SHA,
    }
    inherited = {
        "queries": queries,
        "artifact_sha256": {
            n: str(i) * 64
            for i, n in enumerate(
                ("train-membership.json", "parent.json", "child.json", "training.json")
            )
        },
    }
    summary = {
        "queries": queries,
        "validation_update_points": [0, 8000],
        "train_query_order_sha256": AUDIT.formatted_hash(queries["train"]),
        "script_sha256": "a" * 64,
        "recipe_sha256": "b" * 64,
        "parent_training_identity": parent_identity,
        "corpus_identity": parent["corpus_identity"],
        "inherited_architecture": parent["training_metadata"]["model_config"],
        "workspace_provenance": {"commit": "d" * 40},
        "environment": {"synthetic": True},
        "split_hash": AUDIT.SPLIT_SHA,
        "artifact_sha256": {
            "checkpoint-final.pt": "e" * 64,
            "checkpoint-final.pt.json": "f" * 64,
            "zero-replay.json": "0" * 64,
        },
        "input_adapter_identity": AUDIT.INPUT_ADAPTER,
        "input_amendment_sha256": AUDIT.AMENDMENT_SHA,
        "repair_declaration_sha256": AUDIT.REPAIR_SHA,
    }
    summary["input_adapter"] = {
        "identity": AUDIT.INPUT_ADAPTER,
        "amendment_sha256": AUDIT.AMENDMENT_SHA,
        "encoding": "five prompt IDs; ascending unique targets; EOS; prompt labels -100",
        "inherited_corpus_identity": True,
        "whole_corpus_revalidated": False,
        "evidence_sha256": inherited["artifact_sha256"],
        "partition_sha256": {
            n: AUDIT.formatted_hash(
                [{k: r[k] for k in ("input_ids", "labels", "subject", "dimension")} for r in rows]
            )
            for n, rows in partitions.items()
        },
    }
    inherited["corpus_identity"] = parent["corpus_identity"]
    inherited["inherited_architecture"] = parent["training_metadata"]["model_config"]
    recipe = {"optimizer": {"name": "adamw"}}
    impl = {
        "runner_sha256": summary["script_sha256"],
        "scorer_sha256": AUDIT.AFFINE_SHA,
        "bilinear_helper_sha256": AUDIT.SCORER_SHA,
        "mean_helper_sha256": AUDIT.MEAN_RUNNER_SHA,
        "budget_helper_sha256": AUDIT.BUDGET_RUNNER_SHA,
        "source_archive_sha256": AUDIT.SOURCE_SHA,
        "configs_archive_sha256": AUDIT.CONFIGS_SHA,
        "input_adapter_identity": AUDIT.INPUT_ADAPTER,
        "input_amendment_sha256": AUDIT.AMENDMENT_SHA,
        "repair_declaration_sha256": AUDIT.REPAIR_SHA,
    }
    config = {
        "implementation_identity": impl,
        "inherited_parent_config": old_run["config"],
        "affine_residual_architecture": AUDIT.ARCHITECTURE,
        "affine_residual_recipe": recipe,
    }
    metadata = {
        "implementation_identity": impl,
        "objective": AUDIT.OBJECTIVE,
        "evaluator": AUDIT.EVALUATOR,
        "architecture": AUDIT.ARCHITECTURE,
        "model_config": parent["training_metadata"]["model_config"],
        "model_config_scope": "inherited archived base only; residual architecture is separate",
        "parent_checkpoint_sha256": AUDIT.PARENT_SHA,
        "parent_training_steps": 2000,
        "residual_updates": 8000,
        "trainable_parameters": list(AUDIT.PARAMETERS),
        "inference": AUDIT.INFERENCE,
        "train_query_order_sha256": summary["train_query_order_sha256"],
        "record_count": 1637,
        "recipe_sha256": summary["recipe_sha256"],
        "runner_sha256": summary["script_sha256"],
        "scorer_sha256": AUDIT.AFFINE_SHA,
        "bilinear_helper_sha256": AUDIT.SCORER_SHA,
        "parent_state_sha256": AUDIT.formatted_hash(original),
        "initial_state_sha256": AUDIT.formatted_hash(initial),
        "initial_pre_update_loss": AUDIT.INITIAL_LOSS,
        "final_post_update_loss": 0.001,
        "standard_serving_supported": False,
        "input_adapter_identity": AUDIT.INPUT_ADAPTER,
        "input_amendment_sha256": AUDIT.AMENDMENT_SHA,
        "repair_declaration_sha256": AUDIT.REPAIR_SHA,
    }
    identity = parent_identity | {
        "source_commit": summary["workspace_provenance"]["commit"],
        "resolved_config_hash": AUDIT.formatted_hash(config),
        "evaluator_version": AUDIT.EVALUATOR,
        "environment": summary["environment"]
        | impl
        | {"recipe_sha256": summary["recipe_sha256"], "architecture": AUDIT.ARCHITECTURE},
    }
    summary["child_training_identity"] = identity
    opt, opt_impl = optimizer()
    child = {
        "model": child_state,
        "optimizer": opt,
        "scheduler": None,
        "global_step": 8000,
        "config": config,
        "training_metadata": metadata,
        "experiment_identity": identity,
        "corpus_identity": parent["corpus_identity"],
        "split_hash": AUDIT.SPLIT_SHA,
    }
    training = {
        "complete": True,
        "objective": AUDIT.OBJECTIVE,
        "architecture": AUDIT.ARCHITECTURE,
        "completed_updates": 8000,
        "history": trace(),
        "initial_pre_update_loss": AUDIT.INITIAL_LOSS,
        "final_post_update_loss": 0.001,
        "initial_loss_replay": AUDIT.INITIAL_LOSS,
        "train_query_count": 1637,
        "transformer_forwards_during_fit": 0,
        "validation_labels_used": False,
        "trainable_parameters": list(AUDIT.PARAMETERS),
        "fit_complete": True,
        "phase": "complete",
        "optimizer_steps_attempted": 8000,
        "frozen_state_exact": True,
        "train_query_order_sha256": summary["train_query_order_sha256"],
        "checkpoint": str(tmp_path / "checkpoint-final.pt"),
        "checkpoint_sha256": "e" * 64,
        "checkpoint_sidecar_sha256": "f" * 64,
        "parent_state_before": original,
        "state_before": initial,
        "state_after": final,
        "state_reloaded": final,
        "residual_changed": True,
        "frozen_tensors_unchanged": True,
        "reload_exact": True,
        "optimizer_reload_exact": True,
        "optimizer": {"name": "adamw", "config": recipe["optimizer"], "groups": opt_impl["groups"]},
        "optimizer_implementation": {
            "module": "torch.optim.adamw",
            "class": "AdamW",
            "defaults": opt_impl["defaults"],
        },
        "optimizer_final": AUDIT.optimizer_contract(opt, opt_impl),
        "training_identity": identity,
        "zero_replay_sha256": "0" * 64,
        "zero_replay": {},
        "refit_seconds": 1.0,
    }
    reports = {
        "training.json": training,
        "run.json": {"config": config, "identity": identity, "training_metadata": metadata},
        "train-membership.json": membership,
        "checkpoint-final.pt.json": {},
        "zero-replay.json": {},
    }
    context = (
        inherited,
        {"initial_pre_update_loss": AUDIT.INITIAL_LOSS, "parent_state_before": original},
        membership,
        partitions,
        vocabulary,
        {},
        tmp_path / "parent.pt",
        parent_sidecar,
        old_run,
    )
    # Transport is synthetic; every tensor, optimizer, metadata and adapter contract executes.
    monkeypatch.setattr(
        AUDIT, "load_payload", lambda path, digest: parent if path.name == "parent.pt" else child
    )
    monkeypatch.setattr(AUDIT, "sidecar_matches", lambda *args: None)
    result = AUDIT.training_audit(summary, reports, recipe, context, tmp_path)
    assert result["child_tensor_count"] == 96 and result["updates_verified"] == 8000
    training["optimizer_steps_attempted"] = 8001
    with pytest.raises(ValueError, match="chronology"):
        AUDIT.training_audit(summary, reports, recipe, context, tmp_path)


def test_archive_runtime_binding_includes_packaging_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    monkeypatch.setattr(AUDIT, "dependency_allowlist", lambda: {"fixture.zip"})
    archive = tmp_path / "fixture.zip"
    members = {
        "src/plm/__init__.py": b"# synthetic",
        "pyproject.toml": b"[project]\n",
        "uv.lock": b"version = 1\n",
        "configs/protocol/synthetic.yaml": b"fixture: true",
    }
    with zipfile.ZipFile(archive, "w") as stream:
        for name, data in members.items():
            stream.writestr(name, data)
            destination = tmp_path / "runtime" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    result = AUDIT.archive_inventory(archive, AUDIT.sha(archive), tmp_path / "runtime")
    assert set(result) == set(members)


def test_archive_forbidden_member_never_read(tmp_path, monkeypatch):
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    monkeypatch.setattr(AUDIT, "dependency_allowlist", lambda: {"fixture.zip"})
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("data/records.jsonl", "must stay unopened")

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden archive member was opened")

    monkeypatch.setattr(zipfile.ZipFile, "read", forbidden)
    with pytest.raises(ValueError, match="scope"):
        AUDIT.archive_inventory(archive, AUDIT.sha(archive), tmp_path / "runtime")


def readiness_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(AUDIT, "AUDITOR_TEST_DEPENDENCIES", ("auditor.py", "tests.py"))
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    monkeypatch.setattr(AUDIT, "__file__", str(tmp_path / "auditor.py"))
    monkeypatch.setattr(
        AUDIT,
        "dependency_allowlist",
        lambda: {"auditor.py", "tests.py", "stdout.txt", "receipt.json", "readiness.json"},
    )
    for name in ("auditor.py", "tests.py", "stdout.txt"):
        (tmp_path / name).write_text("synthetic " + name, encoding="utf-8")
    data = receipt() | {
        "tested_script_sha256": AUDIT.sha(tmp_path / "auditor.py"),
        "test_dependencies": {n: AUDIT.sha(tmp_path / n) for n in ("auditor.py", "tests.py")},
        "stdout": {"path": "stdout.txt", "sha256": AUDIT.sha(tmp_path / "stdout.txt")},
    }
    return data


def write_readiness(tmp_path, data):
    (tmp_path / "receipt.json").write_text(json.dumps(data), encoding="utf-8")
    contract = {
        "schema_version": 1,
        "experiment": "symmetric-affine8000-v2",
        "plan_sha256": AUDIT.PLAN_SHA,
        "input_amendment_sha256": AUDIT.AMENDMENT_SHA,
        "repair_declaration_sha256": AUDIT.REPAIR_SHA,
        "input_adapter_identity": AUDIT.INPUT_ADAPTER,
        "ready": True,
        **{
            key: {"path": name, "sha256": AUDIT.sha(tmp_path / name)}
            for key, name in (
                ("auditor", "auditor.py"),
                ("tests", "tests.py"),
                ("test_receipt", "receipt.json"),
            )
        },
    }
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    inputs = {
        str(tmp_path / n): AUDIT.sha(tmp_path / n)
        for n in ("auditor.py", "tests.py", "stdout.txt", "receipt.json", "readiness.json")
    }
    reference = {
        "path": "readiness.json",
        "sha256": AUDIT.sha(path),
        "readiness_only": True,
        "candidate_audited": False,
    }
    return reference, inputs


def test_readiness_complete_chain(tmp_path, monkeypatch):
    data = readiness_fixture(tmp_path, monkeypatch)
    reference, inputs = write_readiness(tmp_path, data)
    assert AUDIT.readiness_contract(reference, inputs) == AUDIT.sha(tmp_path / "receipt.json")


@pytest.mark.parametrize(
    "kind",
    [
        "missing_source",
        "wrong_source",
        "skip",
        "missing_stdout_input",
        "missing_contract_input",
        "wrong_adapter",
        "unknown_dependency",
    ],
)
def test_readiness_admission_rejects(tmp_path, monkeypatch, kind):
    data = readiness_fixture(tmp_path, monkeypatch)
    if kind == "missing_source":
        data.pop("tested_script_sha256")
    elif kind == "wrong_source":
        data["tested_script_sha256"] = "0" * 64
    elif kind == "skip":
        data["tests_skipped"] = 1
    elif kind == "wrong_adapter":
        data["input_adapter_identity"] = "wrong"
    elif kind == "unknown_dependency":
        data["test_dependencies"]["records.jsonl"] = "0" * 64
    reference, inputs = write_readiness(tmp_path, data)
    if kind == "missing_stdout_input":
        inputs.pop(str(tmp_path / "stdout.txt"))
    elif kind == "missing_contract_input":
        inputs.pop(str(tmp_path / "readiness.json"))
    with pytest.raises(ValueError):
        AUDIT.readiness_contract(reference, inputs)


def parent_fixture():
    train = {"seq_len": 512, "optimizer": {"name": "adamw", "lr": 0.0003}}
    model = {"dim": 256, "vocab_size": 0, "max_seq_len": 128, "n_layers": 8}
    resolved = model | {"vocab_size": 2049, "max_seq_len": 512}
    corpus = {"snapshot_hash": "a" * 64, "vocabulary_hash": "b" * 64}
    identity = corpus | {"evaluator_version": "plm-train-causal-v2", "seeds": [1729]}
    metadata = {"objective": "causal-next-token-v1", "model_config": resolved}
    parent = {
        "global_step": 2000,
        "config": copy.deepcopy(train),
        "corpus_identity": corpus,
        "training_metadata": metadata,
        "experiment_identity": identity,
        "split_hash": AUDIT.SPLIT_SHA,
    }
    sidecar = {
        "checkpoint_hash": AUDIT.PARENT_SHA,
        "training_metadata": copy.deepcopy(metadata),
        "split_hash": AUDIT.SPLIT_SHA,
    }
    run = {
        "config": {"train": train, "model": model, "seed": 1729},
        "identity": copy.deepcopy(identity),
    }
    inherited = {
        "corpus_identity": copy.deepcopy(corpus),
        "inherited_architecture": copy.deepcopy(resolved),
    }
    return parent, sidecar, run, inherited, 2049


def test_actual_train_config_schema_and_all_nine_predicates():
    values = parent_fixture()
    assert values[0]["config"] != values[2]["config"]
    assert values[0]["config"] == values[2]["config"]["train"]
    checks = AUDIT.parent_admission(*values)
    assert len(checks) == 9 and all(checks.values())


@pytest.mark.parametrize(
    "kind",
    [
        "root_config",
        "optimizer",
        "sequence",
        "step",
        "bool_step",
        "hash",
        "corpus",
        "sidecar_metadata",
        "objective",
        "model",
        "evaluator",
        "identity_corpus",
        "run_identity",
        "inherited_model",
        "split",
    ],
)
def test_each_parent_admission_predicate_remains_required(kind):
    parent, sidecar, run, inherited, width = parent_fixture()
    if kind == "root_config":
        parent["config"] = copy.deepcopy(run["config"])
    elif kind == "optimizer":
        parent["config"]["optimizer"]["lr"] = 0.1
    elif kind == "sequence":
        parent["config"]["seq_len"] = 256
    elif kind in ("step", "bool_step"):
        parent["global_step"] = 8000 if kind == "step" else True
    elif kind == "hash":
        sidecar["checkpoint_hash"] = "0" * 64
    elif kind == "corpus":
        inherited["corpus_identity"]["snapshot_hash"] = "0" * 64
    elif kind == "sidecar_metadata":
        sidecar["training_metadata"]["objective"] = "changed"
    elif kind in ("objective", "model"):
        for metadata in (parent["training_metadata"], sidecar["training_metadata"]):
            if kind == "objective":
                metadata["objective"] = "changed"
            else:
                metadata["model_config"]["dim"] = 128
    elif kind == "evaluator":
        parent["experiment_identity"]["evaluator_version"] = "changed"
        run["identity"]["evaluator_version"] = "changed"
    elif kind == "identity_corpus":
        parent["experiment_identity"].pop("snapshot_hash")
    elif kind == "run_identity":
        run["identity"]["seeds"] = [1730]
    elif kind == "inherited_model":
        inherited["inherited_architecture"]["dim"] = 128
    else:
        parent["split_hash"] = "0" * 64
    with pytest.raises(ValueError):
        AUDIT.parent_admission(parent, sidecar, run, inherited, width)


@pytest.mark.parametrize("value", [None, "0" * 64])
def test_v2_repair_receipt_binding(value):
    data = receipt()
    if value is None:
        data.pop("repair_declaration_sha256")
    else:
        data["repair_declaration_sha256"] = value
    with pytest.raises(ValueError):
        AUDIT.receipt_contract(data)
