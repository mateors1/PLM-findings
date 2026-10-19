"""Portable synthetic CPU checks; no campaign, real weights, or GPU execution."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/refit_bilinear_residual.py"
ARCHIVED_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _module(SCRIPT, "bilinear_test_runner")


@pytest.fixture(scope="module")
def mean():
    return _module(ROOT / "scripts/refit_membership_projection.py", "bilinear_mean_helper")


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def archived(tmp_path):
    source = ROOT / "tests/fixtures/projection_refit_runtime_source.zip"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == ARCHIVED_SHA
    with zipfile.ZipFile(source) as archive:
        layers = archive.read("src/plm/model/layers.py").decode("utf-8")
        for name in archive.namelist():
            if name.startswith("src/"):
                target = tmp_path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    return tmp_path, layers


@pytest.fixture
def loss_fn(torch, archived):
    tree = ast.parse(archived[1])
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prompt_set_loss"
    )
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "Tensor": torch.Tensor,
        "ENTITY_BASE": 1024,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "archived-balanced-bce", "exec"),
        namespace,
    )
    return namespace["_prompt_set_loss"]


def _tiny(torch, dim=4, products=5):
    torch.manual_seed(1729)
    model = torch.nn.Module()
    model.config = SimpleNamespace(dim=dim)
    model.token_embedding = torch.nn.Embedding(1024 + products, dim)
    model.symmetric_relation_projection = torch.nn.Linear(dim, dim, bias=False)
    model.decoder = torch.nn.Linear(dim, dim)
    return model


def test_hand_computed_off_diagonal_scores_and_dimension_isolation(runner, torch):
    entities = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    parameter = torch.zeros(2, 2, 2)
    parameter[0, 0, 1] = 2
    actual = runner._residual(
        parameter, entities, torch.tensor([0, 0]), torch.tensor([32, 33]), math.sqrt(2)
    )
    expected = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]) / math.sqrt(2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameter[1, 0, 0] = 4
    changed = runner._residual(
        parameter, entities, torch.tensor([0, 0]), torch.tensor([32, 33]), math.sqrt(2)
    )
    assert torch.equal(changed[0], actual[0])
    torch.testing.assert_close(changed[1], torch.tensor([4.0, 0.0, 4.0]) / math.sqrt(2))


def test_zero_parameter_has_hand_computed_nonzero_cross_coordinate_gradient(runner, torch):
    entities = torch.eye(2)
    parameter = torch.zeros(2, 2, 2, requires_grad=True)
    value = runner._residual(
        parameter, entities, torch.tensor([0]), torch.tensor([32]), math.sqrt(2)
    )[0, 1]
    value.backward()
    expected = torch.zeros_like(parameter)
    expected[0, 0, 1] = expected[0, 1, 0] = 1 / (2 * math.sqrt(2))
    torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)


def test_antisymmetric_part_cancels_and_toy_swapped_scores_match(runner, torch):
    entities = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    parameter = torch.tensor([[[1.0, 2.0], [2.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]])
    arguments = (entities, torch.arange(3), torch.full((3,), 32), 2.0)
    expected = runner._residual(parameter, *arguments)
    skew = torch.tensor([[[0.0, 3.0], [-3.0, 0.0]], [[0.0, -2.0], [2.0, 0.0]]])
    assert torch.equal(runner._residual(parameter + skew, *arguments), expected)
    assert torch.equal(expected, expected.T)
    assert torch.count_nonzero(runner._residual(skew, *arguments)).item() == 0


@pytest.mark.parametrize("dimension", [31, 34, 127])
def test_unknown_dimension_rejected(runner, torch, dimension):
    with pytest.raises(ValueError):
        runner._residual(
            torch.zeros(2, 2, 2), torch.eye(2), torch.tensor([0]), torch.tensor([dimension]), 2.0
        )


@pytest.mark.parametrize("field", ["subjects", "dimensions"])
@pytest.mark.parametrize("dtype_name", ["float32", "bool"])
def test_noninteger_index_tensors_rejected(runner, torch, field, dtype_name):
    subjects, dimensions = torch.tensor([0]), torch.tensor([32])
    if field == "subjects":
        subjects = subjects.to(getattr(torch, dtype_name))
    else:
        dimensions = dimensions.to(getattr(torch, dtype_name))
    with pytest.raises(ValueError, match="integer subject/dimension"):
        runner._residual(torch.zeros(2, 2, 2), torch.eye(2), subjects, dimensions, 2.0)


def test_attach_freezes_parent_and_actual_head_zero_replays_without_forward(runner, mean, torch):
    model = _tiny(torch)
    parent_state = mean._state_hashes(model)
    runner._attach(model)
    parameter = model.symmetric_bilinear_residual
    assert parameter.shape == (2, 4, 4) and parameter.dtype == torch.float32
    assert torch.count_nonzero(parameter).item() == 0
    assert [name for name, p in model.named_parameters() if p.requires_grad] == [
        "symmetric_bilinear_residual"
    ]
    after = mean._state_hashes(model)
    assert set(after) == set(parent_state) | {"symmetric_bilinear_residual"}
    assert all(after[name] == digest for name, digest in parent_state.items())
    prompts = torch.tensor([[1, 1024, 32, 34, 5], [1, 1025, 33, 34, 5]])
    features = mean._features(model, prompts)
    expected = mean._head(model.symmetric_relation_projection.weight, features)
    # A research head must not accidentally evaluate the unchanged decoder forward.
    model.forward = lambda *_: (_ for _ in ()).throw(AssertionError("decoder forward forbidden"))
    actual = runner._head(model, features, prompts[:, 2], mean)
    assert torch.equal(actual, expected)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_output = runner._head(model, features, prompts[:, 2], mean)
    assert autocast_output.dtype == torch.float32 and torch.equal(autocast_output, expected)
    assert mean._state_hashes(model) == after
    with pytest.raises(ValueError, match="already attached"):
        runner._attach(model)
    assert mean._state_hashes(model) == after


def test_balanced_loss_through_head_preserves_masks_and_query_weight(runner, mean, torch, loss_fn):
    model = _tiny(torch)
    runner._attach(model)
    prompts = torch.tensor([[1, 1024, 32, 34, 5], [1, 1025, 33, 34, 5]])
    logits = runner._head(model, mean._features(model, prompts), prompts[:, 2], mean)
    logits.retain_grad()
    labels = torch.tensor([[1025, 1026, 1025, 2, -100], [1026, 2, -100, 0, 5]])
    actual = loss_fn(logits, labels, excluded_ids=prompts[:, 1] - 1024)
    softplus = torch.nn.functional.softplus
    expected = (
        0.5 * (softplus(-logits[0, [1, 2]]).mean() + softplus(logits[0, [3, 4]]).mean())
        + 0.5 * (softplus(-logits[1, 2]) + softplus(logits[1, [0, 3, 4]]).mean())
    ) / 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert logits.grad[0, 0] == logits.grad[1, 1] == 0
    assert model.symmetric_bilinear_residual.grad is not None
    assert all(
        p.grad is None
        for name, p in model.named_parameters()
        if name != "symmetric_bilinear_residual"
    )


def test_only_residual_updates_and_endpoint_trace_uses_custom_head(runner, mean, torch, loss_fn):
    model = _tiny(torch)
    runner._attach(model)
    prompts = torch.tensor([[1, 1024, 32, 34, 5], [1, 1025, 33, 34, 5]])
    features = mean._features(model, prompts)
    labels = torch.tensor([[1025, 2], [1026, 2]])
    before = mean._state_hashes(model)
    optimizer = torch.optim.AdamW([model.symmetric_bilinear_residual], lr=0.0003, weight_decay=0)
    initial = loss_fn(
        runner._head(model, features, prompts[:, 2], mean), labels, excluded_ids=features[2]
    ).item()
    receipt = {}
    runner._fit(model, features, prompts[:, 2], labels, loss_fn, optimizer, 3, receipt, mean)
    after = mean._state_hashes(model)
    assert [key for key in before if before[key] != after[key]] == ["symmetric_bilinear_residual"]
    assert receipt["completed_updates"] == 3
    assert [row["update"] for row in receipt["history"]] == [1, 2, 3]
    assert receipt["initial_pre_update_loss"] == receipt["history"][0]["pre_update_loss"] == initial
    assert (
        receipt["final_post_update_loss"]
        == loss_fn(
            runner._head(model, features, prompts[:, 2], mean), labels, excluded_ids=features[2]
        ).item()
    )
    assert all(
        p.grad is None
        for name, p in model.named_parameters()
        if name != "symmetric_bilinear_residual"
    )


def test_nonfinite_update_preserves_completed_work_and_stops(runner, mean, torch, loss_fn):
    model = _tiny(torch)
    runner._attach(model)
    prompts = torch.tensor([[1, 1024, 32, 34, 5]])
    features = mean._features(model, prompts)
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        value = loss_fn(*args, **kwargs)
        return value if count == 1 else value * float("nan")

    receipt = {}
    optimizer = torch.optim.AdamW([model.symmetric_bilinear_residual], lr=0.0003, weight_decay=0)
    with pytest.raises(ValueError, match="nonfinite"):
        runner._fit(
            model,
            features,
            prompts[:, 2],
            torch.tensor([[1025, 2]]),
            fail_second,
            optimizer,
            3,
            receipt,
            mean,
        )
    assert receipt["completed_updates"] == 1 and receipt["failed_update"] == 2
    assert len(receipt["history"]) == 1 and "failed_observation" in receipt


def test_archived_factory_checkpoint_optimizer_and_head_roundtrip(archived, tmp_path):
    pytest.importorskip("torch")
    code = f"""
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, {str(archived[0] / "src")!r})
import torch
from plm.config import ModelConfig, OptimizerConfig
from plm.model.architecture import build_model
from plm.model.layers import _prompt_set_loss
from plm.training.optim import create_optimizer
from plm.training.checkpoint import save_checkpoint, load_checkpoint
assert not torch.cuda.is_available()
def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
r = module({str(SCRIPT)!r}, "bilinear")
mean = module({str(ROOT / "scripts/refit_membership_projection.py")!r}, "mean")
torch.manual_seed(1729)
config = ModelConfig(vocab_size=1029, dim=16, n_layers=1, n_heads=4,
                     n_kv_heads=2, ffn_multiple_of=8, max_seq_len=16,
                     dtype="fp32", symmetric_relation_loss_weight=1.0)
model = r._factory(config, build_model, device="cpu")
assert [n for n,p in model.named_parameters() if p.requires_grad] == ["symmetric_bilinear_residual"]
before = mean._state_hashes(model)
prompts = torch.tensor([[1,1024,32,34,5],[1,1025,33,34,5]])
features = mean._features(model,prompts)
with torch.no_grad():
    old = model(prompts).symmetric_relation_logits
    assert torch.equal(r._head(model,features,prompts[:,2],mean), old)
opt = create_optimizer(model, OptimizerConfig(name="adamw",weight_decay=0.0))
owned = [p for group in opt.param_groups for p in group["params"]]
assert len(owned) == 1 and owned[0] is model.symmetric_bilinear_residual
r._fit(model,features,prompts[:,2],torch.tensor([[1025,2],[1026,2]]),_prompt_set_loss,opt,2,{{}},mean)
trained = mean._state_hashes(model)
assert [n for n in before if before[n] != trained[n]] == ["symmetric_bilinear_residual"]
path = Path({str(tmp_path / "synthetic-child.pt")!r})
save_checkpoint(path,model,optimizer=opt,global_step=2,config=config,
                training_metadata={{"synthetic_test":True}})
unprepared = build_model(config).eval()
assert len(model.state_dict()) == len(unprepared.state_dict()) + 1
try:
    load_checkpoint(path,unprepared,restore_rng=False)
except RuntimeError as exc:
    assert "symmetric_bilinear_residual" in str(exc)
else:
    raise AssertionError("unprepared base model silently loaded residual architecture")
child = r._factory(config, build_model, device="cpu")
childopt = create_optimizer(child, OptimizerConfig(name="adamw",weight_decay=0.0))
state = load_checkpoint(path,child,optimizer=childopt,restore_rng=False)
assert state.global_step == 2
assert mean._state_hashes(child) == trained
assert mean._same_state(opt.state_dict(),childopt.state_dict())
owned = [p for group in childopt.param_groups for p in group["params"]]
assert len(owned) == 1 and owned[0] is child.symmetric_bilinear_residual
with torch.no_grad():
    assert torch.equal(r._head(model,features,prompts[:,2],mean),
                       r._head(child,mean._features(child,prompts),prompts[:,2],mean))
    assert torch.equal(child(prompts).symmetric_relation_logits,old)
print("archived factory/zero replay/residual-only/checkpoint/optimizer contracts passed")
"""
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _record(subject="a", dimension="TYPE", subject_id=1024):
    dimension_id = {"TYPE": 32, "COLOR": 33}[dimension]
    ids = [1, subject_id, dimension_id, 34, 5, 1026, 2]
    return SimpleNamespace(
        subject=subject, dimension=dimension, input_ids=ids, labels=[-100] * 5 + ids[5:]
    )


def test_child_evaluation_retains_partial_raw_evidence(runner, mean, torch, tmp_path, monkeypatch):
    def safe(value):
        if isinstance(value, list):
            return [safe(v) for v in value]
        return (
            {"nonfinite_float": str(value)}
            if isinstance(value, float) and not math.isfinite(value)
            else value
        )

    helper = SimpleNamespace(
        _failure_safe=safe,
        _fp32=lambda values: runner._require(
            len(values) == 1025 and all(math.isfinite(v) for v in values), "nonfinite head"
        ),
    )
    model = _tiny(torch, products=1025)
    runner._attach(model)
    records = [_record(), _record("b", subject_id=1025)]
    wide = [
        {
            **mean._query(record),
            "group": "TYPE_dual",
            "expected_set_ids": [1026],
            "composition": {"selected_set_ids": [1026]},
        }
        for record in records
    ]
    original = runner._head
    calls = 0

    def invalid_second_batch(*args):
        nonlocal calls
        calls += 1
        values = original(*args)
        if calls == 2:
            values[0, 1] = float("nan")
        return values

    monkeypatch.setattr(runner, "_head", invalid_second_batch)
    output = tmp_path / "child.json"
    with pytest.raises(ValueError, match="nonfinite"):
        runner._evaluate(model, records, wide, mean, helper, output, batch_size=1)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert report["query_count"] == len(report["responses"]) == 1
    assert report["failed_observation"]["offset"] == 1
    assert report["failed_observation"]["standalone_logits"][0][1] == {"nonfinite_float": "nan"}


def _child_contract(runner):
    architecture = runner._architecture()
    implementation = {
        "runner_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "mean_helper_sha256": runner._MEAN,
        "source_archive_sha256": ARCHIVED_SHA,
        "configs_archive_sha256": (
            "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
        ),
    }
    identity = {"evaluator_version": runner._EVALUATOR}
    config = {
        "bilinear_residual_architecture": copy.deepcopy(architecture),
        "implementation_identity": copy.deepcopy(implementation),
    }
    metadata = {
        "architecture": copy.deepcopy(architecture),
        "implementation_identity": copy.deepcopy(implementation),
        "runner_sha256": implementation["runner_sha256"],
        "scorer_sha256": implementation["runner_sha256"],
        "objective": runner._OBJECTIVE,
        "evaluator": runner._EVALUATOR,
        "trainable_parameters": ["symmetric_bilinear_residual"],
        "residual_updates": 500,
        "parent_training_steps": 2000,
        "parent_checkpoint_sha256": (
            "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
        ),
    }
    raw = {
        "global_step": 500,
        "experiment_identity": identity,
        "config": config,
        "training_metadata": metadata,
        "corpus_identity": {"fixture": "synthetic"},
        "split_hash": "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d",
    }
    return (
        SimpleNamespace(global_step=500, metadata=raw),
        identity,
        config,
        metadata,
        raw["corpus_identity"],
    )


@pytest.mark.parametrize(
    "key,value",
    [("id", "wrong-architecture"), ("dimension_token_ids", [33, 32]), ("shape", [2, 256, 128])],
)
def test_child_rejects_wrong_architecture_even_when_metadata_and_config_agree(runner, key, value):
    state, identity, config, metadata, corpus = _child_contract(runner)
    runner._validate_child(state, identity, config, metadata, corpus)
    config["bilinear_residual_architecture"][key] = value
    metadata["architecture"][key] = value
    with pytest.raises(ValueError, match="architecture"):
        runner._validate_child(state, identity, config, metadata, corpus)


@pytest.mark.parametrize(
    "key,value",
    [
        ("objective", "causal-next-token-v1"),
        ("parent_checkpoint_sha256", "mean-refit-child"),
        ("trainable_parameters", ["symmetric_relation_projection.weight"]),
    ],
)
def test_child_rejects_objective_or_lineage_impersonation(runner, key, value):
    state, identity, config, metadata, corpus = _child_contract(runner)
    metadata[key] = value
    with pytest.raises(ValueError):
        runner._validate_child(state, identity, config, metadata, corpus)


@pytest.mark.parametrize("changed", ["scorer", "helper"])
def test_child_rejects_inconsistent_implementation_identity(runner, changed):
    state, identity, config, metadata, corpus = _child_contract(runner)
    runner._validate_child(state, identity, config, metadata, corpus)
    if changed == "scorer":
        metadata["scorer_sha256"] = "0" * 64
    else:
        metadata["implementation_identity"]["mean_helper_sha256"] = "0" * 64
        config["implementation_identity"]["mean_helper_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="implementation identity"):
        runner._validate_child(state, identity, config, metadata, corpus)


def test_zero_replay_exact_bytes_mixed_dimensions_final_short_batch(runner, mean, torch, tmp_path):
    model = _tiny(torch)
    runner._attach(model)
    records = [_record(), _record("b", "COLOR", 1025), _record("c", subject_id=1027)]
    parent = {"responses": [], "product_token_ids": list(range(1024, 1029))}
    for offset in range(0, len(records), 2):
        prompts = torch.tensor([r.input_ids[:5] for r in records[offset : offset + 2]])
        logits = mean._head(
            model.symmetric_relation_projection.weight, mean._features(model, prompts)
        )
        parent["responses"].extend({"symmetric_relation_logits": row} for row in logits.tolist())
    helper = SimpleNamespace(_failure_safe=lambda value: value)
    result = runner._zero_replay(
        model, records, parent, mean, helper, tmp_path / "zero.json", batch_size=2
    )
    assert result["complete"] and result["query_count"] == 3
    assert result["batch_sizes"] == [2, 1] and result["residual_nonzero_count"] == 0
    assert result["logits_sha256"] == result["parent_logits_sha256"]
    with torch.no_grad():
        model.symmetric_bilinear_residual[0, 0, 0] = 1
    failure_path = tmp_path / "wrong-zero.json"
    with pytest.raises(ValueError, match="zero-A"):
        runner._zero_replay(model, records, parent, mean, helper, failure_path, batch_size=2)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert not failure["complete"] and failure["failed_observation"]["offset"] == 0
    assert failure["failed_observation"]["residual"]


def test_child_predictions_do_not_depend_on_teacher_targets(runner, mean, torch, tmp_path):
    model = _tiny(torch, products=1025)
    runner._attach(model)
    with torch.no_grad():
        model.symmetric_bilinear_residual[0, 0, 1] = 0.25
    helper = SimpleNamespace(
        _failure_safe=lambda value: value,
        _fp32=lambda values: runner._require(
            len(values) == 1025 and all(math.isfinite(v) for v in values), "finite FP32 vector"
        ),
    )
    results = []
    for target in (1026, 1027):
        records = [_record(), _record("b", subject_id=1025), _record("c", "COLOR", 1028)]
        wide = []
        for record, group in zip(records, ["TYPE_single", "TYPE_dual", "COLOR"], strict=True):
            record.input_ids[5] = record.labels[5] = target
            wide.append(
                {
                    **mean._query(record),
                    "group": group,
                    "expected_set_ids": [target],
                    "composition": {"selected_set_ids": [target]},
                }
            )
        report = runner._evaluate(model, records, wide, mean, helper, tmp_path / f"{target}.json")
        assert report["complete"] and report["query_count"] == 3
        results.append(report["responses"])
    for left, right in zip(*results, strict=True):
        assert left["expected_set_ids"] != right["expected_set_ids"]
        assert left["symmetric_relation_logits"] == right["symmetric_relation_logits"]
        assert left["selected_set_ids"] == right["selected_set_ids"]


@pytest.mark.parametrize(
    "key,value", [("residual_updates", 501), ("threshold", 0.01), ("initialization", "random")]
)
def test_new_recipe_refuses_undeclared_changes(runner, mean, key, value):
    recipe = json.loads(
        (ROOT / "configs/experiments/bilinear_residual_refit_v1.json").read_text(encoding="utf-8")
    )
    inherited = json.loads(
        (ROOT / "configs/experiments/projection_only_refit_v1.json").read_text(encoding="utf-8")
    )
    runner._recipe(recipe, mean, inherited)
    recipe[key] = value
    with pytest.raises(ValueError, match="fixed residual recipe"):
        runner._recipe(recipe, mean, inherited)


def test_wrong_helper_rejected_before_execution(runner, tmp_path):
    path = tmp_path / "scripts/refit_membership_projection.py"
    path.parent.mkdir()
    path.write_text("raise AssertionError('unauthenticated code executed')", encoding="utf-8")
    with pytest.raises(ValueError, match="helper identity"):
        runner._load_mean(tmp_path)


def test_new_execution_failure_retains_immutable_training_and_summary(runner, mean, tmp_path):
    failure = ValueError("synthetic import boundary failure")

    def stop_before_import():
        raise failure

    helper = SimpleNamespace(
        _modules=stop_before_import,
        _failure_safe=lambda value: value,
        _sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    out = tmp_path / "summary.json"
    with pytest.raises(ValueError) as caught:
        runner._execute(tmp_path, out, {}, mean, helper, {"complete": False})
    assert caught.value is failure
    summary = json.loads(out.read_text(encoding="utf-8"))
    training_path = tmp_path / "training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    assert not summary["complete"] and not training["complete"]
    assert training["completed_updates"] == 0 and not training["history"]
    assert summary["error"] == training["error"] == repr(failure)
    assert (
        summary["artifact_sha256"]["training.json"]
        == hashlib.sha256(training_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="immutable"):
        mean._refuse(out)
