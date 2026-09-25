"""Synthetic CPU contracts only; no accepted checkpoint or corpus is executed."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "projection_refit", ROOT / "scripts/refit_membership_projection.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
RECIPE = ROOT / "configs/experiments/projection_only_refit_v1.json"


def _recipe():
    return json.loads(RECIPE.read_text())


@pytest.mark.parametrize(
    "key,value",
    [
        ("projection_updates", 501),
        ("loss_coefficient", 0.1),
        ("seed", 1730),
        ("threshold", 0.01),
        ("autocast", 0),
    ],
)
def test_fixed_recipe_refuses_changes_including_bool_type(key, value):
    recipe = _recipe()
    recipe[key] = value
    with pytest.raises(ValueError, match="fixed recipe"):
        runner._recipe(recipe)


def test_recipe_accepts_only_declared_optimizer():
    recipe = _recipe()
    assert runner._recipe(recipe) is recipe
    recipe["optimizer"]["name"] = "adamw_fused"
    with pytest.raises(ValueError, match="fixed recipe"):
        runner._recipe(recipe)


def test_strict_zero_subject_and_no_truncation():
    assert runner._prediction([9.0, 0.0, -1.0, 2.0], 1024) == [1027]
    assert len(runner._prediction([1.0] * 1025, 1024)) == 1024
    assert runner._prediction([-1.0] * 4, 1024) == []
    with pytest.raises(ValueError, match="nonfinite"):
        runner._prediction([float("nan")], 1024)


@pytest.mark.parametrize("size,compatible", [(0, False), (1, True), (506, True), (507, False)])
def test_serialization_bounds_preserve_raw_sets(size, compatible):
    value = runner._metrics(list(range(1025, 1025 + size)), [1025])
    assert value["set_size"] == size
    assert value["serialization_compatible"] is compatible


def test_metrics_false_positive_negative_and_empty():
    value = runner._metrics([1025, 1027], [1025, 1026])
    assert value["precision"] == value["recall"] == value["f1"] == 0.5
    assert value["false_positive_count"] == value["false_negative_count"] == 1
    assert runner._metrics([], [1025])["f1"] == 0


def test_gate_requires_strong_comparator_and_each_group():
    child = {
        "aggregate": {"serialization_compatible": 222, "exact_count": 202, "f1": 0.98},
        "groups": {
            g: {"exact_count": n}
            for g, n in {"COLOR": 103, "TYPE_single": 50, "TYPE_dual": 49}.items()
        },
    }
    assert runner._gate(child, {"finite": True})["primary_checks_passed"]
    assert not runner._gate(child, {"finite": 1})["primary_checks_passed"]
    for path, value in (("exact_count", 201), ("f1", 0.97), ("serialization_compatible", 221)):
        changed = copy.deepcopy(child)
        changed["aggregate"][path] = value
        assert not runner._gate(changed, {"finite": True})["primary_checks_passed"]
    child["groups"]["COLOR"]["exact_count"] = 102
    assert not runner._gate(child, {"finite": True})["primary_checks_passed"]


def _record(subject="a", dimension="TYPE", subject_id=1024):
    ids = [1, subject_id, 32, 34, 5, 1026, 2]
    return SimpleNamespace(
        subject=subject, dimension=dimension, input_ids=ids, labels=[-100] * 5 + ids[5:]
    )


def test_train_validation_separation_and_ordered_membership():
    train = [_record(), _record("b", subject_id=1025)]
    validation = [_record("c", subject_id=1027)]
    assert runner._split_contract(train, validation, 2, 1)["train"][1]["subject"] == "b"
    assert runner._training_membership(train)["queries"][0]["expected_set_ids"] == [1026]
    with pytest.raises(ValueError, match="leakage"):
        runner._split_contract(train, [train[0]], 2, 1)
    train[0].labels[0] = 1
    with pytest.raises(ValueError, match="teacher alignment"):
        runner._training_membership(train)


def test_immutable_outputs_allow_auditor_but_refuse_primary_remnants(tmp_path):
    (tmp_path / "independent-audit.py").write_text("# owned elsewhere")
    path = tmp_path / "summary.json"
    runner._refuse(path)
    runner._write(path, {"complete": False})
    with pytest.raises((ValueError, FileExistsError)):
        runner._refuse(path)
    with pytest.raises(FileExistsError):
        runner._write(path, {})
    other = tmp_path / "new"
    other.mkdir()
    (other / "train-membership.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        runner._refuse(other / "summary.json")


def test_wrong_helper_hash_rejected_before_import(tmp_path):
    path = tmp_path / "runs/learning/wide-first-choice-v1/summary.script.py"
    path.parent.mkdir(parents=True)
    path.write_text("raise AssertionError('should never execute')")
    with pytest.raises(ValueError, match="helper identity"):
        runner._load_helper(tmp_path)


def test_execution_import_boundary_retains_failure_summary(tmp_path):
    helper = SimpleNamespace(
        _modules=lambda: (_ for _ in ()).throw(ValueError("contaminated")),
        _failure_safe=lambda v: v,
        _sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="contaminated"):
        runner._execute(tmp_path, tmp_path / "summary.json", {}, helper, {"complete": False})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete"] is False and "contaminated" in summary["error"]
    assert "training.json" in summary["artifact_sha256"]


@pytest.fixture
def archived(tmp_path):
    source = ROOT / "tests/fixtures/projection_refit_runtime_source.zip"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
    )
    with zipfile.ZipFile(source) as archive:
        layers = archive.read("src/plm/model/layers.py").decode()
        for name in archive.namelist():
            if name.startswith("src/"):
                target = tmp_path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    return tmp_path, layers


@pytest.fixture
def loss_fn(archived):
    torch = pytest.importorskip("torch")
    tree = ast.parse(archived[1])
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prompt_set_loss"
    )
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "Tensor": torch.Tensor,
        "ENTITY_BASE": 1024,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "authenticated-archived-bce", "exec"),
        namespace,
    )
    return namespace["_prompt_set_loss"]


def _tiny():
    torch = pytest.importorskip("torch")
    torch.manual_seed(19)
    model = torch.nn.Module()
    model.config = SimpleNamespace(dim=4)
    model.token_embedding = torch.nn.Embedding(1029, 4)
    model.symmetric_relation_projection = torch.nn.Linear(4, 4, bias=False)
    model.decoder = torch.nn.Linear(4, 4)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == runner._WEIGHT)
    return model


def test_archived_loss_balance_masks_duplicates_subject_and_gradient(loss_fn):
    torch = pytest.importorskip("torch")
    z = torch.tensor([[99.0, 1.0, -2.0, 0.0, -1.0]], requires_grad=True)
    a = loss_fn(
        z, torch.tensor([[-100, 1, 32, 5, 1025, 1025, 2, 0]]), excluded_ids=torch.tensor([0])
    )
    expected = (
        torch.nn.functional.softplus(-z[0, 1]) + torch.nn.functional.softplus(z[0, 2:]).mean()
    ) * 0.5
    assert torch.equal(a, expected)
    a.backward()
    assert z.grad[0, 0] == 0 and z.grad[0, 1] < 0 and bool((z.grad[0, 2:] > 0).all())
    for labels in ([[2]], [[1024, 1025]], [[1025, 1026, 1027, 1028]]):
        with pytest.raises(ValueError):
            loss_fn(z, torch.tensor(labels), excluded_ids=torch.tensor([0]))


def test_fullbatch_updates_only_w_no_model_forward_and_endpoint_trace(loss_fn):
    torch = pytest.importorskip("torch")
    model = _tiny()
    model.forward = lambda *_: (_ for _ in ()).throw(
        AssertionError("transformer forward forbidden")
    )
    prompts = torch.tensor([[1, 1024, 32, 34, 5], [1, 1025, 33, 34, 5]])
    features = runner._features(model, prompts)
    assert not features[0].requires_grad and not features[1].requires_grad
    labels = torch.tensor([[1025, 2], [1026, 2]])
    before = runner._state_hashes(model)
    optimizer = torch.optim.AdamW(
        [model.symmetric_relation_projection.weight], lr=0.0003, weight_decay=0
    )
    initial = loss_fn(
        runner._head(model.symmetric_relation_projection.weight, features),
        labels,
        excluded_ids=features[2],
    ).item()
    report = {}
    runner._fit(model, features, labels, loss_fn, optimizer, 3, report)
    assert report["completed_updates"] == 3
    assert [r["update"] for r in report["history"]] == [1, 2, 3]
    assert report["initial_pre_update_loss"] == report["history"][0]["pre_update_loss"] == initial
    assert (
        report["final_post_update_loss"]
        == loss_fn(
            runner._head(model.symmetric_relation_projection.weight, features),
            labels,
            excluded_ids=features[2],
        ).item()
    )
    assert runner._frozen(before, runner._state_hashes(model))
    assert all(p.grad is None for n, p in model.named_parameters() if n != runner._WEIGHT)
    with torch.no_grad():
        model.decoder.weight[0, 0] += 1
    with pytest.raises(ValueError, match="frozen state"):
        runner._frozen(before, runner._state_hashes(model))


def test_nonfinite_second_update_retains_first_receipt_and_stops(loss_fn):
    torch = pytest.importorskip("torch")
    model = _tiny()
    features = runner._features(model, torch.tensor([[1, 1024, 32, 34, 5]]))
    count = 0

    def injected(*args, **kwargs):
        nonlocal count
        count += 1
        loss = loss_fn(*args, **kwargs)
        return loss if count == 1 else loss * float("nan")

    report = {}
    optimizer = torch.optim.AdamW([model.symmetric_relation_projection.weight], lr=0.0003)
    with pytest.raises(ValueError, match="nonfinite training loss"):
        runner._fit(model, features, torch.tensor([[1025, 2]]), injected, optimizer, 3, report)
    assert report["completed_updates"] == 1 and report["failed_update"] == 2
    assert len(report["history"]) == 1 and "failed_observation" in report


def test_exact_nested_optimizer_reload_check():
    torch = pytest.importorskip("torch")
    state = {
        "state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.zeros(2)}},
        "param_groups": [{"params": [0], "foreach": None}, {"params": []}],
    }
    same = copy.deepcopy(state)
    assert runner._same_state(state, same)
    same["state"][0]["exp_avg"][0] = 1
    assert not runner._same_state(state, same)
    same = copy.deepcopy(state)
    same["param_groups"][0]["foreach"] = False
    assert not runner._same_state(state, same)


def test_child_contract_rejects_causal_impersonation_and_wrong_lineage():
    identity = {"evaluator_version": runner._EVALUATOR}
    config = {"new_recipe": True}
    metadata = {
        "objective": runner._OBJECTIVE,
        "parent_checkpoint_sha256": runner._PARENT,
        "parent_training_steps": 2000,
        "projection_updates": 500,
    }
    raw = {
        "global_step": 500,
        "experiment_identity": identity,
        "config": config,
        "training_metadata": metadata,
        "corpus_identity": {},
        "split_hash": runner._SPLIT,
    }
    state = SimpleNamespace(global_step=500, metadata=raw)
    runner._validate_child(state, identity, config, metadata, {})
    for key, wrong in (("objective", "causal-next-token-v1"), ("parent_training_steps", 500)):
        bad = copy.deepcopy(metadata)
        bad[key] = wrong
        state.metadata = {**raw, "training_metadata": bad}
        with pytest.raises(ValueError):
            runner._validate_child(state, identity, config, bad, {})


def test_archived_tiny_model_parity_and_checkpoint_optimizer_roundtrip(archived, tmp_path):
    pytest.importorskip("torch")
    code = f"""
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, {str(archived[0] / "src")!r})
import torch
from plm.config import ModelConfig, OptimizerConfig
from plm.model.architecture import build_model
from plm.model.layers import _prompt_set_loss
from plm.training.optim import create_optimizer
from plm.training.checkpoint import save_checkpoint, load_checkpoint
script_path = {str(ROOT / "scripts/refit_membership_projection.py")!r}
spec = importlib.util.spec_from_file_location("refit", script_path)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
torch.manual_seed(1729)
config = ModelConfig(vocab_size=1029, dim=16, n_layers=1, n_heads=4,
                     n_kv_heads=2, ffn_multiple_of=8,
                     max_seq_len=16, dtype="fp32", symmetric_relation_loss_weight=1.0)
model = build_model(config).eval()
for n,p in model.named_parameters(): p.requires_grad_(n == r._WEIGHT)
prompts = torch.tensor([[1,1024,32,34,5],[1,1025,33,34,5]])
features = r._features(model,prompts)
with torch.no_grad():
    expected = model(prompts).symmetric_relation_logits
    assert torch.equal(r._head(model.symmetric_relation_projection.weight,features), expected)
with torch.autocast("cpu",dtype=torch.bfloat16):
    assert r._head(model.symmetric_relation_projection.weight,features).dtype == torch.float32
opt = create_optimizer(model, OptimizerConfig(name="adamw",weight_decay=0.0))
assert len(opt.param_groups[0]["params"]) == 1 and not opt.param_groups[1]["params"]
r._fit(model,features,torch.tensor([[1025,2],[1026,2]]),_prompt_set_loss,opt,2,{{}})
path = Path({str(tmp_path / "synthetic.pt")!r})
save_checkpoint(path,model,optimizer=opt,global_step=2,config=config)
child = build_model(config).eval()
for n,p in child.named_parameters(): p.requires_grad_(n == r._WEIGHT)
childopt = create_optimizer(child, OptimizerConfig(name="adamw",weight_decay=0.0))
load_checkpoint(path,child,optimizer=childopt,restore_rng=False)
assert r._state_hashes(model) == r._state_hashes(child)
assert r._same_state(opt.state_dict(),childopt.state_dict())
assert childopt.param_groups[0]["params"][0] is child.symmetric_relation_projection.weight
print("archived synthetic head/BCE/checkpoint contracts passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_preflight_rejects_wrong_plan_before_consuming_reports(tmp_path):
    helper = SimpleNamespace(
        _preflight=lambda *_: {"inputs": {}},
        _bind=lambda p, digest, inputs: runner._require(
            hashlib.sha256(p.read_bytes()).hexdigest() == digest, "input identity"
        ),
    )
    plan = tmp_path / "plan.md"
    plan.write_text("wrong")
    with pytest.raises(ValueError, match="input identity"):
        runner._preflight(tmp_path, plan, RECIPE, helper, tmp_path / "helper.py")


def test_partial_evaluation_preserves_failed_nonfinite_head(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

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
    model = _tiny()
    model.token_embedding = torch.nn.Embedding(2049, 4)
    records = [_record(), _record("b", subject_id=1025)]
    old = [{**runner._query(r), "group": "TYPE_dual", "expected_set_ids": [1026]} for r in records]
    original = runner._head
    count = 0

    def injected(*args):
        nonlocal count
        count += 1
        result = original(*args)
        if count == 2:
            result[0, 1] = float("nan")
        return result

    monkeypatch.setattr(runner, "_head", injected)
    output = tmp_path / "child.json"
    with pytest.raises(ValueError, match="nonfinite"):
        runner._evaluate(model, records, old, helper, output, "child", batch_size=1)
    report = json.loads(output.read_text())
    assert report["complete"] is False and report["query_count"] == len(report["responses"]) == 1
    assert report["failed_observation"]["offset"] == 1
    assert report["failed_observation"]["standalone_logits"][0][1] == {"nonfinite_float": "nan"}


def test_paired_changes_and_groups():
    rows = [
        {"group": "COLOR", "expected_set_ids": [1025], "metrics": {"exact": True}},
        {"group": "TYPE_dual", "expected_set_ids": [1025], "metrics": {"exact": False}},
    ]
    result = runner._paired(rows, [[1026], [1025]])
    assert result["gains"] == result["losses"] == 1
    assert result["groups"]["COLOR"] == {"gains": 1, "losses": 0}
    assert result["groups"]["TYPE_single"] == {"gains": 0, "losses": 0}
