"""CPU contracts for sole worst loss on the unchanged bilinear parameterization."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/refit_bilinear_worst_boundary.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load(SCRIPT, "bilinear_worst_runner_test")


@pytest.fixture(scope="module")
def bilinear(runner):
    return runner._load_bilinear(ROOT)[0]


@pytest.fixture(scope="module")
def mean(bilinear):
    return bilinear._load_mean(ROOT)[0]


@pytest.fixture(scope="module")
def worst(runner):
    return runner._load_checked(
        ROOT,
        "scripts/refit_worst_boundary_projection.py",
        runner._LOSS_HELPER,
        "worst_callback_test",
    )[0]


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def recipes():
    return [
        json.loads((ROOT / "configs/experiments" / n).read_text())
        for n in ["bilinear_worst_boundary_v1.json", "bilinear_budget2000_v1.json"]
    ]


def test_recipe_changes_only_declared_objective_gate_and_diagnostic(runner):
    new, old = recipes()
    before = copy.deepcopy(old)
    assert runner._recipe(new, old) == new and old == before
    assert new["optimizer"] == old["optimizer"] and new["residual_updates"] == 2000
    assert "initial_training_loss" not in new
    assert new["mean_bce_diagnostic"] == {
        "initial_expected": 0.003822767175734043,
        "update_points": [0, 2000],
        "used_for_updates": False,
    }
    assert {k for k in old if k in new and old[k] != new[k]} == {
        "experiment",
        "objective",
        "evaluator",
        "strong_comparator_exact",
        "strong_comparator_f1",
        "strong_comparator_group_exact",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("residual_updates", 500),
        ("objective", "balanced-bce"),
        ("loss_coefficient", 0.5),
        ("threshold", 0.1),
        ("strong_comparator_exact", 201),
        ("strong_comparator_f1", 0.98),
        ("mean_bce_diagnostic", {"used_for_updates": True}),
        ("optimizer", {"lr": 0.001}),
    ],
)
def test_recipe_rejects_undeclared_changes(runner, field, value):
    new, old = recipes()
    new[field] = value
    with pytest.raises(ValueError, match="fixed worst-boundary recipe"):
        runner._recipe(new, old)


def child():
    return {
        "aggregate": {"exact_count": 208, "f1": 0.9998, "serialization_compatible": 222},
        "groups": {
            g: {"exact_count": v}
            for g, v in {"COLOR": 103, "TYPE_single": 50, "TYPE_dual": 54}.items()
        },
    }


def test_new_gate_strictly_exceeds_balanced_sibling(runner):
    data = child()
    assert runner._gate(data, {"all": True})["primary_checks_passed"]
    data["aggregate"]["exact_count"] = 207
    assert not runner._gate(data, {"all": True})["primary_checks_passed"]


@pytest.mark.parametrize(
    "change", ["f1", "COLOR", "TYPE_single", "TYPE_dual", "serialization", "invariant"]
)
def test_pooled_gain_cannot_rescue_any_required_gate(runner, change):
    data = child()
    invariants = {"all": True}
    data["aggregate"]["exact_count"] = 220
    if change == "f1":
        data["aggregate"]["f1"] = 0.9997
    elif change == "serialization":
        data["aggregate"]["serialization_compatible"] = 221
    elif change == "invariant":
        invariants["all"] = False
    else:
        data["groups"][change]["exact_count"] -= 1
    gate = runner._gate(data, invariants)
    assert (
        not gate["primary_checks_passed"]
        and not gate["accepted"]
        and gate["independent_audit_required"]
    )


def tiny(torch, bilinear):
    torch.manual_seed(1729)
    model = torch.nn.Module()
    model.config = SimpleNamespace(dim=4)
    model.token_embedding = torch.nn.Embedding(1029, 4)
    model.symmetric_relation_projection = torch.nn.Linear(4, 4, bias=False)
    return bilinear._attach(model)


def test_frozen_bilinear_loop_receives_sole_worst_callback(torch, bilinear, mean, worst):
    model = tiny(torch, bilinear)
    prompts = torch.tensor([[1, 1024, 32, 34, 5], [1, 1025, 33, 34, 5]])
    features = mean._features(model, prompts)
    labels = torch.tensor([[1025, 1026, 2], [1026, 1027, 2]])
    before = mean._state_hashes(model)
    calls = []

    def callback(scores, labels, *, excluded_ids):
        assert torch.equal(excluded_ids, prompts[:, 1] - 1024)
        value = worst._worst_loss(scores, labels, excluded_ids=excluded_ids)
        calls.append(float(value.detach()))
        return value

    optimizer = torch.optim.AdamW([model.symmetric_bilinear_residual], lr=0.0003, weight_decay=0)
    receipt = {}
    bilinear._fit(model, features, prompts[:, 2], labels, callback, optimizer, 3, receipt, mean)
    assert len(calls) == 4 and receipt["completed_updates"] == 3 and len(receipt["history"]) == 3
    assert (
        receipt["initial_pre_update_loss"] == calls[0]
        and receipt["final_post_update_loss"] == calls[-1]
    )
    after = mean._state_hashes(model)
    assert mean._frozen(before, after, weight="symmetric_bilinear_residual")
    assert all(
        p.grad is None for n, p in model.named_parameters() if n != "symmetric_bilinear_residual"
    )
    assert optimizer.state[model.symmetric_bilinear_residual]["step"] == 3


def test_mean_diagnostic_has_no_gradient_or_state_effect(torch, runner, bilinear, mean):
    model = tiny(torch, bilinear)
    prompts = torch.tensor([[1, 1024, 32, 34, 5]])
    features = mean._features(model, prompts)
    labels = torch.tensor([[1025, 1026, 2]])
    before = mean._state_hashes(model)
    calls = []

    def diagnostic(scores, labels, *, excluded_ids):
        assert not torch.is_grad_enabled() and not scores.requires_grad
        calls.append(excluded_ids.tolist())
        return torch.nn.functional.softplus(scores).mean()

    value = runner._mean_diagnostic(
        model, features, prompts[:, 2], labels, diagnostic, bilinear, mean
    )
    assert isinstance(value, float) and calls == [[0]]
    assert mean._state_hashes(model) == before and all(p.grad is None for p in model.parameters())


def test_execution_ast_passes_explicit_callback_and_no_old_execute():
    tree = ast.parse(SCRIPT.read_text())
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_execute")
    fits = [
        n
        for n in ast.walk(execute)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_fit"
    ]
    assert len(fits) == 1 and ast.unparse(fits[0].args[4]) == "worst._worst_loss"
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"_execute", "_validate_child"}
    ]


def contract(runner, bilinear):
    old = load(ROOT / "tests/unit/test_bilinear_budget.py", "old_budget_contract_test")
    args = old._contract(runner, bilinear)
    metadata = args[3]
    config = args[2]
    for item in [config, metadata]:
        item["implementation_identity"].update(
            loss_helper_sha256=runner._LOSS_HELPER, budget_helper_sha256=runner._BUDGET
        )
    metadata["loss_helper_sha256"] = runner._LOSS_HELPER
    metadata["mean_bce_diagnostic"] = {
        "initial": 0.003822767175734043,
        "final": 0.01,
        "initial_replay_exact": True,
        "used_for_updates": False,
        "update_points": [0, 2000],
    }
    return args


def test_new_objective_identity_keeps_scorer_and_parent_lineage(runner, bilinear):
    args = contract(runner, bilinear)
    runner._validate_child(*args, bilinear)
    assert (
        runner._OBJECTIVE != bilinear._OBJECTIVE
        and runner._SCORER
        == hashlib.sha256((ROOT / "scripts/refit_bilinear_residual.py").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "changed",
    [
        "loss_helper",
        "scorer",
        "objective",
        "evaluator",
        "parent",
        "step",
        "diagnostic",
        "diagnostic_gradient",
        "diagnostic_points",
    ],
)
def test_checkpoint_rejects_changed_lineage_and_diagnostic(runner, bilinear, changed):
    args = contract(runner, bilinear)
    meta = args[3]
    if changed == "loss_helper":
        meta["loss_helper_sha256"] = "wrong"
    elif changed == "scorer":
        meta["scorer_sha256"] = "wrong"
    elif changed == "objective":
        meta["objective"] = "balanced-bce"
    elif changed == "evaluator":
        args[1]["evaluator_version"] = "old-gate"
    elif changed == "parent":
        meta["parent_checkpoint_sha256"] = "sibling-child"
    elif changed == "step":
        args[0].global_step = 500
    elif changed == "diagnostic":
        meta["mean_bce_diagnostic"]["initial"] = 0
    elif changed == "diagnostic_gradient":
        meta["mean_bce_diagnostic"]["used_for_updates"] = True
    else:
        meta["mean_bce_diagnostic"]["update_points"] = [0, 500]
    with pytest.raises(ValueError):
        runner._validate_child(*args, bilinear)


@pytest.mark.parametrize(
    "relative,pinfield",
    [
        ("scripts/refit_bilinear_budget.py", "_BUDGET"),
        ("scripts/refit_worst_boundary_projection.py", "_LOSS_HELPER"),
    ],
)
def test_frozen_callback_helpers_rejected_before_import(runner, tmp_path, relative, pinfield):
    path = tmp_path / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text("raise AssertionError('must not load')")
    with pytest.raises(ValueError, match="identity"):
        runner._load_checked(tmp_path, relative, getattr(runner, pinfield), "test")


def test_failed_execution_preserves_partial_evidence(runner, bilinear, mean, worst, tmp_path):
    failure = RuntimeError("synthetic failure before runtime")

    def stop():
        raise failure

    helper = SimpleNamespace(
        _modules=stop,
        _failure_safe=lambda v: v,
        _sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
    )
    out = tmp_path / "summary.json"
    with pytest.raises(RuntimeError) as caught:
        runner._execute(tmp_path, out, {}, bilinear, mean, helper, {"complete": False}, worst)
    assert caught.value is failure
    report = json.loads(out.read_text())
    training = json.loads((tmp_path / "training.json").read_text())
    assert (
        not report["complete"]
        and not training["complete"]
        and training["objective"] == runner._OBJECTIVE
    )
    with pytest.raises(ValueError, match="immutable"):
        mean._refuse(out)
