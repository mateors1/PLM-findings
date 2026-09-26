"""Portable CPU contracts for the fixed8000 budget and descriptive prefix evidence."""

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
SCRIPT = ROOT / "scripts/refit_bilinear_budget8000.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load(SCRIPT, "budget8000_test")


@pytest.fixture(scope="module")
def bilinear(runner):
    return runner._load_bilinear(ROOT)[0]


@pytest.fixture(scope="module")
def mean(bilinear):
    return bilinear._load_mean(ROOT)[0]


def recipes():
    return [
        json.loads((ROOT / "configs/experiments" / n).read_text())
        for n in ["bilinear_budget8000_v1.json", "bilinear_budget2000_v1.json"]
    ]


def test_only_declared_budget_campaign_evaluator_and_floors_change(runner):
    new, old = recipes()
    before = copy.deepcopy(old)
    assert runner._recipe(new, old) == new and old == before
    assert {k for k in old if old[k] != new[k]} == {
        "experiment",
        "residual_updates",
        "evaluator",
        "strong_comparator_exact",
        "strong_comparator_f1",
        "strong_comparator_group_exact",
    }
    assert new["objective"] == old["objective"] == runner._OBJECTIVE
    assert new["residual_updates"] == 8000 and new["parent_training_steps"] == 2000


@pytest.mark.parametrize(
    "key,value",
    [
        ("residual_updates", 2000),
        ("residual_updates", 7999),
        ("residual_updates", 8001),
        ("threshold", 0.1),
        ("objective", "worst"),
        ("initialization", "resume"),
        ("strong_comparator_exact", 201),
        ("strong_comparator_f1", 0.98),
        ("optimizer", {"lr": 0.001}),
    ],
)
def test_recipe_rejects_budget_objective_and_policy_drift(runner, key, value):
    new, old = recipes()
    new[key] = value
    with pytest.raises(ValueError, match="fixed 8000-update recipe"):
        runner._recipe(new, old)


def trace():
    old = {
        "history": [{"update": i + 1, "pre_update_loss": 1 / (i + 1)} for i in range(2000)],
        "final_post_update_loss": 0.0004,
    }
    new = {
        "history": copy.deepcopy(old["history"])
        + [{"update": i + 1, "pre_update_loss": 0.0004 / (i - 1999)} for i in range(2000, 8000)]
    }
    return new, old


def test_prefix_separates_preupdate2000_from_aligned_preupdate2001(runner):
    new, old = trace()
    before = copy.deepcopy((new, old))
    result = runner._prefix_diagnostic(new, old)
    assert result == {
        "pre_update_compared_count": 2000,
        "all_pre_update_equal": True,
        "mismatch_count": 0,
        "first_mismatch_update": None,
        "historical_final_post_update_loss": 0.0004,
        "current_pre_update_2001_loss": 0.0004,
        "boundary_loss_equal": True,
        "used_for_selection_or_updates": False,
        "descriptive_only": True,
    }
    assert new["history"][1999]["pre_update_loss"] != old["final_post_update_loss"]
    assert (new, old) == before


@pytest.mark.parametrize("indices", [[0], [1999], [0, 15, 1999]])
def test_prefix_mismatch_is_descriptive_and_never_aborts(runner, indices):
    new, old = trace()
    for i in indices:
        new["history"][i]["pre_update_loss"] += 0.5
    result = runner._prefix_diagnostic(new, old)
    assert result["mismatch_count"] == len(indices)
    assert result["first_mismatch_update"] == indices[0] + 1 and not result["all_pre_update_equal"]
    assert result["boundary_loss_equal"] and result["used_for_selection_or_updates"] is False


def test_boundary_mismatch_does_not_change_prefix_equality(runner):
    new, old = trace()
    new["history"][2000]["pre_update_loss"] = 0.0003
    result = runner._prefix_diagnostic(new, old)
    assert result["all_pre_update_equal"] and not result["boundary_loss_equal"]


@pytest.mark.parametrize("change", ["short_new", "short_old", "order_new", "order_old"])
def test_prefix_rejects_missing_or_misaligned_trace_evidence(runner, change):
    new, old = trace()
    if change == "short_new":
        new["history"].pop()
    elif change == "short_old":
        old["history"].pop()
    elif change == "order_new":
        new["history"][0]["update"] = 2
    else:
        old["history"][0]["update"] = 2
    with pytest.raises(ValueError):
        runner._prefix_diagnostic(new, old)


def passing_child():
    return {
        "aggregate": {"exact_count": 208, "f1": 0.9998, "serialization_compatible": 222},
        "groups": {
            g: {"exact_count": n}
            for g, n in {"COLOR": 103, "TYPE_single": 50, "TYPE_dual": 54}.items()
        },
    }


def test_gate_strict_boundary_and_no_prefix_requirement(runner):
    child = passing_child()
    assert runner._gate(child, {"execution": True})["primary_checks_passed"]
    child["aggregate"]["exact_count"] = 207
    assert not runner._gate(child, {"execution": True})["primary_checks_passed"]
    node = next(
        n
        for n in ast.parse(SCRIPT.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_gate"
    )
    assert "prefix" not in ast.unparse(node)


@pytest.mark.parametrize(
    "change", ["f1", "COLOR", "TYPE_single", "TYPE_dual", "serialization", "invariants"]
)
def test_total_gain_cannot_rescue_required_gate(runner, change):
    child = passing_child()
    child["aggregate"]["exact_count"] = 220
    inv = {"execution": True}
    if change == "f1":
        child["aggregate"]["f1"] = 0.9997
    elif change == "serialization":
        child["aggregate"]["serialization_compatible"] = 221
    elif change == "invariants":
        inv["execution"] = False
    else:
        child["groups"][change]["exact_count"] -= 1
    gate = runner._gate(child, inv)
    assert (
        not gate["primary_checks_passed"]
        and not gate["accepted"]
        and gate["independent_audit_required"]
    )


def contract(runner, bilinear):
    old = load(ROOT / "tests/unit/test_bilinear_budget.py", "budget8000_contract_fixture")
    args = old._contract(runner, bilinear)
    state, _identity, config, meta, _corpus = args
    state.global_step = state.metadata["global_step"] = 8000
    meta["residual_updates"] = 8000
    for item in [config, meta]:
        item["implementation_identity"]["budget_helper_sha256"] = runner._BUDGET
    return args


def test_checkpoint_keeps_parent2000_and_child8000_separate(runner, bilinear):
    args = contract(runner, bilinear)
    runner._validate_child(*args, bilinear)
    assert args[3]["parent_training_steps"] == 2000 and args[3]["residual_updates"] == 8000
    assert runner._OBJECTIVE == bilinear._OBJECTIVE and runner._EVALUATOR != bilinear._EVALUATOR


@pytest.mark.parametrize(
    "change",
    [
        "global_step",
        "payload_step",
        "residual_updates",
        "parent_steps",
        "objective",
        "evaluator",
        "budget_helper",
        "parent",
    ],
)
def test_child_rejects_wrong_step_objective_or_lineage(runner, bilinear, change):
    args = contract(runner, bilinear)
    meta = args[3]
    if change == "global_step":
        args[0].global_step = 2000
    elif change == "payload_step":
        args[0].metadata["global_step"] = 2000
    elif change == "residual_updates":
        meta["residual_updates"] = 2000
    elif change == "parent_steps":
        meta["parent_training_steps"] = 8000
    elif change == "objective":
        meta["objective"] = "worst"
    elif change == "evaluator":
        args[1]["evaluator_version"] = "old-gate"
    elif change == "budget_helper":
        meta["implementation_identity"]["budget_helper_sha256"] = "wrong"
    else:
        meta["parent_checkpoint_sha256"] = "balanced-child"
    with pytest.raises(ValueError):
        runner._validate_child(*args, bilinear)


def test_balanced_callback_fixed_budget_and_owned_lifecycle():
    tree = ast.parse(SCRIPT.read_text())
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_execute")
    fits = [
        n
        for n in ast.walk(execute)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_fit"
    ]
    assert (
        len(fits) == 1
        and ast.unparse(fits[0].args[4]) == "_prompt_set_loss"
        and fits[0].args[6].value == 8000
    )
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"_execute", "_validate_child"}
    ]
    assert "_worst_loss" not in SCRIPT.read_text()


def test_wrong_budget_helper_rejected_before_import(runner, tmp_path):
    path = tmp_path / "scripts/refit_bilinear_budget.py"
    path.parent.mkdir()
    path.write_text("raise AssertionError('must not load')")
    with pytest.raises(ValueError, match="identity"):
        runner._load_budget(tmp_path)


def test_failed_execution_retains_partial_evidence(runner, bilinear, mean, tmp_path):
    failure = RuntimeError("synthetic stop before runtime")

    def stop():
        raise failure

    helper = SimpleNamespace(
        _modules=stop,
        _failure_safe=lambda v: v,
        _sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
    )
    out = tmp_path / "summary.json"
    with pytest.raises(RuntimeError) as caught:
        runner._execute(tmp_path, out, {}, bilinear, mean, helper, {"complete": False})
    assert caught.value is failure
    report = json.loads(out.read_text())
    training = json.loads((tmp_path / "training.json").read_text())
    assert (
        not report["complete"] and not training["complete"] and training["completed_updates"] == 0
    )
    with pytest.raises(ValueError, match="immutable"):
        mean._refuse(out)
