"""Synthetic replication contracts without campaign files or real checkpoints."""

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
SCRIPT = ROOT / "scripts/refit_bilinear_budget8000_replication.py"
PARENTS = {
    1730: "5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2",
    1731: "ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d",
}


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture(scope="module")
def runner():
    return _module(SCRIPT, "replication_test_runner")


@pytest.fixture(scope="module")
def bilinear():
    return _module(ROOT / "scripts/refit_bilinear_residual.py", "replication_bilinear_helper")


@pytest.fixture(scope="module")
def mean():
    return _module(ROOT / "scripts/refit_membership_projection.py", "replication_mean_helper")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tiny(torch):
    model = torch.nn.Module()
    model.config = SimpleNamespace(dim=4)
    model.token_embedding = torch.nn.Embedding(1029, 4)
    model.symmetric_relation_projection = torch.nn.Linear(4, 4, bias=False)
    return model


def test_each_parent_replays_its_own_initial_loss(runner, bilinear, mean):
    torch = pytest.importorskip("torch")
    observed = []
    for seed in (1730, 1731):
        torch.manual_seed(seed)
        model = bilinear._attach(_tiny(torch))
        prompts = torch.tensor([[1, 1024, 32, 34, 5], [1, 1025, 33, 34, 5]])
        features = mean._features(model, prompts)
        calls = []

        def loss_fn(logits, labels, *, excluded_ids):
            # A synthetic scalar callback isolates replay from archived BCE math.
            assert labels.shape == (2, 1)
            assert torch.equal(excluded_ids, prompts[:, 1] - 1024)
            value = torch.nn.functional.softplus(logits).mean()
            calls.append(value.item())
            return value

        before = mean._state_hashes(model)
        runner._initial_replay(
            model, features, prompts[:, 2], torch.tensor([[1025], [1026]]), loss_fn, bilinear, mean
        )
        assert len(calls) == 2 and calls[0] == calls[1]
        assert mean._state_hashes(model) == before
        observed.append(calls[0])
    assert observed[0] != observed[1]


def test_initial_replay_rejects_nonzero_residual(runner, bilinear, mean):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1730)
    model = bilinear._attach(_tiny(torch))
    with torch.no_grad():
        model.symmetric_bilinear_residual[0, 0, 0] = 1.0
    prompts = torch.tensor([[1, 1024, 32, 34, 5]])
    with pytest.raises(ValueError):
        runner._initial_replay(
            model,
            mean._features(model, prompts),
            prompts[:, 2],
            torch.tensor([[1025]]),
            lambda logits, labels, **kwargs: logits.square().mean(),
            bilinear,
            mean,
        )


def _parent_contract(runner, seed):
    base = {
        "seed": 1729,
        "run_name": "national_dex_continuation_control_s1729_v1",
        "data": {"split_seed": 1729},
        "model": {"symmetric_relation_loss_weight": 1.0},
    }
    config = copy.deepcopy(base)
    config.update(seed=seed, run_name=f"national_dex_continuation_control_s{seed}_v1")
    identity = {
        "seeds": [seed],
        "resolved_config_hash": runner._SEED_CONTRACTS[seed]["parent_config_sha256"],
        "split_hash": runner._SPLIT,
    }
    run = {"config": config, "identity": identity}
    sidecar = {
        "checkpoint_hash": PARENTS[seed],
        "experiment_identity": copy.deepcopy(identity),
        "global_step": 2000,
        "split_hash": runner._SPLIT,
        "training_metadata": {"objective": "causal-next-token-v1"},
    }
    result = {
        "checkpoint_hash": PARENTS[seed],
        "identity": copy.deepcopy(identity),
        "global_step": 2000,
    }
    return run, sidecar, result, base


@pytest.mark.parametrize("seed", [1730, 1731])
def test_seed_maps_original_parent_without_changing_data_split(runner, seed):
    args = _parent_contract(runner, seed)
    before = copy.deepcopy(args)
    result = runner._seed_contract(seed, *args)
    assert result["seed"] == result["model_seed"] == seed
    assert result["data_split_seed"] == 1729
    assert result["parent_checkpoint_sha256"] == PARENTS[seed]
    assert result["floors"]["exact_count"] == {1730: 212, 1731: 208}[seed]
    assert result["floors"]["groups"] == {
        "COLOR": 103,
        "TYPE_single": {1730: 51, 1731: 51}[seed],
        "TYPE_dual": {1730: 58, 1731: 54}[seed],
    }
    assert args == before


@pytest.mark.parametrize("changed", ["parent", "seed", "split", "model", "config_hash"])
def test_swapped_parent_or_training_config_cannot_pass_mapping(runner, changed):
    args = _parent_contract(runner, 1730)
    run, sidecar, result, _ = args
    if changed == "parent":
        sidecar["checkpoint_hash"] = result["checkpoint_hash"] = PARENTS[1731]
    elif changed == "seed":
        for identity in (run["identity"], sidecar["experiment_identity"], result["identity"]):
            identity["seeds"] = [1731]
    elif changed == "split":
        run["config"]["data"]["split_seed"] = 1730
    elif changed == "model":
        run["config"]["model"]["symmetric_relation_loss_weight"] = 0.0
    else:
        for identity in (run["identity"], sidecar["experiment_identity"], result["identity"]):
            identity["resolved_config_hash"] = "0" * 64
    with pytest.raises(ValueError):
        runner._seed_contract(1730, *args)


def _quality(runner, seed):
    floors = copy.deepcopy(runner._SEED_CONTRACTS[seed]["floors"])
    child = {
        "aggregate": {
            "query_count": 222,
            "serialization_compatible": 222,
            "exact_count": floors["exact_count"] + 1,
            "f1": floors["f1"],
        },
        "groups": {g: {"exact_count": count} for g, count in floors["groups"].items()},
    }
    return child, {"replay": True, "reload": True}, floors


@pytest.mark.parametrize("seed", [1730, 1731])
def test_gate_uses_matching_seed_floors_and_strict_exact_gain(runner, seed):
    child, invariants, floors = _quality(runner, seed)
    gate = runner._seed_gate(child, invariants, floors)
    assert gate["primary_checks_passed"] and gate["accepted"] is False
    child["aggregate"]["exact_count"] = floors["exact_count"]
    assert not runner._seed_gate(child, invariants, floors)["primary_checks_passed"]


@pytest.mark.parametrize("changed", ["f1", "group", "serialization", "invariant", "empty"])
def test_exact_gain_cannot_rescue_other_gate_failure(runner, changed):
    child, invariants, floors = _quality(runner, 1730)
    if changed == "f1":
        child["aggregate"]["f1"] -= 1e-12
    elif changed == "group":
        child["groups"]["TYPE_dual"]["exact_count"] -= 1
    elif changed == "serialization":
        child["aggregate"]["serialization_compatible"] = 221
    elif changed == "invariant":
        invariants["replay"] = 1  # Truthiness must not impersonate an actual passed check.
    else:
        invariants.clear()
    assert not runner._seed_gate(child, invariants, floors)["primary_checks_passed"]


def _campaign_reports(runner):
    return {
        seed: {
            "seed": seed,
            "complete": True,
            "gate": runner._seed_gate(*_quality(runner, seed)),
        }
        for seed in (1730, 1731)
    }


def test_campaign_requires_both_gates_and_historical_acceptance(runner):
    reports = _campaign_reports(runner)
    assert runner._campaign_gate(reports, True)["primary_checks_passed"]
    assert not runner._campaign_gate(reports, False)["primary_checks_passed"]
    reports[1730]["gate"]["primary_checks_passed"] = False
    result = runner._campaign_gate(reports, True)
    assert not result["primary_checks_passed"]
    assert result["checks"]["fresh_seed1731"] is True
    assert result["accepted"] is False
    # A quality failure must not make a missing/incomplete second seed acceptable.
    reports[1731]["complete"] = False
    with pytest.raises(ValueError, match="complete matching"):
        runner._campaign_gate(reports, True)


@pytest.mark.parametrize("changed", ["missing", "extra_historical", "swapped"])
def test_campaign_rejects_wrong_fresh_seed_coverage(runner, changed):
    reports = _campaign_reports(runner)
    if changed == "missing":
        reports.pop(1731)
    elif changed == "extra_historical":
        reports[1729] = {"seed": 1729, "complete": True}
    else:
        reports[1731]["seed"] = 1730
    with pytest.raises(ValueError):
        runner._campaign_gate(reports, True)


def _observations(mean, wrong=0):
    rows = []
    for index in range(222):
        group = "COLOR" if index < 103 else "TYPE_single" if index < 154 else "TYPE_dual"
        selected = [2047] if index < wrong else [2048]
        rows.append(
            {
                "index": index,
                "subject": f"synthetic-{index}",
                "dimension": "COLOR" if group == "COLOR" else "TYPE",
                "prompt_ids": [1, 1024 + index, 33 if group == "COLOR" else 32, 34, 5],
                "expected_set_ids": [2048],
                "group": group,
                "selected_set_ids": selected,
                "metrics": mean._metrics(selected, [2048]),
                "strict_separation": True,
            }
        )
    return {
        "complete": True,
        "query_count": 222,
        "product_token_ids": list(range(1024, 2049)),
        "responses": rows,
    }


def test_pooling_distinguishes_fresh444_from_historical_plus_fresh666(runner, mean):
    historical = _observations(mean, wrong=15)
    fresh = {1731: _observations(mean, wrong=1), 1730: _observations(mean)}
    result = runner._aggregate_reports(fresh, historical, mean)
    assert result["fresh_two"]["aggregate"]["query_count"] == 444
    assert result["fresh_two"]["aggregate"]["exact_count"] == 443
    assert result["all_three"]["aggregate"]["query_count"] == 666
    assert result["all_three"]["aggregate"]["exact_count"] == 650
    assert result["fresh_two"]["groups"]["TYPE_dual"]["query_count"] == 136
    assert result["all_three"]["groups"]["TYPE_dual"]["query_count"] == 204


@pytest.mark.parametrize("changed", ["order", "labels", "columns"])
def test_pooling_refuses_misaligned_seed_queries(runner, mean, changed):
    historical = _observations(mean)
    fresh = {1730: _observations(mean), 1731: _observations(mean)}
    if changed == "order":
        fresh[1731]["responses"].reverse()
    elif changed == "labels":
        fresh[1731]["responses"][0]["expected_set_ids"] = [2047]
    else:
        fresh[1731]["product_token_ids"].reverse()
    with pytest.raises(ValueError):
        runner._aggregate_reports(fresh, historical, mean)


def test_recipe_preserves_budget_and_removes_single_seed_initial_loss(runner):
    directory = ROOT / "configs/experiments"
    recipe = _read(directory / "bilinear_budget8000_replication_v1.json")
    inherited = _read(directory / "bilinear_budget2000_v1.json")
    before = copy.deepcopy(inherited)
    assert runner._recipe(recipe, inherited) == recipe
    assert inherited == before
    assert recipe["fresh_seeds"] == [1730, 1731] and recipe["historical_seed"] == 1729
    assert "initial_training_loss" not in recipe
    assert recipe["residual_updates"] == 8000
    assert recipe["optimizer"] == inherited["optimizer"]


@pytest.mark.parametrize("changed", ["seeds", "initial_loss", "budget", "floors"])
def test_recipe_refuses_undeclared_replication_changes(runner, changed):
    directory = ROOT / "configs/experiments"
    recipe = _read(directory / "bilinear_budget8000_replication_v1.json")
    inherited = _read(directory / "bilinear_budget2000_v1.json")
    if changed == "seeds":
        recipe["fresh_seeds"] = [1730]
    elif changed == "initial_loss":
        recipe["initial_training_loss"] = inherited["initial_training_loss"]
    elif changed == "budget":
        recipe["residual_updates"] = 2001
    else:
        recipe["seed_contracts"]["1730"]["floors"]["exact_count"] = 201
    with pytest.raises(ValueError, match="fixed replication recipe"):
        runner._recipe(recipe, inherited)


def _child_contract(runner, bilinear, seed):
    # Extend the immutable budget fixture with the newly declared seed fields.
    old_tests = _module(ROOT / "tests/unit/test_bilinear_budget.py", "replication_old_fixture")
    args = old_tests._contract(runner, bilinear)
    _, identity, config, metadata, _ = args
    identity["seeds"] = [seed]
    for item in (config, metadata):
        item.update(seed=seed, data_split_seed=1729, campaign_version=runner._CAMPAIGN)
        item["implementation_identity"]["budget_helper_sha256"] = runner._BUDGET
    metadata["parent_checkpoint_sha256"] = PARENTS[seed]
    metadata["budget_helper_sha256"] = runner._BUDGET
    state = args[0]
    state.global_step = state.metadata["global_step"] = 8000
    metadata["residual_updates"] = 8000
    for item in (config, metadata):
        item["implementation_identity"].update(
            replication_helper_sha256=runner._REPLICATION, budget8000_helper_sha256=runner._EXTENDED
        )
    return args


@pytest.mark.parametrize("seed", [1730, 1731])
def test_child_identity_records_own_seed_and_original_parent(runner, bilinear, seed):
    args = _child_contract(runner, bilinear, seed)
    runner._validate_child(*args, seed, bilinear)
    args[3]["parent_checkpoint_sha256"] = PARENTS[3461 - seed]
    with pytest.raises(ValueError, match="original parent"):
        runner._validate_child(*args, seed, bilinear)


@pytest.mark.parametrize("changed", ["step", "seed", "split_seed", "evaluator"])
def test_child_rejects_wrong_checkpoint_contract(runner, bilinear, changed):
    args = _child_contract(runner, bilinear, 1730)
    state, identity, config, metadata, _ = args
    if changed == "step":
        state.global_step = state.metadata["global_step"] = 500
    elif changed == "seed":
        identity["seeds"] = [1731]
    elif changed == "split_seed":
        config["data_split_seed"] = metadata["data_split_seed"] = 1730
    else:
        identity["evaluator_version"] = "plm-bilinear-residual-screen-v1"
    with pytest.raises(ValueError):
        runner._validate_child(*args, 1730, bilinear)


def test_execution_error_preserves_partial_seed_evidence(runner, bilinear, mean, tmp_path):
    failure = RuntimeError("synthetic boundary failure")

    def stop():
        raise failure

    helper = SimpleNamespace(
        _modules=stop,
        _failure_safe=lambda value: value,
        _sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    out = tmp_path / "summary.json"
    with pytest.raises(RuntimeError) as caught:
        runner._execute(
            tmp_path, out, {"seed": 1730}, bilinear, mean, helper, {"complete": False, "seed": 1730}
        )
    assert caught.value is failure
    report, training = _read(out), _read(tmp_path / "training.json")
    assert report["seed"] == 1730
    assert report["complete"] is training["complete"] is False
    assert training["completed_updates"] == 0
    assert report["error"] == training["error"] == repr(failure)
    assert report["artifact_sha256"]["training.json"] == helper._sha(tmp_path / "training.json")
    with pytest.raises(ValueError, match="immutable"):
        mean._refuse(out)


def test_changed_budget_helper_cannot_execute(runner, tmp_path):
    path = tmp_path / "scripts/refit_bilinear_budget.py"
    path.parent.mkdir()
    path.write_text("raise AssertionError('unverified code executed')", encoding="utf-8")
    with pytest.raises(ValueError, match="budget helper identity"):
        runner._load_budget(tmp_path)


def test_aggregate_output_guard_accepts_clean_directory_then_refuses_prior_bytes(
    runner, mean, tmp_path
):
    out = tmp_path / "summary.json"
    runner._refuse(out, mean, aggregate=True)
    manifest = out.with_suffix(".aggregate-inputs.json")
    manifest.write_text("preserve these exact bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        runner._refuse(out, mean, aggregate=True)
    assert manifest.read_text(encoding="utf-8") == "preserve these exact bytes"


def test_only_new_lifecycle_and_balanced_callback_can_execute(runner):
    tree = ast.parse(SCRIPT.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_execute")
    fit = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_fit"
    ]
    assert len(fit) == 1
    assert isinstance(fit[0].args[4], ast.Name) and fit[0].args[4].id == "_prompt_set_loss"
    assert fit[0].args[6].value == 8000
    source = ast.unparse(function)
    assert "validation_update_points=[0, 8000]" in source
    assert "historical_initial_loss_exact" in source
    assert "training_prefix_diagnostic" in source
    assert "_replace" not in source
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"_execute", "_validate_child"}
    assert "--aggregate" not in SCRIPT.read_text()


@pytest.mark.parametrize("changed", ["summary", "audit", "decision", "link", "quality"])
def test_rejects_unaccepted_historical_chain(runner, tmp_path, changed):
    summary = {"complete": True, "final_identity_check": True}
    audit = {"complete": True, "audit_passed": True, "summary_sha256": "summary"}
    decision = {
        "evidence_accepted": True,
        "fixed_quality_gate_passed": True,
        "summary_sha256": "summary",
        "audit_sha256": "audit",
    }
    if changed == "summary":
        summary["complete"] = False
    elif changed == "audit":
        audit["audit_passed"] = False
    elif changed == "decision":
        decision["evidence_accepted"] = False
    elif changed == "link":
        audit["summary_sha256"] = "other"
    else:
        decision["fixed_quality_gate_passed"] = False
    records = {"summary.json": summary, "independent-audit.json": audit, "decision.json": decision}
    helper = SimpleNamespace(_bind=lambda p, h, inputs: p, _read=lambda p: records[p.name])
    with pytest.raises(ValueError, match="accepted historical chain"):
        runner._accepted_chain(
            tmp_path, "synthetic", "summary.json", ("summary", "audit", "decision"), helper, {}
        )


@pytest.mark.parametrize(
    "field", ["replication_helper_sha256", "budget8000_helper_sha256", "budget_helper_sha256"]
)
def test_child_rejects_helper_identity_drift(runner, bilinear, field):
    args = _child_contract(runner, bilinear, 1730)
    args[3]["implementation_identity"][field] = "bad"
    with pytest.raises(ValueError, match="implementation identity"):
        runner._validate_child(*args, 1730, bilinear)
