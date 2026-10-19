"""Portable orchestration contracts; no campaign inputs, real weights, or GPU."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/refit_bilinear_budget.py"


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _module(SCRIPT, "budget_test_runner")


@pytest.fixture(scope="module")
def bilinear(runner):
    return runner._load_bilinear(ROOT)[0]


@pytest.fixture(scope="module")
def mean():
    return _module(ROOT / "scripts/refit_membership_projection.py", "budget_mean_helper")


def _recipes():
    directory = ROOT / "configs/experiments"
    return tuple(
        json.loads((directory / name).read_text(encoding="utf-8"))
        for name in ("bilinear_budget2000_v1.json", "bilinear_residual_refit_v1.json")
    )


def test_only_declared_budget_and_campaign_change(runner):
    recipe, old = _recipes()
    unchanged = copy.deepcopy(old)
    assert runner._recipe(recipe, old) == recipe
    assert old == unchanged
    assert {k for k in old if old[k] != recipe[k]} == {"experiment", "residual_updates"}
    assert recipe["residual_updates"] == 2000
    assert recipe["evaluator"] == old["evaluator"] == runner._EVALUATOR


@pytest.mark.parametrize(
    "key,value",
    [
        ("residual_updates", 500),
        ("residual_updates", 2001),
        ("initialization", "resume-500-child"),
        ("threshold", 0.01),
        ("validation_batch_size", 1),
        ("strong_comparator_exact", 198),
        ("optimizer", {"name": "adamw", "lr": 0.001}),
    ],
)
def test_fixed_recipe_rejects_budget_resume_or_policy_drift(runner, key, value):
    recipe, old = _recipes()
    recipe[key] = value
    with pytest.raises(ValueError, match="fixed budget recipe"):
        runner._recipe(recipe, old)


def _contract(runner, bilinear):
    implementation = {
        "runner_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "scorer_sha256": runner._SCORER,
        "mean_helper_sha256": runner._MEAN,
        "source_archive_sha256": "synthetic-source",
        "configs_archive_sha256": "synthetic-configs",
    }
    identity = {"evaluator_version": runner._EVALUATOR}
    config = {
        "implementation_identity": copy.deepcopy(implementation),
        "bilinear_residual_architecture": bilinear._architecture(),
    }
    metadata = {
        "implementation_identity": copy.deepcopy(implementation),
        "runner_sha256": implementation["runner_sha256"],
        "scorer_sha256": runner._SCORER,
        "architecture": bilinear._architecture(),
        "objective": runner._OBJECTIVE,
        "evaluator": runner._EVALUATOR,
        "trainable_parameters": [runner._PARAMETER],
        "residual_updates": 2000,
        "parent_training_steps": 2000,
        "parent_checkpoint_sha256": (
            "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
        ),
    }
    corpus = {"fixture": "synthetic"}
    raw = {
        "global_step": 2000,
        "experiment_identity": identity,
        "config": config,
        "training_metadata": metadata,
        "corpus_identity": corpus,
        "split_hash": "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d",
    }
    return SimpleNamespace(global_step=2000, metadata=raw), identity, config, metadata, corpus


def test_new_runner_identity_is_distinct_from_authenticated_frozen_scorer(runner, bilinear):
    args = _contract(runner, bilinear)
    assert args[3]["runner_sha256"] != args[3]["scorer_sha256"]
    runner._validate_child(*args, bilinear)
    assert runner._OBJECTIVE == bilinear._OBJECTIVE
    assert runner._EVALUATOR == bilinear._EVALUATOR


@pytest.mark.parametrize("location", ["state", "payload", "residual_updates"])
def test_500_step_checkpoint_cannot_impersonate_2000_child(runner, bilinear, location):
    args = _contract(runner, bilinear)
    if location == "state":
        args[0].global_step = 500
    elif location == "payload":
        args[0].metadata["global_step"] = 500
    else:
        args[3][location] = 500
    with pytest.raises(ValueError):
        runner._validate_child(*args, bilinear)


@pytest.mark.parametrize(
    "key,value",
    [
        ("parent_checkpoint_sha256", "historical-500-child"),
        ("parent_training_steps", 2500),
        ("trainable_parameters", ["symmetric_relation_projection.weight"]),
        ("objective", "worst-boundary"),
    ],
)
def test_original_parent_and_objective_lineage_remain_strict(runner, bilinear, key, value):
    args = _contract(runner, bilinear)
    args[3][key] = value
    with pytest.raises(ValueError):
        runner._validate_child(*args, bilinear)


@pytest.mark.parametrize("changed", ["scorer", "runner", "mean"])
def test_implementation_identity_contradictions_rejected(runner, bilinear, changed):
    args = _contract(runner, bilinear)
    config, metadata = args[2:4]
    if changed == "scorer":
        # Consistent copies still cannot redefine the frozen scorer.
        for item in (config, metadata):
            item["implementation_identity"]["scorer_sha256"] = "0" * 64
        metadata["scorer_sha256"] = "0" * 64
    elif changed == "runner":
        metadata["runner_sha256"] = "0" * 64
    else:
        for item in (config, metadata):
            item["implementation_identity"]["mean_helper_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="implementation identity"):
        runner._validate_child(*args, bilinear)


def _reports():
    rows = []
    for index, group in enumerate(("COLOR", "TYPE_single", "TYPE_dual")):
        rows.append(
            {
                "index": index,
                "subject": f"synthetic-{index}",
                "dimension": "COLOR" if index == 0 else "TYPE",
                "prompt_ids": [1, 1024 + index, 33 if index == 0 else 32, 34, 5],
                "group": group,
                "expected_set_ids": [1030],
                "selected_set_ids": [1030] if index != 1 else [1031],
                "metrics": {"exact": index != 1},
            }
        )
    current = {
        "complete": True,
        "product_token_ids": list(range(1024, 1032)),
        "query_count": len(rows),
        "responses": rows,
    }
    old = copy.deepcopy(current)
    old["responses"][0]["selected_set_ids"] = [1031]
    old["responses"][1]["selected_set_ids"] = [1030]
    return current, old


def test_historical_pairing_uses_saved_sets_and_preserves_inputs(runner, mean):
    current, old = _reports()
    before = copy.deepcopy((current, old))
    result = runner._historical_pair(current, old, mean)
    assert result["gains"] == result["losses"] == 1
    assert result["groups"] == {
        "COLOR": {"gains": 1, "losses": 0},
        "TYPE_single": {"gains": 0, "losses": 1},
        "TYPE_dual": {"gains": 0, "losses": 0},
    }
    assert (current, old) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("index", 19),
        ("subject", "different"),
        ("dimension", "TYPE"),
        ("prompt_ids", [1, 1024, 32, 34, 5]),
        ("group", "TYPE_dual"),
        ("expected_set_ids", [1031]),
    ],
)
def test_historical_pairing_refuses_misaligned_observations(runner, mean, key, value):
    current, old = _reports()
    old["responses"][0][key] = value
    with pytest.raises(ValueError, match="historical query/label order"):
        runner._historical_pair(current, old, mean)


@pytest.mark.parametrize("changed", ["incomplete", "columns", "count", "order"])
def test_historical_pairing_requires_complete_coverage_and_order(runner, mean, changed):
    current, old = _reports()
    if changed == "incomplete":
        old["complete"] = False
    elif changed == "columns":
        old["product_token_ids"].reverse()
    elif changed == "count":
        old["responses"].pop()
    else:
        old["responses"].reverse()
    with pytest.raises(ValueError):
        runner._historical_pair(current, old, mean)


def test_wrong_scorer_is_rejected_before_import(runner, tmp_path):
    target = tmp_path / "scripts/refit_bilinear_residual.py"
    target.parent.mkdir()
    target.write_text("raise AssertionError('unverified helper executed')", encoding="utf-8")
    with pytest.raises(ValueError, match="bilinear helper identity"):
        runner._load_bilinear(tmp_path)


def test_execution_failure_retains_authenticated_partial_receipts(runner, bilinear, mean, tmp_path):
    failure = RuntimeError("synthetic failure before archived imports")

    def fail():
        raise failure

    helper = SimpleNamespace(
        _modules=fail,
        _failure_safe=lambda value: value,
        _sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    out = tmp_path / "summary.json"
    with pytest.raises(RuntimeError) as caught:
        runner._execute(tmp_path, out, {}, bilinear, mean, helper, {"complete": False})
    assert caught.value is failure
    summary = json.loads(out.read_text(encoding="utf-8"))
    training_path = tmp_path / "training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    assert summary["complete"] is training["complete"] is False
    assert training["completed_updates"] == 0 and not training["history"]
    assert training["error"] == summary["error"] == repr(failure)
    assert summary["artifact_sha256"]["training.json"] == helper._sha(training_path)
    with pytest.raises(ValueError, match="immutable"):
        mean._refuse(out)


@pytest.mark.parametrize("artifact", ["comparisons.json", "zero-replay.json"])
def test_new_artifacts_block_reexecution_before_preflight(
    runner, mean, monkeypatch, tmp_path, artifact
):
    existing = tmp_path / artifact
    existing.write_text("preserve exact prior bytes", encoding="utf-8")
    helper = SimpleNamespace(_modules=lambda: None)
    fake_mean = SimpleNamespace(_load_helper=lambda _: (helper, tmp_path / "helper.py"))
    fake_mean._refuse = mean._refuse
    fake_bilinear = SimpleNamespace(_load_mean=lambda _: (fake_mean, tmp_path / "mean.py"))
    monkeypatch.setattr(runner, "_load_bilinear", lambda _: (fake_bilinear, tmp_path / "old.py"))
    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda *_: pytest.fail("preflight must not read inputs after artifact collision"),
    )
    # Isolate the collision guard from Torch imported by the separately run math tests.
    monkeypatch.setattr(runner, "sys", SimpleNamespace(modules={}))
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(tmp_path / "summary.json")])
    with pytest.raises(ValueError, match="immutable"):
        runner.main()
    assert existing.read_text(encoding="utf-8") == "preserve exact prior bytes"
