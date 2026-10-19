"""Synthetic CPU objective/identity tests; shared lifecycle tests run separately."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "scripts/refit_worst_boundary_projection.py"
SPEC = importlib.util.spec_from_file_location("worst_refit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
mean, MEAN_PATH = runner._load_mean(ROOT)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def mean_loss(torch):
    source = ROOT / "tests/fixtures/projection_refit_runtime_source.zip"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
    )
    with zipfile.ZipFile(source) as archive:
        tree = ast.parse(archive.read("src/plm/model/layers.py").decode())
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
        compile(ast.Module(body=[function], type_ignores=[]), "archived-balanced-bce", "exec"),
        namespace,
    )
    return namespace["_prompt_set_loss"]


def test_exact_extreme_value_exceeds_mean_without_class_cardinality_weight(torch, mean_loss):
    z = torch.tensor([[99.0, 1.0, 3.0, 2.0, -2.0], [99.0, 5.0, 7.0, -3.0, -5.0]])
    labels = torch.tensor([[1025, 1026, 2]] * 2)
    subject = torch.tensor([0, 0])
    expected = (
        (
            torch.nn.functional.softplus(torch.tensor([-1.0, -5.0]))
            + torch.nn.functional.softplus(torch.tensor([2.0, -3.0]))
        )
        * 0.5
    ).mean()
    actual = runner._worst_loss(z, labels, excluded_ids=subject)
    assert torch.equal(actual, expected)
    assert actual > mean_loss(z, labels, excluded_ids=subject)


def test_tied_extrema_distribute_subgradients_and_subject_has_none(torch):
    z = torch.tensor([[99.0, 0.0, 0.0, 0.0, 0.0, -4.0]], requires_grad=True)
    loss = runner._worst_loss(z, torch.tensor([[1025, 1026, 2]]), excluded_ids=torch.tensor([0]))
    loss.backward()
    expected = torch.tensor([[0.0, -0.125, -0.125, 0.125, 0.125, 0.0]])
    torch.testing.assert_close(z.grad, expected, rtol=0, atol=0)


def test_gradient_only_worst_members_and_query_mean_scaling(torch):
    z = torch.tensor([[99.0, 1.0, 3.0, 2.0, -2.0]] * 2, requires_grad=True)
    runner._worst_loss(
        z, torch.tensor([[1025, 1026, 2]] * 2), excluded_ids=torch.tensor([0, 0])
    ).backward()
    assert bool((z.grad[:, [0, 2, 4]] == 0).all())
    torch.testing.assert_close(z.grad[:, 1], -torch.sigmoid(torch.tensor(-1.0)).expand(2) / 4)
    torch.testing.assert_close(z.grad[:, 3], torch.sigmoid(torch.tensor(2.0)).expand(2) / 4)


def test_absolute_zero_boundary_not_shift_invariant_relative_gap(torch):
    a = torch.tensor([[99.0, 1.0, -1.0]])
    b = torch.tensor([[99.0, 3.0, 1.0]])
    labels, excluded = torch.tensor([[1025, 2]]), torch.tensor([0])
    assert (a[0, 2] - a[0, 1]).item() == (b[0, 2] - b[0, 1]).item()
    assert runner._worst_loss(a, labels, excluded_ids=excluded) < runner._worst_loss(
        b, labels, excluded_ids=excluded
    )
    assert mean._prediction(a[0].tolist(), 1024) == [1025]
    assert mean._prediction(b[0].tolist(), 1024) == [1025, 1026]


def test_zero_logits_still_receive_softplus_gradient(torch):
    z = torch.zeros(1, 3, requires_grad=True)
    loss = runner._worst_loss(z, torch.tensor([[1025, 2]]), excluded_ids=torch.tensor([0]))
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert torch.equal(z.grad, torch.tensor([[0.0, -0.25, 0.25]]))


def test_control_padding_duplicate_and_out_of_range_labels_do_not_change_masks(torch):
    z = torch.tensor([[99.0, 1.0, 3.0, 2.0, -2.0]])
    args = {"excluded_ids": torch.tensor([0])}
    plain = runner._worst_loss(z, torch.tensor([[1025, 1026, 2]]), **args)
    padded = runner._worst_loss(
        z, torch.tensor([[-100, 1, 32, 34, 5, 1025, 1026, 1025, 2, 0, 2049]]), **args
    )
    assert torch.equal(plain, padded)


@pytest.mark.parametrize(
    "labels", [[[1024, 1025, 2]], [[-100, 1, 2]], [[1025, 1026, 1027, 1028, 2]]]
)
def test_same_subject_and_empty_class_rejections_match_archived_bce(torch, mean_loss, labels):
    scores, target, excluded = torch.zeros(1, 5), torch.tensor(labels), torch.tensor([0])
    with pytest.raises(ValueError):
        mean_loss(scores, target, excluded_ids=excluded)
    with pytest.raises(ValueError):
        runner._worst_loss(scores, target, excluded_ids=excluded)


def test_fp32_under_cpu_autocast(torch):
    scores = torch.zeros(1, 3, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = runner._worst_loss(scores, torch.tensor([[1025, 2]]), excluded_ids=torch.tensor([0]))
    assert loss.dtype == torch.float32
    loss.backward()
    assert scores.grad[0, 1] == -0.25


def _tiny(torch):
    torch.manual_seed(1729)
    model = torch.nn.Module()
    model.config = SimpleNamespace(dim=4)
    model.token_embedding = torch.nn.Embedding(1029, 4)
    model.symmetric_relation_projection = torch.nn.Linear(4, 4, bias=False)
    for name, p in model.named_parameters():
        p.requires_grad_(name == runner._WEIGHT)
    return model


def test_explicit_callback_uses_shared_loop_and_preserves_frozen_state(torch):
    model = _tiny(torch)
    features = mean._features(model, torch.tensor([[1, 1024, 32, 34, 5]]))
    labels = torch.tensor([[1025, 1026, 2]])
    before = mean._state_hashes(model)
    optimizer = torch.optim.AdamW(
        [model.symmetric_relation_projection.weight], lr=0.0003, weight_decay=0
    )
    calls = 0

    def callback(*args, **kwargs):
        nonlocal calls
        calls += 1
        return runner._worst_loss(*args, **kwargs)

    receipt = {}
    mean._fit(model, features, labels, callback, optimizer, 3, receipt)
    assert calls == 4 and receipt["completed_updates"] == 3
    assert len(receipt["history"]) == 3
    assert mean._frozen(before, mean._state_hashes(model))
    assert (
        receipt["final_post_update_loss"]
        == runner._worst_loss(
            mean._head(model.symmetric_relation_projection.weight, features),
            labels,
            excluded_ids=features[2],
        ).item()
    )


def test_mean_diagnostic_has_no_grad_or_optimizer_effect(torch, mean_loss):
    model = _tiny(torch)
    features = mean._features(model, torch.tensor([[1, 1024, 32, 34, 5]]))
    labels = torch.tensor([[1025, 1026, 2]])
    before = mean._state_hashes(model)

    def head(*args):
        assert not torch.is_grad_enabled()
        return mean._head(*args)

    value = runner._mean_diagnostic(model, features, labels, mean_loss, head)
    assert math.isfinite(value) and before == mean._state_hashes(model)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize(
    "key,value",
    [("projection_updates", 501), ("seed", 1730), ("threshold", 0.01), ("loss_coefficient", 0.1)],
)
def test_fixed_recipe_preserves_all_inherited_settings(key, value):
    original = json.loads((ROOT / "configs/experiments/projection_only_refit_v1.json").read_text())
    new = json.loads((ROOT / "configs/experiments/projection_worst_boundary_v1.json").read_text())
    runner._recipe(new, mean, original)
    new[key] = value
    with pytest.raises(ValueError, match="fixed sibling recipe"):
        runner._recipe(new, mean, original)


def test_new_validator_rejects_previous_objective_and_previous_child_parent():
    identity = {"evaluator_version": runner._EVALUATOR}
    metadata = {
        "objective": runner._OBJECTIVE,
        "parent_checkpoint_sha256": runner._PARENT,
        "parent_training_steps": 2000,
        "projection_updates": 500,
    }
    raw = {
        "global_step": 500,
        "experiment_identity": identity,
        "config": {},
        "training_metadata": metadata,
        "corpus_identity": {},
        "split_hash": runner._SPLIT,
    }
    state = SimpleNamespace(global_step=500, metadata=raw)
    runner._validate_child(state, identity, {}, metadata, {})
    for key, value in (
        ("objective", mean._OBJECTIVE),
        ("parent_checkpoint_sha256", runner._MEAN_CHECKPOINT),
    ):
        bad = {**metadata, key: value}
        state.metadata = {**raw, "training_metadata": bad}
        with pytest.raises(ValueError):
            runner._validate_child(state, identity, {}, bad, {})


def test_no_objective_specific_old_execution_or_global_mutation():
    source = ast.parse(SOURCE.read_text())
    for node in ast.walk(source):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "mean"
        ):
            assert node.attr not in {"_execute", "_validate_child"}
            assert not isinstance(node.ctx, ast.Store)
    assert hashlib.sha256(MEAN_PATH.read_bytes()).hexdigest() == runner._MEAN_HELPER


def test_wrong_helper_and_plan_identities_fail_closed(tmp_path):
    path = tmp_path / "scripts/refit_membership_projection.py"
    path.parent.mkdir()
    path.write_text("raise AssertionError('never import')")
    with pytest.raises(ValueError, match="mean helper identity"):
        runner._load_mean(tmp_path)
    plan = tmp_path / "plan.md"
    plan.write_text("wrong")
    old = SimpleNamespace(_preflight=lambda *_: {"inputs": {}})
    helper = SimpleNamespace(
        _bind=lambda p, h, _: runner._require(
            hashlib.sha256(p.read_bytes()).hexdigest() == h, "input identity"
        )
    )
    with pytest.raises(ValueError, match="input identity"):
        runner._preflight(tmp_path, plan, path, old, path, helper, path)


def test_import_boundary_failure_saved_immutably(tmp_path):
    helper = SimpleNamespace(
        _modules=lambda: (_ for _ in ()).throw(ValueError("contaminated")),
        _failure_safe=lambda v: v,
        _sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="contaminated"):
        runner._execute(tmp_path, tmp_path / "summary.json", {}, mean, helper, {"complete": False})
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["complete"] is False and "training.json" in report["artifact_sha256"]
    with pytest.raises(ValueError, match="immutable"):
        mean._refuse(tmp_path / "summary.json")


def test_paired_exact_and_separation_changes_are_distinct():
    def row(exact, separation, ids):
        return {
            "index": 0,
            "subject": "a",
            "dimension": "TYPE",
            "group": "TYPE_single",
            "prompt_ids": [1, 1024, 32, 34, 5],
            "expected_set_ids": [1025],
            "selected_set_ids": ids,
            "metrics": {"exact": exact},
            "strict_separation": separation,
        }

    child = {"responses": [row(True, True, [1025])]}
    parent = {"responses": [row(False, True, [1025, 1026])]}
    previous = {"responses": [row(False, False, [])]}
    wide = [{"composition": {"selected_set_ids": [1025]}}]
    report = runner._comparisons(child, parent, previous, wide, mean)
    assert report["aggregate"]["parent_dense"]["exact"]["gains"] == 1
    assert report["aggregate"]["parent_dense"]["strict_separation"]["gains"] == 0
    assert report["aggregate"]["mean_refit_dense"]["strict_separation"]["gains"] == 1
    assert report["aggregate"]["width8"]["exact"]["gains"] == 0
    bad = copy.deepcopy(previous)
    bad["responses"][0]["expected_set_ids"] = [1026]
    with pytest.raises(ValueError, match="paired query identity"):
        runner._comparisons(child, parent, bad, wide, mean)
