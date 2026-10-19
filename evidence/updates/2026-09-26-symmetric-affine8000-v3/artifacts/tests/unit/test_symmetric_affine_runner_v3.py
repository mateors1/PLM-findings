"""Portable synthetic CPU contracts; no run artifacts, datasets or CUDA required."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load("refit_symmetric_affine_v3")


@pytest.fixture(autouse=True)
def synthetic_receipt_paths(runner, monkeypatch):
    """Short fixture paths; separate cases exercise exact production admission."""
    monkeypatch.setattr(runner, "_PRIMARY_RECEIPT_PATHS", ("receipt.json",))
    monkeypatch.setattr(runner, "_AUDITOR_CONTRACT_PATH", "contract.json")
    monkeypatch.setattr(
        runner,
        "_AUDITOR_PART_PATHS",
        {"auditor": "auditor.py", "tests": "tests.py", "test_receipt": "receipt.json"},
    )


@pytest.fixture(scope="module")
def mean():
    return load("refit_membership_projection")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Helper:
    @staticmethod
    def _bind(path, digest, inputs):
        path = Path(path).resolve()
        if sha(path) != digest:
            raise ValueError("hash mismatch")
        inputs[str(path)] = digest
        return path

    @staticmethod
    def _read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    _sha = staticmethod(sha)
    _failure_safe = staticmethod(lambda value: value)
    _fp32 = staticmethod(lambda vector: None)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha(path)


def recipe_pair():
    return [
        json.loads((ROOT / "configs/experiments" / name).read_text(encoding="utf-8"))
        for name in ("symmetric_affine8000_v3.json", "bilinear_budget8000_v1.json")
    ]


def test_recipe_exact(runner):
    new, old = recipe_pair()
    before = copy.deepcopy(old)
    assert runner._recipe(new, old) == new
    assert old == before
    assert new["affine_scale"] == 16 and new["residual_updates"] == 8000
    assert new["standard_serving_supported"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("affine_scale", 8),
        ("residual_updates", 7999),
        ("threshold", 0.1),
        ("strong_comparator_exact", 207),
        ("objective", "old"),
        ("inference", "ordinary"),
        ("standard_serving_supported", True),
        ("trainable_parameters", ["symmetric_bilinear_residual"]),
        ("extra", True),
    ],
)
def test_recipe_drift_rejected(runner, key, value):
    new, old = recipe_pair()
    new[key] = value
    with pytest.raises(ValueError, match="fixed affine recipe"):
        runner._recipe(new, old)


def invariants():
    return dict.fromkeys(
        (
            "parent_replay_exact",
            "zero_A_u_b_replay_exact",
            "initial_train_loss_exact",
            "completed_8000_updates",
            "original_93_tensors_unchanged",
            "residual_changed",
            "reload_exact",
            "optimizer_reload_exact",
            "train_only_supervision",
        ),
        True,
    )


def candidate():
    return {
        "aggregate": {
            "serialization_compatible": 222,
            "exact_count": 217,
            "f1": 0.9998843626799785,
        },
        "groups": {
            g: {"exact_count": n}
            for g, n in {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 63}.items()
        },
    }


def test_gate_accepts_only_primary_checks(runner):
    gate = runner._gate(candidate(), invariants())
    assert gate["primary_checks_passed"] is True
    assert gate["accepted"] is False and gate["independent_audit_required"] is True


@pytest.mark.parametrize(
    "key,value",
    [("exact_count", 216), ("serialization_compatible", 221), ("f1", 0.9998843626799784)],
)
def test_gate_metric_floor(runner, key, value):
    child = candidate()
    child["aggregate"][key] = value
    assert runner._gate(child, invariants())["primary_checks_passed"] is False


@pytest.mark.parametrize("group,floor", [("COLOR", 103), ("TYPE_single", 51), ("TYPE_dual", 62)])
def test_gate_group_floor(runner, group, floor):
    child = candidate()
    child["groups"][group]["exact_count"] = floor - 1
    assert runner._gate(child, invariants())["primary_checks_passed"] is False


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"extra": True},
        {**invariants(), "reload_exact": 1},
        {**invariants(), "reload_exact": False},
    ],
)
def test_gate_incomplete_invariants(runner, value):
    assert runner._gate(candidate(), value)["primary_checks_passed"] is False


def receipt_fixture(tmp_path):
    for name in ("auditor.py", "tests.py", "stdout.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    receipt = {
        "passed": True,
        "gpu_used": False,
        "experiment_training_executed": False,
        "exit_code": 0,
        "terminal_completion_observed": True,
        "tests_passed": 4,
        "tests_skipped": 0,
        "input_adapter_identity": "plm-authenticated-partition-membership-v1",
        "input_amendment_sha256": (
            "bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f"
        ),
        "repair_declaration_sha256": (
            "34dcf2a850c58ca96770dce02d6f64c1a4aa821d885dba04ad173421189a9dfc"
        ),
        "tested_script_sha256": "runner",
        "tested_recipe_sha256": "recipe",
        "test_dependencies": {n: sha(tmp_path / n) for n in ("auditor.py", "tests.py")},
        "stdout": {"path": "stdout.txt", "sha256": sha(tmp_path / "stdout.txt")},
    }
    write(tmp_path / "receipt.json", receipt)
    return receipt


def contract_fixture(tmp_path, runner):
    receipt = receipt_fixture(tmp_path)
    receipt["tested_script_sha256"] = sha(tmp_path / "auditor.py")
    for name in runner._AUDITOR_EXTRA_DEPENDENCIES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        receipt["test_dependencies"][name] = sha(path)
    write(tmp_path / "receipt.json", receipt)
    value = {
        "schema_version": 1,
        "experiment": "symmetric-affine8000-v3",
        "plan_sha256": runner._PLAN,
        "ready": True,
        "input_adapter_identity": runner._INPUT_ADAPTER,
        "input_amendment_sha256": runner._AMENDMENT,
        "repair_declaration_sha256": runner._REPAIR,
        **{
            k: {"path": n, "sha256": sha(tmp_path / n)}
            for k, n in (
                ("auditor", "auditor.py"),
                ("tests", "tests.py"),
                ("test_receipt", "receipt.json"),
            )
        },
    }
    write(tmp_path / "contract.json", value)
    return value


def test_primary_receipt_portable(runner, tmp_path):
    receipt_fixture(tmp_path)
    path = tmp_path / "receipt.json"
    bound = {}
    got = runner._test_receipt(
        path,
        sha(path),
        "runner",
        "recipe",
        [tmp_path / "auditor.py", tmp_path / "tests.py"],
        Helper,
        bound,
        tmp_path,
    )
    assert got == (path, tmp_path / "stdout.txt") and len(bound) == 4


def test_auditor_contract_portable(runner, tmp_path):
    contract_fixture(tmp_path, runner)
    path = tmp_path / "contract.json"
    bound = {}
    got = runner._auditor_contract(tmp_path, path, sha(path), Helper, bound)
    assert got["readiness_only"] and got["candidate_audited"] is False
    assert len(bound) == 11


@pytest.mark.parametrize(
    "key,value",
    [
        ("exit_code", False),
        ("exit_code", 1),
        ("exit_code", None),
        ("terminal_completion_observed", False),
        ("tests_passed", True),
        ("tests_passed", 0),
        ("tests_skipped", False),
        ("tests_skipped", 1),
        ("gpu_used", True),
        ("passed", False),
        ("experiment_training_executed", True),
    ],
)
def test_bad_receipt_rejected_for_both_paths(runner, tmp_path, key, value):
    contract = contract_fixture(tmp_path, runner)
    receipt = Helper._read(tmp_path / "receipt.json")
    receipt[key] = value
    digest = write(tmp_path / "receipt.json", receipt)
    contract["test_receipt"]["sha256"] = digest
    pinned = write(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError, match="CPU test receipt"):
        runner._test_receipt(
            tmp_path / "receipt.json",
            digest,
            "runner",
            "recipe",
            [tmp_path / "auditor.py", tmp_path / "tests.py"],
            Helper,
            {},
            tmp_path,
        )
    with pytest.raises(ValueError, match="CPU test receipt"):
        runner._auditor_contract(tmp_path, tmp_path / "contract.json", pinned, Helper, {})


@pytest.mark.parametrize(
    "key,value",
    [("ready", False), ("schema_version", True), ("experiment", "other"), ("plan_sha256", "wrong")],
)
def test_bad_readiness_rejected(runner, tmp_path, key, value):
    contract = contract_fixture(tmp_path, runner)
    contract[key] = value
    digest = write(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError, match="auditor readiness contract"):
        runner._auditor_contract(tmp_path, tmp_path / "contract.json", digest, Helper, {})


def test_missing_readiness_rejected(runner, tmp_path):
    with pytest.raises(ValueError, match="required"):
        runner._auditor_contract(tmp_path, None, None, Helper, {})


@pytest.mark.parametrize("target", ["auditor.py", "tests.py", "stdout.txt", "receipt.json"])
def test_tampered_readiness_dependency_rejected(runner, tmp_path, target):
    contract_fixture(tmp_path, runner)
    path = tmp_path / "contract.json"
    digest = sha(path)
    (tmp_path / target).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner._auditor_contract(tmp_path, path, digest, Helper, {})


@pytest.mark.parametrize("where", ["auditor", "tests", "test_receipt"])
def test_escaped_readiness_path_rejected(runner, tmp_path, where):
    contract = contract_fixture(tmp_path, runner)
    contract[where]["path"] = "../outside.py"
    digest = write(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError, match="escapes repository"):
        runner._auditor_contract(tmp_path, tmp_path / "contract.json", digest, Helper, {})


def test_receipt_dependency_escape_rejected(runner, tmp_path):
    receipt = receipt_fixture(tmp_path)
    receipt["test_dependencies"] = {"../escape.py": "abc"}
    digest = write(tmp_path / "receipt.json", receipt)
    with pytest.raises(ValueError, match="escapes repository"):
        runner._test_receipt(
            tmp_path / "receipt.json", digest, "runner", "recipe", [], Helper, {}, tmp_path
        )


def test_auditor_dependency_missing_rejected(runner, tmp_path):
    contract = contract_fixture(tmp_path, runner)
    receipt = Helper._read(tmp_path / "receipt.json")
    receipt["test_dependencies"].pop("auditor.py")
    contract["test_receipt"]["sha256"] = write(tmp_path / "receipt.json", receipt)
    digest = write(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError, match="test dependency inventory"):
        runner._auditor_contract(tmp_path, tmp_path / "contract.json", digest, Helper, {})


def synthetic_model():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(4))
            self.config = SimpleNamespace(dim=4)

    return load("symmetric_affine_residual").attach(Model())


def step_model(model):
    import torch

    names = load("symmetric_affine_residual").PARAMETER_NAMES
    opt = torch.optim.AdamW([getattr(model, n) for n in names], lr=0.0003, weight_decay=0)
    sum(getattr(model, n).square().sum() + getattr(model, n).sum() for n in names).backward()
    opt.step()
    return opt


def test_three_parameter_optimizer_reload_exact(runner, mean, tmp_path):
    import torch

    model = synthetic_model()
    optimizer = step_model(model)
    report = runner._validate_optimizer(optimizer, model, 1)
    assert set(report["steps"]) == set(runner._PARAMETERS)
    path = tmp_path / "synthetic.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    child = synthetic_model()
    restored = torch.optim.AdamW([getattr(child, n) for n in runner._PARAMETERS])
    saved = torch.load(path, weights_only=True)
    child.load_state_dict(saved["model"])
    restored.load_state_dict(saved["optimizer"])
    assert mean._same_state(model.state_dict(), child.state_dict())
    assert mean._same_state(optimizer.state_dict(), restored.state_dict())
    assert runner._validate_optimizer(restored, child, 1) == report


@pytest.mark.parametrize("mutation", ["missing", "extra", "step", "nan", "shape", "dtype"])
def test_optimizer_final_contract_rejects_drift(runner, mutation):
    import torch

    model = synthetic_model()
    opt = step_model(model)
    p = getattr(model, runner._PARAMETERS[0])
    if mutation == "missing":
        del opt.state[p]
    if mutation == "extra":
        opt.state[model.base] = {}
    if mutation == "step":
        opt.state[p]["step"].fill_(2)
    if mutation == "nan":
        opt.state[p]["exp_avg"].fill_(float("nan"))
    if mutation == "shape":
        opt.state[p]["exp_avg"] = torch.zeros(1)
    if mutation == "dtype":
        opt.state[p]["exp_avg"] = opt.state[p]["exp_avg"].double()
    with pytest.raises(ValueError, match="optimizer"):
        runner._validate_optimizer(opt, model, 1)


def test_immutable_output(mean, tmp_path):
    path = tmp_path / "summary.json"
    mean._write(path, {"complete": False})
    with pytest.raises(FileExistsError):
        mean._write(path, {"complete": True})
    with pytest.raises(ValueError, match="immutable"):
        mean._refuse(path)


def test_early_execution_failure_preserves_receipts(runner, mean, tmp_path):
    def fail():
        raise RuntimeError("synthetic module failure")

    helper = SimpleNamespace(_modules=fail, _failure_safe=lambda value: value, _sha=sha)
    summary = {"complete": False, "acceptance": False}
    with pytest.raises(RuntimeError, match="synthetic module failure"):
        runner._execute(tmp_path, tmp_path / "summary.json", {}, None, mean, helper, summary, None)
    saved = Helper._read(tmp_path / "summary.json")
    train = Helper._read(tmp_path / "training.json")
    assert saved["complete"] is False and train["complete"] is False
    assert train["completed_updates"] == 0 and "error" in saved and "error" in train
    assert saved["artifact_sha256"]["training.json"] == sha(tmp_path / "training.json")


def test_import_has_no_eager_torch():
    import ast

    tree = ast.parse((ROOT / "scripts/refit_symmetric_affine_v3.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name != "torch" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torch")


def metadata_fixture(runner):
    implementation = {
        "runner_sha256": "runner",
        "scorer_sha256": runner._AFFINE,
        "mean_helper_sha256": runner._MEAN,
        "budget_helper_sha256": runner._BUDGET,
        "bilinear_helper_sha256": runner._SCORER,
    }
    metadata = {
        "implementation_identity": implementation,
        "runner_sha256": "runner",
        "scorer_sha256": runner._AFFINE,
        "architecture": runner._ARCHITECTURE,
        "objective": runner._OBJECTIVE,
        "evaluator": runner._EVALUATOR,
        "inference": runner._INFERENCE,
        "trainable_parameters": list(runner._PARAMETERS),
        "standard_serving_supported": False,
        "input_adapter_identity": runner._INPUT_ADAPTER,
        "input_amendment_sha256": runner._AMENDMENT,
        "repair_declaration_sha256": runner._REPAIR,
        "residual_updates": 8000,
        "parent_training_steps": 2000,
        "parent_checkpoint_sha256": (
            "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
        ),
    }
    config = {
        "implementation_identity": implementation,
        "affine_residual_architecture": runner._ARCHITECTURE,
    }
    identity = {"evaluator_version": runner._EVALUATOR}
    corpus = {"fixture": "synthetic"}
    raw = {
        "global_step": 8000,
        "experiment_identity": identity,
        "config": config,
        "training_metadata": metadata,
        "corpus_identity": corpus,
        "split_hash": "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d",
    }
    return SimpleNamespace(global_step=8000, metadata=raw), identity, config, metadata, corpus


def test_child_identity_contract(runner):
    runner._validate_child(*metadata_fixture(runner))


@pytest.mark.parametrize(
    "key,value",
    [
        ("architecture", "old"),
        ("objective", "old"),
        ("inference", "old"),
        ("evaluator", "old"),
        ("trainable_parameters", ["symmetric_bilinear_residual"]),
        ("standard_serving_supported", True),
        ("residual_updates", 2000),
        ("parent_training_steps", 8000),
        ("parent_checkpoint_sha256", "old"),
    ],
)
def test_child_metadata_rejects_wrong_lineage(runner, key, value):
    args = metadata_fixture(runner)
    args[3][key] = value
    with pytest.raises(ValueError):
        runner._validate_child(*args)


def test_failed_optimizer_payload_preserved(runner, mean, tmp_path):
    import torch

    model = synthetic_model()
    optimizer = step_model(model)
    training = {
        "completed_updates": 0,
        "optimizer_steps_attempted": 1,
        "error": "post-step failure",
    }
    runner._preserve_failure(tmp_path, model, optimizer, training, mean, Helper)
    metadata = Helper._read(tmp_path / "failure-state.json")
    payload = torch.load(tmp_path / "failure-state.pt", weights_only=True)
    assert metadata["sha256"] == sha(tmp_path / "failure-state.pt")
    assert metadata["resume_allowed"] is False and payload["diagnostic_only"] is True
    assert payload["completed_updates"] == 0 and payload["optimizer_steps_attempted"] == 1
    assert mean._same_state(payload["model_state"], model.state_dict())
    assert mean._same_state(payload["optimizer_state"], optimizer.state_dict())
    with pytest.raises(FileExistsError):
        runner._preserve_failure(tmp_path, model, optimizer, training, mean, Helper)


def test_output_admission_allows_preregistered_auditor(mean, tmp_path):
    (tmp_path / "independent-audit.py").write_text(
        "# synthetic preregistered auditor", encoding="utf-8"
    )
    mean._refuse(tmp_path / "summary.json")
    (tmp_path / "training.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        mean._refuse(tmp_path / "summary.json")


def evaluation_fixture():
    import torch

    affine = load("symmetric_affine_residual")
    bilinear = load("refit_bilinear_residual")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(dim=256)
            self.token_embedding = torch.nn.Embedding(2049, 256)
            self.symmetric_relation_projection = torch.nn.Linear(256, 256, bias=False)
            with torch.no_grad():
                self.symmetric_relation_projection.weight.zero_()

    model = affine.attach(Model())
    records = []
    rows = []
    for i in range(14):
        dim = 33 if i % 3 == 0 else 32
        subject = 1024 + i
        target = 1050 + i
        prompt = [1, subject, dim, 34, 3]
        record = SimpleNamespace(
            subject=f"PKM_{i}",
            dimension="COLOR" if dim == 33 else "TYPE",
            input_ids=[*prompt, target, 2],
        )
        group = "COLOR" if dim == 33 else "TYPE_single" if i % 3 == 1 else "TYPE_dual"
        records.append(record)
        rows.append(
            {
                "index": i,
                "subject": record.subject,
                "dimension": record.dimension,
                "prompt_ids": prompt,
                "group": group,
                "expected_set_ids": [target],
                "composition": {"selected_set_ids": [target]},
            }
        )
    parent = {
        "responses": [
            {"symmetric_relation_logits": [0.0] * 1025, "selected_set_ids": []} for _ in rows
        ],
        "product_token_ids": list(range(1024, 2049)),
    }
    return model, records, rows, parent, affine, bilinear


def test_zero_replay_and_affine_final_evaluation(runner, mean, tmp_path):
    import torch

    model, records, rows, parent, affine, bilinear = evaluation_fixture()
    before = mean._state_hashes(model)
    result = runner._zero_replay(
        model, records, parent, mean, Helper, tmp_path / "zero.json", affine, bilinear
    )
    assert result["complete"] and result["batch_sizes"] == [8, 6]
    assert result["logits_sha256"] == result["parent_logits_sha256"]
    assert mean._state_hashes(model) == before
    with torch.no_grad():
        model.symmetric_affine_bias.copy_(torch.tensor([100.0, -100.0]))
    result = runner._evaluate(
        model, records, rows, mean, Helper, tmp_path / "child.json", affine, bilinear, parent
    )
    assert result["complete"] and result["query_count"] == 14
    for row in result["responses"]:
        selected = row["selected_set_ids"]
        assert len(selected) == (0 if row["dimension"] == "COLOR" else 1024)
        assert row["prompt_ids"][1] not in selected
        assert selected == sorted(set(selected))
        assert row["metrics"]["serialization_compatible"] is False
    assert result["aggregate"]["serialization_compatible"] == 0


def test_zero_replay_failure_is_immutable(runner, mean, tmp_path):
    import torch

    model, records, _rows, parent, affine, bilinear = evaluation_fixture()
    with torch.no_grad():
        model.symmetric_affine_bias.fill_(1)
    with pytest.raises(ValueError, match="parent replay"):
        runner._zero_replay(
            model, records, parent, mean, Helper, tmp_path / "zero.json", affine, bilinear
        )
    saved = Helper._read(tmp_path / "zero.json")
    assert saved["complete"] is False and "failed_observation" in saved and "error" in saved


def test_final_evaluation_alignment_failure_recorded(runner, mean, tmp_path):
    model, records, rows, parent, affine, bilinear = evaluation_fixture()
    rows[0]["subject"] = "mismatch"
    with pytest.raises(ValueError, match="query identity"):
        runner._evaluate(
            model, records, rows, mean, Helper, tmp_path / "child.json", affine, bilinear, parent
        )
    saved = Helper._read(tmp_path / "child.json")
    assert saved["complete"] is False and saved["query_count"] == 0 and "error" in saved


def test_exact_optimizer_state_distinguishes_signed_zero(runner):
    import torch

    left = {"state": {0: {"exp_avg": torch.tensor([0.0])}}}
    right = {"state": {0: {"exp_avg": torch.tensor([-0.0])}}}
    assert not runner._exact_state(left, right)
    assert runner._exact_state(left, copy.deepcopy(left))
    assert not runner._exact_state({"flag": False}, {"flag": 0})


@pytest.mark.parametrize("value", [None, "another-source"])
def test_auditor_tested_script_identity_required(runner, tmp_path, value):
    contract = contract_fixture(tmp_path, runner)
    receipt = Helper._read(tmp_path / "receipt.json")
    if value is None:
        receipt.pop("tested_script_sha256")
    else:
        receipt["tested_script_sha256"] = value
    contract["test_receipt"]["sha256"] = write(tmp_path / "receipt.json", receipt)
    digest = write(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError, match="auditor tested source identity"):
        runner._auditor_contract(tmp_path, tmp_path / "contract.json", digest, Helper, {})


def partition_fixture(ntrain=3, nval=3):
    tokens = [f"<reserved_{i}>" for i in range(2049)]
    classes = ["reserved"] * 2049
    for i, name in {
        0: "PAD",
        1: "BOS",
        2: "EOS",
        5: "ANSWER",
        32: "TYPE",
        33: "COLOR",
        34: "SAME",
    }.items():
        tokens[i] = name
    for i in range(1024, 2049):
        tokens[i] = f"PKM_{i}"
        classes[i] = "entity"
    rows = []
    for i in range(ntrain + nval):
        subject = 1024 + i // 2
        dim = 32 + i % 2
        prompt = [1, subject, dim, 34, 5]
        targets = [2048, 2047]
        rows.append(
            {
                "index": i if i < ntrain else i - ntrain,
                "subject": tokens[subject],
                "dimension": tokens[dim],
                "prompt_ids": prompt,
                "expected_set_ids": targets,
            }
        )
    train = {"partition": "train", "query_count": ntrain, "queries": rows[:ntrain]}
    parent = {
        "complete": True,
        "query_count": nval,
        "responses": rows[ntrain:],
        "product_token_ids": list(range(1024, 2049)),
    }
    queries = {
        name: [{k: r[k] for k in ("subject", "dimension", "prompt_ids")} for r in group]
        for name, group in (("train", rows[:ntrain]), ("validation", rows[ntrain:]))
    }
    vocab = {"schema_version": 2, "tokens": tokens, "classes": classes}
    return train, parent, queries, vocab


def test_selected_partition_encoding_order_and_masks(runner):
    source = partition_fixture()
    before = copy.deepcopy(source)
    partitions = runner._selected_partitions(*source, expected_train=3, expected_validation=3)
    assert source == before
    for name, records in partitions.items():
        for record, query in zip(records, source[2][name], strict=True):
            assert list(record.input_ids[:5]) == query["prompt_ids"]
            assert record.input_ids[5:] == (2047, 2048, 2)
            assert record.labels[:5] == (-100,) * 5
            assert record.labels[5:] == record.input_ids[5:]
    assert set(partitions) == {"train", "validation"}


@pytest.mark.parametrize(
    "mutation",
    [
        "partition",
        "train_count",
        "val_count",
        "index",
        "bool_index",
        "subject",
        "dimension",
        "prompt",
        "numeric",
        "target_duplicate",
        "self_target",
        "empty",
        "invalid_target",
        "order",
        "duplicate_query",
        "leak",
        "columns",
        "vocab",
    ],
)
def test_selected_partition_rejects_malformed_or_swapped(runner, mutation):
    train, parent, queries, vocab = partition_fixture()
    row = train["queries"][0]
    if mutation == "partition":
        train["partition"] = "test"
    if mutation == "train_count":
        train["query_count"] = 2
    if mutation == "val_count":
        parent["query_count"] = 2
    if mutation == "index":
        row["index"] = 2
    if mutation == "bool_index":
        row["index"] = False
    if mutation == "subject":
        row["subject"] = "PKM_wrong"
    if mutation == "dimension":
        row["dimension"] = "BIOME"
    if mutation == "prompt":
        row["prompt_ids"][4] = 3
    if mutation == "numeric":
        row["prompt_ids"][0] = True
    if mutation == "target_duplicate":
        row["expected_set_ids"] = [2048, 2048]
    if mutation == "self_target":
        row["expected_set_ids"] = [row["prompt_ids"][1]]
    if mutation == "empty":
        row["expected_set_ids"] = []
    if mutation == "invalid_target":
        row["expected_set_ids"] = [2]
    if mutation == "order":
        queries["train"].reverse()
    if mutation == "duplicate_query":
        train["queries"][1] = copy.deepcopy(row)
        train["queries"][1]["index"] = 1
        queries["train"][1] = queries["train"][0]
    if mutation == "leak":
        parent["responses"][0] = copy.deepcopy(row)
        queries["validation"][0] = queries["train"][0]
    if mutation == "columns":
        parent["product_token_ids"].reverse()
    if mutation == "vocab":
        vocab["tokens"][5] = "WRONG"
    with pytest.raises(ValueError):
        runner._selected_partitions(train, parent, queries, vocab, 3, 3)


def test_validation_membership_cannot_change_training_records(runner):
    inputs = partition_fixture()
    original = runner._selected_partitions(*inputs, 3, 3)
    inputs[1]["responses"][0]["expected_set_ids"] = [2046]
    changed = runner._selected_partitions(*inputs, 3, 3)
    assert original["train"] == changed["train"]
    assert original["validation"] != changed["validation"]


def test_archived_bce_loss_and_score_gradient_permutation_invariance():
    import ast
    import zipfile

    import torch

    archive = ROOT / "tests/fixtures/projection_refit_runtime_source.zip"
    with zipfile.ZipFile(archive) as opened:
        tree = ast.parse(opened.read("src/plm/model/layers.py").decode())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prompt_set_loss"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "Tensor": torch.Tensor,
        "torch": torch,
        "F": torch.nn.functional,
        "ENTITY_BASE": 1024,
    }
    exec(compile(module, str(archive) + "!_prompt_set_loss", "exec"), namespace)
    loss_fn = namespace["_prompt_set_loss"]
    scores = torch.linspace(-2, 2, 24).reshape(3, 8).requires_grad_()
    labels = torch.tensor([[-100, 1025, 1027, 2], [-100, 1026, 1028, 2], [-100, 1027, 1029, 2]])
    subjects = torch.tensor([0, 1, 2])
    reference = loss_fn(scores, labels, excluded_ids=subjects)
    gradient = torch.autograd.grad(reference, scores)[0]
    for order in ([3, 2, 1, 0], [1, 0, 3, 2], [2, 1, 0, 3]):
        value = loss_fn(scores, labels[:, order], excluded_ids=subjects)
        actual = torch.autograd.grad(value, scores)[0]
        assert torch.equal(reference, value) and torch.equal(gradient, actual)
    repeated = torch.cat((labels, labels[:, 1:2]), dim=1)
    value = loss_fn(scores, repeated, excluded_ids=subjects)
    assert torch.equal(reference, value)
    assert torch.equal(gradient, torch.autograd.grad(value, scores)[0])


def test_runner_has_no_whole_corpus_loader_or_inherited_preflight():
    import ast

    source = (ROOT / "scripts/refit_symmetric_affine_v3.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "load_inference_runtime",
        "load_corpus_records",
        "require_valid_artifacts",
        "split_queries",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert all(alias.name not in forbidden for alias in node.names)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden | {"_preflight"}


@pytest.mark.parametrize(
    "path", ["data/processed/records.jsonl", "data/graph.sqlite", "other/protected.jsonl"]
)
def test_receipt_cannot_open_corpus_or_graph(runner, tmp_path, path):
    with pytest.raises(ValueError, match="forbidden corpus or graph"):
        runner._resolve(tmp_path, path)


@pytest.mark.parametrize("key", ["input_adapter_identity", "input_amendment_sha256"])
def test_receipt_and_readiness_require_input_amendment(runner, tmp_path, key):
    contract = contract_fixture(tmp_path, runner)
    receipt = Helper._read(tmp_path / "receipt.json")
    receipt[key] = "wrong"
    with pytest.raises(ValueError, match="test input contract"):
        runner._receipt_checks(receipt)
    contract[key] = "wrong"
    digest = write(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError, match="auditor readiness contract"):
        runner._auditor_contract(tmp_path, tmp_path / "contract.json", digest, Helper, {})


def test_preflight_opens_only_selected_allowlist(runner, mean, tmp_path, monkeypatch):
    train, parent, queries, vocab = partition_fixture(1637, 222)
    for row in parent["responses"]:
        row["group"] = (
            "COLOR"
            if row["dimension"] == "COLOR"
            else "TYPE_single"
            if row["index"] % 4
            else "TYPE_dual"
        )
        row["selected_set_ids"] = row["expected_set_ids"]
        row["metrics"] = mean._metrics(row["selected_set_ids"], row["expected_set_ids"])
    historical = copy.deepcopy(parent)
    monkeypatch.setattr(runner, "_MEMBERSHIP", mean._digest(train))
    new_recipe, old_recipe = recipe_pair()
    artifacts = dict.fromkeys(
        ("train-membership.json", "parent.json", "child.json", "training.json"), "hash"
    )
    artifacts["checkpoint-final.pt"] = runner._CHILD
    identity = {"evaluator_version": "plm-train-causal-v2"}
    corpus = {"vocabulary_hash": "vocab"}
    parent_dir = tmp_path / "runs/national_dex_continuation_control_s1729_v1"
    summary = {
        "complete": True,
        "final_identity_check": True,
        "script_sha256": runner._BASE,
        "split_hash": mean._SPLIT,
        "architecture": "historical",
        "objective": "plm-bilinear-residual-balanced-bce-v1",
        "evaluator": "plm-bilinear-budget8000-screen-v1",
        "artifact_sha256": artifacts,
        "parent_training_identity": identity,
        "corpus_identity": corpus,
        "queries": queries,
        "input_sha256": {
            str(parent_dir / n): "hash" for n in ("run.json", "checkpoint-final.pt.json")
        },
        "recipe_sha256": "hash",
        "inherited_architecture": {},
        "environment": {},
    }
    forbidden = tmp_path / "data/processed/records.jsonl"
    summary["input_sha256"][str(forbidden)] = "must-never-open"
    audit = {
        "complete": True,
        "audit_passed": True,
        "summary_sha256": runner._SUMMARY,
        "script_sha256": runner._AUDITOR,
    }
    decision = {
        "evidence_accepted": True,
        "fixed_quality_gate_passed": True,
        "summary_sha256": runner._SUMMARY,
        "audit_sha256": runner._AUDIT,
    }
    old = tmp_path / "runs/learning/bilinear-budget8000-v1"
    objects = {
        old / "summary.json": summary,
        old / "independent-audit.json": audit,
        old / "decision.json": decision,
        old / "train-membership.json": train,
        old / "parent.json": parent,
        old / "child.json": historical,
        old / "training.json": {"parent_state_before": {}},
        old / "summary.recipe.json": old_recipe,
        parent_dir / "run.json": {"identity": identity, "config": {}},
        parent_dir / "checkpoint-final.pt.json": {
            "experiment_identity": identity,
            "global_step": 2000,
            "checkpoint_hash": mean._PARENT,
            "split_hash": mean._SPLIT,
            "training_metadata": {"objective": "causal-next-token-v1"},
            "corpus_identity": corpus,
        },
        tmp_path / "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json": vocab,
        tmp_path / "configs/experiments/symmetric_affine8000_v3.json": new_recipe,
    }
    calls = []

    def bind(path, digest, inputs):
        path = Path(path).resolve()
        assert path != forbidden and path.suffix not in (".db", ".sqlite", ".jsonl")
        assert "data" not in path.parts or path.name == "vocabulary.json"
        calls.append(path)
        inputs[str(path)] = digest
        return path

    def read(path):
        assert Path(path) in objects, f"Unexpected read: {path}"
        return objects[Path(path)]

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("whole-corpus loader called")

    helper = SimpleNamespace(
        _bind=bind,
        _read=read,
        _sha=lambda p: "hash",
        _SOURCE="source",
        _CONFIGS="configs",
        _archive_members=lambda paths: [],
        _preflight=forbidden_loader,
    )
    bilinear = SimpleNamespace(_architecture=lambda: "historical", _preflight=forbidden_loader)
    result = runner._preflight(
        tmp_path,
        tmp_path / "docs/experiments/2026-09-26-symmetric-affine-plan.md",
        tmp_path / "configs/experiments/symmetric_affine8000_v3.json",
        bilinear,
        tmp_path / "scripts/refit_bilinear_residual.py",
        mean,
        tmp_path / "scripts/refit_membership_projection.py",
        helper,
        tmp_path / "runs/learning/wide-first-choice-v1/summary.script.py",
        SimpleNamespace(_preflight=forbidden_loader),
        tmp_path / "scripts/refit_bilinear_budget.py",
    )
    assert (
        len(result["partitions"]["train"]) == 1637
        and len(result["partitions"]["validation"]) == 222
    )
    assert forbidden not in calls
    assert result["input_adapter"]["whole_corpus_revalidated"] is False
    assert set(result["input_adapter"]["partition_sha256"]) == {"train", "validation"}


def test_receipt_extra_dependency_rejected_before_any_dependency_open(runner, tmp_path):
    receipt = receipt_fixture(tmp_path)
    receipt["test_dependencies"]["unapproved.py"] = "unknown"
    digest = write(tmp_path / "receipt.json", receipt)
    opened = []

    class Guard(Helper):
        @staticmethod
        def _bind(path, digest, inputs):
            opened.append(Path(path).name)
            assert Path(path).name == "receipt.json", "dependency opened before inventory admission"
            return Helper._bind(path, digest, inputs)

    with pytest.raises(ValueError, match="test dependency inventory"):
        runner._test_receipt(
            tmp_path / "receipt.json",
            digest,
            "runner",
            "recipe",
            [tmp_path / "auditor.py", tmp_path / "tests.py"],
            Guard,
            {},
            tmp_path,
        )
    assert opened == ["receipt.json"]


def test_receipt_forbidden_dependency_rejected_before_any_dependency_open(runner, tmp_path):
    receipt = receipt_fixture(tmp_path)
    receipt["test_dependencies"]["data/protected.jsonl"] = "unknown"
    digest = write(tmp_path / "receipt.json", receipt)
    opened = []

    class Guard(Helper):
        @staticmethod
        def _bind(path, digest, inputs):
            opened.append(Path(path).name)
            assert Path(path).name == "receipt.json"
            return Helper._bind(path, digest, inputs)

    with pytest.raises(ValueError, match="forbidden corpus or graph"):
        runner._test_receipt(
            tmp_path / "receipt.json",
            digest,
            "runner",
            "recipe",
            [tmp_path / "auditor.py", tmp_path / "tests.py"],
            Guard,
            {},
            tmp_path,
        )
    assert opened == ["receipt.json"]


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "vocabulary",
        "objective",
        "step",
        "config",
        "evaluator",
        "corpus",
        "train_optimizer",
        "train_seq_len",
        "train_max_steps",
        "root_in_checkpoint",
        "checkpoint_hash",
        "sidecar_metadata",
        "resolved_model",
        "experiment_corpus",
    ],
)
def test_selected_runtime_uses_only_checkpoint_and_vocabulary(
    runner, tmp_path, monkeypatch, mutation
):
    import sys
    import zipfile
    from types import ModuleType

    with zipfile.ZipFile(ROOT / "tests/fixtures/projection_refit_runtime_source.zip") as archive:
        config_source = archive.read("src/plm/config.py").decode()
    config_module = ModuleType("synthetic_archived_config")
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    exec(compile(config_source, "archived/config.py", "exec"), config_module.__dict__)
    config = config_module.RootConfig.model_validate(
        {
            "model": {"dim": 256, "n_heads": 8, "n_kv_heads": 2},
            "eval": {"device": "cuda"},
            "train": {"seq_len": 512, "max_steps": 2000, "optimizer": {"lr": 0.0003}},
        }
    )
    model_config = config.model.model_copy(update={"vocab_size": 2049, "max_seq_len": 512})
    corpus = {"vocabulary_hash": "v", "graph_hash": "g", "records_hash": "r"}
    identity = {"evaluator_version": "plm-train-causal-v2", "split_hash": "split", **corpus}
    metadata = {
        "objective": "causal-next-token-v1",
        "model_config": model_config.model_dump(mode="json"),
    }
    context = {
        "parent_run": {
            "config": config.model_dump(mode="json"),
            "identity": identity,
        },
        "parent_sidecar": {"training_metadata": metadata},
        "vocabulary_path": tmp_path / "vocabulary.json",
        "inherited_architecture": model_config.model_dump(mode="json"),
        "parent": tmp_path / "checkpoint.pt",
        "corpus_identity": corpus,
        "partitions": {"train": ("train",), "validation": ("validation",)},
    }
    raw = {
        "config": copy.deepcopy(context["parent_run"]["config"]["train"]),
        "corpus_identity": corpus,
        "training_metadata": metadata,
    }
    state = SimpleNamespace(
        global_step=2000,
        checkpoint_hash="e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1",
        metadata=raw,
    )
    calls = []

    class Vocabulary:
        @classmethod
        def load(cls, path):
            assert path == context["vocabulary_path"]
            calls.append("vocabulary")
            return cls()

        def __len__(self):
            return 2049

        def content_hash(self):
            return "wrong" if mutation == "vocabulary" else "v"

    class Model:
        def to(self, device):
            assert device == "cuda"
            calls.append("device")
            return self

        def eval(self):
            return self

    def load_checkpoint(path, model, **kwargs):
        assert path == context["parent"] and kwargs == {"map_location": "cpu", "restore_rng": False}
        calls.append("checkpoint")
        return state

    def validate(state, **kwargs):
        assert kwargs == {"identity": identity, "split_hash": "split"}
        calls.append("identity")

    modules = {
        "plm.config": {
            "RootConfig": config_module.RootConfig,
            "ModelConfig": config_module.ModelConfig,
        },
        "plm.experimentation.identity": {
            "ExperimentIdentity": SimpleNamespace(from_dict=lambda value: value)
        },
        "plm.model.architecture": {"build_model": lambda cfg: Model()},
        "plm.protocol": {"Vocabulary": Vocabulary},
        "plm.training.checkpoint": {
            "load_checkpoint": load_checkpoint,
            "validate_checkpoint_identity": validate,
        },
    }
    # A closed synthetic package tree cannot import corpus, graph, split or serving loaders.
    for name in ("plm", "plm.model", "plm.training", "plm.experimentation"):
        mod = ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    for name, attributes in modules.items():
        mod = ModuleType(name)
        mod.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, mod)
    if mutation == "objective":
        metadata["objective"] = "old"
    if mutation == "step":
        state.global_step = 0
    if mutation == "config":
        raw["config"] = {}
    if mutation == "evaluator":
        identity["evaluator_version"] = "old"
    if mutation == "corpus":
        raw["corpus_identity"] = {}
    if mutation == "train_optimizer":
        raw["config"]["optimizer"]["lr"] = 0.0007
    if mutation == "train_seq_len":
        raw["config"]["seq_len"] = 256
    if mutation == "train_max_steps":
        raw["config"]["max_steps"] = 1999
    if mutation == "root_in_checkpoint":
        raw["config"] = context["parent_run"]["config"]
    if mutation in {"train_optimizer", "train_seq_len", "train_max_steps", "root_in_checkpoint"}:
        with pytest.raises(ValueError, match="TrainConfig differs"):
            runner._load_selected_runtime(context)
        assert "device" not in calls
        return
    if mutation == "checkpoint_hash":
        state.checkpoint_hash = "0" * 64
    if mutation == "sidecar_metadata":
        raw["training_metadata"] = {**metadata, "train_loss": 1.0}
    if mutation == "resolved_model":
        metadata["model_config"]["dim"] = 128
    if mutation == "experiment_corpus":
        identity["graph_hash"] = "wrong"
    if mutation is not None:
        with pytest.raises(ValueError):
            runner._load_selected_runtime(context)
    else:
        runtime = runner._load_selected_runtime(context)
        assert runtime.split.train == ("train",) and runtime.split.validation == ("validation",)
        assert not hasattr(runtime.split, "test")
        assert calls == ["vocabulary", "checkpoint", "identity", "device"]
        # This is the exact equality that failed in frozen v1.
        assert raw["config"] != context["parent_run"]["config"]
        assert raw["config"] == context["parent_run"]["config"]["train"]


@pytest.mark.parametrize("entry", ["primary", "contract", "auditor", "tests", "test_receipt"])
def test_production_readiness_paths_reject_unknown_before_open(tmp_path, entry):
    # Fresh module retains the exact production filenames (no fixture overrides).
    module = load("refit_symmetric_affine_v3")
    opened = []
    protected = tmp_path / "protected.json"
    protected.write_text("protected fixture sentinel", encoding="utf-8")
    contract_path = tmp_path / module._AUDITOR_CONTRACT_PATH
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "experiment": "symmetric-affine8000-v3",
        "ready": True,
        "plan_sha256": module._PLAN,
        "input_adapter_identity": module._INPUT_ADAPTER,
        "input_amendment_sha256": module._AMENDMENT,
        "repair_declaration_sha256": module._REPAIR,
        **{k: {"path": v, "sha256": "unused"} for k, v in module._AUDITOR_PART_PATHS.items()},
    }
    if entry in module._AUDITOR_PART_PATHS:
        contract[entry] = {"path": "protected.json", "sha256": sha(protected)}
    pinned = write(contract_path, contract)

    class Guard(Helper):
        @staticmethod
        def _bind(path, digest, inputs):
            opened.append(Path(path))
            assert Path(path) == contract_path, "unadmitted file opened"
            return Helper._bind(path, digest, inputs)

    with pytest.raises(ValueError, match="path admission"):
        if entry == "primary":
            module._test_receipt(protected, sha(protected), "s", "r", [], Guard, {}, tmp_path)
        elif entry == "contract":
            module._auditor_contract(tmp_path, protected, sha(protected), Guard, {})
        else:
            module._auditor_contract(tmp_path, contract_path, pinned, Guard, {})
    assert opened == ([] if entry in ("primary", "contract") else [contract_path])


def test_production_readiness_exact_named_paths_pass(tmp_path):
    module = load("refit_symmetric_affine_v3")
    refs = {k: tmp_path / v for k, v in module._AUDITOR_PART_PATHS.items()}
    dependencies = [
        refs["auditor"],
        refs["tests"],
        *(tmp_path / p for p in module._AUDITOR_EXTRA_DEPENDENCIES),
    ]
    for path in dependencies:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic dependency", encoding="utf-8")
    refs["test_receipt"].parent.mkdir(parents=True, exist_ok=True)
    stdout = refs["test_receipt"].parent / "stdout.txt"
    stdout.write_text("4 passed", encoding="utf-8")
    receipt = {
        "passed": True,
        "gpu_used": False,
        "experiment_training_executed": False,
        "exit_code": 0,
        "terminal_completion_observed": True,
        "tests_passed": 4,
        "tests_skipped": 0,
        "input_adapter_identity": module._INPUT_ADAPTER,
        "input_amendment_sha256": module._AMENDMENT,
        "repair_declaration_sha256": module._REPAIR,
        "tested_script_sha256": sha(refs["auditor"]),
        "test_dependencies": {str(p.relative_to(tmp_path)): sha(p) for p in dependencies},
        "stdout": {"path": str(stdout.relative_to(tmp_path)), "sha256": sha(stdout)},
    }
    write(refs["test_receipt"], receipt)
    contract = {
        "schema_version": 1,
        "experiment": "symmetric-affine8000-v3",
        "ready": True,
        "plan_sha256": module._PLAN,
        "input_adapter_identity": module._INPUT_ADAPTER,
        "input_amendment_sha256": module._AMENDMENT,
        "repair_declaration_sha256": module._REPAIR,
        **{k: {"path": str(p.relative_to(tmp_path)), "sha256": sha(p)} for k, p in refs.items()},
    }
    path = tmp_path / module._AUDITOR_CONTRACT_PATH
    digest = write(path, contract)
    result = module._auditor_contract(tmp_path, path, digest, Helper, {})
    assert result["readiness_only"] and result["candidate_audited"] is False


def test_v2_recipe_changes_only_execution_identity_and_repair_binding(runner):
    original = json.loads(
        (ROOT / "configs/experiments/symmetric_affine8000_v1.json").read_text(encoding="utf-8")
    )
    revised = json.loads(
        (ROOT / "configs/experiments/symmetric_affine8000_v3.json").read_text(encoding="utf-8")
    )
    changes = {k for k in original.keys() | revised.keys() if original.get(k) != revised.get(k)}
    assert changes == {"experiment", "evaluator", "repair_declaration_sha256"}
    assert revised["repair_declaration_sha256"] == runner._REPAIR


@pytest.mark.parametrize("where", ["recipe", "receipt", "readiness"])
def test_repair_declaration_binding_required(runner, tmp_path, where):
    if where == "recipe":
        new, old = recipe_pair()
        new.pop("repair_declaration_sha256")
        with pytest.raises(ValueError, match="fixed affine recipe"):
            runner._recipe(new, old)
    elif where == "receipt":
        receipt = receipt_fixture(tmp_path)
        receipt.pop("repair_declaration_sha256")
        with pytest.raises(ValueError, match="test input contract"):
            runner._receipt_checks(receipt)
    else:
        contract = contract_fixture(tmp_path, runner)
        contract.pop("repair_declaration_sha256")
        digest = write(tmp_path / "contract.json", contract)
        with pytest.raises(ValueError, match="auditor readiness contract"):
            runner._auditor_contract(tmp_path, tmp_path / "contract.json", digest, Helper, {})


@pytest.mark.parametrize("missing_mean", [False, True])
def test_full_execute_lifecycle_with_bounded_synthetic_fit(
    runner, mean, tmp_path, monkeypatch, missing_mean
):
    """Real CPU lifecycle; one real optimizer step and simulated 8000-step trace."""
    import ast
    import inspect
    import sys
    import zipfile
    from types import ModuleType

    import torch

    affine = load("symmetric_affine_residual")
    bilinear = load("refit_bilinear_residual")
    trace = []
    with zipfile.ZipFile(ROOT / "tests/fixtures/projection_refit_runtime_source.zip") as archive:
        sources = {
            n: archive.read(n).decode()
            for n in (
                "src/plm/config.py",
                "src/plm/experimentation/identity.py",
                "src/plm/training/checkpoint.py",
                "src/plm/training/optim.py",
                "src/plm/training/data.py",
                "src/plm/model/interfaces.py",
                "src/plm/model/layers.py",
            )
        }
    for name in ("plm", "plm.model", "plm.training", "plm.experimentation", "plm.corpus"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    compiler = ModuleType("plm.corpus.compiler")
    compiler.CorpusRecord = SimpleNamespace
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    loaded = {}
    for name in (
        "plm.config",
        "plm.experimentation.identity",
        "plm.model.interfaces",
        "plm.training.checkpoint",
        "plm.training.optim",
        "plm.training.data",
    ):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        path = "src/" + name.replace(".", "/") + ".py"
        exec(compile(sources[path], "archived/" + path, "exec"), module.__dict__)
        loaded[name] = module
    config = loaded["plm.config"].ModelConfig.model_validate(
        {"dim": 256, "vocab_size": 2049, "max_seq_len": 512}
    )
    tree = ast.parse(sources["src/plm/model/layers.py"])
    loss_node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prompt_set_loss"
    )
    layer_module = ModuleType("plm.model.layers")
    layer_module.__dict__.update(
        Tensor=torch.Tensor, torch=torch, F=torch.nn.functional, ENTITY_BASE=1024
    )
    exec(
        compile(ast.Module(body=[loss_node], type_ignores=[]), "archived/loss", "exec"),
        layer_module.__dict__,
    )
    monkeypatch.setitem(sys.modules, layer_module.__name__, layer_module)
    loss_fn = layer_module._prompt_set_loss

    class SyntheticModel(torch.nn.Module):
        def __init__(self, model_config):
            super().__init__()
            self.config = model_config
            self.token_embedding = torch.nn.Embedding(2049, 256)
            self.symmetric_relation_projection = torch.nn.Linear(256, 256, bias=False)
            with torch.no_grad():
                self.token_embedding.weight.zero_()
                self.symmetric_relation_projection.weight.zero_()
            for i in range(91):
                self.register_buffer(f"frozen_fixture_{i}", torch.tensor([float(i)]))

        def forward(self, prompts):
            features = mean._features(self, prompts)
            return SimpleNamespace(
                symmetric_relation_logits=mean._head(
                    self.symmetric_relation_projection.weight, features
                )
            )

    def build_model(model_config):
        trace.append("build_reload_model")
        assert model_config == config
        return SyntheticModel(model_config)

    architecture = ModuleType("plm.model.architecture")
    architecture.build_model = build_model
    monkeypatch.setitem(sys.modules, architecture.__name__, architecture)

    def seed_everything(seed, deterministic=False):
        assert seed == 1729 and deterministic is False

    reproduction = ModuleType("plm.reproducibility")
    reproduction.seed_everything = seed_everything
    monkeypatch.setitem(sys.modules, reproduction.__name__, reproduction)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(sys, "path", list(sys.path))

    membership, parent, queries, vocabulary = partition_fixture(1637, 222)
    # Canonical authenticated membership fixture, never a repository dataset.
    for row in membership["queries"]:
        row["expected_set_ids"].sort()
    partitions = runner._selected_partitions(membership, parent, queries, vocabulary)
    for row in parent["responses"]:
        row["group"] = (
            "COLOR"
            if row["dimension"] == "COLOR"
            else "TYPE_single"
            if row["index"] % 4 == 1
            else "TYPE_dual"
        )
        row["symmetric_relation_logits"] = [0.0] * 1025
        row["selected_set_ids"] = []
        row["metrics"] = mean._metrics([], row["expected_set_ids"])
    model = SyntheticModel(config)
    assert len(model.state_dict()) == 93
    training_identity = {"protocol_version": "synthetic-protocol"}
    corpus_identity = {
        "graph_hash": "synthetic-graph",
        "records_hash": "synthetic-records",
        "vocabulary_hash": "synthetic-vocabulary",
    }
    runtime = SimpleNamespace(
        model=model,
        device="cpu",
        vocabulary=SimpleNamespace(pad_id=0),
        checkpoint_hash=mean._PARENT,
        training_identity=training_identity,
        corpus_identity=corpus_identity,
        split=SimpleNamespace(**partitions, split_hash=mean._SPLIT),
    )

    def selected_runtime(context):
        trace.append("load_synthetic_runtime")
        assert context["parent_state"] == mean._state_hashes(model)
        return runtime

    monkeypatch.setattr(runner, "_load_selected_runtime", selected_runtime)
    monkeypatch.setattr(runner, "_MEMBERSHIP", mean._digest(membership))
    monkeypatch.setattr(mean, "_git", lambda root: {"commit": "synthetic-source"})
    batch = loaded["plm.training.data"].collate_records(partitions["train"], pad_id=0)
    labels = batch.labels.masked_fill(~batch.attention_mask, -100)[:, 1:]
    subjects = batch.input_ids[:, 1] - 1024
    initial_loss = float(loss_fn(torch.zeros(1637, 1025), labels, excluded_ids=subjects))
    recipe = recipe_pair()[0]
    recipe["initial_training_loss"] = initial_loss
    context = {
        "recipe": recipe,
        "members": [],
        "historical_environment": {"fixture": "CPU"},
        "parent_run": {"identity": training_identity, "config": {"synthetic": True}},
        "parent_state": mean._state_hashes(model),
        "membership": membership,
        "wide": parent,
        "input_adapter": {"synthetic_membership_fixture": True},
        "inputs": {},
        "historical8000": copy.deepcopy(parent),
        "historical8000_child_sha256": "synthetic",
    }

    class ExecutionHelper(Helper):
        _SOURCE = "synthetic-source-archive"
        _CONFIGS = "synthetic-config-archive"

        @staticmethod
        def _modules(runtime=None):
            return {}

        @staticmethod
        def _extract(members, path):
            assert members == []
            (path / "src").mkdir(parents=True)
            return {}

        @staticmethod
        def _environment(torch_module):
            assert torch_module is torch
            return {"fixture": "CPU"}

        @staticmethod
        def _settings(torch_module):
            assert torch_module is torch
            return {}

        @staticmethod
        def _unchanged(hashes):
            assert hashes == {}

        @staticmethod
        def _runtime_unchanged(runtime, hashes):
            assert hashes == {}

    helper = ExecutionHelper
    real_evaluate = runner._evaluate

    # Retain the real function's exact parameter names, signature, and body.
    def tracked_evaluate(
        model,
        records,
        wide_rows,
        mean,
        helper,
        path,
        affine,
        bilinear,
        parent=None,
        batch_size=8,
        phase="child",
    ):
        assert mean is frozen_mean and helper is ExecutionHelper
        assert records is runtime.split.validation and wide_rows is context["wide"]["responses"]
        trace.append("evaluate_" + phase)
        return real_evaluate(
            model,
            records,
            wide_rows,
            mean,
            helper,
            path,
            affine,
            bilinear,
            parent=parent,
            batch_size=batch_size,
            phase=phase,
        )

    frozen_mean = mean
    assert inspect.signature(tracked_evaluate) == inspect.signature(real_evaluate)
    monkeypatch.setattr(runner, "_evaluate", tracked_evaluate)
    real_zero = runner._zero_replay

    def tracked_zero(model, records, parent, mean, helper, path, affine, bilinear, batch_size=8):
        trace.append("zero_replay")
        return real_zero(
            model, records, parent, mean, helper, path, affine, bilinear, batch_size=batch_size
        )

    assert inspect.signature(tracked_zero) == inspect.signature(real_zero)
    monkeypatch.setattr(runner, "_zero_replay", tracked_zero)
    real_attach = affine.attach

    def tracked_attach(model):
        trace.append("attach")
        return real_attach(model)

    monkeypatch.setattr(affine, "attach", tracked_attach)
    real_fit = affine.fit

    def bounded_fit(
        model,
        parent_scores,
        entities,
        subjects,
        dimension_tokens,
        labels,
        loss_fn,
        optimizer,
        updates,
        receipt,
        bilinear_fn,
    ):
        inspect.signature(real_fit).bind(
            model,
            parent_scores,
            entities,
            subjects,
            dimension_tokens,
            labels,
            loss_fn,
            optimizer,
            updates,
            receipt,
            bilinear_fn,
        )
        assert updates == 8000 and receipt == {}
        assert parent_scores.shape == (1637, 1025) and entities.shape == (1025, 256)
        assert subjects.shape == dimension_tokens.shape == (1637,) and labels.shape[0] == 1637
        assert loss_fn is layer_module._prompt_set_loss and bilinear_fn is bilinear._residual
        assert [n for n, p in model.named_parameters() if p.requires_grad] == list(
            runner._PARAMETERS
        )
        trace.append("fit_8000_simulated")
        params = [getattr(model, n) for n in runner._PARAMETERS]
        optimizer.zero_grad(set_to_none=True)
        sum(p.sum() for p in params).backward()
        # One real AdamW mutation, not evidence of 8000 numerical updates.
        optimizer.step()
        for parameter in params:
            optimizer.state[parameter]["step"].fill_(updates)
        receipt.update(
            complete=True,
            history=[
                {
                    "update": i,
                    "pre_update_loss": initial_loss,
                    "gradient_finite": True,
                    "parameters_finite": True,
                }
                for i in range(1, updates + 1)
            ],
            completed_updates=updates,
            optimizer_steps_attempted=updates,
            phase="complete",
            initial_pre_update_loss=initial_loss,
            final_post_update_loss=initial_loss,
            frozen_state_exact=True,
            synthetic_numerical_steps=1,
            synthetic_trajectory=True,
        )

    assert inspect.signature(bounded_fit) == inspect.signature(real_fit)
    monkeypatch.setattr(affine, "fit", bounded_fit)
    checkpoint_module = loaded["plm.training.checkpoint"]
    real_save = checkpoint_module.save_checkpoint
    real_load = checkpoint_module.load_checkpoint

    def save_checkpoint(
        path,
        model,
        *,
        optimizer=None,
        scheduler=None,
        global_step,
        config=None,
        identity=None,
        corpus_identity=None,
        split_hash=None,
        training_metadata=None,
    ):
        trace.append("save_checkpoint")
        return real_save(
            path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            config=config,
            identity=identity,
            corpus_identity=corpus_identity,
            split_hash=split_hash,
            training_metadata=training_metadata,
        )

    def load_checkpoint(
        path, model, *, optimizer=None, scheduler=None, map_location="cpu", restore_rng=True
    ):
        trace.append("load_checkpoint")
        return real_load(
            path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
            restore_rng=restore_rng,
        )

    for wrapper, original in ((save_checkpoint, real_save), (load_checkpoint, real_load)):
        original_signature = inspect.signature(original)
        unannotated = original_signature.replace(
            parameters=[
                p.replace(annotation=inspect.Parameter.empty)
                for p in original_signature.parameters.values()
            ],
            return_annotation=inspect.Signature.empty,
        )
        assert inspect.signature(wrapper) == unannotated

    monkeypatch.setattr(checkpoint_module, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(checkpoint_module, "load_checkpoint", load_checkpoint)
    summary = {
        "complete": False,
        "acceptance": False,
        "script_sha256": "synthetic-runner",
        "recipe_sha256": "synthetic-recipe",
        "snapshot_sha256": {},
    }
    execute = runner._execute
    if missing_mean:
        tree = ast.parse(
            (ROOT / "scripts/refit_symmetric_affine_v3.py").read_text(encoding="utf-8")
        )
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_execute")
        parent_call = next(
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_evaluate"
            and any(k.arg == "phase" for k in n.keywords)
        )
        assert isinstance(parent_call.args[3], ast.Name) and parent_call.args[3].id == "mean"
        del parent_call.args[3]
        namespace = dict(vars(runner))
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "missing-mean-regression", "exec"),
            namespace,
        )
        execute = namespace["_execute"]
        with pytest.raises(TypeError, match="bilinear"):
            execute(
                tmp_path,
                tmp_path / "summary.json",
                context,
                bilinear,
                mean,
                helper,
                summary,
                affine,
            )
        failed = Helper._read(tmp_path / "summary.json")
        assert failed["complete"] is False and not (tmp_path / "parent.json").exists()
        assert (tmp_path / "failure-state.pt").exists()
        assert trace == ["load_synthetic_runtime"]
        return
    execute(tmp_path, tmp_path / "summary.json", context, bilinear, mean, helper, summary, affine)
    saved = Helper._read(tmp_path / "summary.json")
    training = Helper._read(tmp_path / "training.json")
    assert trace == [
        "load_synthetic_runtime",
        "evaluate_parent",
        "attach",
        "zero_replay",
        "fit_8000_simulated",
        "save_checkpoint",
        "build_reload_model",
        "attach",
        "load_checkpoint",
        "evaluate_child",
    ]
    assert saved["complete"] is True and saved["final_identity_check"] is True
    assert training["reload_exact"] and training["optimizer_reload_exact"]
    assert (
        len(training["state_before"])
        == len(training["state_after"])
        == len(training["state_reloaded"])
        == 96
    )
    assert len(training["parent_state_before"]) == 93
    assert training["fit_complete"] and training["complete"] and training["synthetic_trajectory"]
    assert training["synthetic_numerical_steps"] == 1 and len(training["history"]) == 8000
    assert set(training["optimizer_final"]["steps"].values()) == {8000.0}
    assert all(saved["invariants"].values())
    assert (
        saved["gate"]["accepted"] is False and saved["gate"]["independent_audit_required"] is True
    )
    assert saved["gate"]["primary_checks_passed"] is False  # Synthetic model is not accurate.
    for name, digest in saved["artifact_sha256"].items():
        assert sha(tmp_path / name) == digest
    assert Helper._read(tmp_path / "parent.json")["query_count"] == 222
    assert Helper._read(tmp_path / "child.json")["query_count"] == 222
    assert Helper._read(tmp_path / "zero-replay.json")["batch_sizes"] == [8] * 27 + [6]
    assert (tmp_path / "comparisons.json").exists() and not (tmp_path / "failure-state.pt").exists()


def test_all_internal_runner_calls_bind_real_signatures(runner):
    import ast
    import inspect

    source = (ROOT / "scripts/refit_symmetric_affine_v3.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    signatures = {
        n.name: inspect.signature(getattr(runner, n.name))
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    checked = 0
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id not in signatures
        ):
            continue
        assert not any(isinstance(a, ast.Starred) for a in node.args), (
            "unresolved positional expansion"
        )
        assert not any(k.arg is None for k in node.keywords), "unresolved keyword expansion"
        signatures[node.func.id].bind(
            *[object() for _ in node.args], **{k.arg: object() for k in node.keywords}
        )
        checked += 1
    assert checked >= 100
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_execute")
    calls = [
        n
        for n in ast.walk(execute)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_evaluate"
    ]
    assert len(calls) == 2
    for call in calls:
        assert isinstance(call.args[3], ast.Name) and call.args[3].id == "mean"
        assert isinstance(call.args[4], ast.Name) and call.args[4].id == "helper"
    broken = list(calls[0].args)
    del broken[3]
    with pytest.raises(TypeError):
        signatures["_evaluate"].bind(
            *[object() for _ in broken], **{k.arg: object() for k in calls[0].keywords}
        )
