"""Fixed fresh symmetric-affine residual pilot; never a serving promotion.

Own scorer-aware lifecycle with authenticated historical pure utilities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_PLAN = "d102c6d00ab536b89c0a0021097976fb997831184a4e4c2db82d172a4c61a19a"
_REPAIR = "8ac4fbc66d40bf26e249d0ee9a0cd9117d06653e762552629c98e2b53a2c5a18"
_AMENDMENT = "bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f"
_INPUT_ADAPTER = "plm-authenticated-partition-membership-v1"
_PRIMARY_RECEIPT_PATHS = (
    "runs/learning/symmetric-affine-runner-tests-v2/receipt.json",
    "runs/learning/symmetric-affine-runner-tests-v2/test-receipt.json",
)
_AUDITOR_CONTRACT_PATH = "runs/learning/symmetric-affine-auditor-tests-v2/readiness.json"
_AUDITOR_PART_PATHS = {
    "auditor": "runs/learning/symmetric-affine8000-v2/independent-audit.py",
    "tests": "tests/unit/test_symmetric_affine_audit_v2.py",
    "test_receipt": "runs/learning/symmetric-affine-auditor-tests-v2/test-receipt.json",
}
_AUDITOR_EXTRA_DEPENDENCIES = (
    "runs/learning/projection-only-refit-v1/independent-audit.py",
    "runs/learning/bilinear-residual-refit-v1/independent-audit.py",
    "runs/learning/bilinear-budget2000-v1/independent-audit.py",
    "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md",
    "docs/experiments/2026-09-26-symmetric-affine-plan.md",
    "docs/experiments/2026-09-26-symmetric-affine-runtime-repair-v2.md",
)
_AFFINE = "8c665818bbfb4306f954ce8b0d9cf67e64e1de2b6f883ee087f3fbe8b3037fbe"
_SCORER = "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
_MEAN = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
_SUMMARY = "e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8"
_AUDIT = "bdf142f48229fbcdb197bce3a019e38c3a65570ae3f8551a9c9a1f14b2198872"
_DECISION = "2cadf9dce7ee36cc4095a10a746cc1608d5ce4437724872c1eec1c88e915c195"
_AUDITOR = "9ecc66cd6f63b43d6fd13f0a29ec29ce785e4d0ad45b3f861a2e92f406bc7676"
_CHILD = "6979013750eb8f2780917f714192850aae411a48490757236a891fea7e351f91"
_MEMBERSHIP = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"
_ARCHITECTURE = "plm-frozen-symmetric-affine-residual-v1"
_OBJECTIVE = "plm-affine-residual-balanced-bce-v1"
_INFERENCE = "plm-symmetric-affine-positive-set-v1"
_EVALUATOR = "plm-symmetric-affine8000-screen-v2"
_BUDGET = "f4376d4f2d838938bb9f50ae0cfa16ef319639fa0ce917c9fea3b842c0018930"
_BASE = "cd593c14c429247837c26b09754509d5cfd1dd36adb317f56522a80a78a25122"
_PARAMETERS = ("symmetric_bilinear_residual", "symmetric_affine_linear", "symmetric_affine_bias")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _load_bilinear(root):
    path = root / "scripts/refit_bilinear_residual.py"
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == _SCORER, "bilinear helper identity")
    spec = importlib.util.spec_from_file_location("budget_bilinear_helper", path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _load_budget(root):
    path = root / "scripts/refit_bilinear_budget.py"
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == _BUDGET, "budget helper identity")
    spec = importlib.util.spec_from_file_location("budget8000_frozen_budget_helper", path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _load_module(path, expected, name):
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "helper identity")
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recipe(value, inherited):
    expected = json.loads(json.dumps(inherited))
    _require(expected["experiment"] == "bilinear-budget8000-v1", "historical recipe")
    expected.pop("trainable_parameter")
    expected.update(
        experiment="symmetric-affine8000-v2",
        architecture=_ARCHITECTURE,
        objective=_OBJECTIVE,
        inference=_INFERENCE,
        evaluator=_EVALUATOR,
        trainable_parameters=list(_PARAMETERS),
        affine_linear_shape=[2, 256],
        affine_bias_shape=[2],
        affine_scale=16,
        standard_serving_supported=False,
        input_adapter=_INPUT_ADAPTER,
        input_amendment_sha256=_AMENDMENT,
        repair_declaration_sha256=_REPAIR,
        strong_comparator_exact=216,
        strong_comparator_f1=0.9998843626799785,
        strong_comparator_group_exact={"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 62},
    )
    _require(
        json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True),
        "fixed affine recipe differs",
    )
    return value


def _gate(child, invariants):
    metrics, groups = child["aggregate"], child["groups"]
    checks = {
        "execution_invariants": set(invariants)
        == {
            "parent_replay_exact",
            "zero_A_u_b_replay_exact",
            "initial_train_loss_exact",
            "completed_8000_updates",
            "original_93_tensors_unchanged",
            "residual_changed",
            "reload_exact",
            "optimizer_reload_exact",
            "train_only_supervision",
        }
        and all(v is True for v in invariants.values()),
        "serialization_compatible": metrics["serialization_compatible"] == 222,
        "exact_above_historical8000": metrics["exact_count"] > 216,
        "f1_at_least_historical8000": metrics["f1"] >= 0.9998843626799785,
        "group_exact_nonregression": all(
            groups[g]["exact_count"] >= n
            for g, n in {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 62}.items()
        ),
    }
    return {
        "accepted": False,
        "independent_audit_required": True,
        "primary_checks_passed": all(checks.values()),
        "checks": checks,
    }


def _historical_pair(current, historical, mean):
    _require(
        current["complete"] is True and historical["complete"] is True,
        "complete historical pairing",
    )
    _require(
        current["product_token_ids"] == historical["product_token_ids"]
        and current["query_count"]
        == historical["query_count"]
        == len(current["responses"])
        == len(historical["responses"]),
        "historical columns/coverage",
    )
    keys = ("index", "subject", "dimension", "prompt_ids", "group", "expected_set_ids")
    for new, old in zip(current["responses"], historical["responses"], strict=True):
        _require(
            {k: new[k] for k in keys} == {k: old[k] for k in keys}, "historical query/label order"
        )
    return mean._paired(
        current["responses"], [r["selected_set_ids"] for r in historical["responses"]]
    )


def _selected_partitions(
    membership, parent, queries, vocabulary, expected_train=1637, expected_validation=222
):
    """Re-encode selected authenticated membership only; never load a corpus or graph."""
    tokens, classes = vocabulary["tokens"], vocabulary["classes"]
    _require(
        len(tokens) == len(classes) == 2049 and len(set(tokens)) == 2049, "vocabulary inventory"
    )
    _require(
        {i: tokens[i] for i in (0, 1, 2, 5, 32, 33, 34)}
        == {0: "PAD", 1: "BOS", 2: "EOS", 5: "ANSWER", 32: "TYPE", 33: "COLOR", 34: "SAME"},
        "vocabulary protocol IDs",
    )
    _require(
        [i for i, c in enumerate(classes) if c == "entity"] == list(range(1024, 2049)),
        "product columns",
    )
    _require(
        membership.get("partition") == "train"
        and membership.get("query_count") == len(membership["queries"]) == expected_train,
        "train partition coverage",
    )
    _require(
        parent.get("complete") is True
        and parent.get("query_count") == len(parent["responses"]) == expected_validation
        and parent["product_token_ids"] == list(range(1024, 2049)),
        "validation partition coverage",
    )
    partitions = {}
    for name, rows, expected in (
        ("train", membership["queries"], expected_train),
        ("validation", parent["responses"], expected_validation),
    ):
        _require(len(queries[name]) == expected, "ordered query coverage")
        records = []
        for index, row in enumerate(rows):
            prompt, truth = row["prompt_ids"], row["expected_set_ids"]
            _require(type(row["index"]) is int and row["index"] == index, "ordered query index")
            _require(
                isinstance(prompt, list)
                and len(prompt) == 5
                and all(type(i) is int for i in prompt)
                and prompt[0] == 1
                and prompt[3:] == [34, 5]
                and prompt[2] in (32, 33)
                and 1024 <= prompt[1] < 2049,
                "five-token grammar",
            )
            _require(
                tokens[prompt[1]] == row["subject"] and tokens[prompt[2]] == row["dimension"],
                "query token identity",
            )
            _require(
                {k: row[k] for k in ("subject", "dimension", "prompt_ids")} == queries[name][index],
                "accepted partition order",
            )
            _require(
                isinstance(truth, list)
                and 1 <= len(truth) <= 506
                and all(type(i) is int and 1024 <= i < 2049 for i in truth)
                and len(set(truth)) == len(truth)
                and prompt[1] not in truth,
                "membership validity",
            )
            ids = (*prompt, *sorted(truth), 2)
            records.append(
                SimpleNamespace(
                    subject=row["subject"],
                    dimension=row["dimension"],
                    input_ids=ids,
                    labels=(*([-100] * 5), *ids[5:]),
                )
            )
        partitions[name] = tuple(records)
    train_keys = {(r.subject, r.dimension) for r in partitions["train"]}
    val_keys = {(r.subject, r.dimension) for r in partitions["validation"]}
    _require(
        len(train_keys) == expected_train and len(val_keys) == expected_validation,
        "duplicate partition query",
    )
    _require(not train_keys & val_keys, "training/validation query leakage")
    return partitions


def _preflight(
    root,
    plan,
    recipe_path,
    bilinear,
    bilinear_path,
    mean,
    mean_path,
    helper,
    helper_path,
    budget,
    budget_path,
):
    # Every read below is explicitly named. Historical input manifests are evidence,
    # never an instruction to open graph/records/test paths.
    del budget
    _require(
        plan.resolve() == (root / "docs/experiments/2026-09-26-symmetric-affine-plan.md").resolve()
        and recipe_path.resolve()
        == (root / "configs/experiments/symmetric_affine8000_v2.json").resolve(),
        "fixed input document paths",
    )
    inputs = {}
    allowed = {
        "summary.json": _SUMMARY,
        "independent-audit.json": _AUDIT,
        "decision.json": _DECISION,
        "independent-audit.py": _AUDITOR,
    }
    old = root / "runs/learning/bilinear-budget8000-v1"
    values = {
        n: helper._read(helper._bind(old / n, h, inputs))
        for n, h in allowed.items()
        if n.endswith(".json")
    }
    helper._bind(old / "independent-audit.py", _AUDITOR, inputs)
    summary, audit, decision = (
        values[n] for n in ("summary.json", "independent-audit.json", "decision.json")
    )
    _require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and decision["evidence_accepted"] is True
        and decision["fixed_quality_gate_passed"] is True
        and audit["summary_sha256"] == decision["summary_sha256"] == _SUMMARY
        and decision["audit_sha256"] == _AUDIT
        and audit["script_sha256"] == _AUDITOR,
        "accepted historical8000 evidence",
    )
    _require(
        summary["script_sha256"] == _BASE
        and summary["split_hash"] == mean._SPLIT
        and summary["architecture"] == bilinear._architecture()
        and summary["objective"] == "plm-bilinear-residual-balanced-bce-v1"
        and summary["evaluator"] == "plm-bilinear-budget8000-screen-v1",
        "historical identities",
    )
    _require(
        summary["artifact_sha256"]["checkpoint-final.pt"] == _CHILD,
        "historical checkpoint inherited identity",
    )
    selected = {
        n: helper._read(helper._bind(old / n, summary["artifact_sha256"][n], inputs))
        for n in ("train-membership.json", "parent.json", "child.json", "training.json")
    }
    membership, parent, historical, training = (
        selected[n] for n in ("train-membership.json", "parent.json", "child.json", "training.json")
    )
    _require(mean._digest(membership) == _MEMBERSHIP, "training membership identity")
    _historical_pair(parent, historical, mean)
    parent_dir = root / "runs/national_dex_continuation_control_s1729_v1"
    parent_path = helper._bind(parent_dir / "checkpoint-final.pt", mean._PARENT, inputs)

    # Resolve only two named metadata identities; never open the whole input inventory.
    def inherited_metadata(relative):
        path = root / relative
        candidates = [
            h
            for name, h in summary["input_sha256"].items()
            if Path(name).as_posix().replace("\\", "/").endswith("/" + relative)
        ]
        _require(len(candidates) == 1, "explicit metadata identity")
        return helper._read(helper._bind(path, candidates[0], inputs))

    parent_run = inherited_metadata("runs/national_dex_continuation_control_s1729_v1/run.json")
    parent_sidecar = inherited_metadata(
        "runs/national_dex_continuation_control_s1729_v1/checkpoint-final.pt.json"
    )
    _require(
        parent_run["identity"]
        == parent_sidecar["experiment_identity"]
        == summary["parent_training_identity"]
        and parent_sidecar["global_step"] == 2000
        and parent_sidecar["checkpoint_hash"] == mean._PARENT
        and parent_sidecar["split_hash"] == mean._SPLIT
        and parent_sidecar["training_metadata"]["objective"] == "causal-next-token-v1"
        and parent_sidecar["corpus_identity"] == summary["corpus_identity"],
        "original parent metadata",
    )
    vocab_path = root / "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json"
    vocabulary = helper._read(
        helper._bind(
            vocab_path, "1b0d8f1762fede01fb253c3df23d21ced254a10f6cea367709c479565a92fb34", inputs
        )
    )
    partitions = _selected_partitions(membership, parent, summary["queries"], vocabulary)
    source = helper._bind(
        root / "runs/learning/pair-composition-integration-v1/summary.source.zip",
        helper._SOURCE,
        inputs,
    )
    configs = helper._bind(
        root / "runs/learning/pair-composition-integration-v1/summary.configs.zip",
        helper._CONFIGS,
        inputs,
    )
    members = helper._archive_members((source, configs))
    for path, digest in (
        (plan, _PLAN),
        (root / "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md", _AMENDMENT),
        (root / "docs/experiments/2026-09-26-symmetric-affine-runtime-repair-v2.md", _REPAIR),
        (bilinear_path, _SCORER),
        (mean_path, _MEAN),
        (helper_path, mean._HELPER),
        (budget_path, _BUDGET),
        (root / "scripts/refit_bilinear_budget8000.py", _BASE),
    ):
        helper._bind(path, digest, inputs)
    inherited_recipe = helper._read(
        helper._bind(old / "summary.recipe.json", summary["recipe_sha256"], inputs)
    )
    recipe = _recipe(helper._read(recipe_path), inherited_recipe)
    helper._bind(recipe_path, helper._sha(recipe_path), inputs)
    evidence = {name: summary["artifact_sha256"][name] for name in selected}
    adapter = {
        "identity": _INPUT_ADAPTER,
        "amendment_sha256": _AMENDMENT,
        "encoding": "five prompt IDs; ascending unique targets; EOS; prompt labels -100",
        "inherited_corpus_identity": True,
        "whole_corpus_revalidated": False,
        "evidence_sha256": evidence,
        "partition_sha256": {
            n: mean._digest(
                [
                    {
                        "input_ids": list(r.input_ids),
                        "labels": list(r.labels),
                        "subject": r.subject,
                        "dimension": r.dimension,
                    }
                    for r in rows
                ]
            )
            for n, rows in partitions.items()
        },
    }
    return {
        "inputs": inputs,
        "recipe": recipe,
        "recipe_path": recipe_path,
        "membership": membership,
        "parent_state": training["parent_state_before"],
        "parent": parent_path,
        "parent_run": parent_run,
        "parent_sidecar": parent_sidecar,
        "corpus_identity": summary["corpus_identity"],
        "inherited_architecture": summary["inherited_architecture"],
        "vocabulary_path": vocab_path,
        "partitions": partitions,
        "input_adapter": adapter,
        "wide": parent,
        "historical8000": historical,
        "historical_environment": summary["environment"],
        "historical8000_child_sha256": summary["artifact_sha256"]["child.json"],
        "source": source,
        "configs": configs,
        "members": members,
    }


def _load_selected_runtime(context):
    import torch

    from plm.config import ModelConfig, RootConfig
    from plm.experimentation.identity import ExperimentIdentity
    from plm.model.architecture import build_model
    from plm.protocol import Vocabulary
    from plm.training.checkpoint import load_checkpoint, validate_checkpoint_identity

    config = RootConfig.model_validate(context["parent_run"]["config"])
    vocabulary = Vocabulary.load(context["vocabulary_path"])
    _require(
        vocabulary.content_hash() == context["corpus_identity"]["vocabulary_hash"],
        "vocabulary content identity",
    )
    model_config = config.model.model_copy(
        update={"vocab_size": len(vocabulary), "max_seq_len": config.train.seq_len}
    )
    _require(
        model_config.model_dump(mode="json") == context["inherited_architecture"],
        "inherited model configuration",
    )
    model = build_model(model_config)
    state = load_checkpoint(context["parent"], model, map_location="cpu", restore_rng=False)
    identity = context["parent_run"]["identity"]
    validate_checkpoint_identity(
        state, identity=ExperimentIdentity.from_dict(identity), split_hash=identity["split_hash"]
    )
    raw = state.metadata
    _require(state.global_step == 2000, "parent global step")
    _require(
        state.checkpoint_hash == "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1",
        "original parent checkpoint hash",
    )
    _require(
        raw["config"] == context["parent_run"]["config"]["train"],
        "parent checkpoint TrainConfig differs from original run train configuration",
    )
    _require(raw["corpus_identity"] == context["corpus_identity"], "parent corpus identity")
    _require(
        raw["training_metadata"] == context["parent_sidecar"]["training_metadata"],
        "parent payload and sidecar training metadata",
    )
    _require(raw["training_metadata"]["objective"] == "causal-next-token-v1", "parent objective")
    _require(
        ModelConfig.model_validate(raw["training_metadata"]["model_config"]) == model_config,
        "parent resolved model configuration",
    )
    _require(identity["evaluator_version"] == "plm-train-causal-v2", "parent evaluator identity")
    _require(
        all(identity[k] == v for k, v in context["corpus_identity"].items()),
        "parent experiment corpus identity",
    )
    device = config.eval.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _require(device == "cuda", "historical CUDA device")
    model.to(device).eval()
    return SimpleNamespace(
        model=model,
        vocabulary=vocabulary,
        checkpoint_hash=state.checkpoint_hash,
        training_identity=identity,
        corpus_identity=context["corpus_identity"],
        device=device,
        split=SimpleNamespace(**context["partitions"], split_hash=identity["split_hash"]),
    )


def _head(model, features, dimensions, mean, affine, bilinear):
    parent = mean._head(model.symmetric_relation_projection.weight, features)
    return affine.score(
        parent, model, features[0], features[2], dimensions, features[3], bilinear._residual
    )


def _exact_state(left, right):
    """Reload equality includes tensor bytes (unlike numeric equality of signed zero)."""
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and _tensor_bytes(left) == _tensor_bytes(right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_exact_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        return len(left) == len(right) and all(
            _exact_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _validate_optimizer(optimizer, model, updates):
    import torch

    parameters = [getattr(model, n) for n in _PARAMETERS]
    actual = [p for g in optimizer.param_groups for p in g["params"]]
    _require(
        len(actual) == 3 and {id(p) for p in actual} == {id(p) for p in parameters},
        "optimizer exact parameter membership",
    )
    _require(set(optimizer.state) == set(parameters), "optimizer exact state inventory")
    for p in parameters:
        state = optimizer.state[p]
        _require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "optimizer state fields")
        _require(float(state["step"]) == updates, "optimizer step")
        for name in ("exp_avg", "exp_avg_sq"):
            _require(
                state[name].shape == p.shape
                and state[name].dtype == torch.float32
                and bool(torch.isfinite(state[name]).all()),
                "optimizer moments",
            )
    state = optimizer.state_dict()
    names = {id(getattr(model, n)): n for n in _PARAMETERS}
    mapping = {}
    for group, saved in zip(optimizer.param_groups, state["param_groups"], strict=True):
        for parameter, key in zip(group["params"], saved["params"], strict=True):
            mapping[str(key)] = names[id(parameter)]
    return {
        "parameter_id_to_name": mapping,
        "steps": {mapping[str(k)]: float(v["step"]) for k, v in state["state"].items()},
    }


def _resolve(root, value):
    root = root.resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    _require(path.is_relative_to(root), "receipt path escapes repository")
    relative = path.relative_to(root)
    _require(
        not (relative.parts and relative.parts[0] == "data")
        and path.suffix.lower() not in {".jsonl", ".db", ".sqlite", ".sqlite3"},
        "forbidden corpus or graph input",
    )
    return path


def _receipt_checks(receipt):
    _require(
        receipt.get("passed") is True
        and receipt.get("gpu_used") is False
        and receipt.get("experiment_training_executed") is False
        and type(receipt.get("exit_code")) is int
        and receipt["exit_code"] == 0
        and receipt.get("terminal_completion_observed") is True
        and type(receipt.get("tests_passed")) is int
        and receipt["tests_passed"] > 0
        and type(receipt.get("tests_skipped")) is int
        and receipt["tests_skipped"] == 0,
        "CPU test receipt",
    )
    _require(
        receipt.get("input_adapter_identity") == _INPUT_ADAPTER
        and receipt.get("input_amendment_sha256") == _AMENDMENT
        and receipt.get("repair_declaration_sha256") == _REPAIR,
        "test input contract",
    )


def _receipt_dependencies(root, receipt, helper, inputs, required, stdout_path):
    # Resolve and compare the complete inventory before opening a single dependency.
    resolved = {}
    for name, expected in receipt["test_dependencies"].items():
        path = _resolve(root, name)
        _require(path not in resolved, "duplicate dependency path")
        resolved[path] = expected
    _require(set(resolved) == {p.resolve() for p in required}, "test dependency inventory")
    stdout = _resolve(root, receipt["stdout"]["path"])
    _require(stdout == stdout_path.resolve(), "test stdout path")
    for path, expected in resolved.items():
        helper._bind(path, expected, inputs)
    helper._bind(stdout, receipt["stdout"]["sha256"], inputs)
    return resolved


def _test_receipt(path, pinned, digest, recipe_digest, required, helper, inputs, root):
    path = _resolve(root, path)
    _require(
        path in {(root / p).resolve() for p in _PRIMARY_RECEIPT_PATHS},
        "primary receipt path admission",
    )
    path = helper._bind(path, pinned, inputs)
    receipt = helper._read(path)
    _receipt_checks(receipt)
    _require(
        receipt.get("tested_script_sha256") == digest
        and receipt.get("tested_recipe_sha256") == recipe_digest,
        "tested source identity",
    )
    dependencies = _receipt_dependencies(
        root, receipt, helper, inputs, required, path.parent / "stdout.txt"
    )
    _require(set(dependencies) == {p.resolve() for p in required}, "test dependency inventory")
    return path, _resolve(root, receipt["stdout"]["path"])


def _auditor_contract(root, path, pinned, helper, inputs):
    _require(path is not None and pinned, "auditor readiness contract required")
    path = _resolve(root, path)
    _require(path == (root / _AUDITOR_CONTRACT_PATH).resolve(), "auditor contract path admission")
    path = helper._bind(path, pinned, inputs)
    contract = helper._read(path)
    _require(
        type(contract.get("schema_version")) is int
        and contract["schema_version"] == 1
        and contract.get("experiment") == "symmetric-affine8000-v2"
        and contract.get("plan_sha256") == _PLAN
        and contract.get("ready") is True
        and contract.get("input_adapter_identity") == _INPUT_ADAPTER
        and contract.get("input_amendment_sha256") == _AMENDMENT
        and contract.get("repair_declaration_sha256") == _REPAIR,
        "auditor readiness contract",
    )
    # Admit the entire reference trio before opening any member.
    parts = {n: _resolve(root, contract[n]["path"]) for n in _AUDITOR_PART_PATHS}
    _require(
        all(parts[n] == (root / p).resolve() for n, p in _AUDITOR_PART_PATHS.items()),
        "auditor reference path admission",
    )
    for name, path_ref in parts.items():
        helper._bind(path_ref, contract[name]["sha256"], inputs)
    receipt = helper._read(parts["test_receipt"])
    _receipt_checks(receipt)
    _require(
        receipt.get("tested_script_sha256") == contract["auditor"]["sha256"],
        "auditor tested source identity",
    )
    dependencies = _receipt_dependencies(
        root,
        receipt,
        helper,
        inputs,
        [parts["auditor"], parts["tests"], *(root / p for p in _AUDITOR_EXTRA_DEPENDENCIES)],
        parts["test_receipt"].parent / "stdout.txt",
    )
    _require(
        all(dependencies.get(parts[n]) == contract[n]["sha256"] for n in ("auditor", "tests")),
        "auditor test dependency identity",
    )
    return {
        "path": str(path.relative_to(root.resolve())),
        "sha256": pinned,
        "readiness_only": True,
        "candidate_audited": False,
    }


def _tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _zero_replay(model, records, parent, mean, helper, path, affine, bilinear, batch_size=8):
    import torch

    report = {"complete": False, "batches": [], "query_count": 0, "dtype": "float32"}
    actual_hash, parent_hash, residual_hash = (hashlib.sha256() for _ in range(3))
    try:
        _require(len(records) == len(parent["responses"]), "zero replay coverage")
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            prompts = torch.tensor(
                [r.input_ids[:5] for r in batch], device=next(model.parameters()).device
            )
            features = mean._features(model, prompts)
            with torch.no_grad():
                actual = _head(model, features, prompts[:, 2], mean, affine, bilinear)
                residual = actual - mean._head(model.symmetric_relation_projection.weight, features)
            expected = torch.tensor(
                [
                    row["symmetric_relation_logits"]
                    for row in parent["responses"][offset : offset + len(batch)]
                ],
                dtype=torch.float32,
            )
            report["failed_observation"] = {
                "offset": offset,
                "logits": helper._failure_safe(actual.cpu().tolist()),
                "residual": helper._failure_safe(residual.cpu().tolist()),
            }
            parts = [_tensor_bytes(t) for t in (actual, expected, residual)]
            _require(parts[0] == parts[1], "zero-A/u/b actual head parent replay")
            _require(int(torch.count_nonzero(residual)) == 0, "nonzero initial residual")
            for h, data in zip((actual_hash, parent_hash, residual_hash), parts, strict=True):
                h.update(data)
            report["batches"].append(
                {
                    "indices": list(range(offset, offset + len(batch))),
                    "shape": list(actual.shape),
                    "logits_sha256": hashlib.sha256(parts[0]).hexdigest(),
                    "parent_logits_sha256": hashlib.sha256(parts[1]).hexdigest(),
                    "residual_sha256": hashlib.sha256(parts[2]).hexdigest(),
                    "residual_nonzero_count": 0,
                }
            )
            report["query_count"] += len(batch)
            report.pop("failed_observation", None)
        report.update(
            complete=True,
            logits_shape=[len(records), len(parent["product_token_ids"])],
            logits_sha256=actual_hash.hexdigest(),
            parent_logits_sha256=parent_hash.hexdigest(),
            residual_sha256=residual_hash.hexdigest(),
            residual_nonzero_count=0,
            batch_sizes=[b["shape"][0] for b in report["batches"]],
        )
        return report
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        mean._write(path, helper._failure_safe(report))


def _evaluate(
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
    import torch

    report = {
        "phase": phase,
        "complete": False,
        "query_count": 0,
        "product_token_ids": list(range(1024, 2049)),
        "responses": [],
        "head_seconds": 0.0,
    }
    try:
        _require(len(records) == len(wide_rows), "evaluation coverage")
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            prompts = torch.tensor(
                [r.input_ids[:5] for r in batch], device=next(model.parameters()).device
            )
            if prompts.is_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            features = mean._features(model, prompts)
            with torch.no_grad():
                logits = (
                    mean._head(model.symmetric_relation_projection.weight, features)
                    if phase == "parent"
                    else _head(model, features, prompts[:, 2], mean, affine, bilinear)
                )
                if phase == "parent":
                    _require(
                        _tensor_bytes(logits)
                        == _tensor_bytes(model(prompts).symmetric_relation_logits),
                        "standalone archived parent mismatch",
                    )
            if prompts.is_cuda:
                torch.cuda.synchronize()
            report["head_seconds"] += time.perf_counter() - started
            values = logits.cpu().tolist()
            report["failed_observation"] = {
                "offset": offset,
                "prompt_ids": prompts.cpu().tolist(),
                "standalone_logits": helper._failure_safe(values),
            }
            for j, (record, vector) in enumerate(zip(batch, values, strict=True)):
                old = wide_rows[offset + j]
                _require(
                    mean._query(record)
                    == {k: old[k] for k in ("subject", "dimension", "prompt_ids")},
                    "query identity",
                )
                if phase == "parent":
                    expected = torch.tensor(old["symmetric_relation_logits"], dtype=torch.float32)
                    _require(
                        _tensor_bytes(logits[j]) == _tensor_bytes(expected),
                        "historical parent replay",
                    )
                truth = list(old["expected_set_ids"])
                _require(set(record.input_ids[5:-1]) == set(truth), "validation truth")
                helper._fp32(vector)
                selected = mean._prediction(vector, old["prompt_ids"][1])
                excluded = set(truth) | {old["prompt_ids"][1]}
                negatives = [value for c, value in enumerate(vector) if c + 1024 not in excluded]
                report["responses"].append(
                    {
                        "index": offset + j,
                        **mean._query(record),
                        "group": old["group"],
                        "expected_set_ids": truth,
                        "symmetric_relation_logits": vector,
                        "selected_set_ids": selected,
                        "metrics": mean._metrics(selected, truth),
                        "strict_separation": min(vector[i - 1024] for i in truth) > max(negatives),
                    }
                )
                report["query_count"] += 1
            report.pop("failed_observation", None)
        rows = report["responses"]
        report["aggregate"] = mean._totals(rows)
        report["groups"] = {
            g: mean._totals([r for r in rows if r["group"] == g]) for g in mean._GROUPS
        }
        if parent is not None:
            report["paired_vs_parent_dense"] = mean._paired(
                rows, [r["selected_set_ids"] for r in parent["responses"]]
            )
        report.update(complete=True, exact_parent_replay=phase == "parent")
        return report
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        mean._write(path, helper._failure_safe(report))


def _validate_child(state, identity, config, metadata, corpus):
    raw = state.metadata
    _require(state.global_step == raw["global_step"] == 8000, "child update count")
    _require(
        raw["experiment_identity"] == identity and identity["evaluator_version"] == _EVALUATOR,
        "child identity",
    )
    _require(raw["config"] == config and raw["training_metadata"] == metadata, "child metadata")
    _require(
        metadata["implementation_identity"] == config["implementation_identity"]
        and metadata["implementation_identity"]["runner_sha256"] == metadata["runner_sha256"]
        and metadata["implementation_identity"]["scorer_sha256"]
        == metadata["scorer_sha256"]
        == _AFFINE
        and metadata["implementation_identity"]["mean_helper_sha256"] == _MEAN
        and metadata["implementation_identity"]["budget_helper_sha256"] == _BUDGET
        and metadata["implementation_identity"]["bilinear_helper_sha256"] == _SCORER,
        "child implementation identity",
    )
    _require(
        metadata["architecture"] == config["affine_residual_architecture"] == _ARCHITECTURE,
        "child architecture",
    )
    _require(
        metadata["objective"] == _OBJECTIVE
        and metadata["evaluator"] == _EVALUATOR
        and metadata["trainable_parameters"] == list(_PARAMETERS)
        and metadata["inference"] == _INFERENCE
        and metadata["standard_serving_supported"] is False
        and metadata["input_adapter_identity"] == _INPUT_ADAPTER
        and metadata["input_amendment_sha256"] == _AMENDMENT
        and metadata["repair_declaration_sha256"] == _REPAIR
        and metadata["residual_updates"] == 8000
        and metadata["parent_training_steps"] == 2000,
        "child objective/lineage",
    )
    _require(
        metadata["parent_checkpoint_sha256"]
        == "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1",
        "original parent",
    )
    _require(
        raw["corpus_identity"] == corpus
        and raw["split_hash"] == "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d",
        "child corpus/split",
    )


def _preserve_failure(directory, model, optimizer, training, mean, helper):
    import torch

    path = directory / "failure-state.pt"
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "completed_updates": training.get("completed_updates", 0),
        "optimizer_steps_attempted": training.get("optimizer_steps_attempted", 0),
        "diagnostic_only": True,
        "resume_allowed": False,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
    mean._write(
        directory / "failure-state.json",
        {
            "path": path.name,
            "sha256": helper._sha(path),
            "diagnostic_only": True,
            "resume_allowed": False,
            "completed_updates": payload["completed_updates"],
            "optimizer_steps_attempted": payload["optimizer_steps_attempted"],
            "error": training.get("error"),
        },
    )


def _execute(root, out, context, bilinear, mean, helper, summary, affine):
    started = time.perf_counter()
    runtime_root = out.parent / "runtime"
    training = {
        "complete": False,
        "objective": _OBJECTIVE,
        "architecture": _ARCHITECTURE,
        "completed_updates": 0,
        "history": [],
    }
    try:
        helper._modules()
        summary["runtime_files_sha256"] = helper._extract(context["members"], runtime_root)
        sys.path.insert(0, str(runtime_root / "src"))
        import torch

        from plm.config import OptimizerConfig
        from plm.experimentation.identity import ExperimentIdentity
        from plm.model.architecture import build_model
        from plm.model.layers import _prompt_set_loss
        from plm.reproducibility import seed_everything
        from plm.training.checkpoint import load_checkpoint, save_checkpoint
        from plm.training.data import collate_records
        from plm.training.optim import create_optimizer

        recipe = context["recipe"]
        seed_everything(1729, deterministic=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        summary["environment"] = helper._environment(torch)
        _require(
            summary["environment"] == context["historical_environment"],
            "historical environment drift",
        )
        summary["numerical_settings"] = helper._settings(torch) | {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        summary["workspace_provenance"] = mean._git(root)
        runtime = _load_selected_runtime(context)
        _require(
            runtime.checkpoint_hash == mean._PARENT
            and runtime.split.split_hash == mean._SPLIT
            and runtime.training_identity == context["parent_run"]["identity"],
            "loaded parent identity",
        )
        helper._modules(runtime_root)
        model = runtime.model.float().eval()
        _require(
            all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()), "finite parent"
        )
        parent_state = mean._state_hashes(model)
        _require(
            parent_state == context["parent_state"] and len(parent_state) == 93,
            "original 93 tensors",
        )
        training["parent_state_before"] = parent_state
        queries = mean._split_contract(runtime.split.train, runtime.split.validation)
        membership = mean._training_membership(runtime.split.train)
        _require(
            membership == context["membership"] and mean._digest(membership) == _MEMBERSHIP,
            "ordered train manifest",
        )
        mean._write(out.parent / "train-membership.json", membership)
        summary.update(
            queries=queries,
            input_adapter=context["input_adapter"],
            validation_update_points=[0, 8000],
            train_query_order_sha256=mean._digest(queries["train"]),
            parent_training_identity=runtime.training_identity,
            inherited_architecture=model.config.model_dump(mode="json"),
            corpus_identity=runtime.corpus_identity,
            split_hash=mean._SPLIT,
        )
        parent = _evaluate(
            model,
            runtime.split.validation,
            context["wide"]["responses"],
            helper,
            out.parent / "parent.json",
            affine,
            bilinear,
            phase="parent",
        )
        affine.attach(model)
        _require(
            [n for n, p in model.named_parameters() if p.requires_grad] == list(_PARAMETERS),
            "three-parameter trainable mask",
        )
        _require(
            all(int(torch.count_nonzero(getattr(model, n))) == 0 for n in _PARAMETERS),
            "zero residual initialization",
        )
        initial = mean._state_hashes(model)
        _require(
            len(initial) == 96
            and {k: v for k, v in initial.items() if k not in _PARAMETERS} == parent_state,
            "96-entry initial state",
        )
        training["state_before"] = initial
        zero = _zero_replay(
            model,
            runtime.split.validation,
            parent,
            mean,
            helper,
            out.parent / "zero-replay.json",
            affine,
            bilinear,
        )
        _require(mean._state_hashes(model) == initial, "zero replay changed state")
        training["zero_replay_sha256"] = helper._sha(out.parent / "zero-replay.json")
        training["zero_replay"] = zero
        helper._unchanged(context["inputs"])
        batch = collate_records(runtime.split.train, pad_id=runtime.vocabulary.pad_id)
        prompts = batch.input_ids[:, :5].to(runtime.device)
        labels = batch.labels.masked_fill(~batch.attention_mask, -100)[:, 1:].to(runtime.device)
        features = mean._features(model, prompts)
        with torch.no_grad():
            scores = _head(model, features, prompts[:, 2], mean, affine, bilinear)
            initial_loss = float(_prompt_set_loss(scores, labels, excluded_ids=features[2]).cpu())
        _require(
            list(scores.shape) == [1637, 1025] and initial_loss == recipe["initial_training_loss"],
            "zero-A full train loss replay",
        )
        optimizer = create_optimizer(model, OptimizerConfig.model_validate(recipe["optimizer"]))
        _require(
            not optimizer.state and sum(len(g["params"]) for g in optimizer.param_groups) == 3,
            "fresh three-parameter optimizer",
        )
        training.update(
            train_query_count=1637,
            train_query_order_sha256=summary["train_query_order_sha256"],
            trainable_parameters=list(_PARAMETERS),
            initial_loss_replay=initial_loss,
            optimizer={
                "name": "adamw",
                "config": recipe["optimizer"],
                "groups": [
                    {k: v for k, v in group.items() if k != "params"}
                    | {
                        "parameter_names": [
                            n
                            for n, p in model.named_parameters()
                            if any(p is q for q in group["params"])
                        ]
                    }
                    for group in optimizer.param_groups
                ],
            },
            optimizer_implementation={
                "class": type(optimizer).__name__,
                "module": type(optimizer).__module__,
                "defaults": dict(optimizer.defaults),
            },
            transformer_forwards_during_fit=0,
            validation_labels_used=False,
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        fit_started = time.perf_counter()
        with torch.no_grad():
            parent_scores = mean._head(
                model.symmetric_relation_projection.weight, features
            ).detach()
        _require(mean._state_hashes(model) == initial, "state changed before fit")
        _require(
            all(int(torch.count_nonzero(getattr(model, n))) == 0 for n in _PARAMETERS),
            "nonzero residual immediately before fit",
        )
        fit_receipt = {}
        try:
            affine.fit(
                model,
                parent_scores,
                features[0],
                features[2],
                prompts[:, 2],
                labels,
                _prompt_set_loss,
                optimizer,
                8000,
                fit_receipt,
                bilinear._residual,
            )
        finally:
            training.update({k: v for k, v in fit_receipt.items() if k != "complete"})
            training["fit_complete"] = fit_receipt.get("complete", False)
        training["optimizer_final"] = _validate_optimizer(optimizer, model, 8000)
        torch.cuda.synchronize()
        training["refit_seconds"] = time.perf_counter() - fit_started
        _require(training["initial_pre_update_loss"] == initial_loss, "trace initial loss")
        after = mean._state_hashes(model)
        _require(set(after) == set(initial), "state inventory changed")
        _require(any(initial[n] != after[n] for n in _PARAMETERS), "residual did not change")
        _require(
            {k: v for k, v in after.items() if k not in _PARAMETERS} == parent_state,
            "frozen original state",
        )
        training.update(state_after=after, residual_changed=True, frozen_tensors_unchanged=True)
        implementation = {
            "runner_sha256": summary["script_sha256"],
            "scorer_sha256": _AFFINE,
            "input_adapter_identity": _INPUT_ADAPTER,
            "input_amendment_sha256": _AMENDMENT,
            "repair_declaration_sha256": _REPAIR,
            "bilinear_helper_sha256": _SCORER,
            "mean_helper_sha256": _MEAN,
            "budget_helper_sha256": _BUDGET,
            "source_archive_sha256": helper._SOURCE,
            "configs_archive_sha256": helper._CONFIGS,
        }
        child_config = {
            "implementation_identity": implementation,
            "inherited_parent_config": context["parent_run"]["config"],
            "affine_residual_architecture": _ARCHITECTURE,
            "affine_residual_recipe": recipe,
        }
        identity = ExperimentIdentity(
            source_commit=summary["workspace_provenance"]["commit"],
            resolved_config_hash=mean._digest(child_config),
            protocol_version=runtime.training_identity["protocol_version"],
            **runtime.corpus_identity,
            split_hash=mean._SPLIT,
            evaluator_version=_EVALUATOR,
            environment={
                **summary["environment"],
                "source_archive_sha256": helper._SOURCE,
                "configs_archive_sha256": helper._CONFIGS,
                "runner_sha256": summary["script_sha256"],
                "scorer_sha256": _AFFINE,
                "input_adapter_identity": _INPUT_ADAPTER,
                "input_amendment_sha256": _AMENDMENT,
                "repair_declaration_sha256": _REPAIR,
                "bilinear_helper_sha256": _SCORER,
                "recipe_sha256": summary["recipe_sha256"],
                "mean_helper_sha256": _MEAN,
                "budget_helper_sha256": _BUDGET,
                "architecture": _ARCHITECTURE,
            },
            seeds=(1729,),
        )
        metadata = {
            "implementation_identity": implementation,
            "objective": _OBJECTIVE,
            "evaluator": _EVALUATOR,
            "architecture": _ARCHITECTURE,
            "model_config": model.config.model_dump(mode="json"),
            "model_config_scope": "inherited archived base only; residual architecture is separate",
            "parent_checkpoint_sha256": mean._PARENT,
            "parent_training_steps": 2000,
            "residual_updates": 8000,
            "trainable_parameters": list(_PARAMETERS),
            "inference": _INFERENCE,
            "train_query_order_sha256": summary["train_query_order_sha256"],
            "record_count": 1637,
            "recipe_sha256": summary["recipe_sha256"],
            "runner_sha256": summary["script_sha256"],
            "scorer_sha256": _AFFINE,
            "input_adapter_identity": _INPUT_ADAPTER,
            "input_amendment_sha256": _AMENDMENT,
            "repair_declaration_sha256": _REPAIR,
            "bilinear_helper_sha256": _SCORER,
            "parent_state_sha256": mean._digest(parent_state),
            "initial_state_sha256": mean._digest(initial),
            "initial_pre_update_loss": initial_loss,
            "final_post_update_loss": training["final_post_update_loss"],
            "standard_serving_supported": False,
        }
        mean._write(
            out.parent / "run.json",
            {"config": child_config, "identity": identity.to_dict(), "training_metadata": metadata},
        )
        checkpoint = out.parent / "checkpoint-final.pt"
        _require(
            not checkpoint.exists() and not checkpoint.with_suffix(".pt.json").exists(),
            "checkpoint collision",
        )
        saved = save_checkpoint(
            checkpoint,
            model,
            optimizer=optimizer,
            global_step=8000,
            config=child_config,
            identity=identity,
            corpus_identity=runtime.corpus_identity,
            split_hash=mean._SPLIT,
            training_metadata=metadata,
        )
        reloaded = affine.attach(build_model(model.config).float().to(runtime.device).eval())
        reload_optimizer = create_optimizer(
            reloaded, OptimizerConfig.model_validate(recipe["optimizer"])
        )
        loaded = load_checkpoint(
            checkpoint, reloaded, optimizer=reload_optimizer, map_location="cpu", restore_rng=False
        )
        _validate_child(loaded, identity.to_dict(), child_config, metadata, runtime.corpus_identity)
        training["state_reloaded"] = mean._state_hashes(reloaded)
        _require(
            training["state_reloaded"] == after and loaded.checkpoint_hash == saved.checkpoint_hash,
            "child tensor reload",
        )
        _require(
            _exact_state(optimizer.state_dict(), reload_optimizer.state_dict()),
            "optimizer reload",
        )
        _validate_optimizer(reload_optimizer, reloaded, 8000)
        training.update(
            reload_exact=True,
            optimizer_reload_exact=True,
            checkpoint=str(checkpoint),
            checkpoint_sha256=saved.checkpoint_hash,
            checkpoint_sidecar_sha256=helper._sha(checkpoint.with_suffix(".pt.json")),
            training_identity=identity.to_dict(),
        )
        summary["child_training_identity"] = identity.to_dict()
        child = _evaluate(
            reloaded,
            runtime.split.validation,
            context["wide"]["responses"],
            mean,
            helper,
            out.parent / "child.json",
            affine=affine,
            bilinear=bilinear,
            parent=parent,
        )
        comparisons = {
            "historical8000_summary_sha256": _SUMMARY,
            "historical8000_child_sha256": context["historical8000_child_sha256"],
            "comparison_scope": "saved historical output; no rerun",
            "paired_vs_historical8000": _historical_pair(child, context["historical8000"], mean),
        }
        mean._write(out.parent / "comparisons.json", comparisons)
        invariants = {
            "parent_replay_exact": parent["exact_parent_replay"],
            "zero_A_u_b_replay_exact": zero["complete"]
            and zero["logits_sha256"] == zero["parent_logits_sha256"],
            "initial_train_loss_exact": initial_loss == recipe["initial_training_loss"],
            "completed_8000_updates": training["completed_updates"] == 8000,
            "original_93_tensors_unchanged": training["frozen_tensors_unchanged"],
            "residual_changed": training["residual_changed"],
            "reload_exact": training["reload_exact"],
            "optimizer_reload_exact": training["optimizer_reload_exact"],
            "train_only_supervision": training["validation_labels_used"] is False,
        }
        summary.update(
            invariants=invariants,
            gate=_gate(child, invariants),
            parent=parent["aggregate"],
            child={
                "aggregate": child["aggregate"],
                "groups": child["groups"],
                "paired_vs_parent_dense": child["paired_vs_parent_dense"],
                "paired_vs_historical8000": comparisons["paired_vs_historical8000"],
            },
            peak_memory={
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            module_origins=helper._modules(runtime_root),
        )
        helper._runtime_unchanged(runtime_root, summary["runtime_files_sha256"])
        helper._unchanged(context["inputs"])
        helper._unchanged(summary["snapshot_sha256"])
        _require(helper._environment(torch) == summary["environment"], "environment changed")
        training["complete"] = True
        summary.update(final_identity_check=True, complete=True)
    except BaseException as exc:
        summary["error"] = repr(exc)
        training["error"] = repr(exc)
        summary["failure_state_availability"] = {
            "model_available": "model" in locals(),
            "optimizer_available": "optimizer" in locals(),
        }
        if "model" in locals():
            try:
                _preserve_failure(
                    out.parent, model, locals().get("optimizer"), training, mean, helper
                )
            except BaseException as failure_exc:
                summary["failure_state_persistence_error"] = repr(failure_exc)
        raise
    finally:
        mean._write(out.parent / "training.json", helper._failure_safe(training))
        summary["artifact_sha256"] = {
            p.name: helper._sha(p)
            for p in out.parent.iterdir()
            if p.name
            in {
                "parent.json",
                "child.json",
                "training.json",
                "zero-replay.json",
                "comparisons.json",
                "train-membership.json",
                "run.json",
                "checkpoint-final.pt",
                "checkpoint-final.pt.json",
                "failure-state.pt",
                "failure-state.json",
            }
        }
        summary["wall_seconds"] = time.perf_counter() - started
        mean._write(out, helper._failure_safe(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("docs/experiments/2026-09-26-symmetric-affine-plan.md")
    )
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/experiments/symmetric_affine8000_v2.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/symmetric-affine8000-v2/summary.json")
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    parser.add_argument("--auditor-contract", type=Path)
    parser.add_argument("--auditor-contract-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    content = script.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    _require(
        "torch" not in sys.modules and Path.cwd().resolve() == root, "Torch-free root preflight"
    )
    bilinear, bilinear_path = _load_bilinear(root)
    budget, budget_path = _load_budget(root)
    mean, mean_path = bilinear._load_mean(root)
    helper, helper_path = mean._load_helper(root)
    affine_path = root / "scripts/symmetric_affine_residual.py"
    affine = _load_module(affine_path, _AFFINE, "affine_scorer")
    _require(affine.PARAMETER_NAMES == _PARAMETERS, "scorer parameter names")
    helper._modules()
    out = args.out.resolve()
    mean._refuse(out)
    _require(not (out.parent / "zero-replay.json").exists(), "immutable zero replay")
    _require(not (out.parent / "comparisons.json").exists(), "immutable comparisons")
    _require(
        not any((out.parent / n).exists() for n in ("failure-state.pt", "failure-state.json")),
        "immutable failure state",
    )
    context = _preflight(
        root,
        args.plan.resolve(),
        args.recipe.resolve(),
        bilinear,
        bilinear_path,
        mean,
        mean_path,
        helper,
        helper_path,
        budget,
        budget_path,
    )
    helper._bind(script, digest, context["inputs"])
    helper._bind(affine_path, _AFFINE, context["inputs"])
    helper._unchanged(context["inputs"])
    if args.preflight_only:
        _require("torch" not in sys.modules, "preflight imported Torch")
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "torch_imported": False,
                    "input_count": len(context["inputs"]),
                    "script_sha256": digest,
                    "scorer_sha256": _AFFINE,
                    "input_adapter_identity": _INPUT_ADAPTER,
                    "input_amendment_sha256": _AMENDMENT,
                    "repair_declaration_sha256": _REPAIR,
                    "bilinear_helper_sha256": _SCORER,
                    "historical8000_summary_sha256": _SUMMARY,
                    "historical8000_audit_sha256": _AUDIT,
                    "historical8000_decision_sha256": _DECISION,
                    "recipe_sha256": helper._sha(args.recipe),
                }
            )
        )
        return
    _require(
        args.test_receipt is not None and args.test_receipt_sha256,
        "pinned CPU test receipt required",
    )
    required = [
        script,
        affine_path,
        bilinear_path,
        mean_path,
        helper_path,
        budget_path,
        root / "scripts/refit_bilinear_budget8000.py",
        root / "tests/unit/test_symmetric_affine_runner_v2.py",
        root / "tests/unit/test_symmetric_affine_residual.py",
        root / "tests/fixtures/projection_refit_runtime_source.zip",
    ]
    receipt_path, stdout = _test_receipt(
        args.test_receipt,
        args.test_receipt_sha256,
        digest,
        helper._sha(args.recipe),
        required,
        helper,
        context["inputs"],
        root,
    )
    readiness = _auditor_contract(
        root, args.auditor_contract, args.auditor_contract_sha256, helper, context["inputs"]
    )
    helper._unchanged(context["inputs"])
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": content,
        "affine-helper.py": affine_path.read_bytes(),
        "bilinear-helper.py": bilinear_path.read_bytes(),
        "budget-helper.py": budget_path.read_bytes(),
        "mean-helper.py": mean_path.read_bytes(),
        "helper.py": helper_path.read_bytes(),
        "plan.md": args.plan.read_bytes(),
        "runtime-repair.md": (
            root / "docs/experiments/2026-09-26-symmetric-affine-runtime-repair-v2.md"
        ).read_bytes(),
        "input-amendment.md": (
            root / "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md"
        ).read_bytes(),
        "recipe.json": args.recipe.read_bytes(),
        "source.zip": context["source"].read_bytes(),
        "configs.zip": context["configs"].read_bytes(),
        "test-receipt.json": receipt_path.read_bytes(),
        "test-stdout.txt": stdout.read_bytes(),
        "inputs.json": mean._bytes(context["inputs"]),
    }
    hashes = {}
    for suffix, data in snapshots.items():
        path = out.with_suffix("." + suffix)
        with path.open("xb") as stream:
            stream.write(data)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
    summary = {
        "campaign_version": "symmetric-affine8000-v2",
        "auditor_readiness": readiness,
        "complete": False,
        "acceptance": False,
        "architecture": _ARCHITECTURE,
        "objective": _OBJECTIVE,
        "evaluator": _EVALUATOR,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "scorer_sha256": _AFFINE,
        "input_adapter_identity": _INPUT_ADAPTER,
        "input_amendment_sha256": _AMENDMENT,
        "repair_declaration_sha256": _REPAIR,
        "bilinear_helper_sha256": _SCORER,
        "historical8000_summary_sha256": _SUMMARY,
        "historical8000_audit_sha256": _AUDIT,
        "historical8000_decision_sha256": _DECISION,
        "mean_helper_sha256": _MEAN,
        "budget_helper_sha256": _BUDGET,
        "helper_sha256": mean._HELPER,
        "recipe_sha256": helper._sha(args.recipe),
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "protected_test_used": False,
        "standard_serving_supported": False,
        "inference": _INFERENCE,
        "limitations": [
            "Adaptive single-seed screen; no protected evaluation or default promotion.",
            "Historical 8000 scores reused; no matched timing comparison.",
            "Standalone affine scorer is not executed by ordinary decoder serving.",
            "No oracle parity, energy advantage or durable remote weight archive claim.",
        ],
    }
    _execute(root, out, context, bilinear, mean, helper, summary, affine)


if __name__ == "__main__":
    main()
