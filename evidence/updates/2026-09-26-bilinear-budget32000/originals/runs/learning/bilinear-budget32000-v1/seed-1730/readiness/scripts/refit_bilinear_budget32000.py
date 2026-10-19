"""Fixed three-parent 32000-update bilinear campaign; no serving promotion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

_PLAN = "5f1e0243b373ad1287c31dcef8a3c1cfa310cac9701795469a68677506e6f444"
_AMENDMENT = "bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f"
_ADAPTER_SHA = "02b5364bc7c440269c3a7bf123c37fe188b74353bcd1549a08a18ff95f774a6e"
_INPUT_ADAPTER = "plm-authenticated-partition-membership-v1"
_SCORER = "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
_MEAN = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
_MEMBERSHIP = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"
_SPLIT = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
_ARCHITECTURE = "plm-frozen-symmetric-bilinear-residual-v1"
_OBJECTIVE = "plm-bilinear-residual-balanced-bce-v1"
_INFERENCE = "dense-positive-bilinear-residual-subject-excluded-ascending-fp32-b8-last6-v1"
_EVALUATOR = "plm-bilinear-budget32000-screen-v1"
_CAMPAIGN = "bilinear-budget32000-v1"
_PARAMETERS = ("symmetric_bilinear_residual",)
_UPDATES = 32000
_SEEDS = (1729, 1730, 1731)
_PARENTS = {
    1729: "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1",
    1730: "5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2",
    1731: "ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d",
}
_FLOORS = {
    1729: {
        "exact_count": 216,
        "f1": 0.9998843626799785,
        "groups": {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 62},
    },
    1730: {
        "exact_count": 220,
        "f1": 0.9999717006519728,
        "groups": {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 66},
    },
    1731: {
        "exact_count": 214,
        "f1": 0.9998584743646364,
        "groups": {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 60},
    },
}
_INITIAL_LOSSES = {
    1729: 0.003822767175734043,
    1730: 0.003967047668993473,
    1731: 0.004869009833782911,
}
_CONTROL1729 = {
    "summary.json": "e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8",
    "independent-audit.json": "bdf142f48229fbcdb197bce3a019e38c3a65570ae3f8551a9c9a1f14b2198872",
    "decision.json": "2cadf9dce7ee36cc4095a10a746cc1608d5ce4437724872c1eec1c88e915c195",
}
_CONTROL_REPLICATION = {
    "aggregate-v2.json": "77cc2ee2ebf4dc1f8671e71833712302160190c85477eb6e0dd0b5494bb3cde6",
    "independent-audit-v2.json": "49fc5d7edc7763870ac0e9310295385d8e6a3af6e67d3cc6b5a9f0e44bc2ae77",
    "decision.json": "01a9faefb90ee3d3e7525c0d5191d41741131843dc79ea2b44d860c5c8e21d5e",
}


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


def _load_module(path, expected, name):
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "helper identity")
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        state.checkpoint_hash == _PARENTS[context["seed"]],
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

    _require(isinstance(optimizer, torch.optim.AdamW), "AdamW optimizer")
    for group in optimizer.param_groups:
        _require(
            group["lr"] == 0.0003
            and tuple(group["betas"]) == (0.9, 0.999)
            and group["eps"] == 1e-8
            and group["weight_decay"] == 0
            and not group.get("amsgrad", False)
            and not group.get("maximize", False)
            and not group.get("capturable", False)
            and not group.get("differentiable", False)
            and group.get("foreach") is None
            and group.get("fused") in (None, False),
            "fixed AdamW settings",
        )
    parameters = [getattr(model, n) for n in _PARAMETERS]
    actual = [p for g in optimizer.param_groups for p in g["params"]]
    _require(
        len(actual) == 1 and {id(p) for p in actual} == {id(p) for p in parameters},
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


def _tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _evaluate(
    model,
    records,
    wide_rows,
    mean,
    helper,
    path,
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
                    else bilinear._head(model, features, prompts[:, 2], mean)
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


def _preserve_failure(directory, model, optimizer, training, mean, helper):
    import torch

    path = directory / "failure-state.pt"
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "completed_updates": training.get("completed_updates", 0),
        "optimizer_steps_attempted": None,
        "optimizer_steps_attempted_status": "not-instrumented-frozen-fit",
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
            "optimizer_steps_attempted_status": payload["optimizer_steps_attempted_status"],
            "error": training.get("error"),
        },
    )


def _execute(root, out, context, bilinear, mean, helper, summary):
    seed = context["seed"]
    started = time.perf_counter()
    runtime_root = out.parent / "runtime"
    training = {
        "complete": False,
        "fit_complete": False,
        "phase": "initializing",
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
        seed_everything(seed, deterministic=False)
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
            runtime.checkpoint_hash == _PARENTS[seed]
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
            validation_update_points=[0, _UPDATES],
            training_diagnostic_updates=[8000, _UPDATES],
            training_timing_scope=(
                "refit_seconds includes detached diagnostic copy/reduction/JSON "
                "serialization; not comparable to historical uninstrumented timing"
            ),
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
            mean,
            helper,
            out.parent / "parent.json",
            bilinear,
            phase="parent",
        )
        bilinear._attach(model)
        _require(
            [n for n, p in model.named_parameters() if p.requires_grad] == list(_PARAMETERS),
            "one-parameter trainable mask",
        )
        _require(
            all(int(torch.count_nonzero(getattr(model, n))) == 0 for n in _PARAMETERS),
            "zero residual initialization",
        )
        initial = mean._state_hashes(model)
        _require(
            len(initial) == 94
            and {k: v for k, v in initial.items() if k not in _PARAMETERS} == parent_state,
            "94-entry initial state",
        )
        training["state_before"] = initial
        zero = bilinear._zero_replay(
            model,
            runtime.split.validation,
            parent,
            mean,
            helper,
            out.parent / "zero-replay.json",
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
            scores = bilinear._head(model, features, prompts[:, 2], mean)
            initial_loss = float(_prompt_set_loss(scores, labels, excluded_ids=features[2]).cpu())
        _require(
            list(scores.shape) == [1637, 1025] and initial_loss == _INITIAL_LOSSES[seed],
            "zero-A full train loss replay",
        )
        optimizer = create_optimizer(model, OptimizerConfig.model_validate(recipe["optimizer"]))
        _require(
            not optimizer.state and sum(len(g["params"]) for g in optimizer.param_groups) == 1,
            "fresh one-parameter optimizer",
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
        _require(mean._state_hashes(model) == initial, "state changed before fit")
        _require(
            all(int(torch.count_nonzero(getattr(model, n))) == 0 for n in _PARAMETERS),
            "nonzero residual immediately before fit",
        )
        fit_receipt = {}
        observed_loss = _TrainingLossRecorder(
            _prompt_set_loss,
            context["membership"],
            labels,
            features[2],
            out.parent,
            mean,
            helper,
            membership_sha256=helper._sha(out.parent / "train-membership.json"),
        )
        try:
            bilinear._fit(
                model,
                features,
                prompts[:, 2],
                labels,
                observed_loss,
                optimizer,
                _UPDATES,
                fit_receipt,
                mean,
            )
            _require(fit_receipt["completed_updates"] == _UPDATES, "complete fit")
            observed_loss.finish()
            fit_receipt.update(fit_complete=True, phase="complete")
        finally:
            training.update(fit_receipt)
            training["training_diagnostics"] = observed_loss.report
        training["training_diagnostics"] = observed_loss.report
        training["training_prefix_diagnostic"] = _prefix_diagnostic(
            training, context["historical_training"]
        )
        training["optimizer_final"] = _validate_optimizer(optimizer, model, _UPDATES)
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
            "scorer_sha256": _SCORER,
            "input_adapter_identity": _INPUT_ADAPTER,
            "input_amendment_sha256": _AMENDMENT,
            "plan_sha256": _PLAN,
            "bilinear_helper_sha256": _SCORER,
            "mean_helper_sha256": _MEAN,
            "input_adapter_source_sha256": _ADAPTER_SHA,
            "source_archive_sha256": helper._SOURCE,
            "configs_archive_sha256": helper._CONFIGS,
        }
        child_config = {
            "implementation_identity": implementation,
            "inherited_parent_config": context["parent_run"]["config"],
            "bilinear_residual_architecture": bilinear._architecture(),
            "bilinear_residual_recipe": recipe,
            "seed": seed,
            "data_split_seed": 1729,
            "campaign_version": _CAMPAIGN,
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
                "scorer_sha256": _SCORER,
                "input_adapter_identity": _INPUT_ADAPTER,
                "input_amendment_sha256": _AMENDMENT,
                "plan_sha256": _PLAN,
                "bilinear_helper_sha256": _SCORER,
                "recipe_sha256": summary["recipe_sha256"],
                "mean_helper_sha256": _MEAN,
                "input_adapter_source_sha256": _ADAPTER_SHA,
                "architecture": _ARCHITECTURE,
            },
            seeds=(seed,),
        )
        metadata = {
            "implementation_identity": implementation,
            "seed": seed,
            "data_split_seed": 1729,
            "campaign_version": _CAMPAIGN,
            "objective": _OBJECTIVE,
            "evaluator": _EVALUATOR,
            "architecture": _ARCHITECTURE,
            "model_config": model.config.model_dump(mode="json"),
            "model_config_scope": "inherited archived base only; residual architecture is separate",
            "parent_checkpoint_sha256": _PARENTS[seed],
            "parent_training_steps": 2000,
            "residual_updates": _UPDATES,
            "trainable_parameters": list(_PARAMETERS),
            "inference": _INFERENCE,
            "train_query_order_sha256": summary["train_query_order_sha256"],
            "record_count": 1637,
            "recipe_sha256": summary["recipe_sha256"],
            "runner_sha256": summary["script_sha256"],
            "scorer_sha256": _SCORER,
            "input_adapter_identity": _INPUT_ADAPTER,
            "input_amendment_sha256": _AMENDMENT,
            "plan_sha256": _PLAN,
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
            global_step=_UPDATES,
            config=child_config,
            identity=identity,
            corpus_identity=runtime.corpus_identity,
            split_hash=mean._SPLIT,
            training_metadata=metadata,
        )
        reloaded = bilinear._attach(build_model(model.config).float().to(runtime.device).eval())
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
        _validate_optimizer(reload_optimizer, reloaded, _UPDATES)
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
            bilinear=bilinear,
            parent=parent,
        )
        comparisons = {
            "historical8000_summary_sha256": context["historical8000_summary_sha256"],
            "historical8000_child_sha256": context["historical8000_child_sha256"],
            "comparison_scope": "saved historical output; no rerun",
            "paired_vs_historical8000": _historical_pair(child, context["historical8000"], mean),
        }
        mean._write(out.parent / "comparisons.json", comparisons)
        invariants = {
            "parent_replay_exact": parent["exact_parent_replay"],
            "zero_A_replay_exact": zero["complete"]
            and zero["logits_sha256"] == zero["parent_logits_sha256"],
            "initial_train_loss_exact": initial_loss == _INITIAL_LOSSES[seed],
            "completed_32000_updates": training["completed_updates"] == _UPDATES,
            "original_93_tensors_unchanged": training["frozen_tensors_unchanged"],
            "residual_changed": training["residual_changed"],
            "reload_exact": training["reload_exact"],
            "optimizer_reload_exact": training["optimizer_reload_exact"],
            "train_only_supervision": training["validation_labels_used"] is False,
        }
        summary.update(
            invariants=invariants,
            gate=_gate(seed, child, invariants),
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
                "train-scores-post8000.json",
                "train-scores-post32000.json",
                "train-diagnostics.json",
            }
        }
        summary["wall_seconds"] = time.perf_counter() - started
        mean._write(out, helper._failure_safe(summary))


def _adapter(root):
    return _load_module(
        root / "scripts/refit_symmetric_affine_v3.py", _ADAPTER_SHA, "frozen_partition_adapter"
    )


def _selected_partitions(membership, parent, queries, vocabulary, adapter):
    return adapter._selected_partitions(membership, parent, queries, vocabulary)


def _recipe(value, inherited):
    expected = json.loads(json.dumps(inherited))
    for key in (
        "seed",
        "initial_training_loss",
        "strong_comparator_exact",
        "strong_comparator_f1",
        "strong_comparator_group_exact",
    ):
        expected.pop(key)
    expected.update(
        experiment=_CAMPAIGN,
        evaluator=_EVALUATOR,
        inference=_INFERENCE,
        residual_updates=_UPDATES,
        seeds=list(_SEEDS),
        data_split_seed=1729,
        input_adapter=_INPUT_ADAPTER,
        input_amendment_sha256=_AMENDMENT,
        plan_sha256=_PLAN,
        input_adapter_source_sha256=_ADAPTER_SHA,
        standard_serving_supported=False,
        validation_update_points=[0, _UPDATES],
        training_diagnostic_updates=[8000, _UPDATES],
        seed_contracts={
            str(seed): {
                "parent_checkpoint_sha256": _PARENTS[seed],
                "initial_training_loss": _INITIAL_LOSSES[seed],
                "floors": _FLOORS[seed],
            }
            for seed in _SEEDS
        },
    )
    _require(value == expected, "fixed 32000 recipe")
    return value


def _gate(seed, child, invariants):
    _require(type(seed) is int and seed in _SEEDS, "declared seed")
    floors = _FLOORS[seed]
    expected = {
        "parent_replay_exact",
        "zero_A_replay_exact",
        "initial_train_loss_exact",
        "completed_32000_updates",
        "original_93_tensors_unchanged",
        "residual_changed",
        "reload_exact",
        "optimizer_reload_exact",
        "train_only_supervision",
    }
    checks = {
        "execution_invariants": set(invariants) == expected
        and all(v is True for v in invariants.values()),
        "exact_above_matching8000": child["aggregate"]["exact_count"] > floors["exact_count"],
        "f1_at_least_matching8000": child["aggregate"]["f1"] >= floors["f1"],
        "group_exact_nonregression": all(
            child["groups"][g]["exact_count"] >= n for g, n in floors["groups"].items()
        ),
        "serialization_compatible": child["aggregate"]["serialization_compatible"] == 222,
    }
    return {
        "accepted": False,
        "independent_audit_required": True,
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
    }


def _campaign_gate(reports):
    _require(set(reports) == set(_SEEDS), "complete three-seed campaign")
    checks = {}
    for seed in _SEEDS:
        r = reports[seed]
        _require(r["seed"] == seed and r["complete"] is True, "complete matching seed")
        expected = _gate(seed, r["child"], r["invariants"])
        _require(r["gate"] == expected, "seed gate recomputation")
        checks[str(seed)] = expected["primary_checks_passed"]
    return {
        "accepted": False,
        "independent_audit_required": True,
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
    }


def _validate_child(state, identity, config, metadata, corpus):
    raw = state.metadata
    seed = metadata["seed"]
    _require(type(seed) is int and seed in _SEEDS, "child seed")
    _require(state.global_step == raw["global_step"] == _UPDATES, "child update count")
    _require(
        raw["experiment_identity"] == identity
        and identity["seeds"] == [seed]
        and identity["evaluator_version"] == _EVALUATOR,
        "child identity",
    )
    _require(raw["config"] == config and raw["training_metadata"] == metadata, "child metadata")
    _require(
        config["seed"] == seed
        and config["data_split_seed"] == metadata["data_split_seed"] == 1729
        and config["campaign_version"] == metadata["campaign_version"] == _CAMPAIGN,
        "seed/campaign",
    )
    implementation = metadata["implementation_identity"]
    _require(
        implementation == config["implementation_identity"]
        and implementation["runner_sha256"] == metadata["runner_sha256"]
        and implementation["scorer_sha256"] == metadata["scorer_sha256"] == _SCORER
        and implementation["mean_helper_sha256"] == _MEAN
        and implementation["input_adapter_source_sha256"] == _ADAPTER_SHA,
        "implementation identity",
    )
    _require(
        metadata["architecture"] == _ARCHITECTURE
        and config["bilinear_residual_architecture"]["id"] == _ARCHITECTURE
        and metadata["objective"] == _OBJECTIVE
        and metadata["evaluator"] == _EVALUATOR
        and metadata["inference"] == _INFERENCE
        and metadata["trainable_parameters"] == list(_PARAMETERS)
        and metadata["parent_training_steps"] == 2000
        and metadata["residual_updates"] == _UPDATES
        and metadata["parent_checkpoint_sha256"] == _PARENTS[seed],
        "child architecture/objective/lineage",
    )
    _require(
        metadata["standard_serving_supported"] is False
        and metadata["input_adapter_identity"] == _INPUT_ADAPTER
        and metadata["input_amendment_sha256"] == _AMENDMENT
        and metadata["plan_sha256"] == _PLAN,
        "child input declaration",
    )
    _require(raw["corpus_identity"] == corpus and raw["split_hash"] == _SPLIT, "child corpus/split")


def _training_diagnostic(scores, membership, post_update, mean):
    _require(
        len(scores) == membership["query_count"] == len(membership["queries"]),
        "train score coverage",
    )
    rows = []
    for index, (vector, q) in enumerate(zip(scores, membership["queries"], strict=True)):
        _require(q["index"] == index and len(vector) == 1025, "training row/columns")
        subject = q["prompt_ids"][1]
        truth = q["expected_set_ids"]
        selected = mean._prediction(vector, subject)
        expected = set(truth)
        negatives = [i for i in range(1024, 2049) if i != subject and i not in expected]
        _require(
            bool(expected) and bool(negatives) and subject not in expected, "training boundary sets"
        )
        positive_id = min(expected, key=lambda i: (vector[i - 1024], i))
        negative_id = min(negatives, key=lambda i: (-vector[i - 1024], i))
        low, high = vector[positive_id - 1024], vector[negative_id - 1024]
        rows.append(
            {
                **q,
                "group": q["dimension"],
                "selected_set_ids": selected,
                "metrics": mean._metrics(selected, truth),
                "strict_separation": low > high,
                "min_positive": low,
                "min_positive_id": positive_id,
                "max_negative": high,
                "max_negative_id": negative_id,
                "zero_margin": min(low, -high),
                "separation_gap": low - high,
            }
        )
    minimum = min(rows, key=lambda row: (row["zero_margin"], row["index"]))
    return {
        "post_update": post_update,
        "query_count": len(rows),
        "aggregate": mean._totals(rows),
        "groups": {
            g: mean._totals([r for r in rows if r["dimension"] == g])
            for g in ("TYPE", "COLOR")
            if any(r["dimension"] == g for r in rows)
        },
        "minimum_signed_margin": minimum["zero_margin"],
        "minimum_signed_margin_index": minimum["index"],
        "rows": rows,
    }


class _TrainingLossRecorder:
    """Observe detached fit-only call8001 and finalcall32001 without changing loss."""

    def __init__(
        self,
        loss_fn,
        membership,
        labels,
        subjects,
        directory,
        mean,
        helper,
        *,
        membership_sha256,
        endpoints=(8000, 32000),
    ):
        _require(
            isinstance(endpoints, tuple)
            and len(endpoints) == 2
            and all(type(n) is int and n >= 0 for n in endpoints)
            and endpoints[0] < endpoints[1],
            "ordered observation endpoints",
        )
        self.loss_fn = loss_fn
        self.membership = membership
        self.labels = labels
        self.subjects = subjects
        self.directory = directory
        self.mean = mean
        self.helper = helper
        self.membership_sha256 = membership_sha256
        self.endpoints = endpoints
        self.calls = 0
        self.report = {
            "partition": "train",
            "descriptive_only": True,
            "used_for_selection_or_updates": False,
            "complete": False,
            "endpoints": {},
        }

    def __call__(self, scores, labels, *, excluded_ids):
        import torch

        _require(labels is self.labels and excluded_ids is self.subjects, "fit loss input identity")
        _require(self.report["complete"] is False, "observer already complete")
        result = self.loss_fn(scores, labels, excluded_ids=excluded_ids)
        self.calls += 1
        step = self.calls - 1
        if step in self.endpoints:
            _require(str(step) not in self.report["endpoints"], "duplicate observation")
            _require(
                scores.dtype == torch.float32
                and scores.ndim == 2
                and bool(torch.isfinite(scores).all()),
                "observed FP32 scores",
            )
            if self.endpoints == (8000, 32000):
                _require(torch.is_grad_enabled() is (step == 8000), "observation gradient mode")
            values = scores.detach().cpu().tolist()
            reduced = _training_diagnostic(values, self.membership, step, self.mean)
            path = self.directory / f"train-scores-post{step}.json"
            _require(not path.exists(), "immutable train score endpoint")
            raw = {
                "complete": True,
                "partition": "train",
                "post_update": step,
                "dtype": "float32",
                "fit_loss_callback": self.calls,
                "grad_enabled": torch.is_grad_enabled(),
                "affects_optimization": False,
                "capture_alignment": f"pre-update-{step + 1}"
                if step != self.endpoints[-1]
                else "final-post-update",
                "training_membership_sha256": self.membership_sha256,
                "product_token_ids": list(range(1024, 2049)),
                "query_count": len(values),
                "score_bytes_sha256": hashlib.sha256(_tensor_bytes(scores)).hexdigest(),
                "responses": [
                    {**q, "symmetric_relation_logits": v}
                    for q, v in zip(self.membership["queries"], values, strict=True)
                ],
            }
            self.mean._write(path, raw)
            reduced.update(score_artifact=path.name, score_artifact_sha256=self.helper._sha(path))
            self.report["endpoints"][str(step)] = reduced
        return result

    def finish(self):
        _require(self.report["complete"] is False, "observer already complete")
        _require(
            self.calls == max(self.endpoints) + 1
            and set(self.report["endpoints"]) == {str(s) for s in self.endpoints},
            "complete diagnostic chronology",
        )
        self.report.update(complete=True, loss_callback_calls=self.calls)
        self.mean._write(self.directory / "train-diagnostics.json", self.report)


def _prefix_diagnostic(training, historical):
    old = historical["history"]
    _require(len(old) == 8000 and len(training["history"]) == _UPDATES, "training prefix length")
    mismatch = [
        i + 1
        for i, (a, b) in enumerate(zip(training["history"][:8000], old, strict=True))
        if a["pre_update_loss"] != b["pre_update_loss"]
    ]
    return {
        "descriptive_only": True,
        "used_for_selection_or_updates": False,
        "pre_update_compared_count": 8000,
        "mismatch_count": len(mismatch),
        "first_mismatch_update": mismatch[0] if mismatch else None,
        "all_pre_update_equal": not mismatch,
        "historical_final_post_update_loss": historical["final_post_update_loss"],
        "current_pre_update_8001_loss": training["history"][8000]["pre_update_loss"],
        "boundary_loss_equal": training["history"][8000]["pre_update_loss"]
        == historical["final_post_update_loss"],
    }


def _parent_config_hash(config):
    # Original training identities use compact canonical JSON, not report formatting.
    payload = json.dumps(
        {k: v for k, v in config.items() if k != "run_name"},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parent_contract(seed, run, sidecar, summary, mean):
    _require(type(seed) is int and seed in _SEEDS, "original parent seed")
    identity = run["identity"]
    config = run["config"]
    _require(
        config["seed"] == seed
        and config["run_name"] == f"national_dex_continuation_control_s{seed}_v1"
        and config["data"]["split_seed"] == 1729,
        "parent config seed/split",
    )
    _require(
        _parent_config_hash(config) == identity["resolved_config_hash"],
        "parent resolved config hash",
    )
    _require(
        identity == sidecar["experiment_identity"] == summary["parent_training_identity"]
        and identity["seeds"] == [seed]
        and identity["split_hash"] == _SPLIT,
        "original parent identity",
    )
    _require(
        sidecar["global_step"] == 2000
        and sidecar["checkpoint_hash"] == _PARENTS[seed]
        and sidecar["split_hash"] == _SPLIT
        and sidecar["config"] == config["train"]
        and sidecar["training_metadata"]["objective"] == "causal-next-token-v1"
        and sidecar["corpus_identity"] == summary["corpus_identity"],
        "original parent metadata",
    )


def _accepted_control(root, seed, helper, inputs):
    if seed == 1729:
        directory = root / "runs/learning/bilinear-budget8000-v1"
        values = {
            n: helper._read(helper._bind(directory / n, h, inputs)) for n, h in _CONTROL1729.items()
        }
        summary, audit, decision = (
            values[n] for n in ("summary.json", "independent-audit.json", "decision.json")
        )
        pins = {
            "summary": _CONTROL1729["summary.json"],
            "audit": _CONTROL1729["independent-audit.json"],
            "decision": _CONTROL1729["decision.json"],
        }
        _require(decision["fixed_quality_gate_passed"] is True, "accepted historical gate")
    else:
        directory = root / "runs/learning/bilinear-budget8000-replication-v1"
        values = {
            n: helper._read(helper._bind(directory / n, h, inputs))
            for n, h in _CONTROL_REPLICATION.items()
        }
        aggregate, aggregate_audit, decision = (
            values[n] for n in ("aggregate-v2.json", "independent-audit-v2.json", "decision.json")
        )
        _require(
            aggregate["complete"] is True
            and aggregate_audit["audit_passed"] is True
            and aggregate_audit["summary_sha256"]
            == decision["summary_sha256"]
            == _CONTROL_REPLICATION["aggregate-v2.json"]
            and decision["audit_sha256"] == _CONTROL_REPLICATION["independent-audit-v2.json"],
            "accepted replication aggregate",
        )
        records = [c for c in decision["children"] if c["seed"] == seed]
        _require(len(records) == 1, "matching historical seed")
        child = records[0]
        directory = directory / f"seed-{seed}"
        pins = {
            "summary": child["summary_sha256"],
            "audit": child["audit_sha256"],
            "decision": _CONTROL_REPLICATION["decision.json"],
        }
        summary = helper._read(helper._bind(directory / "summary.json", pins["summary"], inputs))
        audit = helper._read(
            helper._bind(directory / "independent-audit-v2.json", pins["audit"], inputs)
        )
        _require(
            summary["seed"] == seed and child["quality_gate_passed"] is True, "accepted seed screen"
        )
    _require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == pins["summary"]
        and decision["evidence_accepted"] is True,
        "accepted seed-matched8000 evidence",
    )
    _require(
        summary["objective"] == _OBJECTIVE and summary["split_hash"] == _SPLIT, "control identities"
    )
    _require(
        summary["child"]["aggregate"]["exact_count"] == _FLOORS[seed]["exact_count"]
        and summary["child"]["aggregate"]["f1"] == _FLOORS[seed]["f1"]
        and all(
            summary["child"]["groups"][g]["exact_count"] == n
            for g, n in _FLOORS[seed]["groups"].items()
        ),
        "fixed comparator floors",
    )
    return directory, summary, audit, pins


def _preflight(root, seed, plan, recipe_path, bilinear, mean, helper, adapter):
    _require(type(seed) is int and seed in _SEEDS, "declared seed")
    _require(
        plan.resolve()
        == (root / "docs/experiments/2026-09-26-bilinear-budget32000-plan.md").resolve()
        and recipe_path.resolve()
        == (root / "configs/experiments/bilinear_budget32000_v1.json").resolve(),
        "fixed declaration paths",
    )
    inputs = {}
    directory, summary, _audit, pins = _accepted_control(root, seed, helper, inputs)
    selected = {
        n: helper._read(helper._bind(directory / n, summary["artifact_sha256"][n], inputs))
        for n in ("train-membership.json", "parent.json", "child.json", "training.json")
    }
    membership, parent, historical, training = (
        selected[n] for n in ("train-membership.json", "parent.json", "child.json", "training.json")
    )
    _require(mean._digest(membership) == _MEMBERSHIP, "selected training membership")
    _require(
        training["initial_pre_update_loss"] == _INITIAL_LOSSES[seed]
        and training["completed_updates"] == 8000,
        "historical training endpoint",
    )
    _historical_pair(parent, historical, mean)
    parent_dir = root / f"runs/national_dex_continuation_control_s{seed}_v1"
    parent_path = helper._bind(parent_dir / "checkpoint-final.pt", _PARENTS[seed], inputs)

    def metadata(name):
        path = parent_dir / name
        relative = path.relative_to(root).as_posix()
        candidates = [
            h
            for n, h in summary["input_sha256"].items()
            if Path(n).as_posix().replace("\\", "/").endswith("/" + relative) or n == relative
        ]
        _require(len(candidates) == 1, "explicit original metadata identity")
        return helper._read(helper._bind(path, candidates[0], inputs))

    parent_run, parent_sidecar = metadata("run.json"), metadata("checkpoint-final.pt.json")
    _parent_contract(seed, parent_run, parent_sidecar, summary, mean)
    vocab_path = root / "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json"
    vocabulary = helper._read(
        helper._bind(
            vocab_path, "1b0d8f1762fede01fb253c3df23d21ced254a10f6cea367709c479565a92fb34", inputs
        )
    )
    partitions = _selected_partitions(membership, parent, summary["queries"], vocabulary, adapter)
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
    for relative, digest in [
        (str(plan.relative_to(root)), _PLAN),
        ("docs/experiments/2026-09-26-symmetric-affine-input-amendment.md", _AMENDMENT),
        ("scripts/refit_symmetric_affine_v3.py", _ADAPTER_SHA),
        ("scripts/refit_bilinear_residual.py", _SCORER),
        ("scripts/refit_membership_projection.py", _MEAN),
        ("runs/learning/wide-first-choice-v1/summary.script.py", mean._HELPER),
    ]:
        helper._bind(root / relative, digest, inputs)
    # Seed1729's archived recipe defines the unchanged scalar/optimizer baseline only.
    recipe_control = root / "runs/learning/bilinear-budget8000-v1/summary.recipe.json"
    control_summary = helper._read(
        helper._bind(
            root / "runs/learning/bilinear-budget8000-v1/summary.json",
            _CONTROL1729["summary.json"],
            inputs,
        )
    )
    inherited_recipe = helper._read(
        helper._bind(recipe_control, control_summary["recipe_sha256"], inputs)
    )
    recipe = _recipe(helper._read(recipe_path), inherited_recipe)
    helper._bind(recipe_path, helper._sha(recipe_path), inputs)
    adapter_receipt = {
        "identity": _INPUT_ADAPTER,
        "amendment_sha256": _AMENDMENT,
        "source_sha256": _ADAPTER_SHA,
        "inherited_corpus_identity": True,
        "whole_corpus_revalidated": False,
        "evidence_sha256": {n: summary["artifact_sha256"][n] for n in selected},
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
        "seed": seed,
        "inputs": inputs,
        "recipe": recipe,
        "membership": membership,
        "parent_state": training["parent_state_before"],
        "parent": parent_path,
        "parent_run": parent_run,
        "parent_sidecar": parent_sidecar,
        "corpus_identity": summary["corpus_identity"],
        "inherited_architecture": summary["inherited_architecture"],
        "vocabulary_path": vocab_path,
        "partitions": partitions,
        "input_adapter": adapter_receipt,
        "wide": parent,
        "historical8000": historical,
        "historical_training": training,
        "historical_environment": summary["environment"],
        "historical8000_child_sha256": summary["artifact_sha256"]["child.json"],
        **{f"historical8000_{k}_sha256": v for k, v in pins.items()},
        "source": source,
        "configs": configs,
        "members": members,
    }


_READINESS_PATH = "runs/learning/bilinear-budget32000-readiness-v1/readiness.json"
_TEST_DEPENDENCIES = {
    "runner": {
        "scripts/refit_bilinear_budget32000.py",
        "tests/unit/test_bilinear_budget32000.py",
        "scripts/refit_bilinear_residual.py",
        "scripts/refit_membership_projection.py",
        "runs/learning/wide-first-choice-v1/summary.script.py",
        "scripts/refit_symmetric_affine_v3.py",
        "tests/unit/test_bilinear_residual.py",
        "tests/fixtures/projection_refit_runtime_source.zip",
        "docs/experiments/2026-09-26-bilinear-budget32000-plan.md",
        "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md",
    },
    "auditor": {
        "runs/learning/bilinear-budget32000-v1/independent-audit.py",
        "tests/unit/test_bilinear_budget32000_audit.py",
        "runs/learning/projection-only-refit-v1/independent-audit.py",
        "docs/experiments/2026-09-26-bilinear-budget32000-plan.md",
        "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md",
    },
    "aggregate": {
        "scripts/aggregate_bilinear_budget32000.py",
        "tests/unit/test_bilinear_budget32000_aggregate.py",
        "scripts/refit_membership_projection.py",
        "docs/experiments/2026-09-26-bilinear-budget32000-plan.md",
    },
}
_TEST_SCRIPTS = {
    "runner": "scripts/refit_bilinear_budget32000.py",
    "auditor": "runs/learning/bilinear-budget32000-v1/independent-audit.py",
    "aggregate": "scripts/aggregate_bilinear_budget32000.py",
}
_TEST_DIRS = {
    role: f"runs/learning/bilinear-budget32000-{role}-tests-v1" for role in _TEST_DEPENDENCIES
}
_READINESS_FILES = (
    {"configs/experiments/bilinear_budget32000_v1.json"}
    | set().union(*_TEST_DEPENDENCIES.values())
    | {
        directory + "/" + name
        for directory in _TEST_DIRS.values()
        for name in ("test-receipt.json", "stdout.txt")
    }
)


def _cpu_receipt(receipt, role, files):
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
        "observed CPU test receipt",
    )
    dependencies = receipt.get("test_dependencies")
    _require(
        type(dependencies) is dict and set(dependencies) == _TEST_DEPENDENCIES[role],
        "exact CPU dependency inventory",
    )
    _require(all(dependencies[n] == files[n] for n in dependencies), "CPU dependency hash identity")
    _require(
        receipt.get("tested_script_sha256") == files[_TEST_SCRIPTS[role]], "tested script identity"
    )
    if role == "runner":
        _require(
            receipt.get("tested_recipe_sha256")
            == files["configs/experiments/bilinear_budget32000_v1.json"],
            "tested recipe identity",
        )
    _require(
        receipt.get("stdout")
        == {
            "path": _TEST_DIRS[role] + "/stdout.txt",
            "sha256": files[_TEST_DIRS[role] + "/stdout.txt"],
        },
        "tested stdout identity",
    )


def _readiness(root, path, pinned, helper, inputs):
    path = _resolve(root, path)
    _require(path == (root / _READINESS_PATH).resolve(), "fixed campaign readiness path")
    _require(
        type(pinned) is str and len(pinned) == 64 and all(c in "0123456789abcdef" for c in pinned),
        "required readiness SHA256",
    )
    contract = helper._read(helper._bind(path, pinned, inputs))
    _require(
        type(contract.get("schema_version")) is int
        and contract["schema_version"] == 1
        and contract.get("campaign_version") == _CAMPAIGN
        and contract.get("plan_sha256") == _PLAN
        and contract.get("ready") is True,
        "campaign readiness identity",
    )
    files = contract.get("files")
    _require(type(files) is dict and set(files) == _READINESS_FILES, "exact readiness inventory")
    _require(
        all(
            type(h) is str and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
            for h in files.values()
        ),
        "readiness hash format",
    )
    # Resolve the complete named inventory before hashing any member.
    paths = {name: _resolve(root, name) for name in files}
    for name, path_ref in paths.items():
        helper._bind(path_ref, files[name], inputs)
    _require(
        files["docs/experiments/2026-09-26-bilinear-budget32000-plan.md"] == _PLAN
        and files["docs/experiments/2026-09-26-symmetric-affine-input-amendment.md"] == _AMENDMENT
        and files["scripts/refit_symmetric_affine_v3.py"] == _ADAPTER_SHA
        and files["scripts/refit_bilinear_residual.py"] == _SCORER
        and files["scripts/refit_membership_projection.py"] == _MEAN,
        "fixed readiness helper identity",
    )
    for role, directory in _TEST_DIRS.items():
        _cpu_receipt(helper._read(paths[directory + "/test-receipt.json"]), role, files)
    return contract


def _instant(value):
    _require(type(value) is str, "timestamp string required")
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})", value
    )
    _require(match is not None, "timezone-aware timestamp required")
    if match[3] != "Z":
        _require(int(match[3][1:3]) < 24 and int(match[3][4:6]) < 60, "invalid UTC offset")
    delta = datetime.fromisoformat(match[1] + match[3]) - datetime(1970, 1, 1, tzinfo=UTC)
    return Fraction(delta.days * 86400 + delta.seconds) + (
        Fraction(int(match[2]), 10 ** len(match[2])) if match[2] else Fraction(0)
    )


def _previous_seed(root, seed, ready, helper, inputs, readiness_sha256, admission_utc):
    if seed == 1729:
        return None
    previous = seed - 1
    directory = root / f"runs/learning/{_CAMPAIGN}/seed-{previous}"

    # Only explicit prior terminal/audit metadata; never a recursive manifest read.
    def read(name):
        path = directory / name
        return helper._read(helper._bind(path, helper._sha(path), inputs))

    summary = read("summary.json")
    execution = read("execution-receipt.json")
    audit = read("independent-audit.json")
    terminal = read("audit-execution-receipt.json")
    summary_sha = helper._sha(directory / "summary.json")
    _require(
        summary["shared_readiness"] == {"path": _READINESS_PATH, "sha256": readiness_sha256}
        and summary["script_sha256"] == ready["files"][_TEST_SCRIPTS["runner"]]
        and summary["recipe_sha256"]
        == ready["files"]["configs/experiments/bilinear_budget32000_v1.json"],
        "same campaign admission",
    )
    _require(
        terminal["auditor_sha256"] == ready["files"][_TEST_SCRIPTS["auditor"]]
        and terminal["test_receipt_sha256"]
        == ready["files"][_TEST_DIRS["auditor"] + "/test-receipt.json"]
        and terminal["cuda_visible_devices"] == "-1",
        "previous tested audit terminal",
    )
    _require(
        _instant(execution["started_utc"])
        <= _instant(execution["completed_utc"])
        <= _instant(terminal["started_utc"])
        <= _instant(terminal["completed_utc"])
        <= _instant(admission_utc),
        "previous audited chronology",
    )
    _require(
        summary["seed"] == previous
        and summary["complete"] is True
        and summary["final_identity_check"] is True
        and summary["gate"]["checks"]["execution_invariants"] is True,
        "previous execution completed",
    )
    _require(
        audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["seed"] == previous
        and audit["plan_sha256"] == _PLAN
        and audit["shared_readiness"] == summary["shared_readiness"]
        and audit["summary_sha256"] == summary_sha
        and audit["script_sha256"] == ready["files"][_TEST_SCRIPTS["auditor"]]
        and audit["test_receipt_sha256"]
        == ready["files"][_TEST_DIRS["auditor"] + "/test-receipt.json"]
        and audit["execution_receipt_sha256"] == helper._sha(directory / "execution-receipt.json"),
        "previous independently audited",
    )
    for receipt in (execution, terminal):
        _require(
            type(receipt.get("exit_code")) is int
            and receipt["exit_code"] == 0
            and receipt.get("terminal_completion_observed_by_primary") is True
            and receipt["seed"] == previous
            and receipt["summary_sha256"] == summary_sha,
            "previous observed terminal",
        )
    _require(
        terminal["audit_sha256"] == helper._sha(directory / "independent-audit.json")
        and terminal["execution_receipt_sha256"]
        == helper._sha(directory / "execution-receipt.json"),
        "previous audit terminal binding",
    )
    helper._bind(directory / "primary-stdout.txt", execution["stdout_sha256"], inputs)
    helper._bind(directory / "audit-stdout.txt", terminal["stdout_sha256"], inputs)
    return {
        "seed": previous,
        "summary_sha256": summary_sha,
        "completed_utc": execution["completed_utc"],
        "execution_receipt_sha256": helper._sha(directory / "execution-receipt.json"),
        "audit_execution_receipt_sha256": helper._sha(directory / "audit-execution-receipt.json"),
    }


def _admit_output(root, seed, out, mean):
    _require(
        out.resolve() == (root / f"runs/learning/{_CAMPAIGN}/seed-{seed}/summary.json").resolve(),
        "fixed seed output path",
    )
    mean._refuse(out)
    for name in (
        "zero-replay.json",
        "comparisons.json",
        "failure-state.pt",
        "failure-state.json",
        "train-scores-post8000.json",
        "train-scores-post32000.json",
        "train-diagnostics.json",
        "readiness",
        "runtime",
    ):
        _require(not (out.parent / name).exists(), "immutable execution output: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=_SEEDS, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/experiments/2026-09-26-bilinear-budget32000-plan.md"),
    )
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/experiments/bilinear_budget32000_v1.json")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--readiness-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    _require(
        Path.cwd().resolve() == root and "torch" not in sys.modules, "Torch-free root preflight"
    )
    script = Path(__file__).resolve()
    content = script.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    bilinear, bilinear_path = _load_bilinear(root)
    mean, mean_path = bilinear._load_mean(root)
    helper, helper_path = mean._load_helper(root)
    adapter = _adapter(root)
    helper._modules()
    readiness_inputs = {}
    ready = _readiness(root, args.readiness, args.readiness_sha256, helper, readiness_inputs)
    out = (args.out or root / f"runs/learning/{_CAMPAIGN}/seed-{args.seed}/summary.json").resolve()
    _admit_output(root, args.seed, out, mean)
    context = _preflight(
        root, args.seed, args.plan.resolve(), args.recipe.resolve(), bilinear, mean, helper, adapter
    )
    context["inputs"].update(readiness_inputs)
    helper._bind(script, digest, context["inputs"])
    helper._unchanged(context["inputs"])
    if args.preflight_only:
        _require("torch" not in sys.modules, "preflight imported torch")
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "seed": args.seed,
                    "script_sha256": digest,
                    "recipe_sha256": helper._sha(args.recipe),
                    "input_count": len(context["inputs"]),
                    "readiness_sha256": args.readiness_sha256,
                    "torch_imported": False,
                }
            )
        )
        return
    predecessor = _previous_seed(
        root,
        args.seed,
        ready,
        helper,
        context["inputs"],
        args.readiness_sha256,
        datetime.now(UTC).isoformat(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": content,
        "bilinear-helper.py": bilinear_path.read_bytes(),
        "mean-helper.py": mean_path.read_bytes(),
        "helper.py": helper_path.read_bytes(),
        "input-adapter.py": (root / "scripts/refit_symmetric_affine_v3.py").read_bytes(),
        "plan.md": args.plan.read_bytes(),
        "input-amendment.md": (
            root / "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md"
        ).read_bytes(),
        "recipe.json": args.recipe.read_bytes(),
        "source.zip": context["source"].read_bytes(),
        "configs.zip": context["configs"].read_bytes(),
        "readiness.json": args.readiness.read_bytes(),
        "inputs.json": mean._bytes(context["inputs"]),
    }
    hashes = {}
    for suffix, data in snapshots.items():
        path = out.with_suffix("." + suffix)
        with path.open("xb") as stream:
            stream.write(data)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
    for relative, expected in ready["files"].items():
        path = out.parent / "readiness" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (root / relative).read_bytes()
        _require(hashlib.sha256(data).hexdigest() == expected, "readiness copy identity")
        with path.open("xb") as stream:
            stream.write(data)
        hashes[str(path)] = expected
    summary = {
        "campaign_version": _CAMPAIGN,
        "seed": args.seed,
        "data_split_seed": 1729,
        "complete": False,
        "acceptance": False,
        "architecture": _ARCHITECTURE,
        "objective": _OBJECTIVE,
        "inference": _INFERENCE,
        "evaluator": _EVALUATOR,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "scorer_sha256": _SCORER,
        "mean_helper_sha256": _MEAN,
        "helper_sha256": mean._HELPER,
        "input_adapter_identity": _INPUT_ADAPTER,
        "input_adapter_source_sha256": _ADAPTER_SHA,
        "input_amendment_sha256": _AMENDMENT,
        "recipe_sha256": helper._sha(args.recipe),
        "shared_readiness": {"path": _READINESS_PATH, "sha256": args.readiness_sha256},
        "prior_seed_execution": predecessor,
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "protected_test_used": False,
        "standard_serving_supported": False,
        **{
            k: context[k]
            for k in (
                "historical8000_summary_sha256",
                "historical8000_audit_sha256",
                "historical8000_decision_sha256",
                "historical8000_child_sha256",
            )
        },
        "limitations": [
            "Three development-selected parents reuse222 validation queries; "
            "no independent666sample claim.",
            "Training diagnostics are detached and include serialization overhead "
            "in refit_seconds.",
            "No threshold tuning, serving integration, oracle parity or durable weight archive.",
        ],
    }
    _execute(root, out, context, bilinear, mean, helper, summary)


if __name__ == "__main__":
    main()
