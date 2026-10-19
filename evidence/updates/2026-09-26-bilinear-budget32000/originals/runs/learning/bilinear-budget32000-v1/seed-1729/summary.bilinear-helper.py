"""Frozen-parent full symmetric bilinear residual screen; standalone head only.

The inherited decoder forward does not evaluate this architecture. Only the
explicit scorer below represents the child; standard serving is unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

_PLAN = "b1cd8a897e5db7d0dc0adc3e3bd40a87b0ae5830714b63ea86649f1599c9e132"
_MEAN = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
_SUMMARY = "0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61"
_AUDIT = "6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc"
_DECISION = "da0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6"
_MEMBERSHIP = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"
_ARCHITECTURE = "plm-frozen-symmetric-bilinear-residual-v1"
_OBJECTIVE = "plm-bilinear-residual-balanced-bce-v1"
_EVALUATOR = "plm-bilinear-residual-screen-v1"
_PARAMETER = "symmetric_bilinear_residual"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _architecture():
    return {
        "id": _ARCHITECTURE,
        "parameter": _PARAMETER,
        "shape": [2, 256, 256],
        "dimension_token_ids": [32, 33],
        "dimensions": ["TYPE", "COLOR"],
        "dtype": "float32",
        "initialization": "exact-zero",
        "symmetrization": "(A+A.transpose(-1,-2))*0.5",
        "score_order": (
            "stack-linear-entities-gather-subject-linear-products-divide16-add-archived-parent-v1"
        ),
    }


def _load_mean(root):
    path = root / "scripts/refit_membership_projection.py"
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == _MEAN, "mean helper identity")
    spec = importlib.util.spec_from_file_location("bilinear_mean_helper", path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    mean = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mean)
    return mean, path


def _recipe(value, mean, inherited):
    expected = json.loads(json.dumps(mean._recipe(inherited)))
    expected.pop("projection_updates")
    expected.pop("projection_shape")
    expected.update(
        experiment="bilinear-residual-refit-v1",
        objective=_OBJECTIVE,
        evaluator=_EVALUATOR,
        architecture=_ARCHITECTURE,
        residual_updates=500,
        trainable_parameter=_PARAMETER,
        residual_shape=[2, 256, 256],
        dimension_token_ids=[32, 33],
        initial_training_loss=0.003822767175734043,
        initialization="exact-zero",
    )
    _require(mean._bytes(value) == mean._bytes(expected), "fixed residual recipe differs")
    return value


def _preflight(root, plan, recipe_path, mean, mean_path, helper, helper_path):
    old = root / "runs/learning/projection-only-refit-v1"
    context = mean._preflight(
        root, old / "summary.plan.md", old / "summary.recipe.json", helper, helper_path
    )
    inputs = context["inputs"]
    helper._bind(plan, _PLAN, inputs)
    helper._bind(mean_path, _MEAN, inputs)
    summary = helper._read(helper._bind(old / "summary.json", _SUMMARY, inputs))
    audit = helper._read(helper._bind(old / "independent-audit.json", _AUDIT, inputs))
    decision = helper._read(helper._bind(old / "decision.json", _DECISION, inputs))
    _require(
        summary["complete"] is True
        and audit["audit_passed"] is True
        and audit["complete"] is True
        and decision["evidence_accepted"] is True
        and audit["summary_sha256"] == decision["summary_sha256"] == _SUMMARY
        and decision["audit_sha256"] == _AUDIT,
        "accepted mean-refit evidence",
    )
    helper._bind(old / "independent-audit.py", audit["script_sha256"], inputs)
    membership = helper._read(helper._bind(old / "train-membership.json", _MEMBERSHIP, inputs))
    training = helper._read(
        helper._bind(old / "training.json", summary["artifact_sha256"]["training.json"], inputs)
    )
    _require(training["initial_pre_update_loss"] == 0.003822767175734043, "initial loss comparator")
    recipe = _recipe(helper._read(recipe_path), mean, context["recipe"])
    helper._bind(recipe_path, helper._sha(recipe_path), inputs)
    context.update(
        recipe=recipe,
        recipe_path=recipe_path,
        membership=membership,
        parent_state=training["state_before"],
    )
    return context


def _attach(model):
    import torch

    _require(not hasattr(model, _PARAMETER), "residual already attached")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    first = next(model.parameters())
    _require(first.dtype == torch.float32, "FP32 base model required")
    dimension = model.config.dim
    model.register_parameter(
        _PARAMETER,
        torch.nn.Parameter(
            torch.zeros((2, dimension, dimension), dtype=torch.float32, device=first.device)
        ),
    )
    return model


def _factory(config, build_model, device="cpu"):
    return _attach(build_model(config).float().to(device).eval())


def _residual(parameter, entities, subjects, dimension_tokens, scale):
    import torch
    from torch.nn import functional as functional

    _require(parameter.shape == (2, entities.shape[1], entities.shape[1]), "residual shape")
    _require(
        dimension_tokens.ndim == subjects.ndim == 1 and dimension_tokens.shape == subjects.shape,
        "query dimension shape",
    )
    _require(
        subjects.dtype == dimension_tokens.dtype == torch.long,
        "integer subject/dimension indices required",
    )
    _require(bool(((dimension_tokens == 32) | (dimension_tokens == 33)).all()), "unknown dimension")
    _require(bool(((subjects >= 0) & (subjects < len(entities))).all()), "subject index")
    _require(parameter.dtype == entities.dtype == torch.float32, "FP32 residual inputs")
    with torch.autocast(device_type=parameter.device.type, enabled=False):
        symmetric = (parameter + parameter.transpose(-1, -2)) * 0.5
        transformed = torch.stack(
            (functional.linear(entities, symmetric[0]), functional.linear(entities, symmetric[1]))
        )
        selected = transformed[dimension_tokens - 32, subjects]
        return functional.linear(selected, entities) / scale


def _head(model, features, dimension_tokens, mean):
    entities, _, subjects, scale = features
    parent = mean._head(model.symmetric_relation_projection.weight, features)
    residual = _residual(getattr(model, _PARAMETER), entities, subjects, dimension_tokens, scale)
    return parent + residual


def _fit(model, features, dimension_tokens, labels, loss_fn, optimizer, updates, receipt, mean):
    import torch

    parameter = getattr(model, _PARAMETER)
    entities, _, subjects, scale = features
    with torch.no_grad():
        parent = mean._head(model.symmetric_relation_projection.weight, features).detach()
    receipt.update(history=[], completed_updates=0)
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = parent + _residual(parameter, entities, subjects, dimension_tokens, scale)
        loss = loss_fn(scores, labels, excluded_ids=subjects)
        receipt["failed_update"] = update
        receipt["failed_observation"] = {"pre_update_loss": float(loss.detach().cpu())}
        _require(
            bool(torch.isfinite(scores).all()) and bool(torch.isfinite(loss)),
            "nonfinite scores/loss",
        )
        loss.backward()
        _require(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
            "nonfinite residual gradient",
        )
        _require(
            all(p.grad is None for n, p in model.named_parameters() if n != _PARAMETER),
            "frozen gradient",
        )
        optimizer.step()
        _require(bool(torch.isfinite(parameter).all()), "nonfinite residual parameter")
        receipt["history"].append(
            {
                "update": update,
                "pre_update_loss": float(loss.detach().cpu()),
                "gradient_finite": True,
                "parameters_finite": True,
            }
        )
        receipt["completed_updates"] = update
        receipt.pop("failed_observation", None)
    with torch.no_grad():
        final = loss_fn(
            parent + _residual(parameter, entities, subjects, dimension_tokens, scale),
            labels,
            excluded_ids=subjects,
        )
    _require(bool(torch.isfinite(final)), "nonfinite final loss")
    receipt["initial_pre_update_loss"] = receipt["history"][0]["pre_update_loss"]
    receipt["final_post_update_loss"] = float(final.cpu())
    receipt.pop("failed_update", None)


def _tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _zero_replay(model, records, parent, mean, helper, path, batch_size=8):
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
                actual = _head(model, features, prompts[:, 2], mean)
                residual = _residual(
                    getattr(model, _PARAMETER), features[0], features[2], prompts[:, 2], features[3]
                )
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
            _require(parts[0] == parts[1], "zero-A actual head parent replay")
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


def _evaluate(model, records, wide_rows, mean, helper, path, parent=None, batch_size=8):
    import torch

    report = {
        "phase": "child",
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
                logits = _head(model, features, prompts[:, 2], mean)
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
        report["paired_vs_width8"] = mean._paired(
            rows, [r["composition"]["selected_set_ids"] for r in wide_rows]
        )
        if parent is not None:
            report["paired_vs_parent_dense"] = mean._paired(
                rows, [r["selected_set_ids"] for r in parent["responses"]]
            )
        report.update(complete=True, exact_parent_replay=False)
        return report
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        mean._write(path, helper._failure_safe(report))


def _validate_child(state, identity, config, metadata, corpus):
    raw = state.metadata
    _require(state.global_step == raw["global_step"] == 500, "child update count")
    _require(
        raw["experiment_identity"] == identity and identity["evaluator_version"] == _EVALUATOR,
        "child identity",
    )
    _require(raw["config"] == config and raw["training_metadata"] == metadata, "child metadata")
    _require(
        metadata["implementation_identity"] == config["implementation_identity"]
        and metadata["implementation_identity"]["runner_sha256"]
        == metadata["runner_sha256"]
        == metadata["scorer_sha256"]
        and metadata["implementation_identity"]["mean_helper_sha256"] == _MEAN,
        "child implementation identity",
    )
    _require(
        metadata["architecture"] == config["bilinear_residual_architecture"] == _architecture(),
        "child architecture",
    )
    _require(
        metadata["objective"] == _OBJECTIVE
        and metadata["evaluator"] == _EVALUATOR
        and metadata["trainable_parameters"] == [_PARAMETER]
        and metadata["residual_updates"] == 500
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


def _execute(root, out, context, mean, helper, summary):
    started = time.perf_counter()
    runtime_root = out.parent / "runtime"
    training = {
        "complete": False,
        "objective": _OBJECTIVE,
        "architecture": _architecture(),
        "completed_updates": 0,
        "history": [],
    }
    try:
        helper._modules()
        summary["runtime_files_sha256"] = helper._extract(context["members"], runtime_root)
        sys.path.insert(0, str(runtime_root / "src"))
        import torch

        from plm.config import OptimizerConfig, RootConfig
        from plm.experimentation.identity import ExperimentIdentity
        from plm.model.architecture import build_model
        from plm.model.layers import _prompt_set_loss
        from plm.protocol import load_protocol
        from plm.reproducibility import seed_everything
        from plm.serving.runtime import load_inference_runtime
        from plm.training.checkpoint import load_checkpoint, save_checkpoint
        from plm.training.data import collate_records
        from plm.training.optim import create_optimizer

        recipe = context["recipe"]
        seed_everything(1729, deterministic=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        summary["environment"] = helper._environment(torch)
        summary["numerical_settings"] = helper._settings(torch) | {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        summary["workspace_provenance"] = mean._git(root)
        config = RootConfig.model_validate(context["references"][1729]["offline"]["config"])
        runtime = load_inference_runtime(
            config,
            context["parent"],
            corpus=(root / config.data.corpus_manifest).parent,
            graph_db=root / config.data.graph_db,
            protocol=load_protocol(runtime_root / "configs/protocol/pokemon_v1.yaml"),
        )
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
            validation_update_points=[0, 500],
            train_query_order_sha256=mean._digest(queries["train"]),
            parent_training_identity=runtime.training_identity,
            inherited_architecture=model.config.model_dump(mode="json"),
            corpus_identity=runtime.corpus_identity,
            split_hash=mean._SPLIT,
        )
        parent = mean._evaluate(
            model,
            runtime.split.validation,
            context["wide"]["responses"],
            helper,
            out.parent / "parent.json",
            "parent",
        )
        _attach(model)
        _require(
            [n for n, p in model.named_parameters() if p.requires_grad] == [_PARAMETER],
            "A-only trainable mask",
        )
        parameter = getattr(model, _PARAMETER)
        _require(
            list(parameter.shape) == [2, 256, 256] and int(torch.count_nonzero(parameter)) == 0,
            "zero residual initialization",
        )
        initial = mean._state_hashes(model)
        _require(
            len(initial) == 94
            and {k: v for k, v in initial.items() if k != _PARAMETER} == parent_state,
            "94-entry initial state",
        )
        training["state_before"] = initial
        zero = _zero_replay(
            model, runtime.split.validation, parent, mean, helper, out.parent / "zero-replay.json"
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
            scores = _head(model, features, prompts[:, 2], mean)
            initial_loss = float(_prompt_set_loss(scores, labels, excluded_ids=features[2]).cpu())
        _require(
            list(scores.shape) == [1637, 1025] and initial_loss == recipe["initial_training_loss"],
            "zero-A full train loss replay",
        )
        optimizer = create_optimizer(model, OptimizerConfig.model_validate(recipe["optimizer"]))
        _require(
            not optimizer.state and sum(len(g["params"]) for g in optimizer.param_groups) == 1,
            "fresh A-only optimizer",
        )
        training.update(
            train_query_count=1637,
            train_query_order_sha256=summary["train_query_order_sha256"],
            trainable_parameters=[_PARAMETER],
            initial_loss_replay=initial_loss,
            optimizer={
                "name": "adamw",
                "config": recipe["optimizer"],
                "groups": [
                    {k: v for k, v in group.items() if k != "params"}
                    | {"parameter_names": [_PARAMETER] if group["params"] else []}
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
        _fit(
            model, features, prompts[:, 2], labels, _prompt_set_loss, optimizer, 500, training, mean
        )
        torch.cuda.synchronize()
        training["refit_seconds"] = time.perf_counter() - fit_started
        _require(training["initial_pre_update_loss"] == initial_loss, "trace initial loss")
        after = mean._state_hashes(model)
        _require(mean._frozen(initial, after, weight=_PARAMETER), "residual did not change")
        _require(
            {k: v for k, v in after.items() if k != _PARAMETER} == parent_state,
            "frozen original state",
        )
        training.update(state_after=after, residual_changed=True, frozen_tensors_unchanged=True)
        implementation = {
            "runner_sha256": summary["script_sha256"],
            "mean_helper_sha256": _MEAN,
            "source_archive_sha256": helper._SOURCE,
            "configs_archive_sha256": helper._CONFIGS,
        }
        child_config = {
            "implementation_identity": implementation,
            "inherited_parent_config": context["parent_run"]["config"],
            "bilinear_residual_architecture": _architecture(),
            "bilinear_residual_recipe": recipe,
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
                "recipe_sha256": summary["recipe_sha256"],
                "mean_helper_sha256": _MEAN,
                "architecture": _ARCHITECTURE,
            },
            seeds=(1729,),
        )
        metadata = {
            "implementation_identity": implementation,
            "objective": _OBJECTIVE,
            "evaluator": _EVALUATOR,
            "architecture": _architecture(),
            "model_config": model.config.model_dump(mode="json"),
            "model_config_scope": "inherited archived base only; residual architecture is separate",
            "parent_checkpoint_sha256": mean._PARENT,
            "parent_training_steps": 2000,
            "residual_updates": 500,
            "trainable_parameters": [_PARAMETER],
            "train_query_order_sha256": summary["train_query_order_sha256"],
            "record_count": 1637,
            "recipe_sha256": summary["recipe_sha256"],
            "runner_sha256": summary["script_sha256"],
            "scorer_sha256": summary["script_sha256"],
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
            global_step=500,
            config=child_config,
            identity=identity,
            corpus_identity=runtime.corpus_identity,
            split_hash=mean._SPLIT,
            training_metadata=metadata,
        )
        reloaded = _factory(model.config, build_model, runtime.device)
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
            mean._same_state(optimizer.state_dict(), reload_optimizer.state_dict()),
            "optimizer reload",
        )
        training.update(
            reload_exact=True,
            optimizer_reload_exact=True,
            checkpoint=str(checkpoint),
            checkpoint_sha256=saved.checkpoint_hash,
            checkpoint_sidecar_sha256=helper._sha(checkpoint.with_suffix(".pt.json")),
            training_identity=identity.to_dict(),
            complete=True,
        )
        summary["child_training_identity"] = identity.to_dict()
        child = _evaluate(
            reloaded,
            runtime.split.validation,
            context["wide"]["responses"],
            mean,
            helper,
            out.parent / "child.json",
            parent=parent,
        )
        invariants = {
            "parent_replay_exact": parent["exact_parent_replay"],
            "zero_A_replay_exact": zero["complete"]
            and zero["logits_sha256"] == zero["parent_logits_sha256"],
            "initial_train_loss_exact": initial_loss == recipe["initial_training_loss"],
            "completed_500_updates": training["completed_updates"] == 500,
            "original_93_tensors_unchanged": training["frozen_tensors_unchanged"],
            "residual_changed": training["residual_changed"],
            "reload_exact": training["reload_exact"],
            "optimizer_reload_exact": training["optimizer_reload_exact"],
            "train_only_supervision": training["validation_labels_used"] is False,
        }
        summary.update(
            invariants=invariants,
            gate=mean._gate(child, invariants),
            parent=parent["aggregate"],
            child={
                "aggregate": child["aggregate"],
                "groups": child["groups"],
                "paired_vs_parent_dense": child["paired_vs_parent_dense"],
                "paired_vs_width8": child["paired_vs_width8"],
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
        summary.update(final_identity_check=True, complete=True)
    except BaseException as exc:
        summary["error"] = repr(exc)
        training["error"] = repr(exc)
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
                "train-membership.json",
                "run.json",
                "checkpoint-final.pt",
                "checkpoint-final.pt.json",
            }
        }
        summary["wall_seconds"] = time.perf_counter() - started
        mean._write(out, helper._failure_safe(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("docs/experiments/2026-09-25-bilinear-residual-plan.md")
    )
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/experiments/bilinear_residual_refit_v1.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/bilinear-residual-refit-v1/summary.json")
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    content = script.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    _require(
        "torch" not in sys.modules and Path.cwd().resolve() == root, "Torch-free root preflight"
    )
    mean, mean_path = _load_mean(root)
    helper, helper_path = mean._load_helper(root)
    helper._modules()
    out = args.out.resolve()
    mean._refuse(out)
    _require(not (out.parent / "zero-replay.json").exists(), "immutable zero replay")
    context = _preflight(
        root, args.plan.resolve(), args.recipe.resolve(), mean, mean_path, helper, helper_path
    )
    helper._bind(script, digest, context["inputs"])
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
                    "recipe_sha256": helper._sha(args.recipe),
                }
            )
        )
        return
    _require(
        args.test_receipt is not None and args.test_receipt_sha256,
        "pinned CPU test receipt required",
    )
    receipt_path = helper._bind(args.test_receipt, args.test_receipt_sha256, context["inputs"])
    receipt = helper._read(receipt_path)
    _require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["experiment_training_executed"] is False
        and receipt["tested_script_sha256"] == digest
        and receipt["tested_recipe_sha256"] == helper._sha(args.recipe),
        "CPU test receipt",
    )
    required = [
        root / "tests/unit/test_bilinear_residual.py",
        root / "tests/unit/test_projection_only_refit.py",
        root / "tests/fixtures/projection_refit_runtime_source.zip",
        mean_path,
        root / "configs/experiments/projection_only_refit_v1.json",
    ]
    _require(
        set(receipt["test_dependencies"]) == {str(p.resolve()) for p in required},
        "test dependency inventory",
    )
    for path, pinned in receipt["test_dependencies"].items():
        helper._bind(path, pinned, context["inputs"])
    _require(
        receipt["test_dependencies"][str(mean_path)] == _MEAN
        and receipt["test_dependencies"][
            str(root / "tests/fixtures/projection_refit_runtime_source.zip")
        ]
        == helper._SOURCE,
        "portable helper/fixture identity",
    )
    stdout = helper._bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"], context["inputs"])
    helper._unchanged(context["inputs"])
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": content,
        "mean-helper.py": mean_path.read_bytes(),
        "helper.py": helper_path.read_bytes(),
        "plan.md": args.plan.read_bytes(),
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
        "campaign_version": "bilinear-residual-refit-v1",
        "complete": False,
        "acceptance": False,
        "architecture": _architecture(),
        "objective": _OBJECTIVE,
        "evaluator": _EVALUATOR,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "mean_helper_sha256": _MEAN,
        "helper_sha256": mean._HELPER,
        "recipe_sha256": helper._sha(args.recipe),
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "protected_test_used": False,
        "standard_serving_supported": False,
        "limitations": [
            "Single-seed adaptive validation screen, no protected evaluation.",
            (
                "Architecture and optimization geometry change together; "
                "no isolated causal capacity claim."
            ),
            "Mathematical symmetry does not promise swapped FP32 bitwise equality.",
            "Inherited decoder forward is not the new membership scorer.",
            "Dense head evaluation does not establish autoregressive generation quality.",
            "No serving throughput/energy or durable remote weight archival claim.",
        ],
    }
    _execute(root, out, context, mean, helper, summary)


if __name__ == "__main__":
    main()
