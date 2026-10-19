"""Frozen-parent, full-batch projection-only BCE screen; no serving contract migration.

The frozen recipe governs every real update. Synthetic tests may exercise private
arithmetic with smaller tensors. Preflight imports only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

_PLAN = "2ac4d157ad8c079561d642199fec8de2c3da1866819baa48f52080e9b0f39bfe"
_HELPER = "2ab48674b526ec6641a6c2366b669ac76bdeac1e8633331e01c772d4393edd39"
_SUMMARY = "1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183"
_AUDIT = "eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95"
_DECISION = "70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601"
_PARENT = "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
_CONFIG = "6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148"
_SPLIT = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
_OBJECTIVE = "plm-projection-only-balanced-bce-v1"
_EVALUATOR = "plm-projection-refit-screen-v1"
_WEIGHT = "symmetric_relation_projection.weight"
_GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_bytes(value))


def _load_helper(root):
    path = root / "runs/learning/wide-first-choice-v1/summary.script.py"
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == _HELPER, "helper identity")
    spec = importlib.util.spec_from_file_location("projection_refit_width8_helper", path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper, path


def _recipe(value):
    expected = {
        "schema_version": 1,
        "experiment": "projection-only-refit-v1",
        "objective": _OBJECTIVE,
        "evaluator": _EVALUATOR,
        "seed": 1729,
        "parent_training_steps": 2000,
        "projection_updates": 500,
        "train_query_count": 1637,
        "validation_query_count": 222,
        "trainable_parameter": _WEIGHT,
        "projection_shape": [256, 256],
        "product_count": 1025,
        "entity_base": 1024,
        "loss_coefficient": 1.0,
        "optimizer": {
            "name": "adamw",
            "lr": 0.0003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "decay_excludes_norm_and_bias": True,
        },
        "scheduler": None,
        "gradient_clipping": None,
        "gradient_accumulation": 1,
        "batching": "full-training-partition-in-split-order",
        "validation_batch_size": 8,
        "parameter_dtype": "float32",
        "autocast": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "float32_matmul_precision": "highest",
        "deterministic_algorithms": False,
        "selection": "ascending-non-subject-product-ids-with-logit-strictly-positive",
        "threshold": 0.0,
        "serialization_min_products": 1,
        "serialization_max_products": 506,
        "strong_comparator_exact": 201,
        "strong_comparator_f1": 0.9799255176742276,
        "strong_comparator_group_exact": {"COLOR": 103, "TYPE_single": 50, "TYPE_dual": 48},
    }
    _require(_bytes(value) == _bytes(expected), "fixed recipe differs")
    return value


def _refuse(out):
    _require(
        not out.exists()
        and not list(out.parent.glob(out.stem + ".*"))
        and not any(
            (out.parent / name).exists()
            for name in (
                "parent.json",
                "child.json",
                "training.json",
                "train-membership.json",
                "run.json",
                "runtime",
                "checkpoint-final.pt",
                "checkpoint-final.pt.json",
            )
        ),
        "immutable output exists",
    )


def _preflight(root, plan, recipe_path, helper, helper_path):
    context = helper._preflight(
        root, root / "docs/experiments/2026-09-25-wide-first-choice-plan.md"
    )
    inputs = context["inputs"]
    helper._bind(plan, _PLAN, inputs)
    helper._bind(helper_path, _HELPER, inputs)
    recipe = _recipe(helper._read(recipe_path))
    helper._bind(recipe_path, helper._sha(recipe_path), inputs)
    directory = root / "runs/learning/wide-first-choice-v1"
    summary = helper._read(helper._bind(directory / "summary.json", _SUMMARY, inputs))
    audit = helper._read(helper._bind(directory / "independent-audit.json", _AUDIT, inputs))
    decision = helper._read(helper._bind(directory / "decision.json", _DECISION, inputs))
    _require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == decision["summary_sha256"] == _SUMMARY
        and decision["evidence_accepted"] is True
        and decision["fixed_quality_gate_passed"] is True
        and decision["audit_sha256"] == _AUDIT
        and decision["auditor_sha256"] == audit["script_sha256"],
        "accepted width8 evidence",
    )
    helper._bind(directory / "independent-audit.py", audit["script_sha256"], inputs)
    refs = [r for r in summary["reports"] if r["seed"] == 1729]
    _require(len(refs) == 1, "parent seed reference")
    wide = helper._read(helper._bind(refs[0]["path"], refs[0]["sha256"], inputs))
    _require(wide["complete"] is True and wide["query_count"] == 222, "wide coverage")
    rows = wide["responses"]
    _require(len(rows) == 222 and wide["product_token_ids"] == list(range(1024, 2049)), "columns")
    for index, row in enumerate(rows):
        helper._fp32(row["symmetric_relation_logits"])
        _require(row["index"] == index and row["group"] in _GROUPS, "wide row order/group")
    _require(
        sum(r["metrics"]["wide"]["exact"] for r in rows) == 201
        and math.fsum(r["metrics"]["wide"]["f1"] for r in rows) / 222
        == recipe["strong_comparator_f1"]
        and {
            g: sum(r["metrics"]["wide"]["exact"] for r in rows if r["group"] == g) for g in _GROUPS
        }
        == recipe["strong_comparator_group_exact"],
        "strong comparator identity",
    )
    parent_dir = root / "runs/national_dex_continuation_control_s1729_v1"
    parent_path = helper._bind(parent_dir / "checkpoint-final.pt", _PARENT, inputs)
    for name in ("run.json", "training-result.json", "checkpoint-final.pt.json"):
        _require(
            str((parent_dir / name).resolve()) in inputs, "missing inherited training identity"
        )
    run = helper._read(parent_dir / "run.json")
    sidecar = helper._read(parent_dir / "checkpoint-final.pt.json")
    result = helper._read(parent_dir / "training-result.json")
    _require(
        run["identity"] == sidecar["experiment_identity"] == result["identity"]
        and run["identity"]["resolved_config_hash"] == _CONFIG
        and sidecar["checkpoint_hash"] == result["checkpoint_hash"] == _PARENT
        and sidecar["global_step"] == result["global_step"] == 2000
        and sidecar["split_hash"] == _SPLIT
        and sidecar["training_metadata"]["objective"] == "causal-next-token-v1",
        "parent training contract",
    )
    context.update(
        recipe=recipe,
        recipe_path=recipe_path,
        wide=wide,
        parent=parent_path,
        parent_run=run,
        parent_sidecar=sidecar,
    )
    return context


def _metrics(selected, truth):
    predicted, expected = set(selected), set(truth)
    _require(len(predicted) == len(selected) and bool(expected), "set/teacher contract")
    hits = len(predicted & expected)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(expected)
    return {
        "exact": predicted == expected,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "set_size": len(predicted),
        "false_positive_count": len(predicted - expected),
        "false_negative_count": len(expected - predicted),
        "serialization_compatible": 1 <= len(predicted) <= 506,
    }


def _prediction(logits, subject, base=1024):
    _require(all(type(z) in (int, float) and math.isfinite(z) for z in logits), "nonfinite head")
    _require(type(subject) is int and base <= subject < base + len(logits), "subject column")
    return [base + c for c, z in enumerate(logits) if z > 0 and base + c != subject]


def _totals(rows):
    _require(bool(rows), "empty aggregate")
    return {
        "query_count": len(rows),
        "exact_count": sum(r["metrics"]["exact"] for r in rows),
        **{
            k: math.fsum(r["metrics"][k] for r in rows) / len(rows)
            for k in ("precision", "recall", "f1", "set_size")
        },
        **{
            k: sum(r["metrics"][k] for r in rows)
            for k in ("false_positive_count", "false_negative_count", "serialization_compatible")
        },
        "strict_separation_count": sum(r["strict_separation"] for r in rows),
    }


def _paired(rows, reference_ids):
    _require(len(rows) == len(reference_ids), "paired coverage")
    values = [
        (r["metrics"]["exact"], set(ids) == set(r["expected_set_ids"]))
        for r, ids in zip(rows, reference_ids, strict=True)
    ]
    result = {
        "gains": sum(a and not b for a, b in values),
        "losses": sum(b and not a for a, b in values),
    }
    result["groups"] = {
        g: {
            "gains": sum(
                a and not b for (a, b), r in zip(values, rows, strict=True) if r["group"] == g
            ),
            "losses": sum(
                b and not a for (a, b), r in zip(values, rows, strict=True) if r["group"] == g
            ),
        }
        for g in _GROUPS
    }
    return result


def _gate(child, invariants):
    aggregate, groups = child["aggregate"], child["groups"]
    checks = {
        "execution_invariants": all(v is True for v in invariants.values()),
        "serialization_compatible": aggregate["serialization_compatible"] == 222,
        "exact_above_201": aggregate["exact_count"] > 201,
        "f1_at_least_strong_comparator": aggregate["f1"] >= 0.9799255176742276,
        "group_exact_nonregression": all(
            groups[g]["exact_count"] >= n
            for g, n in {"COLOR": 103, "TYPE_single": 50, "TYPE_dual": 48}.items()
        ),
    }
    return {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "independent_audit_required": True,
        "accepted": False,
    }


def _state_hashes(model):
    return {
        name: {
            "sha256": hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }
        for name, t in sorted(model.state_dict().items())
    }


def _frozen(before, after, weight=_WEIGHT):
    _require(set(before) == set(after) and weight in before, "state inventory mismatch")
    _require(
        all(before[k] == after[k] for k in before if k != weight), "frozen state tensor changed"
    )
    return before[weight] != after[weight]


def _features(model, prompts):
    import torch
    from torch.nn import functional as functional

    _require(prompts.ndim == 2 and prompts.shape[1] == 5, "five-token head prompts")
    subjects = prompts[:, 1] - 1024
    dimensions = prompts[:, 2]
    weight = model.token_embedding.weight
    _require(bool(((subjects >= 0) & (subjects < weight.shape[0] - 1024)).all()), "product subject")
    _require(bool(((dimensions >= 32) & (dimensions < 128)).all()), "steering dimension")
    with torch.no_grad(), torch.autocast(device_type=prompts.device.type, enabled=False):
        scale = model.config.dim**0.5
        entities = functional.normalize(weight[1024:].float(), dim=-1) * scale
        steering = functional.normalize(weight[dimensions].float(), dim=-1) * scale
    return entities.detach(), steering.detach(), subjects, scale


def _head(weight, features):
    import torch
    from torch.nn import functional as functional

    entities, steering, subjects, scale = features
    with torch.autocast(device_type=weight.device.type, enabled=False):
        relation = functional.linear(steering, weight.float())
        return functional.linear(entities[subjects] * relation, entities) / scale


def _fit(model, features, labels, loss_fn, optimizer, updates, receipt):
    import torch

    weight = dict(model.named_parameters())[_WEIGHT]
    receipt["history"] = []
    receipt["completed_updates"] = 0
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(_head(weight, features), labels, excluded_ids=features[2])
        receipt["failed_update"] = update
        receipt["failed_observation"] = {
            "update": update,
            "pre_update_loss": float(loss.detach().cpu()),
        }
        _require(bool(torch.isfinite(loss)), "nonfinite training loss")
        loss.backward()
        _require(
            weight.grad is not None and bool(torch.isfinite(weight.grad).all()),
            "nonfinite gradient",
        )
        _require(
            all(p.grad is None for n, p in model.named_parameters() if n != _WEIGHT),
            "frozen gradient",
        )
        optimizer.step()
        _require(bool(torch.isfinite(weight).all()), "nonfinite parameter")
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
        final = loss_fn(_head(weight, features), labels, excluded_ids=features[2])
    _require(bool(torch.isfinite(final)), "nonfinite final loss")
    receipt["final_post_update_loss"] = float(final.cpu())
    receipt["initial_pre_update_loss"] = receipt["history"][0]["pre_update_loss"]
    receipt.pop("failed_update", None)


def _query(record):
    return {
        "subject": record.subject,
        "dimension": record.dimension,
        "prompt_ids": list(record.input_ids[:5]),
    }


def _split_contract(train, validation, expected_train=1637, expected_validation=222):
    train_keys = [(r.subject, r.dimension) for r in train]
    validation_keys = [(r.subject, r.dimension) for r in validation]
    _require(len(train_keys) == len(set(train_keys)) == expected_train, "train query coverage")
    _require(
        len(validation_keys) == len(set(validation_keys)) == expected_validation,
        "validation coverage",
    )
    _require(not set(train_keys) & set(validation_keys), "training/validation query leakage")
    return {"train": [_query(r) for r in train], "validation": [_query(r) for r in validation]}


def _training_membership(records):
    rows = []
    for index, record in enumerate(records):
        ids, labels = list(record.input_ids), list(record.labels)
        _require(
            len(ids) == len(labels)
            and labels[:5] == [-100] * 5
            and labels[5:] == ids[5:]
            and ids[-1] == 2,
            "train teacher alignment",
        )
        expected = sorted({i for i in labels if 1024 <= i < 2049})
        _require(expected and ids[1] not in expected, "train SAME membership")
        rows.append({"index": index, **_query(record), "expected_set_ids": expected})
    return {"partition": "train", "query_count": len(rows), "queries": rows}


def _same_state(left, right):
    """Exact nested optimizer-state comparison, including empty groups and flags."""
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _evaluate(model, records, wide_rows, helper, path, phase, parent=None, batch_size=8):
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
            features = _features(model, prompts)
            with torch.no_grad():
                logits = _head(model.symmetric_relation_projection.weight, features)
            if prompts.is_cuda:
                torch.cuda.synchronize()
            report["head_seconds"] += time.perf_counter() - started
            values = logits.detach().cpu().tolist()
            report["failed_observation"] = {
                "offset": offset,
                "prompt_ids": prompts.cpu().tolist(),
                "standalone_logits": helper._failure_safe(values),
            }
            if phase == "parent":
                with torch.no_grad():
                    model_logits = model(prompts).symmetric_relation_logits
                report["failed_observation"]["archived_model_logits"] = helper._failure_safe(
                    model_logits.cpu().tolist()
                )
                _require(torch.equal(logits, model_logits), "standalone/archived head mismatch")
            for j, (record, vector) in enumerate(zip(batch, values, strict=True)):
                old = wide_rows[offset + j]
                _require(
                    _query(record) == {k: old[k] for k in ("subject", "dimension", "prompt_ids")},
                    "query identity",
                )
                truth = list(old["expected_set_ids"])
                _require(set(record.input_ids[5:-1]) == set(truth), "validation truth identity")
                helper._fp32(vector)
                if phase == "parent":
                    _require(
                        vector == old["symmetric_relation_logits"],
                        "historical parent head mismatch",
                    )
                selected = _prediction(vector, old["prompt_ids"][1])
                negatives = [
                    z
                    for c, z in enumerate(vector)
                    if c + 1024 not in set(truth) | {old["prompt_ids"][1]}
                ]
                row = {
                    "index": offset + j,
                    **_query(record),
                    "group": old["group"],
                    "expected_set_ids": truth,
                    "symmetric_relation_logits": vector,
                    "selected_set_ids": selected,
                    "metrics": _metrics(selected, truth),
                    "strict_separation": min(vector[i - 1024] for i in truth) > max(negatives),
                }
                report["responses"].append(row)
                report["query_count"] += 1
            report.pop("failed_observation", None)
        rows = report["responses"]
        report["aggregate"] = _totals(rows)
        report["groups"] = {g: _totals([r for r in rows if r["group"] == g]) for g in _GROUPS}
        report["paired_vs_width8"] = _paired(
            rows, [r["composition"]["selected_set_ids"] for r in wide_rows]
        )
        if parent is not None:
            report["paired_vs_parent_dense"] = _paired(
                rows, [r["selected_set_ids"] for r in parent["responses"]]
            )
        report["exact_parent_replay"] = phase == "parent"
        report["complete"] = True
        return report
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        _write(path, report)


def _validate_child(state, identity, config, metadata, corpus):
    raw = state.metadata
    _require(state.global_step == 500 and raw["global_step"] == 500, "child update count")
    _require(
        raw["experiment_identity"] == identity and identity["evaluator_version"] == _EVALUATOR,
        "child identity",
    )
    _require(
        raw["config"] == config and raw["training_metadata"] == metadata, "child recipe/objective"
    )
    _require(
        metadata["objective"] == _OBJECTIVE
        and metadata["parent_checkpoint_sha256"] == _PARENT
        and metadata["parent_training_steps"] == 2000
        and metadata["projection_updates"] == 500,
        "child lineage",
    )
    _require(
        raw["corpus_identity"] == corpus and raw["split_hash"] == _SPLIT, "child data identity"
    )


def _git(root):
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout

    status = run("status", "--porcelain=v1").decode()
    return {
        "commit": run("rev-parse", "HEAD").decode().strip(),
        "status_porcelain": status,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(run("diff", "HEAD", "--binary")).hexdigest(),
        "scope": "workspace provenance; executed modules are separately bound to archived runtime",
    }


def _execute(root, out, context, helper, summary):
    started = time.perf_counter()
    runtime_root = out.parent / "runtime"
    training = {"complete": False, "objective": _OBJECTIVE, "completed_updates": 0, "history": []}
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
        seed_everything(recipe["seed"], deterministic=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        summary["environment"] = helper._environment(torch)
        summary["numerical_settings"] = helper._settings(torch)
        summary["numerical_settings"].update(
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
        )
        summary["workspace_provenance"] = _git(root)
        config = RootConfig.model_validate(context["references"][1729]["offline"]["config"])
        runtime = load_inference_runtime(
            config,
            context["parent"],
            corpus=(root / config.data.corpus_manifest).parent,
            graph_db=root / config.data.graph_db,
            protocol=load_protocol(runtime_root / "configs/protocol/pokemon_v1.yaml"),
        )
        _require(
            runtime.checkpoint_hash == _PARENT
            and runtime.split.split_hash == _SPLIT
            and runtime.training_identity == context["parent_run"]["identity"],
            "loaded parent identity",
        )
        helper._modules(runtime_root)
        model = runtime.model.float().eval()
        _require(
            all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
            "finite parent state",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name == _WEIGHT)
        _require(
            [n for n, p in model.named_parameters() if p.requires_grad] == [_WEIGHT],
            "W-only trainable mask",
        )
        _require(
            list(model.symmetric_relation_projection.weight.shape) == [256, 256], "projection shape"
        )
        queries = _split_contract(runtime.split.train, runtime.split.validation)
        summary["queries"] = queries
        summary["validation_update_points"] = [0, 500]
        summary["train_query_order_sha256"] = _digest(queries["train"])
        summary["validation_query_order_sha256"] = _digest(queries["validation"])
        summary["parent_training_identity"] = runtime.training_identity
        summary["inherited_architecture"] = model.config.model_dump(mode="json")
        summary["corpus_identity"] = runtime.corpus_identity
        summary["split_hash"] = _SPLIT
        before = _state_hashes(model)
        training["state_before"] = before
        _write(out.parent / "train-membership.json", _training_membership(runtime.split.train))
        torch.cuda.reset_peak_memory_stats()
        parent = _evaluate(
            model,
            runtime.split.validation,
            context["wide"]["responses"],
            helper,
            out.parent / "parent.json",
            "parent",
        )
        helper._unchanged(context["inputs"])
        batch = collate_records(runtime.split.train, pad_id=runtime.vocabulary.pad_id)
        prompts = batch.input_ids[:, :5].to(runtime.device)
        labels = batch.labels.masked_fill(~batch.attention_mask, -100)[:, 1:].to(runtime.device)
        features = _features(model, prompts)
        _require(
            list(_head(model.symmetric_relation_projection.weight, features).shape) == [1637, 1025],
            "full train head shape",
        )
        optimizer = create_optimizer(model, OptimizerConfig.model_validate(recipe["optimizer"]))
        _require(
            not optimizer.state and sum(len(g["params"]) for g in optimizer.param_groups) == 1,
            "fresh W optimizer",
        )
        training.update(
            train_query_count=1637,
            train_query_order_sha256=summary["train_query_order_sha256"],
            trainable_parameters=[_WEIGHT],
            optimizer=recipe["optimizer"],
            optimizer_implementation={
                "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                "defaults": optimizer.defaults,
                "groups": [
                    {k: v for k, v in group.items() if k != "params"}
                    | {"parameter_names": [_WEIGHT] if group["params"] else []}
                    for group in optimizer.param_groups
                ],
            },
            transformer_forwards_during_fit=0,
            validation_labels_used=False,
        )
        torch.cuda.synchronize()
        fit_started = time.perf_counter()
        _fit(model, features, labels, _prompt_set_loss, optimizer, 500, training)
        torch.cuda.synchronize()
        training["refit_seconds"] = time.perf_counter() - fit_started
        after = _state_hashes(model)
        training["state_after"] = after
        training["projection_changed"] = _frozen(before, after)
        _require(training["projection_changed"], "projection did not change")
        training["frozen_tensors_unchanged"] = True
        child_config = {
            "inherited_parent_config": context["parent_run"]["config"],
            "projection_refit_recipe": recipe,
        }
        identity = ExperimentIdentity(
            source_commit=summary["workspace_provenance"]["commit"],
            resolved_config_hash=_digest(child_config),
            protocol_version=runtime.training_identity["protocol_version"],
            **runtime.corpus_identity,
            split_hash=_SPLIT,
            evaluator_version=_EVALUATOR,
            environment={
                **summary["environment"],
                "source_archive_sha256": helper._SOURCE,
                "configs_archive_sha256": helper._CONFIGS,
                "runner_sha256": summary["script_sha256"],
                "recipe_sha256": summary["recipe_sha256"],
            },
            seeds=(1729,),
        )
        metadata = {
            "objective": _OBJECTIVE,
            "model_config": model.config.model_dump(mode="json"),
            "parent_checkpoint_sha256": _PARENT,
            "parent_training_steps": 2000,
            "projection_updates": 500,
            "trainable_parameters": [_WEIGHT],
            "train_query_order_sha256": summary["train_query_order_sha256"],
            "record_count": 1637,
            "recipe_sha256": summary["recipe_sha256"],
            "frozen_state_sha256": _digest({k: v for k, v in before.items() if k != _WEIGHT}),
            "final_post_update_loss": training["final_post_update_loss"],
        }
        _write(
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
            split_hash=_SPLIT,
            training_metadata=metadata,
        )
        reloaded = build_model(model.config).float().to(runtime.device).eval()
        for name, parameter in reloaded.named_parameters():
            parameter.requires_grad_(name == _WEIGHT)
        reload_optimizer = create_optimizer(
            reloaded, OptimizerConfig.model_validate(recipe["optimizer"])
        )
        loaded = load_checkpoint(
            checkpoint, reloaded, optimizer=reload_optimizer, map_location="cpu", restore_rng=False
        )
        _validate_child(loaded, identity.to_dict(), child_config, metadata, runtime.corpus_identity)
        training["state_reloaded"] = _state_hashes(reloaded)
        _require(training["state_reloaded"] == after, "child reload tensor mismatch")
        _require(loaded.checkpoint_hash == saved.checkpoint_hash, "checkpoint reload hash")
        _require(
            _same_state(optimizer.state_dict(), reload_optimizer.state_dict()),
            "optimizer reload mismatch",
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
            helper,
            out.parent / "child.json",
            "child",
            parent,
        )
        invariants = {
            "parent_replay_exact": parent["exact_parent_replay"],
            "train_only": True,
            "complete_500_updates": training["completed_updates"] == 500,
            "frozen_tensors_unchanged": training["frozen_tensors_unchanged"],
            "child_reload_exact": training["reload_exact"],
        }
        summary["gate"] = _gate(child, invariants)
        summary["peak_memory"] = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        summary["module_origins"] = helper._modules(runtime_root)
        helper._runtime_unchanged(runtime_root, summary["runtime_files_sha256"])
        helper._unchanged(context["inputs"])
        helper._unchanged(summary["snapshot_sha256"])
        _require(helper._environment(torch) == summary["environment"], "environment changed")
        summary["final_identity_check"] = True
        summary["complete"] = True
    except BaseException as exc:
        summary["error"] = repr(exc)
        training["error"] = repr(exc)
        raise
    finally:
        _write(out.parent / "training.json", helper._failure_safe(training))
        summary["artifact_sha256"] = {
            p.name: helper._sha(p)
            for p in out.parent.iterdir()
            if p.name
            in {
                "parent.json",
                "child.json",
                "training.json",
                "train-membership.json",
                "run.json",
                "checkpoint-final.pt",
                "checkpoint-final.pt.json",
            }
        }
        summary["wall_seconds"] = time.perf_counter() - started
        _write(out, helper._failure_safe(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/experiments/2026-09-25-projection-only-refit-plan.md"),
    )
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/experiments/projection_only_refit_v1.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/projection-only-refit-v1/summary.json")
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    content = script.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    _require("torch" not in sys.modules, "preflight must be Torch-free")
    _require(Path.cwd().resolve() == root, "run from repository root")
    helper, helper_path = _load_helper(root)
    helper._modules()
    out = args.out.resolve()
    _refuse(out)
    context = _preflight(root, args.plan.resolve(), args.recipe.resolve(), helper, helper_path)
    helper._bind(script, digest, context["inputs"])
    helper._unchanged(context["inputs"])
    if args.preflight_only:
        _require("torch" not in sys.modules, "preflight imported Torch")
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "torch_imported": False,
                    "training_executed": False,
                    "input_count": len(context["inputs"]),
                    "script_sha256": digest,
                    "recipe_sha256": helper._sha(args.recipe),
                }
            )
        )
        return
    _require(
        args.test_receipt is not None and args.test_receipt_sha256, "pinned test receipt required"
    )
    receipt_path = helper._bind(args.test_receipt, args.test_receipt_sha256, context["inputs"])
    receipt = helper._read(receipt_path)
    _require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["experiment_training_executed"] is False
        and receipt["tested_script_sha256"] == digest
        and receipt["tested_recipe_sha256"] == helper._sha(args.recipe),
        "test receipt contract",
    )
    stdout = helper._bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"], context["inputs"])
    helper._bind(receipt["test_source"], receipt["tested_test_sha256"], context["inputs"])
    fixture = root / "tests/fixtures/projection_refit_runtime_source.zip"
    _require(
        receipt["fixtures"] == {str(fixture.resolve()): helper._SOURCE},
        "portable test fixture identity",
    )
    helper._bind(fixture, helper._SOURCE, context["inputs"])
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": content,
        "helper.py": helper_path.read_bytes(),
        "plan.md": args.plan.read_bytes(),
        "recipe.json": args.recipe.read_bytes(),
        "source.zip": context["source"].read_bytes(),
        "configs.zip": context["configs"].read_bytes(),
        "test-receipt.json": receipt_path.read_bytes(),
        "test-stdout.txt": stdout.read_bytes(),
        "inputs.json": _bytes(context["inputs"]),
    }
    hashes = {}
    for suffix, data in snapshots.items():
        path = out.with_suffix("." + suffix)
        with path.open("xb") as stream:
            stream.write(data)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
    summary = {
        "campaign_version": "projection-only-refit-v1",
        "complete": False,
        "acceptance": False,
        "objective": _OBJECTIVE,
        "evaluator": _EVALUATOR,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "helper_sha256": _HELPER,
        "recipe_sha256": helper._sha(args.recipe),
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "protected_test_used": False,
        "standard_serving_supported": False,
        "limitations": [
            "Single seed; fixed final checkpoint without validation selection.",
            "Saved-evidence audit does not repeat neural training.",
            "Dense membership sets do not prove sequence or serving equivalence.",
            "Frozen decoder does not imply unchanged head-guided generation.",
            "Timing is descriptive; no throughput, energy or concurrency claim.",
            "Local checkpoint bytes are not evidence of durable remote weight archival.",
        ],
    }
    _execute(root, out, context, helper, summary)


if __name__ == "__main__":
    main()
