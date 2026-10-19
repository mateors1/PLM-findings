"""Sibling W-only refit with sole worst-member softplus; frozen helper callbacks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

_PLAN = "03bda47032fa65c11bf70904cb05f845fe10a89154e36a5edb5b76b0f5d7be9c"
_MEAN_HELPER = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
_MEAN_SUMMARY = "0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61"
_MEAN_AUDIT = "6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc"
_MEAN_DECISION = "da0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6"
_MEAN_CHECKPOINT = "84a3ba02ab5f43e9073f1f8b8c7afc85ad7a5363e8cb180e1424a77e6a44ab87"
_PARENT = "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
_SPLIT = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
_OBJECTIVE = "plm-projection-worst-boundary-softplus-v1"
_EVALUATOR = "plm-projection-worst-boundary-screen-v1"
_WEIGHT = "symmetric_relation_projection.weight"
_GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _load_mean(root):
    path = root / "scripts/refit_membership_projection.py"
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == _MEAN_HELPER, "mean helper identity")
    spec = importlib.util.spec_from_file_location("authenticated_mean_projection_helper", path)
    _require(spec is not None and spec.loader is not None, "mean helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _recipe(recipe, mean, inherited):
    expected = json.loads(json.dumps(mean._recipe(inherited)))
    expected.update(
        experiment="projection-worst-boundary-v1",
        objective=_OBJECTIVE,
        evaluator=_EVALUATOR,
        loss_definition="query-mean-half-softplus-negative-amin-true-plus-softplus-amax-negative-fp32-v1",
        mean_bce_diagnostic={
            "initial_expected": 0.003822767175734043,
            "update_points": [0, 500],
            "used_for_updates": False,
        },
    )
    _require(mean._bytes(recipe) == mean._bytes(expected), "fixed sibling recipe differs")
    return recipe


def _worst_loss(set_logits, labels, *, excluded_ids):
    """Archived BCE mask semantics, replacing class means only with amin/amax.

    amin/amax distribute subgradients across ties; no indexed extremum choice.
    """
    import torch
    from torch.nn import functional as functional

    with torch.autocast(device_type=set_logits.device.type, enabled=False):
        product_labels = (labels >= 1024) & (labels < 1024 + set_logits.shape[-1])
        targets = torch.zeros_like(set_logits, dtype=torch.float32)
        indices = (labels - 1024).clamp(0, set_logits.shape[-1] - 1)
        targets.scatter_add_(1, indices, product_labels.float())
        targets = (targets > 0).float()
        negatives = 1 - targets
        if targets.gather(1, excluded_ids[:, None]).any():
            raise ValueError("symmetric labels must exclude the SAME subject")
        negatives = negatives.scatter(1, excluded_ids[:, None], 0)
        if (targets.sum(-1) == 0).any() or (negatives.sum(-1) == 0).any():
            raise ValueError("worst-member loss needs positive and negative products per query")
        scores = set_logits.float()
        weakest = scores.masked_fill(~targets.bool(), float("inf")).amin(dim=-1)
        strongest = scores.masked_fill(~negatives.bool(), -float("inf")).amax(dim=-1)
        return ((functional.softplus(-weakest) + functional.softplus(strongest)) * 0.5).mean()


def _mean_diagnostic(model, features, labels, mean_loss, head_fn):
    import torch

    # Explicit frozen helper callback; diagnostics have no autograd graph.
    with torch.no_grad(), torch.autocast(device_type=labels.device.type, enabled=False):
        scores = head_fn(model.symmetric_relation_projection.weight, features)
        value = mean_loss(scores, labels, excluded_ids=features[2])
    _require(bool(torch.isfinite(value)), "nonfinite diagnostic mean BCE")
    return float(value.cpu())


def _preflight(root, plan, recipe_path, mean, mean_path, helper, helper_path):
    directory = root / "runs/learning/projection-only-refit-v1"
    context = mean._preflight(
        root, directory / "summary.plan.md", directory / "summary.recipe.json", helper, helper_path
    )
    inputs = context["inputs"]
    helper._bind(plan, _PLAN, inputs)
    helper._bind(mean_path, _MEAN_HELPER, inputs)
    helper._bind(directory / "summary.script.py", _MEAN_HELPER, inputs)
    summary = helper._read(helper._bind(directory / "summary.json", _MEAN_SUMMARY, inputs))
    audit = helper._read(helper._bind(directory / "independent-audit.json", _MEAN_AUDIT, inputs))
    decision = helper._read(helper._bind(directory / "decision.json", _MEAN_DECISION, inputs))
    _require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and decision["evidence_accepted"] is True
        and decision["fixed_quality_gate_passed"] is False
        and audit["summary_sha256"] == decision["summary_sha256"] == _MEAN_SUMMARY
        and decision["audit_sha256"] == _MEAN_AUDIT
        and decision["auditor_sha256"] == audit["script_sha256"],
        "accepted mean-refit chain",
    )
    helper._bind(directory / "independent-audit.py", audit["script_sha256"], inputs)
    helper._bind(directory / "checkpoint-final.pt", _MEAN_CHECKPOINT, inputs)
    artifacts = {}
    for name in (
        "training.json",
        "parent.json",
        "child.json",
        "run.json",
        "checkpoint-final.pt.json",
    ):
        artifacts[name] = helper._read(
            helper._bind(directory / name, summary["artifact_sha256"][name], inputs)
        )
    _require(
        summary["artifact_sha256"]["child.json"]
        == "b44463dd43df234aaba5f95a8910eb5afea1bee012782f2a9d19896adb221f1f"
        and artifacts["training.json"]["checkpoint_sha256"] == _MEAN_CHECKPOINT
        and artifacts["training.json"]["objective"] == mean._OBJECTIVE
        and artifacts["training.json"]["completed_updates"] == 500
        and artifacts["training.json"]["complete"] is True,
        "historical mean identity/trace",
    )
    for report in (artifacts["parent.json"], artifacts["child.json"]):
        _require(
            report["complete"] is True and report["query_count"] == 222, "historical dense coverage"
        )
        for old, wide in zip(report["responses"], context["wide"]["responses"], strict=True):
            _require(
                all(
                    old[k] == wide[k]
                    for k in (
                        "index",
                        "subject",
                        "dimension",
                        "group",
                        "prompt_ids",
                        "expected_set_ids",
                    )
                ),
                "historical dense query alignment",
            )
    _require(
        artifacts["training.json"]["initial_pre_update_loss"] == 0.003822767175734043,
        "historical initial mean BCE",
    )
    context["recipe"] = _recipe(helper._read(recipe_path), mean, context["recipe"])
    helper._bind(recipe_path, helper._sha(recipe_path), inputs)
    context["mean_child"] = artifacts["child.json"]
    context["mean_comparator"] = {
        "summary_sha256": _MEAN_SUMMARY,
        "audit_sha256": _MEAN_AUDIT,
        "decision_sha256": _MEAN_DECISION,
        "checkpoint_sha256": _MEAN_CHECKPOINT,
        "artifact_sha256": summary["artifact_sha256"],
        "training_history": artifacts["training.json"]["history"],
        "initial_pre_update_loss": artifacts["training.json"]["initial_pre_update_loss"],
        "final_post_update_loss": artifacts["training.json"]["final_post_update_loss"],
        "objective": mean._OBJECTIVE,
        "historical_reuse": True,
        "aggregate": artifacts["child.json"]["aggregate"],
        "groups": artifacts["child.json"]["groups"],
    }
    return context


def _comparisons(child, parent, historical, wide_rows, mean):
    rows = child["responses"]
    aggregate = {}
    changes = []
    for name, reference in (("parent_dense", parent), ("mean_refit_dense", historical)):
        _require(len(reference["responses"]) == len(rows), "paired query coverage")
        selected = []
        separation = []
        for a, b in zip(rows, reference["responses"], strict=True):
            _require(
                all(
                    a[k] == b[k]
                    for k in (
                        "index",
                        "subject",
                        "dimension",
                        "group",
                        "prompt_ids",
                        "expected_set_ids",
                    )
                ),
                "paired query identity",
            )
            selected.append(b["selected_set_ids"])
            separation.append((a["strict_separation"], b["strict_separation"]))
            changes.append(
                {
                    "comparison": name,
                    "index": a["index"],
                    "subject": a["subject"],
                    "dimension": a["dimension"],
                    "group": a["group"],
                    "gained_exact": a["metrics"]["exact"] and not b["metrics"]["exact"],
                    "lost_exact": b["metrics"]["exact"] and not a["metrics"]["exact"],
                    "gained_separation": a["strict_separation"] and not b["strict_separation"],
                    "lost_separation": b["strict_separation"] and not a["strict_separation"],
                }
            )
        aggregate[name] = {
            "exact": mean._paired(rows, selected),
            "strict_separation": {
                "gains": sum(a and not b for a, b in separation),
                "losses": sum(b and not a for a, b in separation),
            },
        }
    aggregate["width8"] = {
        "exact": mean._paired(rows, [r["composition"]["selected_set_ids"] for r in wide_rows])
    }
    return {"aggregate": aggregate, "per_query": changes}


def _validate_child(state, identity, config, metadata, corpus):
    raw = state.metadata
    _require(state.global_step == raw["global_step"] == 500, "new child update count")
    _require(
        raw["experiment_identity"] == identity and identity["evaluator_version"] == _EVALUATOR,
        "new child identity",
    )
    _require(
        raw["config"] == config and raw["training_metadata"] == metadata,
        "new child recipe/objective",
    )
    _require(
        metadata["objective"] == _OBJECTIVE
        and metadata["parent_checkpoint_sha256"] == _PARENT
        and metadata["parent_training_steps"] == 2000
        and metadata["projection_updates"] == 500,
        "original-parent sibling lineage",
    )
    _require(
        raw["corpus_identity"] == corpus and raw["split_hash"] == _SPLIT, "child data identity"
    )


def _execute(root, out, context, mean, helper, summary):
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
        queries = mean._split_contract(runtime.split.train, runtime.split.validation)
        summary["queries"] = queries
        summary["validation_update_points"] = [0, 500]
        summary["train_query_order_sha256"] = mean._digest(queries["train"])
        summary["validation_query_order_sha256"] = mean._digest(queries["validation"])
        summary["parent_training_identity"] = runtime.training_identity
        summary["inherited_architecture"] = model.config.model_dump(mode="json")
        summary["corpus_identity"] = runtime.corpus_identity
        summary["split_hash"] = _SPLIT
        before = mean._state_hashes(model)
        training["state_before"] = before
        mean._write(
            out.parent / "train-membership.json", mean._training_membership(runtime.split.train)
        )
        torch.cuda.reset_peak_memory_stats()
        parent = mean._evaluate(
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
        features = mean._features(model, prompts)
        _require(
            list(mean._head(model.symmetric_relation_projection.weight, features).shape)
            == [1637, 1025],
            "full train head shape",
        )
        training["mean_bce_diagnostic"] = {
            "initial": _mean_diagnostic(model, features, labels, _prompt_set_loss, mean._head),
            "used_for_updates": False,
        }
        _require(
            training["mean_bce_diagnostic"]["initial"]
            == recipe["mean_bce_diagnostic"]["initial_expected"],
            "initial mean BCE replay mismatch",
        )
        training["mean_bce_diagnostic"]["initial_replay_exact"] = True
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
        mean._fit(model, features, labels, _worst_loss, optimizer, 500, training)
        torch.cuda.synchronize()
        training["refit_seconds"] = time.perf_counter() - fit_started
        training["mean_bce_diagnostic"]["final"] = _mean_diagnostic(
            model, features, labels, _prompt_set_loss, mean._head
        )
        after = mean._state_hashes(model)
        training["state_after"] = after
        training["projection_changed"] = mean._frozen(before, after)
        _require(training["projection_changed"], "projection did not change")
        training["frozen_tensors_unchanged"] = True
        child_config = {
            "inherited_parent_config": context["parent_run"]["config"],
            "projection_refit_recipe": recipe,
        }
        identity = ExperimentIdentity(
            source_commit=summary["workspace_provenance"]["commit"],
            resolved_config_hash=mean._digest(child_config),
            protocol_version=runtime.training_identity["protocol_version"],
            **runtime.corpus_identity,
            split_hash=_SPLIT,
            evaluator_version=_EVALUATOR,
            environment={
                **summary["environment"],
                "source_archive_sha256": helper._SOURCE,
                "configs_archive_sha256": helper._CONFIGS,
                "runner_sha256": summary["script_sha256"],
                "mean_helper_sha256": _MEAN_HELPER,
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
            "frozen_state_sha256": mean._digest({k: v for k, v in before.items() if k != _WEIGHT}),
            "final_post_update_loss": training["final_post_update_loss"],
            "mean_bce_diagnostic": training["mean_bce_diagnostic"],
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
        training["state_reloaded"] = mean._state_hashes(reloaded)
        _require(training["state_reloaded"] == after, "child reload tensor mismatch")
        _require(loaded.checkpoint_hash == saved.checkpoint_hash, "checkpoint reload hash")
        _require(
            mean._same_state(optimizer.state_dict(), reload_optimizer.state_dict()),
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
        child = mean._evaluate(
            reloaded,
            runtime.split.validation,
            context["wide"]["responses"],
            helper,
            out.parent / "child.json",
            "child",
            parent,
        )
        comparisons = _comparisons(
            child, parent, context["mean_child"], context["wide"]["responses"], mean
        )
        mean._write(out.parent / "comparisons.json", comparisons)
        summary["comparisons"] = comparisons["aggregate"]
        invariants = {
            "parent_replay_exact": parent["exact_parent_replay"],
            "train_only": True,
            "initial_mean_bce_replay_exact": training["mean_bce_diagnostic"][
                "initial_replay_exact"
            ],
            "complete_500_updates": training["completed_updates"] == 500,
            "frozen_tensors_unchanged": training["frozen_tensors_unchanged"],
            "child_reload_exact": training["reload_exact"],
        }
        summary["gate"] = mean._gate(child, invariants)
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
        mean._write(out.parent / "training.json", helper._failure_safe(training))
        summary["artifact_sha256"] = {
            p.name: helper._sha(p)
            for p in out.parent.iterdir()
            if p.name
            in {
                "parent.json",
                "child.json",
                "training.json",
                "comparisons.json",
                "mean-refit-comparator.json",
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
        "--plan",
        type=Path,
        default=Path("docs/experiments/2026-09-25-projection-worst-boundary-plan.md"),
    )
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/experiments/projection_worst_boundary_v1.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/projection-worst-boundary-v1/summary.json")
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
    mean, mean_path = _load_mean(root)
    helper, helper_path = mean._load_helper(root)
    helper._modules()
    out = args.out.resolve()
    mean._refuse(out)
    _require(
        not any(
            (out.parent / name).exists()
            for name in ("comparisons.json", "mean-refit-comparator.json")
        ),
        "immutable comparison output exists",
    )
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
        and receipt["tested_recipe_sha256"] == helper._sha(args.recipe)
        and receipt["tested_mean_helper_sha256"] == _MEAN_HELPER,
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
    dependencies = receipt["test_dependencies"]
    _require(
        set(dependencies)
        == {
            str(mean_path.resolve()),
            str((root / "tests/unit/test_projection_only_refit.py").resolve()),
            str((root / "configs/experiments/projection_only_refit_v1.json").resolve()),
        },
        "focused shared-test dependency inventory",
    )
    _require(dependencies[str(mean_path.resolve())] == _MEAN_HELPER, "tested mean helper identity")
    for path, pinned in dependencies.items():
        helper._bind(path, pinned, context["inputs"])
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": content,
        "helper.py": helper_path.read_bytes(),
        "mean-helper.py": mean_path.read_bytes(),
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
        "campaign_version": "projection-worst-boundary-v1",
        "complete": False,
        "acceptance": False,
        "objective": _OBJECTIVE,
        "evaluator": _EVALUATOR,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "helper_sha256": mean._HELPER,
        "mean_helper_sha256": _MEAN_HELPER,
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
    mean._write(out.parent / "mean-refit-comparator.json", context["mean_comparator"])
    _execute(root, out, context, mean, helper, summary)


if __name__ == "__main__":
    main()
