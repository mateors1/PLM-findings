"""Fixed 2000-update sibling screen using the immutable bilinear scorer.

Only the optimization budget changes. Historical500 results are saved comparators,
never resumed checkpoints or additional executed validation endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

_PLAN = "18c08693790880740b20cf25633b56880392362ffbc6b398cd7069116b26edbe"
_SCORER = "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
_MEAN = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
_SUMMARY = "92955bf21ce9dc85ce9806436417436cece618d1be9edd3c400f6d90c96bc0b5"
_AUDIT = "2d8f3d49a4d6e25eaee319f3517b7ec69e139a566a814d4be1f01a44cdbef417"
_DECISION = "d7d38687e2d099348af61bd52ed62df28211464e4209cb00df43db21d1d93234"
_AUDITOR = "a046a1e45b3c977fc556ecc74358d64e2b9d0463ebf653c7b7f7c86aed61622b"
_CHILD = "1f6e9a593ac5c5c83f0d003f273adc02213845ab94f7d5118b1b55bf5bab43d9"
_MEMBERSHIP = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"
_ARCHITECTURE = "plm-frozen-symmetric-bilinear-residual-v1"
_OBJECTIVE = "plm-bilinear-residual-balanced-bce-v1"
_EVALUATOR = "plm-bilinear-residual-screen-v1"
_PARAMETER = "symmetric_bilinear_residual"


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


def _recipe(value, inherited):
    expected = json.loads(json.dumps(inherited))
    _require(
        expected["experiment"] == "bilinear-residual-refit-v1"
        and expected["residual_updates"] == 500,
        "historical recipe",
    )
    expected.update(experiment="bilinear-budget2000-v1", residual_updates=2000)
    _require(
        json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True),
        "fixed budget recipe differs",
    )
    return value


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


def _preflight(
    root, plan, recipe_path, bilinear, bilinear_path, mean, mean_path, helper, helper_path
):
    old = root / "runs/learning/bilinear-residual-refit-v1"
    context = bilinear._preflight(
        root,
        old / "summary.plan.md",
        old / "summary.recipe.json",
        mean,
        mean_path,
        helper,
        helper_path,
    )
    inputs = context["inputs"]
    helper._bind(plan, _PLAN, inputs)
    helper._bind(bilinear_path, _SCORER, inputs)
    summary = helper._read(helper._bind(old / "summary.json", _SUMMARY, inputs))
    audit = helper._read(helper._bind(old / "independent-audit.json", _AUDIT, inputs))
    decision = helper._read(helper._bind(old / "decision.json", _DECISION, inputs))
    helper._bind(old / "independent-audit.py", _AUDITOR, inputs)
    _require(
        summary["complete"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and decision["evidence_accepted"] is True
        and audit["summary_sha256"] == decision["summary_sha256"] == _SUMMARY
        and decision["audit_sha256"] == _AUDIT
        and audit["script_sha256"] == _AUDITOR,
        "accepted historical500 evidence",
    )
    _require(
        summary["script_sha256"] == _SCORER
        and summary["split_hash"] == mean._SPLIT
        and summary["architecture"] == bilinear._architecture()
        and summary["objective"] == _OBJECTIVE
        and summary["evaluator"] == _EVALUATOR,
        "historical architecture/split/scorer",
    )
    for name, digest in summary["artifact_sha256"].items():
        helper._bind(old / name, digest, inputs)
    _require(
        summary["artifact_sha256"]["checkpoint-final.pt"] == _CHILD, "historical child identity"
    )
    historical = helper._read(old / "child.json")
    parent = helper._read(old / "parent.json")
    _historical_pair(parent, historical, mean)
    _require(historical["query_count"] == 222, "historical validation coverage")
    _require(
        summary["parent_training_identity"] == context["parent_run"]["identity"],
        "historical original parent",
    )
    for row, wide in zip(historical["responses"], context["wide"]["responses"], strict=True):
        _require(
            all(
                row[k] == wide[k]
                for k in ("subject", "dimension", "prompt_ids", "group", "expected_set_ids")
            ),
            "historical comparator alignment",
        )
    recipe = _recipe(helper._read(recipe_path), context["recipe"])
    helper._bind(recipe_path, helper._sha(recipe_path), inputs)
    context.update(
        recipe=recipe,
        recipe_path=recipe_path,
        historical500=historical,
        historical500_child_sha256=summary["artifact_sha256"]["child.json"],
    )
    return context


def _validate_child(state, identity, config, metadata, corpus, bilinear):
    raw = state.metadata
    _require(state.global_step == raw["global_step"] == 2000, "child update count")
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
        == _SCORER
        and metadata["implementation_identity"]["mean_helper_sha256"] == _MEAN,
        "child implementation identity",
    )
    _require(
        metadata["architecture"]
        == config["bilinear_residual_architecture"]
        == bilinear._architecture(),
        "child architecture",
    )
    _require(
        metadata["objective"] == _OBJECTIVE
        and metadata["evaluator"] == _EVALUATOR
        and metadata["trainable_parameters"] == [_PARAMETER]
        and metadata["residual_updates"] == 2000
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


def _execute(root, out, context, bilinear, mean, helper, summary):
    started = time.perf_counter()
    runtime_root = out.parent / "runtime"
    training = {
        "complete": False,
        "objective": _OBJECTIVE,
        "architecture": bilinear._architecture(),
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
            validation_update_points=[0, 2000],
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
        bilinear._attach(model)
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
        zero = bilinear._zero_replay(
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
            scores = bilinear._head(model, features, prompts[:, 2], mean)
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
        bilinear._fit(
            model,
            features,
            prompts[:, 2],
            labels,
            _prompt_set_loss,
            optimizer,
            2000,
            training,
            mean,
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
            "scorer_sha256": _SCORER,
            "mean_helper_sha256": _MEAN,
            "source_archive_sha256": helper._SOURCE,
            "configs_archive_sha256": helper._CONFIGS,
        }
        child_config = {
            "implementation_identity": implementation,
            "inherited_parent_config": context["parent_run"]["config"],
            "bilinear_residual_architecture": bilinear._architecture(),
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
                "scorer_sha256": _SCORER,
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
            "architecture": bilinear._architecture(),
            "model_config": model.config.model_dump(mode="json"),
            "model_config_scope": "inherited archived base only; residual architecture is separate",
            "parent_checkpoint_sha256": mean._PARENT,
            "parent_training_steps": 2000,
            "residual_updates": 2000,
            "trainable_parameters": [_PARAMETER],
            "train_query_order_sha256": summary["train_query_order_sha256"],
            "record_count": 1637,
            "recipe_sha256": summary["recipe_sha256"],
            "runner_sha256": summary["script_sha256"],
            "scorer_sha256": _SCORER,
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
            global_step=2000,
            config=child_config,
            identity=identity,
            corpus_identity=runtime.corpus_identity,
            split_hash=mean._SPLIT,
            training_metadata=metadata,
        )
        reloaded = bilinear._factory(model.config, build_model, runtime.device)
        reload_optimizer = create_optimizer(
            reloaded, OptimizerConfig.model_validate(recipe["optimizer"])
        )
        loaded = load_checkpoint(
            checkpoint, reloaded, optimizer=reload_optimizer, map_location="cpu", restore_rng=False
        )
        _validate_child(
            loaded, identity.to_dict(), child_config, metadata, runtime.corpus_identity, bilinear
        )
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
        child = bilinear._evaluate(
            reloaded,
            runtime.split.validation,
            context["wide"]["responses"],
            mean,
            helper,
            out.parent / "child.json",
            parent=parent,
        )
        comparisons = {
            "historical500_summary_sha256": _SUMMARY,
            "historical500_child_sha256": context["historical500_child_sha256"],
            "comparison_scope": "saved historical output; no rerun",
            "paired_vs_historical500": _historical_pair(child, context["historical500"], mean),
        }
        mean._write(out.parent / "comparisons.json", comparisons)
        invariants = {
            "parent_replay_exact": parent["exact_parent_replay"],
            "zero_A_replay_exact": zero["complete"]
            and zero["logits_sha256"] == zero["parent_logits_sha256"],
            "initial_train_loss_exact": initial_loss == recipe["initial_training_loss"],
            "completed_2000_updates": training["completed_updates"] == 2000,
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
                "paired_vs_historical500": comparisons["paired_vs_historical500"],
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
                "comparisons.json",
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
        "--plan", type=Path, default=Path("docs/experiments/2026-09-25-bilinear-budget2000-plan.md")
    )
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/experiments/bilinear_budget2000_v1.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/bilinear-budget2000-v1/summary.json")
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
    bilinear, bilinear_path = _load_bilinear(root)
    mean, mean_path = bilinear._load_mean(root)
    helper, helper_path = mean._load_helper(root)
    helper._modules()
    out = args.out.resolve()
    mean._refuse(out)
    _require(not (out.parent / "zero-replay.json").exists(), "immutable zero replay")
    _require(not (out.parent / "comparisons.json").exists(), "immutable comparisons")
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
                    "scorer_sha256": _SCORER,
                    "historical500_summary_sha256": _SUMMARY,
                    "historical500_audit_sha256": _AUDIT,
                    "historical500_decision_sha256": _DECISION,
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
        root / "tests/unit/test_bilinear_budget.py",
        root / "tests/unit/test_bilinear_residual.py",
        bilinear_path,
        root / "configs/experiments/bilinear_residual_refit_v1.json",
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
        "bilinear-helper.py": bilinear_path.read_bytes(),
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
        "campaign_version": "bilinear-budget2000-v1",
        "complete": False,
        "acceptance": False,
        "architecture": bilinear._architecture(),
        "objective": _OBJECTIVE,
        "evaluator": _EVALUATOR,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "scorer_sha256": _SCORER,
        "historical500_summary_sha256": _SUMMARY,
        "historical500_audit_sha256": _AUDIT,
        "historical500_decision_sha256": _DECISION,
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
                "Only declared update budget changes versus historical500; "
                "adaptive single-seed comparison does not establish convergence or generalization."
            ),
            "Historical500 output reused, not freshly rerun or a timing control.",
            "Mathematical symmetry does not promise swapped FP32 bitwise equality.",
            "Inherited decoder forward is not the new membership scorer.",
            "Dense head evaluation does not establish autoregressive generation quality.",
            "No serving throughput/energy or durable remote weight archival claim.",
        ],
    }
    _execute(root, out, context, bilinear, mean, helper, summary)


if __name__ == "__main__":
    main()
