"""Seed-only fixed 8000-update replications; aggregation is a separate program."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

_REPLICATION = "24fc43fc7d31ec1bf7aa84b475af79198717fbb6e34f41453939a3da49e9fdde"
_EXTENDED = "cd593c14c429247837c26b09754509d5cfd1dd36adb317f56522a80a78a25122"
_PREVIOUS_SUMMARY = "338fb6b7ea08128952ce9abc02b3506639e712a9cff6ce63468ed2e7980f79a0"
_PREVIOUS_AUDIT = "ddbecaceabdacc0808f9b4f122e0309aa3e233fe76b0d75694e31833a40083a8"
_PREVIOUS_DECISION = "62a8dcbb7cbe5d038984f9d57aecacec4b0ba612ba55c2a9f6fe335977304978"
_PREVIOUS_SEEDS = {
    1730: "ecc49edfa4a7b7b2428326d752a020cf09ef92eefcb69157b12a3c4a340356e1",
    1731: "3f99d91b87f4531941ae3af11b3f39a42746df15f86df461db7e06386ecefc9c",
}

_PLAN = "750d115b2fd0daf356cb3a633248e826c264f30533ddf8e8df2ee125d2ecdc6b"
_BUDGET = "f4376d4f2d838938bb9f50ae0cfa16ef319639fa0ce917c9fea3b842c0018930"
_SCORER = "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
_MEAN = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
_SUMMARY = "e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8"
_AUDIT = "bdf142f48229fbcdb197bce3a019e38c3a65570ae3f8551a9c9a1f14b2198872"
_DECISION = "2cadf9dce7ee36cc4095a10a746cc1608d5ce4437724872c1eec1c88e915c195"
_MEMBERSHIP = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"
_SPLIT = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
_ARCHITECTURE = "plm-frozen-symmetric-bilinear-residual-v1"
_OBJECTIVE = "plm-bilinear-residual-balanced-bce-v1"
_EVALUATOR = "plm-bilinear-budget8000-replication-v1"
_CAMPAIGN = "bilinear-budget8000-replication-v1"
_PARAMETER = "symmetric_bilinear_residual"
_SEED_CONTRACTS = {
    1730: {
        "parent_checkpoint_sha256": (
            "5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2"
        ),
        "parent_config_sha256": "0eae5252ae9de2d1463992a319c43e695ae8bb4f24bfb2ae94acdf4da6033b31",
        "width8_sha256": "0a398f2adb53877a188bf72d444c460398b1ac21e2f4317f23bc2948c0dcb88a",
        "floors": {
            "exact_count": 212,
            "f1": 0.9997931269673187,
            "groups": {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 58},
        },
    },
    1731: {
        "parent_checkpoint_sha256": (
            "ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d"
        ),
        "parent_config_sha256": "9e029d5fa280492a382b3b138eb5933a8ef8580c6a95a90deb0c87420c3601b3",
        "width8_sha256": "a2004f3a8e98043f7e55fc775e0d33069a1a46a70b13f753a816a0d2ba2d7799",
        "floors": {
            "exact_count": 208,
            "f1": 0.999668812076336,
            "groups": {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 54},
        },
    },
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _load_budget(root):
    path = root / "scripts/refit_bilinear_budget.py"
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == _BUDGET, "budget helper identity")
    spec = importlib.util.spec_from_file_location("replication_budget_helper", path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _recipe(value, inherited):
    expected = json.loads(json.dumps(inherited))
    _require(
        expected["experiment"] == "bilinear-budget2000-v1" and expected["residual_updates"] == 2000,
        "inherited budget recipe",
    )
    for key in (
        "seed",
        "initial_training_loss",
        "strong_comparator_exact",
        "strong_comparator_f1",
        "strong_comparator_group_exact",
    ):
        expected.pop(key)
    expected.update(
        residual_updates=8000,
        experiment=_CAMPAIGN,
        evaluator=_EVALUATOR,
        fresh_seeds=[1730, 1731],
        historical_seed=1729,
        data_split_seed=1729,
        initial_training_replay=(
            "exact matching-parent versus actual zero-A full-training logits and loss"
        ),
        seed_contracts={str(k): v for k, v in _SEED_CONTRACTS.items()},
    )
    _require(
        json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True),
        "fixed replication recipe differs",
    )
    return value


def _seed_contract(seed, parent_run, sidecar, result, base1729_config):
    _require(type(seed) is int and seed in _SEED_CONTRACTS, "fresh seed mapping")
    contract = _SEED_CONTRACTS[seed]
    expected = json.loads(json.dumps(base1729_config))
    expected.update(seed=seed, run_name=f"national_dex_continuation_control_s{seed}_v1")
    _require(
        parent_run["config"] == expected and expected["data"]["split_seed"] == 1729,
        "parent config differs beyond seed/run name",
    )
    identity = parent_run["identity"]
    _require(
        identity == sidecar["experiment_identity"] == result["identity"]
        and identity["seeds"] == [seed]
        and identity["resolved_config_hash"] == contract["parent_config_sha256"],
        "matching parent identity/seed",
    )
    _require(
        sidecar["checkpoint_hash"]
        == result["checkpoint_hash"]
        == contract["parent_checkpoint_sha256"]
        and sidecar["global_step"] == result["global_step"] == 2000
        and sidecar["split_hash"] == identity["split_hash"] == _SPLIT
        and sidecar["training_metadata"]["objective"] == "causal-next-token-v1",
        "original parent checkpoint/split",
    )
    return {
        "seed": seed,
        "model_seed": seed,
        "data_split_seed": 1729,
        **json.loads(json.dumps(contract)),
    }


def _alignment(left, right):
    _require(
        left["product_token_ids"] == right["product_token_ids"] == list(range(1024, 2049))
        and left["query_count"] == right["query_count"] == 222,
        "reference columns/coverage",
    )
    _require(len(left["responses"]) == len(right["responses"]) == 222, "reference rows")
    for a, b in zip(left["responses"], right["responses"], strict=True):
        _require(
            all(
                a[k] == b[k]
                for k in (
                    "index",
                    "subject",
                    "dimension",
                    "prompt_ids",
                    "expected_set_ids",
                    "group",
                )
            ),
            "cross-seed query/label/group order",
        )


def _load_pinned(root, relative, digest, name):
    path = root / relative
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, name + " identity")
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _accepted_chain(root, folder, summary_name, pins, helper, inputs):
    summary_sha, audit_sha, decision_sha = pins
    summary = helper._read(helper._bind(root / folder / summary_name, summary_sha, inputs))
    audit = helper._read(helper._bind(root / folder / "independent-audit.json", audit_sha, inputs))
    decision = helper._read(helper._bind(root / folder / "decision.json", decision_sha, inputs))
    _require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and decision["evidence_accepted"] is True
        and decision["fixed_quality_gate_passed"] is True
        and audit["summary_sha256"] == decision["summary_sha256"] == summary_sha
        and decision["audit_sha256"] == audit_sha,
        "accepted historical chain",
    )
    return summary, audit, decision


def _preflight(
    root,
    plan,
    recipe_path,
    budget,
    budget_path,
    bilinear,
    bilinear_path,
    mean,
    mean_path,
    helper,
    helper_path,
):
    replication, replication_path = _load_pinned(
        root, "scripts/refit_bilinear_replication.py", _REPLICATION, "frozen_replication_preflight"
    )
    extended, extended_path = _load_pinned(
        root, "scripts/refit_bilinear_budget8000.py", _EXTENDED, "frozen_budget8000_arithmetic"
    )
    previous = root / "runs/learning/bilinear-seed-replication-v1"
    context = replication._preflight(
        root,
        previous / "summary.plan.md",
        previous / "summary.recipe.json",
        budget,
        budget_path,
        bilinear,
        bilinear_path,
        mean,
        mean_path,
        helper,
        helper_path,
    )
    inputs = context["inputs"]
    for path, digest in (
        (plan, _PLAN),
        (replication_path, _REPLICATION),
        (extended_path, _EXTENDED),
    ):
        helper._bind(path, digest, inputs)
    old, _, _ = _accepted_chain(
        root,
        previous,
        "aggregate-v2.json",
        (_PREVIOUS_SUMMARY, _PREVIOUS_AUDIT, _PREVIOUS_DECISION),
        helper,
        inputs,
    )
    historical_folder = root / "runs/learning/bilinear-budget8000-v1"
    historical_summary, _, _ = _accepted_chain(
        root,
        historical_folder,
        "summary.json",
        (_SUMMARY, _AUDIT, _DECISION),
        helper,
        inputs,
    )
    _require(
        historical_summary["script_sha256"] == _EXTENDED
        and historical_summary["scorer_sha256"] == _SCORER,
        "historical8000 implementation",
    )
    for name, digest in historical_summary["artifact_sha256"].items():
        helper._bind(historical_folder / name, digest, inputs)
    historical = helper._read(historical_folder / "child.json")
    _alignment(historical, context["wide"])
    for seed, contract in _SEED_CONTRACTS.items():
        folder = previous / f"seed-{seed}"
        refs = [row for row in old["reports"] if row["seed"] == seed]
        _require(
            len(refs) == 1 and refs[0]["sha256"] == _PREVIOUS_SEEDS[seed], "matched old summary"
        )
        report = helper._read(helper._bind(folder / "summary.json", _PREVIOUS_SEEDS[seed], inputs))
        _require(
            report["complete"] is True
            and report["seed"] == seed
            and report["parent_checkpoint_sha256"] == contract["parent_checkpoint_sha256"],
            "matched2000 parent",
        )
        for name, digest in report["artifact_sha256"].items():
            helper._bind(folder / name, digest, inputs)
        child = helper._read(folder / "child.json")
        _alignment(child, context["seeds"][seed]["wide"])
        actual = {
            "exact_count": child["aggregate"]["exact_count"],
            "f1": child["aggregate"]["f1"],
            "groups": {g: child["groups"][g]["exact_count"] for g in mean._GROUPS},
        }
        _require(actual == contract["floors"], "matched2000 floors")
        context["seeds"][seed].update(
            floors=contract["floors"],
            historical2000=child,
            historical2000_training=helper._read(folder / "training.json"),
            historical2000_summary_sha256=_PREVIOUS_SEEDS[seed],
            historical2000_child_sha256=report["artifact_sha256"]["child.json"],
            historical2000_training_sha256=report["artifact_sha256"]["training.json"],
        )
    recipe = _recipe(
        helper._read(recipe_path),
        helper._read(root / "configs/experiments/bilinear_budget2000_v1.json"),
    )
    helper._bind(recipe_path, helper._sha(recipe_path), inputs)
    context.update(
        recipe=recipe,
        extended=extended,
        extended_path=extended_path,
        replication_path=replication_path,
        historical1729=historical,
        historical1729_summary=historical_summary,
        historical1729_child_sha256=historical_summary["artifact_sha256"]["child.json"],
    )
    return context


def _initial_replay(model, features, dimension_tokens, labels, loss_fn, bilinear, mean):
    import torch

    with torch.no_grad():
        parent = mean._head(model.symmetric_relation_projection.weight, features)
        zero = bilinear._head(model, features, dimension_tokens, mean)
        parent_loss = loss_fn(parent, labels, excluded_ids=features[2])
        zero_loss = loss_fn(zero, labels, excluded_ids=features[2])
    _require(
        bool(torch.isfinite(parent).all())
        and bool(torch.isfinite(zero).all())
        and bool(torch.isfinite(parent_loss))
        and bool(torch.isfinite(zero_loss)),
        "finite initial train replay",
    )
    parent_bytes = bilinear._tensor_bytes(parent)
    zero_bytes = bilinear._tensor_bytes(zero)
    _require(
        torch.equal(parent, zero)
        and parent_bytes == zero_bytes
        and torch.equal(parent_loss, zero_loss),
        "matching-parent zero-A train logits/loss",
    )
    return {
        "parent_loss": float(parent_loss.cpu()),
        "zero_loss": float(zero_loss.cpu()),
        "shape": list(parent.shape),
        "parent_logits_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "zero_logits_sha256": hashlib.sha256(zero_bytes).hexdigest(),
        "logits_exact": True,
        "loss_exact": True,
    }


def _seed_gate(child, invariants, floors):
    aggregate, groups = child["aggregate"], child["groups"]
    checks = {
        "execution_invariants": bool(invariants) and all(v is True for v in invariants.values()),
        "serialization_compatible": aggregate["serialization_compatible"] == 222,
        "exact_above_matching2000": aggregate["exact_count"] > floors["exact_count"],
        "f1_at_least_matching2000": aggregate["f1"] >= floors["f1"],
        "group_exact_nonregression": all(
            groups[g]["exact_count"] >= v for g, v in floors["groups"].items()
        ),
    }
    return {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "independent_audit_required": True,
        "accepted": False,
    }


def _campaign_gate(reports, historical_accepted):
    _require(set(reports) == {1730, 1731}, "exact fresh seed coverage")
    for seed, report in reports.items():
        _require(
            report["seed"] == seed and report["complete"] is True, "complete matching fresh seed"
        )
    checks = {
        "historical1729_accepted": historical_accepted is True,
        **{
            f"fresh_seed{seed}": reports[seed]["gate"]["primary_checks_passed"] is True
            for seed in (1730, 1731)
        },
    }
    return {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "independent_audit_required": True,
        "accepted": False,
    }


def _aggregate_reports(fresh, historical, mean):
    _require(set(fresh) == {1730, 1731}, "exact fresh pooling coverage")
    for report in fresh.values():
        _alignment(report, historical)
    newrows = [r for seed in (1730, 1731) for r in fresh[seed]["responses"]]

    def pool(rows):
        return {
            "aggregate": mean._totals(rows),
            "groups": {g: mean._totals([r for r in rows if r["group"] == g]) for g in mean._GROUPS},
        }

    return {"fresh_two": pool(newrows), "all_three": pool(historical["responses"] + newrows)}


def _validate_child(state, identity, config, metadata, corpus, seed, bilinear):
    _require(seed in _SEED_CONTRACTS, "fresh child seed")
    raw = state.metadata
    _require(state.global_step == raw["global_step"] == 8000, "child update count")
    _require(
        raw["experiment_identity"] == identity and identity["evaluator_version"] == _EVALUATOR,
        "child identity",
    )
    _require(
        identity["seeds"] == [seed]
        and metadata["seed"] == config["seed"] == seed
        and metadata["data_split_seed"] == config["data_split_seed"] == 1729
        and metadata["campaign_version"] == config["campaign_version"] == _CAMPAIGN,
        "child seed/campaign contract",
    )
    _require(raw["config"] == config and raw["training_metadata"] == metadata, "child metadata")
    _require(
        metadata["implementation_identity"] == config["implementation_identity"]
        and metadata["implementation_identity"]["runner_sha256"] == metadata["runner_sha256"]
        and metadata["implementation_identity"]["scorer_sha256"]
        == metadata["scorer_sha256"]
        == _SCORER
        and metadata["implementation_identity"]["mean_helper_sha256"] == _MEAN
        and metadata["implementation_identity"]["budget_helper_sha256"] == _BUDGET
        and metadata["implementation_identity"]["replication_helper_sha256"] == _REPLICATION
        and metadata["implementation_identity"]["budget8000_helper_sha256"] == _EXTENDED,
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
        and metadata["residual_updates"] == 8000
        and metadata["parent_training_steps"] == 2000,
        "child objective/lineage",
    )
    _require(
        metadata["parent_checkpoint_sha256"] == _SEED_CONTRACTS[seed]["parent_checkpoint_sha256"],
        "original parent",
    )
    _require(
        raw["corpus_identity"] == corpus
        and raw["split_hash"] == "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d",
        "child corpus/split",
    )


def _execute(root, out, context, bilinear, mean, helper, summary):
    seed = context["seed"]
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
        seed_everything(seed, deterministic=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        summary["environment"] = helper._environment(torch)
        summary["numerical_settings"] = helper._settings(torch) | {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        summary["workspace_provenance"] = mean._git(root)
        config = RootConfig.model_validate(context["references"][seed]["offline"]["config"])
        runtime = load_inference_runtime(
            config,
            context["parent"],
            corpus=(root / config.data.corpus_manifest).parent,
            graph_db=root / config.data.graph_db,
            protocol=load_protocol(runtime_root / "configs/protocol/pokemon_v1.yaml"),
        )
        _require(
            runtime.checkpoint_hash == context["parent_checkpoint_sha256"]
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
            len(parent_state) == 93,
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
            validation_update_points=[0, 8000],
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
        replay = _initial_replay(
            model, features, prompts[:, 2], labels, _prompt_set_loss, bilinear, mean
        )
        training["initial_training_replay"] = replay
        _require(replay["shape"] == [1637, 1025], "full train logit shape")
        initial_loss = replay["zero_loss"]
        _require(
            initial_loss == context["historical2000_training"]["initial_pre_update_loss"],
            "matching historical initial loss",
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
            8000,
            training,
            mean,
        )
        torch.cuda.synchronize()
        training["refit_seconds"] = time.perf_counter() - fit_started
        training["training_prefix_diagnostic"] = context["extended"]._prefix_diagnostic(
            training, context["historical2000_training"]
        )
        summary["training_prefix_diagnostic"] = training["training_prefix_diagnostic"]
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
            "budget_helper_sha256": _BUDGET,
            "replication_helper_sha256": _REPLICATION,
            "budget8000_helper_sha256": _EXTENDED,
            "mean_helper_sha256": _MEAN,
            "source_archive_sha256": helper._SOURCE,
            "configs_archive_sha256": helper._CONFIGS,
        }
        child_config = {
            "seed": seed,
            "data_split_seed": 1729,
            "campaign_version": _CAMPAIGN,
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
                "budget_helper_sha256": _BUDGET,
                "replication_helper_sha256": _REPLICATION,
                "budget8000_helper_sha256": _EXTENDED,
                "recipe_sha256": summary["recipe_sha256"],
                "mean_helper_sha256": _MEAN,
                "architecture": _ARCHITECTURE,
            },
            seeds=(seed,),
        )
        metadata = {
            "seed": seed,
            "data_split_seed": 1729,
            "campaign_version": _CAMPAIGN,
            "implementation_identity": implementation,
            "objective": _OBJECTIVE,
            "evaluator": _EVALUATOR,
            "architecture": bilinear._architecture(),
            "model_config": model.config.model_dump(mode="json"),
            "model_config_scope": "inherited archived base only; residual architecture is separate",
            "parent_checkpoint_sha256": context["parent_checkpoint_sha256"],
            "parent_training_steps": 2000,
            "residual_updates": 8000,
            "trainable_parameters": [_PARAMETER],
            "train_query_order_sha256": summary["train_query_order_sha256"],
            "record_count": 1637,
            "recipe_sha256": summary["recipe_sha256"],
            "runner_sha256": summary["script_sha256"],
            "scorer_sha256": _SCORER,
            "budget_helper_sha256": _BUDGET,
            "replication_helper_sha256": _REPLICATION,
            "budget8000_helper_sha256": _EXTENDED,
            "parent_state_sha256": mean._digest(parent_state),
            "initial_state_sha256": mean._digest(initial),
            "initial_pre_update_loss": initial_loss,
            "final_post_update_loss": training["final_post_update_loss"],
            "training_prefix_diagnostic": training["training_prefix_diagnostic"],
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
        reloaded = bilinear._factory(model.config, build_model, runtime.device)
        reload_optimizer = create_optimizer(
            reloaded, OptimizerConfig.model_validate(recipe["optimizer"])
        )
        loaded = load_checkpoint(
            checkpoint, reloaded, optimizer=reload_optimizer, map_location="cpu", restore_rng=False
        )
        _validate_child(
            loaded,
            identity.to_dict(),
            child_config,
            metadata,
            runtime.corpus_identity,
            seed,
            bilinear,
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
            "historical2000_summary_sha256": context["historical2000_summary_sha256"],
            "historical2000_child_sha256": context["historical2000_child_sha256"],
            "comparison_scope": "saved seed-matched historical output; no rerun",
            "paired_vs_historical2000": context["extended"]._historical_pair(
                child, context["historical2000"], mean
            ),
        }
        mean._write(out.parent / "comparisons.json", comparisons)
        invariants = {
            "parent_replay_exact": parent["exact_parent_replay"],
            "zero_A_replay_exact": zero["complete"]
            and zero["logits_sha256"] == zero["parent_logits_sha256"],
            "initial_train_loss_exact": replay["parent_loss"] == initial_loss
            and replay["logits_exact"] is True,
            "historical_initial_loss_exact": initial_loss
            == context["historical2000_training"]["initial_pre_update_loss"],
            "completed_8000_updates": training["completed_updates"] == 8000,
            "original_93_tensors_unchanged": training["frozen_tensors_unchanged"],
            "residual_changed": training["residual_changed"],
            "reload_exact": training["reload_exact"],
            "optimizer_reload_exact": training["optimizer_reload_exact"],
            "train_only_supervision": training["validation_labels_used"] is False,
        }
        summary.update(
            invariants=invariants,
            gate=_seed_gate(child, invariants, context["floors"]),
            parent=parent["aggregate"],
            child={
                "aggregate": child["aggregate"],
                "groups": child["groups"],
                "paired_vs_parent_dense": child["paired_vs_parent_dense"],
                "paired_vs_width8": child["paired_vs_width8"],
                "paired_vs_historical2000": comparisons["paired_vs_historical2000"],
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
                "comparisons.json",
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


def _refuse(out, mean, aggregate=False):
    mean._refuse(out)
    _require(not (out.parent / "zero-replay.json").exists(), "immutable zero replay")
    if aggregate:
        _require(
            not out.with_suffix(".aggregate-inputs.json").exists(), "immutable aggregate inputs"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1730, 1731))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/experiments/2026-09-25-bilinear-budget8000-replication-plan.md"),
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("configs/experiments/bilinear_budget8000_replication_v1.json"),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    args = parser.parse_args()
    _require(
        args.preflight_only or args.seed is not None,
        "choose --seed",
    )
    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    content = script.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    _require(
        "torch" not in sys.modules and Path.cwd().resolve() == root, "Torch-free root preflight"
    )
    budget, budget_path = _load_budget(root)
    bilinear, bilinear_path = budget._load_bilinear(root)
    mean, mean_path = bilinear._load_mean(root)
    helper, helper_path = mean._load_helper(root)
    helper._modules()
    folder = root / "runs/learning/bilinear-budget8000-replication-v1"
    if args.seed is not None:
        folder = folder / f"seed-{args.seed}"
    out = (args.out or folder / "summary.json").resolve()
    _refuse(out, mean)
    context = _preflight(
        root,
        args.plan.resolve(),
        args.recipe.resolve(),
        budget,
        budget_path,
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
                    "fresh_seeds": [1730, 1731],
                    "input_count": len(context["inputs"]),
                    "script_sha256": digest,
                    "scorer_sha256": _SCORER,
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
        root / "tests/unit/test_bilinear_budget8000_replication.py",
        root / "tests/unit/test_bilinear_replication.py",
        root / "tests/unit/test_bilinear_budget8000.py",
        context["replication_path"],
        context["extended_path"],
        root / "configs/experiments/bilinear_seed_replication_v1.json",
        root / "configs/experiments/bilinear_budget8000_v1.json",
        root / "tests/unit/test_bilinear_budget.py",
        root / "tests/unit/test_bilinear_residual.py",
        root / "tests/unit/test_projection_only_refit.py",
        root / "tests/fixtures/projection_refit_runtime_source.zip",
        budget_path,
        bilinear_path,
        mean_path,
        root / "configs/experiments/bilinear_budget2000_v1.json",
        root / "configs/experiments/bilinear_residual_refit_v1.json",
        root / "configs/experiments/projection_only_refit_v1.json",
    ]
    _require(
        set(receipt["test_dependencies"]) == {str(p.resolve()) for p in required},
        "test dependency inventory",
    )
    for path, pinned in receipt["test_dependencies"].items():
        helper._bind(path, pinned, context["inputs"])
    stdout = helper._bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"], context["inputs"])
    helper._unchanged(context["inputs"])
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": content,
        "budget-helper.py": budget_path.read_bytes(),
        "replication-helper.py": context["replication_path"].read_bytes(),
        "budget8000-helper.py": context["extended_path"].read_bytes(),
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
        "campaign_version": _CAMPAIGN,
        "report_kind": "fresh_seed",
        "complete": False,
        "acceptance": False,
        "architecture": bilinear._architecture(),
        "objective": _OBJECTIVE,
        "evaluator": _EVALUATOR,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "scorer_sha256": _SCORER,
        "budget_helper_sha256": _BUDGET,
        "replication_helper_sha256": _REPLICATION,
        "budget8000_helper_sha256": _EXTENDED,
        "mean_helper_sha256": _MEAN,
        "helper_sha256": mean._HELPER,
        "recipe_sha256": helper._sha(args.recipe),
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "historical1729_summary_sha256": _SUMMARY,
        "historical1729_audit_sha256": _AUDIT,
        "historical1729_decision_sha256": _DECISION,
        "protected_test_used": False,
        "standard_serving_supported": False,
        "limitations": [
            "Two fresh training seeds share 222 validation queries; "
            "666 observations are not 666 independent examples.",
            "Seed 1729 is reused historical development/selection evidence, "
            "not a fresh replication.",
            "Architecture/objective/scorer are unchanged; evaluator binds "
            "matching-seed floors and conjunction.",
            "Dense validation does not establish autoregressive generation, "
            "serving or protected-test quality.",
            "Terminal chronology is an owner observation bound by receipts, "
            "not independent execution replay.",
            "Local checkpoint hashes do not establish durable remote weight archival.",
        ],
    }
    own = {**context, **context["seeds"][args.seed]}
    summary.update(
        seed=args.seed,
        model_seed=args.seed,
        data_split_seed=1729,
        parent_checkpoint_sha256=own["parent_checkpoint_sha256"],
        parent_config_sha256=own["parent_config_sha256"],
        width8_sha256=own["width8_sha256"],
        comparator_floors=own["floors"],
        historical2000_summary_sha256=own["historical2000_summary_sha256"],
        historical2000_child_sha256=own["historical2000_child_sha256"],
        historical2000_training_sha256=own["historical2000_training_sha256"],
        historical2000_aggregate_sha256=_PREVIOUS_SUMMARY,
        historical2000_audit_sha256=_PREVIOUS_AUDIT,
        historical2000_decision_sha256=_PREVIOUS_DECISION,
    )
    _execute(root, out, own, bilinear, mean, helper, summary)


if __name__ == "__main__":
    main()
