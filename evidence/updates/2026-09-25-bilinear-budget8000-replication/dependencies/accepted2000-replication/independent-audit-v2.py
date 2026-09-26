"""Independent saved-evidence audit of two fresh bilinear seed replications.

CPU payload inspection only. No primary prediction/gate imports, neural forward,
optimizer execution, or regeneration of the reported training trajectory.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import zipfile
from datetime import datetime
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PLAN_SHA = "d5a36d48eac3c40a30a5c29c2126f3d4f815f5ecb4d93a03f8cc47f771e75b54"
BUDGET_AUDITOR_SHA = "c674cf8e4a3ab2056ca4a02746ee55ead639e2a10e6b208525cb0e9dc906d1dc"
BUDGET_SUMMARY_SHA = "c507162902e173964c6fec1446448a074ee8fed41f4acb3a4f6d46d6744907a3"
BUDGET_AUDIT_SHA = "475c50f91891ccd13c180472e2d019913b3e5432ab0a9d3b15162735711aac3d"
BUDGET_DECISION_SHA = "edaf01b58cd09ad4524123f5f817c64602d0414c4220e1c709487016cf2f89fb"
BUDGET_RUNNER_SHA = "f4376d4f2d838938bb9f50ae0cfa16ef319639fa0ce917c9fea3b842c0018930"
WIDE_SUMMARY_SHA = "1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183"
WIDE_AUDIT_SHA = "eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95"
WIDE_DECISION_SHA = "70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601"
RUNNER_SHA = "24fc43fc7d31ec1bf7aa84b475af79198717fbb6e34f41453939a3da49e9fdde"
RECIPE_SHA = "4090ae0fcb6040147f9bfcee6bcfaf8bcf5cda95d9d484ba977ecddcc90121f7"
RUNNER_RECEIPT_SHA = "b786a28b9970843ccc14186748757b25ac187285a40775ee6a2cd8a3d8622229"
EVALUATOR = "plm-bilinear-seed-replication-v1"
CAMPAIGN = "bilinear-seed-replication-v1"
SEEDS = (1730, 1731)
PINS = {
    1730: {
        "parent": "5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2",
        "config": "0eae5252ae9de2d1463992a319c43e695ae8bb4f24bfb2ae94acdf4da6033b31",
        "wide": "0a398f2adb53877a188bf72d444c460398b1ac21e2f4317f23bc2948c0dcb88a",
        "exact": 205,
        "f1": 0.9906969338820507,
        "groups": {"COLOR": 103, "TYPE_single": 45, "TYPE_dual": 57},
    },
    1731: {
        "parent": "ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d",
        "config": "9e029d5fa280492a382b3b138eb5933a8ef8580c6a95a90deb0c87420c3601b3",
        "wide": "a2004f3a8e98043f7e55fc775e0d33069a1a46a70b13f753a816a0d2ba2d7799",
        "exact": 197,
        "f1": 0.9798558083418223,
        "groups": {"COLOR": 103, "TYPE_single": 44, "TYPE_dual": 50},
    },
}
INPUTS = {}
V1_SHA = "c1170a43629111d1a9904b9c26c2ef9cba77ff46612486dd868d02b03601f0a0"
V1_FAILED_RECEIPT_SHA = "6545cfbe9e37519888a9d86ff188a71d7bb578e5c8486ace82eba8ffeaf982a5"
V1_FAILED_STDOUT_SHA = "7faf1f1c30cbb0cc815214d4479d66e8bfa595835eed4e6b07dfead045e12275"
REPAIR_PLAN_SHA = "0020f6e9136669109101af048a830690eaa716d9a3607dd1b612e95ab0a57e4f"
AGGREGATION_RUNNER_SHA = "6cd3418a61ba3533445854a847c82aa80ffb9951b9baf602f896b876bf6ed1e4"
AGGREGATION_RECEIPT_SHA = "572b0b7ee24a42116f9b154dab4750b915eded70f67e3b32a3c072287576755f"
FAILED_AGGREGATE_SHA = "c3d79ed3effe7504b67121b25274603b2a5870f9150a5e8043b975f9dc478c2c"
FAILED_AGGREGATE_STDOUT_SHA = "68736e31ddca6d572c4384273d7317dbfb02bdbb37896de8b555df40296fbce3"
FAILED_AGGREGATE_RECEIPT_SHA = "31101de61a287e9e7addee93cffe881d0591ac338af0826448c810cac365837b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helpers():
    path = ROOT / "runs/learning/bilinear-budget2000-v1/independent-audit.py"
    if sha(path) != BUDGET_AUDITOR_SHA:
        raise ValueError("frozen independent budget helper changed")
    spec = importlib.util.spec_from_file_location("independent_seed_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUDGET = load_helpers()
BASE = BUDGET.BASE
require, exact, finite = BASE.require, BASE.exact, BASE.finite
A = BUDGET.A
OBJECTIVE = BUDGET.OBJECTIVE
ARCHITECTURE = BUDGET.ARCHITECTURE_DESCRIPTOR
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")


def bind(path, digest):
    path = Path(path).resolve()
    require(sha(path) == digest, "hash mismatch: " + str(path))
    require(str(path) not in INPUTS or INPUTS[str(path)] == digest, "conflicting file identity")
    INPUTS[str(path)] = digest
    return path


def read(path, digest=None):
    return json.loads(bind(path, digest or sha(path)).read_text(encoding="utf-8"))


def manifest(mapping):
    require(isinstance(mapping, dict) and mapping, "nonempty hash manifest")
    for path, digest in mapping.items():
        bind(path, digest)


def aggregate_inventory_contract(declared_preflight, snapshot, dynamic):
    exact(declared_preflight, snapshot, "explicit preflight inventory")
    require(isinstance(snapshot, dict) and bool(snapshot), "nonempty preflight inventory")
    require(all(dynamic.get(p) == h for p, h in snapshot.items()), "preflight inputs retained")


def load_payload(path, digest):
    import torch

    return torch.load(bind(path, digest), map_location="cpu", weights_only=False)


def seed_contract(seed):
    require(type(seed) is int and seed in SEEDS, "exact fresh seed membership")


def recipe_contract(recipe, inherited):
    expected = json.loads(json.dumps(inherited))
    require(
        expected["experiment"] == "bilinear-budget2000-v1" and expected["residual_updates"] == 2000,
        "accepted budget recipe",
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
        experiment=CAMPAIGN,
        evaluator=EVALUATOR,
        fresh_seeds=list(SEEDS),
        historical_seed=1729,
        data_split_seed=1729,
        initial_training_replay=(
            "exact matching-parent versus actual zero-A full-training logits and loss"
        ),
        seed_contracts={
            str(seed): {
                "parent_checkpoint_sha256": PINS[seed]["parent"],
                "parent_config_sha256": PINS[seed]["config"],
                "width8_sha256": PINS[seed]["wide"],
                "floors": {
                    "exact_count": PINS[seed]["exact"],
                    "f1": PINS[seed]["f1"],
                    "groups": PINS[seed]["groups"],
                },
            }
            for seed in SEEDS
        },
    )
    exact(recipe, expected, "fixed recipe changes only declared replication contract")


def lineage_contract(seed, metadata, identity):
    seed_contract(seed)
    BUDGET.architecture_contract(metadata["architecture"])
    require(
        metadata["seed"] == seed
        and metadata["data_split_seed"] == 1729
        and metadata["campaign_version"] == CAMPAIGN
        and identity["seeds"] == [seed]
        and metadata["parent_checkpoint_sha256"] == PINS[seed]["parent"]
        and metadata["parent_training_steps"] == metadata["residual_updates"] == 2000
        and metadata["objective"] == OBJECTIVE
        and metadata["evaluator"] == EVALUATOR
        and identity["evaluator_version"] == EVALUATOR
        and metadata["record_count"] == 1637,
        "matching fresh seed/parent/new evaluator lineage",
    )
    exact(metadata["trainable_parameters"], [A], "sole residual trainable")


def loss_contract(history, parent_loss, zero_loss, initial, final):
    """Finite trajectory receipts with an independently measured parent endpoint."""
    require(len(history) == 2000, "exactly 2000 updates")
    for number, row in enumerate(history, 1):
        require(type(row["update"]) is int and row["update"] == number, "ordered updates")
        finite(row["pre_update_loss"])
        require(
            row["pre_update_loss"] >= 0
            and row["gradient_finite"] is True
            and row["parameters_finite"] is True,
            "finite update attestations",
        )
    for value in (parent_loss, zero_loss, initial, final):
        finite(value)
        require(value >= 0, "nonnegative loss")
    require(
        parent_loss == zero_loss == initial == history[0]["pre_update_loss"],
        "seed-specific original parent/zero-A/first-update equality",
    )


def parent_config_contract(seed, config, historical_config):
    seed_contract(seed)
    require(config["seed"] == seed and config["data"]["split_seed"] == 1729, "model/split seeds")
    expected = dict(historical_config)
    expected.update(seed=seed, run_name=f"national_dex_continuation_control_s{seed}_v1")
    exact(config, expected, "original recipe differs only in run name and model seed")


def original_parent_schema(run, sidecar, result):
    """Original causal training stores metadata in its sidecar, not run.json."""
    require(set(run) == {"config", "identity"}, "original run schema")
    exact(sidecar["config"], run["config"]["train"], "original checkpoint training config")
    exact(sidecar["experiment_identity"], run["identity"], "original checkpoint identity")
    exact(result["identity"], run["identity"], "original result identity")
    metadata = sidecar["training_metadata"]
    exact(metadata["model_config"], run["config"]["model"], "original model config metadata")
    require(
        metadata["objective"] == "causal-next-token-v1"
        and metadata["record_count"] == 1637
        and metadata["data_order"] == "sequential-v1"
        and metadata["batch_size"] == run["config"]["train"]["batch_size"] == 32
        and metadata["grad_accum_steps"] == run["config"]["train"]["grad_accum_steps"] == 1,
        "original causal training metadata",
    )
    finite(metadata["train_loss"])
    require(metadata["train_loss"] == result["train_loss"], "original training loss receipt")


def training_replay_contract(replay, history, initial, final):
    exact(replay["shape"], [1637, 1025], "full training head shape")
    require(
        replay["logits_exact"] is True and replay["loss_exact"] is True,
        "actual zero-A full-training replay attestations",
    )
    left, right = replay["parent_logits_sha256"], replay["zero_logits_sha256"]
    require(
        isinstance(left, str)
        and len(left) == 64
        and left == right
        and all(c in "0123456789abcdef" for c in left),
        "full-training exact FP32 byte hashes",
    )
    loss_contract(history, replay["parent_loss"], replay["zero_loss"], initial, final)


def seed_gate(child, baseline, invariants):
    """Each seed stands on its own comparator; pooling cannot affect this gate."""
    metrics, groups = child["aggregate"], child["groups"]
    checks = {
        "execution_invariants": bool(invariants) and all(v is True for v in invariants.values()),
        "serialization_compatible": metrics["serialization_compatible"] == 222,
        "exact_above_matching_width8": metrics["exact_count"] > baseline["exact_count"],
        "f1_at_least_matching_width8": metrics["f1"] >= baseline["f1"],
        "group_exact_nonregression": all(
            groups[g]["exact_count"] >= baseline["groups"][g] for g in GROUPS
        ),
    }
    return {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "independent_audit_required": True,
        "accepted": False,
    }


def fresh_conjunction(gates):
    require(set(gates) == set(SEEDS), "exactly both fresh seed gates")
    require(all(type(gates[s]) is bool for s in SEEDS), "boolean seed gate outcomes")
    return all(gates[s] for s in SEEDS)


def aligned_reports(reports):
    require(len(reports) >= 2, "multiple seed reports")
    keys = ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_ids")
    reference = reports[0]
    require(reference["query_count"] == len(reference["responses"]) == 222, "reference coverage")
    for report in reports[1:]:
        require(report["query_count"] == len(report["responses"]) == 222, "seed coverage")
        exact(report["product_token_ids"], reference["product_token_ids"], "shared column order")
        for row, old in zip(report["responses"], reference["responses"], strict=True):
            exact([row[k] for k in keys], [old[k] for k in keys], "shared query/truth/group order")


def pooled_rows(reports):
    aligned_reports(reports)
    rows = [row for report in reports for row in report["responses"]]
    return {
        "aggregate": BASE.totals(rows),
        "groups": {g: BASE.totals([r for r in rows if r["group"] == g]) for g in GROUPS},
    }


def accepted_chain(directory, summary_sha, audit_sha, decision_sha):
    summary = read(directory / "summary.json", summary_sha)
    audit = read(directory / "independent-audit.json", audit_sha)
    decision = read(directory / "decision.json", decision_sha)
    require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == summary_sha
        and decision["summary_sha256"] == summary_sha
        and decision["audit_sha256"] == audit_sha
        and decision["evidence_accepted"] is True,
        "accepted historical evidence chain",
    )
    manifest(summary["input_sha256"])
    return summary, audit, decision


def historical_context():
    directory = ROOT / "runs/learning/bilinear-budget2000-v1"
    summary, audit, decision = accepted_chain(
        directory, BUDGET_SUMMARY_SHA, BUDGET_AUDIT_SHA, BUDGET_DECISION_SHA
    )
    require(
        decision["fixed_quality_gate_passed"] is True
        and audit["script_sha256"] == BUDGET_AUDITOR_SHA
        and summary["script_sha256"] == BUDGET_RUNNER_SHA,
        "accepted historical selection-seed result",
    )
    artifacts = {
        name: read(directory / name, summary["artifact_sha256"][name])
        for name in (
            "child.json",
            "parent.json",
            "training.json",
            "run.json",
            "train-membership.json",
        )
    }
    exact(
        artifacts["child.json"]["aggregate"],
        audit["child"]["aggregate"],
        "historical metric binding",
    )
    exact(artifacts["child.json"]["groups"], audit["child"]["groups"], "historical group binding")
    vocabulary_path = ROOT / "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json"
    vocabulary = read(vocabulary_path, summary["input_sha256"][str(vocabulary_path)])["tokens"]
    records_path = vocabulary_path.with_name("records.jsonl")
    bind(records_path, summary["input_sha256"][str(records_path)])
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    partitions, split = BASE.split_membership(records)
    require(
        split == BUDGET.SPLIT_SHA
        and {k: len(v) for k, v in partitions.items()}
        == {"train": 1637, "validation": 222, "test": 191},
        "shared split identity (not the model seed)",
    )
    wide_dir = ROOT / "runs/learning/wide-first-choice-v1"
    wide_summary, _, wide_decision = accepted_chain(
        wide_dir, WIDE_SUMMARY_SHA, WIDE_AUDIT_SHA, WIDE_DECISION_SHA
    )
    require(wide_decision["evidence_accepted"] is True, "width8 accepted evidence")
    return {
        "summary": summary,
        "audit": audit,
        "artifacts": artifacts,
        "partitions": partitions,
        "vocabulary": vocabulary,
        "wide_summary": wide_summary,
    }


def seed_context(seed, historical):
    seed_contract(seed)
    mapping = PINS[seed]
    wide_summary = historical["wide_summary"]
    entries = [r for r in wide_summary["reports"] if r["seed"] == seed]
    require(
        len(entries) == 1 and entries[0]["sha256"] == mapping["wide"], "matching width8 mapping"
    )
    wide = read(entries[0]["path"], mapping["wide"])
    require(
        wide["seed"] == seed
        and wide["complete"] is True
        and wide["checkpoint_hash"] == mapping["parent"]
        and wide["split_hash"] == BUDGET.SPLIT_SHA,
        "matching seed/parent/split reference",
    )
    exact(wide["product_token_ids"], list(range(1024, 2049)), "width8 columns")
    folder = ROOT / f"runs/national_dex_continuation_control_s{seed}_v1"
    inherited = wide_summary["input_sha256"]
    run = read(folder / "run.json", inherited[str(folder / "run.json")])
    sidecar = read(
        folder / "checkpoint-final.pt.json", inherited[str(folder / "checkpoint-final.pt.json")]
    )
    result = read(folder / "training-result.json", inherited[str(folder / "training-result.json")])
    bind(folder / "checkpoint-final.pt", mapping["parent"])
    parent_config_contract(
        seed,
        run["config"],
        historical["artifacts"]["run.json"]["config"]["inherited_parent_config"],
    )
    require(
        run["identity"]["resolved_config_hash"] == mapping["config"]
        and run["identity"]["seeds"] == [seed]
        and result["checkpoint_hash"] == mapping["parent"]
        and result["global_step"] == 2000,
        "original training checkpoint identities",
    )
    exact(run["identity"], sidecar["experiment_identity"], "parent identity sidecar")
    exact(run["identity"], result["identity"], "parent training result identity")
    exact(run["identity"], wide["training_identity"], "matching width8 training identity")
    require(
        sidecar["checkpoint_hash"] == mapping["parent"]
        and sidecar["global_step"] == 2000
        and sidecar["split_hash"] == BUDGET.SPLIT_SHA,
        "parent sidecar payload/split binding",
    )
    original_parent_schema(run, sidecar, result)
    # Saved complete-width8 evidence authenticates the source paths; independently
    # recompute its set-quality floors without rerunning candidate selection.
    values = []
    for row, record in zip(wide["responses"], historical["partitions"]["validation"], strict=True):
        prompt, truth = BASE.query_labels(record, historical["vocabulary"])
        exact(row["prompt_ids"], prompt, "width8 query order")
        exact(row["expected_set_ids"], truth, "width8 truth")
        selected = row["composition"]["selected_set_ids"]
        values.append({"group": row["group"], "metrics": BASE.metrics(selected, truth)})
    require(len(values) == 222, "width8 coverage")
    baseline = {
        "exact_count": sum(r["metrics"]["exact"] for r in values),
        "f1": math.fsum(r["metrics"]["f1"] for r in values) / 222,
        "groups": {
            g: sum(r["metrics"]["exact"] for r in values if r["group"] == g) for g in GROUPS
        },
    }
    # Width8 stores ordinary sequential mean; use its declared exact aggregation
    # for the comparator identity, while checking each constituent independently.
    baseline["f1"] = sum(r["metrics"]["f1"] for r in values) / 222
    exact(
        baseline,
        {"exact_count": mapping["exact"], "f1": mapping["f1"], "groups": mapping["groups"]},
        "declared per-seed floors",
    )
    return {
        "seed": seed,
        "wide": wide,
        "baseline": baseline,
        "parent_folder": folder,
        "parent_run": run,
        "parent_sidecar": sidecar,
        "parent_result": result,
    }


def training_audit(summary, reports, recipe, context, historical, folder):
    seed = context["seed"]
    parent_path = context["parent_folder"] / "checkpoint-final.pt"
    parent_sidecar = context["parent_sidecar"]
    old_run = context["parent_run"]
    old_membership = historical["artifacts"]["train-membership.json"]
    partitions, vocabulary = historical["partitions"], historical["vocabulary"]
    parent_hash = PINS[seed]["parent"]
    training, run = reports["training.json"], reports["run.json"]
    require(
        training["complete"] is True
        and training["objective"] == OBJECTIVE
        and training["completed_updates"] == 2000,
        "complete new objective training",
    )
    require(
        not any(k in training for k in ("error", "failed_update", "failed_observation")),
        "retained training failure",
    )
    BUDGET.architecture_contract(training["architecture"])
    initial_loss = training["initial_training_replay"]["parent_loss"]
    training_replay_contract(
        training["initial_training_replay"],
        training["history"],
        training["initial_pre_update_loss"],
        training["final_post_update_loss"],
    )
    require(training["initial_loss_replay"] == initial_loss, "seed-specific initial replay")
    require(
        training["train_query_count"] == 1637
        and training["transformer_forwards_during_fit"] == 0
        and training["validation_labels_used"] is False,
        "train-only fit receipts",
    )
    exact(training["trainable_parameters"], [A], "sole trainable residual name")
    queries = {
        name: [BASE.query_descriptor(r) for r in partitions[name]]
        for name in ("train", "validation")
    }
    exact(summary["queries"], queries, "ordered supervised partitions")
    exact(summary["validation_update_points"], [0, 2000], "fixed endpoints only")
    require(
        summary["train_query_order_sha256"]
        == training["train_query_order_sha256"]
        == BASE.formatted_hash(queries["train"]),
        "train query order hash",
    )
    membership = {
        "partition": "train",
        "query_count": 1637,
        "queries": [
            {
                "index": index,
                **BASE.query_descriptor(record),
                "expected_set_ids": BASE.query_labels(record, vocabulary)[1],
            }
            for index, record in enumerate(partitions["train"])
        ],
    }
    exact(reports["train-membership.json"], membership, "independent train labels")
    exact(membership, old_membership, "old training membership identity")
    require(
        sha(folder / "train-membership.json") == BUDGET.MEMBERSHIP_SHA,
        "exact training membership bytes",
    )
    parent = load_payload(parent_path, parent_hash)
    checkpoint = folder / "checkpoint-final.pt"
    require(
        Path(training["checkpoint"]).resolve() == checkpoint
        and training["checkpoint_sha256"] == summary["artifact_sha256"]["checkpoint-final.pt"]
        and training["checkpoint_sidecar_sha256"]
        == summary["artifact_sha256"]["checkpoint-final.pt.json"],
        "child payload paths/hashes",
    )
    child = load_payload(checkpoint, training["checkpoint_sha256"])
    BASE.sidecar_matches(parent, parent_sidecar, parent_hash)
    BASE.sidecar_matches(child, reports["checkpoint-final.pt.json"], training["checkpoint_sha256"])
    lineage_contract(seed, child["training_metadata"], child["experiment_identity"])
    require(
        parent["global_step"] == 2000
        and type(child["global_step"]) is int
        and child["global_step"] == 2000,
        "parent/new update distinction",
    )
    parent_state, initial_state, final_state = BUDGET.compare_states(
        parent["model"], child["model"]
    )
    BUDGET.state_receipts_contract(training, parent_state, initial_state, final_state)
    require(training["optimizer"]["name"] == "adamw", "optimizer family")
    exact(training["optimizer"]["config"], recipe["optimizer"], "optimizer config recipe")
    impl = training["optimizer_implementation"]
    BUDGET.optimizer_contract(
        child["optimizer"],
        {
            "class": impl["module"] + "." + impl["class"],
            "groups": training["optimizer"]["groups"],
            "defaults": impl["defaults"],
        },
    )
    require(child["scheduler"] is None, "no new scheduler")
    config = {
        "seed": seed,
        "data_split_seed": 1729,
        "campaign_version": CAMPAIGN,
        "inherited_parent_config": old_run["config"],
        "bilinear_residual_architecture": ARCHITECTURE,
        "bilinear_residual_recipe": recipe,
        "implementation_identity": {
            "runner_sha256": RUNNER_SHA,
            "scorer_sha256": BUDGET.SCORER_SHA,
            "budget_helper_sha256": BUDGET_RUNNER_SHA,
            "mean_helper_sha256": BUDGET.MEAN_RUNNER_SHA,
            "source_archive_sha256": BUDGET.SOURCE_SHA,
            "configs_archive_sha256": BUDGET.CONFIGS_SHA,
        },
    }
    exact(run["config"], config, "architecture-explicit child config")
    exact(child["config"], config, "checkpoint child config")
    exact(
        summary["parent_training_identity"],
        parent["experiment_identity"],
        "original training identity",
    )
    exact(summary["corpus_identity"], parent["corpus_identity"], "corpus identity receipt")
    exact(
        summary["inherited_architecture"],
        parent["training_metadata"]["model_config"],
        "base architecture kept separately",
    )
    metadata = {
        "seed": seed,
        "data_split_seed": 1729,
        "campaign_version": CAMPAIGN,
        "objective": OBJECTIVE,
        "evaluator": EVALUATOR,
        "architecture": ARCHITECTURE,
        "model_config": parent["training_metadata"]["model_config"],
        "model_config_scope": "inherited archived base only; residual architecture is separate",
        "parent_checkpoint_sha256": parent_hash,
        "parent_training_steps": 2000,
        "residual_updates": 2000,
        "trainable_parameters": [A],
        "train_query_order_sha256": summary["train_query_order_sha256"],
        "record_count": 1637,
        "recipe_sha256": RECIPE_SHA,
        "runner_sha256": RUNNER_SHA,
        "scorer_sha256": BUDGET.SCORER_SHA,
        "budget_helper_sha256": BUDGET_RUNNER_SHA,
        "parent_state_sha256": BASE.formatted_hash(parent_state),
        "initial_state_sha256": BASE.formatted_hash(initial_state),
        "initial_pre_update_loss": initial_loss,
        "final_post_update_loss": training["final_post_update_loss"],
        "standard_serving_supported": False,
        "implementation_identity": config["implementation_identity"],
    }
    exact(run["training_metadata"], metadata, "authoritative residual metadata")
    exact(child["training_metadata"], metadata, "checkpoint metadata/endpoints/source binding")
    identity = {
        **parent["experiment_identity"],
        "source_commit": summary["workspace_provenance"]["commit"],
        "resolved_config_hash": BASE.formatted_hash(config),
        "evaluator_version": EVALUATOR,
        "environment": {
            **summary["environment"],
            "source_archive_sha256": BUDGET.SOURCE_SHA,
            "configs_archive_sha256": BUDGET.CONFIGS_SHA,
            "runner_sha256": RUNNER_SHA,
            "scorer_sha256": BUDGET.SCORER_SHA,
            "budget_helper_sha256": BUDGET_RUNNER_SHA,
            "recipe_sha256": RECIPE_SHA,
            "mean_helper_sha256": BUDGET.MEAN_RUNNER_SHA,
            "architecture": BUDGET.ARCHITECTURE,
        },
        "checkpoint_hash": None,
        "seeds": [seed],
        "repetitions": 1,
    }
    for stored in (
        run["identity"],
        child["experiment_identity"],
        training["training_identity"],
        summary["child_training_identity"],
    ):
        exact(stored, identity, "new training identity")
    require(child["split_hash"] == summary["split_hash"] == BUDGET.SPLIT_SHA, "frozen child split")
    exact(child["corpus_identity"], parent["corpus_identity"], "child corpus unchanged")
    require(
        training["zero_replay_sha256"] == summary["artifact_sha256"]["zero-replay.json"],
        "zero initialization replay hash",
    )
    exact(training["zero_replay"], reports["zero-replay.json"], "zero replay nested receipt")
    finite(training["refit_seconds"])
    require(training["refit_seconds"] >= 0, "descriptive refit time")
    return {
        "seed": seed,
        "parent_checkpoint_sha256": parent_hash,
        "child_checkpoint_sha256": training["checkpoint_sha256"],
        "child_sidecar_sha256": training["checkpoint_sidecar_sha256"],
        "architecture": ARCHITECTURE,
        "global_step": 2000,
        "parent_training_steps": 2000,
        "original_tensor_count": 93,
        "child_tensor_count": 94,
        "unchanged_original_tensor_count": 93,
        "new_trained_tensor": A,
        "initial_residual": initial_state[A],
        "trained_residual": final_state[A],
        "train_query_count": 1637,
        "initial_pre_update_loss": initial_loss,
        "initial_training_replay": training["initial_training_replay"],
        "final_post_update_loss": training["final_post_update_loss"],
        "updates_verified": 2000,
        "optimizer_step": 2000,
        "refit_seconds": training["refit_seconds"],
        "payload_loaded_on": "cpu",
        "scorer_sha256": BUDGET.SCORER_SHA,
        "budget_helper_sha256": BUDGET_RUNNER_SHA,
        "runner_sha256": RUNNER_SHA,
    }


def provenance(summary, folder, historical, context, *, aggregate=False):
    prefix = "aggregate-v2" if aggregate else "summary"
    inherited = historical["summary"]
    seed = context["seed"]
    require(summary["report_kind"] == ("aggregate" if aggregate else "fresh_seed"), "report kind")
    if not aggregate:
        require(
            summary["seed"] == summary["model_seed"] == seed and summary["data_split_seed"] == 1729,
            "report seed/split binding",
        )
        for key, value in (
            ("parent_checkpoint_sha256", PINS[seed]["parent"]),
            ("parent_config_sha256", PINS[seed]["config"]),
            ("width8_sha256", PINS[seed]["wide"]),
            ("comparator_floors", context["baseline"]),
        ):
            exact(summary[key], value, "report matching-seed identity")
    require(
        summary["campaign_version"] == CAMPAIGN
        and summary["complete"] is True
        and summary["final_identity_check"] is True
        and "error" not in summary,
        "complete bilinear campaign",
    )
    require(
        summary["acceptance"] is False
        and summary["protected_test_used"] is False
        and summary["standard_serving_supported"] is False,
        "acceptance/protected/serving boundary",
    )
    for key, digest in (
        ("plan_sha256", PLAN_SHA),
        ("script_sha256", RUNNER_SHA),
        ("recipe_sha256", RECIPE_SHA),
        ("mean_helper_sha256", BUDGET.MEAN_RUNNER_SHA),
        ("helper_sha256", BUDGET.HELPER_SHA),
        ("budget_helper_sha256", BUDGET_RUNNER_SHA),
        ("historical1729_summary_sha256", BUDGET_SUMMARY_SHA),
        ("historical1729_audit_sha256", BUDGET_AUDIT_SHA),
        ("historical1729_decision_sha256", BUDGET_DECISION_SHA),
    ):
        require(summary[key] == digest, "frozen primary identity: " + key)
    require(
        summary["objective"] == OBJECTIVE and summary["evaluator"] == EVALUATOR,
        "new objective/evaluator identity",
    )
    require(
        summary["scorer_sha256"] == BUDGET.SCORER_SHA,
        "unchanged frozen scorer separate from runner",
    )
    BUDGET.architecture_contract(summary["architecture"])
    manifest(summary["input_sha256"])
    manifest(summary["snapshot_sha256"])
    inputs = summary["input_sha256"]
    required = {
        PINS[seed]["parent"],
        PINS[seed]["wide"],
        BUDGET.SOURCE_SHA,
        BUDGET.CONFIGS_SHA,
        BUDGET_SUMMARY_SHA,
        BUDGET_AUDIT_SHA,
        BUDGET_DECISION_SHA,
        BUDGET_AUDITOR_SHA,
        PLAN_SHA,
        RUNNER_SHA,
        RECIPE_SHA,
        RUNNER_RECEIPT_SHA,
        BUDGET.MEMBERSHIP_SHA,
        BUDGET.SCORER_SHA,
        BUDGET.MEAN_RUNNER_SHA,
        BUDGET_RUNNER_SHA,
    }
    require(required <= set(inputs.values()), "required consumed immutable identities")
    for path, digest in historical["wide_summary"]["input_sha256"].items():
        relative = Path(path).relative_to(ROOT)
        if (
            relative.parts[0] == "data"
            or f"national_dex_continuation_control_s{seed}_v1" in relative.parts
        ):
            require(inputs.get(path) == digest, "original exact data/checkpoint input binding")
    snapshots = {
        "script.py",
        "mean-helper.py",
        "helper.py",
        "plan.md",
        "recipe.json",
        "source.zip",
        "configs.zip",
        "test-receipt.json",
        "test-stdout.txt",
        "inputs.json",
        "bilinear-helper.py",
        "budget-helper.py",
    }
    if aggregate:
        snapshots.update(
            {
                "aggregation-script.py",
                "aggregation-test-receipt.json",
                "aggregation-test-stdout.txt",
            }
        )
    require(
        set(summary["snapshot_sha256"])
        == {str(folder / (prefix + "." + suffix)) for suffix in snapshots},
        "snapshot inventory",
    )
    for suffix, digest in (
        ("script.py", RUNNER_SHA),
        ("mean-helper.py", BUDGET.MEAN_RUNNER_SHA),
        ("helper.py", BUDGET.HELPER_SHA),
        ("plan.md", PLAN_SHA),
        ("recipe.json", RECIPE_SHA),
        ("test-receipt.json", RUNNER_RECEIPT_SHA),
        ("bilinear-helper.py", BUDGET.SCORER_SHA),
        ("budget-helper.py", BUDGET_RUNNER_SHA),
        ("source.zip", BUDGET.SOURCE_SHA),
        ("configs.zip", BUDGET.CONFIGS_SHA),
    ):
        bind(folder / (prefix + "." + suffix), digest)
    initial_inputs = read(folder / (prefix + ".inputs.json"))
    if aggregate:
        aggregate_inventory_contract(summary["preflight_input_sha256"], initial_inputs, inputs)
        path = Path(summary["aggregate_inputs_path"])
        require(
            path == folder / (prefix + ".aggregate-inputs.json"), "aggregate dynamic inventory path"
        )
        exact(read(path, summary["aggregate_inputs_sha256"]), inputs, "dynamic aggregate inputs")
    else:
        exact(initial_inputs, inputs, "input manifest snapshot")
    recipe = read(folder / (prefix + ".recipe.json"), RECIPE_SHA)
    recipe_contract(
        recipe,
        read(ROOT / "runs/learning/bilinear-budget2000-v1/summary.recipe.json", BUDGET.RECIPE_SHA),
    )
    receipt = read(folder / (prefix + ".test-receipt.json"), RUNNER_RECEIPT_SHA)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["experiment_training_executed"] is False
        and receipt["tested_script_sha256"] == RUNNER_SHA
        and receipt["tested_recipe_sha256"] == RECIPE_SHA
        and receipt["tested_mean_helper_sha256"] == BUDGET.MEAN_RUNNER_SHA
        and receipt["tested_scorer_sha256"] == BUDGET.SCORER_SHA
        and receipt["tested_budget_helper_sha256"] == BUDGET_RUNNER_SHA
        and receipt["test_count"] == 136
        and receipt["skipped"] == 0,
        "frozen primary synthetic receipt",
    )
    for path, digest in (
        (receipt["stdout"]["path"], receipt["stdout"]["sha256"]),
        (receipt["test_source"], receipt["tested_test_sha256"]),
    ):
        bind(path, digest)
        require(inputs.get(str(Path(path).resolve())) == digest, "test input binding")
    bind(folder / (prefix + ".test-stdout.txt"), receipt["stdout"]["sha256"])
    fixture = ROOT / "tests/fixtures/projection_refit_runtime_source.zip"
    exact(
        receipt["fixtures"], {str(fixture): BUDGET.SOURCE_SHA}, "portable archived runtime fixture"
    )
    require(inputs.get(str(fixture)) == BUDGET.SOURCE_SHA, "fixture input binding")
    bind(fixture, BUDGET.SOURCE_SHA)
    for path, digest in receipt["test_dependencies"].items():
        bind(path, digest)
        require(inputs.get(path) == digest, "shared tested dependency binding")
    if not aggregate:
        runtime = folder / "runtime"
        source = archive_inventory(folder / (prefix + ".source.zip"), BUDGET.SOURCE_SHA, runtime)
        configs = archive_inventory(folder / (prefix + ".configs.zip"), BUDGET.CONFIGS_SHA, runtime)
        require(not (set(source) & set(configs)), "disjoint source/config archives")
        inventory = {str(runtime / name): digest for name, digest in (source | configs).items()}
        exact(summary["runtime_files_sha256"], inventory, "frozen runtime inventory")
        require(
            {str(p) for p in runtime.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
            == set(inventory),
            "no extra/missing runtime files",
        )
        origins = summary["module_origins"]
        require(
            {"plm", "plm.training.optim", "plm.training.checkpoint", "plm.model.layers"}
            <= set(origins),
            "required runtime origins",
        )
        for name, entry in origins.items():
            path = Path(entry["path"])
            require(name == "plm" or name.startswith("plm."), "runtime namespace")
            require(
                path.is_relative_to(runtime / "src")
                and inventory.get(str(path)) == entry["sha256"],
                "runtime module source",
            )
            module = name.replace(".", "/")
            require(
                path.relative_to(runtime / "src").as_posix()
                in (module + ".py", module + "/__init__.py"),
                "module file binding",
            )
    exact(summary["environment"], inherited["environment"], "parent dependency/device environment")
    exact(summary["numerical_settings"], inherited["numerical_settings"], "fixed FP32 settings")
    if not aggregate:
        workspace = summary["workspace_provenance"]
        require(
            len(workspace["commit"]) == 40
            and hashlib.sha256(workspace["status_porcelain"].encode()).hexdigest()
            == workspace["status_sha256"]
            and len(workspace["tracked_diff_sha256"]) == 64,
            "workspace source receipts",
        )
        finite(summary["wall_seconds"])
        require(
            summary["wall_seconds"] >= 0
            and all(type(v) is int and v > 0 for v in summary["peak_memory"].values()),
            "descriptive time/memory",
        )
        artifacts = {
            "parent.json",
            "zero-replay.json",
            "child.json",
            "training.json",
            "train-membership.json",
            "run.json",
            "checkpoint-final.pt",
            "checkpoint-final.pt.json",
        }
        require(
            set(summary["artifact_sha256"]) == artifacts, "complete experiment artifact inventory"
        )
        for name, digest in summary["artifact_sha256"].items():
            bind(folder / name, digest)
    if aggregate:
        recovery_provenance(summary, folder, initial_inputs)
    return recipe


def recovery_provenance(summary, folder, preflight):
    require(
        summary["aggregation_runner_sha256"] == AGGREGATION_RUNNER_SHA
        and summary["aggregation_revision"] == "aware-time-v2",
        "separate frozen aggregation implementation",
    )
    for key, expected in (
        ("failed_aggregate_summary_sha256", FAILED_AGGREGATE_SHA),
        ("failed_aggregate_stdout_sha256", FAILED_AGGREGATE_STDOUT_SHA),
        ("failed_aggregate_execution_receipt_sha256", FAILED_AGGREGATE_RECEIPT_SHA),
    ):
        require(summary[key] == expected, "retained failed aggregate identity")
    failed = read(folder / "summary.json", FAILED_AGGREGATE_SHA)
    receipt = read(folder / "execution-receipt.json", FAILED_AGGREGATE_RECEIPT_SHA)
    bind(folder / "primary-stdout.txt", FAILED_AGGREGATE_STDOUT_SHA)
    require(
        failed["complete"] is False
        and "launch/terminal chronology binding" in failed["error"]
        and receipt["exit_code"] == 1
        and receipt["terminal_completion_observed_by_primary"] is True
        and receipt["summary_sha256"] == FAILED_AGGREGATE_SHA
        and receipt["stdout_sha256"] == FAILED_AGGREGATE_STDOUT_SHA,
        "retained failed chronology processing attempt",
    )
    old_snapshot = read(
        folder / "summary.inputs.json",
        failed["snapshot_sha256"][str(folder / "summary.inputs.json")],
    )
    require(
        all(preflight.get(p) == h for p, h in old_snapshot.items()),
        "old preflight identities preserved",
    )
    source = bind(folder / "aggregate-v2.aggregation-script.py", AGGREGATION_RUNNER_SHA)
    tests = read(folder / "aggregate-v2.aggregation-test-receipt.json", AGGREGATION_RECEIPT_SHA)
    require(
        tests["passed"] is True
        and tests["gpu_used"] is False
        and tests["experiment_training_executed"] is False
        and tests["tested_script_sha256"] == AGGREGATION_RUNNER_SHA
        and tests["frozen_runner_sha256"] == RUNNER_SHA,
        "frozen aggregation-only test receipt",
    )
    bind(folder / "aggregate-v2.aggregation-test-stdout.txt", tests["stdout"]["sha256"])
    for path, digest in tests["test_dependencies"].items():
        bind(path, digest)
        require(preflight.get(path) == digest, "aggregation tested source input")
    bind(tests["stdout"]["path"], tests["stdout"]["sha256"])
    require(
        preflight.get(str(ROOT / "scripts/aggregate_bilinear_replication.py")) == sha(source)
        and REPAIR_PLAN_SHA in preflight.values()
        and AGGREGATION_RECEIPT_SHA in preflight.values(),
        "aggregation/declaration inputs",
    )


def archive_inventory(path, digest, destination):
    bind(path, digest)
    result = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            name = entry.filename
            require(
                entry.orig_filename == name
                and not entry.is_dir()
                and all(p not in ("", ".", "..") for p in name.split("/"))
                and "\\" not in name
                and ":" not in name
                and name.casefold() not in {k.casefold() for k in result},
                "safe unique archive paths",
            )
            require((entry.external_attr >> 16) & 0o170000 != 0o120000, "archive symlink")
            result[name] = hashlib.sha256(archive.read(name)).hexdigest()
            bind(destination / name, result[name])
    return result


def terminal_contract(receipt, seed, summary_sha, stdout_sha, previous=None):
    seed_contract(seed)
    require(
        receipt["seed"] == seed
        and receipt["execution_order"] == seed - 1729
        and type(receipt["exit_code"]) is int
        and receipt["exit_code"] == 0
        and receipt["terminal_completion_observed_by_primary"] is True
        and receipt["summary_sha256"] == summary_sha
        and receipt["stdout_sha256"] == stdout_sha
        and receipt["previous_terminal_receipt_sha256"] == previous,
        "observed terminal completion and predecessor binding",
    )
    start, finish = (
        datetime.fromisoformat(receipt[k]) for k in ("started_at_utc", "finished_at_utc")
    )
    require(
        start.utcoffset() is not None and finish.utcoffset() is not None and finish >= start,
        "aware ordered terminal timestamps",
    )
    require(
        iso_instant(receipt["finished_at_utc"]) >= iso_instant(receipt["started_at_utc"]),
        "exact fractional terminal ordering",
    )
    return start, finish


def iso_instant(value):
    """Exact UTC seconds, retaining PowerShell's seventh fractional digit."""
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})",
        value,
    )
    require(match is not None, "explicit offset timestamp")
    whole, digits, offset = match.groups()
    instant = datetime.fromisoformat(whole + offset)
    delta = instant - datetime.fromisoformat("1970-01-01T00:00:00+00:00")
    fraction = Fraction(int(digits), 10 ** len(digits)) if digits else Fraction(0)
    return Fraction(delta.days * 86400 + delta.seconds) + fraction


def same_instant(left, right):
    first, second = datetime.fromisoformat(left), datetime.fromisoformat(right)
    require(
        first.utcoffset() is not None and second.utcoffset() is not None,
        "timezone-aware launch and receipt timestamps",
    )
    require(
        first == second and iso_instant(left) == iso_instant(right), "launch/receipt same instant"
    )


def execution_receipt(folder, seed, summary_sha):
    path = folder / "execution-receipt.json"
    receipt = read(path)
    stdout = bind(folder / "primary-stdout.txt", receipt["stdout_sha256"])
    previous = None
    if seed == 1731:
        predecessor_path = folder.parent / "seed-1730/execution-receipt.json"
        predecessor = read(predecessor_path)
        previous = sha(predecessor_path)
        old_summary = folder.parent / "seed-1730/summary.json"
        old_stdout = bind(
            folder.parent / "seed-1730/primary-stdout.txt", predecessor["stdout_sha256"]
        )
        _, previous_finish = terminal_contract(predecessor, 1730, sha(old_summary), sha(old_stdout))
    start, _finish = terminal_contract(receipt, seed, summary_sha, sha(stdout), previous)
    if seed == 1731:
        require(start >= previous_finish, "fresh GPU processes do not overlap")
        require(
            iso_instant(receipt["started_at_utc"]) >= iso_instant(predecessor["finished_at_utc"]),
            "exact fractional process nonoverlap",
        )
    launch = folder / "launch.json"
    if launch.exists():
        value = read(launch)
        require(value["seed"] == seed, "launch versus terminal seed")
        same_instant(value["started_at_utc"], receipt["started_at_utc"])
    return {
        "path": str(path),
        "sha256": sha(path),
        "started_at_utc": receipt["started_at_utc"],
        "finished_at_utc": receipt["finished_at_utc"],
        "exit_code": 0,
    }


def own_receipt():
    folder = ROOT / "runs/learning/bilinear-seed-replication-auditor-tests-v2"
    path = folder / "test-receipt.json"
    receipt = read(path)
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["model_execution"] is False
        and receipt["new_diagnostic_executed"] is False
        and receipt["tested_auditor_sha256"] == sha(__file__),
        "frozen independent synthetic receipt",
    )
    bind(folder / "test_auditor.py", receipt["tests_sha256"])
    bind(folder / "stdout.txt", receipt["stdout_sha256"])
    return sha(path)


def audit_header(summary_path):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment")
    bind(ROOT / "runs/learning/bilinear-budget2000-v1/independent-audit.py", BUDGET_AUDITOR_SHA)
    bind(
        ROOT / "runs/learning/bilinear-residual-refit-v1/independent-audit.py",
        BUDGET.BILINEAR_AUDITOR_SHA,
    )
    bind(
        ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py",
        BUDGET.AUDITOR_HELPER_SHA,
    )
    bind(HERE / "independent-audit.py", V1_SHA)
    bind(
        ROOT / "docs/experiments/2026-09-25-bilinear-replication-evidence-repair.md",
        REPAIR_PLAN_SHA,
    )
    failure_dir = ROOT / "runs/learning/bilinear-seed-replication-auditor-tests-v1"
    failed = read(failure_dir / "seed-1730-actual-audit-receipt.json", V1_FAILED_RECEIPT_SHA)
    bind(failure_dir / "seed-1730-actual-stdout.txt", V1_FAILED_STDOUT_SHA)
    require(
        failed["exit_code"] == 1
        and failed["audit_sha256"] is None
        and failed["auditor_sha256"] == V1_SHA
        and failed["stdout"]["sha256"] == V1_FAILED_STDOUT_SHA,
        "retained original failed audit attempt",
    )
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(summary_path),
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN_SHA,
        "test_receipt_sha256": own_receipt(),
        "audit_version": 2,
        "repair_plan_sha256": REPAIR_PLAN_SHA,
        "auditor_repair": {
            "previous_auditor_sha256": V1_SHA,
            "failed_attempt_receipt_sha256": V1_FAILED_RECEIPT_SHA,
            "failed_attempt_stdout_sha256": V1_FAILED_STDOUT_SHA,
            "reason": "Original run metadata belongs to its checkpoint sidecar; "
            "launch and terminal timestamps are compared as aware instants.",
            "prediction_and_gate_arithmetic_changed": False,
            "primary_artifacts_changed": False,
        },
        "limitations": [
            "Independent saved-set arithmetic and CPU payload checks; no training replay.",
            "Full-training loss/logit equality and finite trajectory are authenticated receipts; "
            "they are not independently regenerated.",
            "Reload and terminal chronology are bound owner observations, not execution replay.",
            "666 observations share 222 queries; 1729 is historical selection evidence.",
            "No oracle-parity, protected-quality, serving-support or durable-backup claim.",
        ],
    }


def finalize(result):
    require(BUDGET.INPUTS == BUDGET.OLD.INPUTS == BASE.INPUTS == {}, "no helper global mutation")
    manifest(dict(INPUTS))
    result["input_sha256"] = dict(INPUTS)
    return result


def audit_seed(summary_path, seed):
    seed_contract(seed)
    require(summary_path.parent == HERE / f"seed-{seed}", "per-seed campaign directory")
    result = audit_header(summary_path)
    summary = read(summary_path)
    historical = historical_context()
    context = seed_context(seed, historical)
    recipe = provenance(summary, summary_path.parent, historical, context)
    reports = {
        name: read(summary_path.parent / name, digest)
        for name, digest in summary["artifact_sha256"].items()
        if name.endswith(".json")
    }
    parent, child = reports["parent.json"], reports["child.json"]
    BASE.evaluation(
        parent,
        "parent",
        context["wide"],
        historical["partitions"]["validation"],
        historical["vocabulary"],
    )
    BASE.evaluation(
        child,
        "child",
        context["wide"],
        historical["partitions"]["validation"],
        historical["vocabulary"],
        parent,
    )
    aligned_reports([child, historical["artifacts"]["child.json"]])
    zero = BUDGET.zero_replay_contract(reports["zero-replay.json"], parent)
    checkpoint = training_audit(summary, reports, recipe, context, historical, summary_path.parent)
    names = (
        "parent_replay_exact",
        "zero_A_replay_exact",
        "initial_train_loss_exact",
        "completed_2000_updates",
        "original_93_tensors_unchanged",
        "residual_changed",
        "reload_exact",
        "optimizer_reload_exact",
        "train_only_supervision",
    )
    invariants = dict.fromkeys(names, True)
    exact(summary["invariants"], invariants, "independently checked execution invariants")
    gate = seed_gate(child, context["baseline"], invariants)
    exact(summary["gate"], gate, "matching-seed independent gate")
    exact(summary["parent"], parent["aggregate"], "summary parent aggregate")
    child_summary = {
        k: child[k] for k in ("aggregate", "groups", "paired_vs_parent_dense", "paired_vs_width8")
    }
    exact(summary["child"], child_summary, "summary child aggregate/comparisons")
    result.update(
        seed=seed,
        parent={k: parent[k] for k in ("aggregate", "groups", "paired_vs_width8")},
        child=child_summary,
        checkpoint=checkpoint,
        gate=gate,
        comparator_floors=context["baseline"],
        zero_initialization_replay=zero,
        comparisons={
            "versus_parent_dense": child["paired_vs_parent_dense"],
            "versus_width8": child["paired_vs_width8"],
        },
        execution_receipt=execution_receipt(summary_path.parent, seed, sha(summary_path)),
    )
    return finalize(result)


def campaign_gate(gates):
    fresh_conjunction(gates)
    checks = {"historical1729_accepted": True, **{f"fresh_seed{s}": gates[s] for s in SEEDS}}
    return {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "independent_audit_required": True,
        "accepted": False,
    }


def audit_aggregate(summary_path):
    require(summary_path == HERE / "aggregate-v2.json", "repaired aggregate campaign path")
    result = audit_header(summary_path)
    summary = read(summary_path)
    historical = historical_context()
    contexts = {seed: seed_context(seed, historical) for seed in SEEDS}
    provenance(summary, HERE, historical, contexts[1730], aggregate=True)
    require(
        [r["seed"] for r in summary["reports"]] == list(SEEDS),
        "ordered exact fresh report coverage",
    )
    children, records, gates = [], [], {}
    for entry in summary["reports"]:
        seed = entry["seed"]
        folder = HERE / f"seed-{seed}"
        require(
            Path(entry["path"]) == folder / "summary.json" and entry["role"] == "fresh replication",
            "fresh report path/role",
        )
        current = read(entry["path"], entry["sha256"])
        audit_path = folder / "independent-audit.json"
        checked = read(audit_path)
        require(
            checked["complete"] is True
            and checked["audit_passed"] is True
            and checked["acceptance"] is False
            and checked["seed"] == seed
            and checked["summary_sha256"] == entry["sha256"]
            and checked["script_sha256"] == sha(__file__)
            and checked["plan_sha256"] == PLAN_SHA,
            "completed exact-source per-seed audit required",
        )
        manifest(checked["input_sha256"])
        terminal = execution_receipt(folder, seed, entry["sha256"])
        require(
            entry["execution_receipt"] == terminal["path"]
            and entry["execution_receipt_sha256"] == terminal["sha256"],
            "aggregate terminal receipt binding",
        )
        child = read(folder / "child.json", current["artifact_sha256"]["child.json"])
        exact(current["child"], checked["child"], "audited child receipt")
        exact(current["gate"], checked["gate"], "audited fresh gate")
        exact(
            current["environment"], historical["summary"]["environment"], "same runtime environment"
        )
        exact(
            current["numerical_settings"],
            historical["summary"]["numerical_settings"],
            "same numerical settings",
        )
        children.append(child)
        gates[seed] = checked["gate"]["primary_checks_passed"]
        records.append(
            {
                "seed": seed,
                "summary_sha256": entry["sha256"],
                "audit_sha256": sha(audit_path),
                **{k: checked[k] for k in ("checkpoint", "child", "comparisons", "gate")},
            }
        )
    old = historical["artifacts"]["child.json"]
    # Rebuild pooled arithmetic from all rows, never from rounded per-seed means.
    fresh, all_three = pooled_rows(children), pooled_rows([old, *children])
    exact(summary["fresh_two"], fresh, "fresh-only independently pooled arithmetic")
    exact(summary["all_three"], all_three, "historical plus fresh independently pooled arithmetic")
    historical_record = {
        "role": "historical development/selection seed",
        "fresh_execution": False,
        "summary_sha256": BUDGET_SUMMARY_SHA,
        "audit_sha256": BUDGET_AUDIT_SHA,
        "decision_sha256": BUDGET_DECISION_SHA,
        "child_sha256": historical["summary"]["artifact_sha256"]["child.json"],
        "aggregate": old["aggregate"],
        "groups": old["groups"],
    }
    exact(summary["historical1729"], historical_record, "historical reuse evidence and role")
    scope = {
        "fresh_seed_order": [1730, 1731],
        "fresh_query_seed_observations": 444,
        "all_query_seed_observations": 666,
        "shared_validation_queries": 222,
        "new_checkpoints": 2,
        "historical_seed_retrained": False,
    }
    exact(summary["observation_scope"], scope, "replication versus shared observations")
    gate = campaign_gate(gates)
    exact(summary["gate"], gate, "conjunctive fresh replication gate")
    execution_path = HERE / "aggregate-v2-execution-receipt.json"
    execution = read(execution_path)
    stdout = bind(HERE / "aggregate-v2-stdout.txt", execution["stdout_sha256"])
    require(
        type(execution["exit_code"]) is int
        and execution["exit_code"] == 0
        and execution["terminal_completion_observed_by_primary"] is True
        and execution["summary_sha256"] == sha(summary_path),
        "observed recovered aggregate terminal completion",
    )
    result.update(
        per_seed=records,
        historical1729=historical_record,
        fresh_two=fresh,
        all_three=all_three,
        observation_scope=scope,
        gate=gate,
        aggregation_repair={
            "runner_sha256": AGGREGATION_RUNNER_SHA,
            "test_receipt_sha256": AGGREGATION_RECEIPT_SHA,
            "execution_receipt_sha256": sha(execution_path),
            "stdout_sha256": sha(stdout),
            "failed_summary_sha256": FAILED_AGGREGATE_SHA,
            "failed_stdout_sha256": FAILED_AGGREGATE_STDOUT_SHA,
            "failed_execution_receipt_sha256": FAILED_AGGREGATE_RECEIPT_SHA,
        },
    )
    return finalize(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int, choices=SEEDS)
    mode.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    path = args.summary.resolve()
    output = path.parent / "independent-audit.json"
    require(not output.exists(), "immutable audit output")
    result = audit_aggregate(path) if args.aggregate else audit_seed(path, args.seed)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"audit_passed": True, "audit_sha256": sha(output), "gate": result["gate"]},
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
