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
PLAN_SHA = "750d115b2fd0daf356cb3a633248e826c264f30533ddf8e8df2ee125d2ecdc6b"
BUDGET_AUDITOR_SHA = "c674cf8e4a3ab2056ca4a02746ee55ead639e2a10e6b208525cb0e9dc906d1dc"
BUDGET_SUMMARY_SHA = "c507162902e173964c6fec1446448a074ee8fed41f4acb3a4f6d46d6744907a3"
BUDGET_AUDIT_SHA = "475c50f91891ccd13c180472e2d019913b3e5432ab0a9d3b15162735711aac3d"
BUDGET_DECISION_SHA = "edaf01b58cd09ad4524123f5f817c64602d0414c4220e1c709487016cf2f89fb"
BUDGET_RUNNER_SHA = "f4376d4f2d838938bb9f50ae0cfa16ef319639fa0ce917c9fea3b842c0018930"
WIDE_SUMMARY_SHA = "1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183"
WIDE_AUDIT_SHA = "eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95"
WIDE_DECISION_SHA = "70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601"
RUNNER_SHA = "5da5ad02365f72f5e8a7123183d0b5d2f56e50c3538e64e2ac984e2b491230dd"
RECIPE_SHA = "20fad007be0d1852e86086229edc7c9c760117039f0685288d4e8e22f695075f"
RUNNER_RECEIPT_SHA = "9b01b18b15b00a98653d057f952bc8882b2d002d81b17a1cc20438e5dc1583a0"
EVALUATOR = "plm-bilinear-budget8000-replication-v1"
CAMPAIGN = "bilinear-budget8000-replication-v1"
SEEDS = (1730, 1731)
PINS = {
    1730: {
        "parent": "5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2",
        "config": "0eae5252ae9de2d1463992a319c43e695ae8bb4f24bfb2ae94acdf4da6033b31",
        "wide": "0a398f2adb53877a188bf72d444c460398b1ac21e2f4317f23bc2948c0dcb88a",
        "exact": 212,
        "f1": 0.9997931269673187,
        "groups": {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 58},
    },
    1731: {
        "parent": "ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d",
        "config": "9e029d5fa280492a382b3b138eb5933a8ef8580c6a95a90deb0c87420c3601b3",
        "wide": "a2004f3a8e98043f7e55fc775e0d33069a1a46a70b13f753a816a0d2ba2d7799",
        "exact": 208,
        "f1": 0.999668812076336,
        "groups": {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 54},
    },
}
INPUTS = {}
REPLICATION_AUDITOR_SHA = "462c76e5fb18b8acf3304c14dbe8924397852c14ff642080a50676338afda941"
SCREEN_AUDITOR_SHA = "9ecc66cd6f63b43d6fd13f0a29ec29ce785e4d0ad45b3f861a2e92f406bc7676"
SCREEN_SUMMARY_SHA = "e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8"
SCREEN_AUDIT_SHA = "bdf142f48229fbcdb197bce3a019e38c3a65570ae3f8551a9c9a1f14b2198872"
SCREEN_DECISION_SHA = "2cadf9dce7ee36cc4095a10a746cc1608d5ce4437724872c1eec1c88e915c195"
REPLICATION_RUNNER_SHA = "24fc43fc7d31ec1bf7aa84b475af79198717fbb6e34f41453939a3da49e9fdde"
SCREEN_RUNNER_SHA = "cd593c14c429247837c26b09754509d5cfd1dd36adb317f56522a80a78a25122"
REPLICATION_SUMMARY_SHA = "338fb6b7ea08128952ce9abc02b3506639e712a9cff6ce63468ed2e7980f79a0"
REPLICATION_AUDIT_SHA = "ddbecaceabdacc0808f9b4f122e0309aa3e233fe76b0d75694e31833a40083a8"
REPLICATION_DECISION_SHA = "62a8dcbb7cbe5d038984f9d57aecacec4b0ba612ba55c2a9f6fe335977304978"
AGGREGATION_RUNNER_SHA = "78cae68bdf741ec35c6deff6a62b57673092be0cf61968e88bbf40274878789f"
AGGREGATION_RECEIPT_SHA = "0dc5c87d41db7ae3e1a3d9cd591717d0941501df201e836abcad765ead2e3f36"
HISTORICAL_SEEDS = {
    1730: {
        "summary": "ecc49edfa4a7b7b2428326d752a020cf09ef92eefcb69157b12a3c4a340356e1",
        "training": "a7366f60a7164d8e31d1dccfb4484364cb68919f8e2449c7871138359f2d4fe5",
    },
    1731: {
        "summary": "3f99d91b87f4531941ae3af11b3f39a42746df15f86df461db7e06386ecefc9c",
        "training": "e7f7d4fd2cbfecc7d819b8d691c3a49c9fcd2c46e8d8bf80cf46833d61ad3512",
    },
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticated_helper(relative, digest, name):
    path = ROOT / relative
    if sha(path) != digest:
        raise ValueError("frozen independent helper changed: " + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLICATION = authenticated_helper(
    "runs/learning/bilinear-seed-replication-v1/independent-audit-v2.py",
    REPLICATION_AUDITOR_SHA,
    "independent_replication8000_prior",
)
SCREEN = authenticated_helper(
    "runs/learning/bilinear-budget8000-v1/independent-audit.py",
    SCREEN_AUDITOR_SHA,
    "independent_replication8000_screen",
)


BUDGET = SCREEN.BUDGET
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
        residual_updates=8000,
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
        and metadata["parent_training_steps"] == 2000
        and metadata["residual_updates"] == 8000
        and metadata["objective"] == OBJECTIVE
        and metadata["evaluator"] == EVALUATOR
        and identity["evaluator_version"] == EVALUATOR
        and metadata["record_count"] == 1637,
        "matching fresh seed/parent/new evaluator lineage",
    )
    exact(metadata["trainable_parameters"], [A], "sole residual trainable")


def loss_contract(history, parent_loss, zero_loss, initial, final):
    """Finite trajectory receipts with an independently measured parent endpoint."""
    require(len(history) == 8000, "exactly 8000 updates")
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
        "exact_above_matching2000": metrics["exact_count"] > baseline["exact_count"],
        "f1_at_least_matching2000": metrics["f1"] >= baseline["f1"],
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
    directory = ROOT / "runs/learning/bilinear-budget8000-v1"
    summary, audit, decision = accepted_chain(
        directory, SCREEN_SUMMARY_SHA, SCREEN_AUDIT_SHA, SCREEN_DECISION_SHA
    )
    require(
        decision["fixed_quality_gate_passed"] is True
        and audit["script_sha256"] == SCREEN_AUDITOR_SHA
        and summary["script_sha256"] == SCREEN_RUNNER_SHA,
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
        {
            "exact_count": REPLICATION.PINS[seed]["exact"],
            "f1": REPLICATION.PINS[seed]["f1"],
            "groups": REPLICATION.PINS[seed]["groups"],
        },
        "declared per-seed floors",
    )
    previous = matched_history(seed)
    aligned_reports([previous["child"], historical["artifacts"]["child.json"]])
    BASE.evaluation(
        previous["parent"],
        "parent",
        wide,
        historical["partitions"]["validation"],
        historical["vocabulary"],
    )
    BASE.evaluation(
        previous["child"],
        "child",
        wide,
        historical["partitions"]["validation"],
        historical["vocabulary"],
        previous["parent"],
    )
    matching = {
        "exact_count": previous["child"]["aggregate"]["exact_count"],
        "f1": previous["child"]["aggregate"]["f1"],
        "groups": {g: previous["child"]["groups"][g]["exact_count"] for g in GROUPS},
    }
    exact(
        matching,
        {"exact_count": mapping["exact"], "f1": mapping["f1"], "groups": mapping["groups"]},
        "seed-matched accepted 2000 floors",
    )
    return {
        "seed": seed,
        "wide": wide,
        "baseline": matching,
        "width8_baseline": baseline,
        "previous": previous,
        "parent_folder": folder,
        "parent_run": run,
        "parent_sidecar": sidecar,
        "parent_result": result,
    }


def matched_history(seed):
    """Authenticate the accepted repaired chain, without repeating its training audit."""
    seed_contract(seed)
    folder = ROOT / "runs/learning/bilinear-seed-replication-v1"
    summary = read(folder / "aggregate-v2.json", REPLICATION_SUMMARY_SHA)
    audit = read(folder / "independent-audit.json", REPLICATION_AUDIT_SHA)
    decision = read(folder / "decision.json", REPLICATION_DECISION_SHA)
    require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == REPLICATION_SUMMARY_SHA
        and decision["summary_sha256"] == REPLICATION_SUMMARY_SHA
        and decision["audit_sha256"] == REPLICATION_AUDIT_SHA
        and decision["evidence_accepted"] is True
        and decision["fixed_quality_gate_passed"] is True,
        "accepted matching-2000 chain",
    )
    entries = [r for r in audit["per_seed"] if r["seed"] == seed]
    require(
        len(entries) == 1 and entries[0]["summary_sha256"] == HISTORICAL_SEEDS[seed]["summary"],
        "accepted seed summary mapping",
    )
    directory = folder / f"seed-{seed}"
    own = read(directory / "summary.json", HISTORICAL_SEEDS[seed]["summary"])
    checked = read(directory / "independent-audit.json", entries[0]["audit_sha256"])
    require(
        own["complete"] is True
        and own["final_identity_check"] is True
        and checked["audit_passed"] is True
        and checked["complete"] is True
        and checked["seed"] == seed
        and checked["summary_sha256"] == HISTORICAL_SEEDS[seed]["summary"],
        "accepted seed evidence binding",
    )
    artifacts = {
        name: read(directory / name, own["artifact_sha256"][name])
        for name in ("child.json", "parent.json", "training.json")
    }
    require(
        own["artifact_sha256"]["training.json"] == HISTORICAL_SEEDS[seed]["training"],
        "matched historical trace",
    )
    exact(
        artifacts["child.json"]["aggregate"],
        checked["child"]["aggregate"],
        "historical aggregate audit",
    )
    exact(artifacts["child.json"]["groups"], checked["child"]["groups"], "historical group audit")
    require(
        artifacts["training.json"]["completed_updates"] == 2000
        and len(artifacts["training.json"]["history"]) == 2000,
        "historical budget",
    )
    return {
        "summary": own,
        "audit": checked,
        "child": artifacts["child.json"],
        "parent": artifacts["parent.json"],
        "training": artifacts["training.json"],
        "child_sha256": own["artifact_sha256"]["child.json"],
    }


def prefix_contract(training, historical):
    require(
        training["initial_pre_update_loss"] == historical["initial_pre_update_loss"],
        "matching historical initial loss replay",
    )
    expected = SCREEN.prefix_diagnostic(
        training["history"], historical["history"], historical["final_post_update_loss"]
    )
    exact(
        training["training_prefix_diagnostic"],
        expected,
        "independent descriptive prefix arithmetic",
    )
    return expected


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
        and training["completed_updates"] == 8000,
        "complete balanced objective training",
    )
    require(
        not any(k in training for k in ("error", "failed_update", "failed_observation")),
        "retained training failure",
    )
    BUDGET.architecture_contract(training["architecture"])
    prefix = prefix_contract(training, context["previous"]["training"])
    exact(summary["training_prefix_diagnostic"], prefix, "summary descriptive prefix")
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
    exact(summary["validation_update_points"], [0, 8000], "fixed endpoints only")
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
        and child["global_step"] == 8000,
        "parent/new update distinction",
    )
    parent_state, initial_state, final_state = BUDGET.compare_states(
        parent["model"], child["model"]
    )
    BUDGET.state_receipts_contract(training, parent_state, initial_state, final_state)
    require(training["optimizer"]["name"] == "adamw", "optimizer family")
    exact(training["optimizer"]["config"], recipe["optimizer"], "optimizer config recipe")
    impl = training["optimizer_implementation"]
    SCREEN.optimizer_contract(
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
            "replication_helper_sha256": REPLICATION_RUNNER_SHA,
            "budget8000_helper_sha256": SCREEN_RUNNER_SHA,
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
        "residual_updates": 8000,
        "trainable_parameters": [A],
        "train_query_order_sha256": summary["train_query_order_sha256"],
        "record_count": 1637,
        "recipe_sha256": RECIPE_SHA,
        "runner_sha256": RUNNER_SHA,
        "scorer_sha256": BUDGET.SCORER_SHA,
        "budget_helper_sha256": BUDGET_RUNNER_SHA,
        "replication_helper_sha256": REPLICATION_RUNNER_SHA,
        "budget8000_helper_sha256": SCREEN_RUNNER_SHA,
        "parent_state_sha256": BASE.formatted_hash(parent_state),
        "initial_state_sha256": BASE.formatted_hash(initial_state),
        "initial_pre_update_loss": initial_loss,
        "training_prefix_diagnostic": prefix,
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
            "replication_helper_sha256": REPLICATION_RUNNER_SHA,
            "budget8000_helper_sha256": SCREEN_RUNNER_SHA,
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
        "global_step": 8000,
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
        "training_prefix_diagnostic": prefix,
        "final_post_update_loss": training["final_post_update_loss"],
        "updates_verified": 8000,
        "optimizer_step": 8000,
        "refit_seconds": training["refit_seconds"],
        "payload_loaded_on": "cpu",
        "scorer_sha256": BUDGET.SCORER_SHA,
        "budget_helper_sha256": BUDGET_RUNNER_SHA,
        "replication_helper_sha256": REPLICATION_RUNNER_SHA,
        "budget8000_helper_sha256": SCREEN_RUNNER_SHA,
        "runner_sha256": RUNNER_SHA,
    }


def provenance(summary, folder, historical, context, *, aggregate=False):
    prefix = "aggregate" if aggregate else "summary"
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
    if not aggregate:
        expected_history = {
            "historical2000_summary_sha256": HISTORICAL_SEEDS[seed]["summary"],
            "historical2000_training_sha256": HISTORICAL_SEEDS[seed]["training"],
            "historical2000_child_sha256": context["previous"]["child_sha256"],
            "historical2000_aggregate_sha256": REPLICATION_SUMMARY_SHA,
            "historical2000_audit_sha256": REPLICATION_AUDIT_SHA,
            "historical2000_decision_sha256": REPLICATION_DECISION_SHA,
        }
        exact(
            {k: summary[k] for k in expected_history},
            expected_history,
            "matching accepted historical identities",
        )
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
        ("replication_helper_sha256", REPLICATION_RUNNER_SHA),
        ("budget8000_helper_sha256", SCREEN_RUNNER_SHA),
        ("historical1729_summary_sha256", SCREEN_SUMMARY_SHA),
        ("historical1729_audit_sha256", SCREEN_AUDIT_SHA),
        ("historical1729_decision_sha256", SCREEN_DECISION_SHA),
    ):
        require(summary[key] == digest, "frozen primary identity: " + key)
    require(
        summary["objective"] == OBJECTIVE and summary["evaluator"] == EVALUATOR,
        "unchanged objective/new evaluator identity",
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
        SCREEN_SUMMARY_SHA,
        SCREEN_AUDIT_SHA,
        SCREEN_DECISION_SHA,
        REPLICATION_SUMMARY_SHA,
        REPLICATION_AUDIT_SHA,
        REPLICATION_DECISION_SHA,
        REPLICATION_RUNNER_SHA,
        SCREEN_RUNNER_SHA,
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
        "replication-helper.py",
        "budget8000-helper.py",
    }
    if aggregate:
        snapshots.update(
            {
                "aggregation-script.py",
                "aggregation-test-receipt.json",
                "aggregation-test-stdout.txt",
                "auditor-script.py",
                "auditor-test-receipt.json",
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
        ("replication-helper.py", REPLICATION_RUNNER_SHA),
        ("budget8000-helper.py", SCREEN_RUNNER_SHA),
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
        and receipt["test_count"] == 221
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
    exact(
        set(receipt["test_dependencies"]),
        {str(ROOT / path) for path in SEED_TEST_DEPENDENCIES},
        "seed test dependency inventory",
    )
    for path, digest in receipt["test_dependencies"].items():
        bind(path, digest)
        require(inputs.get(path) == digest, "shared tested dependency binding")
    if aggregate:
        aggregate_provenance(summary, folder, initial_inputs)
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
            "comparisons.json",
        }
        require(
            set(summary["artifact_sha256"]) == artifacts, "complete experiment artifact inventory"
        )
        for name, digest in summary["artifact_sha256"].items():
            bind(folder / name, digest)
    return recipe


SEED_TEST_DEPENDENCIES = (
    "tests/unit/test_bilinear_budget8000_replication.py",
    "tests/unit/test_bilinear_replication.py",
    "tests/unit/test_bilinear_budget8000.py",
    "scripts/refit_bilinear_replication.py",
    "scripts/refit_bilinear_budget8000.py",
    "configs/experiments/bilinear_seed_replication_v1.json",
    "configs/experiments/bilinear_budget8000_v1.json",
    "tests/unit/test_bilinear_budget.py",
    "tests/unit/test_bilinear_residual.py",
    "tests/unit/test_projection_only_refit.py",
    "tests/fixtures/projection_refit_runtime_source.zip",
    "scripts/refit_bilinear_budget.py",
    "scripts/refit_bilinear_residual.py",
    "scripts/refit_membership_projection.py",
    "configs/experiments/bilinear_budget2000_v1.json",
    "configs/experiments/bilinear_residual_refit_v1.json",
    "configs/experiments/projection_only_refit_v1.json",
)


def aggregate_provenance(summary, folder, preflight):
    require(
        summary["aggregation_runner_sha256"] == AGGREGATION_RUNNER_SHA
        and summary["auditor_sha256"] == sha(__file__)
        and summary["auditor_test_receipt_sha256"] == own_receipt(),
        "separate aggregation and auditor identities",
    )
    bind(folder / "aggregate.aggregation-script.py", AGGREGATION_RUNNER_SHA)
    receipt = read(folder / "aggregate.aggregation-test-receipt.json", AGGREGATION_RECEIPT_SHA)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["experiment_training_executed"] is False
        and receipt["tested_script_sha256"] == AGGREGATION_RUNNER_SHA
        and receipt["tested_recipe_sha256"] == RECIPE_SHA
        and receipt["skipped"] == 0
        and receipt["test_count"] == 34,
        "frozen aggregate test receipt",
    )
    expected_dependencies = {
        str(ROOT / p)
        for p in (
            "tests/unit/test_bilinear_budget8000_replication_aggregate.py",
            "scripts/refit_bilinear_budget8000_replication.py",
            "tests/unit/test_bilinear_replication.py",
            "scripts/refit_membership_projection.py",
        )
    }
    exact(
        set(receipt["test_dependencies"]),
        expected_dependencies,
        "aggregation test dependency inventory",
    )
    for path, digest in receipt["test_dependencies"].items():
        bind(path, digest)
        require(preflight.get(path) == digest, "aggregation test dependency preflight binding")
    bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"])
    bind(folder / "aggregate.aggregation-test-stdout.txt", receipt["stdout"]["sha256"])
    bind(folder / "aggregate.auditor-script.py", sha(__file__))
    bind(folder / "aggregate.auditor-test-receipt.json", own_receipt())
    require(
        {AGGREGATION_RUNNER_SHA, AGGREGATION_RECEIPT_SHA, sha(__file__), own_receipt()}
        <= set(preflight.values()),
        "aggregation prerequisites consumed before execution",
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


def audit_receipt_contract(
    receipt, checked, *, seed, summary_sha, audit_sha, stdout_sha, auditor_sha, test_receipt_sha
):
    """Success of an evidence audit does not imply success of its quality gate."""
    seed_contract(seed)
    require(
        checked["complete"] is True
        and checked["audit_passed"] is True
        and checked["acceptance"] is False,
        "complete independent evidence audit",
    )
    require(
        checked["seed"] == seed
        and checked["summary_sha256"] == summary_sha
        and checked["script_sha256"] == auditor_sha
        and checked["test_receipt_sha256"] == test_receipt_sha
        and checked["plan_sha256"] == PLAN_SHA,
        "independent audit identities",
    )
    require(
        type(receipt["exit_code"]) is int
        and receipt["exit_code"] == 0
        and receipt["terminal_completion_observed_by_primary"] is True,
        "observed successful audit terminal",
    )
    expected = {
        "seed": seed,
        "summary_sha256": summary_sha,
        "audit_sha256": audit_sha,
        "stdout_sha256": stdout_sha,
        "auditor_sha256": auditor_sha,
        "test_receipt_sha256": test_receipt_sha,
    }
    exact({k: receipt[k] for k in expected}, expected, "exact audit receipt identities")


def own_receipt():
    folder = ROOT / "runs/learning/bilinear-budget8000-replication-auditor-tests-v1"
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
    require(
        all(
            "PENDING" not in value
            for value in (
                RUNNER_SHA,
                RECIPE_SHA,
                RUNNER_RECEIPT_SHA,
                AGGREGATION_RUNNER_SHA,
                AGGREGATION_RECEIPT_SHA,
            )
        ),
        "audit source not frozen",
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment")
    bind(ROOT / "runs/learning/bilinear-budget2000-v1/independent-audit.py", BUDGET_AUDITOR_SHA)
    bind(ROOT / "runs/learning/bilinear-budget8000-v1/independent-audit.py", SCREEN_AUDITOR_SHA)
    bind(
        ROOT / "runs/learning/bilinear-seed-replication-v1/independent-audit-v2.py",
        REPLICATION_AUDITOR_SHA,
    )
    bind(
        ROOT / "runs/learning/bilinear-residual-refit-v1/independent-audit.py",
        BUDGET.BILINEAR_AUDITOR_SHA,
    )
    bind(
        ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py",
        BUDGET.AUDITOR_HELPER_SHA,
    )
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(summary_path),
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN_SHA,
        "test_receipt_sha256": own_receipt(),
        "audit_version": 1,
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
    require(
        REPLICATION.INPUTS
        == SCREEN.INPUTS
        == BUDGET.INPUTS
        == BUDGET.OLD.INPUTS
        == BASE.INPUTS
        == {},
        "no helper global mutation",
    )
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
        "historical_initial_loss_exact",
        "completed_8000_updates",
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
    aligned_reports([child, context["previous"]["child"]])
    paired = BASE.paired(
        child["responses"],
        [r["selected_set_ids"] for r in context["previous"]["child"]["responses"]],
    )
    comparisons = {
        "historical2000_summary_sha256": HISTORICAL_SEEDS[seed]["summary"],
        "historical2000_child_sha256": context["previous"]["child_sha256"],
        "comparison_scope": "saved seed-matched historical output; no rerun",
        "paired_vs_historical2000": paired,
    }
    exact(reports["comparisons.json"], comparisons, "independent matching-2000 pairing")
    child_summary["paired_vs_historical2000"] = paired
    exact(summary["child"], child_summary, "summary child aggregate/comparisons")
    result.update(
        seed=seed,
        parent={k: parent[k] for k in ("aggregate", "groups", "paired_vs_width8")},
        child=child_summary,
        checkpoint=checkpoint,
        training_prefix_diagnostic=checkpoint["training_prefix_diagnostic"],
        gate=gate,
        comparator_floors=context["baseline"],
        zero_initialization_replay=zero,
        comparisons={
            "versus_parent_dense": child["paired_vs_parent_dense"],
            "versus_width8": child["paired_vs_width8"],
            "versus_historical2000": paired,
        },
        execution_receipt=execution_receipt(summary_path.parent, seed, sha(summary_path)),
    )
    return finalize(result)


def campaign_gate(gates):
    fresh_conjunction(gates)
    checks = {
        "historical1729_accepted": True,
        **{f"fresh_seed{s}": gates[s] for s in SEEDS},
        **{f"fresh_seed{s}_audited": True for s in SEEDS},
    }
    return {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "independent_audit_required": True,
        "accepted": False,
    }


def audit_aggregate(summary_path):
    require(summary_path == HERE / "aggregate.json", "aggregate campaign path")
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
        audit_receipt_path = folder / "audit-execution-receipt.json"
        audit_receipt = read(audit_receipt_path)
        audit_stdout = bind(folder / "audit-stdout.txt", audit_receipt["stdout_sha256"])
        audit_receipt_contract(
            audit_receipt,
            checked,
            seed=seed,
            summary_sha=entry["sha256"],
            audit_sha=sha(audit_path),
            stdout_sha=sha(audit_stdout),
            auditor_sha=sha(__file__),
            test_receipt_sha=own_receipt(),
        )
        expected_audit_entry = {
            "audit_path": str(audit_path),
            "audit_sha256": sha(audit_path),
            "audit_execution_receipt": str(audit_receipt_path),
            "audit_execution_receipt_sha256": sha(audit_receipt_path),
            "audit_stdout_sha256": sha(audit_stdout),
        }
        exact(
            {k: entry[k] for k in expected_audit_entry},
            expected_audit_entry,
            "aggregate audit receipts",
        )
        for path in (audit_path, audit_receipt_path, audit_stdout):
            require(
                summary["input_sha256"].get(str(path)) == sha(path),
                "fresh audit consumed by aggregation",
            )
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
                **{
                    k: checked[k]
                    for k in (
                        "checkpoint",
                        "child",
                        "comparisons",
                        "gate",
                        "training_prefix_diagnostic",
                    )
                },
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
        "summary_sha256": SCREEN_SUMMARY_SHA,
        "audit_sha256": SCREEN_AUDIT_SHA,
        "decision_sha256": SCREEN_DECISION_SHA,
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
    execution_path = HERE / "aggregate-execution-receipt.json"
    execution = read(execution_path)
    stdout = bind(HERE / "aggregate-stdout.txt", execution["stdout_sha256"])
    require(
        type(execution["exit_code"]) is int
        and execution["exit_code"] == 0
        and execution["terminal_completion_observed_by_primary"] is True
        and execution["summary_sha256"] == sha(summary_path),
        "observed aggregate terminal completion",
    )
    result.update(
        per_seed=records,
        historical1729=historical_record,
        fresh_two=fresh,
        all_three=all_three,
        observation_scope=scope,
        gate=gate,
        aggregation={
            "runner_sha256": AGGREGATION_RUNNER_SHA,
            "test_receipt_sha256": AGGREGATION_RECEIPT_SHA,
            "execution_receipt_sha256": sha(execution_path),
            "stdout_sha256": sha(stdout),
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
