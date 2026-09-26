"""Independent bilinear-residual saved-evidence audit; CPU tensor inspection.

No neural forward, optimizer execution, or imports of primary prediction/gate
arithmetic. Pure helpers come from the hash-authenticated earlier independent audit.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PLAN_SHA = "b1cd8a897e5db7d0dc0adc3e3bd40a87b0ae5830714b63ea86649f1599c9e132"
AUDITOR_HELPER_SHA = "78d81a14d666549dd1f6419a7d78792073141a59a729135bc32ac84c88c6ce4a"
MEAN_SUMMARY_SHA = "0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61"
MEAN_AUDIT_SHA = "6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc"
MEAN_DECISION_SHA = "da0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6"
MEAN_RUNNER_SHA = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
PARENT_SHA = "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
MEMBERSHIP_SHA = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"
SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
SPLIT_SHA = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
ARCHITECTURE = "plm-frozen-symmetric-bilinear-residual-v1"
OBJECTIVE = "plm-bilinear-residual-balanced-bce-v1"
EVALUATOR = "plm-bilinear-residual-screen-v1"
A = "symmetric_bilinear_residual"
SHAPE = [2, 256, 256]
INITIAL_LOSS = 0.003822767175734043
INPUTS = {}
RUNNER_SHA = "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
RECIPE_SHA = "7cb54921727046887d888357bc9079e14e6069ae3c7415386e6170afec1a7533"
RUNNER_RECEIPT_SHA = "a761885c5765f02ad43560acd08d4d9d9814ffafe2d3dc5cf378fb09c6846d30"
HELPER_SHA = "2ab48674b526ec6641a6c2366b669ac76bdeac1e8633331e01c772d4393edd39"
ARCHITECTURE_DESCRIPTOR = {
    "id": ARCHITECTURE,
    "parameter": A,
    "shape": SHAPE,
    "dimension_token_ids": [32, 33],
    "dimensions": ["TYPE", "COLOR"],
    "dtype": "float32",
    "initialization": "exact-zero",
    "symmetrization": "(A+A.transpose(-1,-2))*0.5",
    "score_order": (
        "stack-linear-entities-gather-subject-linear-products-divide16-add-archived-parent-v1"
    ),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_auditor():
    path = ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py"
    if sha(path) != AUDITOR_HELPER_SHA:
        raise ValueError("independent arithmetic helper changed")
    spec = importlib.util.spec_from_file_location("bilinear_independent_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_auditor()
require = BASE.require
exact = BASE.exact
finite = BASE.finite
state_manifest = BASE.state_manifest
tensor_digest = BASE.tensor_digest
split_membership = BASE.split_membership
query_labels = BASE.query_labels
query_descriptor = BASE.query_descriptor
formatted_hash = BASE.formatted_hash
sidecar_matches = BASE.sidecar_matches
evaluation = BASE.evaluation
screen_gate = BASE.screen_gate


def bind(path, digest):
    path = Path(path).resolve()
    require(sha(path) == digest, "hash mismatch: " + str(path))
    require(str(path) not in INPUTS or INPUTS[str(path)] == digest, "conflicting input digest")
    INPUTS[str(path)] = digest
    return path


def read(path, digest=None):
    return json.loads(bind(path, digest or sha(path)).read_text(encoding="utf-8"))


def load_payload(path, digest):
    import torch

    return torch.load(bind(path, digest), map_location="cpu", weights_only=False)


def compare_states(parent, child):
    """The original tensors are immutable; A is the sole new trained tensor."""
    import torch

    require(len(parent) == 93 and A not in parent, "exact original 93-tensor inventory")
    require(set(child) == set(parent) | {A}, "child adds only residual tensor")
    for name in parent:
        before, after = parent[name], child[name]
        require(
            isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor), "tensor state"
        )
        require(before.device.type == after.device.type == "cpu", "CPU payload inspection")
        require(
            before.dtype == after.dtype and before.shape == after.shape, "unchanged tensor schema"
        )
        require(
            torch.isfinite(before).all().item() and torch.isfinite(after).all().item(),
            "finite original tensors",
        )
        require(tensor_digest(before) == tensor_digest(after), "changed frozen tensor: " + name)
    residual = child[A]
    require(
        isinstance(residual, torch.Tensor)
        and residual.device.type == "cpu"
        and residual.dtype == torch.float32
        and list(residual.shape) == SHAPE,
        "residual FP32 CPU shape",
    )
    require(
        torch.isfinite(residual).all().item() and torch.count_nonzero(residual).item() > 0,
        "finite trained nonzero residual",
    )
    original = state_manifest(parent)
    initial = {
        **original,
        A: {
            "sha256": tensor_digest(torch.zeros(SHAPE, dtype=torch.float32)),
            "shape": SHAPE,
            "dtype": "torch.float32",
        },
    }
    return original, initial, state_manifest(child)


def optimizer_contract(optimizer, implementation):
    import torch

    require(set(optimizer) == {"state", "param_groups"}, "optimizer payload schema")
    groups = optimizer["param_groups"]
    require(
        len(groups) == 2 and len(groups[0]["params"]) == 1 and groups[1]["params"] == [],
        "sole residual optimizer parameter",
    )
    parameter_id = groups[0]["params"][0]
    require(set(optimizer["state"]) == {parameter_id}, "sole residual optimizer state")
    named = []
    for group in groups:
        require(
            group["lr"] == 0.0003
            and list(group["betas"]) == [0.9, 0.999]
            and group["eps"] == 1e-8
            and group["weight_decay"] == 0,
            "fixed AdamW group",
        )
        require(
            group.get("fused") in (None, False)
            and group["amsgrad"] is False
            and group.get("maximize", False) is False,
            "non-fused ordinary AdamW",
        )
        named.append(
            {k: v for k, v in group.items() if k != "params"}
            | {"parameter_names": [A] if group["params"] else []}
        )
    require(implementation["class"] == "torch.optim.adamw.AdamW", "optimizer class")
    exact(implementation["groups"], named, "serialized parameter ID to named residual")
    defaults = implementation["defaults"]
    require(
        defaults["lr"] == 0.0003
        and list(defaults["betas"]) == [0.9, 0.999]
        and defaults["eps"] == 1e-8
        and defaults.get("fused") in (None, False)
        and defaults["amsgrad"] is False,
        "optimizer default receipt",
    )
    state = optimizer["state"][parameter_id]
    require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "AdamW state fields")
    step = state["step"]
    require(
        isinstance(step, torch.Tensor)
        and step.device.type == "cpu"
        and step.numel() == 1
        and step.item() == 500,
        "residual optimizer step500",
    )
    for name in ("exp_avg", "exp_avg_sq"):
        value = state[name]
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and list(value.shape) == SHAPE
            and torch.isfinite(value).all().item(),
            "residual optimizer moment shape/finite",
        )
    require((state["exp_avg_sq"] >= 0).all().item(), "nonnegative second moment")


def loss_contract(history, initial, final):
    BASE.trace_contract(history)
    BASE.loss_endpoints(history, initial, final)
    require(initial == INITIAL_LOSS, "zero-residual full-training parent loss replay")


def state_receipts_contract(training, original, initial, final):
    exact(training["parent_state_before"], original, "parent state receipt")
    exact(training["state_before"], initial, "initial state is original plus exact zero A")
    exact(training["state_after"], final, "trained payload tensor receipt")
    exact(training["state_reloaded"], final, "reload tensor receipt")
    for name in (
        "residual_changed",
        "frozen_tensors_unchanged",
        "reload_exact",
        "optimizer_reload_exact",
    ):
        require(training[name] is True, "state/reload invariant: " + name)


def fp32_bytes(rows):
    """Lossless FP32 logits from JSON, in declared query/product order."""
    values = []
    for row in rows:
        vector = row["symmetric_relation_logits"]
        BASE.vector(vector)
        values.extend(vector)
    return struct.pack("<" + "f" * len(values), *values)


def architecture_contract(value):
    exact(value, ARCHITECTURE_DESCRIPTOR, "declared residual architecture and operation order")


def lineage_contract(metadata, identity):
    architecture_contract(metadata["architecture"])
    require(
        metadata["objective"] == OBJECTIVE
        and metadata["evaluator"] == EVALUATOR
        and identity["evaluator_version"] == EVALUATOR,
        "new objective/evaluator",
    )
    require(
        metadata["parent_checkpoint_sha256"] == PARENT_SHA
        and metadata["parent_training_steps"] == 2000
        and metadata["residual_updates"] == 500,
        "original parent and separate new updates",
    )
    exact(metadata["trainable_parameters"], [A], "sole named residual trainable")
    require(metadata["record_count"] == 1637, "full training partition")


def zero_replay_contract(report, parent):
    rows = parent["responses"]
    require(
        report["complete"] is True
        and report["query_count"] == len(rows) == 222
        and report["dtype"] == "float32",
        "complete zero-A validation replay",
    )
    require("error" not in report and "failed_observation" not in report, "zero replay failure")
    batches = []
    for offset in range(0, 222, 8):
        part = rows[offset : offset + 8]
        digest = hashlib.sha256(fp32_bytes(part)).hexdigest()
        zeros = bytes(len(part) * 1025 * 4)
        batches.append(
            {
                "indices": list(range(offset, offset + len(part))),
                "shape": [len(part), 1025],
                "logits_sha256": digest,
                "parent_logits_sha256": digest,
                "residual_sha256": hashlib.sha256(zeros).hexdigest(),
                "residual_nonzero_count": 0,
            }
        )
    expected = {
        "complete": True,
        "batches": batches,
        "query_count": 222,
        "dtype": "float32",
        "logits_shape": [222, 1025],
        "logits_sha256": hashlib.sha256(fp32_bytes(rows)).hexdigest(),
        "parent_logits_sha256": hashlib.sha256(fp32_bytes(rows)).hexdigest(),
        "residual_sha256": hashlib.sha256(bytes(222 * 1025 * 4)).hexdigest(),
        "residual_nonzero_count": 0,
        "batch_sizes": [8] * 27 + [6],
    }
    exact(report, expected, "actual zero-A head path replay hashes/batches")
    return {
        k: report[k] for k in ("query_count", "logits_sha256", "residual_sha256", "batch_sizes")
    }


def hash_manifest(mapping):
    require(isinstance(mapping, dict) and mapping, "nonempty hash manifest")
    for path, digest in mapping.items():
        bind(path, digest)


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


def upstream():
    old_dir = ROOT / "runs/learning/projection-only-refit-v1"
    old = read(old_dir / "summary.json", MEAN_SUMMARY_SHA)
    audit = read(old_dir / "independent-audit.json", MEAN_AUDIT_SHA)
    decision = read(old_dir / "decision.json", MEAN_DECISION_SHA)
    require(
        old["complete"] is True
        and old["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == MEAN_SUMMARY_SHA
        and decision["summary_sha256"] == MEAN_SUMMARY_SHA
        and decision["audit_sha256"] == MEAN_AUDIT_SHA
        and decision["evidence_accepted"] is True,
        "accepted mean evidence",
    )
    hash_manifest(old["input_sha256"])
    bind(old_dir / "independent-audit.py", AUDITOR_HELPER_SHA)
    training = read(old_dir / "training.json", old["artifact_sha256"]["training.json"])
    require(training["initial_pre_update_loss"] == INITIAL_LOSS, "historical initial loss")
    vocabulary_path = ROOT / "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json"
    vocabulary = read(vocabulary_path, old["input_sha256"][str(vocabulary_path)])["tokens"]
    records_path = vocabulary_path.with_name("records.jsonl")
    bind(records_path, old["input_sha256"][str(records_path)])
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    partitions, split = split_membership(records)
    require(
        split == SPLIT_SHA
        and {k: len(v) for k, v in partitions.items()}
        == {"train": 1637, "validation": 222, "test": 191},
        "original partition identity",
    )
    wide_path = ROOT / "runs/learning/wide-first-choice-v1/seed-1729.json"
    wide = read(wide_path, old["input_sha256"][str(wide_path)])
    old_membership = read(old_dir / "train-membership.json", MEMBERSHIP_SHA)
    parent_path = ROOT / "runs/national_dex_continuation_control_s1729_v1/checkpoint-final.pt"
    parent_sidecar = read(
        parent_path.with_suffix(".pt.json"),
        old["input_sha256"][str(parent_path.with_suffix(".pt.json"))],
    )
    bind(parent_path, PARENT_SHA)
    old_run_path = parent_path.with_name("run.json")
    old_run = read(old_run_path, old["input_sha256"][str(old_run_path)])
    return (
        old,
        training,
        old_membership,
        partitions,
        vocabulary,
        wide,
        parent_path,
        parent_sidecar,
        old_run,
    )


def recipe_contract(value, inherited):
    expected = json.loads(json.dumps(inherited))
    expected.pop("projection_updates")
    expected.pop("projection_shape")
    expected.update(
        experiment="bilinear-residual-refit-v1",
        objective=OBJECTIVE,
        evaluator=EVALUATOR,
        architecture=ARCHITECTURE,
        residual_updates=500,
        trainable_parameter=A,
        residual_shape=SHAPE,
        dimension_token_ids=[32, 33],
        initial_training_loss=INITIAL_LOSS,
        initialization="exact-zero",
    )
    exact(value, expected, "only declared bilinear recipe changes")


def provenance(summary, folder, inherited):
    require(
        summary["campaign_version"] == "bilinear-residual-refit-v1"
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
        ("mean_helper_sha256", MEAN_RUNNER_SHA),
        ("helper_sha256", HELPER_SHA),
    ):
        require(summary[key] == digest, "frozen primary identity: " + key)
    require(
        summary["objective"] == OBJECTIVE and summary["evaluator"] == EVALUATOR,
        "new objective/evaluator identity",
    )
    architecture_contract(summary["architecture"])
    hash_manifest(summary["input_sha256"])
    hash_manifest(summary["snapshot_sha256"])
    inputs = summary["input_sha256"]
    required = {
        PARENT_SHA,
        SOURCE_SHA,
        CONFIGS_SHA,
        MEAN_SUMMARY_SHA,
        MEAN_AUDIT_SHA,
        MEAN_DECISION_SHA,
        MEAN_RUNNER_SHA,
        AUDITOR_HELPER_SHA,
        PLAN_SHA,
        RUNNER_SHA,
        RECIPE_SHA,
        RUNNER_RECEIPT_SHA,
        MEMBERSHIP_SHA,
    }
    require(required <= set(inputs.values()), "required consumed immutable identities")
    for path, digest in inherited["input_sha256"].items():
        relative = Path(path).relative_to(ROOT)
        if (
            relative.parts[0] == "data"
            or "national_dex_continuation_control_s1729_v1" in relative.parts
        ):
            require(inputs.get(path) == digest, "original data/checkpoint exact input binding")
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
    }
    require(
        set(summary["snapshot_sha256"])
        == {str(folder / ("summary." + suffix)) for suffix in snapshots},
        "snapshot inventory",
    )
    for suffix, digest in (
        ("script.py", RUNNER_SHA),
        ("mean-helper.py", MEAN_RUNNER_SHA),
        ("helper.py", HELPER_SHA),
        ("plan.md", PLAN_SHA),
        ("recipe.json", RECIPE_SHA),
        ("test-receipt.json", RUNNER_RECEIPT_SHA),
    ):
        bind(folder / ("summary." + suffix), digest)
    exact(read(folder / "summary.inputs.json"), inputs, "input manifest snapshot")
    recipe = read(folder / "summary.recipe.json", RECIPE_SHA)
    old_recipe_path = ROOT / "runs/learning/projection-only-refit-v1/summary.recipe.json"
    recipe_contract(
        recipe, read(old_recipe_path, inherited["snapshot_sha256"][str(old_recipe_path)])
    )
    receipt = read(folder / "summary.test-receipt.json", RUNNER_RECEIPT_SHA)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["experiment_training_executed"] is False
        and receipt["tested_script_sha256"] == RUNNER_SHA
        and receipt["tested_recipe_sha256"] == RECIPE_SHA
        and receipt["tested_mean_helper_sha256"] == MEAN_RUNNER_SHA,
        "frozen primary synthetic receipt",
    )
    for path, digest in (
        (receipt["stdout"]["path"], receipt["stdout"]["sha256"]),
        (receipt["test_source"], receipt["tested_test_sha256"]),
    ):
        bind(path, digest)
        require(inputs.get(str(Path(path).resolve())) == digest, "test input binding")
    bind(folder / "summary.test-stdout.txt", receipt["stdout"]["sha256"])
    fixture = ROOT / "tests/fixtures/projection_refit_runtime_source.zip"
    exact(receipt["fixtures"], {str(fixture): SOURCE_SHA}, "portable archived runtime fixture")
    require(inputs.get(str(fixture)) == SOURCE_SHA, "fixture input binding")
    bind(fixture, SOURCE_SHA)
    for path, digest in receipt["test_dependencies"].items():
        bind(path, digest)
        require(inputs.get(path) == digest, "shared tested dependency binding")
    runtime = folder / "runtime"
    source = archive_inventory(folder / "summary.source.zip", SOURCE_SHA, runtime)
    configs = archive_inventory(folder / "summary.configs.zip", CONFIGS_SHA, runtime)
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
            path.is_relative_to(runtime / "src") and inventory.get(str(path)) == entry["sha256"],
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
    require(set(summary["artifact_sha256"]) == artifacts, "complete experiment artifact inventory")
    for name, digest in summary["artifact_sha256"].items():
        bind(folder / name, digest)
    return recipe


def training_audit(summary, reports, recipe, context, folder):
    (
        _old,
        old_training,
        old_membership,
        partitions,
        vocabulary,
        _,
        parent_path,
        parent_sidecar,
        old_run,
    ) = context
    training, run = reports["training.json"], reports["run.json"]
    require(
        training["complete"] is True
        and training["objective"] == OBJECTIVE
        and training["completed_updates"] == 500,
        "complete new objective training",
    )
    require(
        not any(k in training for k in ("error", "failed_update", "failed_observation")),
        "retained training failure",
    )
    architecture_contract(training["architecture"])
    loss_contract(
        training["history"], training["initial_pre_update_loss"], training["final_post_update_loss"]
    )
    require(
        training["initial_loss_replay"] == INITIAL_LOSS == old_training["initial_pre_update_loss"],
        "zero-A original full-training loss identity",
    )
    require(
        training["train_query_count"] == 1637
        and training["transformer_forwards_during_fit"] == 0
        and training["validation_labels_used"] is False,
        "train-only fit receipts",
    )
    exact(training["trainable_parameters"], [A], "sole trainable residual name")
    queries = {
        name: [query_descriptor(r) for r in partitions[name]] for name in ("train", "validation")
    }
    exact(summary["queries"], queries, "ordered supervised partitions")
    exact(summary["validation_update_points"], [0, 500], "fixed endpoints only")
    require(
        summary["train_query_order_sha256"]
        == training["train_query_order_sha256"]
        == formatted_hash(queries["train"]),
        "train query order hash",
    )
    membership = {
        "partition": "train",
        "query_count": 1637,
        "queries": [
            {
                "index": index,
                **query_descriptor(record),
                "expected_set_ids": query_labels(record, vocabulary)[1],
            }
            for index, record in enumerate(partitions["train"])
        ],
    }
    exact(reports["train-membership.json"], membership, "independent train labels")
    exact(membership, old_membership, "old training membership identity")
    require(
        sha(folder / "train-membership.json") == MEMBERSHIP_SHA, "exact training membership bytes"
    )
    parent = load_payload(parent_path, PARENT_SHA)
    checkpoint = folder / "checkpoint-final.pt"
    require(
        Path(training["checkpoint"]).resolve() == checkpoint
        and training["checkpoint_sha256"] == summary["artifact_sha256"]["checkpoint-final.pt"]
        and training["checkpoint_sidecar_sha256"]
        == summary["artifact_sha256"]["checkpoint-final.pt.json"],
        "child payload paths/hashes",
    )
    child = load_payload(checkpoint, training["checkpoint_sha256"])
    sidecar_matches(parent, parent_sidecar, PARENT_SHA)
    sidecar_matches(child, reports["checkpoint-final.pt.json"], training["checkpoint_sha256"])
    lineage_contract(child["training_metadata"], child["experiment_identity"])
    require(
        parent["global_step"] == 2000
        and type(child["global_step"]) is int
        and child["global_step"] == 500,
        "parent/new update distinction",
    )
    parent_state, initial_state, final_state = compare_states(parent["model"], child["model"])
    exact(parent_state, old_training["state_before"], "authenticated original 93 tensors")
    state_receipts_contract(training, parent_state, initial_state, final_state)
    require(training["optimizer"]["name"] == "adamw", "optimizer family")
    exact(training["optimizer"]["config"], recipe["optimizer"], "optimizer config recipe")
    impl = training["optimizer_implementation"]
    optimizer_contract(
        child["optimizer"],
        {
            "class": impl["module"] + "." + impl["class"],
            "groups": training["optimizer"]["groups"],
            "defaults": impl["defaults"],
        },
    )
    require(child["scheduler"] is None, "no new scheduler")
    config = {
        "inherited_parent_config": old_run["config"],
        "bilinear_residual_architecture": ARCHITECTURE_DESCRIPTOR,
        "bilinear_residual_recipe": recipe,
        "implementation_identity": {
            "runner_sha256": RUNNER_SHA,
            "mean_helper_sha256": MEAN_RUNNER_SHA,
            "source_archive_sha256": SOURCE_SHA,
            "configs_archive_sha256": CONFIGS_SHA,
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
        "objective": OBJECTIVE,
        "evaluator": EVALUATOR,
        "architecture": ARCHITECTURE_DESCRIPTOR,
        "model_config": parent["training_metadata"]["model_config"],
        "model_config_scope": "inherited archived base only; residual architecture is separate",
        "parent_checkpoint_sha256": PARENT_SHA,
        "parent_training_steps": 2000,
        "residual_updates": 500,
        "trainable_parameters": [A],
        "train_query_order_sha256": summary["train_query_order_sha256"],
        "record_count": 1637,
        "recipe_sha256": RECIPE_SHA,
        "runner_sha256": RUNNER_SHA,
        "scorer_sha256": RUNNER_SHA,
        "parent_state_sha256": formatted_hash(parent_state),
        "initial_state_sha256": formatted_hash(initial_state),
        "initial_pre_update_loss": INITIAL_LOSS,
        "final_post_update_loss": training["final_post_update_loss"],
        "standard_serving_supported": False,
        "implementation_identity": config["implementation_identity"],
    }
    exact(run["training_metadata"], metadata, "authoritative residual metadata")
    exact(child["training_metadata"], metadata, "checkpoint metadata/endpoints/source binding")
    identity = {
        **parent["experiment_identity"],
        "source_commit": summary["workspace_provenance"]["commit"],
        "resolved_config_hash": formatted_hash(config),
        "evaluator_version": EVALUATOR,
        "environment": {
            **summary["environment"],
            "source_archive_sha256": SOURCE_SHA,
            "configs_archive_sha256": CONFIGS_SHA,
            "runner_sha256": RUNNER_SHA,
            "recipe_sha256": RECIPE_SHA,
            "mean_helper_sha256": MEAN_RUNNER_SHA,
            "architecture": ARCHITECTURE,
        },
        "checkpoint_hash": None,
        "seeds": [1729],
        "repetitions": 1,
    }
    for stored in (
        run["identity"],
        child["experiment_identity"],
        training["training_identity"],
        summary["child_training_identity"],
    ):
        exact(stored, identity, "new training identity")
    require(child["split_hash"] == summary["split_hash"] == SPLIT_SHA, "frozen child split")
    exact(child["corpus_identity"], parent["corpus_identity"], "child corpus unchanged")
    require(
        training["zero_replay_sha256"] == summary["artifact_sha256"]["zero-replay.json"],
        "zero initialization replay hash",
    )
    exact(training["zero_replay"], reports["zero-replay.json"], "zero replay nested receipt")
    finite(training["refit_seconds"])
    require(training["refit_seconds"] >= 0, "descriptive refit time")
    return {
        "parent_checkpoint_sha256": PARENT_SHA,
        "child_checkpoint_sha256": training["checkpoint_sha256"],
        "child_sidecar_sha256": training["checkpoint_sidecar_sha256"],
        "architecture": ARCHITECTURE_DESCRIPTOR,
        "global_step": 500,
        "parent_training_steps": 2000,
        "original_tensor_count": 93,
        "child_tensor_count": 94,
        "unchanged_original_tensor_count": 93,
        "new_trained_tensor": A,
        "initial_residual": initial_state[A],
        "trained_residual": final_state[A],
        "train_query_count": 1637,
        "initial_pre_update_loss": INITIAL_LOSS,
        "final_post_update_loss": training["final_post_update_loss"],
        "updates_verified": 500,
        "optimizer_step": 500,
        "refit_seconds": training["refit_seconds"],
        "payload_loaded_on": "cpu",
    }


def audit(summary_path):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment")
    summary = read(summary_path)
    folder = summary_path.parent
    bind(ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py", AUDITOR_HELPER_SHA)
    context = upstream()
    recipe = provenance(summary, folder, context[0])
    reports = {
        name: read(folder / name, digest)
        for name, digest in summary["artifact_sha256"].items()
        if name.endswith(".json")
    }
    parent, child = reports["parent.json"], reports["child.json"]
    _, _, _, partitions, vocabulary, wide, _, _, _ = context
    evaluation(parent, "parent", wide, partitions["validation"], vocabulary)
    zero = zero_replay_contract(reports["zero-replay.json"], parent)
    evaluation(child, "child", wide, partitions["validation"], vocabulary, parent)
    checkpoint = training_audit(summary, reports, recipe, context, folder)
    invariant_names = (
        "parent_replay_exact",
        "zero_A_replay_exact",
        "initial_train_loss_exact",
        "completed_500_updates",
        "original_93_tensors_unchanged",
        "residual_changed",
        "reload_exact",
        "optimizer_reload_exact",
        "train_only_supervision",
    )
    invariants = dict.fromkeys(invariant_names, True)
    exact(summary["invariants"], invariants, "checked execution invariants")
    gate = screen_gate(child, invariants)
    exact(summary["gate"], gate, "independent unchanged quality gate")
    exact(summary["parent"], parent["aggregate"], "summary parent metrics")
    child_summary = {
        k: child[k] for k in ("aggregate", "groups", "paired_vs_parent_dense", "paired_vs_width8")
    }
    exact(summary["child"], child_summary, "summary child metrics/comparisons")
    tests = ROOT / "runs/learning/bilinear-residual-refit-auditor-tests-v1"
    receipt = read(tests / "test-receipt.json")
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["model_execution"] is False
        and receipt["new_diagnostic_executed"] is False
        and receipt["tested_auditor_sha256"] == sha(__file__),
        "frozen independent synthetic receipt",
    )
    bind(tests / "test_auditor.py", receipt["tests_sha256"])
    bind(tests / "stdout.txt", receipt["stdout_sha256"])
    require(BASE.INPUTS == {}, "authenticated helper globals unchanged")
    hash_manifest(dict(INPUTS))
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(summary_path),
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN_SHA,
        "test_receipt_sha256": sha(tests / "test-receipt.json"),
        "independent_helper_sha256": AUDITOR_HELPER_SHA,
        "parent": {k: parent[k] for k in ("aggregate", "groups", "paired_vs_width8")},
        "child": child_summary,
        "zero_initialization_replay": zero,
        "comparisons": {
            "versus_parent_dense": child["paired_vs_parent_dense"],
            "versus_width8": child["paired_vs_width8"],
        },
        "checkpoint": checkpoint,
        "gate": gate,
        "input_sha256": dict(INPUTS),
        "limitations": [
            "Saved-evidence and CPU checkpoint inspection; no neural forward or repeated training.",
            "Loss/update receipts are authenticated, not independently regenerated.",
            "All 93 parent tensors match; saved A and optimizer moments are inspected.",
            "Reload/optimizer chronology is authenticated, not independently replayed.",
            "Saved logits/zero-initialization hashes are bound, not recomputed on CUDA.",
            "The inherited decoder output does not represent the new architecture/scorer.",
            "Dense validation does not prove serving, protected quality or durable weight backup.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    path = args.summary.resolve()
    out = path.parent / "independent-audit.json"
    require(not out.exists(), "immutable independent audit")
    result = audit(path)
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "audit_sha256": sha(out),
                "checkpoint": result["checkpoint"],
                "child": result["child"]["aggregate"],
                "gate": result["gate"],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
