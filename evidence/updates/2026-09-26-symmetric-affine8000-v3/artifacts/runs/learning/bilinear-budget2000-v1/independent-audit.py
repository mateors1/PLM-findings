"""Independent 2000-update bilinear budget audit; CPU saved-payload inspection.

No neural forward, optimizer execution, or imports of primary prediction/gate
arithmetic. Pure helpers come from the hash-authenticated earlier independent audit.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PLAN_SHA = "18c08693790880740b20cf25633b56880392362ffbc6b398cd7069116b26edbe"
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
BILINEAR_AUDITOR_SHA = "a046a1e45b3c977fc556ecc74358d64e2b9d0463ebf653c7b7f7c86aed61622b"
SCORER_SHA = "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
HISTORICAL_SUMMARY_SHA = "92955bf21ce9dc85ce9806436417436cece618d1be9edd3c400f6d90c96bc0b5"
HISTORICAL_AUDIT_SHA = "2d8f3d49a4d6e25eaee319f3517b7ec69e139a566a814d4be1f01a44cdbef417"
HISTORICAL_DECISION_SHA = "d7d38687e2d099348af61bd52ed62df28211464e4209cb00df43db21d1d93234"
HISTORICAL_CHECKPOINT_SHA = "1f6e9a593ac5c5c83f0d003f273adc02213845ab94f7d5118b1b55bf5bab43d9"
HISTORICAL_RECIPE_SHA = "7cb54921727046887d888357bc9079e14e6069ae3c7415386e6170afec1a7533"
RUNNER_SHA = "f4376d4f2d838938bb9f50ae0cfa16ef319639fa0ce917c9fea3b842c0018930"
RECIPE_SHA = "e454c52a315acb389b2bda3c126367e13df63b96230f9f698517705b531d51da"
RUNNER_RECEIPT_SHA = "7dc624f0f30478d89dc0f65742b47ba1931af580268b14df422ca0c13c703d1b"
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
    path = ROOT / "runs/learning/bilinear-residual-refit-v1/independent-audit.py"
    if sha(path) != BILINEAR_AUDITOR_SHA:
        raise ValueError("frozen bilinear independent helper changed")
    spec = importlib.util.spec_from_file_location("budget2000_independent_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD = load_auditor()
BASE = OLD.BASE
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


compare_states = OLD.compare_states


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
        and step.item() == 2000,
        "residual optimizer step2000",
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
    require(len(history) == 2000, "exactly 2000 independent updates")
    for number, row in enumerate(history, 1):
        require(type(row["update"]) is int and row["update"] == number, "ordered 2000 updates")
        finite(row["pre_update_loss"])
        require(
            row["pre_update_loss"] >= 0
            and row["gradient_finite"] is True
            and row["parameters_finite"] is True,
            "finite update attestations",
        )
    finite(initial)
    finite(final)
    require(
        initial == INITIAL_LOSS == history[0]["pre_update_loss"] and final >= 0,
        "initial replay and final loss endpoint",
    )


state_receipts_contract = OLD.state_receipts_contract


fp32_bytes = OLD.fp32_bytes


architecture_contract = OLD.architecture_contract


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
        and metadata["residual_updates"] == 2000,
        "original parent and separate new updates",
    )
    exact(metadata["trainable_parameters"], [A], "sole named residual trainable")
    require(metadata["record_count"] == 1637, "full training partition")


zero_replay_contract = OLD.zero_replay_contract


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
    require(
        expected["experiment"] == "bilinear-residual-refit-v1"
        and expected["residual_updates"] == 500,
        "historical recipe origin",
    )
    expected.update(experiment="bilinear-budget2000-v1", residual_updates=2000)
    exact(value, expected, "sole declared budget and campaign changes")


def historical_evidence():
    directory = ROOT / "runs/learning/bilinear-residual-refit-v1"
    summary = read(directory / "summary.json", HISTORICAL_SUMMARY_SHA)
    audit = read(directory / "independent-audit.json", HISTORICAL_AUDIT_SHA)
    decision = read(directory / "decision.json", HISTORICAL_DECISION_SHA)
    require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == HISTORICAL_SUMMARY_SHA
        and decision["evidence_accepted"] is True
        and decision["summary_sha256"] == HISTORICAL_SUMMARY_SHA
        and decision["audit_sha256"] == HISTORICAL_AUDIT_SHA,
        "accepted historical500 evidence",
    )
    require(
        audit["script_sha256"] == BILINEAR_AUDITOR_SHA
        and decision["fixed_quality_gate_passed"] is False
        and summary["script_sha256"] == SCORER_SHA,
        "frozen scorer and preserved failed historical gate",
    )
    architecture_contract(summary["architecture"])
    hash_manifest(summary["input_sha256"])
    child = read(directory / "child.json", summary["artifact_sha256"]["child.json"])
    training = read(directory / "training.json", summary["artifact_sha256"]["training.json"])
    run = read(directory / "run.json", summary["artifact_sha256"]["run.json"])
    require(
        training["checkpoint_sha256"] == HISTORICAL_CHECKPOINT_SHA
        and training["completed_updates"] == 500
        and training["complete"] is True
        and training["initial_pre_update_loss"] == INITIAL_LOSS,
        "historical500 training receipts",
    )
    bind(directory / "checkpoint-final.pt", HISTORICAL_CHECKPOINT_SHA)
    bind(
        directory / "checkpoint-final.pt.json",
        summary["artifact_sha256"]["checkpoint-final.pt.json"],
    )
    require(
        run["training_metadata"]["parent_checkpoint_sha256"] == PARENT_SHA
        and run["training_metadata"]["scorer_sha256"] == SCORER_SHA,
        "historical scorer/original parent",
    )
    exact(child["aggregate"], audit["child"]["aggregate"], "accepted historical metric binding")
    exact(child["groups"], audit["child"]["groups"], "accepted historical group binding")
    return summary, child, training, run


def historical_pairing(child, historical, historical_child_sha):
    require(
        child["complete"] is True
        and historical["complete"] is True
        and child["query_count"] == historical["query_count"] == 222,
        "historical pairing coverage",
    )
    exact(
        child["product_token_ids"],
        historical["product_token_ids"],
        "historical product column order",
    )
    current, previous = child["responses"], historical["responses"]
    require(len(current) == len(previous) == 222, "full paired query coverage")
    fields = ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_ids")
    for row, old in zip(current, previous, strict=True):
        exact(
            [row[k] for k in fields],
            [old[k] for k in fields],
            "historical exact query/label/group order",
        )
        for item in (row, old):
            predicted = BASE.predict(item["symmetric_relation_logits"], item["prompt_ids"][1])
            exact(item["selected_set_ids"], predicted, "historical/current dense selection")
            exact(
                item["metrics"],
                BASE.metrics(predicted, item["expected_set_ids"]),
                "paired exact set arithmetic",
            )
    paired = BASE.paired(current, [r["selected_set_ids"] for r in previous])
    return {
        "historical500_summary_sha256": HISTORICAL_SUMMARY_SHA,
        "historical500_child_sha256": historical_child_sha,
        "comparison_scope": "saved historical output; no rerun",
        "paired_vs_historical500": paired,
    }


def provenance(summary, folder, inherited):
    require(
        summary["campaign_version"] == "bilinear-budget2000-v1"
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
        ("historical500_summary_sha256", HISTORICAL_SUMMARY_SHA),
        ("historical500_audit_sha256", HISTORICAL_AUDIT_SHA),
        ("historical500_decision_sha256", HISTORICAL_DECISION_SHA),
    ):
        require(summary[key] == digest, "frozen primary identity: " + key)
    require(
        summary["objective"] == OBJECTIVE and summary["evaluator"] == EVALUATOR,
        "new objective/evaluator identity",
    )
    require(summary["scorer_sha256"] == SCORER_SHA, "unchanged frozen scorer separate from runner")
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
        SCORER_SHA,
        HISTORICAL_SUMMARY_SHA,
        HISTORICAL_AUDIT_SHA,
        HISTORICAL_DECISION_SHA,
        BILINEAR_AUDITOR_SHA,
        HISTORICAL_CHECKPOINT_SHA,
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
        "bilinear-helper.py",
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
        ("bilinear-helper.py", SCORER_SHA),
    ):
        bind(folder / ("summary." + suffix), digest)
    exact(read(folder / "summary.inputs.json"), inputs, "input manifest snapshot")
    recipe = read(folder / "summary.recipe.json", RECIPE_SHA)
    old_recipe_path = ROOT / "runs/learning/bilinear-residual-refit-v1/summary.recipe.json"
    recipe_contract(recipe, read(old_recipe_path, HISTORICAL_RECIPE_SHA))
    receipt = read(folder / "summary.test-receipt.json", RUNNER_RECEIPT_SHA)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["experiment_training_executed"] is False
        and receipt["tested_script_sha256"] == RUNNER_SHA
        and receipt["tested_recipe_sha256"] == RECIPE_SHA
        and receipt["tested_mean_helper_sha256"] == MEAN_RUNNER_SHA
        and receipt["tested_scorer_sha256"] == SCORER_SHA
        and receipt["test_count"] == 91
        and receipt["skipped"] == 0,
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
        "comparisons.json",
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
        and training["completed_updates"] == 2000,
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
    exact(summary["validation_update_points"], [0, 2000], "fixed endpoints only")
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
        and child["global_step"] == 2000,
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
            "scorer_sha256": SCORER_SHA,
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
        "residual_updates": 2000,
        "trainable_parameters": [A],
        "train_query_order_sha256": summary["train_query_order_sha256"],
        "record_count": 1637,
        "recipe_sha256": RECIPE_SHA,
        "runner_sha256": RUNNER_SHA,
        "scorer_sha256": SCORER_SHA,
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
            "scorer_sha256": SCORER_SHA,
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
        "global_step": 2000,
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
        "updates_verified": 2000,
        "optimizer_step": 2000,
        "refit_seconds": training["refit_seconds"],
        "payload_loaded_on": "cpu",
        "scorer_sha256": SCORER_SHA,
        "runner_sha256": RUNNER_SHA,
    }


def audit(summary_path):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment")
    summary = read(summary_path)
    folder = summary_path.parent
    bind(ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py", AUDITOR_HELPER_SHA)
    bind(
        ROOT / "runs/learning/bilinear-residual-refit-v1/independent-audit.py", BILINEAR_AUDITOR_SHA
    )
    context = upstream()
    historical_summary, historical_child, historical_training, historical_run = (
        historical_evidence()
    )
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
    evaluation(historical_child, "child", wide, partitions["validation"], vocabulary, parent)
    comparison = historical_pairing(
        child, historical_child, historical_summary["artifact_sha256"]["child.json"]
    )
    exact(reports["comparisons.json"], comparison, "independent historical500 pairing report")
    exact(
        recipe,
        {
            **historical_run["config"]["bilinear_residual_recipe"],
            "experiment": "bilinear-budget2000-v1",
            "residual_updates": 2000,
        },
        "historical/new recipe compatibility",
    )
    exact(
        reports["training.json"]["state_before"],
        historical_training["state_before"],
        "same zero-A start, not resumed trained child",
    )
    checkpoint = training_audit(summary, reports, recipe, context, folder)
    invariant_names = (
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
    invariants = dict.fromkeys(invariant_names, True)
    exact(summary["invariants"], invariants, "checked execution invariants")
    gate = screen_gate(child, invariants)
    exact(summary["gate"], gate, "independent unchanged quality gate")
    exact(summary["parent"], parent["aggregate"], "summary parent metrics")
    child_summary = {
        k: child[k] for k in ("aggregate", "groups", "paired_vs_parent_dense", "paired_vs_width8")
    }
    child_summary["paired_vs_historical500"] = comparison["paired_vs_historical500"]
    exact(summary["child"], child_summary, "summary child metrics/comparisons")
    tests = ROOT / "runs/learning/bilinear-budget2000-auditor-tests-v1"
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
    require(BASE.INPUTS == {} and OLD.INPUTS == {}, "authenticated helper globals unchanged")
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
        "bilinear_independent_helper_sha256": BILINEAR_AUDITOR_SHA,
        "parent": {k: parent[k] for k in ("aggregate", "groups", "paired_vs_width8")},
        "child": child_summary,
        "zero_initialization_replay": zero,
        "comparisons": {
            "versus_parent_dense": child["paired_vs_parent_dense"],
            "versus_width8": child["paired_vs_width8"],
            "versus_historical500": comparison["paired_vs_historical500"],
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
