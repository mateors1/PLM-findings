"""Independent saved-evidence worst-boundary audit, with authenticated pure helpers.

No prediction/aggregation imports from the primary runner; no neural execution.
The earlier auditor is reused without altering any of its globals or contracts.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN_SHA = "03bda47032fa65c11bf70904cb05f845fe10a89154e36a5edb5b76b0f5d7be9c"
OBJECTIVE = "plm-projection-worst-boundary-softplus-v1"
EVALUATOR = "plm-projection-worst-boundary-screen-v1"
AUDITOR_HELPER_SHA = "78d81a14d666549dd1f6419a7d78792073141a59a729135bc32ac84c88c6ce4a"
MEAN_RUNNER_SHA = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
MEAN_PLAN_SHA = "2ac4d157ad8c079561d642199fec8de2c3da1866819baa48f52080e9b0f39bfe"
MEAN_SUMMARY_SHA = "0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61"
MEAN_AUDIT_SHA = "6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc"
MEAN_DECISION_SHA = "da0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6"
MEAN_CHILD_SHA = "b44463dd43df234aaba5f95a8910eb5afea1bee012782f2a9d19896adb221f1f"
MEAN_CHECKPOINT_SHA = "84a3ba02ab5f43e9073f1f8b8c7afc85ad7a5363e8cb180e1424a77e6a44ab87"
MEAN_INITIAL_LOSS = 0.003822767175734043
RUNNER_SHA = "8598f110d7cbc60085c7d099a821386cba3566f4264c91939fada5baa0f0e480"
RECIPE_SHA = "023cc73601332bc2260339808dd244b8456c4e86d59ac5a3cf5bac9e5b9a8c0c"
RUNNER_RECEIPT_SHA = "57acbcbf1bb8bf926bc6e1444c3cdfef65ff68fa2fd37ecd0a1770d7051cc1a4"
INPUTS = {}


def load_auditor():
    path = ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != AUDITOR_HELPER_SHA:
        raise ValueError("independent helper hash mismatch")
    spec = importlib.util.spec_from_file_location("authenticated_mean_projection_auditor", path)
    if spec is None or spec.loader is None:
        raise ValueError("helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_AUDIT = load_auditor()
require = BASE_AUDIT.require
exact = BASE_AUDIT.exact
finite = BASE_AUDIT.finite
vector = BASE_AUDIT.vector
predict = BASE_AUDIT.predict
metrics = BASE_AUDIT.metrics
separation = BASE_AUDIT.separation
split_membership = BASE_AUDIT.split_membership
query_labels = BASE_AUDIT.query_labels
trace_contract = BASE_AUDIT.trace_contract
tensor_digest = BASE_AUDIT.tensor_digest
compare_states = BASE_AUDIT.compare_states
optimizer_contract = BASE_AUDIT.optimizer_contract
sidecar_matches = BASE_AUDIT.sidecar_matches
loss_endpoints = BASE_AUDIT.loss_endpoints
trainable_manifest = BASE_AUDIT.trainable_manifest
formatted_hash = BASE_AUDIT.formatted_hash
totals = BASE_AUDIT.totals
paired = BASE_AUDIT.paired
evaluation = BASE_AUDIT.evaluation
screen_gate = BASE_AUDIT.screen_gate
state_manifest = BASE_AUDIT.state_manifest
query_descriptor = BASE_AUDIT.query_descriptor

PARENT_SHA = "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
PARENT_CONFIG_SHA = "6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148"
REFERENCE_SHA = "1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183"
REFERENCE_AUDIT_SHA = "eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95"
REFERENCE_DECISION_SHA = "70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601"
SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
SPLIT_SHA = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
W = "symmetric_relation_projection.weight"
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
HELPER_SHA = "2ab48674b526ec6641a6c2366b669ac76bdeac1e8633331e01c772d4393edd39"
SETTINGS = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "deterministic_algorithms": False,
    "autocast_cuda_enabled": False,
    "autocast_cpu_enabled": False,
    "cudnn_deterministic": False,
    "cudnn_benchmark": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bind(path, digest):
    path = Path(path).resolve()
    require(sha(path) == digest, f"hash mismatch: {path}")
    require(str(path) not in INPUTS or INPUTS[str(path)] == digest, "input hash conflict")
    INPUTS[str(path)] = digest
    return path


def read(path, digest):
    return json.loads(bind(path, digest).read_text(encoding="utf-8"))


def source_archive(path, digest, destination=None):
    bind(path, digest)
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len({n.casefold() for n in names}), "duplicate archive path")
        for name in names:
            entry = archive.getinfo(name)
            relative = PurePosixPath(name)
            require(
                entry.orig_filename == name
                and relative.parts
                and not relative.is_absolute()
                and relative.as_posix() == name
                and ".." not in relative.parts
                and "\\" not in name
                and ":" not in name,
                "unsafe archive path",
            )
            require(
                not entry.is_dir() and (entry.external_attr >> 16) & 0o170000 != 0o120000,
                "unexpected directory/symlink",
            )
            result[name] = hashlib.sha256(archive.read(name)).hexdigest()
            if destination is not None:
                bind(destination / name, result[name])
    return result


def manifest(mapping):
    require(isinstance(mapping, dict) and mapping, "missing hash manifest")
    for path, digest in mapping.items():
        bind(path, digest)


def upstream():
    directory = ROOT / "runs/learning/wide-first-choice-v1"
    summary = read(directory / "summary.json", REFERENCE_SHA)
    checked = read(directory / "independent-audit.json", REFERENCE_AUDIT_SHA)
    decision = read(directory / "decision.json", REFERENCE_DECISION_SHA)
    require(
        summary["complete"] is True and summary["final_identity_check"] is True,
        "parent reference incomplete",
    )
    require(
        checked["audit_passed"] is True
        and checked["complete"] is True
        and checked["summary_sha256"] == REFERENCE_SHA,
        "parent reference audit",
    )
    require(
        decision["evidence_accepted"] is True
        and decision["fixed_quality_gate_passed"] is True
        and decision["summary_sha256"] == REFERENCE_SHA
        and decision["audit_sha256"] == REFERENCE_AUDIT_SHA,
        "accepted strong comparator",
    )
    bind(directory / "independent-audit.py", checked["script_sha256"])
    descriptors = [r for r in summary["reports"] if r["seed"] == 1729]
    require(len(descriptors) == 1, "one seed1729 reference")
    reference = read(descriptors[0]["path"], descriptors[0]["sha256"])
    require(
        reference["seed"] == 1729
        and reference["complete"] is True
        and reference["query_count"] == 222
        and len(reference["responses"]) == 222,
        "reference coverage",
    )
    require(
        reference["checkpoint_hash"] == PARENT_SHA and reference["split_hash"] == SPLIT_SHA,
        "parent identity",
    )
    exact(reference["product_token_ids"], list(range(1024, 2049)), "column identity")
    strong = reference["overall"]["wide"]
    require(
        strong["exact_count"] == 201 and strong["f1"] == 0.9799255176742276, "strong baseline201"
    )
    for group, count in [("COLOR", 103), ("TYPE_single", 50), ("TYPE_dual", 48)]:
        require(reference["groups"][group]["wide"]["exact_count"] == count, "strong group baseline")
    inherited = {str(Path(p).resolve()): h for p, h in summary["input_sha256"].items()}
    for path, digest in inherited.items():
        relative = Path(path).relative_to(ROOT)
        if (
            relative.parts[0] == "data"
            or "national_dex_continuation_control_s1729_v1" in relative.parts
        ):
            bind(path, digest)
    corpus = ROOT / "data/processed/pokemon_v1_f1541479_20260924"
    vocabulary = read(corpus / "vocabulary.json", inherited[str(corpus / "vocabulary.json")])[
        "tokens"
    ]
    require(len(vocabulary) == len(set(vocabulary)) == 2049, "vocabulary inventory")
    records_path = bind(corpus / "records.jsonl", inherited[str(corpus / "records.jsonl")])
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    partitions, split_hash = split_membership(records)
    require(
        split_hash == SPLIT_SHA
        and [len(partitions[k]) for k in ("train", "validation", "test")] == [1637, 222, 191],
        "original split assignment",
    )
    # Examine train/validation labels only; test keys establish split identity.
    for name in ("train", "validation"):
        for record in partitions[name]:
            query_labels(record, vocabulary)
    for record, row in zip(partitions["validation"], reference["responses"], strict=True):
        prompt, truth = query_labels(record, vocabulary)
        exact(
            [record["subject"], record["dimension"]],
            [row["subject"], row["dimension"]],
            "validation corpus order",
        )
        exact(prompt, row["prompt_ids"], "validation prompt binding")
        exact(truth, row["expected_set_ids"], "validation truth binding")
    parent_folder = ROOT / "runs/national_dex_continuation_control_s1729_v1"
    checkpoint = bind(parent_folder / "checkpoint-final.pt", PARENT_SHA)
    sidecar_path = parent_folder / "checkpoint-final.pt.json"
    sidecar = read(sidecar_path, inherited[str(sidecar_path)])
    require(
        sidecar["global_step"] == 2000
        and sidecar["checkpoint_hash"] == PARENT_SHA
        and sidecar["experiment_identity"]["resolved_config_hash"] == PARENT_CONFIG_SHA,
        "parent trained identity",
    )
    return summary, reference, vocabulary, partitions, checkpoint, sidecar


def load_payload(path, digest):
    import torch

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment required")
    # Both files are authenticated local experiment checkpoints, including RNG pickles.
    # No module or callable is imported from the primary evaluator.
    bind(path, digest)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict) and "model" in payload, "checkpoint payload")
    return payload


def training_audit(
    summary,
    training,
    run,
    membership,
    recipe,
    partitions,
    vocabulary,
    parent_path,
    parent_sidecar,
    folder,
):
    require(
        training["complete"] is True
        and training["objective"] == OBJECTIVE
        and training["completed_updates"] == 500,
        "complete projection training",
    )
    require(not any(k in training for k in ("error", "failed_update")), "training failure")
    trace_contract(training["history"])
    loss_endpoints(
        training["history"],
        training["initial_pre_update_loss"],
        training["final_post_update_loss"],
    )
    require(
        training["train_query_count"] == 1637
        and training["transformer_forwards_during_fit"] == 0
        and training["validation_labels_used"] is False,
        "train-only execution receipts",
    )
    trainable_manifest(training["trainable_parameters"])
    exact(training["optimizer"], recipe["optimizer"], "training optimizer recipe")
    queries = {
        name: [query_descriptor(r) for r in partitions[name]] for name in ("train", "validation")
    }
    exact(summary["queries"], queries, "ordered train and validation descriptors")
    exact(summary["validation_update_points"], [0, 500], "fixed validation update points")
    for name in ("train", "validation"):
        require(
            summary[name + "_query_order_sha256"] == formatted_hash(queries[name]),
            "query order hash",
        )
    exact(
        training["train_query_order_sha256"],
        summary["train_query_order_sha256"],
        "training order binding",
    )
    train_rows = []
    for index, record in enumerate(partitions["train"]):
        _, truth = query_labels(record, vocabulary)
        train_rows.append({"index": index, **query_descriptor(record), "expected_set_ids": truth})
    exact(
        membership,
        {"partition": "train", "query_count": 1637, "queries": train_rows},
        "complete train membership labels",
    )
    checkpoint = folder / "checkpoint-final.pt"
    require(Path(training["checkpoint"]).resolve() == checkpoint, "child checkpoint location")
    require(
        training["checkpoint_sha256"] == summary["artifact_sha256"]["checkpoint-final.pt"],
        "child checkpoint report binding",
    )
    require(
        training["checkpoint_sidecar_sha256"]
        == summary["artifact_sha256"]["checkpoint-final.pt.json"],
        "sidecar report binding",
    )
    parent = load_payload(parent_path, PARENT_SHA)
    child = load_payload(checkpoint, training["checkpoint_sha256"])
    sidecar = read(checkpoint.with_suffix(".pt.json"), training["checkpoint_sidecar_sha256"])
    sidecar_matches(parent, parent_sidecar, PARENT_SHA)
    sidecar_matches(child, sidecar, training["checkpoint_sha256"])
    lineage_contract(child["training_metadata"], child["experiment_identity"])
    require(
        parent["global_step"] == 2000
        and type(child["global_step"]) is int
        and child["global_step"] == 500,
        "separate parent/refit step identities",
    )
    compare_states(parent["model"], child["model"])
    before, after = state_manifest(parent["model"]), state_manifest(child["model"])
    exact(training["state_before"], before, "before manifest from parent payload")
    exact(training["state_after"], after, "after manifest from child payload")
    exact(training["state_reloaded"], after, "saved reload manifest")
    require(
        all(
            training[k] is True
            for k in (
                "projection_changed",
                "frozen_tensors_unchanged",
                "reload_exact",
                "optimizer_reload_exact",
            )
        ),
        "state/reload receipts",
    )
    optimizer_contract(child["optimizer"])
    require(child["scheduler"] is None, "no child scheduler")
    implementation = training["optimizer_implementation"]
    require(implementation["class"] == "torch.optim.adamw.AdamW", "optimizer implementation")
    groups = [
        {k: v for k, v in group.items() if k != "params"}
        | {"parameter_names": [W] if group["params"] else []}
        for group in child["optimizer"]["param_groups"]
    ]
    exact(implementation["groups"], groups, "named optimizer payload groups")
    defaults = implementation["defaults"]
    require(
        defaults["lr"] == 0.0003
        and list(defaults["betas"]) == [0.9, 0.999]
        and defaults["eps"] == 1e-8
        and defaults.get("fused") in (None, False)
        and defaults["amsgrad"] is False,
        "optimizer defaults receipt",
    )
    old_run_path = ROOT / "runs/national_dex_continuation_control_s1729_v1/run.json"
    old_run = json.loads(old_run_path.read_text(encoding="utf-8"))
    config = {"inherited_parent_config": old_run["config"], "projection_refit_recipe": recipe}
    exact(run["config"], config, "child config inherits exact parent config")
    exact(child["config"], config, "child payload config")
    exact(
        summary["parent_training_identity"],
        parent["experiment_identity"],
        "parent identity receipt",
    )
    exact(summary["corpus_identity"], parent["corpus_identity"], "corpus receipt")
    exact(
        summary["inherited_architecture"],
        parent["training_metadata"]["model_config"],
        "unchanged raw architecture",
    )
    metadata = {
        "objective": OBJECTIVE,
        "model_config": parent["training_metadata"]["model_config"],
        "parent_checkpoint_sha256": PARENT_SHA,
        "parent_training_steps": 2000,
        "projection_updates": 500,
        "trainable_parameters": [W],
        "train_query_order_sha256": summary["train_query_order_sha256"],
        "record_count": 1637,
        "recipe_sha256": summary["recipe_sha256"],
        "frozen_state_sha256": formatted_hash({k: v for k, v in before.items() if k != W}),
        "final_post_update_loss": training["final_post_update_loss"],
        "mean_bce_diagnostic": training["mean_bce_diagnostic"],
    }
    exact(run["training_metadata"], metadata, "new training objective/lineage")
    exact(child["training_metadata"], metadata, "checkpoint authoritative objective/lineage")
    identity = {
        **parent["experiment_identity"],
        "source_commit": summary["workspace_provenance"]["commit"],
        "resolved_config_hash": formatted_hash(config),
        "evaluator_version": EVALUATOR,
        "environment": {
            **summary["environment"],
            "source_archive_sha256": SOURCE_SHA,
            "configs_archive_sha256": CONFIGS_SHA,
            "runner_sha256": summary["script_sha256"],
            "mean_helper_sha256": MEAN_RUNNER_SHA,
            "recipe_sha256": summary["recipe_sha256"],
        },
        "checkpoint_hash": None,
        "seeds": [1729],
        "repetitions": 1,
    }
    for saved in (
        run["identity"],
        child["experiment_identity"],
        training["training_identity"],
        summary["child_training_identity"],
    ):
        exact(saved, identity, "child experiment identity")
    require(child["split_hash"] == summary["split_hash"] == SPLIT_SHA, "child split")
    exact(child["corpus_identity"], parent["corpus_identity"], "child corpus")
    finite(training["refit_seconds"])
    require(training["refit_seconds"] >= 0, "refit timing")
    return {
        "parent_checkpoint_sha256": PARENT_SHA,
        "child_checkpoint_sha256": training["checkpoint_sha256"],
        "child_sidecar_sha256": training["checkpoint_sidecar_sha256"],
        "global_step": 500,
        "parent_training_steps": 2000,
        "changed_tensor_names": [W],
        "state_tensor_count": len(before),
        "train_query_count": 1637,
        "initial_pre_update_loss": training["history"][0]["pre_update_loss"],
        "final_post_update_loss": training["final_post_update_loss"],
        "updates_verified": 500,
        "optimizer_step": 500,
        "refit_seconds": training["refit_seconds"],
        "payload_loaded_on": "cpu",
    }


def provenance(summary, folder, inherited):
    require(
        summary["campaign_version"] == "projection-worst-boundary-v1"
        and summary["complete"] is True
        and summary["final_identity_check"] is True,
        "complete campaign",
    )
    require(
        summary["acceptance"] is False
        and summary["protected_test_used"] is False
        and summary["standard_serving_supported"] is False,
        "scope/acceptance boundary",
    )
    require("error" not in summary, "campaign failure")
    exact(summary["mean_helper_sha256"], MEAN_RUNNER_SHA, "mean primary helper binding")
    exact(
        [
            summary[k]
            for k in (
                "objective",
                "evaluator",
                "plan_sha256",
                "script_sha256",
                "helper_sha256",
                "recipe_sha256",
            )
        ],
        [OBJECTIVE, EVALUATOR, PLAN_SHA, RUNNER_SHA, HELPER_SHA, RECIPE_SHA],
        "frozen primary identities",
    )
    bind(ROOT / "scripts/refit_worst_boundary_projection.py", RUNNER_SHA)
    bind(ROOT / "docs/experiments/2026-09-25-projection-worst-boundary-plan.md", PLAN_SHA)
    bind(ROOT / "configs/experiments/projection_worst_boundary_v1.json", RECIPE_SHA)
    manifest(summary["input_sha256"])
    inputs = {str(Path(p).resolve()): h for p, h in summary["input_sha256"].items()}
    required = {
        PARENT_SHA,
        REFERENCE_SHA,
        REFERENCE_AUDIT_SHA,
        REFERENCE_DECISION_SHA,
        SOURCE_SHA,
        CONFIGS_SHA,
        HELPER_SHA,
        PLAN_SHA,
        RUNNER_SHA,
        RECIPE_SHA,
        RUNNER_RECEIPT_SHA,
        MEAN_RUNNER_SHA,
        MEAN_SUMMARY_SHA,
        MEAN_AUDIT_SHA,
        MEAN_DECISION_SHA,
        MEAN_CHILD_SHA,
        MEAN_CHECKPOINT_SHA,
        MEAN_PLAN_SHA,
        MEAN_RECIPE_SHA,
        AUDITOR_HELPER_SHA,
    }
    require(required <= set(inputs.values()), "required pinned input not consumed")
    for path, digest in inherited["input_sha256"].items():
        relative = Path(path).resolve().relative_to(ROOT)
        if (
            relative.parts[0] == "data"
            or "national_dex_continuation_control_s1729_v1" in relative.parts
        ):
            require(
                inputs.get(str(Path(path).resolve())) == digest,
                "input data/parent identity differs",
            )
    names = (
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
    )
    exact(
        sorted(str(Path(p).resolve()) for p in summary["snapshot_sha256"]),
        sorted(str(folder / ("summary." + n)) for n in names),
        "snapshot inventory",
    )
    manifest(summary["snapshot_sha256"])
    for name, digest in [
        ("script.py", RUNNER_SHA),
        ("mean-helper.py", MEAN_RUNNER_SHA),
        ("helper.py", HELPER_SHA),
        ("plan.md", PLAN_SHA),
        ("recipe.json", RECIPE_SHA),
    ]:
        bind(folder / ("summary." + name), digest)
    exact(
        json.loads((folder / "summary.inputs.json").read_text(encoding="utf-8")),
        summary["input_sha256"],
        "input snapshot",
    )
    recipe = read(folder / "summary.recipe.json", RECIPE_SHA)
    recipe_contract(recipe)
    receipt = read(folder / "summary.test-receipt.json", RUNNER_RECEIPT_SHA)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["experiment_training_executed"] is False
        and receipt["tested_script_sha256"] == RUNNER_SHA
        and receipt["tested_recipe_sha256"] == RECIPE_SHA
        and receipt["tested_mean_helper_sha256"] == MEAN_RUNNER_SHA,
        "focused runner receipt",
    )
    for path, digest in [
        (receipt["stdout"]["path"], receipt["stdout"]["sha256"]),
        (receipt["test_source"], receipt["tested_test_sha256"]),
    ]:
        bind(path, digest)
        require(inputs.get(str(Path(path).resolve())) == digest, "runner tests binding")
    bind(folder / "summary.test-stdout.txt", receipt["stdout"]["sha256"])
    dependencies = {
        str((ROOT / "scripts/refit_membership_projection.py").resolve()): MEAN_RUNNER_SHA,
        str(
            (ROOT / "tests/unit/test_projection_only_refit.py").resolve()
        ): "5207a2632fcfe6b142d008215732a899a00059ec5b2c15d1d1faceb1c89572c6",
        str(
            (ROOT / "configs/experiments/projection_only_refit_v1.json").resolve()
        ): MEAN_RECIPE_SHA,
    }
    exact(receipt["test_dependencies"], dependencies, "shared test dependency pins")
    for path, digest in dependencies.items():
        bind(path, digest)
        require(inputs.get(path) == digest, "shared test input binding")

    fixture = ROOT / "tests/fixtures/projection_refit_runtime_source.zip"
    exact(receipt["fixtures"], {str(fixture.resolve()): SOURCE_SHA}, "portable fixture receipt")
    bind(fixture, SOURCE_SHA)
    require(inputs.get(str(fixture.resolve())) == SOURCE_SHA, "fixture input binding")
    runtime = folder / "runtime"
    source = source_archive(folder / "summary.source.zip", SOURCE_SHA, runtime)
    configs = source_archive(folder / "summary.configs.zip", CONFIGS_SHA, runtime)
    require(
        not ({n.casefold() for n in source} & {n.casefold() for n in configs}),
        "source/config path collision",
    )
    expected = {str((runtime / n).resolve()): h for n, h in (source | configs).items()}
    exact(summary["runtime_files_sha256"], expected, "archived runtime inventory")
    require(
        {
            str(p.resolve())
            for p in runtime.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        == set(expected),
        "extra/missing runtime file",
    )
    origins = summary["module_origins"]
    require(
        {"plm", "plm.training.optim", "plm.training.checkpoint", "plm.model.layers"}
        <= set(origins),
        "required module origins",
    )
    for name, entry in origins.items():
        path = Path(entry["path"]).resolve()
        require(name == "plm" or name.startswith("plm."), "unexpected runtime module")
        require(
            path.is_relative_to(runtime / "src") and expected.get(str(path)) == entry["sha256"],
            "module outside authenticated archive",
        )
        suffix = path.relative_to(runtime / "src").as_posix()
        module = name.replace(".", "/")
        require(suffix in (module + ".py", module + "/__init__.py"), "module/file mismatch")
    exact(summary["environment"], inherited["environment"], "fixed dependency/device environment")
    exact(summary["numerical_settings"], SETTINGS, "fixed FP32 settings")
    workspace = summary["workspace_provenance"]
    require(
        isinstance(workspace["commit"], str) and len(workspace["commit"]) == 40,
        "source commit receipt",
    )
    require(
        hashlib.sha256(workspace["status_porcelain"].encode()).hexdigest()
        == workspace["status_sha256"],
        "dirty status receipt hash",
    )
    require(
        isinstance(workspace["tracked_diff_sha256"], str)
        and len(workspace["tracked_diff_sha256"]) == 64,
        "tracked diff receipt",
    )
    finite(summary["wall_seconds"])
    require(summary["wall_seconds"] >= 0, "wall timing")
    for value in summary["peak_memory"].values():
        require(type(value) is int and value > 0, "positive memory receipt")
    artifacts = (
        "parent.json",
        "child.json",
        "training.json",
        "train-membership.json",
        "comparisons.json",
        "mean-refit-comparator.json",
        "run.json",
        "checkpoint-final.pt",
        "checkpoint-final.pt.json",
    )
    exact(
        sorted(summary["artifact_sha256"]),
        sorted(artifacts),
        "complete campaign artifact inventory",
    )
    for name, digest in summary["artifact_sha256"].items():
        bind(folder / name, digest)
    return recipe


def audit(summary_path):
    summary = read(summary_path, sha(summary_path))
    folder = summary_path.parent
    bind(ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py", AUDITOR_HELPER_SHA)
    old_summary, old_reports = historical_mean()
    inherited, reference, vocabulary, partitions, parent_path, parent_sidecar = upstream()
    recipe = provenance(summary, folder, inherited)
    names = (
        "parent.json",
        "child.json",
        "training.json",
        "run.json",
        "train-membership.json",
        "comparisons.json",
        "mean-refit-comparator.json",
    )
    reports = {name: read(folder / name, summary["artifact_sha256"][name]) for name in names}
    exact(
        reports["mean-refit-comparator.json"],
        historical_comparator_report(old_summary, old_reports),
        "historical comparator full receipt",
    )
    evaluation(
        old_reports["parent.json"], "parent", reference, partitions["validation"], vocabulary
    )
    evaluation(
        old_reports["child.json"],
        "child",
        reference,
        partitions["validation"],
        vocabulary,
        old_reports["parent.json"],
    )
    parent, child = reports["parent.json"], reports["child.json"]
    evaluation(parent, "parent", reference, partitions["validation"], vocabulary)
    evaluation(child, "child", reference, partitions["validation"], vocabulary, parent)
    mean_diagnostic_contract(
        reports["training.json"]["mean_bce_diagnostic"],
        reports["run.json"]["training_metadata"],
        old_reports["training.json"],
    )
    checkpoint = training_audit(
        summary,
        reports["training.json"],
        reports["run.json"],
        reports["train-membership.json"],
        recipe,
        partitions,
        vocabulary,
        parent_path,
        parent_sidecar,
        folder,
    )
    require(
        checkpoint["child_checkpoint_sha256"] not in (PARENT_SHA, MEAN_CHECKPOINT_SHA),
        "distinct sibling derivative",
    )
    derived = comparison_report(child, parent, old_reports["child.json"], reference)
    exact(reports["comparisons.json"], derived, "all paired comparison outputs")
    exact(summary["comparisons"], derived["aggregate"], "summary comparison receipt")
    invariants = {
        "parent_replay_exact": True,
        "train_only": True,
        "initial_mean_bce_replay_exact": True,
        "complete_500_updates": True,
        "frozen_tensors_unchanged": True,
        "child_reload_exact": True,
    }
    fixed = screen_gate(child, invariants)
    exact(summary["gate"], fixed, "independent unchanged strong gate")
    test_dir = ROOT / "runs/learning/projection-worst-boundary-auditor-tests-v1"
    receipt = read(test_dir / "test-receipt.json", sha(test_dir / "test-receipt.json"))
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["model_execution"] is False
        and receipt["new_diagnostic_executed"] is False
        and receipt["tested_auditor_sha256"] == sha(__file__)
        and receipt["tested_recipe_sha256"] == RECIPE_SHA
        and receipt["auditor_helper_sha256"] == AUDITOR_HELPER_SHA
        and receipt["passed_count"] >= 30,
        "frozen auditor tests",
    )
    bind(test_dir / "test_auditor.py", receipt["tests_sha256"])
    bind(test_dir / "stdout.txt", receipt["stdout_sha256"])
    require(BASE_AUDIT.INPUTS == {}, "no mutation of inherited auditor globals")
    manifest(dict(INPUTS))
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(summary_path),
        "script_sha256": sha(__file__),
        "auditor_helper_sha256": AUDITOR_HELPER_SHA,
        "plan_sha256": PLAN_SHA,
        "test_receipt_sha256": sha(test_dir / "test-receipt.json"),
        "parent": {k: parent[k] for k in ("aggregate", "groups", "paired_vs_width8")},
        "child": {
            k: child[k]
            for k in ("aggregate", "groups", "paired_vs_width8", "paired_vs_parent_dense")
        },
        "checkpoint": checkpoint,
        "mean_bce_diagnostic": reports["training.json"]["mean_bce_diagnostic"],
        "comparisons": derived["aggregate"],
        "gate": fixed,
        "input_sha256": dict(INPUTS),
        "limitations": [
            "Independent saved logits/IDs/metrics and CPU checkpoint-byte "
            "inspection; no neural or optimizer execution.",
            "Finite 500-update trace and synchronization/chronology are "
            "authenticated execution receipts, not independent training replication.",
            "New worst-member objective endpoints and diagnostic mean-BCE "
            "endpoints are different quantities and are not interchangeable.",
            "Only W changed from the original parent; historical mean-refit "
            "checkpoint is a comparator, not this sibling's parent.",
            "Shared independent arithmetic helpers were authenticated and "
            "used without modifying their globals; primary arithmetic was not imported.",
            "Dense sets contain no generated EOS and do not establish "
            "serving equivalence or durable weight archival.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()
    summary = args.summary.resolve()
    output = summary.parent / "independent-audit.json"
    require(not output.exists(), "immutable audit output exists")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment")
    result = audit(summary)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "audit_sha256": sha(output),
                "summary_sha256": result["summary_sha256"],
                "parent": result["parent"]["aggregate"],
                "child": result["child"]["aggregate"],
                "gate": result["gate"],
            },
            sort_keys=True,
        )
    )


def mean_comparator_binding(summary, checked, decision):
    require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and summary["objective"] == "plm-projection-only-balanced-bce-v1",
        "complete mean comparator",
    )
    require(
        checked["complete"] is True
        and checked["audit_passed"] is True
        and checked["summary_sha256"] == MEAN_SUMMARY_SHA
        and checked["script_sha256"] == AUDITOR_HELPER_SHA,
        "historical mean audit binding",
    )
    require(
        decision["evidence_accepted"] is True
        and decision["fixed_quality_gate_passed"] is False
        and decision["summary_sha256"] == MEAN_SUMMARY_SHA
        and decision["audit_sha256"] == MEAN_AUDIT_SHA
        and decision["auditor_sha256"] == AUDITOR_HELPER_SHA,
        "historical evidence versus quality decision",
    )
    require(
        summary["artifact_sha256"]["child.json"] == MEAN_CHILD_SHA
        and summary["artifact_sha256"]["checkpoint-final.pt"] == MEAN_CHECKPOINT_SHA,
        "historical child identity",
    )
    require(
        decision["child_checkpoint_sha256"] == MEAN_CHECKPOINT_SHA
        and decision["parent_checkpoint_sha256"] == PARENT_SHA,
        "historical sibling lineage",
    )


def historical_mean():
    folder = ROOT / "runs/learning/projection-only-refit-v1"
    summary = read(folder / "summary.json", MEAN_SUMMARY_SHA)
    checked = read(folder / "independent-audit.json", MEAN_AUDIT_SHA)
    decision = read(folder / "decision.json", MEAN_DECISION_SHA)
    mean_comparator_binding(summary, checked, decision)
    bind(folder / "independent-audit.py", AUDITOR_HELPER_SHA)
    bind(folder / "summary.script.py", MEAN_RUNNER_SHA)
    bind(ROOT / "scripts/refit_membership_projection.py", MEAN_RUNNER_SHA)
    bind(folder / "summary.plan.md", MEAN_PLAN_SHA)
    reports = {
        name: read(folder / name, summary["artifact_sha256"][name])
        for name in (
            "parent.json",
            "child.json",
            "training.json",
            "run.json",
            "train-membership.json",
        )
    }
    bind(folder / "checkpoint-final.pt", MEAN_CHECKPOINT_SHA)
    bind(
        folder / "checkpoint-final.pt.json", summary["artifact_sha256"]["checkpoint-final.pt.json"]
    )
    require(
        reports["child.json"]["aggregate"]["exact_count"] == 122
        and reports["parent.json"]["aggregate"]["exact_count"] == 112,
        "historical dense counts",
    )
    require(
        reports["training.json"]["initial_pre_update_loss"] == MEAN_INITIAL_LOSS,
        "historical initial mean BCE",
    )
    return summary, reports


def mean_diagnostic_contract(diagnostic, metadata, old_training):
    exact(
        diagnostic,
        {
            "initial": diagnostic["initial"],
            "final": diagnostic["final"],
            "initial_replay_exact": True,
            "used_for_updates": False,
        },
        "mean BCE diagnostic flags",
    )
    finite(diagnostic["initial"])
    finite(diagnostic["final"])
    require(
        diagnostic["initial"] == MEAN_INITIAL_LOSS == old_training["initial_pre_update_loss"],
        "exact initial full-batch mean BCE replay",
    )
    require(diagnostic["final"] >= 0, "finite nonnegative final mean BCE")
    exact(metadata["mean_bce_diagnostic"], diagnostic, "mean diagnostic metadata endpoints")
    return diagnostic


def separation_pairs(rows, reference):
    require(len(rows) == len(reference), "separation pair coverage")
    for left, right in zip(rows, reference, strict=True):
        exact(
            [left[k] for k in ("subject", "dimension", "prompt_ids", "expected_set_ids")],
            [right[k] for k in ("subject", "dimension", "prompt_ids", "expected_set_ids")],
            "separation comparator identity",
        )

    def counts(items):
        return {
            "gains": sum(r["strict_separation"] and not q["strict_separation"] for r, q in items),
            "losses": sum(q["strict_separation"] and not r["strict_separation"] for r, q in items),
        }

    pairs = list(zip(rows, reference, strict=True))
    return {
        **counts(pairs),
        "groups": {g: counts([(r, q) for r, q in pairs if r["group"] == g]) for g in GROUPS},
    }


MEAN_RECIPE_SHA = "a395a114f84192cab2e402c62c2d43cfd3cd60173e286c7243f6600b5d420c22"


def recipe_contract(value):
    expected = read(
        ROOT / "runs/learning/projection-only-refit-v1/summary.recipe.json", MEAN_RECIPE_SHA
    )
    expected.update(
        experiment="projection-worst-boundary-v1",
        objective=OBJECTIVE,
        evaluator=EVALUATOR,
        loss_definition="query-mean-half-softplus-negative-amin-true-plus-softplus-amax-negative-fp32-v1",
        mean_bce_diagnostic={
            "initial_expected": MEAN_INITIAL_LOSS,
            "update_points": [0, 500],
            "used_for_updates": False,
        },
    )
    exact(value, expected, "new sole worst-boundary fixed recipe")


def comparison_report(child, parent, historical, wide):
    current = child["responses"]
    aggregates = {}
    changes = []
    for name, reference in [("parent_dense", parent), ("mean_refit_dense", historical)]:
        require(len(reference["responses"]) == len(current), "comparison coverage")
        previous = reference["responses"]
        for row, old in zip(current, previous, strict=True):
            fields = ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_ids")
            exact(
                [row[k] for k in fields], [old[k] for k in fields], "full comparator query identity"
            )
            changes.append(
                {
                    "comparison": name,
                    **{k: row[k] for k in ("index", "subject", "dimension", "group")},
                    "gained_exact": row["metrics"]["exact"] and not old["metrics"]["exact"],
                    "lost_exact": old["metrics"]["exact"] and not row["metrics"]["exact"],
                    "gained_separation": row["strict_separation"] and not old["strict_separation"],
                    "lost_separation": old["strict_separation"] and not row["strict_separation"],
                }
            )
        sep = separation_pairs(current, previous)
        aggregates[name] = {
            "exact": paired(current, [r["selected_set_ids"] for r in previous]),
            "strict_separation": {k: sep[k] for k in ("gains", "losses")},
        }
    aggregates["width8"] = {
        "exact": paired(current, [r["composition"]["selected_set_ids"] for r in wide["responses"]])
    }
    return {"aggregate": aggregates, "per_query": changes}


def historical_comparator_report(summary, reports):
    training = reports["training.json"]
    child = reports["child.json"]
    return {
        "summary_sha256": MEAN_SUMMARY_SHA,
        "audit_sha256": MEAN_AUDIT_SHA,
        "decision_sha256": MEAN_DECISION_SHA,
        "checkpoint_sha256": MEAN_CHECKPOINT_SHA,
        "artifact_sha256": summary["artifact_sha256"],
        "training_history": training["history"],
        "initial_pre_update_loss": training["initial_pre_update_loss"],
        "final_post_update_loss": training["final_post_update_loss"],
        "objective": "plm-projection-only-balanced-bce-v1",
        "historical_reuse": True,
        "aggregate": child["aggregate"],
        "groups": child["groups"],
    }


def lineage_contract(metadata, identity):
    require(
        metadata["objective"] == OBJECTIVE and identity["evaluator_version"] == EVALUATOR,
        "new sibling objective/evaluator",
    )
    require(
        metadata["parent_checkpoint_sha256"] == PARENT_SHA
        and metadata["parent_training_steps"] == 2000
        and metadata["projection_updates"] == 500,
        "original-parent sibling lineage",
    )
    trainable_manifest(metadata["trainable_parameters"])
    require(metadata["record_count"] == 1637, "full train membership count")


if __name__ == "__main__":
    main()
