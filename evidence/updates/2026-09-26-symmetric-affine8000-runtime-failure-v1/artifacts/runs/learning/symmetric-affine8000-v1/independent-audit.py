"""Independent 8000-update symmetric-affine audit; CPU saved-payload inspection.

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
AMENDMENT_SHA = "bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f"
INPUT_ADAPTER = "plm-authenticated-partition-membership-v1"
PLAN_SHA = "d102c6d00ab536b89c0a0021097976fb997831184a4e4c2db82d172a4c61a19a"
AUDITOR_HELPER_SHA = "78d81a14d666549dd1f6419a7d78792073141a59a729135bc32ac84c88c6ce4a"
MEAN_SUMMARY_SHA = "0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61"
MEAN_AUDIT_SHA = "6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc"
MEAN_DECISION_SHA = "da0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6"
AFFINE_SHA = "8c665818bbfb4306f954ce8b0d9cf67e64e1de2b6f883ee087f3fbe8b3037fbe"
HELPER_SHA = "2ab48674b526ec6641a6c2366b669ac76bdeac1e8633331e01c772d4393edd39"
MEAN_RUNNER_SHA = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
PARENT_SHA = "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
MEMBERSHIP_SHA = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"
SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
SPLIT_SHA = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
ARCHITECTURE = "plm-frozen-symmetric-affine-residual-v1"
OBJECTIVE = "plm-affine-residual-balanced-bce-v1"
EVALUATOR = "plm-symmetric-affine8000-screen-v1"
A = "symmetric_bilinear_residual"
SHAPE = [2, 256, 256]
INITIAL_LOSS = 0.003822767175734043
INPUTS = {}
RUNTIME_ALLOWED = set()
AUDITOR_TEST_DEPENDENCIES = (
    "runs/learning/symmetric-affine8000-v1/independent-audit.py",
    "tests/unit/test_symmetric_affine_audit.py",
    "runs/learning/projection-only-refit-v1/independent-audit.py",
    "runs/learning/bilinear-residual-refit-v1/independent-audit.py",
    "runs/learning/bilinear-budget2000-v1/independent-audit.py",
    "docs/experiments/2026-09-26-symmetric-affine-plan.md",
    "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md",
)
BILINEAR_AUDITOR_SHA = "a046a1e45b3c977fc556ecc74358d64e2b9d0463ebf653c7b7f7c86aed61622b"
SCORER_SHA = "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
HISTORICAL_SUMMARY_SHA = "e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8"
HISTORICAL_AUDIT_SHA = "bdf142f48229fbcdb197bce3a019e38c3a65570ae3f8551a9c9a1f14b2198872"
HISTORICAL_DECISION_SHA = "2cadf9dce7ee36cc4095a10a746cc1608d5ce4437724872c1eec1c88e915c195"
HISTORICAL_CHECKPOINT_SHA = "6979013750eb8f2780917f714192850aae411a48490757236a891fea7e351f91"
HISTORICAL_RECIPE_SHA = "f976442315433e32d5c88ce2f4e911509ec5ea56542fc2a247893cefd0ddfefc"
INFERENCE = "plm-symmetric-affine-positive-set-v1"
ARCHITECTURE_DESCRIPTOR = ARCHITECTURE
PARAMETERS = (A, "symmetric_affine_linear", "symmetric_affine_bias")
SHAPES = {A: [2, 256, 256], PARAMETERS[1]: [2, 256], PARAMETERS[2]: [2]}
HISTORICAL_RUNNER_SHA = "cd593c14c429247837c26b09754509d5cfd1dd36adb317f56522a80a78a25122"
HISTORICAL_AUDITOR_SHA = "9ecc66cd6f63b43d6fd13f0a29ec29ce785e4d0ad45b3f861a2e92f406bc7676"
INVARIANTS = {
    "parent_replay_exact",
    "zero_A_u_b_replay_exact",
    "initial_train_loss_exact",
    "completed_8000_updates",
    "original_93_tensors_unchanged",
    "residual_changed",
    "reload_exact",
    "optimizer_reload_exact",
    "train_only_supervision",
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


BUDGET_AUDITOR_SHA = "c674cf8e4a3ab2056ca4a02746ee55ead639e2a10e6b208525cb0e9dc906d1dc"
BUDGET_RUNNER_SHA = "f4376d4f2d838938bb9f50ae0cfa16ef319639fa0ce917c9fea3b842c0018930"


def load_budget_auditor():
    path = ROOT / "runs/learning/bilinear-budget2000-v1/independent-audit.py"
    if sha(path) != BUDGET_AUDITOR_SHA:
        raise ValueError("frozen budget independent helper changed")
    spec = importlib.util.spec_from_file_location("budget8000_independent_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUDGET = load_budget_auditor()


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


def evaluation(report, phase, reference, records, vocabulary, parent=None):
    require(
        report["phase"] == phase
        and report["complete"] is True
        and report["query_count"] == len(report["responses"]) == len(records) == 222,
        "evaluation coverage",
    )
    require(not any(k in report for k in ("error", "failed_observation")), "evaluation failure")
    require(report["exact_parent_replay"] is (phase == "parent"), "parent replay attestation")
    exact(report["product_token_ids"], list(range(1024, 2049)), "product column order")
    finite(report["head_seconds"])
    require(report["head_seconds"] >= 0, "head time")
    rebuilt = []
    for index, (row, old, record) in enumerate(
        zip(report["responses"], reference["responses"], records, strict=True)
    ):
        prompt, truth = query_labels(record, vocabulary)
        logits = row["symmetric_relation_logits"]
        if phase == "parent":
            exact(logits, old["symmetric_relation_logits"], "exact original parent score replay")
        predicted = BASE.predict(logits, prompt[1])
        expected = {
            "index": index,
            "subject": record["subject"],
            "dimension": record["dimension"],
            "prompt_ids": prompt,
            "group": old["group"],
            "expected_set_ids": truth,
            "symmetric_relation_logits": logits,
            "selected_set_ids": predicted,
            "metrics": BASE.metrics(predicted, truth),
            "strict_separation": BASE.separation(logits, truth, prompt[1])["strict_separable"],
        }
        exact(row, expected, "independent full evaluation row")
        rebuilt.append(expected)
    aggregate = BASE.totals(rebuilt)
    groups = {g: BASE.totals([r for r in rebuilt if r["group"] == g]) for g in BASE.GROUPS}
    exact(report["aggregate"], aggregate, "independent aggregate")
    exact(report["groups"], groups, "independent groups")
    if parent is not None:
        exact(
            report["paired_vs_parent_dense"],
            BASE.paired(rebuilt, [r["selected_set_ids"] for r in parent["responses"]]),
            "paired original parent",
        )
    else:
        require("paired_vs_parent_dense" not in report, "unexpected self comparison")
    return rebuilt, aggregate, groups


def allowed_dependency(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), "dependency outside repository")
    relative = path.relative_to(ROOT.resolve()).as_posix()
    require(
        relative in dependency_allowlist() | RUNTIME_ALLOWED,
        "dependency not on explicit selected-partition allowlist: " + relative,
    )
    return path


def dependency_allowlist():
    fixed = {
        "docs/experiments/2026-09-26-symmetric-affine-plan.md",
        "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md",
        "configs/experiments/symmetric_affine8000_v1.json",
        "scripts/refit_symmetric_affine.py",
        "scripts/symmetric_affine_residual.py",
        "scripts/refit_bilinear_residual.py",
        "scripts/refit_membership_projection.py",
        "scripts/refit_bilinear_budget.py",
        "scripts/refit_bilinear_budget8000.py",
        "runs/learning/wide-first-choice-v1/summary.script.py",
        "tests/unit/test_symmetric_affine_audit.py",
        "tests/unit/test_symmetric_affine_runner.py",
        "tests/unit/test_symmetric_affine_residual.py",
        "tests/fixtures/projection_refit_runtime_source.zip",
        "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json",
        "runs/learning/pair-composition-integration-v1/summary.source.zip",
        "runs/learning/pair-composition-integration-v1/summary.configs.zip",
        "runs/learning/projection-only-refit-v1/independent-audit.py",
        "runs/learning/bilinear-residual-refit-v1/independent-audit.py",
        "runs/learning/bilinear-budget2000-v1/independent-audit.py",
    }
    for name in ("checkpoint-final.pt", "checkpoint-final.pt.json", "run.json"):
        fixed.add("runs/national_dex_continuation_control_s1729_v1/" + name)
    for name in (
        "summary.json",
        "independent-audit.json",
        "decision.json",
        "independent-audit.py",
        "training.json",
        "train-membership.json",
        "parent.json",
        "child.json",
        "run.json",
        "checkpoint-final.pt",
        "checkpoint-final.pt.json",
        "summary.recipe.json",
        "summary.source.zip",
        "summary.configs.zip",
    ):
        fixed.add("runs/learning/bilinear-budget8000-v1/" + name)
    for name in (
        "independent-audit.py",
        "summary.json",
        "execution-receipt.json",
        "primary-stdout.txt",
        "parent.json",
        "child.json",
        "zero-replay.json",
        "training.json",
        "train-membership.json",
        "run.json",
        "checkpoint-final.pt",
        "checkpoint-final.pt.json",
        "comparisons.json",
    ):
        fixed.add("runs/learning/symmetric-affine8000-v1/" + name)
    for suffix in (
        "script.py",
        "affine-helper.py",
        "bilinear-helper.py",
        "budget-helper.py",
        "mean-helper.py",
        "helper.py",
        "plan.md",
        "input-amendment.md",
        "recipe.json",
        "source.zip",
        "configs.zip",
        "test-receipt.json",
        "test-stdout.txt",
        "inputs.json",
    ):
        fixed.add("runs/learning/symmetric-affine8000-v1/summary." + suffix)
    for directory in ("symmetric-affine-auditor-tests-v1", "symmetric-affine-runner-tests-v1"):
        for name in (
            "test-receipt.json",
            "receipt.json",
            "stdout.txt",
            "readiness.json",
            "auditor-contract.json",
        ):
            fixed.add("runs/learning/" + directory + "/" + name)
    return fixed


def bind(path, digest):
    path = allowed_dependency(path)
    require(sha(path) == digest, "hash mismatch: " + str(path))
    require(str(path) not in INPUTS or INPUTS[str(path)] == digest, "conflicting input digest")
    INPUTS[str(path)] = digest
    return path


def read(path, digest=None):
    path = allowed_dependency(path)
    return json.loads(bind(path, digest or sha(path)).read_text(encoding="utf-8"))


def load_payload(path, digest):
    import torch

    return torch.load(bind(path, digest), map_location="cpu", weights_only=False)


def compare_states(parent, child):
    """Inspect CPU payload bytes; no model construction or neural execution."""
    import torch

    require(len(parent) == 93 and not (set(parent) & set(PARAMETERS)), "93 original tensors")
    require(set(child) == set(parent) | set(PARAMETERS), "96 state inventory")
    for name, before in parent.items():
        after = child[name]
        require(
            isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor), "tensor state"
        )
        require(before.device.type == after.device.type == "cpu", "CPU state inspection")
        require(before.dtype == after.dtype and before.shape == after.shape, "original schema")
        require(
            torch.isfinite(before).all().item() and torch.isfinite(after).all().item(),
            "finite original state",
        )
        require(tensor_digest(before) == tensor_digest(after), "original tensor bytes: " + name)
    changed = False
    for name, shape in SHAPES.items():
        value = child[name]
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and list(value.shape) == shape,
            "affine FP32 shape: " + name,
        )
        require(torch.isfinite(value).all().item(), "finite residual: " + name)
        changed |= bool(torch.count_nonzero(value).item())
    require(changed, "residual changed")
    original = state_manifest(parent)
    initial = dict(original)
    for name, shape in SHAPES.items():
        initial[name] = {
            "sha256": tensor_digest(torch.zeros(shape, dtype=torch.float32)),
            "shape": shape,
            "dtype": "torch.float32",
        }
    return original, initial, state_manifest(child)


def optimizer_contract(optimizer, implementation):
    """Bind serialized IDs to exact names, shapes, and 8000-step AdamW state."""
    import torch

    require(set(optimizer) == {"state", "param_groups"}, "optimizer payload schema")
    groups = optimizer["param_groups"]
    named_groups = implementation["groups"]
    require(len(groups) == len(named_groups) == 2, "decay and no-decay groups")
    names_by_group = [group["parameter_names"] for group in named_groups]
    exact(
        names_by_group,
        [list(PARAMETERS[:2]), [PARAMETERS[2]]],
        "exact affine optimizer names/order",
    )
    identifiers = [pid for group in groups for pid in group["params"]]
    require(
        len(identifiers) == 3
        and len(set(identifiers)) == 3
        and all(type(x) is int for x in identifiers),
        "three unique integer parameter IDs",
    )
    require(set(optimizer["state"]) == set(identifiers), "exact three optimizer states")
    mapping = {}
    for group, named in zip(groups, named_groups, strict=True):
        require(len(group["params"]) == len(named["parameter_names"]), "named group size")
        exact(
            {k: v for k, v in group.items() if k != "params"},
            {k: v for k, v in named.items() if k != "parameter_names"},
            "optimizer options receipt",
        )
        require(
            group["lr"] == 0.0003
            and list(group["betas"]) == [0.9, 0.999]
            and group["eps"] == 1e-8
            and group["weight_decay"] == 0,
            "fixed AdamW values",
        )
        require(
            group.get("fused") in (None, False)
            and group["amsgrad"] is False
            and group.get("maximize", False) is False
            and group.get("capturable", False) is False
            and group.get("differentiable", False) is False
            and "foreach" in group
            and group["foreach"] is None,
            "ordinary non-fused AdamW",
        )
        for pid, name in zip(group["params"], named["parameter_names"], strict=True):
            mapping[str(pid)] = name
            state = optimizer["state"][pid]
            require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "AdamW state schema")
            step = state["step"]
            require(
                isinstance(step, torch.Tensor)
                and step.device.type == "cpu"
                and step.dtype == torch.float32
                and step.numel() == 1
                and step.item() == 8000,
                "each optimizer state at step 8000",
            )
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                require(
                    isinstance(value, torch.Tensor)
                    and value.device.type == "cpu"
                    and value.dtype == torch.float32
                    and list(value.shape) == SHAPES[name]
                    and torch.isfinite(value).all().item(),
                    "named finite moment shape",
                )
            require((state["exp_avg_sq"] >= 0).all().item(), "nonnegative second moment")
    require(implementation["class"] == "torch.optim.adamw.AdamW", "optimizer class")
    defaults = implementation["defaults"]
    require(
        defaults["lr"] == 0.0003
        and list(defaults["betas"]) == [0.9, 0.999]
        and defaults["eps"] == 1e-8
        and defaults["weight_decay"] == 0.01
        and defaults.get("fused") in (None, False)
        and defaults["amsgrad"] is False
        and "foreach" in defaults
        and defaults["foreach"] is None,
        "optimizer defaults",
    )
    return {"parameter_id_to_name": mapping, "steps": dict.fromkeys(PARAMETERS, 8000.0)}


def loss_contract(history, initial, final):
    require(len(history) == 8000, "exactly 8000 independent updates")
    for number, row in enumerate(history, 1):
        require(type(row["update"]) is int and row["update"] == number, "ordered 8000 updates")
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


def screen_gate(child, invariants):
    aggregate, groups = child["aggregate"], child["groups"]
    checks = {
        "execution_invariants": set(invariants) == INVARIANTS
        and all(v is True for v in invariants.values()),
        "serialization_compatible": aggregate["serialization_compatible"] == 222,
        "exact_above_historical8000": aggregate["exact_count"] > 216,
        "f1_at_least_historical8000": aggregate["f1"] >= 0.9998843626799785,
        "group_exact_nonregression": all(
            groups[group]["exact_count"] >= count
            for group, count in (("COLOR", 103), ("TYPE_single", 51), ("TYPE_dual", 62))
        ),
    }
    return {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "independent_audit_required": True,
        "accepted": False,
    }


state_receipts_contract = OLD.state_receipts_contract


fp32_bytes = OLD.fp32_bytes


def architecture_contract(value):
    exact(value, ARCHITECTURE, "affine architecture identity")


def lineage_contract(metadata, identity):
    architecture_contract(metadata["architecture"])
    require(
        metadata["objective"] == OBJECTIVE
        and metadata["evaluator"] == EVALUATOR
        and identity["evaluator_version"] == EVALUATOR,
        "unchanged objective/new evaluator",
    )
    require(
        metadata["parent_checkpoint_sha256"] == PARENT_SHA
        and metadata["parent_training_steps"] == 2000
        and metadata["residual_updates"] == 8000,
        "original parent and separate new updates",
    )
    exact(metadata["trainable_parameters"], list(PARAMETERS), "all three named residual trainables")
    require(
        metadata["inference"] == INFERENCE and metadata["standard_serving_supported"] is False,
        "inference and unsupported standard serving",
    )
    require(metadata["record_count"] == 1637, "full training partition")


zero_replay_contract = OLD.zero_replay_contract


def hash_manifest(mapping):
    require(isinstance(mapping, dict) and mapping, "nonempty hash manifest")
    for path in mapping:
        allowed_dependency(path)
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
            require(
                name.startswith(("src/", "configs/")) or name in ("pyproject.toml", "uv.lock"),
                "archive source/config scope",
            )
            result[name] = hashlib.sha256(archive.read(name)).hexdigest()
            RUNTIME_ALLOWED.add((destination / name).relative_to(ROOT).as_posix())
            bind(destination / name, result[name])
    return result


def adapter_contract(summary, inherited, partitions):
    require(
        summary["input_adapter_identity"] == INPUT_ADAPTER
        and summary["input_amendment_sha256"] == AMENDMENT_SHA,
        "selected input identity",
    )
    expected = {
        "identity": INPUT_ADAPTER,
        "amendment_sha256": AMENDMENT_SHA,
        "encoding": "five prompt IDs; ascending unique targets; EOS; prompt labels -100",
        "inherited_corpus_identity": True,
        "whole_corpus_revalidated": False,
        "evidence_sha256": {
            name: inherited["artifact_sha256"][name]
            for name in ("train-membership.json", "parent.json", "child.json", "training.json")
        },
        "partition_sha256": {
            name: formatted_hash(
                [
                    {key: row[key] for key in ("input_ids", "labels", "subject", "dimension")}
                    for row in rows
                ]
            )
            for name, rows in partitions.items()
        },
    }
    exact(summary["input_adapter"], expected, "independent selected-partition adapter")
    exact(
        summary["queries"],
        {name: [query_descriptor(row) for row in rows] for name, rows in partitions.items()},
        "selected partition query order",
    )
    exact(summary["queries"], inherited["queries"], "historical selected query identity/order")
    partition_contract(partitions)
    return expected


def partition_contract(partitions):
    require(set(partitions) == {"train", "validation"}, "selected partitions only")
    keys = {}
    for name, rows in partitions.items():
        keys[name] = {(r["subject"], r["dimension"]) for r in rows}
        require(len(keys[name]) == len(rows), "unique selected queries")
        for row in rows:
            subject, dimension = row["subject"], row["dimension"]
            value = int.from_bytes(
                hashlib.sha256(f"1729:{subject}:{dimension}".encode()).digest(), "big"
            ) / float(1 << 256)
            selected = "validation" if value < 0.1 else "test" if value < 0.2 else "train"
            require(selected == name, "query assigned to wrong or protected partition")
    require(not (keys["train"] & keys["validation"]), "disjoint selected partitions")


def records_from_evidence(rows, vocabulary):
    """Rebuild protocol labels from authenticated prior train/validation evidence.

    Protected records are never opened or reconstructed by this auditor.
    """
    records = []
    seen = set()
    for index, row in enumerate(rows):
        require(type(row["index"]) is int and row["index"] == index, "ordered membership index")
        key = (row["subject"], row["dimension"])
        require(key not in seen, "duplicate membership query")
        seen.add(key)
        prompt, truth = row["prompt_ids"], row["expected_set_ids"]
        require(
            key[1] in ("TYPE", "COLOR")
            and isinstance(prompt, list)
            and len(prompt) == 5
            and all(type(i) is int for i in prompt),
            "selected prompt types/dimension",
        )
        require(
            1024 <= prompt[1] <= 2048
            and isinstance(truth, list)
            and 1 <= len(truth) <= 506
            and all(type(i) is int and 1024 <= i <= 2048 for i in truth),
            "selected membership IDs/count",
        )
        ids = prompt + truth + [2]
        record = {
            "subject": key[0],
            "dimension": key[1],
            "input_ids": ids,
            "labels": [-100] * 5 + ids[5:],
            "targets": [vocabulary[i] for i in row["expected_set_ids"]],
        }
        prompt, truth = query_labels(record, vocabulary)
        exact(prompt, row["prompt_ids"], "membership prompt")
        exact(truth, row["expected_set_ids"], "canonical membership labels")
        records.append(record)
    return records


def upstream():
    old_dir = ROOT / "runs/learning/bilinear-budget8000-v1"
    old, _, _, _ = historical_evidence()
    training = read(old_dir / "training.json", old["artifact_sha256"]["training.json"])
    require(training["initial_pre_update_loss"] == INITIAL_LOSS, "historical initial loss")
    vocabulary_path = ROOT / "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json"
    vocabulary = read(vocabulary_path, old["input_sha256"][str(vocabulary_path)])["tokens"]
    old_membership = read(old_dir / "train-membership.json", MEMBERSHIP_SHA)
    saved_parent = read(old_dir / "parent.json", old["artifact_sha256"]["parent.json"])
    partitions = {
        "train": records_from_evidence(old_membership["queries"], vocabulary),
        "validation": records_from_evidence(saved_parent["responses"], vocabulary),
    }
    require(
        len(partitions["train"]) == 1637 and len(partitions["validation"]) == 222,
        "accepted partition coverage",
    )
    require(old["split_hash"] == SPLIT_SHA, "accepted split identity")
    partition_contract(partitions)
    wide = saved_parent
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
        expected["experiment"] == "bilinear-budget8000-v1" and expected["residual_updates"] == 8000,
        "8000 comparator recipe",
    )
    require(expected.pop("trainable_parameter") == A, "historical sole parameter")
    expected.update(
        experiment="symmetric-affine8000-v1",
        architecture=ARCHITECTURE,
        objective=OBJECTIVE,
        inference=INFERENCE,
        evaluator=EVALUATOR,
        trainable_parameters=list(PARAMETERS),
        affine_linear_shape=[2, 256],
        affine_bias_shape=[2],
        affine_scale=16,
        standard_serving_supported=False,
        input_adapter=INPUT_ADAPTER,
        input_amendment_sha256=AMENDMENT_SHA,
        strong_comparator_exact=216,
        strong_comparator_f1=0.9998843626799785,
        strong_comparator_group_exact={"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 62},
    )
    exact(value, expected, "declared scorer-only recipe changes")


def historical_evidence():
    directory = ROOT / "runs/learning/bilinear-budget8000-v1"
    summary = read(directory / "summary.json", HISTORICAL_SUMMARY_SHA)
    audit = read(directory / "independent-audit.json", HISTORICAL_AUDIT_SHA)
    decision = read(directory / "decision.json", HISTORICAL_DECISION_SHA)
    require(
        summary["complete"] is True
        and summary["acceptance"] is False
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == HISTORICAL_SUMMARY_SHA
        and decision["evidence_accepted"] is True
        and decision["summary_sha256"] == HISTORICAL_SUMMARY_SHA
        and decision["audit_sha256"] == HISTORICAL_AUDIT_SHA,
        "accepted historical8000 evidence",
    )
    require(
        audit["script_sha256"] == HISTORICAL_AUDITOR_SHA
        and decision["fixed_quality_gate_passed"] is True
        and summary["script_sha256"] == HISTORICAL_RUNNER_SHA,
        "frozen scorer and preserved accepted historical gate",
    )
    OLD.architecture_contract(summary["architecture"])
    child = read(directory / "child.json", summary["artifact_sha256"]["child.json"])
    training = read(directory / "training.json", summary["artifact_sha256"]["training.json"])
    run = read(directory / "run.json", summary["artifact_sha256"]["run.json"])
    require(
        training["checkpoint_sha256"] == HISTORICAL_CHECKPOINT_SHA
        and training["completed_updates"] == 8000
        and training["complete"] is True
        and training["initial_pre_update_loss"] == INITIAL_LOSS,
        "historical8000 training receipts",
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
        "historical8000_summary_sha256": HISTORICAL_SUMMARY_SHA,
        "historical8000_child_sha256": historical_child_sha,
        "comparison_scope": "saved historical output; no rerun",
        "paired_vs_historical8000": paired,
    }


def receipt_contract(receipt):
    require(
        receipt.get("passed") is True
        and receipt.get("gpu_used") is False
        and receipt.get("experiment_training_executed") is False
        and receipt.get("input_adapter_identity") == INPUT_ADAPTER
        and receipt.get("input_amendment_sha256") == AMENDMENT_SHA,
        "synthetic CPU receipt",
    )
    require(
        type(receipt.get("exit_code")) is int
        and receipt["exit_code"] == 0
        and receipt.get("terminal_completion_observed") is True,
        "observed terminal completion",
    )
    require(
        type(receipt.get("tests_passed")) is int
        and receipt["tests_passed"] > 0
        and type(receipt.get("tests_skipped")) is int
        and receipt["tests_skipped"] == 0,
        "positive tests and zero skips",
    )
    require(bool(receipt["test_dependencies"]), "test dependencies")


def resolve_input(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    require(path.is_relative_to(ROOT.resolve()), "receipt path escapes repository")
    return path


def readiness_contract(reference, inputs):
    require(
        reference["readiness_only"] is True and reference["candidate_audited"] is False,
        "readiness is not candidate audit",
    )
    path = resolve_input(reference["path"])
    require(inputs.get(str(path)) == reference["sha256"], "readiness manifest consumed")
    contract = read(path, reference["sha256"])
    require(
        type(contract["schema_version"]) is int
        and contract["schema_version"] == 1
        and contract["experiment"] == "symmetric-affine8000-v1"
        and contract["plan_sha256"] == PLAN_SHA
        and contract["ready"] is True
        and contract["input_amendment_sha256"] == AMENDMENT_SHA
        and contract["input_adapter_identity"] == INPUT_ADAPTER,
        "readiness identity",
    )
    bound = {}
    for name in ("auditor", "tests", "test_receipt"):
        allowed_dependency(resolve_input(contract[name]["path"]))
    for name in ("auditor", "tests", "test_receipt"):
        ref = contract[name]
        bound[name] = bind(resolve_input(ref["path"]), ref["sha256"])
    require(
        bound["auditor"] == Path(__file__).resolve()
        and contract["auditor"]["sha256"] == sha(__file__),
        "tested auditor identity",
    )
    receipt = read(bound["test_receipt"])
    receipt_contract(receipt)
    require(
        receipt.get("tested_script_sha256") == contract["auditor"]["sha256"],
        "tested auditor source receipt",
    )
    dependencies = {str(resolve_input(k)): v for k, v in receipt["test_dependencies"].items()}
    for name in ("auditor", "tests"):
        require(
            dependencies.get(str(bound[name])) == contract[name]["sha256"],
            "tested artifact dependency",
        )
    require(
        len(dependencies) == len(receipt["test_dependencies"]), "no duplicate dependency aliases"
    )
    require(
        set(dependencies) == {str(ROOT / p) for p in AUDITOR_TEST_DEPENDENCIES},
        "exact auditor test dependencies",
    )
    hash_manifest(dependencies)
    stdout = resolve_input(receipt["stdout"]["path"])
    require(inputs.get(str(stdout)) == receipt["stdout"]["sha256"], "auditor stdout consumed")
    bind(stdout, receipt["stdout"]["sha256"])
    for path, digest in list(INPUTS.items()):
        if path in dependencies or path in {str(v) for v in bound.values()}:
            require(inputs.get(path) == digest, "readiness input binding")
    return sha(bound["test_receipt"])


def provenance(summary, folder, inherited):
    require(
        summary["campaign_version"] == "symmetric-affine8000-v1"
        and summary["complete"] is True
        and summary["final_identity_check"] is True
        and "error" not in summary,
        "completed affine campaign",
    )
    require(
        summary["acceptance"] is False
        and summary["protected_test_used"] is False
        and summary["standard_serving_supported"] is False,
        "scientific boundaries",
    )
    architecture_contract(summary["architecture"])
    require(
        summary["objective"] == OBJECTIVE
        and summary["inference"] == INFERENCE
        and summary["evaluator"] == EVALUATOR,
        "separate scorer identities",
    )
    for key, digest in {
        "plan_sha256": PLAN_SHA,
        "input_amendment_sha256": AMENDMENT_SHA,
        "scorer_sha256": AFFINE_SHA,
        "bilinear_helper_sha256": SCORER_SHA,
        "mean_helper_sha256": MEAN_RUNNER_SHA,
        "budget_helper_sha256": BUDGET_RUNNER_SHA,
        "helper_sha256": HELPER_SHA,
        "historical8000_summary_sha256": HISTORICAL_SUMMARY_SHA,
        "historical8000_audit_sha256": HISTORICAL_AUDIT_SHA,
        "historical8000_decision_sha256": HISTORICAL_DECISION_SHA,
    }.items():
        require(summary[key] == digest, "frozen identity: " + key)
    inputs = summary["input_sha256"]
    hash_manifest(inputs)
    hash_manifest(summary["snapshot_sha256"])
    snapshots = {
        "script.py",
        "affine-helper.py",
        "bilinear-helper.py",
        "budget-helper.py",
        "mean-helper.py",
        "helper.py",
        "plan.md",
        "input-amendment.md",
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
    pins = {
        "script.py": summary["script_sha256"],
        "affine-helper.py": AFFINE_SHA,
        "bilinear-helper.py": SCORER_SHA,
        "budget-helper.py": BUDGET_RUNNER_SHA,
        "mean-helper.py": MEAN_RUNNER_SHA,
        "helper.py": HELPER_SHA,
        "plan.md": PLAN_SHA,
        "input-amendment.md": AMENDMENT_SHA,
        "recipe.json": summary["recipe_sha256"],
        "source.zip": SOURCE_SHA,
        "configs.zip": CONFIGS_SHA,
    }
    for suffix, digest in pins.items():
        bind(folder / ("summary." + suffix), digest)
    require(
        {
            PARENT_SHA,
            SOURCE_SHA,
            CONFIGS_SHA,
            PLAN_SHA,
            AFFINE_SHA,
            SCORER_SHA,
            HISTORICAL_SUMMARY_SHA,
            HISTORICAL_AUDIT_SHA,
            HISTORICAL_DECISION_SHA,
            MEMBERSHIP_SHA,
            summary["script_sha256"],
            summary["recipe_sha256"],
        }
        <= set(inputs.values()),
        "required consumed identities",
    )
    exact(read(folder / "summary.inputs.json"), inputs, "input snapshot")
    recipe = read(folder / "summary.recipe.json", summary["recipe_sha256"])
    recipe_contract(
        recipe,
        read(
            ROOT / "runs/learning/bilinear-budget8000-v1/summary.recipe.json", HISTORICAL_RECIPE_SHA
        ),
    )
    receipt = read(folder / "summary.test-receipt.json")
    receipt_contract(receipt)
    require(
        receipt["tested_script_sha256"] == summary["script_sha256"]
        and receipt["tested_recipe_sha256"] == summary["recipe_sha256"],
        "tested implementation/recipe",
    )
    dependencies = {str(resolve_input(k)): v for k, v in receipt["test_dependencies"].items()}
    required_paths = (
        "scripts/refit_symmetric_affine.py",
        "scripts/symmetric_affine_residual.py",
        "scripts/refit_bilinear_residual.py",
        "scripts/refit_membership_projection.py",
        "scripts/refit_bilinear_budget.py",
        "scripts/refit_bilinear_budget8000.py",
        "runs/learning/wide-first-choice-v1/summary.script.py",
        "tests/unit/test_symmetric_affine_runner.py",
        "tests/unit/test_symmetric_affine_residual.py",
        "tests/fixtures/projection_refit_runtime_source.zip",
    )
    require(
        {str(ROOT / p) for p in required_paths} == set(dependencies),
        "required primary test dependencies",
    )
    hash_manifest(dependencies)
    for path, digest in dependencies.items():
        require(inputs.get(path) == digest, "tested dependency consumed")
    require(
        dependencies[str(ROOT / "scripts/refit_symmetric_affine.py")] == summary["script_sha256"],
        "tested exact runner",
    )
    bind(resolve_input(receipt["stdout"]["path"]), receipt["stdout"]["sha256"])
    bind(folder / "summary.test-stdout.txt", receipt["stdout"]["sha256"])
    readiness_contract(summary["auditor_readiness"], inputs)
    runtime = folder / "runtime"
    source = archive_inventory(folder / "summary.source.zip", SOURCE_SHA, runtime)
    configs = archive_inventory(folder / "summary.configs.zip", CONFIGS_SHA, runtime)
    require(not (set(source) & set(configs)), "disjoint runtime archives")
    inventory = {str(runtime / name): digest for name, digest in (source | configs).items()}
    exact(summary["runtime_files_sha256"], inventory, "runtime inventory")
    require(
        {str(p) for p in runtime.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        == set(inventory),
        "no extra runtime files",
    )
    origins = summary["module_origins"]
    require(
        {"plm", "plm.training.optim", "plm.training.checkpoint", "plm.model.layers"}
        <= set(origins),
        "runtime origins",
    )
    for name, entry in origins.items():
        path = Path(entry["path"])
        require(name == "plm" or name.startswith("plm."), "runtime namespace")
        require(
            path.is_relative_to(runtime / "src") and inventory.get(str(path)) == entry["sha256"],
            "module hash",
        )
        module = name.replace(".", "/")
        require(
            path.relative_to(runtime / "src").as_posix()
            in (module + ".py", module + "/__init__.py"),
            "module path",
        )
    exact(summary["environment"], inherited["environment"], "accepted environment")
    exact(summary["numerical_settings"], inherited["numerical_settings"], "FP32 numerical settings")
    workspace = summary["workspace_provenance"]
    require(
        len(workspace["commit"]) == 40
        and hashlib.sha256(workspace["status_porcelain"].encode()).hexdigest()
        == workspace["status_sha256"]
        and len(workspace["tracked_diff_sha256"]) == 64,
        "workspace receipts",
    )
    finite(summary["wall_seconds"])
    require(
        summary["wall_seconds"] >= 0
        and all(type(v) is int and v > 0 for v in summary["peak_memory"].values()),
        "time/memory",
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
    require(set(summary["artifact_sha256"]) == artifacts, "artifact inventory")
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
        and training["completed_updates"] == 8000,
        "complete unchanged objective with new budget",
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
    exact(training["trainable_parameters"], list(PARAMETERS), "three named trainables")
    require(
        training["fit_complete"] is True
        and training["phase"] == "complete"
        and training["optimizer_steps_attempted"] == 8000
        and training["frozen_state_exact"] is True,
        "completed fresh fit chronology",
    )
    queries = {
        name: [query_descriptor(r) for r in partitions[name]] for name in ("train", "validation")
    }
    exact(summary["queries"], queries, "ordered supervised partitions")
    adapter_contract(summary, _old, partitions)
    exact(summary["validation_update_points"], [0, 8000], "fixed endpoints only")
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
        and child["global_step"] == 8000,
        "parent/new update distinction",
    )
    parent_state, initial_state, final_state = compare_states(parent["model"], child["model"])
    exact(parent_state, old_training["parent_state_before"], "authenticated original 93 tensors")
    state_receipts_contract(training, parent_state, initial_state, final_state)
    require(training["optimizer"]["name"] == "adamw", "optimizer family")
    exact(training["optimizer"]["config"], recipe["optimizer"], "optimizer config recipe")
    impl = training["optimizer_implementation"]
    optimizer_result = optimizer_contract(
        child["optimizer"],
        {
            "class": impl["module"] + "." + impl["class"],
            "groups": training["optimizer"]["groups"],
            "defaults": impl["defaults"],
        },
    )
    exact(training["optimizer_final"], optimizer_result, "final named optimizer states")
    require(child["scheduler"] is None, "no new scheduler")
    config = {
        "inherited_parent_config": old_run["config"],
        "affine_residual_architecture": ARCHITECTURE_DESCRIPTOR,
        "affine_residual_recipe": recipe,
        "implementation_identity": {
            "runner_sha256": summary["script_sha256"],
            "scorer_sha256": AFFINE_SHA,
            "bilinear_helper_sha256": SCORER_SHA,
            "mean_helper_sha256": MEAN_RUNNER_SHA,
            "budget_helper_sha256": BUDGET_RUNNER_SHA,
            "source_archive_sha256": SOURCE_SHA,
            "configs_archive_sha256": CONFIGS_SHA,
            "input_adapter_identity": INPUT_ADAPTER,
            "input_amendment_sha256": AMENDMENT_SHA,
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
        "residual_updates": 8000,
        "trainable_parameters": list(PARAMETERS),
        "inference": INFERENCE,
        "input_adapter_identity": INPUT_ADAPTER,
        "input_amendment_sha256": AMENDMENT_SHA,
        "train_query_order_sha256": summary["train_query_order_sha256"],
        "record_count": 1637,
        "recipe_sha256": summary["recipe_sha256"],
        "runner_sha256": summary["script_sha256"],
        "scorer_sha256": AFFINE_SHA,
        "bilinear_helper_sha256": SCORER_SHA,
        "parent_state_sha256": formatted_hash(parent_state),
        "initial_state_sha256": formatted_hash(initial_state),
        "initial_pre_update_loss": training["initial_pre_update_loss"],
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
            "input_adapter_identity": INPUT_ADAPTER,
            "input_amendment_sha256": AMENDMENT_SHA,
            "runner_sha256": summary["script_sha256"],
            "scorer_sha256": AFFINE_SHA,
            "bilinear_helper_sha256": SCORER_SHA,
            "recipe_sha256": summary["recipe_sha256"],
            "mean_helper_sha256": MEAN_RUNNER_SHA,
            "budget_helper_sha256": BUDGET_RUNNER_SHA,
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
        "global_step": 8000,
        "parent_training_steps": 2000,
        "original_tensor_count": 93,
        "child_tensor_count": 96,
        "unchanged_original_tensor_count": 93,
        "new_trained_tensors": list(PARAMETERS),
        "initial_residual": {n: initial_state[n] for n in PARAMETERS},
        "trained_residual": {n: final_state[n] for n in PARAMETERS},
        "train_query_count": 1637,
        "initial_pre_update_loss": training["initial_pre_update_loss"],
        "final_post_update_loss": training["final_post_update_loss"],
        "updates_verified": 8000,
        "optimizer_step": 8000,
        "refit_seconds": training["refit_seconds"],
        "payload_loaded_on": "cpu",
        "scorer_sha256": AFFINE_SHA,
        "bilinear_helper_sha256": SCORER_SHA,
        "runner_sha256": summary["script_sha256"],
    }


def execution_contract(receipt, summary_hash, stdout_hash):
    require(
        type(receipt["exit_code"]) is int and receipt["exit_code"] == 0, "primary terminal exit"
    )
    require(
        receipt["terminal_completion_observed_by_primary"] is True, "observed primary completion"
    )
    require(receipt["summary_sha256"] == summary_hash, "terminal summary identity")
    require(receipt["stdout_sha256"] == stdout_hash, "terminal stdout identity")


def audit(summary_path):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment")
    summary = read(summary_path)
    folder = summary_path.parent
    terminal = read(folder / "execution-receipt.json")
    stdout = bind(folder / "primary-stdout.txt", terminal["stdout_sha256"])
    execution_contract(terminal, sha(summary_path), sha(stdout))
    bind(ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py", AUDITOR_HELPER_SHA)
    bind(
        ROOT / "runs/learning/bilinear-residual-refit-v1/independent-audit.py", BILINEAR_AUDITOR_SHA
    )
    bind(ROOT / "runs/learning/bilinear-budget2000-v1/independent-audit.py", BUDGET_AUDITOR_SHA)
    context = upstream()
    historical_summary, historical_child, _historical_training, historical_run = (
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
    exact(
        reports["comparisons.json"],
        comparison,
        "independent historical8000 pairing report",
    )
    recipe_contract(recipe, historical_run["config"]["bilinear_residual_recipe"])
    checkpoint = training_audit(summary, reports, recipe, context, folder)
    invariant_names = (
        "parent_replay_exact",
        "zero_A_u_b_replay_exact",
        "initial_train_loss_exact",
        "completed_8000_updates",
        "original_93_tensors_unchanged",
        "residual_changed",
        "reload_exact",
        "optimizer_reload_exact",
        "train_only_supervision",
    )
    invariants = dict.fromkeys(invariant_names, True)
    exact(summary["invariants"], invariants, "checked execution invariants")
    gate = screen_gate(child, invariants)
    exact(summary["gate"], gate, "independent strengthened quality gate")
    exact(summary["parent"], parent["aggregate"], "summary parent metrics")
    child_summary = {k: child[k] for k in ("aggregate", "groups", "paired_vs_parent_dense")}
    child_summary["paired_vs_historical8000"] = comparison["paired_vs_historical8000"]
    exact(summary["child"], child_summary, "summary child metrics/comparisons")
    readiness = read(
        resolve_input(summary["auditor_readiness"]["path"]), summary["auditor_readiness"]["sha256"]
    )
    auditor_receipt_path = resolve_input(readiness["test_receipt"]["path"])
    require(
        BASE.INPUTS == {} and OLD.INPUTS == {} and BUDGET.INPUTS == {},
        "authenticated helper globals unchanged",
    )
    hash_manifest(dict(INPUTS))
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(summary_path),
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN_SHA,
        "test_receipt_sha256": sha(auditor_receipt_path),
        "independent_helper_sha256": AUDITOR_HELPER_SHA,
        "bilinear_independent_helper_sha256": BILINEAR_AUDITOR_SHA,
        "budget_independent_helper_sha256": BUDGET_AUDITOR_SHA,
        "parent": {k: parent[k] for k in ("aggregate", "groups")},
        "child": child_summary,
        "zero_initialization_replay": zero,
        "comparisons": {
            "versus_parent_dense": child["paired_vs_parent_dense"],
            "versus_historical8000": comparison["paired_vs_historical8000"],
        },
        "checkpoint": checkpoint,
        "gate": gate,
        "input_sha256": dict(INPUTS),
        "limitations": [
            "Saved-evidence and CPU checkpoint inspection; no neural forward or repeated training.",
            "Loss/update receipts are authenticated, not independently regenerated.",
            "Corpus/split identity inherited; no whole-corpus revalidation.",
            "All 93 parent tensors match; saved A/u/b and three optimizer moments are inspected.",
            "Reload/optimizer chronology is authenticated, not independently replayed.",
            "Saved logits/zero-initialization hashes are bound, not recomputed on CUDA.",
            "The inherited decoder output does not represent the new architecture/scorer.",
            "Dense validation does not prove serving, protected quality or durable weight backup.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        import sys

        require("torch" not in sys.modules, "metadata preflight must remain torch-free")
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "torch_imported": False,
                    "candidate_audited": False,
                    "plan_sha256": PLAN_SHA,
                    "input_amendment_sha256": AMENDMENT_SHA,
                    "input_adapter_identity": INPUT_ADAPTER,
                    "script_sha256": sha(__file__),
                    "experiment": "symmetric-affine8000-v1",
                    "required_state_tensors": 96,
                    "optimizer_states": 3,
                    "updates": 8000,
                    "architecture": ARCHITECTURE,
                    "inference": INFERENCE,
                    "evaluator": EVALUATOR,
                },
                sort_keys=True,
            )
        )
        return
    require(args.summary is not None, "summary required for actual audit")
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
