"""Independent saved-evidence projection refit audit; CPU payload inspection only.

No primary prediction/aggregation imports and no neural or optimizer execution.
Checkpoint bytes prove saved state; a trace is not independent training replication.
"""

import argparse
import hashlib
import json
import math
import os
import struct
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN_SHA = "2ac4d157ad8c079561d642199fec8de2c3da1866819baa48f52080e9b0f39bfe"
PARENT_SHA = "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
PARENT_CONFIG_SHA = "6e0314a34ddec1ae31eb1b591c14e289339b950a1894f7e66f29ecc3a1508148"
REFERENCE_SHA = "1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183"
REFERENCE_AUDIT_SHA = "eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95"
REFERENCE_DECISION_SHA = "70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601"
SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
SPLIT_SHA = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
W = "symmetric_relation_projection.weight"
OBJECTIVE = "plm-projection-only-balanced-bce-v1"
EVALUATOR = "plm-projection-refit-screen-v1"
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
INPUTS = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def exact(actual, expected, message):
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        message,
    )


def bind(path, digest):
    path = Path(path).resolve()
    require(sha(path) == digest, f"hash mismatch: {path}")
    require(str(path) not in INPUTS or INPUTS[str(path)] == digest, "input hash conflict")
    INPUTS[str(path)] = digest
    return path


def read(path, digest):
    return json.loads(bind(path, digest).read_text(encoding="utf-8"))


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite or nonnumeric scalar")


def vector(values):
    require(isinstance(values, list) and len(values) == 1025, "head width")
    for value in values:
        require(type(value) is float and math.isfinite(value), "nonfinite/nonfloat head value")
        require(struct.unpack("!f", struct.pack("!f", value))[0] == value, "head value not FP32")


def predict(values, subject):
    """Labels, query group, and cardinality never enter the prediction rule."""
    vector(values)
    require(type(subject) is int and 1024 <= subject <= 2048, "subject product ID")
    return [i for i in range(1024, 2049) if i != subject and values[i - 1024] > 0]


def metrics(predicted, truth):
    require(predicted == sorted(set(predicted)), "canonical prediction")
    require(truth == sorted(set(truth)) and truth, "nonempty canonical truth")
    hits = len(set(predicted) & set(truth))
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth)
    return {
        "exact": predicted == truth,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "set_size": len(predicted),
        "false_positive_count": len(predicted) - hits,
        "false_negative_count": len(truth) - hits,
        "serialization_compatible": 1 <= len(predicted) <= 506,
    }


def separation(values, truth, subject):
    vector(values)
    require(
        truth and subject not in truth and truth == sorted(set(truth)), "positive labels/subject"
    )
    positives = [values[i - 1024] for i in truth]
    negatives = [values[i - 1024] for i in range(1024, 2049) if i != subject and i not in truth]
    require(negatives, "nonempty negative group")
    low, high = min(positives), max(negatives)
    return {
        "min_true_logit": low,
        "max_negative_logit": high,
        "gap": low - high,
        "strict_separable": low > high,
        "zero_threshold_correct": low > 0 and high <= 0,
    }


def split_membership(records):
    """Reconstruct query assignment and order, never predict on protected records."""
    partitions = {"train": [], "validation": [], "test": []}
    seen = set()
    for record in records:
        key = (record["subject"], record["dimension"])
        require(key not in seen and key[1] in ("TYPE", "COLOR"), "duplicate/invalid corpus query")
        seen.add(key)
        digest = hashlib.sha256(f"1729:{key[0]}:{key[1]}".encode()).digest()
        value = int.from_bytes(digest, "big") / float(1 << 256)
        name = "validation" if value < 0.1 else "test" if value < 0.2 else "train"
        partitions[name].append(record)
    identity = {
        "algorithm": "sha256-threshold-v2",
        "seed": 1729,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        **{k: [[r["subject"], r["dimension"]] for r in rows] for k, rows in partitions.items()},
    }
    return partitions, canonical_hash(identity)


def query_labels(record, vocabulary):
    ids = record["input_ids"]
    require(
        len(ids) >= 7
        and ids[0] == 1
        and ids[2:5] == [32 if record["dimension"] == "TYPE" else 33, 34, 5]
        and ids[-1] == 2,
        "record protocol",
    )
    require(vocabulary[ids[1]] == record["subject"], "record subject")
    require(all(type(i) is int and 1024 <= i <= 2048 for i in ids[5:-1]), "record product labels")
    exact(record["labels"], [-100] * 5 + ids[5:], "record loss mask")
    exact(record["targets"], [vocabulary[i] for i in ids[5:-1]], "record target mapping")
    truth = sorted(set(ids[5:-1]))
    require(ids[1] not in truth and truth, "truth self-exclusion")
    return ids[:5], truth


def trace_contract(trace):
    require(len(trace) == 500, "exactly 500 updates")
    for index, row in enumerate(trace, 1):
        require(type(row["update"]) is int and row["update"] == index, "sequential update count")
        finite(row["pre_update_loss"])
        require(row["pre_update_loss"] >= 0, "nonnegative loss")
        require(
            row["gradient_finite"] is True and row["parameters_finite"] is True,
            "finite update receipt",
        )


def tensor_digest(tensor):
    # Import torch only inside authenticated payload inspection or synthetic fixtures.
    import torch

    require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu", "CPU state tensor")
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def compare_states(parent, child):
    import torch

    require(set(parent) == set(child) and W in parent, "state inventory")
    require(tuple(parent[W].shape) == tuple(child[W].shape) == (256, 256), "projection shape")
    before, after = {}, {}
    for name in parent:
        left, right = parent[name], child[name]
        require(
            left.dtype == right.dtype and left.shape == right.shape, "state shape/dtype changed"
        )
        require(
            torch.isfinite(left).all().item() and torch.isfinite(right).all().item(),
            "nonfinite state tensor",
        )
        before[name], after[name] = tensor_digest(left), tensor_digest(right)
        if name != W:
            require(before[name] == after[name], "frozen tensor changed: " + name)
        else:
            require(
                left.dtype == torch.float32 and before[name] != after[name], "W not changed in FP32"
            )
    return before, after


def optimizer_contract(optimizer):
    import torch

    require(
        isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"},
        "optimizer payload",
    )
    groups = optimizer["param_groups"]
    require(
        len(groups) == 2 and len(groups[0]["params"]) == 1 and groups[1]["params"] == [],
        "only W optimized",
    )
    for group in groups:
        require(
            group["lr"] == 0.0003
            and list(group["betas"]) == [0.9, 0.999]
            and group["eps"] == 1e-8
            and group["weight_decay"] == 0,
            "fixed AdamW hyperparameters",
        )
        require(
            group.get("fused") in (None, False) and group.get("amsgrad") is False, "nonfused AdamW"
        )
        require(not group.get("maximize", False), "descent optimizer")
    param = groups[0]["params"][0]
    require(set(optimizer["state"]) == {param}, "only W optimizer state")
    state = optimizer["state"][param]
    require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "AdamW state fields")
    require(
        isinstance(state["step"], torch.Tensor) and state["step"].item() == 500, "optimizer step500"
    )
    for key in ("exp_avg", "exp_avg_sq"):
        tensor = state[key]
        require(
            tuple(tensor.shape) == (256, 256)
            and tensor.dtype == torch.float32
            and tensor.device.type == "cpu"
            and torch.isfinite(tensor).all().item(),
            "optimizer moment shape/finite",
        )
    require((state["exp_avg_sq"] >= 0).all().item(), "nonnegative second moment")


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


def sidecar_matches(payload, sidecar, digest):
    require(
        sidecar["schema_version"] == 1 and sidecar["checkpoint_hash"] == digest,
        "checkpoint sidecar digest",
    )
    for key in (
        "global_step",
        "config",
        "experiment_identity",
        "corpus_identity",
        "split_hash",
        "training_metadata",
    ):
        exact(sidecar[key], payload[key], "sidecar payload " + key)


def load_payload(path, digest):
    import torch

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment required")
    # Both files are authenticated local experiment checkpoints, including RNG pickles.
    # No module or callable is imported from the primary evaluator.
    bind(path, digest)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict) and "model" in payload, "checkpoint payload")
    return payload


def loss_endpoints(trace, initial_loss, final_loss):
    trace_contract(trace)
    finite(initial_loss)
    finite(final_loss)
    require(initial_loss >= 0 and final_loss >= 0, "nonnegative loss endpoints")
    exact(initial_loss, trace[0]["pre_update_loss"], "initial loss trace binding")


def trainable_manifest(names):
    exact(names, [W], "exactly named W is trainable")


def formatted_hash(value):
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    ).hexdigest()


def totals(rows):
    require(bool(rows), "nonempty aggregate")
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


def paired(rows, reference_sets):
    require(len(rows) == len(reference_sets), "paired coverage")
    values = [
        (r, set(ids) == set(r["expected_set_ids"]))
        for r, ids in zip(rows, reference_sets, strict=True)
    ]

    def counts(items):
        return {
            "gains": sum(r["metrics"]["exact"] and not old for r, old in items),
            "losses": sum(old and not r["metrics"]["exact"] for r, old in items),
        }

    return {
        **counts(values),
        "groups": {g: counts([(r, old) for r, old in values if r["group"] == g]) for g in GROUPS},
    }


def evaluation(report, phase, reference, records, vocabulary, parent=None):
    require(
        report["phase"] == phase
        and report["complete"] is True
        and report["query_count"] == 222
        and len(report["responses"]) == 222,
        "evaluation completeness",
    )
    require(
        not any(k in report for k in ("error", "failed_observation")), "evaluation failure record"
    )
    require(report["exact_parent_replay"] is (phase == "parent"), "parent replay receipt")
    exact(report["product_token_ids"], list(range(1024, 2049)), "evaluation columns")
    finite(report["head_seconds"])
    require(report["head_seconds"] >= 0, "head timing")
    rebuilt = []
    for index, (row, old, record) in enumerate(
        zip(report["responses"], reference["responses"], records, strict=True)
    ):
        prompt, truth = query_labels(record, vocabulary)
        values = row["symmetric_relation_logits"]
        if phase == "parent":
            exact(values, old["symmetric_relation_logits"], "exact parent historical logits")
        selected = predict(values, prompt[1])
        diagnostic = separation(values, truth, prompt[1])
        expected = {
            "index": index,
            "subject": record["subject"],
            "dimension": record["dimension"],
            "prompt_ids": prompt,
            "group": old["group"],
            "expected_set_ids": truth,
            "symmetric_relation_logits": values,
            "selected_set_ids": selected,
            "metrics": metrics(selected, truth),
            "strict_separation": diagnostic["strict_separable"],
        }
        exact(row, expected, "independent complete evaluation row")
        rebuilt.append(expected)
    aggregate = totals(rebuilt)
    groups = {g: totals([r for r in rebuilt if r["group"] == g]) for g in GROUPS}
    exact(report["aggregate"], aggregate, "dense aggregate")
    exact(report["groups"], groups, "dense groups")
    versus_wide = paired(
        rebuilt, [r["composition"]["selected_set_ids"] for r in reference["responses"]]
    )
    exact(report["paired_vs_width8"], versus_wide, "paired strong comparison")
    if parent is not None:
        exact(
            report["paired_vs_parent_dense"],
            paired(rebuilt, [r["selected_set_ids"] for r in parent["responses"]]),
            "paired dense comparison",
        )
    else:
        require("paired_vs_parent_dense" not in report, "unexpected self comparison")
    return rebuilt, aggregate, groups


def screen_gate(child, invariants):
    aggregate, groups = child["aggregate"], child["groups"]
    checks = {
        "execution_invariants": bool(invariants) and all(v is True for v in invariants.values()),
        "serialization_compatible": aggregate["serialization_compatible"] == 222,
        "exact_above_201": aggregate["exact_count"] > 201,
        "f1_at_least_strong_comparator": aggregate["f1"] >= 0.9799255176742276,
        "group_exact_nonregression": all(
            groups[g]["exact_count"] >= n
            for g, n in [("COLOR", 103), ("TYPE_single", 50), ("TYPE_dual", 48)]
        ),
    }
    return {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "independent_audit_required": True,
        "accepted": False,
    }


def recipe_contract(value):
    expected = {
        "schema_version": 1,
        "experiment": "projection-only-refit-v1",
        "objective": OBJECTIVE,
        "evaluator": EVALUATOR,
        "seed": 1729,
        "parent_training_steps": 2000,
        "projection_updates": 500,
        "train_query_count": 1637,
        "validation_query_count": 222,
        "trainable_parameter": W,
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
    exact(value, expected, "fixed standalone recipe")


def state_manifest(state):
    return {
        name: {
            "sha256": tensor_digest(tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in sorted(state.items())
    }


def query_descriptor(record):
    return {k: record[k] for k in ("subject", "dimension")} | {
        "prompt_ids": record["input_ids"][:5]
    }


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


RUNNER_SHA = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
RUNNER_RECEIPT_SHA = "17e61b9c9f6182defb52534c4962d378a33569396bbec2c504563c7fb5f2418f"
RECIPE_SHA = "a395a114f84192cab2e402c62c2d43cfd3cd60173e286c7243f6600b5d420c22"
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


def provenance(summary, folder, inherited):
    require(
        summary["campaign_version"] == "projection-only-refit-v1"
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
    bind(ROOT / "scripts/refit_membership_projection.py", RUNNER_SHA)
    bind(ROOT / "docs/experiments/2026-09-25-projection-only-refit-plan.md", PLAN_SHA)
    bind(ROOT / "configs/experiments/projection_only_refit_v1.json", RECIPE_SHA)
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
        and receipt["tested_recipe_sha256"] == RECIPE_SHA,
        "focused runner receipt",
    )
    for path, digest in [
        (receipt["stdout"]["path"], receipt["stdout"]["sha256"]),
        (receipt["test_source"], receipt["tested_test_sha256"]),
    ]:
        bind(path, digest)
        require(inputs.get(str(Path(path).resolve())) == digest, "runner tests binding")
    bind(folder / "summary.test-stdout.txt", receipt["stdout"]["sha256"])
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
    inherited, reference, vocabulary, partitions, parent_path, parent_sidecar = upstream()
    recipe = provenance(summary, folder, inherited)
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
    parent, child = reports["parent.json"], reports["child.json"]
    parent_rows, _, _ = evaluation(
        parent, "parent", reference, partitions["validation"], vocabulary
    )
    child_rows, _, _ = evaluation(
        child, "child", reference, partitions["validation"], vocabulary, parent
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
    invariants = {
        "parent_replay_exact": True,
        "train_only": True,
        "complete_500_updates": True,
        "frozen_tensors_unchanged": True,
        "child_reload_exact": True,
    }
    fixed = screen_gate(child, invariants)
    exact(summary["gate"], fixed, "independent declared gate")
    comparisons = {}
    for name, oldsets in [
        ("parent_dense", [r["selected_set_ids"] for r in parent_rows]),
        ("width8", [r["composition"]["selected_set_ids"] for r in reference["responses"]]),
    ]:
        comparisons[name] = {
            "paired": paired(child_rows, oldsets),
            "changed_exact": [
                {
                    "index": r["index"],
                    "subject": r["subject"],
                    "dimension": r["dimension"],
                    "group": r["group"],
                    "gained_exact": r["metrics"]["exact"],
                    "lost_exact": not r["metrics"]["exact"],
                }
                for r, ids in zip(child_rows, oldsets, strict=True)
                if r["metrics"]["exact"] != (set(ids) == set(r["expected_set_ids"]))
            ],
        }
    test_dir = ROOT / "runs/learning/projection-only-refit-auditor-tests-v1"
    receipt = read(test_dir / "test-receipt.json", sha(test_dir / "test-receipt.json"))
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["model_execution"] is False
        and receipt["new_diagnostic_executed"] is False
        and receipt["tested_auditor_sha256"] == sha(__file__)
        and receipt["passed_count"] >= 70
        and receipt["tested_recipe_sha256"] == RECIPE_SHA,
        "auditor tested source",
    )
    bind(test_dir / "test_auditor.py", receipt["tests_sha256"])
    bind(test_dir / "stdout.txt", receipt["stdout_sha256"])
    manifest(dict(INPUTS))
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(summary_path),
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN_SHA,
        "test_receipt_sha256": sha(test_dir / "test-receipt.json"),
        "parent": {k: parent[k] for k in ("aggregate", "groups", "paired_vs_width8")},
        "child": {
            k: child[k]
            for k in ("aggregate", "groups", "paired_vs_width8", "paired_vs_parent_dense")
        },
        "checkpoint": checkpoint,
        "comparisons": comparisons,
        "gate": fixed,
        "input_sha256": dict(INPUTS),
        "limitations": [
            "Independent saved logits, IDs, label/split, metrics and "
            "checkpoint-byte audit; no neural forwards or repeated training.",
            "The 500 finite ordered loss/gradient receipts are "
            "attestations, not an independently reconstructed optimization trajectory.",
            "Saved optimizer state identifies the final step and moments; "
            "fresh initialization and update chronology remain frozen source/receipt evidence.",
            "All non-W state bytes match the authenticated parent; saved "
            "reload manifests match the child payload, without another model construction.",
            "Saved head logits are authenticated but are not independently "
            "regenerated from child weights.",
            "Workspace git receipts identify the recorded dirty state; "
            "executable modules are separately authenticated archive members.",
            "Dense ID sets contain neither generated EOS nor an "
            "autoregressive sequence; no serving or durable weight-archive claim.",
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


if __name__ == "__main__":
    main()
