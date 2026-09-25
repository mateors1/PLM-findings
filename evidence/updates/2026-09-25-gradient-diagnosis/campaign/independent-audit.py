"""Independent NumPy audit of saved margin-gradient diagnostic evidence.

No Torch/model imports, neural replay, optimizer work or alternative policies.
Saved-vector consistency does not independently prove autograd correctness.
"""

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN_SHA = "a0b5af7a3d31ad9e1ac5a5616b13152dd946cec771e5164dbc98dff703f5bbd6"
PRIMARY_SHA = "39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701"
PRIMARY_AUDIT_SHA = "17afc9ca0fedf184ac9e9f2794947306883d4288860b22124cd9eb0c7751bcd9"
SOURCE_SHA = "e806549e1c778b5fc1529423e346ccf915b1484e3a5b322072fff5316c35ab6b"
RUNNER_SHA = "ed889b7785e346065ceb732a89ae78fdaec7a52df3254e2f1502628a73ec400c"
OBJECTIVE = "hard-boundary-hinge-fp32-amin-amax-query-mean-v1"
SLICES = ((0, 32), (800, 832), (1600, 1632))
PARAMETERS = {
    "token_embedding.weight": (2049, 256),
    "symmetric_relation_projection.weight": (256, 256),
}
CHECKPOINTS = {
    ("control", 500): (
        "ba31f3d3e7e3aa56cd5ee7020a7458c93f16767c018b72ff8b4fdc4ca7337fac",
        "503423dddc0ec5560e159e34c8ed296a199cdfdede73827eacf04a6d1e1065e9",
    ),
    ("treatment", 500): (
        "f74cf4774905d70bc8d8fc9643373c1ae1c53778d9bca2ade5c7b1fb6b45efd8",
        "da4b2d7a63c1b3af5fab22758f30ca55ab9c7aeec397d391a82005c67ea248d7",
    ),
    ("control", 2000): (
        "8771fce8a76cf7fab9a67913a30d85490f8edce979ec353928cf4c500479c487",
        "cff070bab97f2ed9a741d01f32d3f457ad976d74de0cecda26e95ea4f3c714ff",
    ),
    ("treatment", 2000): (
        "0bb697a4a0c8a05441fb2c825ffcfc47023f91596be8f48cda639d4e57c1e8ef",
        "339ccb185d1bc7cc05349f1f1c7defbf973d21f2f5813cd924d4f8fe91b0ff7c",
    ),
}
INPUTS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path, expected):
    path = Path(path).resolve()
    actual = sha(path)
    require(actual == expected, f"hash mismatch: {path}")
    require(str(path) not in INPUTS or INPUTS[str(path)] == actual, f"input drift: {path}")
    INPUTS[str(path)] = actual
    return path


def read(path, expected):
    return json.loads(bind(path, expected).read_text(encoding="utf-8"))


def exact(actual, expected, label):
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        label,
    )


def close(actual, expected, *, rtol, atol, label):
    require(
        type(actual) in (float, int)
        and math.isfinite(actual)
        and math.isfinite(expected)
        and abs(actual - expected) <= atol + rtol * abs(expected),
        label,
    )


def checked_array(value, shape, dtype, label):
    require(isinstance(value, np.ndarray), label + " not an array")
    require(value.shape == shape and value.dtype == np.dtype(dtype), label + " dtype/shape")
    require(np.isfinite(value).all(), label + " nonfinite")
    return value


def magnitude(vector):
    flat = np.asarray(vector, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.sum(flat * flat, dtype=np.float64)))


def dot(left, right):
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    require(a.shape == b.shape, "dot shape mismatch")
    return float(np.sum(a * b, dtype=np.float64))


def directions(left, right):
    a, b = magnitude(left), magnitude(right)
    if a == 0.0 or b == 0.0:
        return {
            "cosine": None,
            "norm_ratio": None,
            "reason": "both_zero" if a == b else ("left_zero" if a == 0.0 else "right_zero"),
        }
    return {"cosine": dot(left, right) / (a * b), "norm_ratio": a / b, "reason": None}


def margin_masks(logits, labels, inputs, attention):
    """Derive membership from shifted masked labels; exclude each prompt subject."""
    require(
        inputs.ndim == 2 and inputs.shape == labels.shape == attention.shape, "label dimensions"
    )
    require(inputs.dtype == labels.dtype == np.dtype("int64"), "input/label dtype")
    require(attention.dtype == np.dtype("bool"), "attention dtype")
    require(logits.shape == (inputs.shape[0], 1025), "logit dimensions")
    require(logits.dtype == np.dtype("float32") and np.isfinite(logits).all(), "logit dtype/finite")
    labels = np.where(attention, labels, -100)[:, 1:]
    positive = np.zeros(logits.shape, dtype=bool)
    negative = np.ones(logits.shape, dtype=bool)
    for row in range(len(inputs)):
        wanted = {int(v) - 1024 for v in labels[row] if 1024 <= v <= 2048}
        subject = int(inputs[row, 1]) - 1024
        require(0 <= subject <= 1024 and subject not in wanted, "subject mask/teacher contract")
        for column in wanted:
            positive[row, column] = True
        negative[row, list(wanted | {subject})] = False
        require(positive[row].any() and negative[row].any(), "empty margin class")
    weakest = np.min(np.where(positive, logits, np.inf), axis=1)
    strongest = np.max(np.where(negative, logits, -np.inf), axis=1)
    hinge = np.maximum(np.float32(0), np.float32(1) + strongest - weakest)
    active = hinge > 0
    radial = float(
        np.mean(np.where(active, strongest.astype(np.float64) - weakest.astype(np.float64), 0.0))
    )
    return {
        "positive": positive,
        "negative": negative,
        "min_true": weakest,
        "max_negative": strongest,
        "hinge": hinge,
        "active": active,
        "margin": float(np.mean(hinge, dtype=np.float64)),
        "radial_rhs": radial,
    }


def balanced_symmetric_bce(logits, masks):
    scores = logits.astype(np.float64)
    positives = masks["positive"]
    negatives = masks["negative"]
    positive_loss = np.sum(np.logaddexp(0.0, -scores) * positives, axis=1, dtype=np.float64)
    positive_loss /= np.sum(positives, axis=1)
    negative_loss = np.sum(np.logaddexp(0.0, scores) * negatives, axis=1, dtype=np.float64)
    negative_loss /= np.sum(negatives, axis=1)
    return float(np.mean((positive_loss + negative_loss) * 0.5, dtype=np.float64))


def gradient_vectors(raw, unused, parameter):
    require(
        set(raw) == set(unused) == {"ce", "prompt_bce", "symmetric_bce", "margin"},
        "raw objective inventory",
    )
    for name, value in raw.items():
        require(type(unused[name]) is bool, "unused flag must be a boolean")
        checked_array(value, parameter.shape, np.float32, "gradient " + name)
        if unused[name]:
            require(np.count_nonzero(value) == 0, "unused parameter has nonzero filled gradient")
    vectors = {key: value.astype(np.float64) for key, value in raw.items()}
    vectors["weighted_margin"] = 0.1 * vectors["margin"]
    vectors["control_sum"] = vectors["ce"] + vectors["prompt_bce"] + vectors["symmetric_bce"]
    vectors["hypothetical_sum"] = vectors["control_sum"] + vectors["weighted_margin"]
    return vectors


def checkpoint_evidence(primary, primary_audit):
    """Bind authoritative sidecars and bytes; payload proof remains runner-attested."""
    require(
        primary["complete"] is True
        and primary_audit["audit_passed"] is True
        and primary_audit["summary_sha256"] == PRIMARY_SHA,
        "accepted primary identity",
    )
    results = {}
    for arm, step in CHECKPOINTS:
        index = 0 if arm == "control" else 1
        training = primary["training"][index]
        directory = Path(training["checkpoint"]).parent
        suffix = "final" if step == 2000 else "step-500"
        checkpoint = directory / f"checkpoint-{suffix}.pt"
        expected_checkpoint, expected_sidecar = CHECKPOINTS[arm, step]
        bind(checkpoint, expected_checkpoint)
        sidecar = read(checkpoint.with_suffix(".pt.json"), expected_sidecar)
        require(
            sidecar["checkpoint_hash"] == expected_checkpoint and sidecar["global_step"] == step,
            "checkpoint identity/step",
        )
        exact(sidecar["experiment_identity"], training["identity"], "checkpoint full run identity")
        exact(sidecar["config"], training["config"]["train"], "checkpoint train configuration")
        exact(
            sidecar["training_metadata"]["model_config"],
            training["config"]["model"],
            "checkpoint model config",
        )
        require(
            sidecar["training_metadata"]["objective"] == "causal-next-token-v1",
            "checkpoint objective",
        )
        enabled = arm == "treatment"
        require(
            ("symmetric_margin_objective" in sidecar["training_metadata"]) == enabled,
            "margin descriptor presence",
        )
        if enabled:
            require(
                sidecar["training_metadata"]["symmetric_margin_objective"] == OBJECTIVE,
                "margin objective version",
            )
        exact(
            sidecar["corpus_identity"],
            {k: training["identity"][k] for k in ("graph_hash", "records_hash", "vocabulary_hash")},
            "checkpoint corpus",
        )
        require(sidecar["split_hash"] == training["identity"]["split_hash"], "checkpoint split")
        results[arm, step] = sidecar
    return results


def archive_inventory(path, expected):
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "duplicate archive member")
        require(set(archive.namelist()) == set(expected), "archive inventory")
        for name, digest in expected.items():
            require(
                hashlib.sha256(archive.read(name)).hexdigest() == digest, "archive byte identity"
            )


def training_slices(primary, accepted_audit):
    config = primary["training"][0]["config"]
    identity = primary["training"][0]["identity"]
    corpus = Path(config["data"]["corpus_manifest"]).parent
    available = {
        str(Path(name).resolve()): digest for name, digest in accepted_audit["input_sha256"].items()
    }
    paths = {
        name: (ROOT / corpus / name).resolve()
        for name in ("manifest.json", "vocabulary.json", "records.jsonl")
    }
    for path in paths.values():
        require(str(path) in available, "corpus artifact absent from accepted audit")
        bind(path, available[str(path)])
    manifest = json.loads(paths["manifest.json"].read_text())
    vocabulary = json.loads(paths["vocabulary.json"].read_text())
    require(
        sha(paths["records.jsonl"]) == manifest["records_hash"] == identity["records_hash"],
        "corpus record identity",
    )
    require(
        hashlib.sha256(
            json.dumps(vocabulary, ensure_ascii=True, sort_keys=True).encode()
        ).hexdigest()
        == identity["vocabulary_hash"]
        == manifest["tokenizer_hash"],
        "vocabulary content identity",
    )
    tokens = vocabulary["tokens"]
    require(len(tokens) == len(set(tokens)) == 2049, "vocabulary cardinality")
    assignments = {"train": [], "validation": [], "test": []}
    selected = {}
    wanted = {i for start, end in SLICES for i in range(start, end)}
    data = config["data"]
    for line in paths["records.jsonl"].read_text().splitlines():
        record = json.loads(line)
        query = [record["subject"], record["dimension"]]
        digest = hashlib.sha256(f"{data['split_seed']}:{query[0]}:{query[1]}".encode()).digest()
        fraction = int.from_bytes(digest, "big") / float(1 << 256)
        if fraction < data["validation_query_fraction"]:
            split = "validation"
        elif fraction < data["validation_query_fraction"] + data["test_query_fraction"]:
            split = "test"
        else:
            split = "train"
        index = len(assignments[split])
        assignments[split].append(query)
        # Target fields are accessed only for the 96 fixed training indices.
        if split == "train" and index in wanted:
            selected[index] = record
    split_object = {
        "algorithm": "sha256-threshold-v2",
        "seed": data["split_seed"],
        "validation_fraction": data["validation_query_fraction"],
        "test_fraction": data["test_query_fraction"],
        **assignments,
    }
    require(
        hashlib.sha256(
            json.dumps(
                split_object, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        == identity["split_hash"],
        "split identity",
    )
    require(set(selected) == wanted and len(selected) == 96, "fixed training query coverage")
    batches = {}
    for start, end in SLICES:
        records = [selected[i] for i in range(start, end)]
        width = max(len(r["input_ids"]) for r in records)
        inputs = np.full((32, width), tokens.index("PAD"), dtype=np.int64)
        labels = np.full((32, width), -100, dtype=np.int64)
        attention = np.zeros((32, width), dtype=bool)
        for row, record in enumerate(records):
            size = len(record["input_ids"])
            require(size == len(record["labels"]), "teacher sequence shape")
            inputs[row, :size] = record["input_ids"]
            labels[row, :size] = record["labels"]
            attention[row, :size] = True
            prompt = [
                tokens.index(t)
                for t in ("BOS", record["subject"], record["dimension"], "SAME", "ANSWER")
            ]
            require(
                record["input_ids"][:5] == prompt and record["labels"][:5] == [-100] * 5,
                "teacher prompt mask",
            )
            require(
                record["input_ids"][5:-1] == [tokens.index(t) for t in record["targets"]],
                "teacher target membership",
            )
        batches[start] = {
            "records": records,
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention,
        }
    return batches, tokens


def comparison_record(left, right):
    result = directions(left, right)
    reason = result["reason"] + "_norm" if result["reason"] else None
    return {key: {"value": result[key], "reason": reason} for key in ("cosine", "norm_ratio")}


def group_statistics(arrays, unused, dimension_ids):
    output = {}
    objectives = ("ce", "prompt_bce", "symmetric_bce", "margin")
    require(set(unused) == set(objectives), "unused objective inventory")
    for objective in objectives:
        require(set(unused[objective]) == {"embedding", "projection"}, "unused block inventory")
        require(all(type(v) is bool for v in unused[objective].values()), "unused nonboolean")
        require(unused[objective]["embedding"] is False, "embedding unexpectedly unused")
        require(
            unused[objective]["projection"] is (objective in ("ce", "prompt_bce")),
            "projection ownership",
        )
    for group, block, rows in (
        ("all_embeddings", "embedding", slice(None)),
        ("products", "embedding", slice(1024, None)),
        ("steering", "embedding", dimension_ids),
        ("projection", "projection", slice(None)),
    ):
        parameter = arrays[f"parameter__{block}"][rows]
        raw = {name: arrays[f"gradient__{name}__{block}"][rows] for name in objectives}
        ownership = {name: unused[name][block] for name in objectives}
        vectors = gradient_vectors(raw, ownership, parameter)
        ownership["weighted_margin"] = ownership["margin"]
        ownership["control_sum"] = all(ownership[name] for name in objectives[:3])
        ownership["hypothetical_sum"] = all(ownership[name] for name in objectives)
        pairs = [("weighted_margin", "symmetric_bce"), ("weighted_margin", "control_sum")]
        if block == "embedding":
            pairs.extend((("symmetric_bce", "ce"), ("symmetric_bce", "prompt_bce")))
        output[group] = {
            "parameter_shape": list(parameter.shape),
            "vectors": {
                name: {
                    "l2_norm": magnitude(vector),
                    "parameter_dot": dot(vector, parameter),
                    "unused": ownership[name],
                }
                for name, vector in vectors.items()
            },
            "comparisons": {
                f"{left}_vs_{right}": comparison_record(vectors[left], vectors[right])
                for left, right in pairs
            },
        }
        if group == "projection":
            require(
                np.array_equal(vectors["control_sum"], vectors["symmetric_bce"]),
                "projection control sum differs from symmetric BCE",
            )
    return output


def aggregate_statistics(groups):
    # Select numeric leaves by walking the persisted schema, never average vectors.
    values = {}
    for group in groups:
        pending = [("", group)]
        while pending:
            prefix, item = pending.pop()
            for name, value in item.items():
                key = prefix + "/" + name
                if isinstance(value, dict):
                    pending.append((key, value))
                elif name != "reason" and (type(value) in (float, int) or value is None):
                    values.setdefault(key, []).append(value)
    result = {}
    for name, entries in values.items():
        require(len(entries) == len(groups), "aggregate leaf missing in batch")
        defined = [value for value in entries if value is not None]
        result[name] = {
            "mean_of_defined_batch_values": math.fsum(defined) / len(defined) if defined else None,
            "defined_count": len(defined),
            "batch_count": len(groups),
        }
    return result


def expected_queries(batch, offset):
    return [
        {
            "train_index": offset + i,
            "subject": record["subject"],
            "dimension": record["dimension"],
            "targets": record["targets"],
            "input_ids": record["input_ids"],
            "labels": record["labels"],
            "prompt_ids": record["input_ids"][:5],
        }
        for i, record in enumerate(batch["records"])
    ]


def observation_evidence(report, arrays, state, offset, batch):
    require(report["complete"] is True and not report.get("error"), "incomplete observation")
    require(
        report["state"] == state and report["train_slice"] == [offset, offset + 32],
        "observation state/slice",
    )
    exact(report["queries"], expected_queries(batch, offset), "ordered training queries")
    require(
        report["state_unchanged"] is True
        and report["all_parameter_grad_fields_none"] is True
        and report["state_before_sha256"] == report["state_after_sha256"],
        "state mutated or accumulated gradients",
    )
    dimensions = sorted({int(value) for value in batch["input_ids"][:, 2]})
    exact(report["dimension_ids"], dimensions, "steering dimension IDs")
    exact(
        report["dimension_counts"],
        {d: sum(r["dimension"] == d for r in batch["records"]) for d in ("TYPE", "COLOR")},
        "dimension counts",
    )
    shapes = {}
    for block, shape in (("embedding", (2049, 256)), ("projection", (256, 256))):
        shapes[f"parameter__{block}"] = (shape, "float32")
        for objective in ("ce", "prompt_bce", "symmetric_bce", "margin"):
            shapes[f"gradient__{objective}__{block}"] = (shape, "float32")
    sequence_shape = batch["input_ids"].shape
    shifted_shape = (32, sequence_shape[1] - 1)
    for key in ("input_ids", "labels", "attention_mask"):
        shapes[key] = (sequence_shape, "bool" if key == "attention_mask" else "int64")
    shapes.update(
        shifted_labels=(shifted_shape, "int64"),
        positive_mask=((32, 1025), "bool"),
        negative_mask=((32, 1025), "bool"),
        symmetric_logits=((32, 1025), "float32"),
        prompt_logits=((32, 1025), "float32"),
        margin_hinges=((32,), "float32"),
        token_ce_unreduced=(shifted_shape, "float32"),
    )
    for loss in ("ce", "prompt_bce", "symmetric_bce", "margin", "total"):
        shapes[f"losses__{loss}"] = ((), "float32")
    require(set(arrays) == set(shapes), "unexpected/missing NPZ arrays")
    for name, (shape, dtype) in shapes.items():
        checked_array(arrays[name], shape, dtype, name)
    exact(
        report["array_manifest"],
        {name: {"shape": list(shape), "dtype": dtype} for name, (shape, dtype) in shapes.items()},
        "array manifest",
    )
    for key in ("input_ids", "labels", "attention_mask"):
        require(np.array_equal(arrays[key], batch[key]), "teacher collation differs: " + key)
    shifted = np.where(batch["attention_mask"], batch["labels"], -100)[:, 1:]
    require(np.array_equal(arrays["shifted_labels"], shifted), "shifted loss mask")
    masks = margin_masks(
        arrays["symmetric_logits"], arrays["labels"], arrays["input_ids"], arrays["attention_mask"]
    )
    for name, key in (
        ("positive_mask", "positive"),
        ("negative_mask", "negative"),
        ("margin_hinges", "hinge"),
    ):
        require(
            np.array_equal(arrays[name], masks[key]), "reconstructed mask/hinge differs: " + name
        )
    losses = {
        name: float(arrays[f"losses__{name}"])
        for name in ("ce", "prompt_bce", "symmetric_bce", "margin", "total")
    }
    exact(report["losses"], losses, "loss scalar/array binding")
    close(
        losses["symmetric_bce"],
        balanced_symmetric_bce(arrays["symmetric_logits"], masks),
        rtol=1e-5,
        atol=1e-6,
        label="recomputed symmetric BCE",
    )
    close(losses["margin"], masks["margin"], rtol=1e-5, atol=1e-6, label="recomputed raw margin")
    require(np.all(arrays["token_ce_unreduced"][shifted == -100] == 0), "ignored token CE not zero")
    require(np.all(arrays["token_ce_unreduced"] >= 0), "negative token CE")
    close(
        losses["ce"],
        float(np.mean(arrays["token_ce_unreduced"][shifted != -100], dtype=np.float64)),
        rtol=1e-5,
        atol=1e-6,
        label="saved token CE reduction",
    )
    summed = np.float32(
        np.float32(arrays["losses__ce"] + arrays["losses__prompt_bce"])
        + arrays["losses__symmetric_bce"]
    )
    if state.startswith("treatment_"):
        summed = np.float32(summed + np.float32(0.1) * arrays["losses__margin"])
    close(losses["total"], float(summed), rtol=1e-5, atol=1e-6, label="loss component sum")
    exact(
        report["total_loss_residual"],
        float(np.float32(arrays["losses__total"] - summed)),
        "loss residual",
    )
    require(
        report["margin_query_count"] == 32
        and report["margin_active_count"] == int(masks["active"].sum()),
        "margin count",
    )
    groups = group_statistics(arrays, report["unused"], dimensions)
    exact(report["groups"], groups, "independent gradient statistics")
    lhs = dot(arrays["gradient__margin__projection"], arrays["parameter__projection"])
    rhs = masks["radial_rhs"]
    radial = {
        "gradient_parameter_dot": lhs,
        "active_extrema_mean": rhs,
        "residual": lhs - rhs,
        "relative_tolerance": 1e-4,
        "absolute_tolerance": 1e-5,
        "passed": math.isclose(lhs, rhs, rel_tol=1e-4, abs_tol=1e-5),
        "active_count": int(masks["active"].sum()),
        "query_count": 32,
    }
    exact(report["radial"], radial, "independent radial identity")
    require(radial["passed"], "radial identity outside fixed tolerance")
    return {"groups": groups, "radial": radial, "losses": losses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    require("torch" not in sys.modules, "NumPy-only auditor")
    path = args.summary.resolve()
    output = path.parent / "independent-audit.json"
    require(not output.exists(), "refusing existing independent audit")
    require(path.is_file(), "completed primary gradient summary missing")
    own_sha = sha(__file__)
    summary = read(path, sha(path))
    require(summary["complete"] is True and not summary.get("error"), "incomplete diagnostic")
    require(
        summary["independent_audit_status"] == "required"
        and summary["diagnostic_accepted"] is False,
        "diagnostic acceptance boundary",
    )
    require(
        summary["parameter_updates"] == 0 and summary["training_executed"] is False,
        "no-update boundary",
    )
    require(
        summary["plan_sha256"] == PLAN_SHA and summary["source_archive_sha256"] == SOURCE_SHA,
        "declared plan/source",
    )
    bind(ROOT / "docs/experiments/2026-09-25-margin-gradient-diagnosis-plan.md", PLAN_SHA)
    for name, digest in summary["input_sha256"].items():
        bind(name, digest)
    require(
        set(summary["snapshot_sha256"])
        == {"script.py", "plan.md", "source.zip", "configs.zip", "helper.py"},
        "snapshot coverage",
    )
    for suffix, digest in summary["snapshot_sha256"].items():
        require(Path(suffix).name == suffix, "unsafe snapshot name")
        bind(path.with_suffix("." + suffix), digest)
    require(
        summary["snapshot_sha256"]["script.py"] == summary["script_sha256"] == RUNNER_SHA,
        "executed script snapshot",
    )
    bind(ROOT / "scripts/diagnose_margin_gradients.py", summary["script_sha256"])
    require(
        summary["snapshot_sha256"]["plan.md"] == PLAN_SHA
        and summary["snapshot_sha256"]["source.zip"] == SOURCE_SHA,
        "pinned snapshots",
    )
    primary_directory = ROOT / "runs/learning/symmetric-margin-screen-v1"
    primary = read(primary_directory / "summary.json", PRIMARY_SHA)
    accepted_audit = read(primary_directory / "independent-audit.json", PRIMARY_AUDIT_SHA)
    bind(primary_directory / "independent-audit.py", accepted_audit["script_sha256"])
    bind(primary_directory / "summary.script.py", primary["snapshot_sha256"]["script.py"])
    require(
        summary["snapshot_sha256"]["helper.py"] == primary["snapshot_sha256"]["script.py"],
        "authenticated helper snapshot",
    )
    checkpoint_evidence(primary, accepted_audit)
    for index, arm in enumerate(("control", "treatment")):
        exact(
            summary["effective_configs"][arm],
            primary["training"][index]["config"],
            "effective training recipe",
        )
        model = summary["effective_configs"][arm]["model"]
        require(
            model["first_target_loss_weight"]
            == model["prompt_set_loss_weight"]
            == model["symmetric_relation_loss_weight"]
            == 1.0,
            "component objective weights",
        )
        require(
            model["symmetric_margin"] == 1.0
            and model["symmetric_margin_loss_weight"] == (0.1 if arm == "treatment" else 0.0),
            "margin settings",
        )
        require(
            model["continuation_set_loss_weight"] == model["z_loss_weight"] == 0
            and model["moe"]["enabled"] is False,
            "extra objectives enabled",
        )
        require(
            model["embed_dropout"] == model["resid_dropout"] == model["attention"]["dropout"] == 0,
            "nonzero dropout",
        )
    exact(summary["environment"], primary["environment"], "same source/runtime")
    require(summary["source_commit"] == primary["source_commit"], "source revision")
    exact(
        summary["live_source_config_sha256"],
        primary["live_source_config_sha256"],
        "frozen source/config identity",
    )
    live = summary["live_source_config_sha256"]
    source_names = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")} | {
        "pyproject.toml",
        "uv.lock",
    }
    config_names = {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").rglob("*.yaml")}
    require(set(live) == source_names | config_names, "live source/config inventory")
    for name, digest in live.items():
        bind(ROOT / name, digest)
    archive_inventory(path.with_suffix(".source.zip"), {name: live[name] for name in source_names})
    archive_inventory(path.with_suffix(".configs.zip"), {name: live[name] for name in config_names})
    settings = summary["numerical_settings"]
    for key, value in {
        "parameter_dtype": "float32",
        "device": "cuda",
        "model_training": False,
        "autocast_enabled": False,
        "matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }.items():
        exact(settings[key], value, "numerical setting " + key)
    for key in ("deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark"):
        require(type(settings[key]) is bool, "deterministic setting record")
    initialization = summary["initialization"]
    require(
        initialization["regenerated"] is True
        and initialization["historically_saved"] is False
        and initialization["seed"] == 1729
        and initialization["every_state_tensor_exactly_equal"] is True,
        "initial state provenance",
    )
    require(
        initialization["control_state_sha256"] == initialization["treatment_state_sha256"],
        "initial state equality receipt",
    )
    batches, _ = training_slices(primary, accepted_audit)
    queries = [q for start, _ in SLICES for q in expected_queries(batches[start], start)]
    require(len({(q["subject"], q["dimension"]) for q in queries}) == 96, "distinct queries")
    exact(summary["queries"], queries, "fixed complete training slices")
    identity = primary["training"][0]["identity"]
    exact(
        summary["corpus_identity"],
        {k: identity[k] for k in ("graph_hash", "records_hash", "vocabulary_hash")},
        "corpus identity",
    )
    require(summary["split_hash"] == identity["split_hash"], "split identity")
    states = ("initial", "control_500", "treatment_500", "control_2000", "treatment_2000")
    expected_order = [(state, start) for state in states for start, _ in SLICES]
    exact(
        [(r["state"], r["offset"]) for r in summary["observations"]],
        expected_order,
        "all fifteen observations/order",
    )
    expected_artifacts = {
        f"{state}-batch-{start}.{suffix}"
        for state, start in expected_order
        for suffix in ("npz", "json")
    }
    require(set(summary["artifact_sha256"]) == expected_artifacts, "complete raw/report inventory")
    for filename, digest in summary["artifact_sha256"].items():
        bind(path.parent / filename, digest)
    results = {state: [] for state in states}
    parameters, state_hashes = {}, {}
    for descriptor in summary["observations"]:
        state, offset = descriptor["state"], descriptor["offset"]
        stem = f"{state}-batch-{offset}"
        require(
            descriptor["raw"] == stem + ".npz" and descriptor["report"] == stem + ".json",
            "observation artifact mapping",
        )
        report = read(
            path.parent / descriptor["report"], summary["artifact_sha256"][descriptor["report"]]
        )
        require(
            report["raw_sha256"] == summary["artifact_sha256"][descriptor["raw"]],
            "raw/report hash binding",
        )
        with np.load(path.parent / descriptor["raw"], allow_pickle=False) as raw:
            require(len(raw.files) == len(set(raw.files)), "duplicate NPZ array")
            arrays = {key: raw[key] for key in raw.files}
        result = observation_evidence(report, arrays, state, offset, batches[offset])
        require(
            descriptor["state_sha256"]
            == report["state_before_sha256"]
            == report["state_after_sha256"],
            "observation state identity",
        )
        if state not in state_hashes:
            state_hashes[state] = descriptor["state_sha256"]
        require(descriptor["state_sha256"] == state_hashes[state], "state changed across batches")
        if state == "initial":
            require(
                state_hashes[state] == initialization["control_state_sha256"],
                "initial regenerated identity",
            )
        for block in ("embedding", "projection"):
            array = arrays[f"parameter__{block}"]
            key = (state, block)
            if key not in parameters:
                parameters[key] = array.copy()
            require(
                np.array_equal(array, parameters[key]), "saved parameter changed across batches"
            )
        results[state].append(result)
        print(f"Audited saved gradients: {state} batch {offset}", flush=True)
    rebuilt = {
        state: aggregate_statistics([r["groups"] for r in results[state]]) for state in states
    }
    exact(summary["state_aggregates"], rebuilt, "defined-only per-batch aggregates")
    require(
        all(sha(name) == digest for name, digest in INPUTS.items()),
        "input/source/checkpoint changed during audit",
    )
    require(
        sha(__file__) == own_sha and "torch" not in sys.modules, "auditor drift or Torch import"
    )
    result = {
        "complete": True,
        "audit_passed": True,
        "all_saved_vector_reductions_equal": True,
        "diagnostic_accepted": False,
        "autograd_independently_proved": False,
        "causal_optimizer_claim_established": False,
        "neural_replay_executed": False,
        "summary_sha256": sha(path),
        "script_sha256": own_sha,
        "plan_sha256": PLAN_SHA,
        "distinct_training_queries": 96,
        "parameter_states": 5,
        "observations": 15,
        "query_observations": 480,
        "numpy_version": np.__version__,
        "state_aggregates": rebuilt,
        "radial_checks": {state: [r["radial"] for r in results[state]] for state in states},
        "input_sha256": dict(sorted(INPUTS.items())),
        "limitations": [
            "Saved arrays authenticate numerical reductions, "
            "not autograd correctness or neural outputs.",
            "Checkpoint payload loading and all-model state/.grad immutability are "
            "runner-attested; the auditor binds bytes/sidecars and checks saved parameter "
            "blocks across batches.",
            "Initialization is a seeded regenerated state, not a saved historical checkpoint.",
            "Symmetric BCE and margin are recomputed from saved logits; token CE is "
            "mask-aggregated from saved unreduced losses; prompt BCE is authenticated only.",
            "Loss checks use fixed relative 1e-5 and absolute 1e-6; radial identity uses "
            "fixed math.isclose relative 1e-4 and absolute 1e-5. "
            "Saved reductions match exactly.",
            "No optimizer trajectory, validation answer quality or unmeasured "
            "transformer-gradient claim follows from this diagnostic.",
        ],
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"audit_passed": True, "audit_sha256": sha(output), "diagnostic_accepted": False}
        )
    )


if __name__ == "__main__":
    main()
