"""Fixed, update-free gradient diagnosis of the rejected symmetric-margin screen.

Five states, three predeclared training batches, four objectives and two parameter
blocks. This diagnostic neither selects a coefficient nor replays an optimizer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

_PLAN = "a0b5af7a3d31ad9e1ac5a5616b13152dd946cec771e5164dbc98dff703f5bbd6"
_SUMMARY = "39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701"
_AUDIT = "17afc9ca0fedf184ac9e9f2794947306883d4288860b22124cd9eb0c7751bcd9"
_SOURCE = "e806549e1c778b5fc1529423e346ccf915b1484e3a5b322072fff5316c35ab6b"
_STATES = (
    (
        "control_500",
        "control",
        "checkpoint-step-500.pt",
        500,
        "ba31f3d3e7e3aa56cd5ee7020a7458c93f16767c018b72ff8b4fdc4ca7337fac",
        "503423dddc0ec5560e159e34c8ed296a199cdfdede73827eacf04a6d1e1065e9",
    ),
    (
        "treatment_500",
        "treatment",
        "checkpoint-step-500.pt",
        500,
        "f74cf4774905d70bc8d8fc9643373c1ae1c53778d9bca2ade5c7b1fb6b45efd8",
        "da4b2d7a63c1b3af5fab22758f30ca55ab9c7aeec397d391a82005c67ea248d7",
    ),
    (
        "control_2000",
        "control",
        "checkpoint-final.pt",
        2000,
        "8771fce8a76cf7fab9a67913a30d85490f8edce979ec353928cf4c500479c487",
        "cff070bab97f2ed9a741d01f32d3f457ad976d74de0cecda26e95ea4f3c714ff",
    ),
    (
        "treatment_2000",
        "treatment",
        "checkpoint-final.pt",
        2000,
        "0bb697a4a0c8a05441fb2c825ffcfc47023f91596be8f48cda639d4e57c1e8ef",
        "339ccb185d1bc7cc05349f1f1c7defbf973d21f2f5813cd924d4f8fe91b0ff7c",
    ),
)
_OFFSETS = (0, 800, 1600)
_OBJECTIVES = ("ce", "prompt_bce", "symmetric_bce", "margin")
_BLOCKS = ("embedding", "projection")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bound(path, digest, inputs):
    path = Path(path).resolve()
    _require(path.is_file() and _sha(path) == digest, f"identity mismatch: {path}")
    inputs[str(path)] = digest
    return path


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _helper(directory, summary, inputs):
    path = _bound(directory / "summary.script.py", summary["snapshot_sha256"]["script.py"], inputs)
    spec = importlib.util.spec_from_file_location("authenticated_margin_screen", path)
    _require(spec is not None and spec.loader is not None, "archive helper import")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _refuse(out):
    _require(
        not out.exists()
        and not list(out.parent.glob(out.stem + ".*"))
        and not list(out.parent.glob("*-batch-*.npz"))
        and not list(out.parent.glob("*-batch-*.json")),
        "immutable diagnostic output exists",
    )


def _preflight(root, plan, out):
    inputs = {}
    _bound(plan, _PLAN, inputs)
    directory = root / "runs/learning/symmetric-margin-screen-v1"
    summary = _json(_bound(directory / "summary.json", _SUMMARY, inputs))
    audit = _json(_bound(directory / "independent-audit.json", _AUDIT, inputs))
    _require(
        summary["complete"] is True
        and audit["audit_passed"] is True
        and audit["all_recomputed_outputs_equal"] is True
        and audit["summary_sha256"] == _SUMMARY,
        "completed screen audit required",
    )
    _bound(directory / "independent-audit.py", audit["script_sha256"], inputs)
    helper = _helper(directory, summary, inputs)
    source = helper._source_bytes(root)
    _require(hashlib.sha256(source).hexdigest() == _SOURCE, "declared source archive mismatch")
    _bound(directory / "summary.source.zip", _SOURCE, inputs)
    live = helper._inventory(root)
    _require(live == summary["live_source_config_sha256"], "source/config changed since screen")
    from plm.configuration import validate_saved_config
    from plm.corpus import load_corpus_records, require_valid_artifacts
    from plm.evaluation.split import split_queries
    from plm.protocol.tokenizer import Vocabulary

    arms = {}
    for arm in ("control", "treatment"):
        name = f"training-{arm}.json"
        receipt = _json(_bound(directory / name, summary["report_sha256"][name], inputs))
        config = validate_saved_config(receipt["config"], receipt["config_hash"])
        m = config.model
        _require(
            config.seed == 1729
            and m.vocab_size == 2049
            and m.dim == 256
            and m.max_seq_len == config.train.seq_len == 512
            and m.prompt_set_loss_weight
            == m.symmetric_relation_loss_weight
            == m.first_target_loss_weight
            == 1.0
            and m.symmetric_margin == 1.0
            and m.symmetric_margin_loss_weight == (0.0 if arm == "control" else 0.1)
            and m.continuation_set_loss_weight == m.z_loss_weight == 0
            and not m.moe.enabled
            and m.embed_dropout == m.resid_dropout == m.attention.dropout == 0,
            "fixed loss/model recipe",
        )
        arms[arm] = {"config": config, "receipt": receipt}
    for name, digest in summary["input_sha256"].items():
        path = Path(name).resolve()
        if path.is_relative_to(root / "data"):
            _bound(path, digest, inputs)
    config = arms["control"]["config"]
    corpus = root / Path(config.data.corpus_manifest).parent
    for path in (
        corpus / "manifest.json",
        corpus / "records.jsonl",
        corpus / "vocabulary.json",
        root / config.data.graph_db,
    ):
        _require(str(path.resolve()) in inputs, "corpus input not authenticated")
    validated = require_valid_artifacts(corpus, graph_db=root / config.data.graph_db)
    _require(validated.manifest is not None, "corpus manifest missing")
    split = split_queries(
        load_corpus_records(corpus),
        config.data.validation_query_fraction,
        config.data.test_query_fraction,
        config.data.split_seed,
    )
    vocabulary = Vocabulary.load(corpus / "vocabulary.json")
    corpus_identity = {
        "graph_hash": validated.manifest.graph_hash,
        "records_hash": validated.manifest.records_hash,
        "vocabulary_hash": validated.manifest.tokenizer_hash,
    }
    _require(
        len(vocabulary) == 2049 and vocabulary.entity_ids() == list(range(1024, 2049)),
        "vocabulary columns",
    )
    for arm in arms.values():
        identity = arm["receipt"]["identity"]
        _require(
            identity["split_hash"] == split.split_hash
            and all(identity[k] == v for k, v in corpus_identity.items()),
            "training corpus/split mismatch",
        )
    states = []
    for name, arm, filename, step, cp_hash, sidecar_hash in _STATES:
        config = arms[arm]["config"]
        path = root / config.paths.run_root / config.run_name / filename
        _bound(path, cp_hash, inputs)
        sidecar = _json(_bound(path.with_suffix(".pt.json"), sidecar_hash, inputs))
        _check_metadata(sidecar, step, arms[arm], corpus_identity, split.split_hash)
        states.append({"name": name, "arm": arm, "path": path, "sidecar": sidecar})
    batches, identities = {}, []
    for offset in _OFFSETS:
        records = split.train[offset : offset + 32]
        _require(len(records) == 32, "fixed training slice unavailable")
        batches[offset] = records
        for index, record in enumerate(records, offset):
            _require(
                len(record.input_ids) <= 512
                and list(record.input_ids[:5])
                == vocabulary.encode(["BOS", record.subject, record.dimension, "SAME", "ANSWER"]),
                "training prompt mismatch",
            )
            identities.append(
                {
                    "train_index": index,
                    "subject": record.subject,
                    "dimension": record.dimension,
                    "targets": list(record.targets),
                    "input_ids": list(record.input_ids),
                    "labels": list(record.labels),
                    "prompt_ids": list(record.input_ids[:5]),
                }
            )
    _require(
        len({(q["subject"], q["dimension"]) for q in identities}) == 96, "distinct training queries"
    )
    _refuse(out)
    context = {
        "root": root,
        "inputs": inputs,
        "live_inventory": live,
        "helper": helper,
        "source": source,
        "arms": arms,
        "states": states,
        "batches": batches,
        "queries": identities,
        "vocabulary": vocabulary,
        "corpus_identity": corpus_identity,
        "split_hash": split.split_hash,
        "screen_environment": summary["environment"],
    }
    helper._unchanged(context)
    _require("torch" not in sys.modules, "preflight imported Torch")
    return context


def _check_metadata(metadata, step, arm, corpus, split_hash):
    from plm.configuration import validate_checkpoint_model_config
    from plm.model.interfaces import SYMMETRIC_MARGIN_OBJECTIVE

    config = arm["config"]
    training = metadata["training_metadata"]
    _require(
        metadata["global_step"] == step
        and metadata["experiment_identity"] == arm["receipt"]["identity"]
        and metadata["corpus_identity"] == corpus
        and metadata["split_hash"] == split_hash
        and metadata["config"] == config.train.model_dump(mode="json")
        and validate_checkpoint_model_config(training["model_config"]) == config.model
        and training["model_config"] == config.model.model_dump(mode="json")
        and training["objective"] == "causal-next-token-v1",
        "checkpoint objective/config/identity mismatch",
    )
    active = config.model.symmetric_margin_loss_weight > 0
    _require(
        ("symmetric_margin_objective" in training) == active
        and training.get("symmetric_margin_objective")
        == (SYMMETRIC_MARGIN_OBJECTIVE if active else None),
        "margin descriptor mismatch",
    )


def _norm(vector):
    import numpy as np

    values = np.asarray(vector, dtype=np.float64)
    return float(np.sqrt(np.sum(values * values, dtype=np.float64)))


def _dot(a, b):
    import numpy as np

    return float(
        np.sum(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64), dtype=np.float64)
    )


def _comparison(a, b):
    an, bn = _norm(a), _norm(b)
    reason = (
        "both_zero_norm"
        if an == bn == 0
        else "left_zero_norm"
        if an == 0
        else "right_zero_norm"
        if bn == 0
        else None
    )
    # Even a zero numerator is left undefined as requested by this diagnostic contract.
    return {
        "cosine": {"value": None if reason else _dot(a, b) / (an * bn), "reason": reason},
        "norm_ratio": {"value": None if reason else an / bn, "reason": reason},
    }


def _analysis(arrays, unused, dimensions):
    import numpy as np

    result = {}
    for group, block, indices in (
        ("all_embeddings", "embedding", slice(None)),
        ("products", "embedding", slice(1024, None)),
        ("steering", "embedding", dimensions),
        ("projection", "projection", slice(None)),
    ):
        parameter = arrays[f"parameter__{block}"][indices].astype(np.float64)
        vectors = {
            k: arrays[f"gradient__{k}__{block}"][indices].astype(np.float64) for k in _OBJECTIVES
        }
        vectors["weighted_margin"] = 0.1 * vectors["margin"]
        vectors["control_sum"] = vectors["ce"] + vectors["prompt_bce"] + vectors["symmetric_bce"]
        vectors["hypothetical_sum"] = vectors["control_sum"] + vectors["weighted_margin"]
        ownership = {k: unused[k][block] for k in _OBJECTIVES}
        ownership.update(
            weighted_margin=unused["margin"][block],
            control_sum=all(unused[k][block] for k in _OBJECTIVES[:3]),
            hypothetical_sum=all(unused[k][block] for k in _OBJECTIVES),
        )
        pairs = [("weighted_margin", "symmetric_bce"), ("weighted_margin", "control_sum")]
        if block == "embedding":
            pairs += [("symmetric_bce", "ce"), ("symmetric_bce", "prompt_bce")]
        result[group] = {
            "parameter_shape": list(parameter.shape),
            "vectors": {
                k: {
                    "l2_norm": _norm(v),
                    "parameter_dot": _dot(v, parameter),
                    "unused": ownership[k],
                }
                for k, v in vectors.items()
            },
            "comparisons": {f"{a}_vs_{b}": _comparison(vectors[a], vectors[b]) for a, b in pairs},
        }
    _require(
        unused["ce"]["projection"] is True and unused["prompt_bce"]["projection"] is True,
        "projection must be unused by CE and prompt BCE",
    )
    _require(
        np.array_equal(
            arrays["gradient__ce__projection"], np.zeros_like(arrays["parameter__projection"])
        )
        and np.array_equal(
            arrays["gradient__prompt_bce__projection"],
            np.zeros_like(arrays["parameter__projection"]),
        ),
        "unused parameter vector must be explicit zero fill",
    )
    return result


def _radial(arrays):
    import numpy as np

    logits = arrays["symmetric_logits"].astype(np.float64)
    a = np.min(np.where(arrays["positive_mask"], logits, np.inf), axis=1)
    b = np.max(np.where(arrays["negative_mask"], logits, -np.inf), axis=1)
    # Activation follows the measured FP32 helper; the radial reduction uses FP64.
    active = arrays["margin_hinges"] > 0
    rhs = float(np.sum(np.where(active, b - a, 0), dtype=np.float64) / len(a))
    lhs = _dot(arrays["gradient__margin__projection"], arrays["parameter__projection"])
    return {
        "gradient_parameter_dot": lhs,
        "active_extrema_mean": rhs,
        "residual": lhs - rhs,
        "relative_tolerance": 1e-4,
        "absolute_tolerance": 1e-5,
        "passed": math.isclose(lhs, rhs, rel_tol=1e-4, abs_tol=1e-5),
        "active_count": int(np.sum(active)),
        "query_count": len(a),
    }


def _state_identity(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        descriptor = json.dumps(
            [name, str(array.dtype), list(array.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "big") + descriptor)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _measure(model, records, pad_id, state, offset, directory, queries, *, device="cuda"):
    import numpy as np
    import torch
    import torch.nn.functional as functional

    from plm.model.layers import _prompt_set_masks, _symmetric_margin_loss
    from plm.training.data import collate_records

    stem = f"{state}-batch-{offset}"
    arrays, unused = {}, {}
    report = {
        "state": state,
        "train_slice": [offset, offset + len(records)],
        "complete": False,
        "queries": queries,
        "unused": unused,
        "state_before_sha256": _state_identity(model),
    }
    try:
        batch = collate_records(records, pad_id=pad_id)
        ids, labels, attention = (
            x.to(device) for x in (batch.input_ids, batch.labels, batch.attention_mask)
        )
        shifted = labels.masked_fill(~attention, -100)[:, 1:]
        parameters = (model.token_embedding.weight, model.symmetric_relation_projection.weight)
        for block, parameter in zip(_BLOCKS, parameters, strict=True):
            arrays[f"parameter__{block}"] = parameter.detach().cpu().numpy().copy()
        for key, value in {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": attention,
            "shifted_labels": shifted,
        }.items():
            arrays[key] = value.detach().cpu().numpy().copy()
        output = model(ids, labels=labels, attention_mask=attention)
        logits = output.symmetric_relation_logits
        excluded = ids[:, 1] - 1024
        positive, negative, _, _ = _prompt_set_masks(logits, shifted, excluded_ids=excluded)
        margin, count, active_count = _symmetric_margin_loss(
            logits, shifted, excluded_ids=excluded, margin=1.0
        )
        active_config = model.config.symmetric_margin_loss_weight > 0
        _require(
            (output.symmetric_margin_loss is not None) == active_config, "model margin presence"
        )
        if active_config:
            _require(
                torch.equal(margin, output.symmetric_margin_loss)
                and count == output.symmetric_margin_query_count
                and active_count == output.symmetric_margin_active_count,
                "helper/model margin mismatch",
            )
        losses = {
            "ce": output.task_loss,
            "prompt_bce": output.prompt_set_loss,
            "symmetric_bce": output.symmetric_relation_loss,
            "margin": margin,
            "total": output.loss,
        }
        reconstructed = losses["ce"] + losses["prompt_bce"] + losses["symmetric_bce"]
        if active_config:
            reconstructed = reconstructed + 0.1 * margin
        _require(
            torch.isclose(output.loss, reconstructed, rtol=1e-5, atol=1e-6).item(),
            "total loss decomposition",
        )
        tensors = {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": attention,
            "shifted_labels": shifted,
            "positive_mask": positive.bool(),
            "negative_mask": negative.bool(),
            "symmetric_logits": logits,
            "prompt_logits": output.prompt_set_logits,
            "margin_hinges": functional.relu(
                1.0
                + logits.masked_fill(~negative.bool(), -torch.inf).amax(-1)
                - logits.masked_fill(~positive.bool(), torch.inf).amin(-1)
            ),
            "token_ce_unreduced": functional.cross_entropy(
                output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                shifted.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(shifted.shape),
        }
        for name, tensor in {**tensors, **{f"losses__{k}": v for k, v in losses.items()}}.items():
            arrays[name] = tensor.detach().cpu().numpy().copy()
        _require(
            all(np.isfinite(a).all() for a in arrays.values()), "nonfinite forward observation"
        )
        report.update(
            losses={k: float(v.detach()) for k, v in losses.items()},
            margin_query_count=count,
            margin_active_count=active_count,
            dimension_ids=sorted({int(v) for v in ids[:, 2].tolist()}),
            dimension_counts={d: sum(r.dimension == d for r in records) for d in ("TYPE", "COLOR")},
            total_loss_residual=float((output.loss - reconstructed).detach()),
        )
        for index, objective in enumerate(_OBJECTIVES):
            gradients = torch.autograd.grad(
                losses[objective],
                parameters,
                allow_unused=True,
                retain_graph=index < len(_OBJECTIVES) - 1,
            )
            unused[objective] = {}
            for block, gradient in zip(_BLOCKS, gradients, strict=True):
                unused[objective][block] = gradient is None
                arrays[f"gradient__{objective}__{block}"] = (
                    np.zeros_like(arrays[f"parameter__{block}"])
                    if gradient is None
                    else gradient.detach().cpu().numpy().copy()
                )
        _require(
            all(np.isfinite(a).all() and a.dtype != object for a in arrays.values()),
            "nonfinite/raw object array",
        )
        report["groups"] = _analysis(arrays, unused, report["dimension_ids"])
        report["radial"] = _radial(arrays)
        _require(report["radial"]["passed"], "radial derivative identity failed")
        report["complete"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["state_after_sha256"] = _state_identity(model)
        report["state_unchanged"] = report["state_after_sha256"] == report["state_before_sha256"]
        report["all_parameter_grad_fields_none"] = all(p.grad is None for p in model.parameters())
        report["complete"] = (
            report["complete"]
            and report["state_unchanged"]
            and report["all_parameter_grad_fields_none"]
        )
        raw_path = directory / (stem + ".npz")
        with raw_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report["raw_sha256"] = _sha(raw_path)
        report["array_manifest"] = {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()
        }
        with (directory / (stem + ".json")).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    _require(report["complete"], "state mutated or gradients accumulated")
    return report


def _aggregate(reports):
    # Scalar means only: never average gradient vectors or relabel mean cosines.
    def leaves(value, prefix=""):
        for key, item in value.items():
            path = f"{prefix}/{key}"
            if isinstance(item, dict):
                yield from leaves(item, path)
            elif key != "reason" and (type(item) in (float, int) or item is None):
                yield path, item

    grouped = {}
    for report in reports:
        for path, value in leaves(report["groups"]):
            grouped.setdefault(path, []).append(value)
    return {
        path: {
            "mean_of_defined_batch_values": math.fsum(v for v in values if v is not None)
            / sum(v is not None for v in values)
            if any(v is not None for v in values)
            else None,
            "defined_count": sum(v is not None for v in values),
            "batch_count": len(reports),
        }
        for path, values in grouped.items()
    }


def main():
    script = Path(__file__).resolve()
    script_bytes = script.read_bytes()
    root = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=root / "runs/learning/margin-gradient-diagnosis-v1/summary.json"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=root / "docs/experiments/2026-09-25-margin-gradient-diagnosis-plan.md",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    _require(
        Path.cwd().resolve() == root and "torch" not in sys.modules,
        "standalone repository-root launch required",
    )
    out = args.out.resolve()
    context = _preflight(root, args.plan, out)
    helper = context["helper"]
    context["inputs"][str(script)] = hashlib.sha256(script_bytes).hexdigest()
    helper._unchanged(context)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "torch_imported": False,
                    "model_executed": False,
                    "states": 5,
                    "training_query_count": 96,
                    "batch_offsets": _OFFSETS,
                }
            )
        )
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": script_bytes,
        "helper.py": Path(helper.__file__).read_bytes(),
        "plan.md": args.plan.read_bytes(),
        "source.zip": context["source"],
        "configs.zip": helper._archive(
            root, sorted(k for k in context["live_inventory"] if k.startswith("configs/"))
        ),
    }
    for name, content in snapshots.items():
        with out.with_suffix("." + name).open("xb") as stream:
            stream.write(content)
    result = {
        "complete": False,
        "independent_audit_status": "required",
        "diagnostic_accepted": False,
        "plan_sha256": _PLAN,
        "source_archive_sha256": _SOURCE,
        "script_sha256": context["inputs"][str(script)],
        "input_sha256": context["inputs"],
        "live_source_config_sha256": context["live_inventory"],
        "snapshot_sha256": {k: _sha(out.with_suffix("." + k)) for k in snapshots},
        "effective_configs": {
            a: d["config"].model_dump(mode="json") for a, d in context["arms"].items()
        },
        "corpus_identity": context["corpus_identity"],
        "split_hash": context["split_hash"],
        "queries": context["queries"],
        "observations": [],
        "state_aggregates": {},
        "parameter_updates": 0,
        "training_executed": False,
        "limitations": [
            (
                "One seed, 96 distinct training queries, five states; "
                "480 query observations are not independent queries."
            ),
            "FP32 eval-mode gradients differ from original BF16 training and do not replay AdamW.",
            (
                "Embedding subgroups overlap the full embedding block; "
                "no transformer-block gradients measured."
            ),
            "Saved-vector audit cannot independently prove autograd or neural forward outputs.",
        ],
    }
    model = None
    try:
        import torch

        from plm.experimentation.provenance import capture_runtime_provenance
        from plm.model.architecture import build_model
        from plm.reproducibility import seed_everything
        from plm.training.checkpoint import load_checkpoint

        _require(torch.cuda.is_available(), "CUDA unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        revision, environment = capture_runtime_provenance()
        _require(
            environment == context["screen_environment"],
            "runtime differs from authenticated screen",
        )
        result.update(
            source_commit=revision,
            environment=environment,
            numerical_settings={
                "parameter_dtype": "float32",
                "device": "cuda",
                "model_training": False,
                "autocast_enabled": torch.is_autocast_enabled("cuda"),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
        )
        configs = [context["arms"][arm]["config"] for arm in ("control", "treatment")]
        seed_everything(1729, deterministic=configs[0].deterministic)
        model = build_model(configs[0].model)
        initial_hash = _state_identity(model)
        seed_everything(1729, deterministic=configs[1].deterministic)
        second = build_model(configs[1].model)
        _require(
            model.state_dict().keys() == second.state_dict().keys()
            and all(torch.equal(v, second.state_dict()[k]) for k, v in model.state_dict().items()),
            "control/treatment initialization differs",
        )
        result["initialization"] = {
            "regenerated": True,
            "historically_saved": False,
            "seed": 1729,
            "control_state_sha256": initial_hash,
            "treatment_state_sha256": _state_identity(second),
            "every_state_tensor_exactly_equal": True,
        }
        del second
        for state in [{"name": "initial", "arm": "control", "path": None}, *context["states"]]:
            helper._unchanged(context)
            if state["path"] is not None:
                model = build_model(context["arms"][state["arm"]]["config"].model)
                loaded = load_checkpoint(
                    state["path"], model, map_location="cpu", restore_rng=False
                )
                _require(
                    dict(loaded.metadata) == state["sidecar"],
                    "authoritative payload/sidecar mismatch",
                )
                _check_metadata(
                    loaded.metadata,
                    state["sidecar"]["global_step"],
                    context["arms"][state["arm"]],
                    context["corpus_identity"],
                    context["split_hash"],
                )
            model = model.float().to("cuda").eval()
            _require(
                all(p.dtype == torch.float32 for p in model.parameters())
                and not torch.is_autocast_enabled("cuda"),
                "FP32/no-autocast contract",
            )
            state_hash = _state_identity(model)
            observations = []
            for offset in _OFFSETS:
                helper._unchanged(context)
                report = _measure(
                    model,
                    context["batches"][offset],
                    context["vocabulary"].pad_id,
                    state["name"],
                    offset,
                    out.parent,
                    [q for q in context["queries"] if offset <= q["train_index"] < offset + 32],
                )
                _require(
                    report["state_before_sha256"] == report["state_after_sha256"] == state_hash,
                    "cross-batch parameter drift",
                )
                observations.append(report)
                result["observations"].append(
                    {
                        "state": state["name"],
                        "offset": offset,
                        "state_sha256": state_hash,
                        "report": f"{state['name']}-batch-{offset}.json",
                        "raw": f"{state['name']}-batch-{offset}.npz",
                    }
                )
                print(f"gradient diagnostic {state['name']} batch {offset}: complete", flush=True)
            result["state_aggregates"][state["name"]] = _aggregate(observations)
            model = None
            gc.collect()
            torch.cuda.empty_cache()
        helper._unchanged(context)
        _require(capture_runtime_provenance() == (revision, environment), "runtime changed")
        _require(
            all(_sha(out.with_suffix("." + k)) == v for k, v in result["snapshot_sha256"].items()),
            "snapshot changed",
        )
        result["complete"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        model = None
        result["artifact_sha256"] = {
            p.name: _sha(p) for p in out.parent.glob("*-batch-*.*") if p.suffix in (".npz", ".json")
        }
        helper._write(out, result)
    print(
        json.dumps(
            {
                "complete": True,
                "diagnostic_accepted": False,
                "observations": len(result["observations"]),
                "summary_sha256": _sha(out),
            }
        )
    )


if __name__ == "__main__":
    main()
