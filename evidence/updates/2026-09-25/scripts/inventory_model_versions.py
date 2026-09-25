"""Inventory final training artifacts without importing Torch or loading pickle payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_archive(path: Path, environment: dict[str, Any]) -> int:
    _require(_digest(path) == environment["source_archive_sha256"], "source archive hash differs")
    tree = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "duplicate source archive members")
        _require(names[-2:] == ["pyproject.toml", "uv.lock"], "source archive dependency inventory")
        _require(
            bool(names[:-2])
            and names[:-2] == sorted(names[:-2])
            and all(n.startswith("src/") and n.endswith(".py") for n in names[:-2]),
            "source archive Python inventory",
        )
        for name in names:
            encoded = name.encode("utf-8")
            payload = archive.read(name)
            tree.update(len(encoded).to_bytes(8, "big") + encoded)
            tree.update(len(payload).to_bytes(8, "big") + payload)
        _require(
            hashlib.sha256(archive.read("uv.lock")).hexdigest() == environment["uv_lock_sha256"],
            "archived lock hash differs",
        )
    _require(tree.hexdigest() == environment["source_tree_sha256"], "archived source tree differs")
    return len(names)


def _run(directory: Path, root: Path) -> dict[str, Any]:
    paths = [
        directory / name
        for name in (
            "run.json",
            "training-result.json",
            "checkpoint-final.pt",
            "checkpoint-final.pt.json",
            "source.zip",
        )
    ]
    hashes = {p.relative_to(root).as_posix(): _digest(p) for p in paths}
    run, training, sidecar = _read(paths[0]), _read(paths[1]), _read(paths[3])
    identity, config = run["identity"], run["config"]
    _require(
        identity == training["identity"] == sidecar["experiment_identity"], "identity disagreement"
    )
    _require(config["train"] == sidecar["config"], "run/sidecar training config disagreement")
    metadata = sidecar.get("training_metadata", {})
    _require(isinstance(metadata, dict), "training metadata must be an object")
    comparisons = {
        "model_config": config["model"],
        "batch_size": config["train"]["batch_size"],
        "grad_accum_steps": config["train"]["grad_accum_steps"],
        "train_loss": training.get("train_loss"),
    }
    metadata_checks = {}
    for name, expected in comparisons.items():
        if name not in metadata or (name == "train_loss" and "train_loss" not in training):
            metadata_checks[name] = "unavailable"
        else:
            _require(metadata[name] == expected, f"training metadata {name} disagreement")
            metadata_checks[name] = "consistent"
    _require(config["run_name"] == directory.name, "run label disagreement")
    historical = {key: value for key, value in config.items() if key != "run_name"}
    _require(
        _canonical(historical) == identity["resolved_config_hash"], "historical config hash differs"
    )
    checkpoint_hash = hashes[paths[2].relative_to(root).as_posix()]
    _require(
        checkpoint_hash == training["checkpoint_hash"] == sidecar["checkpoint_hash"],
        "checkpoint bytes differ from receipts",
    )
    recorded_path = Path(training["checkpoint"])
    resolved = recorded_path if recorded_path.is_absolute() else root / recorded_path
    _require(
        resolved.resolve() == paths[2].resolve(), "training result points at another checkpoint"
    )
    step = training["global_step"]
    _require(
        type(step) is int and step >= 0 and step == sidecar["global_step"], "step disagreement"
    )
    seed = config["seed"]
    _require(type(seed) is int and identity["seeds"] == [seed], "seed disagreement")
    _require(sidecar["split_hash"] == identity["split_hash"], "split disagreement")
    _require(
        sidecar["corpus_identity"]
        == {
            k: identity[k]
            for k in (
                "graph_hash",
                "records_hash",
                "vocabulary_hash",
            )
        },
        "corpus identity disagreement",
    )
    members = _source_archive(paths[4], identity["environment"])
    # Rehash after parsing: the inventory is invalid if any input changes while read.
    _require(
        all(_digest(root / name) == digest for name, digest in hashes.items()),
        "artifact changed during inventory",
    )
    return {
        "run_id": directory.name,
        "status": "byte_and_receipt_consistent",
        "seed": seed,
        "global_step": step,
        "checkpoint_path": paths[2].relative_to(root).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "experiment_identity": identity,
        "resolved_config": config,
        "artifact_sha256": hashes,
        "source_archive_member_count": members,
        "training_metadata_checks": metadata_checks,
        "training_parent": {
            "status": "not_established_by_these_receipts",
            "checkpoint_sha256": None,
        },
        "quality_or_promotion_claim": False,
    }


def _inventory(root: Path, run_root: Path) -> dict[str, Any]:
    directories = sorted(p.parent for p in run_root.glob("*/run.json"))
    records: list[dict[str, Any]] = []
    for directory in directories:
        try:
            records.append(_run(directory, root))
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            records.append({"run_id": directory.name, "status": "failed", "error": str(exc)})
    verified = sum(row["status"] == "byte_and_receipt_consistent" for row in records)
    return {
        "schema_version": 1,
        "inventory_kind": "plm-final-checkpoint-byte-and-receipt-inventory-v1",
        "scope": (
            "Immediate child run.json directories only; final checkpoints, not periodic checkpoints"
        ),
        "run_root": run_root.relative_to(root).as_posix(),
        "discovered_runs": len(records),
        "verified_runs": verified,
        "failed_runs": len(records) - verified,
        "all_discovered_runs_consistent": bool(records) and verified == len(records),
        "limitations": [
            "No checkpoint payload deserialization, tensor inspection or numerical replay",
            "No independent corpus-file or evaluation-result revalidation",
            "No inferred training parentage, quality ranking or promotion",
            "No archive upload, backup guarantee or coverage outside the declared directory scope",
        ],
        "runs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_root = (root / args.run_root).resolve()
    run_root.relative_to(root)
    if args.out.exists():
        raise FileExistsError("inventory is immutable; choose a new output path")
    script_sha = _digest(Path(__file__))
    result = _inventory(root, run_root)
    _require(_digest(Path(__file__)) == script_sha, "inventory script changed during execution")
    result["script_sha256"] = script_sha
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "discovered_runs",
                    "verified_runs",
                    "failed_runs",
                    "all_discovered_runs_consistent",
                )
            }
        )
    )
    if not result["all_discovered_runs_consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
