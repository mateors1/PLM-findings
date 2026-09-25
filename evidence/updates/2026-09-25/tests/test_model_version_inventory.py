"""The academic index must distinguish evidence consistency from model validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts/inventory_model_versions.py"
_SPEC = importlib.util.spec_from_file_location("model_version_inventory", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
inventory = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inventory)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def saved_run(tmp_path: Path) -> tuple[Path, Path]:
    directory = tmp_path / "runs" / "example_s17_v1"
    directory.mkdir(parents=True)
    checkpoint = directory / "checkpoint-final.pt"
    # Deliberately not a Torch pickle: this inventory must not deserialize it.
    checkpoint.write_bytes(b"opaque-checkpoint-fixture")
    archive = directory / "source.zip"
    members = {"src/example.py": b"x = 1\n", "pyproject.toml": b"fixture", "uv.lock": b"lock"}
    tree = hashlib.sha256()
    with zipfile.ZipFile(archive, "w") as stream:
        for name, payload in members.items():
            stream.writestr(name, payload)
            encoded = name.encode()
            tree.update(len(encoded).to_bytes(8, "big") + encoded)
            tree.update(len(payload).to_bytes(8, "big") + payload)
    config = {
        "run_name": directory.name,
        "seed": 17,
        "model": {"d_model": 8},
        "train": {"batch_size": 4, "grad_accum_steps": 2},
    }
    identity: dict[str, Any] = {
        "resolved_config_hash": inventory._canonical(
            {k: v for k, v in config.items() if k != "run_name"}
        ),
        "seeds": [17],
        "split_hash": "split",
        "graph_hash": "graph",
        "records_hash": "records",
        "vocabulary_hash": "vocabulary",
        "environment": {
            "source_archive_sha256": inventory._digest(archive),
            "source_tree_sha256": tree.hexdigest(),
            "uv_lock_sha256": hashlib.sha256(b"lock").hexdigest(),
        },
    }
    sidecar = {
        "config": config["train"],
        "training_metadata": {
            "model_config": config["model"],
            "batch_size": 4,
            "grad_accum_steps": 2,
            "train_loss": 0.25,
        },
        "experiment_identity": identity,
        "global_step": 8,
        "checkpoint_hash": inventory._digest(checkpoint),
        "split_hash": "split",
        "corpus_identity": {
            k: identity[k] for k in ("graph_hash", "records_hash", "vocabulary_hash")
        },
    }
    _write(directory / "run.json", {"config": config, "identity": identity})
    _write(directory / "checkpoint-final.pt.json", sidecar)
    _write(
        directory / "training-result.json",
        {
            "identity": identity,
            "global_step": 8,
            "train_loss": 0.25,
            "checkpoint_hash": inventory._digest(checkpoint),
            "checkpoint": str(checkpoint.relative_to(tmp_path)),
        },
    )
    return tmp_path, directory


def test_preserves_unknown_lineage_and_limits(saved_run: tuple[Path, Path]) -> None:
    root, directory = saved_run
    result = inventory._inventory(root, root / "runs")
    assert result["all_discovered_runs_consistent"] is True
    assert result["verified_runs"] == 1
    row = result["runs"][0]
    assert row["checkpoint_sha256"] == inventory._digest(directory / "checkpoint-final.pt")
    assert row["training_parent"]["checkpoint_sha256"] is None
    assert row["quality_or_promotion_claim"] is False
    assert "No checkpoint payload deserialization" in result["limitations"][0]


@pytest.mark.parametrize(
    "mutation", ["weights", "config", "identity", "step", "source", "missing", "target"]
)
def test_inconsistent_history_is_saved_as_failure(
    saved_run: tuple[Path, Path], mutation: str
) -> None:
    root, directory = saved_run
    if mutation == "weights":
        (directory / "checkpoint-final.pt").write_bytes(b"different weights")
    elif mutation == "source":
        (directory / "source.zip").write_bytes(b"different source archive")
    elif mutation == "missing":
        (directory / "training-result.json").unlink()
    else:
        path = directory / ("run.json" if mutation == "config" else "training-result.json")
        value = inventory._read(path)
        if mutation == "config":
            value["config"]["model"]["d_model"] = 16
        elif mutation == "identity":
            value["identity"]["split_hash"] = "another split"
        elif mutation == "step":
            value["global_step"] = 9
        else:
            value["checkpoint"] = "runs/another/checkpoint-final.pt"
        _write(path, value)
    result = inventory._inventory(root, root / "runs")
    assert result["all_discovered_runs_consistent"] is False
    assert result["failed_runs"] == 1
    assert result["runs"][0]["error"]


def test_empty_inventory_cannot_pass(tmp_path: Path) -> None:
    result = inventory._inventory(tmp_path, tmp_path / "runs")
    assert result["discovered_runs"] == 0
    assert result["all_discovered_runs_consistent"] is False


def test_archive_tree_and_lock_are_independently_checked(saved_run: tuple[Path, Path]) -> None:
    _, directory = saved_run
    environment = inventory._read(directory / "run.json")["identity"]["environment"]
    for key in ("source_tree_sha256", "uv_lock_sha256"):
        changed = {**environment, key: "0" * 64}
        with pytest.raises(ValueError):
            inventory._source_archive(directory / "source.zip", changed)


@pytest.mark.parametrize("field", ["model_config", "batch_size", "grad_accum_steps", "train_loss"])
def test_metadata_contradictions_fail(saved_run: tuple[Path, Path], field: str) -> None:
    root, directory = saved_run
    path = directory / "checkpoint-final.pt.json"
    sidecar = inventory._read(path)
    sidecar["training_metadata"][field] = {"d_model": 16} if field == "model_config" else 99
    _write(path, sidecar)
    with pytest.raises(ValueError, match=f"training metadata {field} disagreement"):
        inventory._run(directory, root)


def test_missing_metadata_is_explicitly_unavailable(saved_run: tuple[Path, Path]) -> None:
    root, directory = saved_run
    path = directory / "checkpoint-final.pt.json"
    sidecar = inventory._read(path)
    del sidecar["training_metadata"]
    _write(path, sidecar)
    row = inventory._run(directory, root)
    assert set(row["training_metadata_checks"].values()) == {"unavailable"}


def test_existing_inventory_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("retained evidence", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--out", str(output)])
    with pytest.raises(FileExistsError, match="immutable"):
        inventory.main()
    assert output.read_text(encoding="utf-8") == "retained evidence"
