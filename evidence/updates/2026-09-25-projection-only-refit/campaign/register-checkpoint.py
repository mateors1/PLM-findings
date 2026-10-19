"""Append the audited derivative and freshly verify final checkpoint file bytes."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


root = Path.cwd()
run = root / "runs/learning/projection-only-refit-v1"
prior_path = root / "docs/experiments/2026-09-25-weak-margin-model-version-inventory.json"
destination = root / "docs/experiments/2026-09-25-projection-only-model-version-inventory.json"
require(not destination.exists(), "immutable inventory output")
require(
    sha(run / "decision.json")
    == "da0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6",
    "audited owner decision",
)
prior, decision = read(prior_path), read(run / "decision.json")
require(prior["verified_runs"] == 27 and prior["all_discovered_runs_consistent"], "prior inventory")
require(decision["evidence_accepted"] is True, "accepted evidence")
child_relative = "runs/learning/projection-only-refit-v1/checkpoint-final.pt"
expected = {row["checkpoint_path"]: row["checkpoint_sha256"] for row in prior["runs"]}
require(len(expected) == 27 and child_relative not in expected, "new distinct checkpoint path")
expected[child_relative] = decision["child_checkpoint_sha256"]
discovered = {p.relative_to(root).as_posix() for p in (root / "runs").rglob("checkpoint-final.pt")}
require(discovered == set(expected), "final checkpoint scope differs")
for relative, digest in expected.items():
    require(sha(root / relative) == digest, "checkpoint byte mismatch: " + relative)
sidecar = read(run / "checkpoint-final.pt.json")
require(sidecar["checkpoint_hash"] == expected[child_relative], "child sidecar binding")
write(
    destination,
    {
        "schema_version": 1,
        "inventory_kind": "additive-projection-derivative-and-final-byte-verification-v1",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "prior_inventory_path": prior_path.relative_to(root).as_posix(),
        "prior_inventory_sha256": sha(prior_path),
        "prior_registered_count": 27,
        "added_count": 1,
        "registered_final_count": 28,
        "verified_final_checkpoint_bytes": expected,
        "new_artifact": {
            "run_id": "projection-only-refit-v1",
            "checkpoint_path": child_relative,
            "checkpoint_sha256": expected[child_relative],
            "sidecar_sha256": sha(run / "checkpoint-final.pt.json"),
            "parent_checkpoint_sha256": decision["parent_checkpoint_sha256"],
            "parent_training_steps": 2000,
            "projection_updates": 500,
            "sidecar": sidecar,
            "summary_sha256": decision["summary_sha256"],
            "audit_sha256": decision["audit_sha256"],
            "decision_sha256": sha(run / "decision.json"),
            "evidence_accepted": True,
            "fixed_quality_gate_passed": False,
            "promoted": False,
            "durable_weight_archive": None,
        },
        "script_sha256": sha(__file__),
        "limitations": [
            "Fresh byte hashing covers all28 final files; older payloads/configurations are not revalidated.",
            "The historical inventory retains its original immediate-directory discovery scope.",
            "The new nested derivative has its own independently audited checkpoint contract.",
            "Periodic checkpoints are outside this scope; local hashes do not establish durable backup.",
        ],
    },
)
write(
    run / "execution-receipt.json",
    {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "terminal_session_id": 94155,
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "stdout_sha256": sha(run / "execution-stdout.txt"),
        "summary_sha256": decision["summary_sha256"],
        "gpu_campaign_complete": True,
        "lm_studio_backend": "Stopped by owner at user's request before run; remained offline.",
        "timing_scope": "Descriptive local measurements; no isolated serving comparison.",
    },
)
print(json.dumps({"verified_final_count": 28, "inventory_sha256": sha(destination)}))
