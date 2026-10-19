"""Add two audited seed replicas and freshly verify final checkpoint bytes."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--decision-sha256", required=True)
args = parser.parse_args()
root = Path.cwd()
run = root / "runs/learning/bilinear-budget8000-replication-v1"
prior_path = root / "docs/experiments/2026-09-25-bilinear-budget8000-model-version-inventory.json"
out = (
    root
    / "docs/experiments/2026-09-25-bilinear-budget8000-replication-model-version-inventory.json"
)
require(not out.exists(), "immutable inventory")
require(
    sha(prior_path) == "58b8de7ab229c29b59adb426e41b623df5f55fb6fdbc647a44ccd67e551c1ee1",
    "prior inventory identity",
)
require(sha(run / "decision.json") == args.decision_sha256, "owner decision identity")
prior, decision = read(prior_path), read(run / "decision.json")
require(
    decision["evidence_accepted"] is True
    and decision["new_checkpoint_count"] == 2
    and decision["fresh_seeds"] == [1730, 1731]
    and decision["reused_selection_seed"] == 1729,
    "completed two-seed evidence",
)
require(sha(run / "aggregate-v2.json") == decision["summary_sha256"], "audited aggregate identity")
require(sha(run / "independent-audit-v2.json") == decision["audit_sha256"], "audited aggregate")
require(prior["registered_final_count"] == 35, "prior inventory count")
expected = dict(prior["verified_final_checkpoint_bytes"])
require(len(expected) == 35, "prior final files")
parents = {
    1730: "5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2",
    1731: "ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d",
}
require([c["seed"] for c in decision["children"]] == [1730, 1731], "distinct ordered children")
added = []
for child in decision["children"]:
    seed = child["seed"]
    folder = run / f"seed-{seed}"
    relative = (folder / "checkpoint-final.pt").relative_to(root).as_posix()
    require(child["checkpoint_path"] == relative and relative not in expected, "new child path")
    require(sha(folder / "summary.json") == child["summary_sha256"], "audited seed summary")
    require(sha(folder / "independent-audit-v2.json") == child["audit_sha256"], "audited seed")
    summary = read(folder / "summary.json")
    require(
        child["checkpoint_sha256"] == summary["artifact_sha256"]["checkpoint-final.pt"]
        and child["sidecar_sha256"] == summary["artifact_sha256"]["checkpoint-final.pt.json"]
        and sha(folder / "checkpoint-final.pt.json") == child["sidecar_sha256"],
        "authenticated checkpoint and sidecar",
    )
    sidecar = read(folder / "checkpoint-final.pt.json")
    metadata = sidecar["training_metadata"]
    require(
        sidecar["checkpoint_hash"] == child["checkpoint_sha256"]
        and sidecar["global_step"] == 8000
        and metadata["parent_training_steps"] == 2000
        and metadata["residual_updates"] == 8000
        and metadata["objective"] == "plm-bilinear-residual-balanced-bce-v1"
        and metadata["parent_checkpoint_sha256"] == parents[seed]
        and sidecar["experiment_identity"]["seeds"] == [seed]
        and sidecar["experiment_identity"]["evaluator_version"]
        == "plm-bilinear-budget8000-replication-v1",
        "seed-specific parent and training identity",
    )
    expected[relative] = child["checkpoint_sha256"]
    added.append(
        {
            **child,
            "parent_checkpoint_sha256": parents[seed],
            "parent_training_steps": 2000,
            "residual_updates": 8000,
            "sidecar": sidecar,
            "promoted": False,
            "durable_weight_archive": None,
        }
    )
found = {p.relative_to(root).as_posix() for p in (root / "runs").rglob("checkpoint-final.pt")}
require(len(expected) == 37 and found == set(expected), "final inventory scope")
for path, digest in expected.items():
    require(sha(root / path) == digest, "checkpoint bytes changed: " + path)
result = {
    "schema_version": 1,
    "inventory_kind": (
        "additive-bilinear-budget8000-two-seed-derivatives-and-final-byte-verification-v1"
    ),
    "observed_at_utc": datetime.now(UTC).isoformat(),
    "prior_inventory_path": prior_path.relative_to(root).as_posix(),
    "prior_inventory_sha256": sha(prior_path),
    "prior_registered_count": 35,
    "added_count": 2,
    "registered_final_count": 37,
    "verified_final_checkpoint_bytes": expected,
    "new_artifacts": added,
    "decision_sha256": args.decision_sha256,
    "script_sha256": sha(__file__),
    "limitations": [
        "Fresh byte hashes cover 37 final files; older payloads/configs not all revalidated.",
        "Each new94-state derivative has a separate independently audited seed lineage.",
        "Reused1729 creates no new checkpoint; periodic checkpoints remain outside scope.",
        "Local hashes establish identity, not durable remote checkpoint availability.",
    ],
}
with out.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps({"verified_final_count": 37, "inventory_sha256": sha(out)}))
