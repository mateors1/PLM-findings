"""Register an audited residual derivative and freshly verify final weight bytes."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--decision-sha256", required=True)
args = parser.parse_args()
root = Path.cwd()
run = root / "runs/learning/bilinear-worst-boundary-v1"
prior_path = (
    root / "docs/experiments/2026-09-25-bilinear-seed-replication-model-version-inventory.json"
)
out = root / "docs/experiments/2026-09-25-bilinear-worst-boundary-model-version-inventory.json"
require(not out.exists(), "immutable inventory")
require(
    sha(prior_path) == "5e84f1546fda3ae4e230ad28c98b936043d09e6834e4f26046e7ab9eb2f26034",
    "prior inventory identity",
)
require(sha(run / "decision.json") == args.decision_sha256, "owner decision identity")
prior, decision = read(prior_path), read(run / "decision.json")
require(sha(run / "summary.json") == decision["summary_sha256"], "audited primary identity")
summary = read(run / "summary.json")
require(
    sha(run / "checkpoint-final.pt.json") == summary["artifact_sha256"]["checkpoint-final.pt.json"],
    "audited child sidecar bytes",
)
require(
    decision["child_checkpoint_sha256"] == summary["artifact_sha256"]["checkpoint-final.pt"],
    "audited child checkpoint identity",
)
require(prior["registered_final_count"] == 33, "prior inventory count")
require(
    decision["evidence_accepted"] is True
    and decision["new_checkpoint_created"] is True
    and decision["architecture"] == "plm-frozen-symmetric-bilinear-residual-v1",
    "audited trained derivative",
)
relative = "runs/learning/bilinear-worst-boundary-v1/checkpoint-final.pt"
expected = dict(prior["verified_final_checkpoint_bytes"])
require(len(expected) == 33 and relative not in expected, "new distinct derivative")
expected[relative] = decision["child_checkpoint_sha256"]
found = {p.relative_to(root).as_posix() for p in (root / "runs").rglob("checkpoint-final.pt")}
require(found == set(expected), "final checkpoint inventory scope differs")
for path, digest in expected.items():
    require(sha(root / path) == digest, "checkpoint bytes changed: " + path)
require(len(expected) == 34, "final count")
require(sha(run / "independent-audit.json") == decision["audit_sha256"], "audit identity")
sidecar = read(run / "checkpoint-final.pt.json")
require(sidecar["checkpoint_hash"] == expected[relative], "child sidecar identity")
require(
    sidecar["training_metadata"]["objective"] == "plm-bilinear-worst-boundary-softplus-v1"
    and sidecar["experiment_identity"]["evaluator_version"]
    == "plm-bilinear-worst-boundary-screen-v1"
    and sidecar["experiment_identity"]["seeds"] == [1729]
    and sidecar["training_metadata"]["parent_checkpoint_sha256"]
    == "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1",
    "new objective and original-parent lineage",
)
require(
    sidecar["global_step"] == 2000
    and sidecar["training_metadata"]["parent_training_steps"] == 2000
    and sidecar["training_metadata"]["residual_updates"] == 2000,
    "separate parent and residual steps",
)
result = {
    "schema_version": 1,
    "inventory_kind": "additive-bilinear-worst-boundary-derivative-and-final-byte-verification-v1",
    "observed_at_utc": datetime.now(UTC).isoformat(),
    "prior_inventory_path": prior_path.relative_to(root).as_posix(),
    "prior_inventory_sha256": sha(prior_path),
    "prior_registered_count": 33,
    "added_count": 1,
    "registered_final_count": 34,
    "verified_final_checkpoint_bytes": expected,
    "new_artifact": {
        "run_id": "bilinear-worst-boundary-v1",
        "checkpoint_path": relative,
        "checkpoint_sha256": expected[relative],
        "sidecar_sha256": sha(run / "checkpoint-final.pt.json"),
        "parent_checkpoint_sha256": decision["parent_checkpoint_sha256"],
        "parent_training_steps": 2000,
        "residual_updates": 2000,
        "sidecar": sidecar,
        "summary_sha256": decision["summary_sha256"],
        "audit_sha256": decision["audit_sha256"],
        "decision_sha256": args.decision_sha256,
        "evidence_accepted": True,
        "fixed_quality_gate_passed": decision["fixed_quality_gate_passed"],
        "promoted": False,
        "durable_weight_archive": None,
    },
    "script_sha256": sha(__file__),
    "limitations": [
        "Fresh byte hashes cover all34 final files; older payloads/configs not revalidated.",
        "The new94-state derivative has a separate independently audited checkpoint contract.",
        "Periodic checkpoints are outside scope; local hashes do not establish remote backup.",
    ],
}
with out.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
print(json.dumps({"verified_final_count": 34, "inventory_sha256": sha(out)}))
