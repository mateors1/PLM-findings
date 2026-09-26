"""Bind a completed two-seed audit to a separate owner outcome decision."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
root = Path.cwd()
run = root / "runs/learning/bilinear-seed-replication-v1"
destination = root / "docs/experiments/2026-09-25-bilinear-seed-replication.json"
require(not destination.exists() and not (run / "decision.json").exists(), "immutable outcome")
for seed in (1730, 1731):
    require(
        not (
            destination.parent / f"2026-09-25-bilinear-seed-replication-training-{seed}.json"
        ).exists(),
        "immutable training history",
    )
require(sha(run / "aggregate-v2.json") == args.summary_sha256, "aggregate primary identity")
require(sha(run / "independent-audit.json") == args.audit_sha256, "aggregate audit identity")
summary, audit = read(run / "aggregate-v2.json"), read(run / "independent-audit.json")
require(summary["complete"] is True, "complete aggregate")
require(audit["complete"] is True and audit["audit_passed"] is True, "completed aggregate audit")
require(audit["summary_sha256"] == args.summary_sha256, "audited aggregate")
require(audit["script_sha256"] == sha(run / "independent-audit-v2.py"), "frozen auditor")
require(
    summary["plan_sha256"]
    == audit["plan_sha256"]
    == "d5a36d48eac3c40a30a5c29c2126f3d4f815f5ecb4d93a03f8cc47f771e75b54",
    "frozen replication plan",
)
require(summary["gate"] == audit["gate"], "independently reconstructed gate")
aggregate_execution = read(run / "aggregate-v2-execution-receipt.json")
require(
    aggregate_execution["exit_code"] == 0
    and aggregate_execution["terminal_completion_observed_by_primary"] is True
    and aggregate_execution["summary_sha256"] == args.summary_sha256
    and aggregate_execution["stdout_sha256"] == sha(run / "aggregate-v2-stdout.txt"),
    "observed aggregate terminal completion",
)
for path, digest in audit["input_sha256"].items():
    require(sha(path) == digest, "audited input drift: " + path)
require([item["seed"] for item in audit["per_seed"]] == [1730, 1731], "both ordered fresh seeds")
children = []
details = []
for item in audit["per_seed"]:
    seed = item["seed"]
    folder = run / f"seed-{seed}"
    require(sha(folder / "summary.json") == item["summary_sha256"], "per-seed primary identity")
    require(
        sha(folder / "independent-audit.json") == item["audit_sha256"], "per-seed audit identity"
    )
    primary = read(folder / "summary.json")
    execution = read(folder / "execution-receipt.json")
    require(
        execution["seed"] == seed
        and execution["exit_code"] == 0
        and execution["terminal_completion_observed_by_primary"] is True
        and execution["summary_sha256"] == item["summary_sha256"]
        and execution["stdout_sha256"] == sha(folder / "primary-stdout.txt"),
        "observed primary terminal completion",
    )
    training = read(folder / "training.json")
    require(training["complete"] is True and training["completed_updates"] == 2000, "completed fit")
    for name, digest in primary["artifact_sha256"].items():
        require(sha(folder / name) == digest, "per-seed artifact drift: " + name)
    require(sha(folder / "checkpoint-final.pt") == training["checkpoint_sha256"], "child weights")
    children.append(
        {
            "seed": seed,
            "checkpoint_path": (folder / "checkpoint-final.pt").relative_to(root).as_posix(),
            "checkpoint_sha256": training["checkpoint_sha256"],
            "sidecar_sha256": sha(folder / "checkpoint-final.pt.json"),
            "summary_sha256": item["summary_sha256"],
            "audit_sha256": item["audit_sha256"],
            "execution_receipt_sha256": sha(folder / "execution-receipt.json"),
            "quality_gate_passed": item["gate"]["primary_checks_passed"],
        }
    )
    history_path = destination.parent / f"2026-09-25-bilinear-seed-replication-training-{seed}.json"
    write(
        history_path,
        {
            "seed": seed,
            "source_training_sha256": sha(folder / "training.json"),
            "completed_updates": training["completed_updates"],
            "initial_pre_update_loss": training["initial_pre_update_loss"],
            "final_post_update_loss": training["final_post_update_loss"],
            "history": training["history"],
        },
    )
    details.append(
        {
            "seed": seed,
            "training_history_artifact": {
                "path": history_path.relative_to(root).as_posix(),
                "sha256": sha(history_path),
            },
            "initial_pre_update_loss": training["initial_pre_update_loss"],
            "final_post_update_loss": training["final_post_update_loss"],
            "checkpoint": item["checkpoint"],
            "child": item["child"],
            "comparisons": item["comparisons"],
            "gate": item["gate"],
            "environment": primary["environment"],
            "wall_seconds": primary["wall_seconds"],
            "peak_memory": primary["peak_memory"],
        }
    )
passed = audit["gate"]["primary_checks_passed"]
require(type(passed) is bool, "Boolean campaign gate")
require(passed == all(c["quality_gate_passed"] for c in children), "fresh seed conjunction")
decision = {
    "date": "2026-09-25",
    "campaign_version": "bilinear-seed-replication-v1",
    "evidence_accepted": True,
    "fixed_quality_gate_passed": passed,
    "policy_promoted": False,
    "default_changed": False,
    "protected_test_used": False,
    "standard_serving_supported": False,
    "new_checkpoint_count": 2,
    "fresh_seeds": [1730, 1731],
    "reused_selection_seed": 1729,
    "architecture": "plm-frozen-symmetric-bilinear-residual-v1",
    "objective": "plm-bilinear-residual-balanced-bce-v1",
    "evaluator": "plm-bilinear-seed-replication-v1",
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "auditor_sha256": audit["script_sha256"],
    "decision_script_sha256": sha(__file__),
    "aggregation_runner_sha256": summary["aggregation_runner_sha256"],
    "evidence_repair_declaration_sha256": sha(
        root / "docs/experiments/2026-09-25-bilinear-replication-evidence-repair.md"
    ),
    "failed_aggregate_summary_sha256": sha(run / "summary.json"),
    "failed_aggregate_execution_receipt_sha256": sha(run / "execution-receipt.json"),
    "original_owner_helper_sha256": sha(run / "record-decision.py"),
    "aggregate_execution_receipt_sha256": sha(run / "aggregate-v2-execution-receipt.json"),
    "children": children,
    "gate": audit["gate"],
    "outcome": "Both fresh seeds pass their matching fixed gates; no automatic promotion."
    if passed
    else "Replication quality gate fails; preserve both children and all per-seed evidence.",
    "basis": "Independent saved-evidence and CPU checkpoint audits, with observed terminal "
    "completion for both fresh fits. No independent replay of CUDA training.",
}
portable = {
    "date": "2026-09-25",
    "campaign_version": decision["campaign_version"],
    "status": "replication screen passed" if passed else "replication quality gate rejected",
    "plan_sha256": audit["plan_sha256"],
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "decision": decision,
    "per_seed": details,
    "historical1729": audit["historical1729"],
    "fresh_two": audit["fresh_two"],
    "all_three": audit["all_three"],
    "gate": audit["gate"],
    "input_sha256": audit["input_sha256"],
    "limitations": audit["limitations"],
}
write(run / "decision.json", decision)
portable["decision_sha256"] = sha(run / "decision.json")
write(destination, portable)
print(
    json.dumps(
        {
            "quality_passed": passed,
            "decision_sha256": sha(run / "decision.json"),
            "portable_sha256": sha(destination),
        }
    )
)
