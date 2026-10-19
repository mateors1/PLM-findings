"""Bind the owner's diagnostic decision to a completed independent audit."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
root = Path.cwd()
run = root / "runs/learning/diagonal-feasibility-v1"
portable_path = root / "docs/experiments/2026-09-25-diagonal-feasibility.json"
require(not portable_path.exists() and not (run / "decision.json").exists(), "immutable decision")
require(sha(run / "summary.json") == args.summary_sha256, "primary identity")
require(sha(run / "independent-audit.json") == args.audit_sha256, "audit identity")
summary = read(run / "summary.json")
audit = read(run / "independent-audit.json")
execution = read(run / "execution-receipt.json")
require(summary["complete"] is True and summary["final_identity_check"] is True, "complete run")
require(audit["complete"] is True and audit["audit_passed"] is True, "completed audit")
require(summary["acceptance"] is False and audit["acceptance"] is False, "separate acceptance")
require(audit["summary_sha256"] == args.summary_sha256, "audited summary")
require(audit["script_sha256"] == sha(run / "independent-audit.py"), "auditor identity")
require(summary["script_sha256"] == sha(run / "summary.script.py"), "executed source")
require(
    summary["plan_sha256"]
    == audit["plan_sha256"]
    == "7b2e575dfd651d10edbf2d8c2a5f4a9f3268d6ad350248fafd35f1e09efe694b",
    "frozen plan",
)
require(summary["outcome"] == audit["outcome"], "independent outcome")
require(
    execution["exit_code"] == 0
    and execution["terminal_completion_observed_by_primary"] is True
    and execution["summary_sha256"] == args.summary_sha256,
    "observed primary terminal completion",
)
require(sha(run / "primary-stdout.txt") == execution["stdout_sha256"], "primary stdout")
for path, digest in audit["input_sha256"].items():
    require(sha(path) == digest, "audited input drift: " + path)
for name, digest in summary["artifact_sha256"].items():
    require(sha(run / name) == digest, "primary artifact drift: " + name)
for worker in summary["workers"]:
    require(
        worker["reaped"] is True
        and worker["process_running"] is False
        and not any(k in worker for k in ("parent_error", "cleanup_error", "report_error")),
        "worker execution integrity",
    )
    require(worker["timed_out"] or worker["exit_code"] == 0, "unexpected worker exit")
    if not worker["timed_out"]:
        report = read(run / (worker["dimension"] + ".json"))
        require(
            report["complete"] is True
            and report["final_identity_check"] is True
            and "error" not in report,
            "unexpected solver software failure",
        )
require(
    all(
        summary[k] == 0
        for k in (
            "validation_predictions",
            "protected_test_predictions",
            "neural_checkpoints_created",
        )
    ),
    "diagnostic-only scope",
)
decision = {
    "date": "2026-09-25",
    "campaign_version": summary["campaign_version"],
    "evidence_accepted": True,
    "diagnostic_outcome": audit["outcome"],
    "quality_gate_applicable": False,
    "policy_promoted": False,
    "default_changed": False,
    "training_executed": False,
    "new_checkpoint_created": False,
    "protected_test_used": False,
    "registered_final_neural_checkpoints": 29,
    "parent_checkpoint_sha256": summary["extraction"]["parent_checkpoint_sha256"],
    "feature_file_sha256": summary["extraction"]["file_sha256"],
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "auditor_sha256": audit["script_sha256"],
    "execution_receipt_sha256": sha(run / "execution-receipt.json"),
    "decision_script_sha256": sha(__file__),
    "per_dimension": audit["per_dimension"],
    "basis": "Independent saved-evidence audit of frozen provenance, training labels, "
    "bounded solver traces and exact witnesses where present. Inconclusive is not proof "
    "of feasibility or impossibility. No new validation or serving claim.",
}
write(run / "decision.json", decision)
portable = {
    "date": "2026-09-25",
    "campaign_version": summary["campaign_version"],
    "evaluator": summary["evaluator"],
    "outcome": audit["outcome"],
    "scope": summary["scope"],
    "plan_sha256": summary["plan_sha256"],
    "runner_sha256": summary["script_sha256"],
    "recipe_sha256": summary["recipe_sha256"],
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "decision_sha256": sha(run / "decision.json"),
    "decision": decision,
    "training_queries": audit["training_queries"],
    "constraint_count": audit["constraint_count"],
    "per_dimension": audit["per_dimension"],
    "workers": summary["workers"],
    "solver_environment": summary["solver_environment"],
    "feature_identity": summary["extraction"],
    "wall_seconds": summary["wall_seconds"],
    "input_sha256": summary["input_sha256"],
    "snapshot_sha256": summary["snapshot_sha256"],
    "artifact_sha256": summary["artifact_sha256"],
    "limitations": summary["limitations"] + audit["limitations"],
}
write(portable_path, portable)
print(
    json.dumps(
        {
            "outcome": audit["outcome"],
            "decision_sha256": sha(run / "decision.json"),
            "portable_sha256": sha(portable_path),
        }
    )
)
