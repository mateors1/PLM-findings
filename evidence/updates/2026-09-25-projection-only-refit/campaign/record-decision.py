"""Record the owner's audited decision without changing primary evidence."""

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
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
parser.add_argument("--runner-sha256", required=True)
parser.add_argument("--auditor-sha256", required=True)
args = parser.parse_args()
root = Path.cwd()
run = root / "runs/learning/projection-only-refit-v1"
destination = root / "docs/experiments/2026-09-25-projection-only-refit.json"
require(not destination.exists() and not (run / "decision.json").exists(), "immutable decision")
require(sha(run / "summary.json") == args.summary_sha256, "primary identity")
require(sha(run / "independent-audit.json") == args.audit_sha256, "audit identity")
require(sha(run / "summary.script.py") == args.runner_sha256, "executed runner identity")
require(sha(run / "independent-audit.py") == args.auditor_sha256, "auditor identity")
summary, audit = read(run / "summary.json"), read(run / "independent-audit.json")
require(summary["complete"] is True and summary["final_identity_check"] is True, "complete run")
require(audit["complete"] is True and audit["audit_passed"] is True, "accepted audit")
require(summary["acceptance"] is False and audit["acceptance"] is False, "separate decision")
require(summary["script_sha256"] == args.runner_sha256, "primary script binding")
require(audit["script_sha256"] == args.auditor_sha256, "audit script binding")
require(audit["summary_sha256"] == args.summary_sha256, "audited primary")
require(
    audit["plan_sha256"]
    == summary["plan_sha256"]
    == "2ac4d157ad8c079561d642199fec8de2c3da1866819baa48f52080e9b0f39bfe",
    "frozen plan",
)
require(summary["gate"] == audit["gate"], "independent gate agreement")
for path, digest in audit["input_sha256"].items():
    require(sha(path) == digest, "audited input changed: " + path)
for name, digest in summary["artifact_sha256"].items():
    require(sha(run / name) == digest, "primary artifact changed: " + name)
parent, child, training = (
    read(run / name) for name in ("parent.json", "child.json", "training.json")
)
for name, report in (("parent", parent), ("child", child)):
    require(report["complete"] is True and report["query_count"] == 222, "complete validation")
    for key, value in audit[name].items():
        require(report[key] == value, "audited report mismatch")
passed = summary["gate"]["primary_checks_passed"]
require(type(passed) is bool, "Boolean quality gate")
decision = {
    "date": "2026-09-25",
    "campaign_version": summary["campaign_version"],
    "evidence_accepted": True,
    "fixed_quality_gate_passed": passed,
    "policy_promoted": False,
    "default_changed": False,
    "training_executed": True,
    "new_checkpoint_created": True,
    "protected_test_used": False,
    "standard_serving_supported": False,
    "parent_checkpoint_sha256": audit["checkpoint"]["parent_checkpoint_sha256"],
    "child_checkpoint_sha256": training["checkpoint_sha256"],
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "auditor_sha256": args.auditor_sha256,
    "decision_script_sha256": sha(__file__),
    "gate": summary["gate"],
    "basis": "Independent saved-evidence audit verifies labels, split, predictions, metrics, "
    "final checkpoint and optimizer state, frozen tensors, lineage and declared gate. "
    "It does not repeat training or independently regenerate logits.",
    "outcome": (
        "Single-seed screen passes; fixed replication may be proposed. No automatic promotion."
        if passed
        else "Fixed quality screen fails; retain the trained derivative and negative evidence."
    ),
}
rows = []
for old, new in zip(parent["responses"], child["responses"], strict=True):
    require(old["index"] == new["index"] and old["prompt_ids"] == new["prompt_ids"], "paired query")
    truth = set(new["expected_set_ids"])
    selected = set(new["selected_set_ids"])
    rows.append(
        {
            **{k: new[k] for k in ("index", "subject", "dimension", "group")},
            "expected_count": len(truth),
            "parent_dense_metrics": old["metrics"],
            "child_dense_metrics": new["metrics"],
            "parent_strict_separation": old["strict_separation"],
            "child_strict_separation": new["strict_separation"],
            "child_false_positive_ids": sorted(selected - truth),
            "child_false_negative_ids": sorted(truth - selected),
        }
    )
portable = {
    "date": "2026-09-25",
    "campaign_version": summary["campaign_version"],
    "status": "single-seed quality screen passed" if passed else "rejected by fixed quality gate",
    "objective": summary["objective"],
    "evaluator": summary["evaluator"],
    "inference_policy": "dense-positive-head-subject-excluded-ascending-ids-fp32-b8-last6-v1",
    "plan_sha256": summary["plan_sha256"],
    "runner_sha256": args.runner_sha256,
    "recipe_sha256": summary["recipe_sha256"],
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "decision": decision,
    "parent": audit["parent"],
    "child": audit["child"],
    "comparisons": audit["comparisons"],
    "checkpoint": audit["checkpoint"],
    "child_training_identity": summary["child_training_identity"],
    "environment": summary["environment"],
    "numerical_settings": summary["numerical_settings"],
    "peak_memory": summary["peak_memory"],
    "wall_seconds": summary["wall_seconds"],
    "training_loss_history": training["history"],
    "per_query_diagnostics": rows,
    "input_sha256": summary["input_sha256"],
    "snapshot_sha256": summary["snapshot_sha256"],
    "limitations": summary["limitations"] + audit["limitations"],
}
write(run / "decision.json", decision)
portable["decision_sha256"] = sha(run / "decision.json")
write(destination, portable)
print(
    json.dumps(
        {
            "quality_passed": passed,
            "decision_sha256": portable["decision_sha256"],
            "portable_sha256": sha(destination),
        }
    )
)
