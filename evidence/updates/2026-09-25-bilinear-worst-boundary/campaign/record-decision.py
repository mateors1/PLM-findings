"""Record a separate owner decision for an audited bilinear residual screen."""

import argparse
import hashlib
import json
from collections import Counter
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
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    require(len(data.encode()) <= 512 * 1024, "portable size bound")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
root = Path.cwd()
run = root / "runs/learning/bilinear-worst-boundary-v1"
destination = root / "docs/experiments/2026-09-25-bilinear-worst-boundary.json"
require(not destination.exists() and not (run / "decision.json").exists(), "immutable decision")
require(sha(run / "summary.json") == args.summary_sha256, "primary identity")
require(sha(run / "independent-audit.json") == args.audit_sha256, "audit identity")
summary, audit = read(run / "summary.json"), read(run / "independent-audit.json")
require(summary["complete"] is True and summary["final_identity_check"] is True, "complete run")
require(audit["complete"] is True and audit["audit_passed"] is True, "completed audit")
require(summary["acceptance"] is False and audit["acceptance"] is False, "separate acceptance")
require(audit["summary_sha256"] == args.summary_sha256, "audited summary")
require(audit["script_sha256"] == sha(run / "independent-audit.py"), "auditor identity")
require(summary["script_sha256"] == sha(run / "summary.script.py"), "executed runner")
require(
    summary["scorer_sha256"] == "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
    and summary["script_sha256"] != summary["scorer_sha256"],
    "distinct new orchestrator and unchanged scorer",
)
require(
    summary["plan_sha256"]
    == audit["plan_sha256"]
    == "d2e7ee91ceb1f0128b0d255dde5a22a7f0c3550503497120569e94c6490ef337",
    "frozen plan",
)
require(summary["gate"] == audit["gate"], "independent gate agreement")
require(
    summary["loss_helper_sha256"]
    == "8598f110d7cbc60085c7d099a821386cba3566f4264c91939fada5baa0f0e480",
    "frozen loss",
)
for path, digest in audit["input_sha256"].items():
    require(sha(path) == digest, "audited input drift: " + path)
for name, digest in summary["artifact_sha256"].items():
    require(sha(run / name) == digest, "primary artifact drift: " + name)
execution = read(run / "execution-receipt.json")
require(
    execution["exit_code"] == 0
    and execution["terminal_completion_observed_by_primary"] is True
    and execution["summary_sha256"] == args.summary_sha256
    and execution["stdout_sha256"] == sha(run / "primary-stdout.txt"),
    "observed primary terminal completion",
)
training = read(run / "training.json")
require(training["complete"] is True and training["completed_updates"] == 2000, "complete fit")
require(sha(run / "checkpoint-final.pt") == training["checkpoint_sha256"], "child weights")
require(
    sha(run / "checkpoint-final.pt.json") == training["checkpoint_sidecar_sha256"],
    "child sidecar",
)
passed = audit["gate"]["primary_checks_passed"]
require(type(passed) is bool, "Boolean gate")
child = read(run / "child.json")
per_query = []
for row in child["responses"]:
    truth, selected = set(row["expected_set_ids"]), set(row["selected_set_ids"])
    per_query.append(
        {
            **{key: row[key] for key in ("index", "subject", "dimension", "group")},
            "expected_count": len(truth),
            "selected_count": len(selected),
            "false_positive_ids": sorted(selected - truth),
            "false_negative_ids": sorted(truth - selected),
            "exact": selected == truth,
            "strict_separation": row["strict_separation"],
        }
    )
require(len(per_query) == 222, "diagnostic query coverage")
require(
    sum(row["exact"] for row in per_query) == audit["child"]["aggregate"]["exact_count"],
    "diagnostic exact count",
)
errors = Counter(
    len(row["false_positive_ids"]) + len(row["false_negative_ids"]) for row in per_query
)
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
    "architecture": "plm-frozen-symmetric-bilinear-residual-v1",
    "objective": "plm-bilinear-worst-boundary-softplus-v1",
    "evaluator": "plm-bilinear-worst-boundary-screen-v1",
    "parent_checkpoint_sha256": "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1",
    "child_checkpoint_sha256": training["checkpoint_sha256"],
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "auditor_sha256": audit["script_sha256"],
    "execution_receipt_sha256": sha(run / "execution-receipt.json"),
    "decision_script_sha256": sha(__file__),
    "runner_sha256": summary["script_sha256"],
    "scorer_sha256": summary["scorer_sha256"],
    "loss_helper_sha256": summary["loss_helper_sha256"],
    "gate": audit["gate"],
    "basis": "Independent saved-evidence audit of split, labels, selected sets, metrics, "
    "zero-start replay records, tensor lineage, optimizer and declared gate. "
    "This does not independently repeat neural training or regenerate CUDA logits.",
    "outcome": "Single-seed screen passes; separately declared replication may follow."
    if passed
    else "Fixed quality screen fails; preserve the derivative and negative evidence.",
}
history_path = destination.with_name(destination.stem + "-training.json")
require(not history_path.exists(), "immutable training trace")
write(
    history_path,
    {"source_training_sha256": sha(run / "training.json"), "history": training["history"]},
)
portable = {
    "date": "2026-09-25",
    "campaign_version": summary["campaign_version"],
    "status": "single-seed screen passed" if passed else "rejected by fixed quality gate",
    "architecture": decision["architecture"],
    "objective": decision["objective"],
    "evaluator": decision["evaluator"],
    "inference_policy": (
        "dense-positive-bilinear-residual-subject-excluded-ascending-fp32-b8-last6-v1"
    ),
    "plan_sha256": summary["plan_sha256"],
    "runner_sha256": summary["script_sha256"],
    "scorer_sha256": summary["scorer_sha256"],
    "loss_helper_sha256": summary["loss_helper_sha256"],
    "recipe_sha256": summary["recipe_sha256"],
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "decision": decision,
    **{key: audit[key] for key in ("parent", "child", "comparisons", "checkpoint")},
    "training_history_artifact": {"path": history_path.name, "sha256": sha(history_path)},
    "mean_bce_diagnostic": training["mean_bce_diagnostic"],
    "descriptive_posthoc_diagnostics": {
        "error_count_histogram": {str(k): errors[k] for k in sorted(errors)},
        "strictly_separated_nonexact": sum(
            row["strict_separation"] and not row["exact"] for row in per_query
        ),
        "per_query": per_query,
        "used_for_selection_or_updates": False,
    },
    "initial_pre_update_loss": training["initial_pre_update_loss"],
    "final_post_update_loss": training["final_post_update_loss"],
    "zero_replay": read(run / "zero-replay.json"),
    "child_training_identity": summary["child_training_identity"],
    "environment": summary["environment"],
    "numerical_settings": summary["numerical_settings"],
    "peak_memory": summary["peak_memory"],
    "wall_seconds": summary["wall_seconds"],
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
            "decision_sha256": sha(run / "decision.json"),
            "portable_sha256": sha(destination),
        }
    )
)
