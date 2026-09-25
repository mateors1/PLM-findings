"""Bind an owner decision to a completed, independently audited weak-margin screen."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path.cwd()
run = root / "runs/learning/weak-margin-screen-v1"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


assert sha(run / "summary.json") == args.summary_sha256
assert sha(run / "independent-audit.json") == args.audit_sha256
summary = json.loads((run / "summary.json").read_text())
audit = json.loads((run / "independent-audit.json").read_text())
assert summary["complete"] is True and audit["complete"] is True
assert audit["audit_passed"] and audit["all_recomputed_outputs_equal"]
assert audit["summary_sha256"] == args.summary_sha256
assert audit["gate"] == summary["gate"]
assert audit["distinct_validation_queries"] == 222
assert audit["source_path_count"] == 1776 and audit["slot_count"] == 4440
assert audit["plan_sha256"] == summary["plan_sha256"]
for name, digest in summary["report_sha256"].items():
    assert sha(run / name) == digest, name
for name, digest in summary["snapshot_sha256"].items():
    assert sha(run / ("summary." + name)) == digest, name
for name, digest in audit["input_sha256"].items():
    assert sha(Path(name)) == digest, name
assert sha(run / "independent-audit.py") == audit["script_sha256"]
passed = summary["gate"]["numerical_gates_passed"]
assert type(passed) is bool and passed == all(summary["gate"]["checks"].values())
decision = {
    "date": "2026-09-25",
    "experiment": summary["campaign_version"],
    "evidence_accepted": True,
    "stage_one_accepted": passed,
    "replication_authorized": passed,
    "variant_accepted_across_seeds": False,
    "inference_default_changed": False,
    "protected_test_used": False,
    "decision": ("Accept stage-one screen only; execute fixed additional-seed replication before any family acceptance." if passed else "Reject fixed m=1 lambda=.001 variant at stage one; stop before seeds1730/1731 and retain both checkpoints."),
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "decision_script_sha256": sha(Path(__file__)),
    "plan_sha256": summary["plan_sha256"],
    "gate": summary["gate"],
    "limitations": audit["limitations"],
    "primary_and_audit_preserved": True,
}
write(run / "decision.json", decision)
portable = {
    "date": "2026-09-25",
    "status": "audited single-seed pass; replication required" if passed else "audited single-seed negative result; variant rejected",
    "decision": decision,
    "decision_sha256": sha(run / "decision.json"),
    "seed": 1729,
    "distinct_validation_queries": 222,
    "query_arm_observations": 444,
    "source_paths": 1776,
    "candidate_slots": 4440,
    "treatment": {"symmetric_margin": 1.0, "symmetric_margin_loss_weight": 0.001},
    "fresh_control_used_for_gate": True,
    "arms": audit["arms"],
    "training": [{k: arm[k] for k in ("arm", "run_name", "config_hash", "checkpoint_hash", "identity", "global_step", "training_wall_seconds", "training_history")} for arm in summary["training"]],
    "environment": summary["environment"],
    "source_commit": summary["source_commit"],
    "snapshot_sha256": summary["snapshot_sha256"],
    "raw_reports_sha256": summary["report_sha256"],
    "local_evidence_directory": str(run.relative_to(root)),
    "focused_verification": {"runner_tests_passed": 14, "auditor_tests_passed": 34, "stdout_and_tested_script_hashes_archived": True},
    "reused_verification": {"core_cpu_tests_passed": 818, "core_cpu_tests_skipped": 9, "cpu_default_exact_tensor_comparisons": 92, "synthetic_cuda_bf16_steps": 3, "scope": "Authenticated prior unchanged core source/config/environment; original full-suite stdout is owner-attested, not archived."},
    "interpretation": "Adaptively chosen development screen; compares a fresh matched control with a fixed weaker coefficient. No protected-test, unbiased generalization, optimizer-cause or serving-benefit claim.",
}
destination = root / "docs/experiments/2026-09-25-weak-margin-screen.json"
write(destination, portable)
print(json.dumps({"stage_one_accepted": passed, "decision_sha256": sha(run / "decision.json"), "portable_sha256": sha(destination)}))
