"""Record owner rejection of the completed, independently audited fixed screen."""
import hashlib
import json
from pathlib import Path

root = Path.cwd()
directory = root / "runs/learning/symmetric-margin-screen-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(path.name, sha(path))


summary_path = directory / "summary.json"
audit_path = directory / "independent-audit.json"
assert sha(summary_path) == "39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701"
assert sha(audit_path) == "17afc9ca0fedf184ac9e9f2794947306883d4288860b22124cd9eb0c7751bcd9"
summary = json.loads(summary_path.read_text())
audit = json.loads(audit_path.read_text())
assert summary["complete"] is True and audit["audit_passed"] is True
assert audit["all_recomputed_outputs_equal"] is True
assert audit["summary_sha256"] == sha(summary_path)
assert audit["gate"] == summary["gate"]
assert summary["gate"]["numerical_gates_passed"] is False
for name, digest in summary["report_sha256"].items():
    assert sha(directory / name) == digest, name
decision = {
    "date": "2026-09-25",
    "experiment": "plm-symmetric-margin-stage1-v1",
    "evidence_accepted_as_completed_negative_result": True,
    "variant_accepted": False,
    "replication_authorized": False,
    "inference_default_changed": False,
    "protected_test_used": False,
    "decision": "Reject fixed m=1.0 lambda=0.1 variant; stop before seeds 1730/1731, retain both checkpoints and all failed gates.",
    "summary_sha256": sha(summary_path),
    "audit_sha256": sha(audit_path),
    "decision_script_sha256": sha(Path(__file__)),
    "plan_sha256": summary["plan_sha256"],
    "checks": summary["gate"]["checks"],
    "gained_exact_count": summary["gate"]["gained_exact_count"],
    "lost_exact_count": summary["gate"]["lost_exact_count"],
    "limitations": audit["limitations"],
}
write(directory / "decision.json", decision)
portable = {
    "date": "2026-09-25",
    "status": "completed and independently audited; fixed variant rejected",
    "decision": decision,
    "decision_sha256": sha(directory / "decision.json"),
    "seed": 1729,
    "distinct_validation_queries": 222,
    "query_arm_observations": 444,
    "protected_test_queries": 191,
    "source_paths": 1776,
    "candidate_slots": 4440,
    "treatment": {"symmetric_margin": 1.0, "symmetric_margin_loss_weight": 0.1},
    "historical_exact_context": 191,
    "fresh_control_used_for_gate": True,
    "arms": audit["arms"],
    "training": [{key: arm[key] for key in ("arm", "run_name", "config_hash", "checkpoint_hash", "global_step", "training_wall_seconds", "training_history")} for arm in summary["training"]],
    "environment": summary["environment"],
    "source_commit": summary["source_commit"],
    "snapshot_sha256": summary["snapshot_sha256"],
    "raw_reports_sha256": {name: digest for name, digest in summary["report_sha256"].items() if name.startswith("evaluation-")},
    "local_evidence_directory": "runs/learning/symmetric-margin-screen-v1",
    "model_inventory_sha256": sha(root / "docs/experiments/2026-09-25-margin-model-version-inventory.json"),
    "verification": {
        "model_cpu_suite_passed": 818, "model_cpu_suite_skipped": 9,
        "runner_tests_passed": 25, "independent_auditor_synthetic_tests_passed": 25,
        "archived_source_exact_cpu_tensor_comparisons": 92,
        "synthetic_cuda_bf16_steps": 3,
        "ruff_and_format_passed": True, "strict_mypy_source_files": 62,
        "cpu_original_stdout_archived": False,
    },
    "interpretation": "This fixed objective setting failed at seed 1729. It does not establish that every margin, curriculum or architecture fails; no alternative is tested here.",
}
write(root / "docs/experiments/2026-09-25-symmetric-margin-screen.json", portable)
