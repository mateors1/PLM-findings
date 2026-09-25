"""Record owner acceptance without rewriting primary or independent evidence."""
import hashlib
import json
from pathlib import Path

root = Path.cwd()
run = root / "runs/learning/margin-gradient-diagnosis-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


summary = json.loads((run / "summary.json").read_text())
audit = json.loads((run / "independent-audit.json").read_text())
assert sha(run / "summary.json") == "5a0927acf533a2a170d829a781e52e45686961e909e5a7f697ad193de3dcc607"
assert sha(run / "independent-audit.json") == "c2fe33de149f1ade6d28475b952833a304739c9fd32aafd0064b5ea080b0cbd5"
assert audit["summary_sha256"] == sha(run / "summary.json")
assert summary["complete"] and audit["complete"] and audit["audit_passed"]
assert audit["all_saved_vector_reductions_equal"]
assert audit["observations"] == len(summary["observations"]) == 15
assert audit["distinct_training_queries"] == 96 and audit["parameter_states"] == 5
assert summary["parameter_updates"] == 0 and not summary["training_executed"]
assert summary["state_aggregates"] == audit["state_aggregates"]
assert sha(run / "independent-audit.py") == audit["script_sha256"]
for path, digest in audit["input_sha256"].items():
    assert sha(Path(path)) == digest, path
for path, digest in summary["artifact_sha256"].items():
    assert sha(run / path) == digest, path
reports = []
for observation in summary["observations"]:
    report = json.loads((run / observation["report"]).read_text())
    assert report["complete"] and report["state_unchanged"]
    assert report["all_parameter_grad_fields_none"]
    assert report["state_before_sha256"] == report["state_after_sha256"]
    reports.append({k: report[k] for k in ("state", "train_slice", "groups", "radial", "margin_active_count", "margin_query_count")})
assert all(check["passed"] for state in audit["radial_checks"].values() for check in state)
decision = {
    "date": "2026-09-25",
    "diagnostic_accepted": True,
    "decision": "Accept the bounded training-only gradient measurement and saved-vector audit; no model promotion or causal optimizer claim.",
    "summary_sha256": sha(run / "summary.json"),
    "independent_audit_sha256": sha(run / "independent-audit.json"),
    "acceptance_script_sha256": sha(Path(__file__)),
    "plan_sha256": summary["plan_sha256"],
    "observations": 15,
    "distinct_training_queries": 96,
    "parameter_states": 5,
    "model_promoted": False,
    "new_checkpoints": 0,
    "protected_test_measured": False,
    "limitations": audit["limitations"],
    "primary_and_audit_preserved": True,
    "tests": {"runner_synthetic_passed": 14, "auditor_synthetic_passed": 22},
}
write_new(run / "acceptance.json", decision)
portable = {
    "date": "2026-09-25",
    "status": "accepted bounded diagnostic",
    "diagnostic_accepted": True,
    "decision": decision,
    "acceptance_sha256": sha(run / "acceptance.json"),
    "evidence_directory": "runs/learning/margin-gradient-diagnosis-v1",
    "aggregation": "Arithmetic means of defined per-batch metrics with counts; not metrics of mean vectors or ratios of mean norms.",
    "split": "train",
    "train_slices": [[0, 32], [800, 832], [1600, 1632]],
    "parameter_blocks": {"token_embedding.weight": [2049, 256], "symmetric_relation_projection.weight": [256, 256]},
    "margin": 1.0,
    "margin_weight": 0.1,
    "audit": {k: v for k, v in audit.items() if k not in ("input_sha256", "state_aggregates")},
    "observations": reports,
    **{k: summary[k] for k in ("state_aggregates", "artifact_sha256", "input_sha256", "snapshot_sha256", "initialization", "source_commit", "source_archive_sha256", "script_sha256", "plan_sha256", "corpus_identity", "split_hash", "environment", "numerical_settings", "limitations")},
}
write_new(root / "docs/experiments/2026-09-25-margin-gradient-diagnosis.json", portable)
print(json.dumps({"accepted": True, "acceptance_sha256": sha(run / "acceptance.json"), "portable_sha256": sha(root / "docs/experiments/2026-09-25-margin-gradient-diagnosis.json")}))
