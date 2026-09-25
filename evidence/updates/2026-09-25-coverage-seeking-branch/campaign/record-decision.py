"""Accept independent evidence separately from the declared quality gate."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path.cwd()
run = root / "runs/learning/coverage-seeking-branch-v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
assert sha(run / "summary.json") == args.summary_sha256
assert sha(run / "independent-audit.json") == args.audit_sha256
summary = json.loads((run / "summary.json").read_text())
audit = json.loads((run / "independent-audit.json").read_text())
assert summary["complete"] and summary["final_identity_check"]
assert audit["complete"] and audit["audit_passed"]
assert not summary["acceptance"] and not audit["acceptance"]
assert audit["summary_sha256"] == args.summary_sha256
assert audit["plan_sha256"] == summary["plan_sha256"] == "4d718ca6b19d77d8387ef7b90089e2692058bfc99577ac9affcafe765e5a1551"
assert summary["script_sha256"] == "ff41f533bd20e004391ed0337c1d7e8a5ecf24b159331fe1db27945d06609e7f"
assert audit["script_sha256"] == sha(run / "independent-audit.py")
for key in ("overall", "groups", "gate", "accounting"):
    assert summary[key] == audit[key], key
for path, digest in audit["input_sha256"].items():
    assert sha(Path(path)) == digest, path
reports = []
rows = []
for descriptor in summary["reports"]:
    path = Path(descriptor["path"])
    assert sha(path) == descriptor["sha256"]
    report = json.loads(path.read_text())
    assert report["complete"] and report["replay_exact"]
    reports.append({
        **{k: report[k] for k in ("seed", "overall", "groups", "generation_seconds", "head_seconds", "set_scoring_seconds", "peak_gpu_allocated_bytes", "wall_seconds")},
        "report_sha256": descriptor["sha256"],
    })
    for row in report["responses"]:
        rows.append({
            "seed": report["seed"],
            **{k: row[k] for k in ("index", "subject", "dimension", "group", "anchor", "anchor_correct", "metrics", "gained_exact", "lost_exact")},
            "selected_slot": row["composition"]["selected_slot"],
            "baseline_selected_slot": row["baseline"]["selected_slot"],
            "candidate_count": len(row["composition"]["slots"]),
            "error_counts": {k: len(row[k]) for k in ("newly_covered_true_ids", "newly_covered_false_ids", "false_positive_ids", "false_negative_ids")},
            "availability": {arm: {k: v for k, v in values.items() if k != "source_union_ids"} for arm, values in row["availability"].items()},
        })
passed = summary["gate"]["quality_passed"]
decision = {
    "date": "2026-09-25", "campaign_version": summary["campaign_version"],
    "evidence_accepted": True, "fixed_quality_gate_passed": passed,
    "policy_promoted": False, "default_changed": False,
    "training_executed": False, "new_checkpoint_created": False,
    "protected_test_used": False, "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256, "auditor_sha256": audit["script_sha256"],
    "decision_script_sha256": sha(Path(__file__)), "gate": summary["gate"],
    "basis": "Independent reconstruction verifies anchors, controls, head equality, raw paths, candidates, canonical scores, metrics, work counts, identities and fixed quality gate.",
    "outcome": "Quality screen passed; no automatic serving promotion." if passed else "Reject this prediction variant under the fixed quality gate; retain complete evidence and diagnosis.",
}
destination = root / "docs/experiments/2026-09-25-coverage-seeking-branch.json"
assert not destination.exists() and not (run / "decision.json").exists()
write(run / "decision.json", decision)
write(destination, {
    "date": "2026-09-25", "campaign_version": summary["campaign_version"],
    "status": "fixed quality gate passed" if passed else "rejected by fixed quality gate",
    "inference_policy_suffix": "first8-pair28-plus-positive-uncovered-anchor9-unions8-v1",
    "plan_sha256": summary["plan_sha256"], "script_sha256": summary["script_sha256"],
    "summary_sha256": args.summary_sha256, "audit_sha256": args.audit_sha256,
    "decision_sha256": sha(run / "decision.json"), "decision": decision,
    "input_sha256": summary["input_sha256"], "snapshot_sha256": summary["snapshot_sha256"],
    **{k: summary[k] for k in ("overall", "groups", "gate", "accounting")},
    "per_seed": reports, "per_query_diagnostics": rows,
    "limitations": summary["limitations"] + [
        "set_scoring_seconds includes postprocessing, scoring and per-query diagnostic analysis; it is not isolated scorer latency.",
        "Large raw reports and weight payloads remain local hash-bound dependencies. This is not a durable checkpoint archive or self-contained neural reproduction bundle.",
        "Historical width-eight paths are reused; fresh controls and new paths are measured at fixed B8/last6 shapes. No protected-test use or deployment-quality claim.",
    ],
})
print(json.dumps({"quality_passed": passed, "decision_sha256": sha(run / "decision.json"), "portable_sha256": sha(destination)}))
