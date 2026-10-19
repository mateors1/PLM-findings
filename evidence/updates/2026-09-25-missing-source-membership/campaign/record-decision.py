"""Record owner acceptance after independent saved-output verification."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path.cwd()
run = root / "runs/learning/missing-source-membership-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
summary_hash = "47bf9d54967009a2be6e46bedc28f7949231be9bc14e18798fd66ad776c59805"
assert sha(run / "summary.json") == summary_hash
assert sha(run / "independent-audit.json") == args.audit_sha256
summary = json.loads((run / "summary.json").read_text())
audit = json.loads((run / "independent-audit.json").read_text())
assert summary["complete"] and summary["final_identity_check"]
assert audit["complete"] and audit["audit_passed"]
assert audit["summary_sha256"] == summary_hash
assert audit["plan_sha256"] == summary["plan_sha256"]
assert not summary["acceptance"] and not audit["acceptance"]
for key in ("overall", "coverage_summary"):
    assert summary[key] == audit[key], key
for path, digest in audit["input_sha256"].items():
    assert sha(Path(path)) == digest, path
reports = []
for descriptor in summary["reports"]:
    path = Path(descriptor["path"])
    assert sha(path) == descriptor["sha256"]
    report = json.loads(path.read_text())
    assert report["complete"]
    reports.append(report)
decision = {
    "date": "2026-09-25", "campaign_version": summary["campaign_version"],
    "evidence_accepted": True, "quality_gate_applicable": False,
    "quality_gain_claimed": False, "policy_promoted": False,
    "default_changed": False, "training_executed": False,
    "new_checkpoint_created": False, "protected_test_used": False,
    "summary_sha256": summary_hash, "audit_sha256": args.audit_sha256,
    "auditor_sha256": sha(run / "independent-audit.py"),
    "decision_script_sha256": sha(Path(__file__)),
    "basis": "Independent reconstruction agrees on complete coverage, focused membership signs, ranks, false-member comparisons and aggregates.",
    "outcome": "Accept diagnostic evidence. Most omitted true members receive positive head scores and oracle-cardinality top-K ranks. Investigate candidate construction separately; no causal attribution, calibrated probability or prediction-quality gain established.",
}
destination = root / "docs/experiments/2026-09-25-missing-source-membership.json"
assert not destination.exists() and not (run / "decision.json").exists()
write(run / "decision.json", decision)
write(destination, {
    "date": "2026-09-25", "campaign_version": summary["campaign_version"],
    "status": "independently audited diagnostic; no new prediction policy",
    "plan_sha256": summary["plan_sha256"], "script_sha256": summary["script_sha256"],
    "summary_sha256": summary_hash, "audit_sha256": args.audit_sha256,
    "decision_sha256": sha(run / "decision.json"), "decision": decision,
    "input_sha256": summary["input_sha256"], "snapshot_sha256": summary["snapshot_sha256"],
    "overall": summary["overall"], "coverage_summary": summary["coverage_summary"],
    "per_seed": [{"seed": r["seed"], "overall": r["overall"], "coverage_summary": r["coverage_summary"]} for r in reports],
    "reports": summary["reports"],
    "limitations": summary["limitations"] + ["Raw upstream reports and weights remain local hash-bound dependencies. Publication does not provide a durable checkpoint archive.", "Full per-member diagnostic reports accompany the dated research evidence addition; this portable report contains aggregates."],
})
print(json.dumps({"decision_sha256": sha(run / "decision.json"), "portable_sha256": sha(destination)}))
