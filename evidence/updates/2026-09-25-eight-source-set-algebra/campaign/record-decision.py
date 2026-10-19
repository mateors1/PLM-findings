"""Accept audited evidence separately from the fixed prediction-quality gate."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
RUN = ROOT / "runs/learning/eight-source-set-algebra-v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
summary_hash = "1fe6e9cb1980753e6c2315c6dac890148a52105061c881261da9bb73db1c387d"
assert sha(RUN / "summary.json") == summary_hash
assert sha(RUN / "independent-audit.json") == args.audit_sha256
assert sha(RUN / "independent-audit.py") == "2044a66979274b0a67943e07e692b2891af7bad664605a4b98f24f19f0825dd4"
primary = json.loads((RUN / "summary.json").read_text())
audit = json.loads((RUN / "independent-audit.json").read_text())
assert primary["complete"] and primary["final_identity_check"] and not primary["acceptance"]
assert audit["complete"] and audit["audit_passed"] and not audit["acceptance"]
assert audit["summary_sha256"] == summary_hash
assert audit["plan_sha256"] == primary["plan_sha256"] == "7fb6f28e9d432799590fd7dff9b9269ffc732ecce8c4f496ba0c1e3521c27a5f"
for key in ("overall", "groups", "gate", "accounting", "derived_bound"):
    assert primary[key] == audit[key], key
for path, digest in audit["input_sha256"].items():
    assert sha(Path(path)) == digest, path
rows = []
per_seed = []
for descriptor in primary["reports"]:
    path = Path(descriptor["path"])
    assert sha(path) == descriptor["sha256"]
    report = json.loads(path.read_text())
    assert report["complete"] and report["baseline_reproduction_exact"]
    item = {key: report[key] for key in ("seed", "overall", "groups")}
    assert item in audit["per_seed"]
    per_seed.append({**item, "raw_report_sha256": descriptor["sha256"]})
    for row in report["responses"]:
        rows.append({
            "seed": report["seed"],
            **{key: row[key] for key in ("index", "subject", "dimension", "group", "metrics", "availability", "diagnosis", "gained_exact", "lost_exact", "selection_changed")},
            "selected_slots": {arm: row[arm]["selected_slot"] for arm in ("baseline", "prediction")},
            "selected_source_ranks": {arm: row[arm]["selected_source_ranks"] for arm in ("baseline", "prediction")},
            "error_counts": {arm: {key: len(value) for key, value in row["errors"][arm].items()} for arm in ("baseline", "prediction")},
        })
passed = primary["gate"]["quality_passed"]
decision = {
    "date": "2026-09-25",
    "campaign_version": primary["campaign_version"],
    "evidence_accepted": True,
    "fixed_quality_gate_passed": passed,
    "policy_promoted": False,
    "default_changed": False,
    "training_executed": False,
    "new_checkpoint_created": False,
    "protected_test_used": False,
    "summary_sha256": summary_hash,
    "audit_sha256": args.audit_sha256,
    "auditor_sha256": audit["script_sha256"],
    "decision_script_sha256": sha(Path(__file__)),
    "gate": primary["gate"],
    "basis": "Independent bitset and rational-sum reconstruction agrees on all64 candidate slots, exhaustive pure-operation witnesses, errors, metrics, identities and fixed gate.",
    "outcome": "Quality screen passed; no default promotion." if passed else "Reject prediction variant: seed1729 macro F1 regresses despite two pooled exact gains. Retain negative result and oracle diagnostics.",
}
destination = ROOT / "docs/experiments/2026-09-25-eight-source-set-algebra.json"
assert not destination.exists() and not (RUN / "decision.json").exists()
write(RUN / "decision.json", decision)
portable = {
    "date": "2026-09-25",
    "campaign_version": primary["campaign_version"],
    "status": "quality gate passed" if passed else "rejected by fixed quality gate",
    "inference_policy_suffix": "+first8-pair28-intersection28-symmetric-set-logit-sum-v1",
    "candidate_order": "existing36, then28 lexicographic pair intersections; duplicates and empty sets retained",
    "plan_sha256": primary["plan_sha256"],
    "script_sha256": primary["script_sha256"],
    "summary_sha256": summary_hash,
    "audit_sha256": args.audit_sha256,
    "decision_sha256": sha(RUN / "decision.json"),
    "input_sha256": primary["input_sha256"],
    "snapshot_sha256": primary["snapshot_sha256"],
    **{key: primary[key] for key in ("overall", "groups", "accounting", "derived_bound")},
    "per_seed": per_seed,
    "per_query_diagnostics": rows,
    "decision": decision,
    "limitations": audit["limitations"] + ["Large raw per-seed reports and checkpoint payloads remain local hash-bound dependencies. This compact report is not an independent neural reproduction bundle or durable checkpoint archive.", "No measured serving-performance or energy result; the policy was evaluated only from saved predictions."],
}
write(destination, portable)
print(json.dumps({"quality_passed": passed, "decision_sha256": sha(RUN / "decision.json"), "portable_sha256": sha(destination), "diagnostic_rows": len(rows)}))
