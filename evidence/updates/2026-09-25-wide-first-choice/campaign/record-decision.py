"""Bind the reviewed completed campaign and independent audit to a separate decision."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
RUN = ROOT / "runs/learning/wide-first-choice-v1"

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
assert sha(RUN / "summary.json") == args.summary_sha256
assert sha(RUN / "independent-audit.json") == args.audit_sha256
assert sha(RUN / "independent-audit.py") == "b5b7854ed823b86b93747e120147e0b508b015bc56137ffa60602e22743459e4"
primary = json.loads((RUN / "summary.json").read_text())
audit = json.loads((RUN / "independent-audit.json").read_text())
assert primary["complete"] and primary["final_identity_check"] and not primary["acceptance"]
assert audit["complete"] and audit["audit_passed"] and not audit["acceptance"]
assert audit["summary_sha256"] == args.summary_sha256
assert audit["plan_sha256"] == primary["plan_sha256"] == "02faaedcf88c94c65e70228488431a85e6687e34ea6d6f530f90ec51f43e1461"
for key in ("overall", "groups", "gate", "accounting"):
    assert audit[key] == primary[key], key
for name, digest in audit["input_sha256"].items():
    assert sha(Path(name)) == digest, name
passed = primary["gate"]["quality_passed"]
decision = {
    "date": "2026-09-25",
    "campaign_version": primary["campaign_version"],
    "evidence_accepted": True,
    "fixed_quality_gate_passed": passed,
    "eligible_for_serving_integration_consideration": passed,
    "policy_promoted": False,
    "default_changed": False,
    "new_checkpoint_created": False,
    "protected_test_used": False,
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "auditor_sha256": audit["script_sha256"],
    "decision_script_sha256": sha(Path(__file__)),
    "gate": primary["gate"],
    "basis": "Independent reconstruction of all three seeds, raw path semantics, 36-slot pools, scores, selections, metrics, work counts and fixed gate agrees with the completed campaign.",
    "scope": "Offline validation quality/work evidence only. No serving, energy, protected-test or neural-independent-replay claim.",
}
write(RUN / "decision.json", decision)
seeds = []
for item in primary["reports"]:
    path = Path(item["path"])
    assert sha(path) == item["sha256"]
    report = json.loads(path.read_text())
    assert report["complete"] and report["baseline_replay_exact"] and not report["cleanup_errors"]
    seeds.append({
        "report_sha256": item["sha256"],
        **{key: report[key] for key in ("seed", "checkpoint_hash", "training_identity", "corpus_identity", "split_hash", "numerical_settings", "overall", "groups", "baseline_receipt", "head_seconds", "set_scoring_seconds", "wall_seconds", "peak_gpu_allocated_bytes")},
    })
portable = {
    "date": "2026-09-25",
    "campaign_version": primary["campaign_version"],
    "status": "offline quality gate passed" if passed else "rejected by fixed quality gate",
    "plan_sha256": primary["plan_sha256"],
    "script_sha256": primary["script_sha256"],
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "decision_sha256": sha(RUN / "decision.json"),
    "inference_policy_suffix": "+first8-pair28-symmetric-set-logit-sum-v1",
    "candidate_order": "legacy 10, originals 5-8, then remaining 22 lexicographic pairs",
    "environment": primary["environment"],
    "snapshot_sha256": primary["snapshot_sha256"],
    "input_sha256": primary["input_sha256"],
    "overall": primary["overall"],
    "groups": primary["groups"],
    "per_seed": seeds,
    "changed_queries": primary["changed_queries"],
    "work": audit["work"],
    "accounting": primary["accounting"],
    "decision": decision,
    "limitations": audit["limitations"] + ["Per-rank timings are sequential observations with initialization and order effects, not a randomized or repeated performance benchmark.", "Checkpoint payloads and large raw neural reports remain locally hash-bound; no durable public model archive is established."],
}
destination = ROOT / "docs/experiments/2026-09-25-wide-first-choice.json"
write(destination, portable)
print(json.dumps({"quality_passed": passed, "decision_sha256": sha(RUN / "decision.json"), "portable_sha256": sha(destination)}))
