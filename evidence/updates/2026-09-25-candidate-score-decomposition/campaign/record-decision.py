"""Record separate owner acceptance of the independently audited diagnostic."""
import hashlib
import json
from pathlib import Path

root = Path.cwd()
run = root / "runs/learning/candidate-score-decomposition-v1"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")

expected = {
    "summary.json": "c325c76d0137da15ca73a157b1289e1743063489397690a8b6976276e2beef2d",
    "independent-audit.json": "7d65ffca77cf4a34f991880c9c0ebcf76d47a40e57f4e4bdefd49f2dac9ab23e",
    "independent-audit.py": "5f5af47c73fd641987d4dcf3d3e146672e739c37d7a29df57f5f0062df3ac8fa",
}
assert all(sha(run / name) == digest for name, digest in expected.items())
primary = json.loads((run / "summary.json").read_text())
audit = json.loads((run / "independent-audit.json").read_text())
assert primary["complete"] and audit["complete"] and audit["audit_passed"]
assert audit["all_recomputed_outputs_equal"]
assert audit["summary_sha256"] == expected["summary.json"]
assert primary["overall"] == audit["overall"] and primary["groups"] == audit["groups"]
assert not primary["diagnostic_accepted"] and not audit["diagnostic_accepted"]
for path, digest in audit["input_sha256"].items():
    assert sha(Path(path)) == digest, path
decision = {
    "date": "2026-09-25",
    "diagnostic_version": primary["diagnostic_version"],
    "evidence_accepted": True,
    "model_or_policy_promoted": False,
    "weak_margin_gate_reopened": False,
    "additional_seed_replication_authorized": False,
    "protected_test_used": False,
    "summary_sha256": expected["summary.json"],
    "audit_sha256": expected["independent-audit.json"],
    "auditor_sha256": expected["independent-audit.py"],
    "decision_script_sha256": sha(Path(__file__)),
    "basis": "Independent reconstruction matches all 888 cells, 8880 slot scores and 1776 source paths. Both diagonals reproduce original evidence exactly.",
    "conclusion": "Neither arm misses an available exact candidate. Fixed-pool exact counts depend on the saved pool and remain unchanged under the two saved score vectors. This does not establish causal module attribution.",
    "next_question": "Improve candidate availability under an explicitly declared generation experiment; do not tune a selector that already reaches these fixed-pool exact ceilings.",
}
write(run / "decision.json", decision)
portable = {
    "date": "2026-09-25",
    "diagnostic_version": primary["diagnostic_version"],
    "status": "independently audited saved-evidence diagnostic accepted",
    "plan_sha256": primary["plan_sha256"],
    "script_sha256": primary["script_sha256"],
    "summary_sha256": expected["summary.json"],
    "audit_sha256": expected["independent-audit.json"],
    "decision_sha256": sha(run / "decision.json"),
    "input_identities": primary["input_identities"],
    "input_sha256": primary["input_sha256"],
    "overall": primary["overall"],
    "groups": primary["groups"],
    "changed_exact": primary["changed_exact"],
    "verification": {k: audit[k] for k in ("distinct_queries", "cells_checked", "slot_scores_checked", "source_paths_checked", "test_receipt_sha256", "runner_test_receipt_sha256")},
    "decision": decision,
    "limitations": audit["limitations"] + ["Raw neural evaluation reports and checkpoints remain local hash-bound artifacts; no durable backup or public release is established."],
}
destination = root / "docs/experiments/2026-09-25-candidate-score-decomposition.json"
write(destination, portable)
print(json.dumps({"decision_sha256": sha(run / "decision.json"), "portable_sha256": sha(destination), "evidence_accepted": True}))
