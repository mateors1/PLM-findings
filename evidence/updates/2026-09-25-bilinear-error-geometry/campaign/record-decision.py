"""Accept an independently audited saved-score diagnostic, without a quality gate."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(data) <= 512 * 1024, "portable size bound")
    with path.open("xb") as stream:
        stream.write(data)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
root = Path.cwd()
run = root / "runs/learning/bilinear-error-geometry-v1"
out = root / "docs/experiments/2026-09-25-bilinear-error-geometry.json"
require(not out.exists() and not (run / "decision.json").exists(), "immutable outcome")
require(sha(run / "summary.json") == args.summary_sha256, "primary identity")
require(sha(run / "independent-audit.json") == args.audit_sha256, "audit identity")
summary, audit = read(run / "summary.json"), read(run / "independent-audit.json")
require(summary["complete"] is True, "complete diagnostic")
require(audit["complete"] is True and audit["audit_passed"] is True, "completed audit")
require(audit["summary_sha256"] == args.summary_sha256, "audited result")
require(audit["script_sha256"] == sha(run / "independent-audit.py"), "frozen auditor")
require(
    summary["plan_sha256"]
    == audit["plan_sha256"]
    == "7453c260275c58c4dd8f34fe5be1b2f054251f8ddf1e9b039ec608c76d96e232",
    "frozen plan",
)
execution = read(run / "execution-receipt.json")
require(
    execution["exit_code"] == 0
    and execution["terminal_completion_observed_by_primary"] is True
    and execution["summary_sha256"] == args.summary_sha256
    and execution["stdout_sha256"] == sha(run / "primary-stdout.txt"),
    "observed primary completion",
)
for path, digest in audit["input_sha256"].items():
    require(sha(path) == digest, "audited input drift: " + path)
for key in ("fresh_two", "all_three", "overlap"):
    require(summary[key] == audit[key], "independent reduction agreement: " + key)
require([item["seed"] for item in summary["seed_artifacts"]] == [1729, 1730, 1731], "seed scope")
details = []
for item in summary["seed_artifacts"]:
    seed = item["seed"]
    path = run / f"seed-{seed}.json"
    require(Path(item["path"]).resolve() == path, "seed report location")
    require(sha(path) == item["sha256"], "seed report identity")
    require(audit["input_sha256"][str(path)] == item["sha256"], "seed report audited")
    report = read(path)
    require(report["complete"] is True and len(report["rows"]) == 222, "complete seed geometry")
    checked = [entry for entry in audit["per_seed"] if entry["seed"] == seed]
    require(len(checked) == 1, "one independent seed reduction")
    require(
        checked[0]["aggregate"] == report["aggregate"] and checked[0]["groups"] == report["groups"],
        "independent seed arithmetic",
    )
    details.append(
        {
            "seed": seed,
            "role": "historical selection" if seed == 1729 else "fresh replication",
            "report_sha256": item["sha256"],
            "aggregate": report["aggregate"],
            "groups": report["groups"],
            "nonexact_rows": [row for row in report["rows"] if not row["exact"]],
        }
    )
decision = {
    "date": "2026-09-25",
    "campaign_version": "bilinear-error-geometry-v1",
    "evidence_accepted": True,
    "quality_gate_applicable": False,
    "posthoc_validation_diagnostic": True,
    "threshold_selected": False,
    "alternative_predictions_generated": False,
    "new_checkpoint_created": False,
    "registered_final_count_unchanged": 33,
    "protected_test_used": False,
    "policy_promoted": False,
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "auditor_sha256": audit["script_sha256"],
    "execution_receipt_sha256": sha(run / "execution-receipt.json"),
    "decision_script_sha256": sha(__file__),
    "basis": "Independent reconstruction of saved-score reductions; upstream accepted "
    "audits remain authority for saved labels and scores. No neural replay.",
}
write(run / "decision.json", decision)
portable = {
    "status": "accepted descriptive diagnostic; no quality gate",
    "decision": decision,
    "decision_sha256": sha(run / "decision.json"),
    "plan_sha256": summary["plan_sha256"],
    "per_seed": details,
    "fresh_two": summary["fresh_two"],
    "all_three": summary["all_three"],
    "overlap": summary["overlap"],
    "input_sha256": audit["input_sha256"],
    "limitations": summary["limitations"],
}
write(out, portable)
print(json.dumps({"decision_sha256": sha(run / "decision.json"), "portable_sha256": sha(out)}))
