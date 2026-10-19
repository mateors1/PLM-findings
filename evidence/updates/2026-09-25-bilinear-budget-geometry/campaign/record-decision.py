"""Accept the completed paired-budget diagnostic and preserve all reduced rows."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write(path, content):
    require(len(content) <= 512 * 1024, "bounded publication artifact")
    with Path(path).open("xb") as stream:
        stream.write(content)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
root = Path.cwd().resolve()
run = root / "runs/learning/bilinear-budget-geometry-v1"
out = root / "docs/experiments/2026-09-25-bilinear-budget-geometry.json"
require(not out.exists() and not (run / "decision.json").exists(), "immutable acceptance")
require(sha(run / "summary.json") == args.summary_sha256, "summary identity")
require(sha(run / "independent-audit.json") == args.audit_sha256, "audit identity")
s, a = read(run / "summary.json"), read(run / "independent-audit.json")
require(
    s["complete"] is True and a["complete"] is True and a["audit_passed"] is True,
    "completed independent verification",
)
require(a["summary_sha256"] == args.summary_sha256, "correct audited result")
require(a["script_sha256"] == sha(run / "independent-audit.py"), "executed auditor identity")
require(
    s["plan_sha256"]
    == a["plan_sha256"]
    == "9dc2186cb9cf19ee6972c78d136e0680f05ca4ce4f85fc7a8f407217500477e2",
    "frozen plan",
)
for name, result_sha, output in (
    ("execution-receipt.json", args.summary_sha256, "primary-stdout.txt"),
    ("audit-execution-receipt.json", args.audit_sha256, "audit-stdout.txt"),
):
    receipt = read(run / name)
    field = "audit_sha256" if name.startswith("audit-") else "summary_sha256"
    require(
        type(receipt["exit_code"]) is int
        and receipt["exit_code"] == 0
        and receipt["terminal_completion_observed_by_primary"] is True,
        "observed terminal completion",
    )
    require(
        receipt[field] == result_sha and receipt["stdout_sha256"] == sha(run / output),
        "completion artifact binding",
    )
    if name.startswith("audit-"):
        require(receipt["summary_sha256"] == args.summary_sha256, "audited summary receipt")
for path, digest in a["input_sha256"].items():
    require(sha(path) == digest, "audited input drift: " + path)
for key in ("budgets", "transitions"):
    require(s[key] == a[key], "independently reconstructed " + key)
require(all(value is True for value in s["checks"].values()), "primary invariants")
require([entry["seed"] for entry in s["per_seed"]] == [1729, 1730, 1731], "seed coverage")
copies, details = [], []
for entry in s["per_seed"]:
    seed = entry["seed"]
    require(set(entry["budgets"]) == {"2000", "8000"}, "budget coverage")
    detail = {
        "seed": seed,
        "role": "historical development-selected" if seed == 1729 else "fresh budget replication",
        "budgets": {},
    }
    for role, source_name, exported_name, record in [
        *[
            (
                str(b),
                f"seed-{seed}-budget-{b}.json",
                f"budget-{b}-seed-{seed}",
                entry["budgets"][str(b)],
            )
            for b in (2000, 8000)
        ],
        ("paired", f"seed-{seed}-paired.json", f"paired-seed-{seed}", entry["paired"]),
    ]:
        source = run / source_name
        require(Path(record["path"]).resolve() == source, "reduced artifact location")
        require(
            sha(source) == record["sha256"] == a["input_sha256"][str(source)],
            "audited reduced artifact",
        )
        report = read(source)
        require(
            report["complete"] is True
            and report["seed"] == seed
            and report["query_count"] == 222
            and len(report["rows"]) == 222,
            "full query coverage",
        )
        require([row["index"] for row in report["rows"]] == list(range(222)), "query order")
        require(
            report["aggregate"] == record["aggregate"] and report["groups"] == record["groups"],
            "summary agrees with rows",
        )
        target = out.with_name(out.stem + "-" + exported_name + ".json")
        content = source.read_bytes()
        require(not target.exists() and len(content) <= 512 * 1024, "immutable bounded report")
        copies.append((target, content))
        item = {
            "path": target.name,
            "sha256": sha(source),
            "aggregate": report["aggregate"],
            "groups": report["groups"],
        }
        if role == "paired":
            detail["paired"] = item
        else:
            item["checkpoint_sha256"] = record["checkpoint_sha256"]
            detail["budgets"][role] = item
    details.append(detail)
require(len(copies) == 9, "all six budget and three paired reports")
inv = (
    root
    / "docs/experiments/2026-09-25-bilinear-budget8000-replication-model-version-inventory.json"
)
require(
    sha(inv) == "96fe9238fe1ea8bee910f06ba7aa2c8d2bb2e0381730c35c29338b73c03ef709"
    and read(inv)["registered_final_count"] == 37,
    "prior inventory identity, no payload reread",
)
decision = {
    "date": "2026-09-25",
    "campaign_version": "bilinear-budget-geometry-v1",
    "evidence_accepted": True,
    "quality_gate_applicable": False,
    "posthoc_validation_diagnostic": True,
    "threshold_selected": False,
    "alternative_predictions_generated": False,
    "new_checkpoint_created": False,
    "registered_final_count_unchanged": 37,
    "prior_inventory_sha256": sha(inv),
    "protected_test_used": False,
    "policy_promoted": False,
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "auditor_sha256": a["script_sha256"],
    "decision_script_sha256": sha(__file__),
    "execution_receipt_sha256": sha(run / "execution-receipt.json"),
    "audit_execution_receipt_sha256": sha(run / "audit-execution-receipt.json"),
    "basis": "Independent saved-score reconstruction and paired-budget comparisons; "
    "no new predictions or checkpoint payload access.",
}
portable = {
    "status": "accepted descriptive diagnostic; no quality gate",
    "decision": decision,
    "decision_sha256": hashlib.sha256(encoded(decision)).hexdigest(),
    "plan_sha256": s["plan_sha256"],
    "per_seed": details,
    "budgets": a["budgets"],
    "transitions": a["transitions"],
    "input_sha256": a["input_sha256"],
    "limitations": s["limitations"],
}
require(len(encoded(portable)) <= 512 * 1024, "portable overview bound")
for target, content in copies:
    write(target, content)
write(run / "decision.json", encoded(decision))
write(out, encoded(portable))
print(
    json.dumps(
        {
            "decision_sha256": sha(run / "decision.json"),
            "portable_sha256": sha(out),
            "complete_reduced_reports": 9,
        }
    )
)
