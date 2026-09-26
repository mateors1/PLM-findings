"""Bind a completed two-seed audit to a separate owner outcome decision."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    require(len(data.encode("utf-8")) <= 512 * 1024, "publication file bound")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary-sha256", required=True)
parser.add_argument("--audit-sha256", required=True)
args = parser.parse_args()
root = Path.cwd()
run = root / "runs/learning/bilinear-budget8000-replication-v1"
destination = root / "docs/experiments/2026-09-25-bilinear-budget8000-replication.json"
require(not destination.exists() and not (run / "decision.json").exists(), "immutable outcome")
for seed in (1730, 1731):
    for part in range(1, 5):
        require(
            not destination.with_name(
                destination.stem + f"-training-{seed}-part-{part:02}.json"
            ).exists(),
            "immutable training chunk",
        )
require(sha(run / "aggregate-v2.json") == args.summary_sha256, "aggregate primary identity")
require(sha(run / "independent-audit-v2.json") == args.audit_sha256, "aggregate audit identity")
summary, audit = read(run / "aggregate-v2.json"), read(run / "independent-audit-v2.json")
require(summary["complete"] is True, "complete aggregate")
require(audit["complete"] is True and audit["audit_passed"] is True, "completed aggregate audit")
require(audit["summary_sha256"] == args.summary_sha256, "audited aggregate")
require(audit["script_sha256"] == sha(run / "independent-audit-v2.py"), "frozen auditor")
require(
    summary["plan_sha256"]
    == audit["plan_sha256"]
    == "750d115b2fd0daf356cb3a633248e826c264f30533ddf8e8df2ee125d2ecdc6b",
    "frozen replication plan",
)
require(summary["gate"] == audit["gate"], "independently reconstructed gate")
repair_path = (
    root / "docs/experiments/2026-09-25-bilinear-budget8000-replication-evidence-repair.md"
)
require(
    sha(repair_path) == "130fd639ecc9b1d771f44f87e1997d1690eb6255b39f85c81f959616f84a89eb",
    "repair declaration",
)
require(audit["audit_version"] == 2, "versioned aggregate audit")
failed_receipt = run / "seed-1730/audit-execution-receipt.json"
require(
    sha(failed_receipt) == "d25163d1315aafbc98d9a2cc2bd7f1c7e9c8bdd1355ba27a5d4e3fab2b2cbbb6",
    "retained failed audit",
)
require(read(failed_receipt)["exit_code"] == 1, "retained failure status")
require(
    sha(run / "seed-1730/audit-stdout.txt")
    == "8c5773b049fc57518f2c389bbd616046d70743704c90a8eba7f9c343732cd8fb",
    "failed audit stdout",
)
require(
    sha(run / "seed-1730/summary.json")
    == "a7feed80dbab71e4f66e271a4ba5500b49a19ee068578840b5a6c32b6e8808fe",
    "same trained seed1730",
)

aggregate_execution = read(run / "aggregate-v2-execution-receipt.json")
require(
    aggregate_execution["exit_code"] == 0
    and aggregate_execution["terminal_completion_observed_by_primary"] is True
    and aggregate_execution["summary_sha256"] == args.summary_sha256
    and aggregate_execution["stdout_sha256"] == sha(run / "aggregate-v2-stdout.txt"),
    "observed aggregate terminal completion",
)
for path, digest in audit["input_sha256"].items():
    require(sha(path) == digest, "audited input drift: " + path)
require([item["seed"] for item in audit["per_seed"]] == [1730, 1731], "both ordered fresh seeds")
children = []
details = []
for item in audit["per_seed"]:
    seed = item["seed"]
    folder = run / f"seed-{seed}"
    require(sha(folder / "summary.json") == item["summary_sha256"], "per-seed primary identity")
    require(
        sha(folder / "independent-audit-v2.json") == item["audit_sha256"], "per-seed audit identity"
    )
    primary = read(folder / "summary.json")
    execution = read(folder / "execution-receipt.json")
    require(
        execution["seed"] == seed
        and execution["exit_code"] == 0
        and execution["terminal_completion_observed_by_primary"] is True
        and execution["summary_sha256"] == item["summary_sha256"]
        and execution["stdout_sha256"] == sha(folder / "primary-stdout.txt"),
        "observed primary terminal completion",
    )
    training = read(folder / "training.json")
    require(training["complete"] is True and training["completed_updates"] == 8000, "completed fit")
    for name, digest in primary["artifact_sha256"].items():
        require(sha(folder / name) == digest, "per-seed artifact drift: " + name)
    require(sha(folder / "checkpoint-final.pt") == training["checkpoint_sha256"], "child weights")
    children.append(
        {
            "seed": seed,
            "checkpoint_path": (folder / "checkpoint-final.pt").relative_to(root).as_posix(),
            "checkpoint_sha256": training["checkpoint_sha256"],
            "sidecar_sha256": sha(folder / "checkpoint-final.pt.json"),
            "summary_sha256": item["summary_sha256"],
            "audit_sha256": item["audit_sha256"],
            "execution_receipt_sha256": sha(folder / "execution-receipt.json"),
            "quality_gate_passed": item["gate"]["primary_checks_passed"],
        }
    )
    require(len(training["history"]) == 8000, "complete 8000-update trace")
    require([r["update"] for r in training["history"]] == list(range(1, 8001)), "trace order")
    parts, reconstructed = [], []
    for part in range(1, 5):
        start, end = (part - 1) * 2000, part * 2000
        history_path = destination.with_name(
            destination.stem + f"-training-{seed}-part-{part:02}.json"
        )
        write(
            history_path,
            {
                "seed": seed,
                "source_training_sha256": sha(folder / "training.json"),
                "update_start": start + 1,
                "update_end": end,
                "history": training["history"][start:end],
            },
        )
        reconstructed.extend(read(history_path)["history"])
        parts.append(
            {
                "path": history_path.name,
                "sha256": sha(history_path),
                "update_start": start + 1,
                "update_end": end,
            }
        )
    require(reconstructed == training["history"], "exact trace concatenation")
    require(
        training["training_prefix_diagnostic"] == primary["training_prefix_diagnostic"],
        "recorded prefix agrees",
    )
    details.append(
        {
            "seed": seed,
            "training_history_artifacts": parts,
            "training_prefix_diagnostic": training["training_prefix_diagnostic"],
            "initial_pre_update_loss": training["initial_pre_update_loss"],
            "final_post_update_loss": training["final_post_update_loss"],
            "checkpoint": item["checkpoint"],
            "child": item["child"],
            "comparisons": item["comparisons"],
            "gate": item["gate"],
            "environment": primary["environment"],
            "wall_seconds": primary["wall_seconds"],
            "peak_memory": primary["peak_memory"],
        }
    )
passed = audit["gate"]["primary_checks_passed"]
require(type(passed) is bool, "Boolean campaign gate")
require(passed == all(c["quality_gate_passed"] for c in children), "fresh seed conjunction")
decision = {
    "date": "2026-09-25",
    "campaign_version": "bilinear-budget8000-replication-v1",
    "evidence_accepted": True,
    "fixed_quality_gate_passed": passed,
    "policy_promoted": False,
    "default_changed": False,
    "protected_test_used": False,
    "standard_serving_supported": False,
    "new_checkpoint_count": 2,
    "fresh_seeds": [1730, 1731],
    "reused_selection_seed": 1729,
    "architecture": "plm-frozen-symmetric-bilinear-residual-v1",
    "objective": "plm-bilinear-residual-balanced-bce-v1",
    "evaluator": "plm-bilinear-budget8000-replication-v1",
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "auditor_sha256": audit["script_sha256"],
    "decision_script_sha256": sha(__file__),
    "evidence_repair_declaration_sha256": sha(repair_path),
    "failed_seed1730_audit_receipt_sha256": sha(failed_receipt),
    "original_auditor_sha256": sha(run / "independent-audit.py"),
    "original_aggregate_executed": False,
    "neural_execution_repeated_for_evidence_repair": False,
    "aggregation_runner_sha256": summary["aggregation_runner_sha256"],
    "aggregate_execution_receipt_sha256": sha(run / "aggregate-v2-execution-receipt.json"),
    "children": children,
    "gate": audit["gate"],
    "outcome": "Both fresh seeds pass their matching fixed gates; no automatic promotion."
    if passed
    else "Replication quality gate fails; preserve both children and all per-seed evidence.",
    "basis": "Independent saved-evidence and CPU checkpoint audits, with observed terminal "
    "completion for both fresh fits. No independent replay of CUDA training.",
}
portable = {
    "date": "2026-09-25",
    "campaign_version": decision["campaign_version"],
    "status": "replication screen passed" if passed else "replication quality gate rejected",
    "plan_sha256": audit["plan_sha256"],
    "summary_sha256": args.summary_sha256,
    "audit_sha256": args.audit_sha256,
    "decision": decision,
    "per_seed": details,
    "historical1729": audit["historical1729"],
    "fresh_two": audit["fresh_two"],
    "all_three": audit["all_three"],
    "gate": audit["gate"],
    "input_sha256": audit["input_sha256"],
    "limitations": audit["limitations"],
}
write(run / "decision.json", decision)
portable["decision_sha256"] = sha(run / "decision.json")
write(destination, portable)
print(
    json.dumps(
        {
            "quality_passed": passed,
            "decision_sha256": sha(run / "decision.json"),
            "portable_sha256": sha(destination),
        }
    )
)
