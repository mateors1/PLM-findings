"""Independently verify the CPU graph offset run and its derived artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent
RAW = RUN_DIR / "graph-arm-offset120.json"
RUNNER = RUN_DIR / "graph_arm_offset120.py"
AGGREGATOR = RUN_DIR / "aggregate_graph_cpu_offset120.py"
SUMMARY = ROOT / "docs/experiments/2026-09-26-graph-cpu-saturation-offset120.json"
PLAN = ROOT / "docs/experiments/2026-09-26-graph-cpu-saturation-offset120-plan.md"
REPORT = ROOT / "docs/experiments/2026-09-26-graph-cpu-saturation-offset120-report.md"
RECEIPT = ROOT / "runs/learning/native-saturation-20260925/deployments/native-a2hhcdo7/serving.json"
GRAPH_REFERENCE = ROOT / "docs/experiments/2026-09-25-native-graph-oracle.json"
PRIOR = RUN_DIR / "graph-arm-clean-v2.json"
AUDIT = ROOT / "docs/experiments/2026-09-26-graph-cpu-saturation-offset120-audit.json"
FIGURES = [
    ROOT / "docs/experiments/2026-09-26-graph-cpu-offset120-throughput.svg",
    ROOT / "docs/experiments/2026-09-26-graph-cpu-offset120-power.svg",
    ROOT / "docs/experiments/2026-09-26-graph-cpu-offset120-efficiency.svg",
]
OFFSET = 120.0
EXPECTED_LEVELS = [1, 4, 8, 16, 32, 48, 56, 64, 72, 80, 96, 112, 128]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{label}: actual={actual!r}, expected={expected!r}")


def main() -> None:
    raw: dict[str, Any] = json.loads(RAW.read_text(encoding="utf-8"))
    summary: dict[str, Any] = json.loads(SUMMARY.read_text(encoding="utf-8"))
    prior: dict[str, Any] = json.loads(PRIOR.read_text(encoding="utf-8"))
    serving_receipt: dict[str, Any] = json.loads(RECEIPT.read_text(encoding="utf-8"))
    graph_reference: dict[str, Any] = json.loads(GRAPH_REFERENCE.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            raise ValueError(f"audit failed: {name}: {detail}")

    record("run_completed", raw["status"] == "complete", str(raw["status"]))
    record("no_power_target", raw.get("power_target_watts") is None, str(raw.get("power_target_watts")))
    record("requested_offset", raw["cpu_power_offset_watts"] == OFFSET, f"{raw['cpu_power_offset_watts']} W")
    record("thirteen_levels", [r["client_concurrency"] for r in raw["levels"]] == EXPECTED_LEVELS, f"observed={len(raw['levels'])}")
    record("summary_references_raw", summary["source_sha256"]["raw_run"] == sha(RAW), "SHA256 matches raw JSON")
    record("summary_references_runner", summary["source_sha256"]["runner"] == sha(RUNNER) == raw["runner_sha256"], "runner source SHA256 matches")
    record("summary_references_aggregator", summary["source_sha256"]["aggregator"] == sha(AGGREGATOR), "aggregator source SHA256 matches")
    record("plan_identity", sha(PLAN) == raw["plan_sha256"], "frozen plan SHA256 matches run receipt")
    record("serving_receipt_identity", sha(RECEIPT) == raw["serving_receipt_sha256"], "graph-serving receipt SHA256 matches")
    record("graph_reference_identity", sha(GRAPH_REFERENCE) == raw["graph_reference_sha256"], "oracle reference SHA256 matches")
    record("snapshot_identity", serving_receipt["snapshot_sha256"] == raw["graph_snapshot_sha256"], "run uses the receipt-bound SQLite snapshot")
    record("graph_identity", serving_receipt["corpus_identity"]["graph_hash"] == raw["graph_hash"], "run uses the receipt-bound graph hash")
    record("prior_run_complete", prior["status"] == "complete" and len(prior["levels"]) == 13, "prior clean comparison run completed all levels")

    for row in raw["levels"]:
        level = int(row["client_concurrency"])
        samples = [float(sample["package_watts"]) for sample in row["cpu_package_power_samples"]]
        expected_count = int(row["completed_servings"])
        expected_rate = expected_count / float(row["timed_seconds"])
        record(f"level_{level}_quality", row["failed_responses"] == 0 and row["all_successes_returned_16_ids"], "zero failed responses; runner records all accepted target lists equal the frozen oracle")
        close(float(row["completed_servings_per_second"]), expected_rate, f"level {level} serving-rate calculation")
        close(float(row["cpu_package_power_median_watts"]), statistics.median(samples), f"level {level} raw RAPL median")
        close(float(row["cpu_package_power_min_watts"]), min(samples), f"level {level} raw RAPL minimum")
        close(float(row["cpu_package_power_max_watts"]), max(samples), f"level {level} raw RAPL maximum")
        close(float(row["offset_adjusted_power_median_watts"]), statistics.median(samples) + OFFSET, f"level {level} adjusted median")
        close(float(row["offset_adjusted_power_min_watts"]), min(samples) + OFFSET, f"level {level} adjusted minimum")
        close(float(row["offset_adjusted_power_max_watts"]), max(samples) + OFFSET, f"level {level} adjusted maximum")
        close(float(row["servings_per_raw_cpu_package_watt"]), expected_rate / statistics.median(samples), f"level {level} raw servings/W")
        close(float(row["servings_per_offset_adjusted_watt"]), expected_rate / (statistics.median(samples) + OFFSET), f"level {level} adjusted servings/W")
        record(f"level_{level}_power_samples", len(samples) == row["cpu_package_power_sample_count"] and len(samples) >= 9, f"{len(samples)} one-second samples")

    prior_by_level = {row["client_concurrency"]: row for row in prior["levels"]}
    current_by_level = {row["client_concurrency"]: row for row in raw["levels"]}
    comparisons = []
    for level in (1, 4, 8):
        previous = prior_by_level[level]
        current = current_by_level[level]
        old_rate = float(previous["completed_servings_per_second"])
        new_rate = float(current["completed_servings_per_second"])
        old_watts = float(previous["cpu_package_power_median_watts"])
        new_watts = float(current["cpu_package_power_median_watts"])
        comparisons.append(
            {
                "client_concurrency": level,
                "prior_servings_per_second": old_rate,
                "current_servings_per_second": new_rate,
                "throughput_change_percent": (new_rate / old_rate - 1) * 100,
                "prior_package_median_watts": old_watts,
                "current_package_median_watts": new_watts,
                "package_power_change_watts": new_watts - old_watts,
            }
        )

    source_hashes = {
        "raw_run": sha(RAW),
        "runner": sha(RUNNER),
        "aggregator": sha(AGGREGATOR),
        "plan": sha(PLAN),
        "serving_receipt": sha(RECEIPT),
        "graph_reference": sha(GRAPH_REFERENCE),
        "graph_snapshot": raw["graph_snapshot_sha256"],
        "portable_summary": sha(SUMMARY),
        "report": sha(REPORT),
        "figures": {path.name: sha(path) for path in FIGURES},
    }
    audit = {
        "audit_id": "graph-cpu-saturation-offset120-audit-20260926",
        "experiment_id": "graph-cpu-saturation-offset120-20260926",
        "status": "complete",
        "audit_scope": "secondary source-hash, raw-sample, offset-arithmetic, rate, quality-flag and prior-run comparison checks",
        "audit_method": "separate verifier re-read frozen raw inputs, recomputed per-level medians/ranges/rates/efficiencies and cross-bound the run to its serving receipt and frozen graph reference",
        "checks_passed": len(checks),
        "checks_failed": 0,
        "checks": checks,
        "prior_clean_run_comparison": comparisons,
        "not_independently_recorded": [
            "Individual HTTP bodies are not stored; exact target equality is enforced in the frozen runner and corroborated by zero rejected responses plus its source hash.",
            "CPU utilization, frequency, temperature and other package workloads were not recorded, so run-to-run power changes are not causally attributed.",
            "The additive 120 W offset is user-specified arithmetic and has no independent whole-system power-meter validation.",
        ],
        "source_sha256": source_hashes,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(AUDIT), "checks_passed": len(checks), "prior_comparison": comparisons}, indent=2))


if __name__ == "__main__":
    main()
