from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/learning/16id-serving-saturation-20260926"
DOCS = ROOT / "docs/experiments"
RAW_FILES = (
    RUN / "results.json",
    RUN / "results-large-batches.json",
    RUN / "results-near-target.json",
    RUN / "results-fp32-vram.json",
    RUN / "results-fp32-vram-extension.json",
)
SERIES = RUN / "fp32-16id-vram-series.json"
CSV = RUN / "fp32-16id-vram-series.csv"
STOP = RUN / "fp32-user-stop-amendment.json"
OUT = DOCS / "2026-09-26-16id-fp32-vram-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9)


def source_rows(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" in data:
        return [
            {
                "batch_size": row["batch_size"],
                "rate": statistics.median(rep["completed_servings_per_second"] for rep in row["repetitions"]),
                "tokens": statistics.median(rep["completed_ids_per_second"] for rep in row["repetitions"]),
                "power": statistics.median(rep["power_median_watts"] for rep in row["repetitions"]),
                "memory_gib": max(rep["device_memory_peak_bytes"] for rep in row["repetitions"]) / 1024**3,
                "agreement": float(all(rep["first_16_id_serial_parity"] for rep in row["repetitions"])),
                "repetitions": row["repetitions"],
            }
            for row in data["results"]
        ]
    return [
        {
            "batch_size": row["batch_size"],
            "rate": statistics.median(rep["completed_servings_per_second"] for rep in row["repetitions"]),
            "tokens": statistics.median(rep["completed_ids_per_second"] for rep in row["repetitions"]),
            "power": statistics.median(rep["power_median_watts"] for rep in row["repetitions"]),
            "memory_gib": row["nvml_peak_device_memory_used_bytes"] / 1024**3,
            "agreement": statistics.mean(rep["archived_fp32_first16_agreement"] for rep in row["repetitions"]),
            "repetitions": row["repetitions"],
        }
        for row in data["rows"]
        if row.get("status") == "complete"
    ]


def main() -> None:
    report = json.loads(SERIES.read_text(encoding="utf-8"))
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in RAW_FILES}
    source = [row for path in RAW_FILES for row in source_rows(path)]
    source.sort(key=lambda row: int(row["batch_size"]))
    plotted = report["completed_rows"]
    checks: list[dict[str, object]] = []
    checks.append({"name": "28 complete source rows merged", "passed": len(source) == 28 and len(plotted) == 28})
    checks.append({"name": "unique increasing batch sizes", "passed": len({row["batch_size"] for row in source}) == len(source)})
    for raw, derived in zip(source, plotted, strict=True):
        reps = raw["repetitions"]
        if any(rep.get("seconds", 0) < 5 for rep in reps):
            raise AssertionError(f"batch {raw['batch_size']} has a short timed repetition")
        batch_ok = (
            raw["batch_size"] == derived["batch_size"]
            and near(float(raw["rate"]), derived["servings_per_second"])
            and near(float(raw["tokens"]), derived["tokens_per_second"])
            and near(float(raw["power"]), derived["power_median_watts"])
            and near(float(raw["memory_gib"]), derived["device_memory_peak_gib"])
            and near(float(raw["agreement"]), derived["first16_fp32_sequence_agreement"])
            and near(float(derived["servings_per_watt"]), derived["servings_per_second"] / derived["power_median_watts"])
            and near(float(derived["tokens_per_watt"]), derived["tokens_per_second"] / derived["power_median_watts"])
            and float(raw["agreement"]) == 1.0
            and len(reps) == 3
        )
        if not batch_ok:
            raise AssertionError(f"derived row audit failed at batch {raw['batch_size']}")
    checks.append({"name": "all row rates, powers, memory, parity and per-watt math recompute", "passed": True})
    checks.append({"name": "all complete rows have three repetitions of at least five seconds", "passed": True})
    first_plateau = next(row for row in plotted if row["servings_per_second"] >= 9600 * 0.99)
    raw_peak = max(plotted, key=lambda row: row["servings_per_second"])
    checks.append({"name": "first row within one percent of decreed 9,600 servings/s", "passed": first_plateau["batch_size"] == 8192})
    checks.append({"name": "highest raw complete rate retained separately", "passed": raw_peak["batch_size"] == 45056})

    stop = json.loads(STOP.read_text(encoding="utf-8"))
    partial = report["interrupted_partial_row"]
    checks.append({"name": "65,536 two-repetition observation marked partial", "passed": stop["saturation_reached"] is False and partial["status"] == "partial_user_stopped" and partial["repetitions"] == 2})
    checks.append({"name": "no CUDA OOM saturation claim", "passed": report["cuda_oom_saturation_observed"] is False})
    checks.append({"name": "raw input hashes match aggregation record", "passed": report["source_result_sha256"] == hashes})
    with CSV.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    checks.append({"name": "CSV includes 28 complete rows and one explicit partial row", "passed": len(csv_rows) == 29 and sum(row["status"] == "complete" for row in csv_rows) == 28 and sum(row["status"] != "complete" for row in csv_rows) == 1})

    svg_names = sorted(path.name for path in DOCS.glob("2026-09-26-16id-fp32-*.svg"))
    if len(svg_names) != 6:
        raise AssertionError(f"expected six FP32 figures, found {len(svg_names)}")
    for name in svg_names:
        ElementTree.parse(DOCS / name)
    checks.append({"name": "all six SVG figures parse as XML", "passed": True, "files": svg_names})

    audit = {
        "audit_id": "16id-fp32-vram-series-audit-v1",
        "date": "2026-09-26",
        "status": "complete",
        "auditor": "primary-task independent recomputation from raw repetition rows; not a second-agent review or GPU rerun",
        "checks": checks,
        "check_count": len(checks),
        "all_passed": all(bool(check["passed"]) for check in checks),
        "recomputed_first_plateau": {
            "batch_size": first_plateau["batch_size"],
            "servings_per_second": first_plateau["servings_per_second"],
            "device_memory_peak_gib": first_plateau["device_memory_peak_gib"],
        },
        "recomputed_completed_peak": {
            "batch_size": raw_peak["batch_size"],
            "servings_per_second": raw_peak["servings_per_second"],
        },
        "saturation_status": "not observed; user stopped the FP32 run before CUDA OOM",
        "raw_result_sha256": hashes,
        "derived_series_sha256": sha(SERIES),
        "derived_csv_sha256": sha(CSV),
        "stop_amendment_sha256": sha(STOP),
    }
    OUT.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(OUT), "checks": len(checks), "all_passed": audit["all_passed"], "first_plateau_batch": first_plateau["batch_size"], "completed_peak_batch": raw_peak["batch_size"]}, indent=2))


if __name__ == "__main__":
    main()
