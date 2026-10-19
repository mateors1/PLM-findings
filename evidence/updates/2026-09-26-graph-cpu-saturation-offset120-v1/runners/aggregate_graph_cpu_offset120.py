"""Aggregate the CPU graph saturation receipt and render portable charts."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent
RAW = RUN_DIR / "graph-arm-offset120.json"
OUT_JSON = ROOT / "docs/experiments/2026-09-26-graph-cpu-saturation-offset120.json"
OUT_CSV = RUN_DIR / "graph-arm-offset120.csv"
FIGURES = ROOT / "docs/experiments"
OFFSET_WATTS = 120.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def svg_chart(
    path: Path,
    title: str,
    subtitle: str,
    y_label: str,
    series: list[dict[str, Any]],
    y_min: float,
    y_max: float,
    y_ticks: list[float],
) -> None:
    width, height = 980, 620
    left, right, top, bottom = 88, 38, 91, 170
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = 0.0, 7.0
    x_ticks = [(1, 0), (4, 2), (8, 3), (16, 4), (32, 5), (64, 6), (128, 7)]

    def xpos(concurrency: int) -> float:
        return left + (math.log2(concurrency) - x_min) / (x_max - x_min) * plot_width

    def ypos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{html.escape(title)}</title>",
        f"<desc id=\"desc\">{html.escape(subtitle)} Client concurrency uses a base-two logarithmic x axis.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Segoe UI,Arial,sans-serif" font-size="22" font-weight="650" fill="#152337">{html.escape(title)}</text>',
        f'<text x="{left}" y="59" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#526176">{html.escape(subtitle)}</text>',
    ]
    for tick in y_ticks:
        y = ypos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dce3ec" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#526176">{tick:g}</text>')
    for label, exponent in x_ticks:
        x = left + (exponent - x_min) / (x_max - x_min) * plot_width
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_height}" stroke="#eef1f5" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_height+22}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#526176">{label}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#738198" stroke-width="1.2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#738198" stroke-width="1.2"/>')
    parts.append(f'<text x="{left+plot_width/2:.1f}" y="{height-42}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#334155">Client concurrency (log₂ scale)</text>')
    parts.append(f'<text x="23" y="{top+plot_height/2:.1f}" transform="rotate(-90 23 {top+plot_height/2:.1f})" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="#334155">{html.escape(y_label)}</text>')

    for item in series:
        points: list[str] = []
        for row in item["rows"]:
            x = xpos(int(row["client_concurrency"]))
            value = float(row[item["value"]])
            y = ypos(value)
            points.append(f"{x:.2f},{y:.2f}")
            if item.get("low") and item.get("high"):
                low = ypos(float(row[item["low"]]))
                high = ypos(float(row[item["high"]]))
                parts.append(f'<line x1="{x:.2f}" y1="{high:.2f}" x2="{x:.2f}" y2="{low:.2f}" stroke="{item["color"]}" stroke-opacity="0.38" stroke-width="2"/>')
                parts.append(f'<line x1="{x-4:.2f}" y1="{high:.2f}" x2="{x+4:.2f}" y2="{high:.2f}" stroke="{item["color"]}" stroke-opacity="0.55"/>')
                parts.append(f'<line x1="{x-4:.2f}" y1="{low:.2f}" x2="{x+4:.2f}" y2="{low:.2f}" stroke="{item["color"]}" stroke-opacity="0.55"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{item["color"]}" stroke-width="2.7" stroke-linejoin="round" stroke-linecap="round"/>')
        for row in item["rows"]:
            x = xpos(int(row["client_concurrency"]))
            y = ypos(float(row[item["value"]]))
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#ffffff" stroke="{item["color"]}" stroke-width="2.2"/>')

    legend_y = height - 84
    total_width = sum(item["legend_width"] for item in series)
    legend_x = (width - total_width) / 2
    for item in series:
        parts.append(f'<line x1="{legend_x:.1f}" y1="{legend_y}" x2="{legend_x+27:.1f}" y2="{legend_y}" stroke="{item["color"]}" stroke-width="2.7"/>')
        parts.append(f'<circle cx="{legend_x+13.5:.1f}" cy="{legend_y}" r="3.4" fill="#ffffff" stroke="{item["color"]}" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x+36:.1f}" y="{legend_y+4}" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#334155">{html.escape(item["label"])}</text>')
        legend_x += item["legend_width"]
    parts.append(f'<text x="50%" y="{height-14}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="10" fill="#78869a">Power whiskers show observed sample ranges. Offset-adjusted watts = measured RAPL + 120 W.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    expected_levels = [1, 4, 8, 16, 32, 48, 56, 64, 72, 80, 96, 112, 128]
    if raw["status"] != "complete":
        raise ValueError(f"raw run status is {raw['status']!r}")
    if raw["cpu_power_offset_watts"] != OFFSET_WATTS or raw["power_target_watts"] is not None:
        raise ValueError("run does not match the requested offset-only design")
    if [row["client_concurrency"] for row in raw["levels"]] != expected_levels:
        raise ValueError("run is missing, duplicating, or reordering a concurrency level")

    rows: list[dict[str, Any]] = []
    for row in raw["levels"]:
        if row["failed_responses"] != 0 or not row["all_successes_returned_16_ids"]:
            raise ValueError(f"response correctness failed at concurrency {row['client_concurrency']}")
        if row["completed_servings"] != row["completed_servings_per_second"] * row["timed_seconds"]:
            raise ValueError(f"serving rate arithmetic mismatch at concurrency {row['client_concurrency']}")
        if not row["cpu_package_power_samples"]:
            raise ValueError(f"missing RAPL samples at concurrency {row['client_concurrency']}")
        raw_median = float(row["cpu_package_power_median_watts"])
        adjusted_median = float(row["offset_adjusted_power_median_watts"])
        if not math.isclose(adjusted_median - raw_median, OFFSET_WATTS, abs_tol=1e-9):
            raise ValueError(f"offset arithmetic mismatch at concurrency {row['client_concurrency']}")
        if len(row["cpu_package_power_samples"]) != row["cpu_package_power_sample_count"]:
            raise ValueError(f"RAPL sample count mismatch at concurrency {row['client_concurrency']}")
        rows.append(
            {
                "client_concurrency": row["client_concurrency"],
                "completed_servings": row["completed_servings"],
                "timed_seconds": row["timed_seconds"],
                "servings_per_second": row["completed_servings_per_second"],
                "failed_responses": row["failed_responses"],
                "median_latency_ms": row["latency_seconds_median"] * 1000,
                "p95_latency_ms": row["latency_seconds_p95"] * 1000,
                "power_samples": row["cpu_package_power_sample_count"],
                "cpu_package_watts_median": raw_median,
                "cpu_package_watts_min": row["cpu_package_power_min_watts"],
                "cpu_package_watts_max": row["cpu_package_power_max_watts"],
                "offset_adjusted_watts_median": adjusted_median,
                "offset_adjusted_watts_min": row["offset_adjusted_power_min_watts"],
                "offset_adjusted_watts_max": row["offset_adjusted_power_max_watts"],
                "servings_per_cpu_package_watt": row["servings_per_raw_cpu_package_watt"],
                "servings_per_offset_adjusted_watt": row["servings_per_offset_adjusted_watt"],
            }
        )

    plateau = [row for row in rows if row["client_concurrency"] >= 8]
    peak = max(rows, key=lambda row: row["servings_per_second"])
    max_concurrent = max(plateau, key=lambda row: row["servings_per_second"])
    summary = {
        "experiment_id": "graph-cpu-saturation-offset120-20260926",
        "status": "complete",
        "question": "SQLite graph HTTP saturation with measured CPU package power and an additive 120 W reporting offset",
        "workload": {
            "path": "loopback Uvicorn HTTP to deterministic SQLite GraphOracle",
            "response_cap_ids": 16,
            "queries_per_level": len(raw["query_prompts"]),
            "seconds_per_level": raw["level_duration_seconds"],
            "concurrency_levels": expected_levels,
            "successful_outputs_match_frozen_graph_reference": True,
            "failed_responses": sum(row["failed_responses"] for row in rows),
        },
        "power": {
            "source": raw["cpu_power_source"],
            "offset_watts": OFFSET_WATTS,
            "offset_formula": "offset_adjusted_watts = measured_cpu_package_watts + 120",
            "offset_is_direct_wall_power_measurement": False,
            "power_target_watts": None,
        },
        "saturation_summary": {
            "fastest_point_concurrency": peak["client_concurrency"],
            "fastest_point_servings_per_second": peak["servings_per_second"],
            "concurrent_plateau_scope": "tested concurrency >= 8; qualitative plateau, not a CPU utilization/OOM boundary",
            "best_tested_concurrent_point": max_concurrent["client_concurrency"],
            "best_tested_concurrent_servings_per_second": max_concurrent["servings_per_second"],
            "concurrent_plateau_mean_servings_per_second": sum(row["servings_per_second"] for row in plateau) / len(plateau),
            "concurrent_plateau_min_servings_per_second": min(row["servings_per_second"] for row in plateau),
            "concurrent_plateau_max_servings_per_second": max(row["servings_per_second"] for row in plateau),
        },
        "source_sha256": {
            "raw_run": sha(RAW),
            "runner": raw["runner_sha256"],
            "serving_receipt": raw["serving_receipt_sha256"],
            "graph_reference": raw["graph_reference_sha256"],
            "graph_snapshot": raw["graph_snapshot_sha256"],
            "plan": raw["plan_sha256"],
            "aggregator": sha(Path(__file__)),
        },
        "levels": rows,
        "interpretation_limits": [
            "The +120 W series is an arithmetic offset to CPU-package RAPL; it is not measured wall power or a validated system idle baseline.",
            "This is a local loopback graph-oracle HTTP benchmark with an output cap, not a production capacity or latency SLO.",
            "Observed throughput and package-power plateaus describe the tested concurrency range; they do not establish a hardware maximum or thermal/power limit.",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    FIGURES.mkdir(parents=True, exist_ok=True)
    svg_chart(
        FIGURES / "2026-09-26-graph-cpu-offset120-throughput.svg",
        "SQLite graph HTTP throughput",
        "Exact 16-ID capped responses; all 13 levels returned zero failures.",
        "Completed 16-ID servings per second",
        [{"label": "Completed servings/s", "color": "#2457a7", "rows": rows, "value": "servings_per_second", "legend_width": 190}],
        0,
        1100,
        [0, 200, 400, 600, 800, 1000],
    )
    svg_chart(
        FIGURES / "2026-09-26-graph-cpu-offset120-power.svg",
        "Measured CPU package power and +120 W series",
        "RAPL package measurements preserved; offset-adjusted values are derived, not wall power.",
        "CPU package power (watts)",
        [
            {"label": "Measured RAPL median", "color": "#2457a7", "rows": rows, "value": "cpu_package_watts_median", "low": "cpu_package_watts_min", "high": "cpu_package_watts_max", "legend_width": 230},
            {"label": "Measured + 120 W", "color": "#c36a14", "rows": rows, "value": "offset_adjusted_watts_median", "low": "offset_adjusted_watts_min", "high": "offset_adjusted_watts_max", "legend_width": 220},
        ],
        40,
        250,
        [40, 80, 120, 160, 200, 240],
    )
    svg_chart(
        FIGURES / "2026-09-26-graph-cpu-offset120-efficiency.svg",
        "Graph servings per watt",
        "Throughput divided by median measured package watts or the median after adding 120 W.",
        "Completed servings per watt",
        [
            {"label": "Using measured RAPL", "color": "#2457a7", "rows": rows, "value": "servings_per_cpu_package_watt", "legend_width": 230},
            {"label": "Using RAPL + 120 W", "color": "#c36a14", "rows": rows, "value": "servings_per_offset_adjusted_watt", "legend_width": 230},
        ],
        0,
        14,
        [0, 2, 4, 6, 8, 10, 12, 14],
    )
    print(json.dumps({"summary": summary["saturation_summary"], "levels": rows}, indent=2))


if __name__ == "__main__":
    main()
