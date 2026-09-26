from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "runs/learning/16id-serving-saturation-20260926"
DOC_DIR = ROOT / "docs/experiments"
SOURCES = (
    RUN_DIR / "results.json",
    RUN_DIR / "results-large-batches.json",
    RUN_DIR / "results-near-target.json",
    RUN_DIR / "results-fp32-vram.json",
    RUN_DIR / "results-fp32-vram-extension.json",
)
PLATEAU = 9600.0
PLATEAU_TOLERANCE = 0.01
TOTAL_MEMORY_GIB = 17066033152 / 1024**3
OUT_JSON = RUN_DIR / "fp32-16id-vram-series.json"
OUT_CSV = RUN_DIR / "fp32-16id-vram-series.csv"
THROUGHPUT_SVG = DOC_DIR / "2026-09-26-16id-fp32-throughput-efficiency.svg"
RESOURCE_SVG = DOC_DIR / "2026-09-26-16id-fp32-memory-power.svg"
REPORT = DOC_DIR / "2026-09-26-16id-fp32-vram-stopped-report.md"
STOP_NOTE = RUN_DIR / "fp32-user-stop-amendment.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_rows(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    found: list[dict[str, Any]] = []
    for row in report.get("results", []):
        repetitions = row["repetitions"]
        memory = max(
            (rep.get("device_memory_peak_bytes", 0.0) for rep in repetitions), default=0.0
        )
        found.append(
            {
                "batch_size": row["batch_size"],
                "servings_per_second": row["median_servings_per_second"],
                "tokens_per_second": row["median_ids_per_second"],
                "power_median_watts": row["median_repetition_power_watts"],
                "device_memory_peak_gib": memory / 1024**3,
                "first16_fp32_sequence_agreement": float(
                    all(rep.get("first_16_id_serial_parity", False) for rep in repetitions)
                ),
                "protocol_valid_and_cap_complete_rate": None,
                "macro_f1_at_16_first8": None,
                "repetitions": len(repetitions),
                "status": "complete",
                "source_result": str(path.relative_to(ROOT)),
                "experiment_segment": report.get("experiment_id", path.stem),
            }
        )
    return found


def saturation_rows(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    found: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        if row.get("status") != "complete":
            continue
        repetitions = row["repetitions"]
        label_metrics = [rep["last_work_unit_label_metrics"] for rep in repetitions]
        found.append(
            {
                "batch_size": row["batch_size"],
                "servings_per_second": row["median_servings_per_second"],
                "tokens_per_second": row["median_ids_per_second"],
                "power_median_watts": row["median_power_watts"],
                "device_memory_peak_gib": row["nvml_peak_device_memory_used_bytes"] / 1024**3,
                "first16_fp32_sequence_agreement": row["median_fp32_first16_agreement"],
                "protocol_valid_and_cap_complete_rate": statistics.mean(
                    rep["protocol_valid_and_cap_complete_rate"] for rep in repetitions
                ),
                "macro_f1_at_16_first8": statistics.mean(
                    metric["macro_f1_at_16"] for metric in label_metrics
                ),
                "repetitions": len(repetitions),
                "status": "complete",
                "source_result": str(path.relative_to(ROOT)),
                "experiment_segment": report.get("experiment_id", path.stem),
            }
        )
    return found


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(old_rows(path) if "results" in json.loads(path.read_text(encoding="utf-8")) else saturation_rows(path))
    rows.sort(key=lambda row: row["batch_size"])
    seen: set[int] = set()
    for row in rows:
        batch = row["batch_size"]
        if batch in seen:
            raise ValueError(f"duplicate batch size in merged inputs: {batch}")
        seen.add(batch)
        power = row["power_median_watts"]
        row["servings_per_watt"] = row["servings_per_second"] / power
        row["tokens_per_watt"] = row["tokens_per_second"] / power
        row["tokens_per_second_definition"] = "returned generated entity-ID tokens per second; EOS excluded"
    return rows


def partial_stop_row() -> dict[str, Any]:
    servings = statistics.median((3798.5, 3782.8))
    tokens = statistics.median((60775.9, 60524.6))
    power = statistics.median((79.3, 77.1))
    memory_gib = 15.79
    return {
        "batch_size": 65536,
        "servings_per_second": servings,
        "tokens_per_second": tokens,
        "power_median_watts": power,
        "device_memory_peak_gib": memory_gib,
        "first16_fp32_sequence_agreement": 1.0,
        "protocol_valid_and_cap_complete_rate": None,
        "macro_f1_at_16_first8": None,
        "repetitions": 2,
        "status": "partial_user_stopped",
        "source_result": "runner stdout before user stop; partial attempt is not in the completed JSON rows",
        "experiment_segment": "fp32-vram-extension-interrupted-at-65536",
        "servings_per_watt": servings / power,
        "tokens_per_watt": tokens / power,
        "tokens_per_second_definition": "returned generated entity-ID tokens per second; EOS excluded",
        "partial_attempt_note": "Two of three planned repetitions finished; row did not finalize and is excluded from completed-row peak/plateau statistics.",
        "memory_value_is_rounded_gib": True,
    }


def fmt(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}"


def svg_chart(
    title: str,
    description: str,
    data: list[dict[str, Any]],
    series: list[dict[str, Any]],
    *,
    y_max: float,
    y_label: str,
    include_partial: bool = True,
    plateau_line: bool = False,
    plateau_annotation: bool = False,
    total_memory_line: bool = False,
    width: int = 650,
    height: int = 370,
) -> str:
    left, right, top, bottom = 78, 22, 48, 66
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = math.log10(320), math.log10(65536)

    def px(batch: int) -> float:
        return left + (math.log10(batch) - x_min) / (x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (1 - value / y_max) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="25" font-family="Segoe UI,Arial,sans-serif" font-size="17" font-weight="600" fill="#16212b">{escape(title)}</text>',
    ]
    y_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    for fraction in y_ticks:
        value = fraction * y_max
        y = py(value)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d9e0e5" stroke-width="1"/>')
        if y_max < 30:
            label = f"{value:.1f}"
        elif y_max >= 10000:
            label = f"{value/1000:.0f}k"
        elif y_max >= 1000:
            label = f"{value:,.0f}"
        else:
            label = f"{value:.0f}"
        elements.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="11" fill="#35424c">{label}</text>')
    x_ticks = (320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 65536)
    for batch in x_ticks:
        x = px(batch)
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#e7ecef" stroke-width="1"/>')
        label = f"{batch:,}" if batch < 10000 else f"{batch/1000:g}k"
        elements.append(f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="10.5" fill="#35424c">{label}</text>')
    elements.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#34414b" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#34414b" stroke-width="1.2"/>',
            f'<text x="{left + plot_w/2:.1f}" y="{height-18}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#16212b">Batch size (responses)</text>',
            f'<text x="19" y="{top+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 19 {top+plot_h/2:.1f})" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#16212b">{escape(y_label)}</text>',
        ]
    )
    if plateau_line:
        y = py(PLATEAU)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#7a5195" stroke-width="1.8" stroke-dasharray="7 5"/>')
        elements.append(f'<text x="{width-right-2}" y="{y-7:.1f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="10.5" fill="#68427f">decreed peak 9,600 servings/s</text>')
    if total_memory_line:
        y = py(TOTAL_MEMORY_GIB)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#555f68" stroke-width="1.5" stroke-dasharray="5 4"/>')
        elements.append(f'<text x="{width-right-2}" y="{y+14:.1f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="10.5" fill="#555f68">device total {TOTAL_MEMORY_GIB:.2f} GiB</text>')

    completed = [row for row in data if row["status"] == "complete"]
    partial = [row for row in data if row["status"] != "complete"]
    for item in series:
        key, color = item["key"], item["color"]
        values = [(row["batch_size"], row[key]) for row in completed if row.get(key) is not None]
        if values:
            coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in values)
            elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
            for batch, value in values:
                label = f"{batch:,}: {value:,.2f}" if y_max < 1000 else f"{batch:,}: {value:,.0f}"
                elements.append(f'<circle cx="{px(batch):.1f}" cy="{py(value):.1f}" r="3.3" fill="{color}"><title>{escape(item["label"])}; batch {batch:,}; {value:,.3f} {escape(y_label)}</title></circle>')
        if include_partial:
            for row in partial:
                value = row.get(key)
                if value is None:
                    continue
                x, y = px(row["batch_size"]), py(value)
                elements.append(f'<path d="M{x:.1f} {y-5:.1f} L{x+5:.1f} {y:.1f} L{x:.1f} {y+5:.1f} L{x-5:.1f} {y:.1f} Z" fill="#ffffff" stroke="#d55e00" stroke-width="2"><title>Partial, two repetitions only; batch {row["batch_size"]:,}; {value:,.3f} {escape(y_label)}</title></path>')
    if plateau_annotation and completed:
        first = next((row for row in completed if row["servings_per_second"] >= PLATEAU * (1 - PLATEAU_TOLERANCE)), None)
        if first:
            annotation_value = (
                first["device_memory_peak_gib"]
                if y_label == "Device memory used (GiB)"
                else first["servings_per_second"]
            )
            x, y = px(first["batch_size"]), py(annotation_value)
            label_x, label_y = max(left + 20, x - 28), min(height - bottom - 28, y + 40)
            elements.append(f'<line x1="{x:.1f}" y1="{y+4:.1f}" x2="{label_x+72:.1f}" y2="{label_y-10:.1f}" stroke="#35424c" stroke-width="1"/>')
            elements.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-family="Segoe UI,Arial,sans-serif" font-size="10.5" fill="#16212b">first ≥99%: batch {first["batch_size"]:,}</text>')
            if y_label == "Device memory used (GiB)":
                memory = first["device_memory_peak_gib"]
                elements.append(f'<text x="{label_x:.1f}" y="{label_y+14:.1f}" font-family="Segoe UI,Arial,sans-serif" font-size="10.5" fill="#16212b">{memory:.2f} GiB device use</text>')
    if partial and include_partial:
        elements.append(f'<text x="{width-right}" y="{height-43}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="10.5" fill="#b14d00">◇ 65,536 partial (2/3 reps)</text>')
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def write_svg(path: Path, *args: Any, **kwargs: Any) -> None:
    path.write_text(svg_chart(*args, **kwargs), encoding="utf-8")


def main() -> None:
    completed = build_rows()
    partial = partial_stop_row()
    all_plot_rows = completed + [partial]
    if not completed:
        raise ValueError("no completed batch rows found")
    first_plateau = next(
        row
        for row in completed
        if row["servings_per_second"] >= PLATEAU * (1 - PLATEAU_TOLERANCE)
    )
    raw_peak = max(completed, key=lambda row: row["servings_per_second"])
    best_efficiency = max(completed, key=lambda row: row["servings_per_watt"])
    last_complete = completed[-1]
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES}
    stop_note = {
        "status": "user_stopped_fp32_before_cuda_oom",
        "date": "2026-09-26",
        "reason": "User directed stopping FP32 benchmarking and converting the collected data into graphs.",
        "last_completed_batch": 57344,
        "last_completed_result_path": "runs/learning/16id-serving-saturation-20260926/results-fp32-vram-extension.json",
        "incomplete_attempt": {
            "batch_size": 65536,
            "completed_repetitions": 2,
            "planned_repetitions": 3,
            "observed_servings_per_second": [3798.5, 3782.8],
            "observed_entity_id_tokens_per_second": [60775.9, 60524.6],
            "observed_median_board_power_watts": [79.3, 77.1],
            "observed_peak_device_memory_gib_rounded": 15.79,
            "first16_fp32_sequence_agreement": [1.0, 1.0],
            "limitations": [
                "Console observations only; the interrupted attempt did not flush a complete raw row or NVML samples.",
                "Not included in completed-run peak or saturation-boundary statistics.",
                "No CUDA out-of-memory event was observed before the user stop.",
            ],
        },
        "saturation_reached": False,
        "source_result_sha256": sources,
    }
    STOP_NOTE.write_text(json.dumps(stop_note, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    for row in completed:
        row["at_or_above_99_percent_of_decreed_peak"] = row["servings_per_second"] >= PLATEAU * (1 - PLATEAU_TOLERANCE)
    partial["at_or_above_99_percent_of_decreed_peak"] = False
    report = {
        "experiment_id": "16id-fp32-capacity-series-v1",
        "status": "fp32_benchmark_stopped_by_user_before_oom",
        "engine": "FP32",
        "gpu": "NVIDIA GeForce RTX 5070 Ti",
        "user_decreed_practical_peak_servings_per_second": PLATEAU,
        "plateau_first_point_rule": f"first completed median at least {100*(1-PLATEAU_TOLERANCE):.0f}% of decreed peak",
        "first_plateau_point": first_plateau,
        "highest_observed_completed_throughput_point": raw_peak,
        "highest_completed_servings_per_watt_point": best_efficiency,
        "last_completed_point": last_complete,
        "last_completed_batch_memory_gib": last_complete["device_memory_peak_gib"],
        "total_device_memory_gib": TOTAL_MEMORY_GIB,
        "cuda_oom_saturation_observed": False,
        "completed_rows": completed,
        "interrupted_partial_row": partial,
        "source_result_sha256": sources,
        "stop_amendment_path": str(STOP_NOTE.relative_to(ROOT)),
        "definitions": {
            "servings_per_second": "completed bounded response slots per second; EOS or exactly 16 entity IDs",
            "tokens_per_second": "returned generated entity-ID tokens per second; EOS excluded",
            "servings_per_watt": "servings per second divided by median device board power in watts",
            "tokens_per_watt": "generated entity-ID tokens per second divided by median device board power in watts",
            "device_memory_peak_gib": "NVML device-wide peak memory used during warmup and timed phases",
            "agreement": "exact ordered agreement of the first 16 entity IDs with archived FP32 serial output",
        },
    }
    OUT_JSON.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    fields = (
        "batch_size",
        "status",
        "repetitions",
        "servings_per_second",
        "tokens_per_second",
        "power_median_watts",
        "servings_per_watt",
        "tokens_per_watt",
        "device_memory_peak_gib",
        "first16_fp32_sequence_agreement",
        "protocol_valid_and_cap_complete_rate",
        "macro_f1_at_16_first8",
        "at_or_above_99_percent_of_decreed_peak",
        "source_result",
    )
    with OUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_plot_rows:
            writer.writerow({field: row.get(field) for field in fields})

    throughput_series = [
        {"label": "16-ID servings/s", "key": "servings_per_second", "color": "#0072b2"},
        {"label": "entity-ID tokens/s", "key": "tokens_per_second", "color": "#d55e00"},
    ]
    resource_series = [
        {"label": "device memory", "key": "device_memory_peak_gib", "color": "#009e73"}
    ]
    power_series = [{"label": "board power", "key": "power_median_watts", "color": "#cc79a7"}]
    write_svg(
        THROUGHPUT_SVG.with_name("2026-09-26-16id-fp32-servings-throughput.svg"),
        "16-ID FP32 servings per second",
        "Completed fixed-batch 16-ID FP32 servings per second. The dashed line marks the user-decreed 9,600-serving/s practical peak; the first completed batch within one percent is annotated.",
        all_plot_rows,
        series=[throughput_series[0]],
        y_max=11000,
        y_label="Servings per second",
        plateau_line=True,
        plateau_annotation=True,
    )
    write_svg(
        THROUGHPUT_SVG.with_name("2026-09-26-16id-fp32-tokens-throughput.svg"),
        "16-ID FP32 generated tokens per second",
        "Completed fixed-batch FP32 generated entity-ID tokens per second; EOS is excluded. The open point is an interrupted two-repetition observation.",
        all_plot_rows,
        series=[throughput_series[1]],
        y_max=180000,
        y_label="Entity-ID tokens per second",
    )
    write_svg(
        THROUGHPUT_SVG.with_name("2026-09-26-16id-fp32-servings-per-watt.svg"),
        "16-ID FP32 servings per watt",
        "Completed 16-ID serving rate divided by median device board power. The open point is an interrupted two-repetition observation.",
        all_plot_rows,
        series=[{"label": "servings/W", "key": "servings_per_watt", "color": "#0072b2"}],
        y_max=65,
        y_label="Servings per watt",
    )
    write_svg(
        THROUGHPUT_SVG.with_name("2026-09-26-16id-fp32-tokens-per-watt.svg"),
        "16-ID FP32 generated tokens per watt",
        "Generated entity-ID token rate divided by median device board power; EOS is excluded. The open point is an interrupted two-repetition observation.",
        all_plot_rows,
        series=[{"label": "tokens/W", "key": "tokens_per_watt", "color": "#d55e00"}],
        y_max=1200,
        y_label="Entity-ID tokens per watt",
    )
    write_svg(
        RESOURCE_SVG,
        "16-ID FP32 VRAM use",
        "Device-wide NVML peak memory used across the fixed-batch FP32 saturation series. The open point at batch 65,536 is partial; no CUDA OOM was recorded before the user stopped the run.",
        all_plot_rows,
        series=resource_series,
        y_max=TOTAL_MEMORY_GIB + 0.5,
        y_label="Device memory used (GiB)",
        total_memory_line=True,
        plateau_annotation=True,
    )
    write_svg(
        RESOURCE_SVG.with_name("2026-09-26-16id-fp32-board-power.svg"),
        "16-ID FP32 board power",
        "Median device board power for each completed fixed-batch FP32 row. The open point at batch 65,536 is a two-repetition partial observation.",
        all_plot_rows,
        series=power_series,
        y_max=200,
        y_label="Median board power (W)",
    )

    report_text = f"""# 16-ID FP32 batch curve — stopped before saturation

Date: 2026-09-26  
Status: FP32 measurement was stopped by the user before a CUDA out-of-memory boundary was observed.

## Reading the graphs

The practical peak is user-decreed as **9,600 servings/s** for the RTX 5070 Ti. For a reproducible first-plateau point, this report marks the first completed median within 1% of 9,600: **batch {first_plateau['batch_size']:,}**, using **{first_plateau['device_memory_peak_gib']:.2f} GiB** device-wide peak memory. The highest completed raw median in this series was **{raw_peak['servings_per_second']:,.1f} servings/s** at batch {raw_peak['batch_size']:,}; it is shown as a measured point while 9,600 remains the declared practical peak.

At batch {49152:,}, throughput fell to about 5,134 servings/s and at {57344:,} to about 846/s, while exact first-16 sequence agreement remained 100%. The last complete row used {last_complete['device_memory_peak_gib']:.2f} GiB device-wide memory. A 65,536 attempt produced two console-only repetitions near 3,791 servings/s at about 15.79 GiB, then was interrupted. It is marked as partial and excluded from completed-row peak calculations. Since no CUDA OOM was observed, this is **not a measured saturation point**.

## Metrics

- A serving is one bounded response ending on model EOS or exactly 16 generated entity IDs; no synthetic EOS is added.
- Tokens/s means returned generated entity-ID tokens/s; EOS is excluded. The measured token count is reported directly, not estimated from the serving rate.
- Servings/W and tokens/W divide the respective median rate by the median NVML device board power.
- The first-16 agreement column is exact ordered-prefix parity against the archived serial FP32 reference. The higher-batch runner also recorded protocol/cap validity and label-based macro F1 on a diagnostic eight-query validation slice; these do not control the stop rule.
- Memory and power are device-wide NVML readings under Windows WDDM. Desktop GPU processes remained resident.

## Data and evidence

- Merged rates, watts, efficiency, memory, parity and partial status: [`fp32-16id-vram-series.csv`](../../runs/learning/16id-serving-saturation-20260926/fp32-16id-vram-series.csv)
- Structured derivation and source hashes: [`fp32-16id-vram-series.json`](../../runs/learning/16id-serving-saturation-20260926/fp32-16id-vram-series.json)
- Stop/interruption evidence: [`fp32-user-stop-amendment.json`](../../runs/learning/16id-serving-saturation-20260926/fp32-user-stop-amendment.json)
- Full batch raw results: `runs/learning/16id-serving-saturation-20260926/results.json`, `results-large-batches.json`, `results-near-target.json`, `results-fp32-vram.json`, and `results-fp32-vram-extension.json`.

The plotted full-decode-then-cap data from the earlier 130 W campaign is excluded. Only actual `max_new_tokens=16` runs enter this series.
"""
    REPORT.write_text(report_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "completed_rows": len(completed),
                "partial_batch": partial["batch_size"],
                "first_plateau_batch": first_plateau["batch_size"],
                "first_plateau_memory_gib": first_plateau["device_memory_peak_gib"],
                "raw_completed_peak_batch": raw_peak["batch_size"],
                "raw_completed_peak_servings_per_second": raw_peak["servings_per_second"],
                "best_completed_servings_per_watt_batch": best_efficiency["batch_size"],
                "csv": str(OUT_CSV),
                "json": str(OUT_JSON),
                "report": str(REPORT),
                "svg": [
                    str(THROUGHPUT_SVG.with_name("2026-09-26-16id-fp32-servings-throughput.svg")),
                    str(THROUGHPUT_SVG.with_name("2026-09-26-16id-fp32-tokens-throughput.svg")),
                    str(THROUGHPUT_SVG.with_name("2026-09-26-16id-fp32-servings-per-watt.svg")),
                    str(THROUGHPUT_SVG.with_name("2026-09-26-16id-fp32-tokens-per-watt.svg")),
                    str(RESOURCE_SVG),
                    str(RESOURCE_SVG.with_name("2026-09-26-16id-fp32-board-power.svg")),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
