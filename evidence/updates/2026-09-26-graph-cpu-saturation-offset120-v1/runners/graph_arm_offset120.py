"""Measure capped SQLite graph HTTP throughput with a derived +120 W power offset."""

from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio
import uvicorn

from plm.protocol.config import load_protocol
from plm.serving.native import PredictRequest

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent
RECEIPT = ROOT / "runs/learning/native-saturation-20260925/deployments/native-a2hhcdo7/serving.json"
REFERENCE = ROOT / "docs/experiments/2026-09-25-native-graph-oracle.json"
GPU_REFERENCE = ROOT / "runs/learning/batch-saturation-20260925/summary.json"
PLAN = ROOT / "docs/experiments/2026-09-26-graph-cpu-saturation-offset120-plan.md"
PROTOCOL = ROOT / "configs/protocol/pokemon_v1.yaml"
OUT = RUN_DIR / "graph-arm-offset120.json"
LEVELS = (1, 4, 8, 16, 32, 48, 56, 64, 72, 80, 96, 112, 128)
DURATION_SECONDS = 10.0
RESPONSE_LIMIT = 16
POWER_OFFSET_WATTS = 120.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_power_samples(
    samples: list[dict[str, Any]], lock: threading.Lock
) -> tuple[subprocess.Popen[str], threading.Thread]:
    command = r"""
$ErrorActionPreference = 'Stop'
$counter = '\Energy Meter(RAPL_Package0_PKG)\Power'
while ($true) {
  $sample = Get-Counter -Counter $counter -ErrorAction Stop
  [pscustomobject]@{
    utc = $sample.Timestamp.ToUniversalTime().ToString('o')
    watts = [double]($sample.CounterSamples[0].CookedValue / 1000.0)
  } | ConvertTo-Json -Compress
}
"""
    process = subprocess.Popen(
        [shutil.which("pwsh.exe") or shutil.which("powershell.exe") or "pwsh.exe", "-NoProfile", "-Command", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("failed to start the package-power sampler")

    def collect() -> None:
        for line in process.stdout:
            try:
                value = json.loads(line)
                sample = {
                    "monotonic_seconds": time.perf_counter(),
                    "utc": value["utc"],
                    "package_watts": float(value["watts"]),
                }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            with lock:
                samples.append(sample)

    reader = threading.Thread(target=collect, name="rapl-reader", daemon=True)
    reader.start()
    time.sleep(2)
    if process.poll() is not None:
        error = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"RAPL power sampler exited: {error}")
    return process, reader


class ThreadLocalGraph:
    def __init__(self, snapshot: Path, protocol: Any, graph_hash: str, snapshot_hash: str) -> None:
        self.snapshot = snapshot
        self.protocol = protocol
        self.graph_hash = graph_hash
        self.local = threading.local()
        self.instances: list[Any] = []
        self.lock = threading.Lock()
        self.deployment = {
            "ready": True,
            "backend": "sqlite-graph-oracle-v1",
            "graph_hash": graph_hash,
            "snapshot_sha256": snapshot_hash,
            "protocol_version": protocol.version,
        }

    def _predictor(self) -> Any:
        predictor = getattr(self.local, "predictor", None)
        if predictor is None:
            from benchmark_plm_graph import GraphOracle

            predictor = GraphOracle(self.snapshot, self.protocol, self.graph_hash)
            self.local.predictor = predictor
            with self.lock:
                self.instances.append(predictor)
        return predictor

    def predict(self, request: PredictRequest) -> dict[str, Any]:
        return self._predictor().predict(request)

    def close(self) -> None:
        for predictor in self.instances:
            predictor.close()


def make_server(graph: ThreadLocalGraph) -> Any:
    from benchmark_plm_graph import _graph_app

    app = _graph_app(graph)

    @app.on_event("startup")
    async def increase_request_workers() -> None:
        anyio.to_thread.current_default_thread_limiter().total_tokens = max(128, len(LEVELS))

    return app


def run_level(
    base_url: str,
    queries: list[tuple[str, str]],
    expected: dict[tuple[str, str], tuple[str, ...]],
    concurrency: int,
) -> dict[str, Any]:
    endpoint = urlsplit(base_url)
    payloads = [
        json.dumps(
            PredictRequest(subject=subject, dimension=dimension, mode="SAME", limit=RESPONSE_LIMIT)
            .model_dump(),
            separators=(",", ":"),
        )
        for subject, dimension in queries
    ]
    barrier = threading.Barrier(concurrency + 1)
    start_event = threading.Event()
    deadline_holder = [0.0]
    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()

    def client(worker: int) -> None:
        connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=30)
        index = worker % len(payloads)
        barrier.wait()
        start_event.wait()
        while time.perf_counter() < deadline_holder[0]:
            started = time.perf_counter()
            try:
                connection.request(
                    "POST",
                    "/v1/predict",
                    body=payloads[index],
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                body = response.read()
                finished = time.perf_counter()
                success = False
                if response.status == 200:
                    decoded = json.loads(body)
                    targets = tuple(item["key"] for item in decoded["result"]["targets"])
                    success = targets == expected[queries[index]][:RESPONSE_LIMIT]
                with results_lock:
                    results.append(
                        {"finished": finished, "success": success, "latency": finished - started}
                    )
                index = (index + 1) % len(payloads)
                if response.will_close:
                    connection.close()
                    connection = http.client.HTTPConnection(
                        endpoint.hostname, endpoint.port, timeout=30
                    )
            except Exception as exc:
                finished = time.perf_counter()
                with results_lock:
                    results.append(
                        {
                            "finished": finished,
                            "success": False,
                            "latency": finished - started,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                connection.close()
                connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=30)
        connection.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(client, worker) for worker in range(concurrency)]
        barrier.wait()
        started = time.perf_counter()
        deadline = started + DURATION_SECONDS
        deadline_holder[0] = deadline
        start_event.set()
        for future in futures:
            future.result()
    accepted = [row for row in results if row["success"] and row["finished"] <= deadline]
    failures = [row for row in results if not row["success"] and row["finished"] <= deadline]
    latencies = [row["latency"] for row in accepted]
    return {
        "client_concurrency": concurrency,
        "timed_seconds": DURATION_SECONDS,
        "completed_servings": len(accepted),
        "completed_servings_per_second": len(accepted) / DURATION_SECONDS,
        "failed_responses": len(failures),
        "all_successes_returned_16_ids": not failures and len(accepted) > 0,
        "latency_seconds_median": statistics.median(latencies) if latencies else None,
        "latency_seconds_p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        if latencies
        else None,
        "errors": [row.get("error") for row in failures[:5]],
    }


def stop_sampler(process: subprocess.Popen[str], reader: threading.Thread) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    reader.join(timeout=2)


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_plm_graph import _serve

    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    raw_reference_path = ROOT / reference["raw_report"]
    raw_reference = json.loads(raw_reference_path.read_text(encoding="utf-8"))
    prior_gpu = json.loads(GPU_REFERENCE.read_text(encoding="utf-8"))
    queries = [tuple(item) for item in prior_gpu["workload"]["base_prompts"]]
    expected = {
        (row["subject"], row["dimension"]): tuple(row["predicted_targets"])
        for row in raw_reference["arms"]["graph"]["responses"][:8]
    }
    snapshot = Path(receipt["snapshot"])
    snapshot_hash = sha(snapshot)
    if snapshot_hash != receipt["snapshot_sha256"]:
        raise ValueError("graph snapshot differs from the serving receipt")
    graph_hash = receipt["corpus_identity"]["graph_hash"]
    protocol = load_protocol(PROTOCOL)
    graph = ThreadLocalGraph(snapshot, protocol, graph_hash, snapshot_hash)
    app = make_server(graph)
    samples: list[dict[str, Any]] = []
    samples_lock = threading.Lock()
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    output: dict[str, Any] = {
        "status": "running",
        "scope": "loopback SQLite graph HTTP; CPU package RAPL; fixed 16-ID response cap",
        "date": "2026-09-26",
        "plan_sha256": sha(PLAN),
        "runner_sha256": sha(Path(__file__)),
        "serving_receipt_sha256": sha(RECEIPT),
        "graph_reference_sha256": sha(REFERENCE),
        "graph_raw_reference_sha256": sha(raw_reference_path),
        "graph_snapshot_sha256": snapshot_hash,
        "graph_hash": graph_hash,
        "query_prompts": queries,
        "response_limit_ids": RESPONSE_LIMIT,
        "cpu_power_source": "Windows Energy Meter(RAPL_Package0_PKG) Power counter; CookedValue / 1000",
        "cpu_power_offset_watts": POWER_OFFSET_WATTS,
        "power_offset_method": "additive derived series: measured CPU package RAPL watts + 120 W",
        "power_offset_is_wall_power_measurement": False,
        "power_target_watts": None,
        "power_sample_interval_seconds": 1.0,
        "level_duration_seconds": DURATION_SECONDS,
        "levels": [],
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    try:
        with _serve(app, "graph-power") as base_url:
            warm = http.client.HTTPConnection(urlsplit(base_url).hostname, urlsplit(base_url).port, timeout=10)
            warm.request(
                "POST",
                "/v1/predict",
                body=json.dumps(
                    PredictRequest(subject=queries[0][0], dimension=queries[0][1], limit=RESPONSE_LIMIT)
                    .model_dump(),
                    separators=(",", ":"),
                ),
                headers={"Content-Type": "application/json"},
            )
            response = warm.getresponse()
            body = json.loads(response.read())
            if response.status != 200 or len(body["result"]["targets"]) != RESPONSE_LIMIT:
                raise RuntimeError("graph warmup failed or did not return exactly 16 IDs")
            expected_first = list(expected[(queries[0][0], queries[0][1])][:RESPONSE_LIMIT])
            actual_first = [item["key"] for item in body["result"]["targets"]]
            if actual_first != expected_first:
                raise RuntimeError("capped graph response differs from the frozen graph reference")
            warm.close()
            process, reader = read_power_samples(samples, samples_lock)
            for concurrency in LEVELS:
                power_start = time.perf_counter()
                row = run_level(base_url, queries, expected, concurrency)
                power_end = time.perf_counter()
                with samples_lock:
                    scoped_samples = [
                        sample
                        for sample in samples
                        if power_start <= sample["monotonic_seconds"] <= power_end
                    ]
                watts = [sample["package_watts"] for sample in scoped_samples]
                row["cpu_package_power_sample_count"] = len(watts)
                row["cpu_package_power_median_watts"] = statistics.median(watts) if watts else None
                row["cpu_package_power_min_watts"] = min(watts) if watts else None
                row["cpu_package_power_max_watts"] = max(watts) if watts else None
                row["cpu_package_power_samples"] = scoped_samples
                row["power_offset_watts"] = POWER_OFFSET_WATTS
                row["offset_adjusted_power_median_watts"] = (
                    row["cpu_package_power_median_watts"] + POWER_OFFSET_WATTS
                    if row["cpu_package_power_median_watts"] is not None
                    else None
                )
                row["offset_adjusted_power_min_watts"] = (
                    row["cpu_package_power_min_watts"] + POWER_OFFSET_WATTS
                    if row["cpu_package_power_min_watts"] is not None
                    else None
                )
                row["offset_adjusted_power_max_watts"] = (
                    row["cpu_package_power_max_watts"] + POWER_OFFSET_WATTS
                    if row["cpu_package_power_max_watts"] is not None
                    else None
                )
                raw_median = row["cpu_package_power_median_watts"]
                adjusted_median = row["offset_adjusted_power_median_watts"]
                rate = row["completed_servings_per_second"]
                row["servings_per_raw_cpu_package_watt"] = (
                    rate / raw_median if raw_median is not None and raw_median > 0 else None
                )
                row["servings_per_offset_adjusted_watt"] = (
                    rate / adjusted_median
                    if adjusted_median is not None and adjusted_median > 0
                    else None
                )
                output["levels"].append(row)
                OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
                print(
                    f"concurrency={concurrency} servings/s={row['completed_servings_per_second']:.1f} "
                    f"CPU-package-W={row['cpu_package_power_median_watts']} "
                    f"offset-adjusted-W={row['offset_adjusted_power_median_watts']}",
                    flush=True,
                )
    except BaseException as exc:
        output["status"] = "failed"
        output["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None and reader is not None:
            stop_sampler(process, reader)
        graph.close()
        output["cpu_package_power_samples_all"] = samples
        if output["status"] == "running":
            output["status"] = "complete"
        OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
