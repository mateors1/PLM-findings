"""Paired FP32/FP8 validation quality and native serving benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from plm.config import RootConfig
from plm.serving.fp8 import quantize_fp8_decoder
from plm.serving.generation_batch import generate_responses
from plm.serving.native import NativePredictor, PredictRequest
from plm.serving.parser import parse_generated
from plm.serving.postprocess import postprocess_response
from plm.serving.runtime import InferenceRuntime, load_inference_runtime


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality(
    runtime: InferenceRuntime, batch_size: int
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    records = runtime.split.validation
    policy = runtime.config.eval
    total_f1 = 0.0
    exact = valid = terminated = 0
    raw: dict[str, tuple[int, ...]] = {}
    for offset in range(0, len(records), batch_size):
        group = records[offset : offset + batch_size]
        results = generate_responses(
            runtime.model,
            runtime.vocabulary,
            [(row.subject, row.dimension) for row in group],
            max_new_tokens=policy.max_new_tokens,
            device=runtime.device,
            prevent_repeated_targets=policy.prevent_repeated_targets,
            first_target_guidance_alpha=policy.first_target_guidance_alpha,
        )
        for record, result in zip(group, results, strict=True):
            raw[f"{record.subject}:{record.dimension}"] = result.token_ids
            valid += result.protocol_valid
            terminated += result.terminated
            if result.protocol_valid and result.terminated:
                parsed = parse_generated(result.token_ids, runtime.vocabulary)
                selected = set(postprocess_response(parsed).response.targets)
            else:
                selected = set()
            expected = set(record.targets)
            precision = len(selected & expected) / len(selected) if selected else 0.0
            recall = len(selected & expected) / len(expected)
            total_f1 += 2 * precision * recall / (precision + recall) if precision + recall else 0
            exact += result.protocol_valid and result.terminated and selected == expected
    count = len(records)
    return (
        {
            "query_count": count,
            "macro_f1": total_f1 / count,
            "exact_set_count": exact,
            "exact_set_accuracy": exact / count,
            "protocol_valid_rate": valid / count,
            "termination_rate": terminated / count,
            "evaluation": "postprocessed SAME sets on the full validation split",
        },
        raw,
    )


def _percentile(values: list[float], percent: int) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * percent / 100) - 1)]


def _serving(predictor: NativePredictor, concurrency: int, requests: int) -> dict[str, Any]:
    records = predictor.runtime.split.validation[:8]
    payloads = [PredictRequest(subject=row.subject, dimension=row.dimension) for row in records]
    for payload in payloads:
        predictor.predict(payload)

    def invoke(index: int) -> tuple[float, float, float, int]:
        started = time.perf_counter()
        response = predictor.predict(payloads[index % len(payloads)])
        raw = response["raw"]
        if not raw["protocol_valid"] or not raw["terminated"]:
            raise ValueError("native serving produced an invalid or unfinished response")
        return (
            time.perf_counter() - started,
            response["timing"]["queue_seconds"],
            response["timing"]["generation_seconds"],
            len(raw["token_ids"]) - 5,
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        observations = list(pool.map(invoke, range(requests)))
    elapsed = time.perf_counter() - started
    return {
        "path": "NativePredictor.predict; direct call including postprocessing and hydration",
        "concurrency": concurrency,
        "requests": requests,
        "completed_rps": requests / elapsed,
        "completed_output_tps": sum(row[3] for row in observations) / elapsed,
        "client_seconds_p50": statistics.median(row[0] for row in observations),
        "client_seconds_p95": _percentile([row[0] for row in observations], 95),
        "queue_seconds_p95": _percentile([row[1] for row in observations], 95),
        "generation_seconds_p50": statistics.median(row[2] for row in observations),
    }


def _batch(runtime: InferenceRuntime, size: int, repetitions: int) -> dict[str, Any]:
    import torch

    records = runtime.split.validation[:8]
    queries = [(row.subject, row.dimension) for row in records]
    queries = [queries[index % len(queries)] for index in range(size)]
    samples: list[tuple[float, float, int]] = []
    for iteration in range(repetitions + 1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        results = generate_responses(
            runtime.model,
            runtime.vocabulary,
            queries,
            max_new_tokens=runtime.config.eval.max_new_tokens,
            device=runtime.device,
            prevent_repeated_targets=runtime.config.eval.prevent_repeated_targets,
            first_target_guidance_alpha=runtime.config.eval.first_target_guidance_alpha,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if any(not row.protocol_valid or not row.terminated for row in results):
            raise ValueError(f"batch {size} produced invalid or unfinished output")
        if iteration:
            samples.append(
                (elapsed, size / elapsed, sum(len(row.token_ids) - 5 for row in results))
            )
    return {
        "batch_size": size,
        "repetitions": repetitions,
        "median_seconds": statistics.median(row[0] for row in samples),
        "median_completed_rps": statistics.median(row[1] for row in samples),
        "median_completed_output_tps": statistics.median(row[2] / row[0] for row in samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--frontier-report", type=Path, required=True)
    parser.add_argument("--frontier-seed", type=int, default=1730)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality-batch-size", type=int, default=8)
    parser.add_argument("--serving-requests", type=int, default=16)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--batch-repetitions", type=int, default=2)
    args = parser.parse_args()
    if (
        args.out.exists()
        or min(
            args.quality_batch_size,
            args.serving_requests,
            args.batch_repetitions,
            *args.batch_sizes,
        )
        < 1
    ):
        raise ValueError("output must be new and benchmark counts must be positive")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    frontier = json.loads(args.frontier_report.read_text(encoding="utf-8"))
    if not frontier.get("integration_accepted") or frontier.get("protected_test_used"):
        raise ValueError("frontier must be an accepted validation-only integration report")
    matches = [row for row in frontier["runs"] if row["seed"] == args.frontier_seed]
    if len(matches) != 1 or matches[0]["offline_metrics"]["query_count"] != 222:
        raise ValueError("frontier seed must identify one full validation result")
    denominator = matches[0]["offline_metrics"]["f1"]
    if not 0 < denominator <= 1:
        raise ValueError("frontier macro F1 must be in (0, 1]")
    config = RootConfig.model_validate(receipt["config"])
    if config.eval.symmetric_set_reranking or config.eval.pair_set_composition:
        raise ValueError("this benchmark evaluates the guided decoder, not set selection")
    if not config.eval.use_kv_cache or not config.eval.constrained_decoding:
        raise ValueError("benchmark requires constrained cached serving")
    result: dict[str, Any] = {
        "scope": "paired native serving and offline fixed-group decoding on validation prompts",
        "reference_checkpoint_hash": receipt["checkpoint_hash"],
        "receipt_sha256": _sha(args.receipt),
        "frontier": {
            "definition": (
                "best accepted single-seed PLM validation macro F1 "
                "in the selected report"
            ),
            "report_sha256": _sha(args.frontier_report),
            "seed": args.frontier_seed,
            "checkpoint_and_policy": "pair-composition shape-aware v2",
            "query_count": 222,
            "macro_f1": denominator,
        },
        "script_sha256": _sha(Path(__file__)),
        "arms": {},
    }
    fp32_raw: dict[str, tuple[int, ...]] | None = None
    for precision in ("fp32", "fp8"):
        runtime = load_inference_runtime(
            config, receipt["checkpoint"], graph_db=receipt["snapshot"]
        )
        if runtime.checkpoint_hash != receipt["checkpoint_hash"] or runtime.device != "cuda":
            raise ValueError("selected deployment identity or CUDA device differs")
        conversion = quantize_fp8_decoder(runtime.model) if precision == "fp8" else None
        quality, raw = _quality(runtime, args.quality_batch_size)
        if quality["query_count"] != result["frontier"]["query_count"]:
            raise ValueError("frontier and measured query counts differ")
        quality["percent_of_frontier_f1"] = 100 * quality["macro_f1"] / denominator
        if fp32_raw is None:
            fp32_raw = raw
        else:
            quality["raw_token_agreement_with_fp32"] = sum(
                raw[key] == value for key, value in fp32_raw.items()
            ) / len(fp32_raw)
        predictor = NativePredictor(
            runtime, Path(receipt["snapshot"]), max_pending=max(args.batch_sizes)
        )
        serving = [_serving(predictor, level, args.serving_requests) for level in (1, 4)]
        batches = [_batch(runtime, size, args.batch_repetitions) for size in args.batch_sizes]
        result["arms"][precision] = {
            "conversion": conversion,
            "quality": quality,
            "serving": serving,
            "offline_batches": batches,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"{precision}: F1={quality['macro_f1']:.6f}, "
            f"frontier={quality['percent_of_frontier_f1']:.2f}%",
            flush=True,
        )
        del predictor, runtime


if __name__ == "__main__":
    main()
