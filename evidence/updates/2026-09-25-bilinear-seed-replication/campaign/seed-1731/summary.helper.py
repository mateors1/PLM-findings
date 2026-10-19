"""Fixed width-eight inference experiment; archived runtime, exact width-four replay.

Preflight is stdlib-only. Actual execution imports only the authenticated isolated
runtime. Saved-output auditing cannot independently prove the neural first-logit
rank ordering; that behavior is bound to the archived generation implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import platform
import struct
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

_PLAN = "02faaedcf88c94c65e70228488431a85e6687e34ea6d6f530f90ec51f43e1461"
_SUMMARY = "77f9d9bba8eb9c5e0cf5df86f2eb29f522cf00ad50600ee9eea76267ec041c8f"
_AUDIT = "40eba8d9dac72dbe762eb4b54e7c053128b08580f905dd7539a89961c144ebde"
_ACCEPTANCE = "1d5835709646a9dbf7f8db2727e3adabef2c451ff75be932e2eb22833acdd9d4"
_SOURCE = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
_CONFIGS = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
_SPLIT = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
_SEEDS = (1729, 1730, 1731)
_GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
_COLUMNS = list(range(1024, 2049))
_ORDER10 = ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
_ORDER36 = (
    *_ORDER10,
    (5,),
    (6,),
    (7,),
    (8,),
    *(p for p in itertools.combinations(range(1, 9), 2) if p[1] > 4),
)
_BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _failure_safe(value):
    """Retain invalid neural floats without making failure JSON unwritable."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {k: _failure_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_failure_safe(v) for v in value]
    return value


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _bind(path, digest, inputs):
    path = Path(path).resolve()
    _require(path.is_file() and _sha(path) == digest, f"input identity mismatch: {path}")
    inputs[str(path)] = digest
    return path


def _unchanged(inputs):
    for path, digest in inputs.items():
        _require(_sha(path) == digest, f"input changed: {path}")


def _refuse(output):
    _require(
        not output.exists()
        and not list(output.parent.glob(output.stem + ".*"))
        and not list(output.parent.glob("seed-*.json"))
        and not list(output.parent.glob("baseline-*.json"))
        and not (output.parent / "runtime").exists(),
        "immutable output exists",
    )


def _archive_members(archives):
    result = {}
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name = info.filename
                parts = PurePosixPath(name)
                _require(
                    not info.is_dir()
                    and info.orig_filename == name
                    and not parts.is_absolute()
                    and ".." not in parts.parts
                    and "\\" not in name
                    and ":" not in name
                    and parts.as_posix() == name
                    and name.casefold() not in {n.casefold() for n in result}
                    and (info.external_attr >> 16) & 0o170000 != 0o120000,
                    "unsafe or duplicate archive member",
                )
                result[name] = archive.read(info)
    _require(
        "src/plm/__init__.py" in result
        and "configs/config.yaml" in result
        and "configs/protocol/pokemon_v1.yaml" in result,
        "missing archived package/config",
    )
    return result


def _extract(members, destination):
    destination.mkdir()  # Deliberately refuses even an empty existing runtime.
    hashes = {}
    for name, content in members.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        hashes[str(target.resolve())] = hashlib.sha256(content).hexdigest()
    return hashes


def _modules(runtime=None):
    loaded = {k: v for k, v in sys.modules.items() if k == "plm" or k.startswith("plm.")}
    origins = {}
    if runtime is None:
        _require(not loaded, "plm was imported before runtime isolation")
    else:
        for name, module in loaded.items():
            path = getattr(module, "__file__", None)
            _require(
                path is not None and Path(path).resolve().is_relative_to(runtime / "src"),
                f"runtime module contamination: {name}",
            )
            origins[name] = {"path": str(Path(path).resolve()), "sha256": _sha(path)}
    return origins


def _runtime_unchanged(runtime, hashes):
    actual = {
        str(p.resolve()) for p in runtime.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    _require(actual == set(hashes), "extracted runtime inventory changed")
    _unchanged(hashes)


def _sources(paths, prompt, vocabulary):
    first = []
    for rank, path in enumerate(paths, 1):
        ids = list(path.token_ids)
        products = ids[5:-1] if path.terminated else ids[5:]
        _require(ids[:5] == prompt and 6 <= len(ids) <= 512, "source prompt/length")
        _require(all(i in _COLUMNS for i in products), "source product mask")
        _require(not path.terminated or ids[-1] == vocabulary.eos_id, "source EOS")
        _require(len(products) == len(set(products)), "source uniqueness mask")
        _require(list(vocabulary.encode(path.targets)) == products, "source raw target alignment")
        _require(
            path.decoding == _BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            "source rank descriptor",
        )
        first.append(products[0])
    _require(len(first) == len(set(first)), "distinct first-choice branches")


def _preflight(root, plan):
    inputs = {}
    _bind(plan, _PLAN, inputs)
    directory = root / "runs/learning/pair-composition-integration-v2"
    summary = _read(_bind(directory / "summary.json", _SUMMARY, inputs))
    audit = _read(_bind(directory / "independent-audit.json", _AUDIT, inputs))
    acceptance = _read(_bind(directory / "acceptance.json", _ACCEPTANCE, inputs))
    _require(
        summary["verification_complete"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == _SUMMARY
        and acceptance["integration_accepted"] is True
        and acceptance["summary_sha256"] == _SUMMARY
        and acceptance["independent_audit_sha256"] == _AUDIT,
        "accepted integration chain",
    )
    _bind(directory / "independent-audit.py", audit["script_sha256"], inputs)
    inherited = {str(Path(p).resolve()): digest for p, digest in summary["input_sha256"].items()}
    prior = root / "runs/learning/pair-composition-integration-v1"
    source = _bind(prior / "summary.source.zip", _SOURCE, inputs)
    configs = _bind(prior / "summary.configs.zip", _CONFIGS, inputs)
    members = _archive_members((source, configs))
    # Bind only inputs actually consumed, not unrelated current source/config files.
    for path, digest in inherited.items():
        relative = Path(path).relative_to(root)
        if relative.parts[0] == "data" or any(
            f"national_dex_continuation_control_s{seed}_v1" in relative.parts for seed in _SEEDS
        ):
            _bind(path, digest, inputs)
    references = {}
    for reused in summary["offline_reuse"]:
        seed = reused["seed"]
        _require(seed in _SEEDS and seed not in references, "offline seed identity")
        offline = _read(_bind(reused["report"], reused["report_sha256"], inputs))
        path = root / f"runs/learning/pair-unions-v1/seed-{seed}.json"
        pair = _read(_bind(path, inherited[str(path.resolve())], inputs))
        _require(
            offline["exact_parity"] is True
            and offline["seed"] == pair["seed"] == seed
            and offline["split_hash"] == pair["split_hash"] == _SPLIT
            and len(offline["responses"]) == len(pair["responses"]) == 222,
            "reference coverage/parity",
        )
        for left, right in zip(offline["responses"], pair["responses"], strict=True):
            _require(
                all(left[k] == right[k] for k in ("subject", "dimension", "expected", "group")),
                "reference query alignment",
            )
        references[seed] = {"offline": offline, "pair": pair}
    _require(set(references) == set(_SEEDS), "three reference seeds")
    return {
        "inputs": inputs,
        "members": members,
        "references": references,
        "source": source,
        "configs": configs,
    }


def _fp32(values):
    _require(len(values) == 1025, "head column count")
    for value in values:
        _require(type(value) in (int, float) and math.isfinite(value), "nonfinite head")
        _require(struct.unpack("f", struct.pack("f", value))[0] == value, "not FP32 head")


def _compose(paths, logits, vocabulary, query, sets_and_scores, score_set):
    """Truth-free selection; canonical membership/scoring remain archived helpers."""
    _fp32(logits)
    _require(len(paths) in (4, 8), "source width")
    sets, _, eligible = sets_and_scores(paths, logits, vocabulary, query)
    order = _ORDER10 if len(paths) == 4 else _ORDER36
    slots = []
    for index, ranks in enumerate(order, 1):
        ids = sorted(set().union(*(sets[r - 1] for r in ranks)))
        slots.append(
            {
                "slot": index,
                "kind": "original" if len(ranks) == 1 else "pair_composition",
                "source_ranks": list(ranks),
                "set_ids": ids,
                "source_eligible": all(eligible[r - 1] for r in ranks),
                "score": score_set(ids, logits),
            }
        )
    choices = [slot for slot in slots if slot["source_eligible"]]
    selected = (
        min(choices, key=lambda slot: (-slot["score"], slot["slot"])) if choices else slots[0]
    )
    return {
        "source_paths": [_json(asdict(p)) for p in paths],
        "slots": slots,
        **{
            f"selected_{key}": selected[key]
            for key in ("slot", "kind", "source_ranks", "set_ids", "source_eligible")
        },
        "fallback_no_valid_source": not choices,
        "policy": _BASE
        + (
            "+first4-pair6-symmetric-set-logit-sum-v1"
            if len(paths) == 4
            else "+first8-pair28-symmetric-set-logit-sum-v1"
        ),
    }


def _legacy(actual, logits, offline, pair):
    _require(logits == pair["symmetric_relation_logits"], "exact historical head replay failed")
    _require(actual == offline["composition"], "exact historical composition replay failed")


def _metrics(composition, truth):
    success = (
        composition["selected_source_eligible"] and not composition["fallback_no_valid_source"]
    )
    predicted = set(composition["selected_set_ids"]) if success else set()
    expected = set(truth)
    _require(expected, "empty expected set")
    matches = len(predicted & expected)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(expected)
    return {
        "exact": bool(success and predicted == expected),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "set_size": len(predicted),
        "success": bool(success),
    }


def _availability(composition, truth):
    eligible = [s for s in composition["slots"] if s["source_eligible"]]
    available = any(s["set_ids"] == truth for s in eligible)
    originals = [s for s in eligible if s["kind"] == "original"]
    union = sorted(set().union(*(s["set_ids"] for s in originals)))
    return {
        "exact_available": available,
        "available_exact_miss": available and not _metrics(composition, truth)["exact"],
        "source_union_contains_truth": set(truth) <= set(union),
        "source_union_ids": union,
        "distinct_eligible_sets": len({tuple(s["set_ids"]) for s in eligible}),
    }


def _row(base, wide):
    truth = base["expected_set_ids"]
    compositions = {"baseline": base["composition"], "wide": wide}
    metrics = {arm: _metrics(c, truth) for arm, c in compositions.items()}
    return {
        **{k: v for k, v in base.items() if k != "composition"},
        "baseline": base["composition"],
        "composition": wide,
        "metrics": metrics,
        "availability": {arm: _availability(c, truth) for arm, c in compositions.items()},
        "subject_removed": [
            p["protocol_valid"] and p["terminated"] and base["subject"] in p["targets"]
            for p in wide["source_paths"]
        ],
        "gained_exact": metrics["wide"]["exact"] and not metrics["baseline"]["exact"],
        "lost_exact": metrics["baseline"]["exact"] and not metrics["wide"]["exact"],
    }


def _totals(rows):
    _require(rows, "empty aggregation")
    result = {
        "query_count": len(rows),
        "gains": sum(r["gained_exact"] for r in rows),
        "losses": sum(r["lost_exact"] for r in rows),
    }
    for arm, key in (("baseline", "baseline"), ("wide", "composition")):
        result[arm] = {
            "exact_count": sum(r["metrics"][arm]["exact"] for r in rows),
            **{
                metric: math.fsum(r["metrics"][arm][metric] for r in rows) / len(rows)
                for metric in ("precision", "recall", "f1", "set_size")
            },
            **{
                metric: sum(r["availability"][arm][metric] for r in rows)
                for metric in (
                    "exact_available",
                    "available_exact_miss",
                    "source_union_contains_truth",
                )
            },
            "mean_distinct_eligible_sets": math.fsum(
                r["availability"][arm]["distinct_eligible_sets"] for r in rows
            )
            / len(rows),
            "original_selections": sum(
                r["metrics"][arm]["success"] and r[key]["selected_kind"] == "original" for r in rows
            ),
            "pair_selections": sum(
                r["metrics"][arm]["success"] and r[key]["selected_kind"] == "pair_composition"
                for r in rows
            ),
            "added_slot_selections": sum(
                r["metrics"][arm]["success"] and r[key]["selected_slot"] > 10 for r in rows
            ),
            "failures": sum(not r["metrics"][arm]["success"] for r in rows),
        }
    return result


def _aggregate(rows):
    return {
        "overall": _totals(rows),
        "groups": {group: _totals([r for r in rows if r["group"] == group]) for group in _GROUPS},
    }


def _gate(reports, pooled):
    checks = {
        "complete_exact_replay": len(reports) == 3
        and all(
            r["complete"] and r["baseline_replay_exact"] and r["query_count"] == 222
            for r in reports
        ),
        "all_sources_valid_terminated_unique": all(
            all(
                p["protocol_valid"]
                and p["terminated"]
                and p["error"] is None
                and len(p["targets"]) == len(set(p["targets"]))
                for p in row["composition"]["source_paths"]
            )
            for report in reports
            for row in report["responses"]
        ),
        "all_selected_eligible": pooled["overall"]["wide"]["failures"] == 0,
        "per_seed_exact_nonregression": all(
            r["overall"]["wide"]["exact_count"] >= r["overall"]["baseline"]["exact_count"]
            for r in reports
        ),
        "per_seed_f1_nonregression": all(
            r["overall"]["wide"]["f1"] >= r["overall"]["baseline"]["f1"] for r in reports
        ),
        "pooled_exact_improvement": pooled["overall"]["wide"]["exact_count"] > 569,
        "pooled_group_exact_nonregression": all(
            g["wide"]["exact_count"] >= g["baseline"]["exact_count"]
            for g in pooled["groups"].values()
        ),
        "pooled_dual_exact_improvement": pooled["groups"]["TYPE_dual"]["wide"]["exact_count"]
        > pooled["groups"]["TYPE_dual"]["baseline"]["exact_count"],
    }
    return {"checks": checks, "quality_passed": all(checks.values())}


def _work(paths, offset, rank, seconds):
    lengths = [len(p.token_ids) - 5 for p in paths]
    calls = max(lengths)
    return {
        "offset": offset,
        "batch_size": len(paths),
        "rank": rank,
        "wall_seconds": seconds,
        "decode_calls": calls,
        "padded_decode_positions": len(paths) * (5 + calls - 1),
        "useful_emitted_tokens": sum(lengths),
        "guidance_forward_calls": 1,
    }


def _settings(torch):
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "autocast_cuda_enabled": torch.is_autocast_enabled("cuda"),
        "autocast_cpu_enabled": torch.is_autocast_enabled("cpu"),
    }


def _environment(torch):
    _require(torch.cuda.is_available(), "CUDA required for historical replay")
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    _require(
        env["python"] == "3.12.13"
        and env["torch"] == "2.11.0+cu128"
        and env["cuda"] == "12.8"
        and env["gpu"] == "NVIDIA GeForce RTX 5070 Ti",
        "historical dependency/device mismatch",
    )
    _require(
        _settings(torch)
        == {
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "deterministic_algorithms": False,
            "autocast_cuda_enabled": False,
            "autocast_cpu_enabled": False,
        },
        "historical numerical settings mismatch",
    )
    return env


def _seed(root, output, seed, reference, runtime_root):
    import torch

    from plm.config import RootConfig
    from plm.protocol import load_protocol
    from plm.serving.generation import GenerationResult
    from plm.serving.parser import prompt_ids
    from plm.serving.runtime import load_inference_runtime
    from plm.serving.set_reranking import _candidate_sets_and_scores, _generate_rank, _score_set_ids

    report = {
        "seed": seed,
        "complete": False,
        "baseline_replay_exact": False,
        "query_count": 0,
        "responses": [],
        "baseline_observations": [],
        "batch_work": [],
        "head_seconds": 0.0,
        "set_scoring_seconds": 0.0,
        "cleanup_errors": [],
        "product_token_ids": _COLUMNS,
    }
    runtime = None
    started = time.perf_counter()
    try:
        _modules(runtime_root)
        old, pair = reference["offline"], reference["pair"]
        config = RootConfig.model_validate(old["config"])
        runtime = load_inference_runtime(
            config,
            root / f"runs/national_dex_continuation_control_s{seed}_v1/checkpoint-final.pt",
            corpus=(root / config.data.corpus_manifest).parent,
            graph_db=root / config.data.graph_db,
            protocol=load_protocol(runtime_root / "configs/protocol/pokemon_v1.yaml"),
        )
        _modules(runtime_root)
        _require(
            runtime.device == "cuda"
            and not runtime.model.training
            and all(p.dtype == torch.float32 for p in runtime.model.parameters()),
            "FP32 evaluation runtime",
        )
        _require(
            runtime.checkpoint_hash == old["checkpoint_hash"]
            and runtime.training_identity == old["training_identity"]
            and runtime.corpus_identity == old["corpus_identity"]
            and runtime.split.split_hash == _SPLIT,
            "runtime identity",
        )
        report.update(
            config=old["config"],
            checkpoint_hash=runtime.checkpoint_hash,
            training_identity=runtime.training_identity,
            corpus_identity=runtime.corpus_identity,
            split_hash=runtime.split.split_hash,
            numerical_settings=_settings(torch),
        )
        records = runtime.split.validation
        _require(len(records) == 222, "validation coverage")
        torch.cuda.reset_peak_memory_stats()
        for offset in range(0, 222, 8):
            batch = records[offset : offset + 8]
            queries = [(r.subject, r.dimension) for r in batch]
            prompts = [list(prompt_ids(*q, runtime.vocabulary)) for q in queries]
            generated = []
            report["failed_observation"] = {
                "phase": "baseline",
                "offset": offset,
                "prompts": prompts,
                "rank_outputs": [],
            }
            with torch.inference_mode():
                for rank in range(1, 5):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    paths = _generate_rank(
                        runtime.model,
                        runtime.vocabulary,
                        prompts,
                        rank,
                        max_new_tokens=507,
                        device=runtime.device,
                        alpha=16.0,
                        decoding=_BASE,
                    )
                    torch.cuda.synchronize()
                    report["batch_work"].append(
                        _work(paths, offset, rank, time.perf_counter() - tick)
                    )
                    generated.append(paths)
                    report["failed_observation"]["rank_outputs"].append(
                        [_json(asdict(p)) for p in paths]
                    )
                torch.cuda.synchronize()
                tick = time.perf_counter()
                head = runtime.model(
                    torch.tensor(prompts, dtype=torch.long, device=runtime.device)
                ).symmetric_relation_logits
                _require(
                    head is not None
                    and head.dtype == torch.float32
                    and tuple(head.shape) == (len(batch), 1025),
                    "head shape/dtype",
                )
                vectors = head.cpu().tolist()
                torch.cuda.synchronize()
                report["head_seconds"] += time.perf_counter() - tick
            report["failed_observation"]["symmetric_relation_logits"] = vectors
            for local, record in enumerate(batch):
                index = offset + local
                ref = old["responses"][index]
                _require(
                    (record.subject, record.dimension, sorted(record.targets))
                    == (ref["subject"], ref["dimension"], sorted(ref["expected"])),
                    "validation query order/truth",
                )
                tick = time.perf_counter()
                _sources(tuple(g[local] for g in generated), prompts[local], runtime.vocabulary)
                composition = _compose(
                    tuple(g[local] for g in generated),
                    vectors[local],
                    runtime.vocabulary,
                    queries[local],
                    _candidate_sets_and_scores,
                    _score_set_ids,
                )
                report["set_scoring_seconds"] += time.perf_counter() - tick
                _legacy(composition, vectors[local], ref, pair["responses"][index])
                report["baseline_observations"].append(
                    {
                        "index": index,
                        "subject": record.subject,
                        "dimension": record.dimension,
                        "group": ref["group"],
                        "prompt_ids": prompts[local],
                        "expected": sorted(record.targets),
                        "expected_set_ids": sorted(runtime.vocabulary.encode(record.targets)),
                        "symmetric_relation_logits": vectors[local],
                        "composition": composition,
                    }
                )
            del report["failed_observation"]
            print(f"seed {seed} legacy replay {offset + len(batch)}/222", flush=True)
        baseline_path = output.parent / f"baseline-{seed}.json"
        _write(
            baseline_path,
            {
                "seed": seed,
                "exact_replay": True,
                "responses": report["baseline_observations"],
                "batch_work": report["batch_work"],
                "head_seconds": report["head_seconds"],
                "product_token_ids": _COLUMNS,
            },
        )
        report["baseline_receipt"] = {"path": str(baseline_path), "sha256": _sha(baseline_path)}
        report["baseline_replay_exact"] = True
        for offset in range(0, 222, 8):
            bases = report["baseline_observations"][offset : offset + 8]
            prompts = [b["prompt_ids"] for b in bases]
            generated = []
            report["failed_observation"] = {
                "phase": "wide",
                "offset": offset,
                "prompts": prompts,
                "rank_outputs": [],
                "baseline_sha256": report["baseline_receipt"]["sha256"],
            }
            with torch.inference_mode():
                for rank in range(5, 9):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    paths = _generate_rank(
                        runtime.model,
                        runtime.vocabulary,
                        prompts,
                        rank,
                        max_new_tokens=507,
                        device=runtime.device,
                        alpha=16.0,
                        decoding=_BASE,
                    )
                    torch.cuda.synchronize()
                    report["batch_work"].append(
                        _work(paths, offset, rank, time.perf_counter() - tick)
                    )
                    generated.append(paths)
                    report["failed_observation"]["rank_outputs"].append(
                        [_json(asdict(p)) for p in paths]
                    )
            for local, base in enumerate(bases):
                previous = tuple(
                    GenerationResult(
                        **{**p, "token_ids": tuple(p["token_ids"]), "targets": tuple(p["targets"])}
                    )
                    for p in base["composition"]["source_paths"]
                )
                paths = previous + tuple(g[local] for g in generated)
                tick = time.perf_counter()
                _sources(paths, base["prompt_ids"], runtime.vocabulary)
                wide = _compose(
                    paths,
                    base["symmetric_relation_logits"],
                    runtime.vocabulary,
                    (base["subject"], base["dimension"]),
                    _candidate_sets_and_scores,
                    _score_set_ids,
                )
                report["set_scoring_seconds"] += time.perf_counter() - tick
                _require(
                    wide["slots"][:10] == base["composition"]["slots"],
                    "legacy slots changed in wider pool",
                )
                report["responses"].append(_row(base, wide))
                report["query_count"] = len(report["responses"])
            del report["failed_observation"]
            print(f"seed {seed} wide search {report['query_count']}/222", flush=True)
        report.update(_aggregate(report["responses"]))
        _require(
            report["overall"]["baseline"]["exact_count"] == old["metrics"]["exact_count"]
            and report["overall"]["baseline"]["f1"] == old["metrics"]["f1"],
            "historical baseline metric arithmetic",
        )
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["generation_seconds"] = {
            name: math.fsum(
                w["wall_seconds"]
                for w in report["batch_work"]
                if (w["rank"] <= 4) == (name == "ranks1_4")
            )
            for name in ("ranks1_4", "ranks5_8")
        }
        try:
            report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if runtime is not None:
                del runtime
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as exc:
            report["cleanup_errors"].append(f"{type(exc).__name__}: {exc}")
            report["complete"] = False
        _write(output.parent / f"seed-{seed}.json", _failure_safe(report))
    _require(report["complete"], "seed cleanup failed")
    return report


def _execute(root, output, context, summary):
    started = time.perf_counter()
    try:
        _modules()
        runtime = output.parent / "runtime"
        summary["runtime_files_sha256"] = _extract(context["members"], runtime)
        sys.path.insert(0, str(runtime / "src"))
        import torch

        summary["environment"] = _environment(torch)
        summary["numerical_settings"] = _settings(torch)
        reports = []
        for seed in _SEEDS:
            report = _seed(root, output, seed, context["references"][seed], runtime)
            reports.append(report)
            path = output.parent / f"seed-{seed}.json"
            summary["reports"].append({"seed": seed, "path": str(path), "sha256": _sha(path)})
            _modules(runtime)
        rows = [r for report in reports for r in report["responses"]]
        summary.update(_aggregate(rows))
        _require(
            summary["overall"]["baseline"]["exact_count"] == 569
            and summary["overall"]["baseline"]["f1"] == 0.9668960529174291
            and summary["overall"]["baseline"]["exact_available"] == 570,
            "pooled historical baseline",
        )
        summary["gate"] = _gate(reports, summary)
        summary["changed_queries"] = [
            {
                "seed": report["seed"],
                "index": row["index"],
                "subject": row["subject"],
                "dimension": row["dimension"],
                "gained_exact": row["gained_exact"],
                "lost_exact": row["lost_exact"],
            }
            for report in reports
            for row in report["responses"]
            if row["gained_exact"] or row["lost_exact"]
        ]
        summary["accounting"] = {
            "query_seed_observations": len(rows),
            "distinct_queries": 222,
            "source_paths": sum(len(r["composition"]["source_paths"]) for r in rows),
            "candidate_slots": sum(len(r["composition"]["slots"]) for r in rows),
        }
        summary["complete"] = True
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["retained_partial_reports"] = {
            str(p): _sha(p) for p in output.parent.glob("seed-*.json")
        }
        raise
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        try:
            _unchanged(summary["input_sha256"])
            _unchanged(summary["snapshot_sha256"])
            if "runtime_files_sha256" in summary:
                _runtime_unchanged(output.parent / "runtime", summary["runtime_files_sha256"])
                summary["imported_modules"] = _modules(output.parent / "runtime")
            summary["final_identity_check"] = True
        except BaseException as exc:
            summary["complete"] = False
            summary["final_identity_check"] = False
            summary["identity_error"] = f"{type(exc).__name__}: {exc}"
        _write(output, summary)
    _require(summary["complete"], "campaign final identity check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("docs/experiments/2026-09-25-wide-first-choice-plan.md")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/wide-first-choice-v1/summary.json")
    )
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    script = Path(__file__).resolve()
    content = script.read_bytes()
    script_sha = hashlib.sha256(content).hexdigest()
    root = script.parent.parent
    _require(Path.cwd().resolve() == root, "run from repository root")
    _require("torch" not in sys.modules, "standalone Torch-free preflight required")
    _modules()
    output = args.out.resolve()
    _refuse(output)
    context = _preflight(root, args.plan.resolve())
    context["inputs"][str(script)] = script_sha
    _unchanged(context["inputs"])
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "torch_imported": "torch" in sys.modules,
                    "input_count": len(context["inputs"]),
                    "script_sha256": script_sha,
                }
            )
        )
        return
    _require(
        args.test_receipt is not None and args.test_receipt_sha256,
        "execution requires pinned focused-test receipt",
    )
    receipt_path = _bind(args.test_receipt, args.test_receipt_sha256, context["inputs"])
    receipt = _read(receipt_path)
    _require(
        receipt["passed"] is True
        and receipt["tested_script_sha256"] == script_sha
        and receipt["gpu_used"] is False,
        "focused test receipt contract",
    )
    stdout = _bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"], context["inputs"])
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": content,
        "plan.md": args.plan.read_bytes(),
        "source.zip": context["source"].read_bytes(),
        "configs.zip": context["configs"].read_bytes(),
        "test-receipt.json": receipt_path.read_bytes(),
        "test-stdout.txt": stdout.read_bytes(),
        "inputs.json": (json.dumps(context["inputs"], sort_keys=True, indent=2) + "\n").encode(),
    }
    snapshot_hashes = {}
    for suffix, data in snapshots.items():
        path = output.with_suffix("." + suffix)
        with path.open("xb") as stream:
            stream.write(data)
        snapshot_hashes[str(path)] = hashlib.sha256(data).hexdigest()
    summary = {
        "campaign_version": "wide-first-choice-v1",
        "complete": False,
        "acceptance": False,
        "plan_sha256": _PLAN,
        "script_sha256": script_sha,
        "input_sha256": context["inputs"],
        "snapshot_sha256": snapshot_hashes,
        "reports": [],
        "training_executed": False,
        "protected_test_used": False,
        "limitations": [
            "Saved-output audit does not independently reproduce neural forwards "
            "or first-logit ordering.",
            "Quality/work experiment; no serving throughput, energy or concurrency claim.",
            "222 distinct validation queries, three seeds; no policy promotion.",
            "Executed archived runtime differs from unrelated current workspace source.",
        ],
    }
    _execute(root, output, context, summary)


if __name__ == "__main__":
    main()
