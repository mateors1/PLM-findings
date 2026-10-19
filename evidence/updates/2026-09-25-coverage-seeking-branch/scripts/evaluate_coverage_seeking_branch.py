"""One fixed truth-free coverage anchor, with exact historical continuation controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

_PLAN = "4d718ca6b19d77d8387ef7b90089e2692058bfc99577ac9affcafe765e5a1551"
_HELPER = "2ab48674b526ec6641a6c2366b669ac76bdeac1e8633331e01c772d4393edd39"
_SUMMARY = "1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183"
_AUDIT = "eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95"
_DECISION = "70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601"
_DIAG = "47bf9d54967009a2be6e46bedc28f7949231be9bc14e18798fd66ad776c59805"
_DIAG_AUDIT = "dc4bda47ac580f2c49d8cdfdd4e8cecc6c17d75c421ba970a1383268403ee16a"
_DIAG_DECISION = "ffd804c63618d79f1de986305bf31ec7cc9699a0bb965abbc1399af52261c422"
_RANK_SOURCE = "b5fc41463695c6758ff98471e6c1af5a2cb5ddfb7ca9a85a545b6616908629da"
_SEEDS = (1729, 1730, 1731)
_GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
_FORCED = "forced-first-explicit-greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1"
_POLICY = "first8-pair28-plus-positive-uncovered-anchor9-unions8-v1"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _load_helper(root):
    path = root / "runs/learning/wide-first-choice-v1/summary.script.py"
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == _HELPER, "research helper identity")
    spec = importlib.util.spec_from_file_location("authenticated_width8_helper", path)
    _require(spec is not None and spec.loader is not None, "helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _preflight(root, plan, helper, helper_path):
    context = helper._preflight(
        root, root / "docs/experiments/2026-09-25-wide-first-choice-plan.md"
    )
    inputs = context["inputs"]
    helper._bind(helper_path, _HELPER, inputs)
    helper._bind(plan, _PLAN, inputs)
    reports = {}
    for directory, summary_sha, audit_sha, decision_sha in (
        ("wide-first-choice-v1", _SUMMARY, _AUDIT, _DECISION),
        ("missing-source-membership-v1", _DIAG, _DIAG_AUDIT, _DIAG_DECISION),
    ):
        base = root / "runs/learning" / directory
        summary = helper._read(helper._bind(base / "summary.json", summary_sha, inputs))
        audit = helper._read(helper._bind(base / "independent-audit.json", audit_sha, inputs))
        decision = helper._read(helper._bind(base / "decision.json", decision_sha, inputs))
        _require(
            summary["complete"] is True
            and summary["final_identity_check"] is True
            and audit["complete"] is True
            and audit["audit_passed"] is True
            and audit["summary_sha256"] == decision["summary_sha256"] == summary_sha
            and decision["evidence_accepted"] is True
            and decision["audit_sha256"] == audit_sha
            and decision["auditor_sha256"] == audit["script_sha256"],
            "accepted prior chain",
        )
        helper._bind(base / "independent-audit.py", audit["script_sha256"], inputs)
        if directory == "wide-first-choice-v1":
            _require(decision["fixed_quality_gate_passed"] is True, "accepted width8 quality")
            for ref in summary["reports"]:
                seed = ref["seed"]
                _require(seed in _SEEDS and seed not in reports, "seed receipt")
                report = helper._read(helper._bind(ref["path"], ref["sha256"], inputs))
                _require(
                    report["complete"] is True
                    and report["query_count"] == 222
                    and report["product_token_ids"] == helper._COLUMNS,
                    "baseline coverage",
                )
                reports[seed] = report
    _require(set(reports) == set(_SEEDS), "three baseline seeds")
    _require(
        hashlib.sha256(context["members"]["src/plm/serving/set_reranking.py"]).hexdigest()
        == _RANK_SOURCE,
        "archived rank-loop identity",
    )
    context["wide_reports"] = reports
    return context


def _baseline(row, helper):
    prompt, composition = row["prompt_ids"], row["composition"]
    _require(
        len(prompt) == 5
        and all(type(i) is int for i in prompt)
        and prompt[0] == 1
        and prompt[1] in helper._COLUMNS
        and prompt[3:] == [34, 5]
        and prompt[2] == {"TYPE": 32, "COLOR": 33}.get(row["dimension"]),
        "baseline prompt",
    )
    helper._fp32(row["symmetric_relation_logits"])
    paths = composition["source_paths"]
    _require(len(paths) == 8 and len(composition["slots"]) == 36, "baseline pool width")
    sets = []
    first_ids = []
    for rank, path in enumerate(paths, 1):
        ids = path["token_ids"]
        _require(
            path["protocol_valid"] is True
            and path["terminated"] is True
            and path["error"] is None
            and 7 <= len(ids) <= 512
            and all(type(i) is int for i in ids)
            and ids[:5] == prompt
            and ids[-1] == 2
            and all(i in helper._COLUMNS for i in ids[5:-1])
            and len(ids[5:-1]) == len(set(ids[5:-1])) == len(path["targets"])
            and len(path["targets"]) == len(set(path["targets"]))
            and path["decoding"] == helper._BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            "baseline raw path",
        )
        first_ids.append(ids[5])
        sets.append(sorted(set(ids[5:-1]) - {prompt[1]}))
    _require(len(set(first_ids)) == 8, "baseline distinct first IDs")
    for index, (slot, ranks) in enumerate(
        zip(composition["slots"], helper._ORDER36, strict=True), 1
    ):
        members = sorted(set().union(*(sets[r - 1] for r in ranks)))
        expected = {
            "slot": index,
            "kind": "original" if len(ranks) == 1 else "pair_composition",
            "source_ranks": list(ranks),
            "set_ids": members,
            "source_eligible": True,
            "score": math.fsum(row["symmetric_relation_logits"][i - 1024] for i in members),
        }
        _require(slot == expected, "baseline canonical slot/score")
    _require(
        _selection(composition["slots"])
        == {k: composition[k] for k in _selection(composition["slots"])},
        "baseline selection",
    )
    truth = row["expected_set_ids"]
    _require(
        truth == sorted(set(truth))
        and truth
        and all(type(i) is int and i in helper._COLUMNS for i in truth)
        and prompt[1] not in truth
        and row["group"] in _GROUPS,
        "baseline truth/group",
    )
    _require(helper._metrics(composition, truth) == row["metrics"]["wide"], "baseline metrics")
    return sets


def _prepare(context, helper):
    previous = None
    for seed in _SEEDS:
        rows = context["wide_reports"][seed]["responses"]
        _require(len(rows) == 222, "baseline row coverage")
        keys = []
        for index, row in enumerate(rows):
            _require(row["index"] == index, "baseline order")
            _baseline(row, helper)
            keys.append((row["subject"], row["dimension"], row["group"], row["expected_set_ids"]))
        _require(
            len({(k[0], k[1]) for k in keys}) == 222 and (previous is None or previous == keys),
            "cross-seed query order",
        )
        previous = keys
    rows = [r for seed in _SEEDS for r in context["wide_reports"][seed]["responses"]]
    _require(
        sum(r["metrics"]["wide"]["exact"] for r in rows) == 603
        and math.fsum(r["metrics"]["wide"]["f1"] for r in rows) / 666 == 0.9834927532993669
        and sum(r["metrics"]["wide"]["exact"] for r in rows if r["group"] == "TYPE_dual") == 155,
        "baseline fixed aggregates",
    )


def _anchor(source_sets, logits, subject):
    union = set().union(*(set(s) for s in source_sets))
    choices = [
        i for i in range(1024, 2049) if i != subject and i not in union and logits[i - 1024] > 0
    ]
    token = min(choices, key=lambda i: (-logits[i - 1024], i)) if choices else None
    return {
        "active": token is not None,
        "token_id": token,
        "logit": logits[token - 1024] if token is not None else None,
        "uncovered_positive_count": len(choices),
    }


def _generate_forced_first(
    model, vocabulary, prompts, first_ids, *, max_new_tokens, device, guide, make_result
):
    """Archived _generate_rank loop; only step-zero selection accepts explicit IDs.

    The prefill guidance call remains for identical control execution, although
    its ranking does not choose the forced entity. No later guidance is applied.
    """
    import torch

    _require(
        prompts and len(prompts) == len(first_ids) and all(len(p) == 5 for p in prompts),
        "forced batch shape",
    )
    _require(
        type(max_new_tokens) is int
        and max_new_tokens >= 1
        and 5 + max_new_tokens <= model.config.max_seq_len,
        "forced completion budget",
    )
    entities = list(vocabulary.entity_ids())
    _require(all(type(i) is int and i in entities for i in first_ids), "forced first product IDs")
    prefixes = [list(prompt) for prompt in prompts]
    inputs = torch.tensor(prefixes, dtype=torch.long, device=device)
    count = len(prefixes)
    forced = torch.tensor(first_ids, dtype=torch.long, device=device)
    allowed = torch.zeros(len(vocabulary), dtype=torch.bool, device=device)
    allowed[entities] = True
    seen = torch.zeros(count, len(vocabulary), dtype=torch.bool, device=device)
    active = torch.ones(count, dtype=torch.bool, device=device)
    finished = [False] * count
    cache = None
    for step in range(max_new_tokens):
        output = model.decode(inputs, cache=cache)
        if step == 0:
            output = guide(model, inputs, output, 16.0)
        cache = output.cache
        _require(
            output.logits.shape == (count, inputs.shape[1], len(vocabulary)), "forced logits shape"
        )
        logits = output.logits[:, -1]
        _require(bool(torch.isfinite(logits[active]).all()), "finite active forced logits")
        if step:
            allowed[vocabulary.eos_id] = True
        masked = logits.masked_fill(~allowed[None, :], -float("inf"))
        masked = masked.masked_fill(seen, -float("inf"))
        next_ids = forced if step == 0 else masked.argmax(dim=-1)
        next_ids = torch.where(active, next_ids, vocabulary.eos_id)
        for row, token in enumerate(next_ids.tolist()):
            if not finished[row]:
                prefixes[row].append(token)
                finished[row] = token == vocabulary.eos_id
        if all(finished):
            break
        seen.scatter_(1, next_ids[:, None], True)
        seen[:, vocabulary.eos_id] = False
        active = active & (next_ids != vocabulary.eos_id)
        inputs = next_ids[:, None]
    return tuple(
        make_result(prefix, vocabulary, done, _FORCED)
        for prefix, done in zip(prefixes, finished, strict=True)
    )


def _selection(slots):
    eligible = [s for s in slots if s["source_eligible"]]
    chosen = min(eligible, key=lambda s: (-s["score"], s["slot"])) if eligible else slots[0]
    return {
        **{
            f"selected_{k}": chosen[k]
            for k in ("slot", "kind", "source_ranks", "set_ids", "source_eligible")
        },
        "fallback_no_valid_source": not eligible,
    }


def _extend(baseline, anchor, new_path, new_ids, eligible, logits):
    slots, paths = list(baseline["slots"]), list(baseline["source_paths"])
    if anchor["active"]:
        old_union = set().union(*(set(slots[i]["set_ids"]) for i in (0, 1, 2, 3, 10, 11, 12, 13)))
        _require(anchor["token_id"] not in old_union, "anchor outside old coverage")
        if eligible:
            _require(
                anchor["token_id"] in new_ids and set(new_ids) - old_union,
                "valid source contains anchor and expands coverage",
            )
        paths.append(new_path)
        old_sets = [slots[i]["set_ids"] for i in (0, 1, 2, 3, 10, 11, 12, 13)]
        for ranks in ((9,), *((i, 9) for i in range(1, 9))):
            ids = sorted(set(new_ids) | (set(old_sets[ranks[0] - 1]) if len(ranks) == 2 else set()))
            slots.append(
                {
                    "slot": len(slots) + 1,
                    "kind": "original" if len(ranks) == 1 else "pair_composition",
                    "source_ranks": list(ranks),
                    "set_ids": ids,
                    "source_eligible": eligible,
                    "score": math.fsum(logits[i - 1024] for i in ids),
                }
            )
    return {"source_paths": paths, "slots": slots, **_selection(slots), "policy": _POLICY}


def _row(raw, control, anchor, new_path, new_ids, eligible, helper):
    baseline, truth = raw["composition"], set(raw["expected_set_ids"])
    composition = _extend(
        baseline, anchor, new_path, new_ids, eligible, raw["symmetric_relation_logits"]
    )
    old_union = set().union(
        *(set(baseline["slots"][i]["set_ids"]) for i in (0, 1, 2, 3, 10, 11, 12, 13))
    )
    added = set(new_ids) - old_union if anchor["active"] and eligible else set()
    metrics = {
        "baseline": helper._metrics(baseline, truth),
        "new": helper._metrics(composition, truth),
    }
    return {
        **{
            k: raw[k]
            for k in (
                "index",
                "subject",
                "dimension",
                "group",
                "prompt_ids",
                "expected",
                "expected_set_ids",
                "symmetric_relation_logits",
            )
        },
        "baseline": baseline,
        "control_path": control,
        "anchor": anchor,
        "forced_first_id": anchor["token_id"]
        if anchor["active"]
        else baseline["source_paths"][0]["token_ids"][5],
        "new_path": new_path,
        "new_source_set_ids": new_ids,
        "new_source_eligible": eligible,
        "composition": composition,
        "metrics": metrics,
        "availability": {
            "baseline": helper._availability(baseline, sorted(truth)),
            "new": helper._availability(composition, sorted(truth)),
        },
        "anchor_correct": anchor["token_id"] in truth if anchor["active"] else None,
        "newly_covered_true_ids": sorted(added & truth),
        "newly_covered_false_ids": sorted(added - truth),
        "false_positive_ids": sorted(set(composition["selected_set_ids"]) - truth),
        "false_negative_ids": sorted(truth - set(composition["selected_set_ids"])),
        "gained_exact": metrics["new"]["exact"] and not metrics["baseline"]["exact"],
        "lost_exact": metrics["baseline"]["exact"] and not metrics["new"]["exact"],
    }


def _totals(rows):
    result = {
        "query_count": len(rows),
        "active_anchors": sum(r["anchor"]["active"] for r in rows),
        "inactive_anchors": sum(not r["anchor"]["active"] for r in rows),
        "correct_anchors": sum(r["anchor_correct"] is True for r in rows),
        "false_anchors": sum(r["anchor_correct"] is False for r in rows),
        "newly_covered_true_occurrences": sum(len(r["newly_covered_true_ids"]) for r in rows),
        "newly_covered_false_occurrences": sum(len(r["newly_covered_false_ids"]) for r in rows),
        "false_positive_occurrences": sum(len(r["false_positive_ids"]) for r in rows),
        "false_negative_occurrences": sum(len(r["false_negative_ids"]) for r in rows),
        "gains": sum(r["gained_exact"] for r in rows),
        "losses": sum(r["lost_exact"] for r in rows),
    }
    for arm, key in (("baseline", "baseline"), ("new", "composition")):
        result[arm] = {
            "exact_count": sum(r["metrics"][arm]["exact"] for r in rows),
            **{
                m: math.fsum(r["metrics"][arm][m] for r in rows) / len(rows)
                for m in ("precision", "recall", "f1", "set_size")
            },
            **{
                m: sum(r["availability"][arm][m] for r in rows)
                for m in ("exact_available", "available_exact_miss", "source_union_contains_truth")
            },
            "added_slot_selections": sum(r[key]["selected_slot"] > 36 for r in rows),
            "failures": sum(not r["metrics"][arm]["success"] for r in rows),
        }
    return result


def _aggregate(rows):
    return {
        "overall": _totals(rows),
        "groups": {g: _totals([r for r in rows if r["group"] == g]) for g in _GROUPS},
    }


def _gate(reports, pooled):
    checks = {
        "complete_exact_replay": len(reports) == 3
        and all(r["complete"] and r["replay_exact"] and r["query_count"] == 222 for r in reports),
        "active_sources_valid": all(
            r["new_source_eligible"]
            for report in reports
            for r in report["responses"]
            if r["anchor"]["active"]
        ),
        "all_selected_eligible": pooled["overall"]["new"]["failures"] == 0,
        "per_seed_exact_nonregression": all(
            r["overall"]["new"]["exact_count"] >= r["overall"]["baseline"]["exact_count"]
            for r in reports
        ),
        "per_seed_f1_nonregression": all(
            r["overall"]["new"]["f1"] >= r["overall"]["baseline"]["f1"] for r in reports
        ),
        "pooled_exact_improvement": pooled["overall"]["new"]["exact_count"] > 603,
        "pooled_dual_improvement": pooled["groups"]["TYPE_dual"]["new"]["exact_count"] > 155,
        "pooled_group_nonregression": all(
            g["new"]["exact_count"] >= g["baseline"]["exact_count"]
            for g in pooled["groups"].values()
        ),
    }
    return {"checks": checks, "quality_passed": all(checks.values())}


def _work(paths, active, offset, phase, seconds):
    lengths = [len(p.token_ids) - 5 for p in paths]
    calls = max(lengths)
    count = sum(active)
    return {
        "phase": phase,
        "offset": offset,
        "batch_size": len(paths),
        "decode_calls": calls,
        "wall_seconds": seconds,
        "padded_decode_positions": len(paths) * (5 + calls - 1),
        "total_emitted_tokens": sum(lengths),
        "useful_emitted_tokens": sum(n for n, flag in zip(lengths, active, strict=True) if flag),
        "guidance_forwards": 1,
        "active_rows": count,
        "padding_rows": len(paths) - count,
        "active_decode_positions": count * (5 + calls - 1),
        "padding_decode_positions": (len(paths) - count) * (5 + calls - 1),
        "active_emitted_tokens": sum(n for n, flag in zip(lengths, active, strict=True) if flag),
        "padding_emitted_tokens": sum(
            n for n, flag in zip(lengths, active, strict=True) if not flag
        ),
    }


def _seed(root, output, seed, context, helper, runtime_root):
    import torch

    from plm.config import RootConfig
    from plm.protocol import load_protocol
    from plm.serving.generation import _generation_result
    from plm.serving.guidance import apply_first_target_guidance
    from plm.serving.runtime import load_inference_runtime
    from plm.serving.set_reranking import _candidate_sets_and_scores

    runtime = None
    report = {
        "seed": seed,
        "complete": False,
        "replay_exact": False,
        "query_count": 0,
        "responses": [],
        "controls": [],
        "work": [],
        "head_seconds": 0.0,
        "set_scoring_seconds": 0.0,
        "cleanup_errors": [],
    }
    started = time.perf_counter()
    try:
        old = context["wide_reports"][seed]
        config = RootConfig.model_validate(old["config"])
        runtime = load_inference_runtime(
            config,
            root / f"runs/national_dex_continuation_control_s{seed}_v1/checkpoint-final.pt",
            corpus=(root / config.data.corpus_manifest).parent,
            graph_db=root / config.data.graph_db,
            protocol=load_protocol(runtime_root / "configs/protocol/pokemon_v1.yaml"),
        )
        helper._modules(runtime_root)
        _require(
            runtime.device == "cuda"
            and not runtime.model.training
            and all(p.dtype == torch.float32 for p in runtime.model.parameters()),
            "FP32 runtime",
        )
        _require(
            runtime.checkpoint_hash == old["checkpoint_hash"]
            and runtime.training_identity == old["training_identity"]
            and runtime.corpus_identity == old["corpus_identity"]
            and runtime.split.split_hash == old["split_hash"],
            "runtime identity",
        )
        _require(
            [(r.subject, r.dimension, sorted(r.targets)) for r in runtime.split.validation]
            == [(r["subject"], r["dimension"], r["expected"]) for r in old["responses"]],
            "runtime validation alignment",
        )
        report.update(
            {
                k: old[k]
                for k in (
                    "config",
                    "checkpoint_hash",
                    "training_identity",
                    "corpus_identity",
                    "split_hash",
                )
            }
        )
        report["numerical_settings"] = helper._settings(torch)
        torch.cuda.reset_peak_memory_stats()
        for offset in range(0, 222, 8):
            batch = old["responses"][offset : offset + 8]
            prompts = [r["prompt_ids"] for r in batch]
            report["failed_observation"] = {"phase": "control", "offset": offset}
            with torch.inference_mode():
                torch.cuda.synchronize()
                tick = time.perf_counter()
                head = runtime.model(
                    torch.tensor(prompts, dtype=torch.long, device=runtime.device)
                ).symmetric_relation_logits
                _require(
                    head is not None
                    and head.dtype == torch.float32
                    and tuple(head.shape) == (len(batch), 1025),
                    "head tensor shape/dtype",
                )
                vectors = head.cpu().tolist()
                torch.cuda.synchronize()
                report["head_seconds"] += time.perf_counter() - tick
                report["failed_observation"]["fresh_logits"] = vectors
                _require(
                    vectors == [r["symmetric_relation_logits"] for r in batch],
                    "exact fresh head replay",
                )
                tick = time.perf_counter()
                paths = _generate_forced_first(
                    runtime.model,
                    runtime.vocabulary,
                    prompts,
                    [r["composition"]["source_paths"][0]["token_ids"][5] for r in batch],
                    max_new_tokens=507,
                    device=runtime.device,
                    guide=apply_first_target_guidance,
                    make_result=_generation_result,
                )
                torch.cuda.synchronize()
                report["work"].append(
                    _work(paths, [True] * len(batch), offset, "control", time.perf_counter() - tick)
                )
            raw_paths = [helper._json(asdict(p)) for p in paths]
            report["failed_observation"]["control_paths"] = raw_paths
            for i, (raw, path) in enumerate(zip(batch, raw_paths, strict=True), offset):
                prior = raw["composition"]["source_paths"][0]
                _require(
                    {k: v for k, v in path.items() if k != "decoding"}
                    == {k: v for k, v in prior.items() if k != "decoding"},
                    "exact forced-first control replay",
                )
                report["controls"].append(
                    {
                        "index": i,
                        "symmetric_relation_logits": vectors[i - offset],
                        "control_path": path,
                    }
                )
            del report["failed_observation"]
            print(f"seed {seed}: controls {offset + len(batch)}/222", flush=True)
        seal = output.parent / f"controls-{seed}.json"
        helper._write(seal, {"seed": seed, "replay_exact": True, "responses": report["controls"]})
        report["control_receipt"] = {"path": str(seal), "sha256": helper._sha(seal)}
        report["replay_exact"] = True
        for offset in range(0, 222, 8):
            batch = old["responses"][offset : offset + 8]
            sets = [_baseline(r, helper) for r in batch]
            anchors = [
                _anchor(s, r["symmetric_relation_logits"], r["prompt_ids"][1])
                for s, r in zip(sets, batch, strict=True)
            ]
            first_ids = [
                a["token_id"]
                if a["active"]
                else r["composition"]["source_paths"][0]["token_ids"][5]
                for a, r in zip(anchors, batch, strict=True)
            ]
            report["failed_observation"] = {
                "phase": "anchor",
                "offset": offset,
                "anchors": anchors,
                "forced_first_ids": first_ids,
            }
            with torch.inference_mode():
                torch.cuda.synchronize()
                tick = time.perf_counter()
                paths = _generate_forced_first(
                    runtime.model,
                    runtime.vocabulary,
                    [r["prompt_ids"] for r in batch],
                    first_ids,
                    max_new_tokens=507,
                    device=runtime.device,
                    guide=apply_first_target_guidance,
                    make_result=_generation_result,
                )
                torch.cuda.synchronize()
                report["work"].append(
                    _work(
                        paths,
                        [a["active"] for a in anchors],
                        offset,
                        "anchor",
                        time.perf_counter() - tick,
                    )
                )
            report["failed_observation"]["paths"] = [helper._json(asdict(p)) for p in paths]
            for local, (raw, path, anchor) in enumerate(zip(batch, paths, anchors, strict=True)):
                tick = time.perf_counter()
                new_sets, _, eligible = _candidate_sets_and_scores(
                    (path,),
                    raw["symmetric_relation_logits"],
                    runtime.vocabulary,
                    (raw["subject"], raw["dimension"]),
                )
                report["responses"].append(
                    _row(
                        raw,
                        report["controls"][offset + local]["control_path"],
                        anchor,
                        helper._json(asdict(path)),
                        list(new_sets[0]),
                        eligible[0],
                        helper,
                    )
                )
                report["set_scoring_seconds"] += time.perf_counter() - tick
                report["query_count"] = len(report["responses"])
            del report["failed_observation"]
            print(f"seed {seed}: anchors {report['query_count']}/222", flush=True)
        report.update(_aggregate(report["responses"]))
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["generation_seconds"] = {
            phase: math.fsum(w["wall_seconds"] for w in report["work"] if w["phase"] == phase)
            for phase in ("control", "anchor")
        }
        try:
            report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if runtime is not None:
                del runtime
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as exc:
            report["cleanup_errors"].append(str(exc))
            report["complete"] = False
        _write_compact(output.parent / f"seed-{seed}.json", helper._failure_safe(report))
    _require(report["complete"], "seed cleanup failed")
    return report


def _write_compact(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )


def _execute(root, output, context, helper, summary):
    started = time.perf_counter()
    try:
        helper._modules()
        runtime_root = output.parent / "runtime"
        summary["runtime_files_sha256"] = helper._extract(context["members"], runtime_root)
        sys.path.insert(0, str(runtime_root / "src"))
        import torch

        summary["environment"] = helper._environment(torch)
        summary["numerical_settings"] = helper._settings(torch)
        reports = []
        for seed in _SEEDS:
            reports.append(_seed(root, output, seed, context, helper, runtime_root))
            path = output.parent / f"seed-{seed}.json"
            summary["reports"].append(
                {"seed": seed, "path": str(path), "sha256": helper._sha(path)}
            )
            helper._modules(runtime_root)
        rows = [r for report in reports for r in report["responses"]]
        summary.update(_aggregate(rows))
        summary["gate"] = _gate(reports, summary)
        summary["accounting"] = {
            "query_seed_observations": len(rows),
            "distinct_queries": 222,
            "reused_source_paths": 8 * len(rows),
            "fresh_control_paths": len(rows),
            "fresh_active_paths": sum(r["anchor"]["active"] for r in rows),
            "fresh_padding_paths": sum(not r["anchor"]["active"] for r in rows),
            "candidate_slots": sum(len(r["composition"]["slots"]) for r in rows),
        }
        summary["complete"] = True
    except BaseException as exc:
        summary["complete"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["retained_partial_reports"] = {
            str(p): helper._sha(p) for p in output.parent.glob("seed-*.json")
        }
        raise
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        try:
            helper._unchanged(summary["input_sha256"])
            helper._unchanged(summary["snapshot_sha256"])
            if "runtime_files_sha256" in summary:
                helper._runtime_unchanged(
                    output.parent / "runtime", summary["runtime_files_sha256"]
                )
                summary["imported_modules"] = helper._modules(output.parent / "runtime")
            summary["final_identity_check"] = True
        except BaseException as exc:
            summary["complete"] = summary["final_identity_check"] = False
            summary["identity_error"] = str(exc)
        helper._write(output, summary)
    _require(summary["complete"], "final identity check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/experiments/2026-09-25-coverage-seeking-branch-plan.md"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/coverage-seeking-branch-v1/summary.json")
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    args = parser.parse_args()
    script = Path(__file__).resolve()
    content = script.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    root, output = script.parent.parent, args.out.resolve()
    _require(
        Path.cwd().resolve() == root and "torch" not in sys.modules,
        "standalone Torch-free preflight from repository root",
    )
    helper, helper_path = _load_helper(root)
    helper._refuse(output)
    _require(not list(output.parent.glob("controls-*.json")), "immutable control receipt exists")
    context = _preflight(root, args.plan.resolve(), helper, helper_path)
    context["inputs"][str(script)] = digest
    _prepare(context, helper)
    helper._unchanged(context["inputs"])
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "torch_imported": False,
                    "new_branch_measured": False,
                    "script_sha256": digest,
                    "input_count": len(context["inputs"]),
                }
            )
        )
        return
    _require(
        args.test_receipt is not None and args.test_receipt_sha256,
        "pinned focused-test receipt required",
    )
    receipt_path = helper._bind(args.test_receipt, args.test_receipt_sha256, context["inputs"])
    receipt = helper._read(receipt_path)
    _require(
        receipt["passed"] is True
        and receipt["tested_script_sha256"] == digest
        and receipt["gpu_used"] is False,
        "test receipt contract",
    )
    stdout = helper._bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"], context["inputs"])
    helper._bind(receipt["test_source"], receipt["tested_test_sha256"], context["inputs"])
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": content,
        "helper.py": helper_path.read_bytes(),
        "plan.md": args.plan.read_bytes(),
        "source.zip": context["source"].read_bytes(),
        "configs.zip": context["configs"].read_bytes(),
        "test-receipt.json": receipt_path.read_bytes(),
        "test-stdout.txt": stdout.read_bytes(),
        "inputs.json": (json.dumps(context["inputs"], sort_keys=True, indent=2) + "\n").encode(),
    }
    hashes = {}
    for suffix, data in snapshots.items():
        path = output.with_suffix("." + suffix)
        with path.open("xb") as stream:
            stream.write(data)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
    summary = {
        "campaign_version": "coverage-seeking-branch-v1",
        "complete": False,
        "acceptance": False,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "helper_sha256": _HELPER,
        "archived_rank_source_sha256": _RANK_SOURCE,
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "reports": [],
        "training_executed": False,
        "protected_test_used": False,
        "limitations": [
            "Eight historical source paths are reused; "
            "fresh head and forced rank-one controls are replayed.",
            "Saved-output audit is not a second neural run; "
            "forced anchors are not decoder preferences.",
            "Timing is descriptive; no serving throughput, energy or concurrency claim.",
            "LM Studio service is preserved; incidental service use may affect timings.",
        ],
    }
    _execute(root, output, context, helper, summary)


if __name__ == "__main__":
    main()
