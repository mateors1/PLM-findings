"""Saved-only fixed pair-intersection screen and pure-operation oracle diagnosis.

The authenticated upstream audit establishes neural/runtime provenance. This
stdlib program neither loads weights nor independently reruns neural generation.
Oracle witnesses never enter prediction construction, scoring or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import struct
import sys
import time
from pathlib import Path

_PLAN = "7fb6f28e9d432799590fd7dff9b9269ffc732ecce8c4f496ba0c1e3521c27a5f"
_SUMMARY = "1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183"
_AUDIT = "eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95"
_DECISION = "70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601"
_SEEDS = (1729, 1730, 1731)
_GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
_COLUMNS = list(range(1024, 2049))
_PAIRS = tuple(itertools.combinations(range(1, 9), 2))
_ORDER36 = (
    (1,),
    (2,),
    (3,),
    (4,),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (5,),
    (6,),
    (7,),
    (8,),
    *(p for p in _PAIRS if p[1] > 4),
)
_BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"
_STATES = (
    "selected_exact",
    "available_exact_but_missed",
    "unavailable_and_truth_missing_from_all_sources",
    "unavailable_with_truth_covered",
)
_SELECTION = (
    "selected_slot",
    "selected_kind",
    "selected_source_ranks",
    "selected_set_ids",
    "selected_source_eligible",
    "fallback_no_valid_source",
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value, *, compact=False):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                value,
                sort_keys=True,
                indent=None if compact else 2,
                separators=(",", ":") if compact else None,
                allow_nan=False,
            )
            + "\n"
        )


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
        and not list(output.parent.glob("seed-*.json")),
        "immutable output exists",
    )


def _authenticate(root, plan):
    inputs = {}
    _bind(plan, _PLAN, inputs)
    directory = root / "runs/learning/wide-first-choice-v1"
    summary = _read(_bind(directory / "summary.json", _SUMMARY, inputs))
    audit = _read(_bind(directory / "independent-audit.json", _AUDIT, inputs))
    decision = _read(_bind(directory / "decision.json", _DECISION, inputs))
    _require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and decision["evidence_accepted"] is True
        and decision["fixed_quality_gate_passed"] is True
        and audit["summary_sha256"] == decision["summary_sha256"] == _SUMMARY
        and decision["audit_sha256"] == _AUDIT
        and decision["auditor_sha256"] == audit["script_sha256"],
        "accepted upstream chain",
    )
    _bind(directory / "independent-audit.py", audit["script_sha256"], inputs)
    reports = {}
    for receipt in summary["reports"]:
        seed = receipt["seed"]
        _require(seed in _SEEDS and seed not in reports, "upstream seed receipt")
        report = _read(_bind(receipt["path"], receipt["sha256"], inputs))
        _require(
            report["complete"] is True
            and report["seed"] == seed
            and report["query_count"] == len(report["responses"]) == 222
            and report["product_token_ids"] == _COLUMNS,
            "upstream coverage/columns",
        )
        reports[seed] = report
    _require(set(reports) == set(_SEEDS), "three upstream seeds")
    return {"inputs": inputs, "reports": reports}


def _ids(ids, *, nonempty=False):
    _require(
        isinstance(ids, list)
        and all(type(i) is int and i in _COLUMNS for i in ids)
        and ids == sorted(set(ids))
        and (bool(ids) or not nonempty),
        "canonical product IDs",
    )


def _logits(values):
    _require(isinstance(values, list) and len(values) == 1025, "head columns")
    for value in values:
        _require(type(value) in (int, float) and math.isfinite(value), "finite numeric head")
        try:
            rounded = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError("FP32 head range") from exc
        _require(value == rounded, "exact FP32 head")


def _name(token, name, names):
    _require(type(name) is str and name.startswith("PKM_"), "product name")
    _require(token not in names or names[token] == name, "ID/name conflict")
    _require(name not in names.values() or names.get(token) == name, "name/ID conflict")
    names[token] = name


def _sources(row, names):
    prompt, paths = row["prompt_ids"], row["composition"]["source_paths"]
    _require(
        len(prompt) == 5
        and all(type(i) is int for i in prompt)
        and prompt[0] == 1
        and prompt[1] in _COLUMNS
        and prompt[3:] == [34, 5]
        and prompt[2] == {"TYPE": 32, "COLOR": 33}.get(row["dimension"]),
        "prompt grammar",
    )
    _require(
        row["group"] in _GROUPS and (row["group"] == "COLOR") == (row["dimension"] == "COLOR"),
        "dimension/group",
    )
    _name(prompt[1], row["subject"], names)
    _ids(row["expected_set_ids"], nonempty=True)
    _require(prompt[1] not in row["expected_set_ids"], "truth excludes subject")
    _require(row["expected"] == sorted(set(row["expected"])), "canonical expected names")
    _logits(row["symmetric_relation_logits"])
    _require(len(paths) == 8, "eight raw sources")
    sets, removed, first = [], [], []
    for rank, path in enumerate(paths, 1):
        ids = path["token_ids"]
        _require(
            path["protocol_valid"] is True and path["terminated"] is True and path["error"] is None,
            "all raw sources must be valid and terminated",
        )
        _require(
            isinstance(ids, list)
            and 7 <= len(ids) <= 512
            and all(type(i) is int for i in ids)
            and ids[:5] == prompt
            and ids[-1] == 2
            and all(type(i) is int and i in _COLUMNS for i in ids[5:-1]),
            "raw source grammar/mask",
        )
        products = ids[5:-1]
        _require(
            len(products) == len(set(products)) and len(products) == len(path["targets"]),
            "raw source uniqueness/targets",
        )
        _require(
            path["decoding"] == _BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            "source rank descriptor",
        )
        for token, name in zip(products, path["targets"], strict=True):
            _name(token, name, names)
        sets.append(sorted(set(products) - {prompt[1]}))
        removed.append(prompt[1] in products)
        first.append(products[0])
    _require(len(set(first)) == 8, "distinct ranked first products")
    _require(removed == row["subject_removed"], "subject-removal reference")
    return sets, removed


def _slots(sets, logits, *, intersections):
    """Prediction-only arithmetic; no truth, group, expected size or error inputs."""
    slots = []
    orders = [(ranks, False) for ranks in _ORDER36]
    if intersections:
        orders.extend((ranks, True) for ranks in _PAIRS)
    for ranks, intersect in orders:
        members = set(sets[ranks[0] - 1])
        for rank in ranks[1:]:
            if intersect:
                members.intersection_update(sets[rank - 1])
            else:
                members.update(sets[rank - 1])
        ids = sorted(members)
        slots.append(
            {
                "slot": len(slots) + 1,
                "kind": "pair_intersection"
                if intersect
                else "original"
                if len(ranks) == 1
                else "pair_composition",
                "source_ranks": list(ranks),
                "set_ids": ids,
                "source_eligible": True,
                "score": math.fsum(logits[i - 1024] for i in ids),
            }
        )
    return slots


def _select(slots):
    _require(slots and all(s["source_eligible"] is True for s in slots), "eligible fixed pool")
    selected = min(slots, key=lambda s: (-s["score"], s["slot"]))
    return {
        **{
            f"selected_{k}": selected[k]
            for k in ("slot", "kind", "source_ranks", "set_ids", "source_eligible")
        },
        "fallback_no_valid_source": False,
    }


def _metrics(selection, truth):
    predicted, expected = set(selection["selected_set_ids"]), set(truth)
    _require(expected, "nonempty truth")
    matches = len(predicted & expected)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(expected)
    return {
        "exact": predicted == expected,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "set_size": len(predicted),
        "success": True,
    }


def _prepare(reports):
    """Reconstruct only the historical 36-slot baseline; no new experimental sets."""
    prepared, names, identities = {}, {}, None
    for seed in _SEEDS:
        rows = []
        for index, raw in enumerate(reports[seed]["responses"]):
            _require(raw["index"] == index, "ordered query index")
            sets, removed = _sources(raw, names)
            slots = _slots(sets, raw["symmetric_relation_logits"], intersections=False)
            baseline = _select(slots)
            _require(
                slots == raw["composition"]["slots"]
                and baseline == {k: raw["composition"][k] for k in _SELECTION},
                "exact baseline slots/selection",
            )
            metrics = _metrics(baseline, raw["expected_set_ids"])
            _require(metrics == raw["metrics"]["wide"], "exact baseline metrics")
            rows.append(
                {
                    "raw": raw,
                    "source_sets": sets,
                    "removed": removed,
                    "baseline": baseline,
                    "baseline_metrics": metrics,
                    "slots": slots,
                }
            )
        keys = [
            (
                r["raw"]["subject"],
                r["raw"]["dimension"],
                r["raw"]["group"],
                r["raw"]["expected_set_ids"],
            )
            for r in rows
        ]
        _require(
            len(rows) == 222
            and len({(k[0], k[1]) for k in keys}) == 222
            and (identities is None or identities == keys),
            "query alignment/coverage",
        )
        identities = keys
        prepared[seed] = rows
    for rows in prepared.values():
        for row in rows:
            raw = row["raw"]
            _require(
                sorted(names[i] for i in raw["expected_set_ids"]) == raw["expected"],
                "expected ID/name identity",
            )
    all_rows = [r for rows in prepared.values() for r in rows]
    _require(
        sum(r["baseline_metrics"]["exact"] for r in all_rows) == 603
        and math.fsum(r["baseline_metrics"]["f1"] for r in all_rows) / 666 == 0.9834927532993669,
        "fixed historical aggregate reproduction",
    )
    return prepared


def _witnesses(sets, truth):
    expected = set(truth)
    result = {
        name: {"exact_available": False, "min_sources": None, "witness": None}
        for name in ("unions", "intersections")
    }
    for size in range(1, 9):
        for ranks in itertools.combinations(range(1, 9), size):
            union, intersection = set(), set(sets[ranks[0] - 1])
            for rank in ranks:
                union.update(sets[rank - 1])
                intersection.intersection_update(sets[rank - 1])
            for name, value in (("unions", union), ("intersections", intersection)):
                if value == expected and not result[name]["exact_available"]:
                    result[name] = {
                        "exact_available": True,
                        "min_sources": size,
                        "witness": list(ranks),
                    }
    return result


def _errors(ids, truth):
    return {
        "false_positive_ids": sorted(set(ids) - set(truth)),
        "false_negative_ids": sorted(set(truth) - set(ids)),
    }


def _row(prepared):
    raw, sets = prepared["raw"], prepared["source_sets"]
    truth = raw["expected_set_ids"]
    slots = _slots(sets, raw["symmetric_relation_logits"], intersections=True)
    baseline, prediction = prepared["baseline"], _select(slots)
    metrics = {"baseline": prepared["baseline_metrics"], "prediction": _metrics(prediction, truth)}
    witnesses = _witnesses(sets, truth)
    baseline_available = any(s["set_ids"] == truth for s in slots[:36])
    covered = set(truth) <= set().union(*(set(s) for s in sets))
    state = (
        _STATES[0]
        if metrics["baseline"]["exact"]
        else _STATES[1]
        if baseline_available
        else _STATES[3]
        if covered
        else _STATES[2]
    )
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
        "source_paths": raw["composition"]["source_paths"],
        "source_set_ids": sets,
        "source_subject_removed": prepared["removed"],
        "slots": slots,
        "baseline": baseline,
        "prediction": prediction,
        "metrics": metrics,
        "availability": {
            "baseline": baseline_available,
            "policy": any(s["set_ids"] == truth for s in slots),
            "all_unions": witnesses["unions"]["exact_available"],
            "all_intersections": witnesses["intersections"]["exact_available"],
            "either_pure_family": any(w["exact_available"] for w in witnesses.values()),
        },
        "diagnosis": {"baseline_state": state, **witnesses, "source_union_contains_truth": covered},
        "errors": {
            "sources": [_errors(s, truth) for s in sets],
            "baseline": _errors(baseline["selected_set_ids"], truth),
            "prediction": _errors(prediction["selected_set_ids"], truth),
        },
        "gained_exact": metrics["prediction"]["exact"] and not metrics["baseline"]["exact"],
        "lost_exact": metrics["baseline"]["exact"] and not metrics["prediction"]["exact"],
        "selection_changed": baseline["selected_set_ids"] != prediction["selected_set_ids"],
    }


def _totals(rows):
    _require(rows, "nonempty aggregation")
    result = {
        "query_count": len(rows),
        "gains": sum(r["gained_exact"] for r in rows),
        "losses": sum(r["lost_exact"] for r in rows),
        "changed_selection_count": sum(r["selection_changed"] for r in rows),
        "pure_family_availability": {
            key: sum(r["availability"][key] for r in rows)
            for key in ("all_unions", "all_intersections", "either_pure_family")
        },
        "baseline_state_counts": {
            state: sum(r["diagnosis"]["baseline_state"] == state for r in rows) for state in _STATES
        },
    }
    for arm, availability in (("baseline", "baseline"), ("prediction", "policy")):
        result[arm] = {
            "exact_count": sum(r["metrics"][arm]["exact"] for r in rows),
            **{
                key: math.fsum(r["metrics"][arm][key] for r in rows) / len(rows)
                for key in ("precision", "recall", "f1")
            },
            "mean_set_size": math.fsum(r["metrics"][arm]["set_size"] for r in rows) / len(rows),
            "exact_available": sum(r["availability"][availability] for r in rows),
            "available_exact_misses": sum(
                r["availability"][availability] and not r["metrics"][arm]["exact"] for r in rows
            ),
            "intersection_selections": sum(
                r[arm]["selected_kind"] == "pair_intersection" for r in rows
            ),
            "empty_selections": sum(not r[arm]["selected_set_ids"] for r in rows),
        }
    return result


def _aggregate(rows):
    return {
        "overall": _totals(rows),
        "groups": {group: _totals([r for r in rows if r["group"] == group]) for group in _GROUPS},
    }


def _gate(reports, pooled):
    checks = {
        "complete_baseline_reproduction": len(reports) == 3
        and all(
            r["complete"] and r["query_count"] == 222 and r["baseline_reproduction_exact"]
            for r in reports
        ),
        "per_seed_exact_nonregression": all(
            r["overall"]["prediction"]["exact_count"] >= r["overall"]["baseline"]["exact_count"]
            for r in reports
        ),
        "per_seed_f1_nonregression": all(
            r["overall"]["prediction"]["f1"] >= r["overall"]["baseline"]["f1"] for r in reports
        ),
        "pooled_exact_improvement": pooled["overall"]["prediction"]["exact_count"] > 603,
        "pooled_group_exact_nonregression": all(
            g["prediction"]["exact_count"] >= g["baseline"]["exact_count"]
            for g in pooled["groups"].values()
        ),
    }
    return {"checks": checks, "quality_passed": all(checks.values())}


def _execute(output, prepared, summary):
    started = time.perf_counter()
    reports = []
    try:
        for seed in _SEEDS:
            report = {
                "seed": seed,
                "complete": False,
                "query_count": 0,
                "responses": [],
                "baseline_reproduction_exact": True,
                "product_token_ids": _COLUMNS,
            }
            path = output.parent / f"seed-{seed}.json"
            try:
                for item in prepared[seed]:
                    report["failed_query"] = {
                        "index": item["raw"]["index"],
                        "subject": item["raw"]["subject"],
                        "dimension": item["raw"]["dimension"],
                    }
                    report["responses"].append(_row(item))
                    report["query_count"] = len(report["responses"])
                    del report["failed_query"]
                report.update(_aggregate(report["responses"]))
                report["complete"] = True
            except BaseException as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                _write(path, report, compact=True)
                summary["reports"].append({"seed": seed, "path": str(path), "sha256": _sha(path)})
            reports.append(report)
        rows = [row for report in reports for row in report["responses"]]
        summary.update(_aggregate(rows))
        summary["gate"] = _gate(reports, summary)
        states = summary["overall"]["baseline_state_counts"]
        summary["derived_bound"] = {
            "missing_union_count": states[_STATES[2]],
            "existing_available_miss_count": states[_STATES[1]],
            "selected_exact_upper_bound": len(rows) - states[_STATES[2]] - states[_STATES[1]],
            "scope": "appended pure union/intersection sets; unchanged scorer and old-first ties",
        }
        summary["accounting"] = {
            "query_seed_observations": len(rows),
            "distinct_queries": 222,
            "candidate_slots": len(rows) * 64,
            "source_paths": len(rows) * 8,
            "subsets_per_operation_per_query": 255,
            "pure_operation_subsets_examined": len(rows) * 510,
        }
        summary["complete"] = True
    except BaseException as exc:
        summary["complete"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        try:
            _unchanged(summary["input_sha256"])
            _unchanged(summary["snapshot_sha256"])
            summary["final_identity_check"] = True
        except BaseException as exc:
            summary["final_identity_check"] = summary["complete"] = False
            summary["identity_error"] = f"{type(exc).__name__}: {exc}"
        _write(output, summary)
    _require(summary["complete"], "final identity check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/experiments/2026-09-25-eight-source-set-algebra-plan.md"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/eight-source-set-algebra-v1/summary.json")
    )
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    script = Path(__file__).resolve()
    source = script.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    root, output = script.parent.parent, args.out.resolve()
    _require(Path.cwd().resolve() == root, "run from repository root")
    _require("torch" not in sys.modules, "standalone stdlib-only execution required")
    _refuse(output)
    context = _authenticate(root, args.plan.resolve())
    context["inputs"][str(script)] = digest
    prepared = _prepare(context["reports"])
    _unchanged(context["inputs"])
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "baseline_reproduction_exact": True,
                    "new_policy_measured": False,
                    "torch_imported": False,
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
    receipt_path = _bind(args.test_receipt, args.test_receipt_sha256, context["inputs"])
    receipt = _read(receipt_path)
    _require(
        receipt["passed"] is True
        and receipt["tested_script_sha256"] == digest
        and receipt["gpu_used"] is False,
        "focused-test receipt contract",
    )
    stdout = _bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"], context["inputs"])
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": source,
        "plan.md": args.plan.read_bytes(),
        "test-receipt.json": receipt_path.read_bytes(),
        "test-stdout.txt": stdout.read_bytes(),
        "inputs.json": (json.dumps(context["inputs"], sort_keys=True, indent=2) + "\n").encode(),
    }
    hashes = {}
    for suffix, content in snapshots.items():
        path = output.with_suffix("." + suffix)
        with path.open("xb") as stream:
            stream.write(content)
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
    summary = {
        "campaign_version": "eight-source-set-algebra-v1",
        "complete": False,
        "acceptance": False,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "reports": [],
        "training_executed": False,
        "neural_generation_executed": False,
        "protected_test_used": False,
        "limitations": [
            "Upstream authenticated audit establishes neural/runtime provenance; "
            "no independent neural replay.",
            "Oracle witnesses cover pure unions and pure intersections, "
            "not arbitrary nested expressions.",
            "Validation-only fixed screen, no policy promotion; "
            "222 distinct queries under three seeds.",
        ],
    }
    _execute(output, prepared, summary)


if __name__ == "__main__":
    main()
