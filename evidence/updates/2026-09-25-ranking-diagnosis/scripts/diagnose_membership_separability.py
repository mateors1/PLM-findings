"""Diagnose frozen membership rankings with explicitly oracle-assisted ceilings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import struct
import sys
from collections import Counter
from pathlib import Path

_PLAN_SHA = "b189638ba7c89e380934a0239537fd8a8f6a4f34f68058c619fe86fa5e7bca87"
_HELPER_SHA = "4f9baef52a7f365bfafdb46041ede98cf4d04ff27c130e27c053bb4726f4e4c6"
_HELPER_PLAN_SHA = "4004639897b88d54090ce94bfecf26adf5b2a75686be4b8cdf97da2155d19da0"
_DIRECT_SHA = "2cbe82aad296b6ff4a920a951de38a5999a23e0eb8507d40444254e5d404ceab"
_DIRECT_AUDIT_SHA = "8111692647153787e095da94c64ec497ebc41537b6f6c3ea8ad1d5b046b8df42"
_PAIR_SHA = "4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992"
_PAIR_AUDIT_SHA = "13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057"
_SEEDS = (1729, 1730, 1731)
_GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
_IDENTITY = ("checkpoint_hash", "training_identity", "corpus_identity", "split_hash")
_CROSS_FIELDS = (
    "pair_exact",
    "direct_exact",
    "strict_separable",
    "separable_with_wrong_zero_threshold",
    "boundary_tie",
    "strict_rank_overlap",
    "oracle_top_k_exact",
    "source_union_lacks_truth",
)
_FLAGS = (*_CROSS_FIELDS, "oracle_top_k_gained_exact", "oracle_top_k_lost_exact")
_CASES = ("mixed", "empty_truth", "full_truth", "empty_universe")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _bind(path, digest, inputs):
    path = Path(path).resolve()
    _require(path.is_file() and _sha(path) == digest, f"identity mismatch: {path}")
    _require(
        str(path) not in inputs or inputs[str(path)] == digest, f"conflicting identity: {path}"
    )
    inputs[str(path)] = digest
    return path


def _unchanged(inputs):
    _require(
        all(Path(path).is_file() and _sha(path) == digest for path, digest in inputs.items()),
        "authenticated input or executing script/plan changed",
    )
    _require("torch" not in sys.modules, "diagnostic must remain Torch-free")


def _live_inventory(root):
    paths = [
        *(root / "src").rglob("*.py"),
        *(root / "configs").rglob("*.yaml"),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    return {path.relative_to(root).as_posix(): _sha(path) for path in sorted(paths)}


def _scores(logits, product_ids, subject_id, truth):
    _require(
        isinstance(product_ids, list)
        and all(type(i) is int and i >= 1024 for i in product_ids)
        and product_ids == sorted(set(product_ids)),
        "invalid product IDs/order",
    )
    _require(type(subject_id) is int and subject_id in product_ids, "invalid subject ID")
    _require(isinstance(logits, list) and len(logits) == len(product_ids), "logit alignment")
    for value in logits:
        _require(type(value) in (float, int) and math.isfinite(value), "nonfinite/nonnumeric logit")
        try:
            exact = struct.unpack("!f", struct.pack("!f", value))[0] == value
        except (OverflowError, struct.error) as exc:
            raise ValueError("non-FP32 logit") from exc
        _require(exact, "non-FP32 logit")
    allowed = {i: z for i, z in zip(product_ids, logits, strict=True) if i != subject_id}
    _require(
        isinstance(truth, list)
        and all(type(i) is int for i in truth)
        and truth == sorted(set(truth))
        and set(truth) <= allowed.keys(),
        "invalid expected IDs",
    )
    return allowed


def _metrics(prediction, truth):
    predicted, expected = set(prediction), set(truth)
    tp = len(predicted & expected)
    return {
        "exact": predicted == expected,
        "tp": tp,
        "fp": len(predicted - expected),
        "fn": len(expected - predicted),
        "set_size": len(predicted),
        "precision": tp / len(predicted) if predicted else float(not expected),
        "recall": tp / len(expected) if expected else 1.0,
        "f1": 2 * tp / (len(predicted) + len(expected)) if predicted or expected else 1.0,
    }


def _geometry(logits, product_ids, subject_id, truth):
    """Use labels only for declared separability and oracle-cardinality diagnostics."""
    allowed = _scores(logits, product_ids, subject_id, truth)
    positives, negatives = set(truth), set(allowed) - set(truth)
    a = min((allowed[i] for i in positives), default=None)
    b = max((allowed[i] for i in negatives), default=None)
    gap = a - b if a is not None and b is not None else None
    if not allowed:
        case, separable, semantics = "empty_universe", True, "any finite t"
    elif not positives:
        case, separable, semantics = "empty_truth", True, "t >= lower; upper unbounded"
    elif not negatives:
        case, separable, semantics = "full_truth", True, "lower unbounded; t < upper"
    else:
        case, separable = "mixed", b < a
        semantics = "lower <= t < upper" if separable else "empty interval"
    separation = {
        "case": case,
        "strict_separable": separable,
        "min_true_logit": a,
        "max_negative_logit": b,
        "gap": gap,
        "min_true_ids": sorted(i for i in positives if allowed[i] == a),
        "max_negative_ids": sorted(i for i in negatives if allowed[i] == b),
        "threshold_interval": {
            "exists": separable,
            "lower": b,
            "upper": a,
            "lower_inclusive": True if b is not None else None,
            "upper_inclusive": False if a is not None else None,
            "semantics": semantics,
        },
    }
    ranked = sorted(allowed, key=lambda i: (-allowed[i], i))
    k = len(truth)
    chosen, excluded = ranked[:k], ranked[k:]
    last = allowed[chosen[-1]] if chosen else None
    first = allowed[excluded[0]] if excluded else None
    has_cut = bool(chosen and excluded)
    tied = has_cut and last == first
    tie_ids = sorted(i for i in allowed if allowed[i] == last) if tied else []
    top_k = {
        "k": k,
        "selected_ranked_ids": chosen,
        "selected_set_ids": sorted(chosen),
        "metrics": _metrics(chosen, truth),
        "boundary": {
            "has_cut": has_cut,
            "selected_min_logit": last,
            "excluded_max_logit": first,
            "tie": tied,
            "tied_product_ids": tie_ids,
            "tied_true_ids": sorted(set(tie_ids) & positives),
            "tied_negative_ids": sorted(set(tie_ids) & negatives),
            "selected_tied_ids": sorted(set(tie_ids) & set(chosen)),
            "excluded_tied_ids": sorted(set(tie_ids) & set(excluded)),
        },
    }
    _require(not separable or top_k["metrics"]["exact"], "strict separation/top-K invariant")
    return separation, top_k


def _alignment(prepared, direct_reports, product_ids):
    """Reproduce the already measured zero rule, without running new geometry."""
    _require(product_ids == list(range(1024, 2049)), "historical product columns changed")
    _require(len(prepared) == len(direct_reports) == 3, "seed coverage")
    partition, all_rows = None, []
    for seed, (reference, rows), direct in zip(_SEEDS, prepared, direct_reports, strict=True):
        _require(
            reference["seed"] == direct["seed"] == seed
            and direct["query_count"] == len(rows) == len(direct["responses"]) == 222,
            "seed/row coverage or order",
        )
        _require(all(reference[k] == direct[k] for k in _IDENTITY), "direct/pair identity mismatch")
        current = [
            (r["subject"], r["dimension"], r["group"], tuple(r["expected_set_ids"])) for r in rows
        ]
        _require(
            len({(r[0], r[1]) for r in current}) == 222
            and (partition is None or current == partition),
            "query partition/order",
        )
        partition = current
        for row, saved in zip(rows, direct["responses"], strict=True):
            _require(
                all(saved[k] == v for k, v in row.items() if k != "logits"),
                "direct/pair row alignment or baseline evidence",
            )
            allowed = _scores(
                row["logits"], product_ids, row["subject_id"], row["expected_set_ids"]
            )
            direct_ids = sorted(i for i, z in allowed.items() if z > 0)
            direct_metrics = _metrics(direct_ids, row["expected_set_ids"])
            pair_metrics = _metrics(row["baseline_set_ids"], row["expected_set_ids"])
            _require(
                direct_ids == saved["predicted_set_ids"]
                and direct_metrics == saved["direct_metrics"]
                and pair_metrics == saved["baseline_metrics"]
                and saved["gained_exact"] == (direct_metrics["exact"] and not pair_metrics["exact"])
                and saved["lost_exact"] == (pair_metrics["exact"] and not direct_metrics["exact"]),
                "zero-rule reconstruction differs from saved report",
            )
            all_rows.append((pair_metrics["exact"], direct_metrics["exact"]))
    counts = {
        "query_count": len(all_rows),
        "pair_exact": sum(p for p, _ in all_rows),
        "direct_exact": sum(d for _, d in all_rows),
        "direct_gains": sum(d and not p for p, d in all_rows),
        "direct_losses": sum(p and not d for p, d in all_rows),
    }
    _require(
        counts
        == {
            "query_count": 666,
            "pair_exact": 569,
            "direct_exact": 339,
            "direct_gains": 3,
            "direct_losses": 233,
        },
        "frozen direct/baseline counts differ",
    )
    return counts


def _preflight(root, plan):
    inputs = {}
    live = _live_inventory(root)
    _bind(plan, _PLAN_SHA, inputs)
    directory = root / "runs/learning/direct-membership-v1"
    helper_path = _bind(directory / "summary.script.py", _HELPER_SHA, inputs)
    helper_plan = _bind(directory / "summary.plan.md", _HELPER_PLAN_SHA, inputs)
    spec = importlib.util.spec_from_file_location("_separability_archived_direct", helper_path)
    _require(spec is not None and spec.loader is not None, "helper loader unavailable")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    pair, reports, tokens, inherited, historical = helper._authenticate(root, helper_plan)
    for path, digest in inherited.items():
        _bind(path, digest, inputs)
    prepared, product_ids = helper._prepare(reports, pair, tokens)
    direct = _read(_bind(directory / "summary.json", _DIRECT_SHA, inputs))
    audit = _read(_bind(directory / "independent-audit.json", _DIRECT_AUDIT_SHA, inputs))
    _require(
        audit["complete"] is True
        and audit["all_recomputed_outputs_equal"] is True
        and audit["summary_sha256"] == _DIRECT_SHA
        and direct["script_sha256"] == _HELPER_SHA
        and direct["plan_sha256"] == _HELPER_PLAN_SHA
        and direct["pair_summary_sha256"] == _PAIR_SHA
        and direct["pair_audit_sha256"] == _PAIR_AUDIT_SHA,
        "direct audit/helper/parent bindings",
    )
    _bind(directory / "independent-audit.py", audit["script_sha256"], inputs)
    for manifest in (direct, audit):
        for path, digest in manifest["input_sha256"].items():
            _bind(path, digest, inputs)
    _require(
        set(direct["report_sha256"]) == {f"seed-{s}.json" for s in _SEEDS},
        "direct report inventory",
    )
    direct_reports = [
        _read(
            _bind(directory / f"seed-{s}.json", direct["report_sha256"][f"seed-{s}.json"], inputs)
        )
        for s in _SEEDS
    ]
    counts = _alignment(prepared, direct_reports, product_ids)
    comparison = direct["comparison"]
    _require(
        comparison["baseline"]["exact_count"] == counts["pair_exact"]
        and comparison["direct"]["exact_count"] == counts["direct_exact"]
        and comparison["gained_exact"] == counts["direct_gains"]
        and comparison["lost_exact"] == counts["direct_losses"],
        "direct summary counts",
    )
    _unchanged(inputs)
    _require(_live_inventory(root) == live, "live source/config inventory changed")
    return {
        "prepared": prepared,
        "product_ids": product_ids,
        "inputs": inputs,
        "historical_archived_inputs": historical,
        "baseline_reconstruction": counts,
        "helper_path": helper_path,
        "helper_plan": helper_plan,
        "historical_model_environment": pair["historical_model_environment"],
        "live_source_config_sha256": live,
    }


def _measure(seed, index, row, product_ids):
    separation, top_k = _geometry(
        row["logits"], product_ids, row["subject_id"], row["expected_set_ids"]
    )
    direct = sorted(
        i
        for i, z in zip(product_ids, row["logits"], strict=True)
        if i != row["subject_id"] and z > 0
    )
    pair_metrics = _metrics(row["baseline_set_ids"], row["expected_set_ids"])
    direct_metrics = _metrics(direct, row["expected_set_ids"])
    pair_exact, direct_exact = pair_metrics["exact"], direct_metrics["exact"]
    top_exact = top_k["metrics"]["exact"]
    return {
        "seed": seed,
        "index": index,
        **{
            k: row[k]
            for k in (
                "subject",
                "subject_id",
                "dimension",
                "group",
                "expected_set_ids",
                "baseline_set_ids",
                "source_union_missing_true_ids",
            )
        },
        "allowed_product_count": len(product_ids) - 1,
        "direct_set_ids": direct,
        "baseline_selection": {
            k: row["baseline_selection"][k]
            for k in ("selected_slot", "selected_kind", "selected_source_ranks")
        },
        "pair_metrics": pair_metrics,
        "direct_metrics": direct_metrics,
        "separation": separation,
        "oracle_top_k": top_k,
        "pair_exact": pair_exact,
        "direct_exact": direct_exact,
        "strict_separable": separation["strict_separable"],
        "separable_with_wrong_zero_threshold": separation["strict_separable"] and not direct_exact,
        "boundary_tie": separation["gap"] == 0 if separation["gap"] is not None else False,
        "strict_rank_overlap": separation["gap"] < 0 if separation["gap"] is not None else False,
        "oracle_top_k_exact": top_exact,
        "oracle_top_k_gained_exact": top_exact and not pair_exact,
        "oracle_top_k_lost_exact": pair_exact and not top_exact,
        "source_union_lacks_truth": bool(row["source_union_missing_true_ids"]),
    }


def _aggregate(rows):
    def measure(field):
        values = [
            r["oracle_top_k"]["metrics"] if field == "oracle_top_k" else r[field] for r in rows
        ]
        return {
            "exact_count": sum(r["exact"] for r in values),
            **{key: sum(r[key] for r in values) for key in ("tp", "fp", "fn")},
            **{
                key: math.fsum(r[key] for r in values) / len(values) if values else None
                for key in ("precision", "recall", "f1", "set_size")
            },
        }

    cross = Counter(tuple(row[k] for k in _CROSS_FIELDS) for row in rows)
    return {
        "query_count": len(rows),
        "pair_metrics": measure("pair_metrics"),
        "direct_metrics": measure("direct_metrics"),
        "oracle_top_k_metrics": measure("oracle_top_k"),
        "counts": {key: sum(row[key] for row in rows) for key in _FLAGS},
        "separation_cases": {
            case: sum(r["separation"]["case"] == case for r in rows) for case in _CASES
        },
        "cross_tab": [
            {**dict(zip(_CROSS_FIELDS, key, strict=True)), "query_count": count}
            for key, count in sorted(cross.items())
        ],
    }


def _summarize(rows):
    failures = [r for r in rows if not r["pair_exact"]]
    return {
        "totals": _aggregate(rows),
        "groups": {g: _aggregate([r for r in rows if r["group"] == g]) for g in _GROUPS},
        "pair_failures": {
            "totals": _aggregate(failures),
            "groups": {g: _aggregate([r for r in failures if r["group"] == g]) for g in _GROUPS},
        },
    }


def _refuse(output):
    _require(
        not output.exists()
        and not list(output.parent.glob(output.stem + ".*"))
        and not list(output.parent.glob("seed-*.json")),
        "immutable output already exists",
    )


def main():
    script = Path(__file__).resolve()
    executed = script.read_bytes()
    root = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--plan",
        type=Path,
        default=root / "docs/experiments/2026-09-25-membership-separability-plan.md",
    )
    parser.add_argument(
        "--out", type=Path, default=root / "runs/learning/membership-separability-v1/summary.json"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    _require("torch" not in sys.modules, "diagnostic must remain Torch-free")
    output = args.out.resolve()
    _refuse(output)
    context = _preflight(args.root.resolve(), args.plan)
    context["inputs"][str(script)] = hashlib.sha256(executed).hexdigest()
    _unchanged(context["inputs"])
    _require(
        _live_inventory(args.root.resolve()) == context["live_source_config_sha256"],
        "live source/config inventory changed",
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    **context["baseline_reconstruction"],
                    "separability_measured": False,
                    "torch_imported": False,
                    "model_execution": False,
                }
            )
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": executed,
        "plan.md": args.plan.read_bytes(),
        "helper.py": context["helper_path"].read_bytes(),
        "helper-plan.md": context["helper_plan"].read_bytes(),
    }
    snapshot_hashes = {}
    for suffix, payload in snapshots.items():
        target = output.with_suffix("." + suffix)
        with target.open("xb") as stream:
            stream.write(payload)
        snapshot_hashes[suffix] = _sha(target)
    summaries, all_rows, report_hashes = [], [], {}
    for reference, prepared_rows in context["prepared"]:
        rows = [
            _measure(reference["seed"], index, row, context["product_ids"])
            for index, row in enumerate(prepared_rows)
        ]
        report = {
            "seed": reference["seed"],
            "query_count": len(rows),
            **{k: reference[k] for k in _IDENTITY},
            **_summarize(rows),
            "responses": rows,
        }
        target = output.parent / f"seed-{reference['seed']}.json"
        _write(target, report)
        report_hashes[target.name] = _sha(target)
        summaries.append({k: v for k, v in report.items() if k != "responses"})
        all_rows.extend(rows)
    _unchanged(context["inputs"])
    _require(
        _live_inventory(args.root.resolve()) == context["live_source_config_sha256"],
        "live source/config inventory changed",
    )
    _require(
        all(
            _sha(output.with_suffix("." + suffix)) == digest
            for suffix, digest in snapshot_hashes.items()
        ),
        "snapshot changed during measurement",
    )
    _require(
        all(_sha(output.parent / name) == digest for name, digest in report_hashes.items()),
        "report changed during measurement",
    )
    result = {
        "diagnostic_version": "plm-membership-separability-v1",
        "diagnostic_complete": True,
        "diagnostic_accepted": False,
        "independent_audit_status": "required",
        "query_count": len(all_rows),
        "distinct_validation_queries": 222,
        "training_seed_replicas": 3,
        "product_token_ids": context["product_ids"],
        "saved_logit_batch_shape": {"batch_size": 8, "final_batch_size": 6},
        "baseline_reconstruction": context["baseline_reconstruction"],
        **_summarize(all_rows),
        "runs": summaries,
        "cross_tab_fields": list(_CROSS_FIELDS),
        "input_sha256": context["inputs"],
        "historical_archived_inputs": context["historical_archived_inputs"],
        "snapshot_sha256": snapshot_hashes,
        "report_sha256": report_hashes,
        "plan_sha256": _PLAN_SHA,
        "helper_sha256": _HELPER_SHA,
        "pair_summary_sha256": _PAIR_SHA,
        "pair_audit_sha256": _PAIR_AUDIT_SHA,
        "direct_summary_sha256": _DIRECT_SHA,
        "direct_audit_sha256": _DIRECT_AUDIT_SHA,
        "historical_model_environment": context["historical_model_environment"],
        "live_source_config_sha256": context["live_source_config_sha256"],
        "analysis_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_imported": False,
            "model_execution": False,
        },
        "empty_metric_convention": (
            "empty/empty precision=recall=F1=1; empty truth recall=1; "
            "empty prediction precision=0 when truth nonempty"
        ),
        "limitations": [
            "Separation and oracle top-K exactness diagnose exact-answer attainability; "
            "top-K precision/recall/F1 are privileged diagnostics, "
            "not threshold-performance upper bounds.",
            "Three training seeds repeat the same 222 validation queries.",
            "No threshold fitting, calibrator, training, forward, protected test "
            "or application promotion.",
        ],
    }
    _write(output, result)
    print(
        json.dumps(
            {
                "diagnostic_complete": True,
                "diagnostic_accepted": False,
                "summary_sha256": _sha(output),
            }
        )
    )


if __name__ == "__main__":
    main()
