"""Saved-head diagnosis of true products missing from all eight source paths.

No predictor, threshold selection, model forward or recursive training audit.
Head ranks are not decoder-plus-guidance ranks; oracle K is diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
import sys
from pathlib import Path

_PLAN = "bda59d3987d98bb746f3b0dea7445269ec5044f82abd5f0fce3df9404b855907"
_SUMMARY = "1fe6e9cb1980753e6c2315c6dac890148a52105061c881261da9bb73db1c387d"
_AUDIT = "27e0c5ff2931d88fed9fae612c73c64c7835e01282097fc290fa78cd9a7e330c"
_DECISION = "01a51a5a9d5957ec68f32b0d81d23acace29d7eb3aa5684d99939582e88cb32a"
_SEEDS = (1729, 1730, 1731)
_COUNTS = {1729: 14, 1730: 6, 1731: 10}
_COLUMNS = list(range(1024, 2049))
_BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"


def _require(value, message):
    if not value:
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
    directory = root / "runs/learning/eight-source-set-algebra-v1"
    summary = _read(_bind(directory / "summary.json", _SUMMARY, inputs))
    audit = _read(_bind(directory / "independent-audit.json", _AUDIT, inputs))
    decision = _read(_bind(directory / "decision.json", _DECISION, inputs))
    _require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and decision["evidence_accepted"] is True
        and decision["fixed_quality_gate_passed"] is False
        and summary["gate"]["quality_passed"] is False
        and audit["summary_sha256"] == decision["summary_sha256"] == _SUMMARY
        and decision["audit_sha256"] == _AUDIT
        and decision["auditor_sha256"] == audit["script_sha256"],
        "accepted evidence / failed quality chain",
    )
    _bind(directory / "independent-audit.py", audit["script_sha256"], inputs)
    reports = {}
    for receipt in summary["reports"]:
        seed = receipt["seed"]
        _require(seed in _SEEDS and seed not in reports, "seed receipt")
        report = _read(_bind(receipt["path"], receipt["sha256"], inputs))
        _require(
            report["complete"] is True
            and report["seed"] == seed
            and report["query_count"] == len(report["responses"]) == 222
            and report["product_token_ids"] == _COLUMNS,
            "report coverage/columns",
        )
        reports[seed] = report
    _require(set(reports) == set(_SEEDS), "three seed reports")
    return {"inputs": inputs, "reports": reports}


def _coverage(row):
    prompt, truth = row["prompt_ids"], row["expected_set_ids"]
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
        row["group"] in ("COLOR", "TYPE_single", "TYPE_dual")
        and (row["group"] == "COLOR") == (row["dimension"] == "COLOR"),
        "dimension/group",
    )
    _require(
        truth
        and truth == sorted(set(truth))
        and all(type(i) is int and i in _COLUMNS for i in truth)
        and prompt[1] not in truth,
        "canonical subject-excluded truth",
    )
    logits = row["symmetric_relation_logits"]
    _require(len(logits) == 1025, "1025 head columns")
    for value in logits:
        _require(type(value) in (int, float) and math.isfinite(value), "finite head")
        try:
            exact = struct.unpack("f", struct.pack("f", value))[0] == value
        except (OverflowError, struct.error) as exc:
            raise ValueError("FP32 head range") from exc
        _require(exact, "exact FP32 head")
    _require(len(row["source_paths"]) == 8, "eight sources")
    sets, removed, first = [], [], []
    for rank, path in enumerate(row["source_paths"], 1):
        ids = path["token_ids"]
        _require(
            path["protocol_valid"] is True
            and path["terminated"] is True
            and path["error"] is None
            and 7 <= len(ids) <= 512
            and all(type(i) is int for i in ids)
            and ids[:5] == prompt
            and ids[-1] == 2
            and all(i in _COLUMNS for i in ids[5:-1]),
            "valid terminated source grammar",
        )
        products = ids[5:-1]
        _require(
            len(products) == len(set(products)) == len(path["targets"])
            and len(path["targets"]) == len(set(path["targets"])),
            "raw uniqueness/target alignment",
        )
        _require(
            path["decoding"] == _BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            "rank descriptor",
        )
        sets.append(sorted(set(products) - {prompt[1]}))
        removed.append(prompt[1] in products)
        first.append(products[0])
    _require(
        len(set(first)) == 8
        and sets == row["source_set_ids"]
        and removed == row["source_subject_removed"],
        "raw/source alignment",
    )
    union = sorted(set().union(*(set(s) for s in sets)))
    missing, covered = sorted(set(truth) - set(union)), sorted(set(truth) & set(union))
    _require(
        row["diagnosis"]["source_union_contains_truth"] is (not missing)
        and (row["diagnosis"]["baseline_state"] == "unavailable_and_truth_missing_from_all_sources")
        == bool(missing),
        "saved coverage diagnosis",
    )
    return {
        "raw": row,
        "union": union,
        "missing": missing,
        "covered": covered,
        "coverage": {
            **{k: row[k] for k in ("index", "subject", "dimension", "group")},
            "true_member_count": len(truth),
            "covered_member_count": len(covered),
            "missing_member_count": len(missing),
            "missing_ids": missing,
        },
    }


def _aligned(rows, previous, *, count=222):
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
        len(rows) == count
        and len({(k[0], k[1]) for k in keys}) == count
        and (previous is None or previous == keys),
        "cross-seed ordered query/truth alignment",
    )
    return keys


def _prepare(reports):
    prepared, previous = {}, None
    for seed in _SEEDS:
        rows = []
        for index, row in enumerate(reports[seed]["responses"]):
            _require(row["index"] == index, "query index order")
            rows.append(_coverage(row))
        previous = _aligned(rows, previous)
        focused = [r for r in rows if r["missing"]]
        _require(
            len(focused) == _COUNTS[seed]
            and all(r["raw"]["group"] == "TYPE_dual" for r in focused),
            "fixed focused subset coverage",
        )
        prepared[seed] = rows
    return prepared


def _member_summary(members):
    def spread(key):
        values = [m[key] for m in members]
        return (
            {"min": min(values), "median": statistics.median(values), "max": max(values)}
            if values
            else None
        )

    return {
        "count": len(members),
        "positive_count": sum(m["logit"] > 0 for m in members),
        "zero_count": sum(m["logit"] == 0 for m in members),
        "negative_count": sum(m["logit"] < 0 for m in members),
        "logit": spread("logit"),
        "rank": spread("rank"),
    }


def _focused(item):
    _require(item["missing"], "focus requires omitted truth before head analysis")
    row = item["raw"]
    truth, subject, logits = (
        set(row["expected_set_ids"]),
        row["prompt_ids"][1],
        row["symmetric_relation_logits"],
    )
    universe = [i for i in _COLUMNS if i != subject]
    order = sorted(universe, key=lambda i: (-logits[i - 1024], i))
    ranks = {token: index for index, token in enumerate(order, 1)}
    false = set(universe) - truth

    def members(ids):
        return [
            {
                "token_id": token,
                "logit": logits[token - 1024],
                "rank": ranks[token],
                "false_strictly_higher": sum(
                    logits[i - 1024] > logits[token - 1024] for i in false
                ),
                "false_tied": sum(logits[i - 1024] == logits[token - 1024] for i in false),
                "oracle_top_k": ranks[token] <= len(truth),
            }
            for token in ids
        ]

    omitted, covered = members(item["missing"]), members(item["covered"])
    count = sum(m["logit"] > 0 for m in omitted)
    return {
        **{
            k: row[k]
            for k in ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_ids")
        },
        "source_union_ids": item["union"],
        "oracle_k": len(truth),
        "omitted_members": omitted,
        "covered_members": covered,
        "omitted_summary": _member_summary(omitted),
        "covered_summary": _member_summary(covered),
        "positive_omitted_category": "all"
        if count == len(omitted)
        else "some"
        if count
        else "none",
    }


def _totals(rows):
    result = {
        "focused_queries": len(rows),
        "distinct_focused_queries": len({(r["subject"], r["dimension"]) for r in rows}),
        "omitted_oracle_top_k_count": sum(
            m["oracle_top_k"] for r in rows for m in r["omitted_members"]
        ),
        "positive_omitted_queries": {
            category: sum(r["positive_omitted_category"] == category for r in rows)
            for category in ("all", "some", "none")
        },
    }
    for name in ("omitted", "covered"):
        groups = [r[name + "_summary"] for r in rows]
        nonempty = [g for g in groups if g["count"]]
        count, positive = sum(g["count"] for g in groups), sum(g["positive_count"] for g in groups)
        result[name] = {
            "member_occurrences": count,
            "positive_count": positive,
            "zero_count": sum(g["zero_count"] for g in groups),
            "negative_count": sum(g["negative_count"] for g in groups),
            "positive_member_fraction": positive / count if count else None,
            "nonempty_query_count": len(nonempty),
            "mean_query_positive_fraction": math.fsum(
                g["positive_count"] / g["count"] for g in nonempty
            )
            / len(nonempty)
            if nonempty
            else None,
        }
    return result


def _coverage_totals(rows):
    return {
        "query_count": len(rows),
        "focused_queries": sum(r["missing_member_count"] > 0 for r in rows),
        "fully_covered_queries": sum(r["missing_member_count"] == 0 for r in rows),
        "true_member_occurrences": sum(r["true_member_count"] for r in rows),
        "covered_member_occurrences": sum(r["covered_member_count"] for r in rows),
        "omitted_member_occurrences": sum(r["missing_member_count"] for r in rows),
    }


def _execute(output, prepared, summary):
    all_focused, all_coverage = [], []
    try:
        for seed in _SEEDS:
            report = {
                "seed": seed,
                "complete": False,
                "coverage": [r["coverage"] for r in prepared[seed]],
                "focused_rows": [],
            }
            path = output.parent / f"seed-{seed}.json"
            try:
                for item in prepared[seed]:
                    if item["missing"]:
                        report["failed_query"] = item["coverage"]
                        report["focused_rows"].append(_focused(item))
                        del report["failed_query"]
                report["overall"] = _totals(report["focused_rows"])
                report["coverage_summary"] = _coverage_totals(report["coverage"])
                report["complete"] = True
            except BaseException as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                _write(path, report)
                summary["reports"].append({"seed": seed, "path": str(path), "sha256": _sha(path)})
            all_focused.extend(report["focused_rows"])
            all_coverage.extend(report["coverage"])
        summary["overall"] = _totals(all_focused)
        summary["coverage_summary"] = _coverage_totals(all_coverage)
        summary["complete"] = True
    except BaseException as exc:
        summary["complete"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            _unchanged(summary["input_sha256"])
            _unchanged(summary["snapshot_sha256"])
            summary["final_identity_check"] = True
        except BaseException as exc:
            summary["complete"] = summary["final_identity_check"] = False
            summary["identity_error"] = f"{type(exc).__name__}: {exc}"
        _write(output, summary)
    _require(summary["complete"], "final identity check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/experiments/2026-09-25-missing-source-membership-plan.md"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/learning/missing-source-membership-v1/summary.json")
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
        "standalone stdlib execution from repository root",
    )
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
                    "coverage_reconstructed": True,
                    "diagnostic_measured": False,
                    "torch_imported": False,
                    "input_count": len(context["inputs"]),
                    "script_sha256": digest,
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
        "script.py": content,
        "plan.md": args.plan.read_bytes(),
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
        "campaign_version": "missing-source-membership-v1",
        "complete": False,
        "acceptance": False,
        "plan_sha256": _PLAN,
        "script_sha256": digest,
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "reports": [],
        "upstream_evidence_accepted": True,
        "upstream_quality_gate_passed": False,
        "neural_generation_executed": False,
        "training_executed": False,
        "protected_test_used": False,
        "new_prediction_policy": False,
        "limitations": [
            "Upstream audits establish source/runtime provenance; "
            "no new neural replay or recursive checkpoint audit.",
            "Relation-head ranks are not decoder-plus-guidance first-token ranks; "
            "oracle K uses labels.",
            "Positive logits are neither calibrated probabilities nor causal proof; "
            "repeated products are not independent samples.",
            "Focused query counts are query-seed observations; "
            "distinct query identities are reported separately.",
        ],
    }
    _execute(output, prepared, summary)


if __name__ == "__main__":
    main()
