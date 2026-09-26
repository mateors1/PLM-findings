"""Stdlib-only saved bilinear score geometry; no threshold selection or predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path

_PLAN = "7453c260275c58c4dd8f34fe5be1b2f054251f8ddf1e9b039ec608c76d96e232"
_PINS = {
    "aggregate-v2.json": "338fb6b7ea08128952ce9abc02b3506639e712a9cff6ce63468ed2e7980f79a0",
    "independent-audit.json": "ddbecaceabdacc0808f9b4f122e0309aa3e233fe76b0d75694e31833a40083a8",
    "decision.json": "62a8dcbb7cbe5d038984f9d57aecacec4b0ba612ba55c2a9f6fe335977304978",
}
_CATEGORIES = ("exact_zero", "separated_missing", "separated_extra", "overlap_or_tie")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_bound(path, digest, inputs):
    path = Path(path).resolve()
    _require(path.suffix == ".json", "only JSON evidence inputs")
    _require(_sha(path) == digest, "evidence identity: " + str(path))
    _require(str(path) not in inputs or inputs[str(path)] == digest, "identity conflict")
    inputs[str(path)] = digest
    return json.loads(path.read_text(encoding="utf-8"))


def _preflight(root, plan):
    _require(_sha(plan) == _PLAN, "frozen geometry plan")
    inputs = {str(plan.resolve()): _PLAN}
    folder = root / "runs/learning/bilinear-seed-replication-v1"
    summary, audit, decision = [
        _read_bound(folder / name, pin, inputs) for name, pin in _PINS.items()
    ]
    _require(
        summary["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == _PINS["aggregate-v2.json"]
        and decision["summary_sha256"] == _PINS["aggregate-v2.json"]
        and decision["audit_sha256"] == _PINS["independent-audit.json"]
        and decision["evidence_accepted"] is True,
        "accepted replication chain",
    )
    children, child_hashes = {}, {}
    historical = summary["historical1729"]
    old = root / "runs/learning/bilinear-budget2000-v1"
    hs = _read_bound(old / "summary.json", historical["summary_sha256"], inputs)
    ha = _read_bound(old / "independent-audit.json", historical["audit_sha256"], inputs)
    hd = _read_bound(old / "decision.json", historical["decision_sha256"], inputs)
    _require(
        hs["complete"] is True
        and ha["audit_passed"] is True
        and ha["summary_sha256"] == historical["summary_sha256"]
        and hd["summary_sha256"] == historical["summary_sha256"]
        and hd["audit_sha256"] == historical["audit_sha256"]
        and hd["evidence_accepted"] is True,
        "accepted historical chain",
    )
    digest = hs["artifact_sha256"]["child.json"]
    _require(digest == historical["child_sha256"], "historical child identity")
    children[1729] = _read_bound(old / "child.json", digest, inputs)
    child_hashes[1729] = digest
    _require([r["seed"] for r in summary["reports"]] == [1730, 1731], "ordered fresh seeds")
    for entry in summary["reports"]:
        seed = entry["seed"]
        own = folder / f"seed-{seed}"
        _require(
            Path(entry["path"]).resolve() == (own / "summary.json").resolve(),
            "fresh summary location",
        )
        report = _read_bound(own / "summary.json", entry["sha256"], inputs)
        audited = next(r for r in audit["per_seed"] if r["seed"] == seed)
        checked = _read_bound(own / "independent-audit.json", audited["audit_sha256"], inputs)
        _require(
            report["complete"] is True
            and checked["audit_passed"] is True
            and checked["summary_sha256"] == entry["sha256"]
            and audited["summary_sha256"] == entry["sha256"],
            "fresh accepted chain",
        )
        digest = report["artifact_sha256"]["child.json"]
        _require(
            checked["input_sha256"][str((own / "child.json").resolve())] == digest,
            "audited child identity",
        )
        children[seed] = _read_bound(own / "child.json", digest, inputs)
        child_hashes[seed] = digest
    for child in children.values():
        _require(
            child["complete"] is True
            and child["query_count"] == 222
            and child["product_token_ids"] == list(range(1024, 2049))
            and len(child["responses"]) == 222,
            "complete product-column contract",
        )
    _alignment(children)
    return children, child_hashes, inputs


def _alignment(children):
    keys = ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_ids")
    baseline = children[1729]["responses"]
    for child in children.values():
        _require(len(child["responses"]) == len(baseline), "query count drift")
        for left, right in zip(baseline, child["responses"], strict=True):
            _require(all(left[k] == right[k] for k in keys), "aligned query identity")


def _finite_fp32(value):
    _require(type(value) in (int, float) and math.isfinite(value), "finite numeric score")
    try:
        packed = struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("FP32 score required") from exc
    _require(packed == value, "FP32 score required")
    return value


def _reduce(row, columns):
    _require(columns == sorted(set(columns)), "ascending unique product columns")
    prompt = row["prompt_ids"]
    _require(len(prompt) == 5, "subject prompt")
    subject = prompt[1]
    _require(subject in columns, "subject prompt")
    truth = row["expected_set_ids"]
    selected = row["selected_set_ids"]
    _require(
        truth == sorted(set(truth)) and set(truth) <= set(columns) and subject not in truth,
        "valid nonself truth",
    )
    _require(
        selected == sorted(set(selected)) and subject not in selected, "ascending nonself selection"
    )
    scores = [_finite_fp32(v) for v in row["symmetric_relation_logits"]]
    _require(len(scores) == len(columns), "score column count")
    values = dict(zip(columns, scores, strict=True))
    target = set(truth)
    negatives = set(columns) - target - {subject}
    _require(bool(target) and bool(negatives), "nonempty true and false sets")
    replay = [i for i in columns if i != subject and values[i] > 0]
    _require(replay == selected, "saved zero-threshold prediction")
    fp = sorted(set(replay) - target)
    fn = sorted(target - set(replay))
    a = min(values[i] for i in target)
    b = max(values[i] for i in negatives)
    exact = not fp and not fn
    separated = a > b
    _require(exact == (a > 0 and b <= 0), "zero-boundary equivalence")
    metrics = row["metrics"]
    _require(
        type(metrics["exact"]) is bool
        and metrics["exact"] == exact
        and metrics["false_positive_count"] == len(fp)
        and metrics["false_negative_count"] == len(fn)
        and type(row["strict_separation"]) is bool
        and row["strict_separation"] == separated,
        "saved error metrics",
    )
    category = (
        "exact_zero"
        if exact
        else (
            "overlap_or_tie"
            if not separated
            else ("separated_missing" if a <= 0 else "separated_extra")
        )
    )
    return {
        "index": row["index"],
        "subject": row["subject"],
        "dimension": row["dimension"],
        "group": row["group"],
        "prompt_ids": prompt,
        "expected_set_sha256": hashlib.sha256(_bytes(truth)).hexdigest(),
        "a": a,
        "b": b,
        "g": a - b,
        "minimum_true_id": min(i for i in target if values[i] == a),
        "maximum_false_id": min(i for i in negatives if values[i] == b),
        "exact": exact,
        "strict_separation": separated,
        "category": category,
        "false_positive": [{"id": i, "score": values[i]} for i in fp],
        "false_negative": [{"id": i, "score": values[i]} for i in fn],
    }


def _totals(rows, interval=True):
    _require(bool(rows), "nonempty reduction group")
    counts = Counter(row["category"] for row in rows)
    result = {
        "query_count": len(rows),
        "exact_count": sum(r["exact"] for r in rows),
        "strict_separation_count": sum(r["strict_separation"] for r in rows),
        "false_positive_count": sum(len(r["false_positive"]) for r in rows),
        "false_negative_count": sum(len(r["false_negative"]) for r in rows),
        "category_counts": {k: counts[k] for k in _CATEGORIES},
    }
    if interval:
        a = min(r["a"] for r in rows)
        b = max(r["b"] for r in rows)
        result["shared_threshold"] = {
            "A": a,
            "B": b,
            "exists": b < a,
            "lower_inclusive": True,
            "upper_exclusive": True,
            "A_query_index": min(r["index"] for r in rows if r["a"] == a),
            "B_query_index": min(r["index"] for r in rows if r["b"] == b),
        }
    return result


def _reconcile(totals, accepted):
    for key in (
        "query_count",
        "exact_count",
        "strict_separation_count",
        "false_positive_count",
        "false_negative_count",
    ):
        _require(totals[key] == accepted[key], "accepted metric reconciliation: " + key)


def _seed(seed, child, digest):
    rows = [_reduce(row, child["product_token_ids"]) for row in child["responses"]]
    _require([r["index"] for r in rows] == list(range(len(rows))), "ordered query indexes")
    aggregate = _totals(rows)
    _reconcile(aggregate, child["aggregate"])
    groups = {g: _totals([r for r in rows if r["group"] == g]) for g in sorted(child["groups"])}
    for g, metrics in groups.items():
        _reconcile(metrics, child["groups"][g])
    _require({r["group"] for r in rows} == set(groups), "group coverage")
    return {
        "complete": True,
        "seed": seed,
        "child_sha256": digest,
        "query_count": len(rows),
        "rows": rows,
        "aggregate": aggregate,
        "groups": groups,
    }


def _pool(reports, seeds):
    rows = [r for seed in seeds for r in reports[seed]["rows"]]
    return {
        "aggregate": _totals(rows, False),
        "groups": {
            g: _totals([r for r in rows if r["group"] == g], False)
            for g in sorted({r["group"] for r in rows})
        },
    }


def _overlap(reports):
    histogram = Counter()
    failures = []
    for position, row in enumerate(reports[1729]["rows"]):
        seeds = [s for s in (1729, 1730, 1731) if not reports[s]["rows"][position]["exact"]]
        histogram[",".join(map(str, seeds)) or "none"] += 1
        if seeds:
            failures.append(
                {k: row[k] for k in ("index", "subject", "dimension", "group")} | {"seeds": seeds}
            )
    return {
        "shared_queries": len(reports[1729]["rows"]),
        "failure_seed_histogram": dict(sorted(histogram.items())),
        "failed_queries": failures,
    }


def _write(path, value):
    content = value if isinstance(value, bytes) else _bytes(value)
    _require(len(content) <= 512 * 1024, "portable file exceeds 512KiB")
    with path.open("xb") as stream:
        stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    _require(Path.cwd().resolve() == root and "torch" not in sys.modules, "stdlib root boundary")
    plan = root / "docs/experiments/2026-09-25-bilinear-error-geometry-plan.md"
    children, child_hashes, inputs = _preflight(root, plan)
    script = Path(__file__).resolve()
    digest = _sha(script)
    inputs[str(script)] = digest
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "input_count": len(inputs),
                    "torch_imported": False,
                    "script_sha256": digest,
                }
            )
        )
        return
    _require(args.test_receipt is not None and args.test_receipt_sha256, "pinned tests required")
    receipt = _read_bound(args.test_receipt, args.test_receipt_sha256, inputs)
    _require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["diagnostic_executed"] is False
        and receipt["tested_script_sha256"] == digest,
        "synthetic test receipt",
    )
    test = root / "tests/unit/test_bilinear_error_geometry.py"
    _require(_sha(test) == receipt["test_sha256"], "tested source identity")
    inputs[str(test)] = receipt["test_sha256"]
    stdout = Path(receipt["stdout"]["path"]).resolve()
    _require(_sha(stdout) == receipt["stdout"]["sha256"], "test stdout identity")
    inputs[str(stdout)] = receipt["stdout"]["sha256"]
    out = root / "runs/learning/bilinear-error-geometry-v1"
    names = ["summary.json", "seed-1729.json", "seed-1730.json", "seed-1731.json"]
    snapshots = {
        "summary.script.py": script.read_bytes(),
        "summary.plan.md": plan.read_bytes(),
        "summary.test-receipt.json": args.test_receipt.read_bytes(),
        "summary.test-stdout.txt": stdout.read_bytes(),
        "summary.inputs.json": _bytes(inputs),
    }
    _require(all(not (out / n).exists() for n in names + list(snapshots)), "immutable outputs")
    out.mkdir(parents=True, exist_ok=True)
    snapshot_hashes = {}
    for name, content in snapshots.items():
        _write(out / name, content)
        snapshot_hashes[str(out / name)] = _sha(out / name)
    summary = {
        "campaign_version": "bilinear-error-geometry-v1",
        "complete": False,
        "acceptance": False,
        "script_sha256": digest,
        "plan_sha256": _PLAN,
        "input_sha256": inputs,
        "snapshot_sha256": snapshot_hashes,
        "seed_artifacts": [],
        "limitations": [
            "Label-informed descriptive reductions; no threshold chosen or alternative answers.",
            "666 observations share222 validation queries;1729 is historical selection evidence.",
            "No training, checkpoint, protected-test access, serving promotion or quality gate.",
        ],
    }
    try:
        reports = {}
        for seed in (1729, 1730, 1731):
            reports[seed] = _seed(seed, children[seed], child_hashes[seed])
            path = out / f"seed-{seed}.json"
            _write(path, reports[seed])
            summary["seed_artifacts"].append(
                {"seed": seed, "path": str(path), "sha256": _sha(path)}
            )
        summary.update(
            fresh_two=_pool(reports, [1730, 1731]),
            all_three=_pool(reports, [1729, 1730, 1731]),
            overlap=_overlap(reports),
        )
        _require(
            all(_sha(p) == h for p, h in (inputs | snapshot_hashes).items()), "final identity drift"
        )
        summary["complete"] = True
    except BaseException as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        _write(out / "summary.json", summary)
    _require("torch" not in sys.modules, "stdlib execution boundary")
    print(json.dumps({"complete": True, "summary_sha256": _sha(out / "summary.json")}))


if __name__ == "__main__":
    main()
