"""Independent saved-score audit; no model, threshold selection, or primary imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

PLAN = "7453c260275c58c4dd8f34fe5be1b2f054251f8ddf1e9b039ec608c76d96e232"
RUNNER = "66093bfbd06c8622b61535eeaffd0b09886c0cee05ec9eae71292a6e85925789"
TEST_RECEIPT = "1201c714fbfd9bb348373fb65793375d0c0332e549c55e52ed6b55ae2aad760a"
PINS = {
    "aggregate-v2.json": "338fb6b7ea08128952ce9abc02b3506639e712a9cff6ce63468ed2e7980f79a0",
    "independent-audit.json": "ddbecaceabdacc0808f9b4f122e0309aa3e233fe76b0d75694e31833a40083a8",
    "decision.json": "62a8dcbb7cbe5d038984f9d57aecacec4b0ba612ba55c2a9f6fe335977304978",
}
SEEDS = (1729, 1730, 1731)
GROUPS = ("COLOR", "TYPE_dual", "TYPE_single")
CATEGORIES = ("exact_zero", "separated_missing", "separated_extra", "overlap_or_tie")


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def same(actual, expected, message):
    require(encoded(actual) == encoded(expected), message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bound(path, digest, inputs):
    path = Path(path).resolve()
    require(sha(path) == digest, "input hash: " + str(path))
    require(str(path) not in inputs or inputs[str(path)] == digest, "conflicting identity")
    inputs[str(path)] = digest
    return read(path) if path.suffix == ".json" else path.read_bytes()


def accepted(summary, audit, decision, summary_hash, audit_hash):
    require(summary["complete"] is True, "completed upstream")
    require(audit["complete"] is True and audit["audit_passed"] is True, "upstream audit")
    same(audit["summary_sha256"], summary_hash, "upstream audit binding")
    same(decision["summary_sha256"], summary_hash, "upstream decision summary")
    same(decision["audit_sha256"], audit_hash, "upstream decision audit")
    require(decision["evidence_accepted"] is True, "upstream owner acceptance")


def authenticated_children(root, inputs):
    """Open only explicit accepted JSONs, never their historical dependency manifests."""
    folder = root / "runs/learning/bilinear-seed-replication-v1"
    summary, audit, decision = [bound(folder / n, h, inputs) for n, h in PINS.items()]
    accepted(summary, audit, decision, PINS["aggregate-v2.json"], PINS["independent-audit.json"])
    history = summary["historical1729"]
    old = root / "runs/learning/bilinear-budget2000-v1"
    hs, ha, hd = [
        bound(old / n, history[k], inputs)
        for n, k in (
            ("summary.json", "summary_sha256"),
            ("independent-audit.json", "audit_sha256"),
            ("decision.json", "decision_sha256"),
        )
    ]
    accepted(hs, ha, hd, history["summary_sha256"], history["audit_sha256"])
    same(hs["artifact_sha256"]["child.json"], history["child_sha256"], "historical child")
    digests = {1729: history["child_sha256"]}
    children = {1729: bound(old / "child.json", digests[1729], inputs)}
    same([r["seed"] for r in summary["reports"]], [1730, 1731], "fresh seed coverage")
    same([r["seed"] for r in audit["per_seed"]], [1730, 1731], "audited seed coverage")
    for entry, audited in zip(summary["reports"], audit["per_seed"], strict=True):
        seed = entry["seed"]
        own = folder / f"seed-{seed}"
        require(Path(entry["path"]).resolve() == (own / "summary.json").resolve(), "seed path")
        ss = bound(own / "summary.json", entry["sha256"], inputs)
        aa = bound(own / "independent-audit.json", audited["audit_sha256"], inputs)
        require(ss["complete"] is True and aa["audit_passed"] is True, "fresh accepted evidence")
        same(aa["summary_sha256"], entry["sha256"], "seed audit binding")
        same(audited["summary_sha256"], entry["sha256"], "aggregate seed binding")
        digests[seed] = ss["artifact_sha256"]["child.json"]
        same(aa["input_sha256"][str((own / "child.json").resolve())], digests[seed], "child audit")
        children[seed] = bound(own / "child.json", digests[seed], inputs)
    return children, digests


def fp32(number):
    require(type(number) in (int, float) and math.isfinite(number), "finite numeric score")
    try:
        rounded = struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("finite FP32 representability") from error
    require(rounded == number, "exact FP32 representability")
    return number


def metrics(prediction, truth):
    hits = len(set(prediction) & set(truth))
    precision = hits / len(prediction) if prediction else 0.0
    recall = hits / len(truth)
    return {
        "exact": set(prediction) == set(truth),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "set_size": len(prediction),
        "false_positive_count": len(prediction) - hits,
        "false_negative_count": len(truth) - hits,
        "serialization_compatible": 1 <= len(prediction) <= 506,
    }


def reduce_row(row, columns):
    require(columns == sorted(set(columns)) and all(type(i) is int for i in columns), "columns")
    prompt = row["prompt_ids"]
    require(len(prompt) == 5 and all(type(i) is int for i in prompt), "prompt shape")
    subject = prompt[1]
    dimension = row["dimension"]
    require(dimension in ("TYPE", "COLOR"), "dimension")
    same(prompt, [1, subject, 32 if dimension == "TYPE" else 33, 34, 5], "protocol prompt")
    require(subject in columns, "subject column")
    require(
        row["group"] in (("COLOR",) if dimension == "COLOR" else ("TYPE_single", "TYPE_dual")),
        "group",
    )
    truth = row["expected_set_ids"]
    require(all(type(i) is int for i in truth), "integer truth")
    require(truth == sorted(set(truth)) and set(truth) <= set(columns), "canonical truth")
    require(subject not in truth and truth, "nonself nonempty truth")
    false_ids = sorted(set(columns).difference(truth, [subject]))
    require(false_ids, "nonempty negative set")
    logits = row["symmetric_relation_logits"]
    require(len(logits) == len(columns), "logit width")
    scores = {i: fp32(z) for i, z in zip(columns, logits, strict=True)}
    selected = [i for i in columns if i != subject and scores[i] > 0]
    same(row["selected_set_ids"], selected, "saved fixed zero rule")
    calculated = metrics(selected, truth)
    same(row["metrics"], calculated, "saved row metrics")
    low = min(truth, key=lambda i: (scores[i], i))
    high = min(false_ids, key=lambda i: (-scores[i], i))
    a, b = scores[low], scores[high]
    separated = a > b
    same(row["strict_separation"], separated, "saved separation")
    exact = a > 0 and b <= 0
    require(exact == calculated["exact"], "zero rule equivalence")
    if exact:
        category = "exact_zero"
    elif a <= b:
        category = "overlap_or_tie"
    elif a <= 0:
        category = "separated_missing"
        require(calculated["false_positive_count"] == 0, "missing-only separation")
    else:
        category = "separated_extra"
        require(b > 0 and calculated["false_negative_count"] == 0, "extra-only separation")
    return {
        **{k: row[k] for k in ("index", "subject", "dimension", "group", "prompt_ids")},
        "expected_set_sha256": hashlib.sha256(encoded(truth)).hexdigest(),
        "a": a,
        "b": b,
        "g": a - b,
        "minimum_true_id": low,
        "maximum_false_id": high,
        "exact": exact,
        "strict_separation": separated,
        "category": category,
        "false_positive": [{"id": i, "score": scores[i]} for i in selected if i not in truth],
        "false_negative": [{"id": i, "score": scores[i]} for i in truth if i not in selected],
    }


def totals(rows, interval=True):
    require(rows, "nonempty aggregation")
    result = {
        "query_count": len(rows),
        "exact_count": sum(r["exact"] for r in rows),
        "strict_separation_count": sum(r["strict_separation"] for r in rows),
        "false_positive_count": sum(len(r["false_positive"]) for r in rows),
        "false_negative_count": sum(len(r["false_negative"]) for r in rows),
        "category_counts": {c: sum(r["category"] == c for r in rows) for c in CATEGORIES},
    }
    if interval:
        lower = min(rows, key=lambda r: (-r["b"], r["index"]))
        upper = min(rows, key=lambda r: (r["a"], r["index"]))
        result["shared_threshold"] = {
            "A": upper["a"],
            "B": lower["b"],
            "exists": lower["b"] < upper["a"],
            "lower_inclusive": True,
            "upper_exclusive": True,
            "A_query_index": upper["index"],
            "B_query_index": lower["index"],
        }
    return result


def source_totals(rows):
    return {
        "query_count": len(rows),
        "exact_count": sum(r["metrics"]["exact"] for r in rows),
        **{
            k: math.fsum(r["metrics"][k] for r in rows) / len(rows)
            for k in ("precision", "recall", "f1", "set_size")
        },
        **{
            k: sum(r["metrics"][k] for r in rows)
            for k in ("false_positive_count", "false_negative_count", "serialization_compatible")
        },
        "strict_separation_count": sum(r["strict_separation"] for r in rows),
    }


def align(children):
    reference = children[1729]["responses"]
    require(
        len({(r["subject"], r["dimension"]) for r in reference}) == len(reference), "unique queries"
    )
    keys = ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_ids")
    for child in children.values():
        require(len(child["responses"]) == len(reference), "aligned count")
        for index, (left, right) in enumerate(zip(reference, child["responses"], strict=True)):
            same(right["index"], index, "sequential indexes")
            same({k: left[k] for k in keys}, {k: right[k] for k in keys}, "query alignment")


def seed_report(seed, child, digest):
    require(child["complete"] is True and child["query_count"] == 222, "source completion")
    same(child["product_token_ids"], list(range(1024, 2049)), "full product columns")
    require(len(child["responses"]) == 222, "full query coverage")
    rows = [reduce_row(r, child["product_token_ids"]) for r in child["responses"]]
    same(child["aggregate"], source_totals(child["responses"]), "accepted aggregate metrics")
    same(sorted(child["groups"]), list(GROUPS), "source groups")
    for group in GROUPS:
        source = [r for r in child["responses"] if r["group"] == group]
        same(child["groups"][group], source_totals(source), "accepted group metrics")
    return {
        "complete": True,
        "seed": seed,
        "child_sha256": digest,
        "query_count": len(rows),
        "rows": rows,
        "aggregate": totals(rows),
        "groups": {g: totals([r for r in rows if r["group"] == g]) for g in GROUPS},
    }


def pool(reports, seeds):
    rows = [r for seed in seeds for r in reports[seed]["rows"]]
    return {
        "aggregate": totals(rows, False),
        "groups": {g: totals([r for r in rows if r["group"] == g], False) for g in GROUPS},
    }


def overlap(reports):
    histogram, failures = {}, []
    for index, row in enumerate(reports[1729]["rows"]):
        seeds = [s for s in SEEDS if not reports[s]["rows"][index]["exact"]]
        key = ",".join(str(s) for s in seeds) if seeds else "none"
        histogram[key] = histogram.get(key, 0) + 1
        if seeds:
            failures.append(
                {**{k: row[k] for k in ("index", "subject", "dimension", "group")}, "seeds": seeds}
            )
    return {
        "shared_queries": len(reports[1729]["rows"]),
        "failure_seed_histogram": dict(sorted(histogram.items())),
        "failed_queries": failures,
    }


def execution_receipt(receipt, summary_hash, stdout_hash):
    require(type(receipt["exit_code"]) is int and receipt["exit_code"] == 0, "terminal exit")
    require(receipt["terminal_completion_observed_by_primary"] is True, "observed completion")
    same(receipt["summary_sha256"], summary_hash, "terminal summary")
    same(receipt["stdout_sha256"], stdout_hash, "terminal stdout")


def audit(summary_path):
    root = Path(__file__).resolve().parents[3]
    folder = root / "runs/learning/bilinear-error-geometry-v1"
    require(summary_path.resolve() == (folder / "summary.json").resolve(), "campaign location")
    summary = read(summary_path)
    require(
        summary["complete"] is True and summary["acceptance"] is False,
        "complete unaccepted primary",
    )
    same(summary["campaign_version"], "bilinear-error-geometry-v1", "campaign")
    same(summary["script_sha256"], RUNNER, "frozen primary script")
    same(summary["plan_sha256"], PLAN, "frozen plan")
    inputs = {}
    plan = root / "docs/experiments/2026-09-25-bilinear-error-geometry-plan.md"
    bound(plan, PLAN, inputs)
    children, digests = authenticated_children(root, inputs)
    script = root / "scripts/diagnose_bilinear_errors.py"
    bound(script, RUNNER, inputs)
    receipt_path = root / "runs/learning/bilinear-error-geometry-runner-tests-v1/receipt.json"
    receipt = bound(receipt_path, TEST_RECEIPT, inputs)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["diagnostic_executed"] is False,
        "synthetic tests",
    )
    same(receipt["tested_script_sha256"], RUNNER, "tested runner")
    bound(root / "tests/unit/test_bilinear_error_geometry.py", receipt["test_sha256"], inputs)
    stdout = Path(receipt["stdout"]["path"])
    bound(stdout, receipt["stdout"]["sha256"], inputs)
    same(summary["input_sha256"], inputs, "exact allowed input inventory")
    snapshots = {
        "script.py": script.read_bytes(),
        "plan.md": plan.read_bytes(),
        "test-receipt.json": receipt_path.read_bytes(),
        "test-stdout.txt": stdout.read_bytes(),
        "inputs.json": encoded(inputs),
    }
    snapshot_hashes = {
        str(folder / ("summary." + k)): hashlib.sha256(v).hexdigest() for k, v in snapshots.items()
    }
    same(summary["snapshot_sha256"], snapshot_hashes, "exact snapshot inventory")
    for path, digest in snapshot_hashes.items():
        bound(path, digest, inputs)
    summary_hash = sha(summary_path)
    bound(summary_path, summary_hash, inputs)
    terminal_path = folder / "execution-receipt.json"
    terminal = bound(terminal_path, sha(terminal_path), inputs)
    actual_stdout = folder / "primary-stdout.txt"
    bound(actual_stdout, terminal["stdout_sha256"], inputs)
    execution_receipt(terminal, summary_hash, sha(actual_stdout))
    align(children)
    same([r["seed"] for r in summary["seed_artifacts"]], list(SEEDS), "result seed coverage")
    reports, per_seed = {}, []
    for entry in summary["seed_artifacts"]:
        seed = entry["seed"]
        path = folder / f"seed-{seed}.json"
        require(Path(entry["path"]).resolve() == path.resolve(), "seed result location")
        saved = bound(path, entry["sha256"], inputs)
        reports[seed] = seed_report(seed, children[seed], digests[seed])
        same(saved, reports[seed], "independent full seed reductions")
        per_seed.append(
            {
                "seed": seed,
                "report_sha256": entry["sha256"],
                "child_sha256": digests[seed],
                "aggregate": reports[seed]["aggregate"],
                "groups": reports[seed]["groups"],
            }
        )
    fresh, all_three, shared = pool(reports, (1730, 1731)), pool(reports, SEEDS), overlap(reports)
    same(summary["fresh_two"], fresh, "fresh counts")
    same(summary["all_three"], all_three, "contextual counts")
    same(summary["overlap"], shared, "shared query overlap")
    require(all(sha(p) == h for p, h in inputs.items()), "final evidence invariance")
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": summary_hash,
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN,
        "primary_script_sha256": RUNNER,
        "input_sha256": inputs,
        "per_seed": per_seed,
        "fresh_two": fresh,
        "all_three": all_three,
        "overlap": shared,
        "limitations": [
            "Independent arithmetic on accepted saved scores, not a neural replay.",
            "666 observations repeat 222 validation identities; seed1729 was used for selection.",
            "Intervals use labels; they do not select thresholds or estimate new performance.",
            "No checkpoint, training examples, graph or protected-test artifacts opened.",
        ],
    }


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    out = args.summary.parent / "independent-audit.json"
    require(not out.exists(), "audit output already exists")
    result = audit(args.summary)
    require("torch" not in sys.modules, "stdlib audit boundary")
    write(out, result)
    print(
        json.dumps(
            {
                "audit_passed": True,
                "audit_sha256": sha(out),
                "summary_sha256": result["summary_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
