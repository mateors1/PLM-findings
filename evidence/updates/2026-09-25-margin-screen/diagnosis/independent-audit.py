"""Saved-only independent audit of the descriptive margin failure diagnosis.

Uses standard-library arithmetic directly on authenticated primary rows and logs.
Imports neither diagnosis nor primary evaluator helpers. No neural execution or
alternative policy evaluation. Exact comparisons, without numerical tolerances.
"""

import ast
import hashlib
import json
import math
import struct
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIMARY = ROOT / "runs/learning/symmetric-margin-screen-v1"
SUMMARY_SHA = "4d5c507efabb36ff2385cad1c77e50f36c3cd7466e9d48d65b895a2317b0f3c1"
DIAGNOSIS_SHA = "872972a489fe1979b5bf13a5fd3bf29d356863105a6a3411c7bb36cfa92d111f"
PRIMARY_SHA = "39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701"
PRIMARY_AUDIT_SHA = "17afc9ca0fedf184ac9e9f2794947306883d4288860b22124cd9eb0c7751bcd9"
EVALUATION_SHA = {
    "control": "2860849b418df41153fefabf6d5b3c6a7d9b5489e696af161e596842f1236d3e",
    "treatment": "04baa56af9a77858d084c7108a527a84499a30df3e595cddbce4e87cd222832b",
}
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
MEASURES = (
    "logit_min",
    "logit_max",
    "logit_range",
    "logit_mean",
    "logit_population_std",
    "positive_mean",
    "negative_mean",
    "positive_minus_negative_mean",
    "min_true_logit",
    "max_negative_logit",
    "gap",
    "unit_margin_hinge",
)
INPUTS = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path, expected):
    path = Path(path).resolve()
    digest = sha(path)
    require(digest == expected, f"hash mismatch: {path}")
    require(str(path) not in INPUTS or INPUTS[str(path)] == digest, f"input changed: {path}")
    INPUTS[str(path)] = digest
    return path


def read(path, expected):
    return json.loads(bind(path, expected).read_text(encoding="utf-8"))


def equal(actual, expected, message):
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        message,
    )


def population_std(values):
    # Independent exact rational second moment, then high-precision square root.
    # No call to the diagnosis's statistics.pstdev implementation.
    rationals = [Fraction.from_float(v) for v in values]
    count = len(rationals)
    first = sum(rationals, Fraction(0)) / count
    variance = sum((v * v for v in rationals), Fraction(0)) / count - first * first
    with localcontext() as context:
        context.prec = 100
        return float((Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt())


def midpoint(values):
    ordered = sorted(values)
    half = len(ordered) // 2
    return ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2


def describe(record):
    vector = record["symmetric_relation_logits"]
    require(len(vector) == 1025, "wrong head width")
    require(
        all(
            type(v) is float
            and math.isfinite(v)
            and struct.unpack("!f", struct.pack("!f", v))[0] == v
            for v in vector
        ),
        "nonfinite or non-FP32 saved vector",
    )
    subject = record["prompt_ids"][1]
    wanted = set(record["expected_set_ids"])
    require(type(subject) is int and 1024 <= subject <= 2048, "invalid subject column")
    require(
        record["expected_set_ids"] == sorted(wanted)
        and all(type(i) is int and 1024 <= i <= 2048 and i != subject for i in wanted),
        "invalid subject-excluded truth IDs",
    )
    positive, negative = [], []
    values = []
    for offset, value in enumerate(vector):
        token = 1024 + offset
        if token != subject:
            values.append(value)
            (positive if token in wanted else negative).append(value)
    require(positive and negative and len(values) == 1024, "empty class or bad universe")
    low, high = min(positive), max(negative)
    gap = low - high
    equal(
        [record[k] for k in ("min_true_logit", "max_negative_logit", "gap", "strict_separable")],
        [low, high, gap, low > high],
        "primary saved extrema inconsistent",
    )
    positive_mean = math.fsum(positive) / len(positive)
    negative_mean = math.fsum(negative) / len(negative)
    return {
        **{k: record[k] for k in ("index", "subject", "dimension", "group")},
        "non_subject_count": len(values),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "logit_min": min(values),
        "logit_max": max(values),
        "logit_range": max(values) - min(values),
        "logit_mean": math.fsum(values) / len(values),
        "logit_population_std": population_std(values),
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "positive_minus_negative_mean": positive_mean - negative_mean,
        "min_true_logit": low,
        "max_negative_logit": high,
        "gap": gap,
        "unit_margin_hinge": max(0.0, 1.0 - gap),
        "unit_margin_active": gap < 1.0,
        "strict_separable": low > high,
    }


def aggregate(rows):
    require(rows, "empty diagnostic partition")
    result = {
        "query_count": len(rows),
        "strict_separation_count": sum(r["strict_separable"] for r in rows),
        "unit_margin_active_count": sum(r["unit_margin_active"] for r in rows),
        "statistics": {},
    }
    for key in MEASURES:
        values = [r[key] for r in rows]
        result["statistics"][key] = {
            "mean": math.fsum(values) / len(values),
            "median": midpoint(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def main():
    output = HERE / "independent-audit.json"
    require(not output.exists(), "refusing existing audit")
    require("torch" not in sys.modules, "stdlib-only execution required")
    own_sha = sha(__file__)
    summary = read(HERE / "summary.json", SUMMARY_SHA)
    bind(HERE / "diagnose.py", DIAGNOSIS_SHA)
    bind(HERE / "executed-script.py", DIAGNOSIS_SHA)
    require(
        summary["script_sha256"] == DIAGNOSIS_SHA and summary["complete"] is True,
        "diagnosis completion binding",
    )
    docstring = ast.get_docstring(ast.parse((HERE / "executed-script.py").read_text()), clean=False)
    equal(summary["scope"], docstring, "declared executed diagnosis scope")
    for name, digest in summary["input_sha256"].items():
        bind(name, digest)
    primary = read(PRIMARY / "summary.json", PRIMARY_SHA)
    accepted_audit = read(PRIMARY / "independent-audit.json", PRIMARY_AUDIT_SHA)
    bind(PRIMARY / "independent-audit.py", accepted_audit["script_sha256"])
    require(
        primary["complete"] is True
        and accepted_audit["complete"] is True
        and accepted_audit["audit_passed"] is True
        and accepted_audit["all_recomputed_outputs_equal"] is True
        and accepted_audit["summary_sha256"] == PRIMARY_SHA,
        "primary evidence not completely audited",
    )
    require(primary["gate"]["numerical_gates_passed"] is False, "wrong scientific context")
    require([arm["arm"] for arm in summary["arms"]] == ["control", "treatment"], "arm order")
    rebuilt = []
    for observed in summary["arms"]:
        arm = observed["arm"]
        name = f"evaluation-{arm}.json"
        require(primary["report_sha256"][name] == EVALUATION_SHA[arm], "raw evaluation binding")
        evaluation = read(PRIMARY / name, EVALUATION_SHA[arm])
        require(
            evaluation["complete"] is True and evaluation["arm"] == arm, "evaluation completion"
        )
        equal(evaluation["product_token_ids"], list(range(1024, 2049)), "canonical product mapping")
        require(evaluation["query_count"] == len(evaluation["responses"]) == 222, "coverage")
        equal(
            [r["index"] for r in evaluation["responses"]], list(range(222)), "query index coverage"
        )
        rows = [describe(row) for row in evaluation["responses"]]
        name = f"training-{arm}.json"
        training = read(PRIMARY / name, primary["report_sha256"][name])
        histories = []
        for artifact_path, digest in training["artifact_sha256"].items():
            if Path(artifact_path).name in ("metrics.jsonl", "training-result.json"):
                bound = bind(artifact_path, digest)
                if bound.name == "metrics.jsonl":
                    histories.append([json.loads(line) for line in bound.read_text().splitlines()])
                else:
                    histories.append(json.loads(bound.read_text())["history"])
        require(len(histories) == 2, "direct training journal/result missing")
        history = training["training_history"]
        for value in histories:
            equal(value, history, "direct journal/training receipt differs")
        equal(
            [r["step"] for r in history],
            [float(i) for i in range(200, 2001, 200)],
            "trajectory coverage",
        )
        bce = [
            {
                "step": r["step"],
                "validation_symmetric_bce": r["validation_symmetric_relation_loss"],
                "ln2_minus_validation_symmetric_bce": math.log(2)
                - r["validation_symmetric_relation_loss"],
                "validation_symmetric_bce_over_ln2": r["validation_symmetric_relation_loss"]
                / math.log(2),
            }
            for r in history
        ]
        result = {
            "arm": arm,
            "quality": evaluation["metrics"],
            "overall": aggregate(rows),
            "groups": {g: aggregate([r for r in rows if r["group"] == g]) for g in GROUPS},
            "rows": rows,
            "training_trajectory": history,
            "bce_log2_comparison": bce,
        }
        equal(observed, result, arm + " rows/aggregates/trajectory mismatch")
        equal(
            result["overall"]["strict_separation_count"],
            evaluation["head_diagnostics"]["strict_separation_count"],
            "primary separation count",
        )
        rebuilt.append(result)
    control, treatment = rebuilt
    equal(
        [[r[k] for k in ("subject", "dimension", "group")] for r in control["rows"]],
        [[r[k] for k in ("subject", "dimension", "group")] for r in treatment["rows"]],
        "paired query identities",
    )
    comparisons = {
        key: {
            "control_mean": control["overall"]["statistics"][key]["mean"],
            "treatment_mean": treatment["overall"]["statistics"][key]["mean"],
            "treatment_minus_control_mean": treatment["overall"]["statistics"][key]["mean"]
            - control["overall"]["statistics"][key]["mean"],
        }
        for key in MEASURES
    }
    equal(summary["descriptive_comparison"], comparisons, "descriptive comparison")
    equal(summary["ln2_reference"], math.log(2), "zero-logit BCE reference")
    expected_checks = {
        "primary_audit_authenticated": True,
        "raw_reports_authenticated": True,
        "saved_extrema_and_separation_exactly_reconstructed": True,
        "all_222_treatment_saved_vector_hinges_active": treatment["overall"][
            "unit_margin_active_count"
        ]
        == 222,
        "all_222_treatment_logged_validation_hinges_active_at_every_logged_step": all(
            r["validation_symmetric_margin_active_count"]
            == r["validation_symmetric_margin_query_count"]
            == 222
            and r["validation_symmetric_margin_active_fraction"] == 1.0
            for r in treatment["training_trajectory"]
        ),
        "torch_not_imported": "torch" not in sys.modules,
    }
    equal(summary["checks"], expected_checks, "diagnostic factual checks")
    require(
        all(
            summary[k] is False
            for k in ("model_executed", "training_executed", "selection_or_tuning_performed")
        ),
        "diagnostic execution boundary",
    )
    require(
        "not a causal mechanism test" in " ".join(summary["limitations"]),
        "causal limitation absent",
    )
    require(
        "does not prove every score" in " ".join(summary["limitations"]),
        "zero-logit limitation absent",
    )
    require(all(sha(name) == digest for name, digest in INPUTS.items()), "input drift")
    require(sha(__file__) == own_sha and "torch" not in sys.modules, "auditor drift/import")
    result = {
        "complete": True,
        "audit_passed": True,
        "all_recomputed_outputs_equal": True,
        "summary_sha256": SUMMARY_SHA,
        "script_sha256": own_sha,
        "primary_summary_sha256": PRIMARY_SHA,
        "primary_audit_sha256": PRIMARY_AUDIT_SHA,
        "query_count": 444,
        "queries_per_arm": 222,
        "non_subject_values_per_row": 1024,
        "non_subject_values_checked": 444 * 1024,
        "training_logged_steps_checked": 20,
        "model_executed": False,
        "causal_claim_established": False,
        "comparisons": comparisons,
        "arms": [{k: arm[k] for k in ("arm", "overall", "groups")} for arm in rebuilt],
        "checks": expected_checks,
        "input_sha256": dict(sorted(INPUTS.items())),
        "limitations": [
            "Descriptive saved-evidence check, not an optimization mechanism experiment.",
            "Primary quality arithmetic inherits the authenticated accepted primary audit; new diagnosis arithmetic is independently recomputed.",
            "Population variance uses exact rational moments and a 100-digit Decimal square root; all saved floating results match exactly without tolerance.",
            "Saved FP32 inference vectors and training-mode validation logs remain distinct measurements.",
            "No new selection, model forward, training, threshold search or protected-test measurement.",
        ],
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "audit_sha256": sha(output),
                "query_count": 444,
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
