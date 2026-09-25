"""Post-hoc descriptive diagnosis of the failed, audited margin screen.

Declared before execution: consume only authenticated saved validation vectors and
training logs. Describe non-subject score spread (range and population standard
deviation), positive/negative means, extrema, separation gaps and unit-margin hinge
activity; summarize each quantity by query-weighted mean and median, with group
partitions. Preserve all logged CE, symmetric BCE and margin trajectories. Compare
logged symmetric BCE with ln(2), the zero-logit BCE reference, without claiming an
optimization mechanism has been established. No model import/forward, training,
new selection policy, parameter tuning, protected-test measurement or promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
import sys
from pathlib import Path

PINS = {
    "summary.json": "39aafceac5c81d73bc7f38b2cd4f619c23d1480829456e031a352c5c1c6bd701",
    "independent-audit.json": "17afc9ca0fedf184ac9e9f2794947306883d4288860b22124cd9eb0c7751bcd9",
    "evaluation-control.json": "2860849b418df41153fefabf6d5b3c6a7d9b5489e696af161e596842f1236d3e",
    "evaluation-treatment.json": "04baa56af9a77858d084c7108a527a84499a30df3e595cddbce4e87cd222832b",
}
COLUMNS = list(range(1024, 2049))
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate(path, expected, inputs):
    require(path.is_file() and sha(path) == expected, f"input identity mismatch: {path}")
    inputs[str(path)] = expected
    return json.loads(path.read_text(encoding="utf-8"))


def average(values):
    return math.fsum(values) / len(values)


def measure(row):
    logits = row["symmetric_relation_logits"]
    require(
        len(logits) == 1025
        and all(
            type(v) is float
            and math.isfinite(v)
            and struct.unpack("f", struct.pack("f", v))[0] == v
            for v in logits
        ),
        "FP32 vector",
    )
    subject = row["prompt_ids"][1]
    truth = set(row["expected_set_ids"])
    allowed = set(COLUMNS) - {subject}
    require(
        subject in COLUMNS
        and truth < allowed
        and truth
        and len(truth) == len(row["expected_set_ids"]),
        "subject-excluded nonempty positive/negative masks",
    )
    values = [logits[i - 1024] for i in sorted(allowed)]
    positive = [logits[i - 1024] for i in sorted(truth)]
    negative = [logits[i - 1024] for i in sorted(allowed - truth)]
    a, b = min(positive), max(negative)
    gap = a - b
    require(
        a == row["min_true_logit"]
        and b == row["max_negative_logit"]
        and gap == row["gap"]
        and (b < a) == row["strict_separable"],
        "saved extrema or separation mismatch",
    )
    return {
        "index": row["index"],
        "subject": row["subject"],
        "dimension": row["dimension"],
        "group": row["group"],
        "non_subject_count": len(values),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "logit_min": min(values),
        "logit_max": max(values),
        "logit_range": max(values) - min(values),
        "logit_mean": average(values),
        "logit_population_std": statistics.pstdev(values),
        "positive_mean": average(positive),
        "negative_mean": average(negative),
        "positive_minus_negative_mean": average(positive) - average(negative),
        "min_true_logit": a,
        "max_negative_logit": b,
        "gap": gap,
        "unit_margin_hinge": max(0.0, 1.0 - gap),
        "unit_margin_active": gap < 1.0,
        "strict_separable": b < a,
    }


def aggregate(rows):
    require(rows, "empty partition")
    return {
        "query_count": len(rows),
        "strict_separation_count": sum(r["strict_separable"] for r in rows),
        "unit_margin_active_count": sum(r["unit_margin_active"] for r in rows),
        "statistics": {
            key: {
                "mean": average([r[key] for r in rows]),
                "median": statistics.median(r[key] for r in rows),
                "min": min(r[key] for r in rows),
                "max": max(r[key] for r in rows),
            }
            for key in MEASURES
        },
    }


def main():
    require("torch" not in sys.modules, "standalone stdlib-only execution required")
    script = Path(__file__).resolve()
    executing_bytes = script.read_bytes()
    root = script.parents[3]
    directory = root / "runs/learning/symmetric-margin-screen-v1"
    out = script.parent / "summary.json"
    require(
        not out.exists() and not (script.parent / "executed-script.py").exists(),
        "immutable output exists",
    )
    inputs = {str(script): hashlib.sha256(executing_bytes).hexdigest()}
    loaded = {name: authenticate(directory / name, pin, inputs) for name, pin in PINS.items()}
    summary, audit = loaded["summary.json"], loaded["independent-audit.json"]
    require(
        summary["complete"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["all_recomputed_outputs_equal"] is True
        and audit["summary_sha256"] == PINS["summary.json"],
        "primary audit incomplete or unbound",
    )
    auditor = directory / "independent-audit.py"
    require(sha(auditor) == audit["script_sha256"], "primary auditor changed")
    inputs[str(auditor)] = audit["script_sha256"]
    result = {
        "scope": __doc__,
        "complete": False,
        "script_sha256": inputs[str(script)],
        "input_sha256": inputs,
        "arms": [],
        "ln2_reference": math.log(2),
        "model_executed": False,
        "training_executed": False,
        "selection_or_tuning_performed": False,
        "limitations": [
            "Post-hoc descriptive analysis of one failed seed; not a causal mechanism test.",
            "Last-batch training losses are not epoch means.",
            (
                "Saved inference vectors are FP32; logged training validation used "
                "the original training numerical mode."
            ),
            (
                "Zero-logit BCE is ln(2); closeness alone does not prove every score "
                "is zero or every prediction random."
            ),
        ],
    }
    for arm in ("control", "treatment"):
        name = f"evaluation-{arm}.json"
        require(summary["report_sha256"][name] == PINS[name], "raw evaluation manifest binding")
        evaluation = loaded[name]
        require(
            evaluation["complete"] is True
            and evaluation["arm"] == arm
            and evaluation["query_count"] == len(evaluation["responses"]) == 222
            and evaluation["product_token_ids"] == COLUMNS,
            "validation row coverage",
        )
        training_name = f"training-{arm}.json"
        training = authenticate(
            directory / training_name, summary["report_sha256"][training_name], inputs
        )
        rows = [measure(row) for row in evaluation["responses"]]
        require([r["index"] for r in rows] == list(range(222)), "validation order")
        history = training["training_history"]
        require(
            [r["step"] for r in history] == list(range(200, 2001, 200)),
            "logged trajectory coverage",
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
        overall = aggregate(rows)
        require(
            overall["strict_separation_count"]
            == evaluation["head_diagnostics"]["strict_separation_count"],
            "head diagnostic count mismatch",
        )
        result["arms"].append(
            {
                "arm": arm,
                "quality": evaluation["metrics"],
                "overall": overall,
                "groups": {g: aggregate([r for r in rows if r["group"] == g]) for g in GROUPS},
                "rows": rows,
                "training_trajectory": history,
                "bce_log2_comparison": bce,
            }
        )
    control, treatment = result["arms"]
    require(
        [(r["subject"], r["dimension"], r["group"]) for r in control["rows"]]
        == [(r["subject"], r["dimension"], r["group"]) for r in treatment["rows"]],
        "paired identity",
    )
    result["checks"] = {
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
    result["descriptive_comparison"] = {
        key: {
            "control_mean": control["overall"]["statistics"][key]["mean"],
            "treatment_mean": treatment["overall"]["statistics"][key]["mean"],
            "treatment_minus_control_mean": treatment["overall"]["statistics"][key]["mean"]
            - control["overall"]["statistics"][key]["mean"],
        }
        for key in MEASURES
    }
    require(
        all(Path(name).is_file() and sha(Path(name)) == pin for name, pin in inputs.items()),
        "input/script drift",
    )
    with (script.parent / "executed-script.py").open("xb") as stream:
        stream.write(executing_bytes)
    result["complete"] = True
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "summary_sha256": sha(out),
                "checks": result["checks"],
                "comparison": result["descriptive_comparison"],
                "final_training": {a["arm"]: a["training_trajectory"][-1] for a in result["arms"]},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
