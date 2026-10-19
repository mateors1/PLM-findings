"""Independent saved-only set-algebra audit using bitsets and rational sums.

No runner, runtime or model imports. All pure-operation subset witnesses are
label-assisted diagnostics, never candidate selection inputs.
"""

import argparse
import hashlib
import itertools
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN_SHA = "7fb6f28e9d432799590fd7dff9b9269ffc732ecce8c4f496ba0c1e3521c27a5f"
UPSTREAM_SHA = "1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183"
UPSTREAM_AUDIT_SHA = "eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95"
DECISION_SHA = "70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601"
SEEDS = (1729, 1730, 1731)
BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"
POLICY8 = BASE + "+first8-pair28-symmetric-set-logit-sum-v1"
ORDER4 = ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
PAIRS = tuple(itertools.combinations(range(1, 9), 2))
ORDER36 = ORDER4 + tuple((i,) for i in range(5, 9)) + tuple(p for p in PAIRS if p not in ORDER4)
SUBSETS = tuple(
    ranks for size in range(1, 9) for ranks in itertools.combinations(range(1, 9), size)
)
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
INPUTS = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path, digest):
    path = Path(path).resolve()
    require(sha(path) == digest, f"hash mismatch: {path}")
    require(str(path) not in INPUTS or INPUTS[str(path)] == digest, f"input drift: {path}")
    INPUTS[str(path)] = digest
    return path


def read(path, digest):
    return json.loads(bind(path, digest).read_text(encoding="utf-8"))


def exact(actual, expected, message):
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        message,
    )


def vector(values):
    require(isinstance(values, list) and len(values) == 1025, "wrong membership vector width")
    for value in values:
        require(type(value) is float and math.isfinite(value), "invalid head value")
        require(struct.unpack("!f", struct.pack("!f", value))[0] == value, "head value not FP32")


def canonical_sum(ids, values):
    require(
        ids == sorted(set(ids)) and all(type(i) is int and 1024 <= i <= 2048 for i in ids),
        "noncanonical product set",
    )
    return float(sum((Fraction.from_float(values[i - 1024]) for i in ids), Fraction(0)))


def source_set(raw, prompt, rank):
    sequence = raw["token_ids"]
    require(isinstance(sequence, list) and all(type(i) is int for i in sequence), "raw IDs")
    require(sequence[:5] == prompt and 6 <= len(sequence) <= 512, "raw prompt/budget")
    ended = sequence[-1] == 2
    products = sequence[5:-1] if ended else sequence[5:]
    require(all(1024 <= i <= 2048 for i in products), "raw product grammar")
    require(ended or len(sequence) == 512, "source stopped before bound")
    valid = ended and len(sequence) >= 7
    error = (
        None
        if valid
        else (
            "generated response is shorter than the protocol grammar"
            if len(sequence) < 7
            else "response must terminate with EOS"
        )
    )
    require(
        raw["terminated"] is ended and raw["protocol_valid"] is valid and raw["error"] == error,
        "raw completion flags",
    )
    require(
        raw["decoding"] == BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
        "source decoding policy",
    )
    require(
        len(raw["targets"]) == len(products)
        and all(isinstance(t, str) and t.startswith("PKM_") for t in raw["targets"]),
        "source target identifiers",
    )
    eligible = valid and ended and len(products) == len(set(products))
    # SAME postprocessing belongs to the frozen policy; raw subject emissions are legal.
    members = set(products) - {prompt[1]} if valid and ended else set(products)
    return sorted(members), eligible, list(zip(products, raw["targets"], strict=True))


def bits(ids):
    require(
        isinstance(ids, list)
        and ids == sorted(set(ids))
        and all(type(i) is int and 1024 <= i <= 2048 for i in ids),
        "canonical product IDs",
    )
    return sum(1 << (i - 1024) for i in ids)


def ids(mask):
    require(type(mask) is int and mask >= 0 and mask.bit_length() <= 1025, "invalid product bitset")
    result = []
    while mask:
        low = mask & -mask
        result.append(1024 + low.bit_length() - 1)
        mask ^= low
    return result


def operate(source_bits, ranks, intersection=False):
    require(bool(ranks), "empty source subset")
    mask = source_bits[ranks[0] - 1]
    for rank in ranks[1:]:
        mask = mask & source_bits[rank - 1] if intersection else mask | source_bits[rank - 1]
    return mask


def slots(source_bits, values, eligible=None, include_intersections=True):
    """Predictor construction has no truth, group or error inputs."""
    vector(values)
    require(len(source_bits) == 8, "eight sources required")
    eligible = [True] * 8 if eligible is None else eligible
    require(len(eligible) == 8 and all(type(x) is bool for x in eligible), "source eligibility")
    order = [(ranks, False) for ranks in ORDER36]
    if include_intersections:
        order += [(ranks, True) for ranks in PAIRS]
    result = []
    for index, (ranks, intersection) in enumerate(order, 1):
        members = ids(operate(source_bits, ranks, intersection))
        result.append(
            {
                "slot": index,
                "kind": "pair_intersection"
                if intersection
                else "original"
                if len(ranks) == 1
                else "pair_composition",
                "source_ranks": list(ranks),
                "set_ids": members,
                "source_eligible": all(eligible[r - 1] for r in ranks),
                "score": canonical_sum(members, values),
            }
        )
    return result


def select(candidates):
    eligible = [slot for slot in candidates if slot["source_eligible"]]
    chosen = max(eligible, key=lambda s: (s["score"], -s["slot"])) if eligible else candidates[0]
    return {
        f"selected_{k}": chosen[k]
        for k in ("slot", "kind", "source_ranks", "set_ids", "source_eligible")
    } | {"fallback_no_valid_source": not eligible}


def metric(selection, truth):
    successful = selection["selected_source_eligible"] and not selection["fallback_no_valid_source"]
    predicted = bits(selection["selected_set_ids"]) if successful else 0
    wanted = bits(truth)
    require(wanted != 0, "empty validation teacher")
    tp = (predicted & wanted).bit_count()
    precision = tp / predicted.bit_count() if predicted else 0.0
    recall = tp / wanted.bit_count()
    return {
        "exact": bool(successful and predicted == wanted),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "set_size": predicted.bit_count(),
        "success": bool(successful),
    }


def errors(predicted, wanted):
    return {
        "false_positive_ids": ids(predicted & ~wanted),
        "false_negative_ids": ids(wanted & ~predicted),
    }


def witness(source_bits, truth, intersection=False):
    """Label-assisted exact attainability; never called by selection."""
    wanted = bits(truth)
    first = None
    # Enumerate all255 even after a witness so coverage is explicit and fixed.
    for ranks in SUBSETS:
        if operate(source_bits, ranks, intersection) == wanted and first is None:
            first = list(ranks)
    return {
        "exact_available": first is not None,
        "min_sources": len(first) if first is not None else None,
        "witness": first,
    }


def available(candidates, truth):
    wanted = bits(truth)
    return any(s["source_eligible"] and bits(s["set_ids"]) == wanted for s in candidates)


def baseline_state(exact, available_exact, covered):
    if exact:
        require(available_exact, "selected exact absent from pool")
        return "selected_exact"
    if available_exact:
        return "available_exact_but_missed"
    return (
        "unavailable_with_truth_covered"
        if covered
        else "unavailable_and_truth_missing_from_all_sources"
    )


def validate_base(row, vocabulary, index):
    prompt = row["prompt_ids"]
    require(row["index"] == index and type(row["index"]) is int, "query index")
    require(row["dimension"] in ("TYPE", "COLOR") and row["group"] in GROUPS, "dimension/group")
    require((row["group"] == "COLOR") == (row["dimension"] == "COLOR"), "dimension/group agreement")
    require(len(prompt) == 5 and all(type(i) is int for i in prompt), "prompt type/shape")
    require(
        prompt == [1, prompt[1], 32 if row["dimension"] == "TYPE" else 33, 34, 5]
        and 1024 <= prompt[1] <= 2048,
        "protocol prompt",
    )
    require(vocabulary[prompt[1]] == row["subject"], "subject mapping")
    truth = row["expected_set_ids"]
    require(bits(truth) != 0 and prompt[1] not in truth, "teacher subject/nonempty")
    exact(sorted(vocabulary[i] for i in truth), row["expected"], "teacher ID/key mapping")
    comp = row["composition"]
    paths = comp["source_paths"]
    require(len(paths) == 8, "source count")
    first = []
    source_bits = []
    removed = []
    for rank, path in enumerate(paths, 1):
        members, eligible, mapping = source_set(path, prompt, rank)
        require(eligible and path["error"] is None, "upstream source invalid/incomplete/repeated")
        require(all(vocabulary[i] == key for i, key in mapping), "raw path ID/key mapping")
        first.append(path["token_ids"][5])
        source_bits.append(bits(members))
        removed.append(prompt[1] in path["token_ids"][5:-1])
    require(
        len(set(first)) == 8 and all(1024 <= i <= 2048 for i in first),
        "branch-rank first-token distinctness",
    )
    reconstructed = slots(
        source_bits, row["symmetric_relation_logits"], include_intersections=False
    )
    chosen = select(reconstructed)
    exact(
        comp,
        {"source_paths": paths, "slots": reconstructed, **chosen, "policy": POLICY8},
        "exact upstream36 reconstruction",
    )
    exact(row["metrics"]["wide"], metric(chosen, truth), "upstream selected metrics")
    exact(row["subject_removed"], removed, "upstream subject removal")
    require(
        row["availability"]["wide"]["exact_available"] == available(reconstructed, truth),
        "upstream availability",
    )
    return source_bits, removed, reconstructed, chosen


def upstream():
    directory = ROOT / "runs/learning/wide-first-choice-v1"
    summary = read(directory / "summary.json", UPSTREAM_SHA)
    audit = read(directory / "independent-audit.json", UPSTREAM_AUDIT_SHA)
    decision = read(directory / "decision.json", DECISION_SHA)
    require(
        summary["complete"] is True and summary["final_identity_check"] is True,
        "upstream incomplete",
    )
    require(
        audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == UPSTREAM_SHA,
        "upstream audit chain",
    )
    require(
        decision["evidence_accepted"] is True and decision["fixed_quality_gate_passed"] is True,
        "upstream not accepted",
    )
    require(
        decision["summary_sha256"] == UPSTREAM_SHA
        and decision["audit_sha256"] == UPSTREAM_AUDIT_SHA,
        "decision chain",
    )
    require(
        summary["acceptance"] is False
        and audit["acceptance"] is False
        and decision["policy_promoted"] is False,
        "separate owner authority",
    )
    require(
        summary["training_executed"] is False and summary["protected_test_used"] is False,
        "upstream scope",
    )
    exact(summary["gate"], audit["gate"], "upstream audit gate")
    exact(decision["gate"], audit["gate"], "upstream owner gate")
    bind(directory / "independent-audit.py", audit["script_sha256"])
    bind(directory / "summary.script.py", summary["script_sha256"])
    records = summary["reports"]
    require([r["seed"] for r in records] == list(SEEDS), "upstream seeds")
    reports = []
    for record in records:
        report = read(record["path"], record["sha256"])
        require(
            report["complete"] is True
            and report["baseline_replay_exact"] is True
            and report["seed"] == record["seed"],
            "upstream report incomplete",
        )
        require(
            report["query_count"] == 222 and len(report["responses"]) == 222,
            "upstream row coverage",
        )
        exact(report["product_token_ids"], list(range(1024, 2049)), "upstream product columns")
        reports.append(report)
    manifest = {str(Path(p).resolve()): h for p, h in summary["input_sha256"].items()}
    matches = [(p, h) for p, h in manifest.items() if Path(p).name == "vocabulary.json"]
    require(len(matches) == 1, "ambiguous vocabulary input")
    vocabulary = read(*matches[0])["tokens"]
    require(len(vocabulary) == 2049 and len(set(vocabulary)) == 2049, "vocabulary shape")
    return summary, audit, decision, reports, vocabulary


def baseline_metric_totals(metrics):
    require(bool(metrics), "empty metric aggregate")
    return {
        "exact_count": sum(m["exact"] for m in metrics),
        **{
            k: math.fsum(m[k] for m in metrics) / len(metrics)
            for k in ("precision", "recall", "f1", "set_size")
        },
    }


def quality_gate(seed_metrics, overall, groups, baseline_ok=True, invariants=True):
    checks = {
        "complete_baseline_reproduction": baseline_ok and invariants,
        "per_seed_exact_nonregression": all(
            s["prediction"]["exact_count"] >= s["baseline"]["exact_count"] for s in seed_metrics
        ),
        "per_seed_f1_nonregression": all(
            s["prediction"]["f1"] >= s["baseline"]["f1"] for s in seed_metrics
        ),
        "pooled_exact_improvement": overall["prediction"]["exact_count"] > 603,
        "pooled_group_exact_nonregression": all(
            g["prediction"]["exact_count"] >= g["baseline"]["exact_count"] for g in groups.values()
        ),
    }
    return {"checks": checks, "quality_passed": all(checks.values())}


def reconstruct(base, vocabulary, index, validated=None):
    source, removed, old_slots, baseline = (
        validate_base(base, vocabulary, index) if validated is None else validated
    )
    head = base["symmetric_relation_logits"]
    candidates = slots(source, head)
    exact(candidates[:36], old_slots, "old-first36 preservation")
    prediction = select(candidates)
    # Truth enters only after both fixed predictor selections are complete.
    truth = base["expected_set_ids"]
    wanted = bits(truth)
    measures = {"baseline": metric(baseline, truth), "prediction": metric(prediction, truth)}
    unions = witness(source, truth)
    intersections = witness(source, truth, True)
    covered = (wanted & ~operate(source, tuple(range(1, 9)))) == 0
    old_available = available(old_slots, truth)
    availability = {
        "baseline": old_available,
        "policy": available(candidates, truth),
        "all_unions": unions["exact_available"],
        "all_intersections": intersections["exact_available"],
        "either_pure_family": unions["exact_available"] or intersections["exact_available"],
    }
    return {
        **{
            k: base[k]
            for k in (
                "index",
                "subject",
                "dimension",
                "group",
                "prompt_ids",
                "expected",
                "expected_set_ids",
            )
        },
        "source_paths": base["composition"]["source_paths"],
        "source_set_ids": [ids(s) for s in source],
        "source_subject_removed": removed,
        "symmetric_relation_logits": head,
        "slots": candidates,
        "baseline": baseline,
        "prediction": prediction,
        "metrics": measures,
        "availability": availability,
        "diagnosis": {
            "baseline_state": baseline_state(measures["baseline"]["exact"], old_available, covered),
            "unions": unions,
            "intersections": intersections,
            "source_union_contains_truth": covered,
        },
        "errors": {
            "sources": [errors(s, wanted) for s in source],
            "baseline": errors(bits(baseline["selected_set_ids"]), wanted),
            "prediction": errors(bits(prediction["selected_set_ids"]), wanted),
        },
        "gained_exact": measures["prediction"]["exact"] and not measures["baseline"]["exact"],
        "lost_exact": measures["baseline"]["exact"] and not measures["prediction"]["exact"],
        "selection_changed": baseline["selected_set_ids"] != prediction["selected_set_ids"],
    }


STATES = (
    "selected_exact",
    "available_exact_but_missed",
    "unavailable_and_truth_missing_from_all_sources",
    "unavailable_with_truth_covered",
)


def totals(rows):
    result = {
        "query_count": len(rows),
        "gains": sum(r["gained_exact"] for r in rows),
        "losses": sum(r["lost_exact"] for r in rows),
        "changed_selection_count": sum(r["selection_changed"] for r in rows),
    }
    for arm, availability_key in (("baseline", "baseline"), ("prediction", "policy")):
        measures = baseline_metric_totals([r["metrics"][arm] for r in rows])
        measures["mean_set_size"] = measures.pop("set_size")
        result[arm] = {
            **measures,
            "exact_available": sum(r["availability"][availability_key] for r in rows),
            "available_exact_misses": sum(
                r["availability"][availability_key] and not r["metrics"][arm]["exact"] for r in rows
            ),
            "intersection_selections": sum(
                r[arm]["selected_kind"] == "pair_intersection" for r in rows
            ),
            "empty_selections": sum(not r[arm]["selected_set_ids"] for r in rows),
        }
    result["pure_family_availability"] = {
        key: sum(r["availability"][key] for r in rows)
        for key in ("all_unions", "all_intersections", "either_pure_family")
    }
    result["baseline_state_counts"] = {
        state: sum(r["diagnosis"]["baseline_state"] == state for r in rows) for state in STATES
    }
    return result


def aggregate(rows):
    return {
        "overall": totals(rows),
        "groups": {group: totals([r for r in rows if r["group"] == group]) for group in GROUPS},
    }


RUNNER_SHA = "9ea168346238a2b941d4132bfb249cc4169ce271f9631762d8e43aea2786d598"
RUNNER_RECEIPT_SHA = "d7197da002be4961e73b2063123d08cc8321ae029c048ee6334ce12c0cf6651c"


def check_manifest(manifest):
    require(isinstance(manifest, dict) and bool(manifest), "empty manifest")
    for path, digest in manifest.items():
        bind(path, digest)


def provenance(summary, folder, upstream_summary, upstream_audit):
    require(summary["campaign_version"] == "eight-source-set-algebra-v1", "campaign version")
    require(
        summary["complete"] is True and summary["final_identity_check"] is True,
        "incomplete execution",
    )
    require("error" not in summary and "identity_error" not in summary, "execution failure")
    for name in (
        "acceptance",
        "training_executed",
        "neural_generation_executed",
        "protected_test_used",
    ):
        require(summary[name] is False, f"scope violation: {name}")
    require(
        summary["plan_sha256"] == PLAN_SHA and summary["script_sha256"] == RUNNER_SHA,
        "frozen script/plan",
    )
    plan = ROOT / "docs/experiments/2026-09-25-eight-source-set-algebra-plan.md"
    script = ROOT / "scripts/evaluate_source_intersections.py"
    bind(plan, PLAN_SHA)
    bind(script, RUNNER_SHA)
    required = {str(plan.resolve()): PLAN_SHA, str(script.resolve()): RUNNER_SHA}
    directory = ROOT / "runs/learning/wide-first-choice-v1"
    required.update(
        {
            str((directory / name).resolve()): digest
            for name, digest in (
                ("summary.json", UPSTREAM_SHA),
                ("independent-audit.json", UPSTREAM_AUDIT_SHA),
                ("decision.json", DECISION_SHA),
                ("independent-audit.py", upstream_audit["script_sha256"]),
            )
        }
    )
    required.update(
        {str(Path(r["path"]).resolve()): r["sha256"] for r in upstream_summary["reports"]}
    )
    normalized = {str(Path(p).resolve()): h for p, h in summary["input_sha256"].items()}
    check_manifest(summary["input_sha256"])
    require(
        all(normalized.get(p) == h for p, h in required.items()),
        "required authenticated input missing",
    )
    snapshots = summary["snapshot_sha256"]
    expected = {
        str((folder / f"summary.{name}").resolve())
        for name in ("script.py", "plan.md", "test-receipt.json", "test-stdout.txt", "inputs.json")
    }
    require({str(Path(p).resolve()) for p in snapshots} == expected, "snapshot inventory")
    check_manifest(snapshots)
    bind(folder / "summary.script.py", RUNNER_SHA)
    bind(folder / "summary.plan.md", PLAN_SHA)
    receipt = read(folder / "summary.test-receipt.json", RUNNER_RECEIPT_SHA)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["tested_script_sha256"] == RUNNER_SHA
        and receipt["new_policy_measured"] is False
        and receipt["test_count"] == 38,
        "runner test receipt",
    )
    require(RUNNER_RECEIPT_SHA in normalized.values(), "runner receipt absent from input manifest")
    bind(ROOT / "tests/unit/test_source_intersections.py", receipt["tested_test_sha256"])
    bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"])
    bind(folder / "summary.test-stdout.txt", receipt["stdout"]["sha256"])
    exact(
        json.loads((folder / "summary.inputs.json").read_text(encoding="utf-8")),
        summary["input_sha256"],
        "input manifest snapshot",
    )
    require(
        type(summary["wall_seconds"]) is float
        and math.isfinite(summary["wall_seconds"])
        and summary["wall_seconds"] >= 0,
        "wall-time receipt",
    )


def own_tests():
    directory = ROOT / "runs/learning/eight-source-set-algebra-auditor-tests-v1"
    receipt = read(directory / "test-receipt.json", sha(directory / "test-receipt.json"))
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["model_execution"] is False
        and receipt["new_diagnostic_executed"] is False,
        "auditor test receipt",
    )
    require(
        receipt["tested_auditor_sha256"] == sha(__file__) and receipt["passed_count"] >= 39,
        "untested auditor bytes",
    )
    bind(directory / "test_auditor.py", receipt["tests_sha256"])
    bind(directory / "stdout.txt", receipt["stdout_sha256"])


def baseline_checks(reports, vocabulary, upstream_summary):
    validated = {}
    metrics = []
    identities = None
    for report in reports:
        seed = report["seed"]
        prepared = [
            validate_base(r, vocabulary, index) for index, r in enumerate(report["responses"])
        ]
        keys = [
            (r["subject"], r["dimension"], r["group"], r["expected_set_ids"])
            for r in report["responses"]
        ]
        require(len({(k[0], k[1]) for k in keys}) == 222, "duplicate validation query")
        if identities is not None:
            exact(keys, identities, "seed query order/labels")
        identities = keys
        metric_rows = [
            metric(p[3], r["expected_set_ids"])
            for p, r in zip(prepared, report["responses"], strict=True)
        ]
        totals = baseline_metric_totals(metric_rows)
        for key, value in totals.items():
            exact(value, report["overall"]["wide"][key], f"upstream seed aggregate {key}")
        for group in GROUPS:
            grouped = baseline_metric_totals(
                [
                    m
                    for m, r in zip(metric_rows, report["responses"], strict=True)
                    if r["group"] == group
                ]
            )
            for key, value in grouped.items():
                exact(
                    value, report["groups"][group]["wide"][key], f"upstream group aggregate {key}"
                )
        metrics.extend(metric_rows)
        validated[seed] = prepared
    pooled = baseline_metric_totals(metrics)
    require(
        pooled["exact_count"] == 603 and pooled["f1"] == 0.9834927532993669,
        "fixed historical baseline",
    )
    for key, value in pooled.items():
        exact(value, upstream_summary["overall"]["wide"][key], f"upstream pooled metric {key}")
    return validated


def audit_seed(report, upstream_report, vocabulary, validated):
    require(
        report["seed"] == upstream_report["seed"]
        and report["complete"] is True
        and report["baseline_reproduction_exact"] is True,
        "seed incomplete",
    )
    require(report["query_count"] == 222 and len(report["responses"]) == 222, "seed coverage")
    require("error" not in report and "failed_query" not in report, "seed failure evidence")
    exact(report["product_token_ids"], list(range(1024, 2049)), "column mapping")
    rows = []
    for index, (actual, base, prepared) in enumerate(
        zip(report["responses"], upstream_report["responses"], validated, strict=True)
    ):
        rebuilt = reconstruct(base, vocabulary, index, prepared)
        exact(actual, rebuilt, f"seed {report['seed']} row {index} independent reconstruction")
        rows.append(rebuilt)
    for name, value in aggregate(rows).items():
        exact(report[name], value, f"seed {name} aggregate")
    return rows


def derived_bound(overall):
    states = overall["baseline_state_counts"]
    missing = states["unavailable_and_truth_missing_from_all_sources"]
    missed = states["available_exact_but_missed"]
    return {
        "missing_union_count": missing,
        "existing_available_miss_count": missed,
        "selected_exact_upper_bound": overall["query_count"] - missing - missed,
        "scope": "appended pure union/intersection sets; unchanged scorer and old-first ties",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()
    path = args.summary.resolve()
    folder = path.parent
    output = folder / "independent-audit.json"
    require(not output.exists(), "refusing to overwrite independent audit")
    require(path.exists(), "complete primary summary required")
    summary = read(path, sha(path))
    require(summary.get("complete") is True, "incomplete primary cannot receive successful audit")
    old, old_audit, _, old_reports, vocabulary = upstream()
    provenance(summary, folder, old, old_audit)
    own_tests()
    bind(__file__, sha(__file__))
    validated = baseline_checks(old_reports, vocabulary, old)
    descriptors = summary["reports"]
    require([r["seed"] for r in descriptors] == list(SEEDS), "ordered three seed reports")
    reports = []
    rows = []
    for descriptor, reference in zip(descriptors, old_reports, strict=True):
        require(
            Path(descriptor["path"]).resolve() == folder / f"seed-{descriptor['seed']}.json",
            "seed output path",
        )
        report = read(descriptor["path"], descriptor["sha256"])
        rows.extend(audit_seed(report, reference, vocabulary, validated[reference["seed"]]))
        reports.append(report)
    pooled = aggregate(rows)
    for name, value in pooled.items():
        exact(summary[name], value, f"pooled {name}")
    gate = quality_gate([r["overall"] for r in reports], pooled["overall"], pooled["groups"])
    exact(summary["gate"], gate, "fixed quality gate")
    accounting = {
        "query_seed_observations": 666,
        "distinct_queries": 222,
        "candidate_slots": 42624,
        "source_paths": 5328,
        "subsets_per_operation_per_query": 255,
        "pure_operation_subsets_examined": 339660,
    }
    exact(summary["accounting"], accounting, "full accounting")
    bound = derived_bound(pooled["overall"])
    exact(summary["derived_bound"], bound, "unchanged-scorer append-only bound")
    require(
        "torch" not in sys.modules
        and not any(k == "plm" or k.startswith("plm.") for k in sys.modules),
        "non-stdlib model imported",
    )
    for bound_path, digest in INPUTS.items():
        require(sha(bound_path) == digest, f"input changed during audit: {bound_path}")
    result = {
        "audit_version": "eight-source-set-algebra-independent-v1",
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(path),
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN_SHA,
        "accounting": accounting,
        **pooled,
        "gate": gate,
        "derived_bound": bound,
        "per_seed": [
            {"seed": r["seed"], "overall": r["overall"], "groups": r["groups"]} for r in reports
        ],
        "input_sha256": dict(sorted(INPUTS.items())),
        "limitations": [
            (
                "Saved-only independent bitset and rational-sum audit; upstream accepted "
                "receipts establish neural/runtime provenance."
            ),
            (
                "No checkpoint payloads loaded, neural generation, training, or "
                "protected-test predictions."
            ),
            (
                "All 255 pure unions and all 255 pure intersections per query are oracle "
                "diagnostics; they are never predictor inputs."
            ),
            (
                "Pure-operation attainability does not evaluate arbitrary nested "
                "expressions, subtraction, complements, new sources, or changed learned "
                "weights."
            ),
            (
                "Scientific quality gate, faithful evidence acceptance and policy "
                "promotion are separate; owner decision remains required."
            ),
        ],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "audit_sha256": sha(output),
                "summary_sha256": sha(path),
                "accounting": accounting,
                "overall": pooled["overall"],
                "gate": gate,
                "derived_bound": bound,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
