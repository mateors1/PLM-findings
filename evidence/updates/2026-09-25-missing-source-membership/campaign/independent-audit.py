"""Independent stdlib saved-membership diagnosis audit; no neural or policy work."""

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN_SHA = "bda59d3987d98bb746f3b0dea7445269ec5044f82abd5f0fce3df9404b855907"
UPSTREAM_SHA = "1fe6e9cb1980753e6c2315c6dac890148a52105061c881261da9bb73db1c387d"
UPSTREAM_AUDIT_SHA = "27e0c5ff2931d88fed9fae612c73c64c7835e01282097fc290fa78cd9a7e330c"
DECISION_SHA = "01a51a5a9d5957ec68f32b0d81d23acace29d7eb3aa5684d99939582e88cb32a"
BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"
SEEDS = (1729, 1730, 1731)
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


def membership(row, index, names):
    prompt = row["prompt_ids"]
    require(row["index"] == index and type(row["index"]) is int, "row index")
    require(
        row["dimension"] in ("TYPE", "COLOR") and row["group"] in GROUPS, "query dimension/group"
    )
    require((row["dimension"] == "COLOR") == (row["group"] == "COLOR"), "dimension/group mismatch")
    require(len(prompt) == 5 and all(type(i) is int for i in prompt), "five integer prompt IDs")
    require(
        prompt == [1, prompt[1], 32 if row["dimension"] == "TYPE" else 33, 34, 5]
        and 1024 <= prompt[1] <= 2048,
        "protocol prompt",
    )
    truth = row["expected_set_ids"]
    require(
        isinstance(truth, list)
        and truth
        and truth == sorted(set(truth))
        and all(type(i) is int and 1024 <= i <= 2048 and i != prompt[1] for i in truth),
        "canonical subject-excluded teacher IDs",
    )
    paths = row["source_paths"]
    require(len(paths) == 8, "eight raw sources")
    sets = []
    removed = []
    mapping = [(prompt[1], row["subject"])]
    first = []
    for rank, path in enumerate(paths, 1):
        members, eligible, pairs = source_set(path, prompt, rank)
        require(eligible and path["error"] is None, "invalid/incomplete/repeated raw source")
        sets.append(members)
        mapping.extend(pairs)
        removed.append(prompt[1] in path["token_ids"][5:-1])
        first.append(path["token_ids"][5])
    require(len(set(first)) == 8, "first-rank token uniqueness")
    for token, name in mapping:
        require(isinstance(name, str) and name.startswith("PKM_"), "product name")
        require(token not in names or names[token] == name, "token/name mismatch")
        names[token] = name
    require(len(set(names.values())) == len(names), "nonbijective token names")
    exact(sets, row["source_set_ids"], "saved source membership")
    exact(removed, row["source_subject_removed"], "SAME subject removal")
    union = set().union(*(set(s) for s in sets))
    missing = sorted(set(truth) - union)
    covered = sorted(set(truth) & union)
    require(row["diagnosis"]["source_union_contains_truth"] is (not missing), "saved coverage flag")
    require(
        (row["diagnosis"]["baseline_state"] == "unavailable_and_truth_missing_from_all_sources")
        == bool(missing),
        "saved missing state",
    )
    vector(row["symmetric_relation_logits"])
    return sorted(union), missing, covered


def member(token, truth, subject, head):
    """Rank by independent greater-score/equal-score-smaller-ID counting."""
    require(token in truth and token != subject, "scored member must be non-subject truth")
    score = head[token - 1024]
    universe = [i for i in range(1024, 2049) if i != subject]
    rank = 1 + sum(
        head[i - 1024] > score or (head[i - 1024] == score and i < token) for i in universe
    )
    false = [i for i in universe if i not in truth]
    return {
        "token_id": token,
        "logit": score,
        "rank": rank,
        "false_strictly_higher": sum(head[i - 1024] > score for i in false),
        "false_tied": sum(head[i - 1024] == score for i in false),
        "oracle_top_k": rank <= len(truth),
    }


def extent(values):
    if not values:
        return None
    ordered = sorted(values)
    half = len(ordered) // 2
    median = ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def describe(members):
    return {
        "count": len(members),
        "positive_count": sum(m["logit"] > 0 for m in members),
        "zero_count": sum(m["logit"] == 0 for m in members),
        "negative_count": sum(m["logit"] < 0 for m in members),
        "logit": extent([m["logit"] for m in members]),
        "rank": extent([m["rank"] for m in members]),
    }


def reconstruct(row, union, missing, covered):
    require(bool(missing), "focus must be selected from missing membership first")
    head = row["symmetric_relation_logits"]
    truth = set(row["expected_set_ids"])
    subject = row["prompt_ids"][1]
    omitted = [member(i, truth, subject, head) for i in missing]
    present = [member(i, truth, subject, head) for i in covered]
    omitted_summary = describe(omitted)
    positives = omitted_summary["positive_count"]
    return {
        **{
            k: row[k]
            for k in ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_ids")
        },
        "source_union_ids": union,
        "oracle_k": len(truth),
        "omitted_members": omitted,
        "covered_members": present,
        "omitted_summary": omitted_summary,
        "covered_summary": describe(present),
        "positive_omitted_category": "all"
        if positives == len(missing)
        else "some"
        if positives
        else "none",
    }


def coverage_row(row, missing, covered):
    return {
        **{k: row[k] for k in ("index", "subject", "dimension", "group")},
        "true_member_count": len(row["expected_set_ids"]),
        "covered_member_count": len(covered),
        "missing_member_count": len(missing),
        "missing_ids": missing,
    }


def aggregate(focused):
    result = {
        "focused_queries": len(focused),
        "distinct_focused_queries": len({(r["subject"], r["dimension"]) for r in focused}),
        "omitted_oracle_top_k_count": sum(
            m["oracle_top_k"] for r in focused for m in r["omitted_members"]
        ),
        "positive_omitted_queries": {
            category: sum(r["positive_omitted_category"] == category for r in focused)
            for category in ("all", "some", "none")
        },
    }
    for group in ("omitted", "covered"):
        summaries = [r[group + "_summary"] for r in focused]
        count = sum(s["count"] for s in summaries)
        positive = sum(s["positive_count"] for s in summaries)
        fractions = [s["positive_count"] / s["count"] for s in summaries if s["count"]]
        result[group] = {
            "member_occurrences": count,
            "positive_count": positive,
            "zero_count": sum(s["zero_count"] for s in summaries),
            "negative_count": sum(s["negative_count"] for s in summaries),
            "positive_member_fraction": positive / count if count else None,
            "nonempty_query_count": len(fractions),
            "mean_query_positive_fraction": math.fsum(fractions) / len(fractions)
            if fractions
            else None,
        }
    return result


def coverage_totals(rows):
    return {
        "query_count": len(rows),
        "focused_queries": sum(r["missing_member_count"] > 0 for r in rows),
        "fully_covered_queries": sum(r["missing_member_count"] == 0 for r in rows),
        "true_member_occurrences": sum(r["true_member_count"] for r in rows),
        "covered_member_occurrences": sum(r["covered_member_count"] for r in rows),
        "omitted_member_occurrences": sum(r["missing_member_count"] for r in rows),
    }


def aligned(rows, previous, count=222):
    keys = [
        (r["subject"], r["dimension"], r["group"], r["prompt_ids"], r["expected_set_ids"])
        for r in rows
    ]
    require(
        len(keys) == count and len({(k[0], k[1]) for k in keys}) == count,
        "coverage/duplicate query",
    )
    if previous is not None:
        exact(keys, previous, "cross-seed query/teacher mismatch")
    return keys


def prepare(reports):
    prepared = {}
    names = {}
    previous = None
    for report, expected_count in zip(reports, (14, 6, 10), strict=True):
        seed = report["seed"]
        rows = report["responses"]
        previous = aligned(rows, previous)
        memberships = [membership(row, index, names) for index, row in enumerate(rows)]
        focused = [r for r, m in zip(rows, memberships, strict=True) if m[1]]
        require(
            len(focused) == expected_count and all(r["group"] == "TYPE_dual" for r in focused),
            "fixed missing subset coverage",
        )
        prepared[seed] = memberships
    return prepared


def upstream():
    directory = ROOT / "runs/learning/eight-source-set-algebra-v1"
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
        decision["evidence_accepted"] is True and decision["fixed_quality_gate_passed"] is False,
        "evidence acceptance versus failed quality",
    )
    require(
        decision["summary_sha256"] == UPSTREAM_SHA
        and decision["audit_sha256"] == UPSTREAM_AUDIT_SHA
        and decision["auditor_sha256"] == audit["script_sha256"],
        "owner binding",
    )
    exact(summary["gate"], audit["gate"], "upstream gate audit")
    exact(decision["gate"], audit["gate"], "upstream gate decision")
    require(
        summary["gate"]["quality_passed"] is False and decision["policy_promoted"] is False,
        "failed upstream not promoted",
    )
    bind(directory / "independent-audit.py", audit["script_sha256"])
    require([r["seed"] for r in summary["reports"]] == list(SEEDS), "ordered upstream seeds")
    reports = []
    for descriptor in summary["reports"]:
        report = read(descriptor["path"], descriptor["sha256"])
        require(
            report["complete"] is True
            and report["seed"] == descriptor["seed"]
            and report["query_count"] == len(report["responses"]) == 222,
            "upstream report coverage",
        )
        exact(report["product_token_ids"], list(range(1024, 2049)), "upstream product columns")
        reports.append(report)
    return summary, audit, reports


RUNNER_SHA = "f88cfff322c6f7c79174c781715238c4405f72f09af703e400398a92c63dd829"
RUNNER_RECEIPT_SHA = "612f268e8be6e50a5b422648148ba3ad40e7c03b07922007838de9246b012e7a"


def check_manifest(manifest):
    require(isinstance(manifest, dict) and bool(manifest), "empty hash manifest")
    for path, digest in manifest.items():
        bind(path, digest)


def provenance(summary, folder, old, old_audit):
    require(summary["campaign_version"] == "missing-source-membership-v1", "campaign version")
    require(
        summary["complete"] is True and summary["final_identity_check"] is True,
        "primary incomplete",
    )
    require("error" not in summary and "identity_error" not in summary, "primary execution failure")
    for key in (
        "acceptance",
        "upstream_quality_gate_passed",
        "neural_generation_executed",
        "training_executed",
        "protected_test_used",
        "new_prediction_policy",
    ):
        require(summary[key] is False, f"scope violation {key}")
    require(summary["upstream_evidence_accepted"] is True, "upstream evidence acceptance")
    require(
        summary["plan_sha256"] == PLAN_SHA and summary["script_sha256"] == RUNNER_SHA,
        "frozen source/plan",
    )
    plan = ROOT / "docs/experiments/2026-09-25-missing-source-membership-plan.md"
    script = ROOT / "scripts/diagnose_missing_source_membership.py"
    bind(plan, PLAN_SHA)
    bind(script, RUNNER_SHA)
    directory = ROOT / "runs/learning/eight-source-set-algebra-v1"
    required = {
        str(plan.resolve()): PLAN_SHA,
        str(script.resolve()): RUNNER_SHA,
        **{
            str((directory / name).resolve()): digest
            for name, digest in (
                ("summary.json", UPSTREAM_SHA),
                ("independent-audit.json", UPSTREAM_AUDIT_SHA),
                ("decision.json", DECISION_SHA),
                ("independent-audit.py", old_audit["script_sha256"]),
            )
        },
    }
    required.update({str(Path(r["path"]).resolve()): r["sha256"] for r in old["reports"]})
    normalized = {str(Path(p).resolve()): h for p, h in summary["input_sha256"].items()}
    check_manifest(summary["input_sha256"])
    require(
        all(normalized.get(p) == h for p, h in required.items()), "required pinned input missing"
    )
    snapshots = summary["snapshot_sha256"]
    require(
        {str(Path(p).resolve()) for p in snapshots}
        == {
            str((folder / f"summary.{suffix}").resolve())
            for suffix in (
                "script.py",
                "plan.md",
                "test-receipt.json",
                "test-stdout.txt",
                "inputs.json",
            )
        },
        "snapshot inventory",
    )
    check_manifest(snapshots)
    bind(folder / "summary.script.py", RUNNER_SHA)
    bind(folder / "summary.plan.md", PLAN_SHA)
    receipt = read(folder / "summary.test-receipt.json", RUNNER_RECEIPT_SHA)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["tested_script_sha256"] == RUNNER_SHA
        and receipt["diagnostic_measured"] is False
        and receipt["test_count"] == 26,
        "runner tests",
    )
    require(RUNNER_RECEIPT_SHA in normalized.values(), "runner receipt not input bound")
    bind(ROOT / "tests/unit/test_missing_source_membership.py", receipt["tested_test_sha256"])
    bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"])
    bind(folder / "summary.test-stdout.txt", receipt["stdout"]["sha256"])
    exact(
        json.loads((folder / "summary.inputs.json").read_text(encoding="utf-8")),
        summary["input_sha256"],
        "snapshotted input manifest",
    )


def own_tests():
    directory = ROOT / "runs/learning/missing-source-membership-auditor-tests-v1"
    receipt = read(directory / "test-receipt.json", sha(directory / "test-receipt.json"))
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["model_execution"] is False
        and receipt["new_diagnostic_executed"] is False,
        "auditor test receipt",
    )
    require(
        receipt["tested_auditor_sha256"] == sha(__file__) and receipt["passed_count"] >= 37,
        "untested auditor",
    )
    bind(directory / "test_auditor.py", receipt["tests_sha256"])
    bind(directory / "stdout.txt", receipt["stdout_sha256"])


def audit_seed(report, reference, memberships):
    require(
        report["complete"] is True and report["seed"] == reference["seed"], "seed complete/identity"
    )
    require("error" not in report and "failed_query" not in report, "failed seed")
    coverage = []
    focused = []
    for row, (union, missing, covered) in zip(reference["responses"], memberships, strict=True):
        coverage.append(coverage_row(row, missing, covered))
        if missing:
            focused.append(reconstruct(row, union, missing, covered))
    exact(report["coverage"], coverage, "all-row coverage reconstruction")
    exact(report["focused_rows"], focused, "focused membership/rank/error statistics")
    exact(report["overall"], aggregate(focused), "seed member/query aggregate")
    exact(report["coverage_summary"], coverage_totals(coverage), "seed coverage totals")
    return coverage, focused


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
    old, old_audit, references = upstream()
    provenance(summary, folder, old, old_audit)
    own_tests()
    bind(__file__, sha(__file__))
    memberships = prepare(references)
    require([r["seed"] for r in summary["reports"]] == list(SEEDS), "ordered three output reports")
    reports = []
    all_coverage = []
    all_focused = []
    for descriptor, reference in zip(summary["reports"], references, strict=True):
        require(
            Path(descriptor["path"]).resolve() == folder / f"seed-{descriptor['seed']}.json",
            "seed output path",
        )
        report = read(descriptor["path"], descriptor["sha256"])
        coverage, focused = audit_seed(report, reference, memberships[reference["seed"]])
        reports.append(report)
        all_coverage.extend(coverage)
        all_focused.extend(focused)
    overall = aggregate(all_focused)
    coverage = coverage_totals(all_coverage)
    exact(summary["overall"], overall, "pooled member/query fractions")
    exact(summary["coverage_summary"], coverage, "pooled coverage counts")
    require(
        overall["focused_queries"] == 30 and coverage["query_count"] == 666,
        "declared focus/coverage",
    )
    require(
        "torch" not in sys.modules
        and not any(k == "plm" or k.startswith("plm.") for k in sys.modules),
        "runtime imported",
    )
    for bound_path, digest in INPUTS.items():
        require(sha(bound_path) == digest, f"input drift: {bound_path}")
    result = {
        "audit_version": "missing-source-membership-independent-v1",
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(path),
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN_SHA,
        "overall": overall,
        "coverage_summary": coverage,
        "per_seed": [
            {"seed": r["seed"], "overall": r["overall"], "coverage_summary": r["coverage_summary"]}
            for r in reports
        ],
        "input_sha256": dict(sorted(INPUTS.items())),
        "limitations": [
            "Independent saved-output arithmetic, not neural replay or a new checkpoint audit.",
            "Relation-head rank is not decoder-plus-guidance rank; oracle K uses teacher labels.",
            "Positive scores are not calibrated membership probabilities or causal proof.",
            "Repeated products across queries and seeds are not independent statistical samples.",
            "Diagnostic only: no prediction policy, generated-quality claim or protected-test use.",
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
                "overall": overall,
                "coverage_summary": coverage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
