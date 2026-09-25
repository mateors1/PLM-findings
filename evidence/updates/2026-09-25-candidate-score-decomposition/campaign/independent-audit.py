"""Independent saved-only audit of candidate-pool/head-score substitution.

No model, corpus or primary diagnostic imports. Canonical scores are rebuilt with
exact rational sums of saved binary floats, then rounded once to binary64.
"""

import argparse
import hashlib
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN_SHA = "cafbe8df1ae0507cee417b8ebf56f49fab18dbaec8bffca090c1dc2f1300b952"
WEAK_SHA = "3d535d1e298497609169dda8791aef3920d3d8993afab9189df359644b8c7a93"
WEAK_AUDIT_SHA = "5cf9d5b556f06f068bcfda4709d737f351260f21144bc70ec42c305feef9caab"
DECISION_SHA = "607b576806a618d9b658cc3213d4108d99f8eefaa61030e7b3588151018c7797"
RUNNER_SHA = "3665fae41075851852a7cec787f3d75d6815697da0ab3ac7a9c26f365c078d51"
RUNNER_TEST_SHA = "a7fb0b90e399efe3d0eee293a52b016e1318ca3c7890529f8829c74031e27548"
BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"
POLICY = BASE + "+first4-pair6-symmetric-set-logit-sum-v1"
ORDER = ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
ARMS = ("control", "treatment")
STATES = ("selected_exact", "exact_available_but_missed", "exact_unavailable")
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


def pool(row):
    prompt = row["prompt_ids"]
    require(
        len(prompt) == 5
        and all(type(i) is int for i in prompt)
        and prompt == [1, prompt[1], 32 if row["dimension"] == "TYPE" else 33, 34, 5],
        "protocol prompt",
    )
    require(type(prompt[1]) is int and 1024 <= prompt[1] <= 2048, "subject product ID")
    require(row["dimension"] in ("TYPE", "COLOR") and row["group"] in GROUPS, "query metadata")
    require((row["dimension"] == "COLOR") == (row["group"] == "COLOR"), "dimension/group mismatch")
    truth = row["expected_set_ids"]
    require(
        truth
        and truth == sorted(set(truth))
        and all(type(i) is int and 1024 <= i <= 2048 and i != prompt[1] for i in truth),
        "subject-excluded teacher set",
    )
    raw = row["composition"]["source_paths"]
    require(len(raw) == 4, "four sources required")
    source_sets, eligibility, mapping = [], [], [(prompt[1], row["subject"])]
    for rank, source in enumerate(raw, 1):
        ids, ok, pairs = source_set(source, prompt, rank)
        source_sets.append(set(ids))
        eligibility.append(ok)
        mapping.extend(pairs)
    slots = []
    for index, ranks in enumerate(ORDER, 1):
        members = set().union(*(source_sets[rank - 1] for rank in ranks))
        slots.append(
            {
                "slot": index,
                "kind": "original" if index <= 4 else "pair_composition",
                "source_ranks": list(ranks),
                "set_ids": sorted(members),
                "source_eligible": all(eligibility[r - 1] for r in ranks),
            }
        )
    return slots, mapping


def choose(slots, values):
    """Label-free selection over the fixed ten slots; no source regeneration."""
    vector(values)
    require(len(slots) == 10, "ten slots required")
    scored = [{**slot, "score": canonical_sum(slot["set_ids"], values)} for slot in slots]
    eligible = [slot for slot in scored if slot["source_eligible"]]
    winner = (
        max(eligible, key=lambda slot: (slot["score"], -slot["slot"])) if eligible else scored[0]
    )
    return {
        "slots": scored,
        "selected_slot": winner["slot"],
        "selected_kind": winner["kind"],
        "selected_source_ranks": winner["source_ranks"],
        "selected_set_ids": winner["set_ids"],
        "selected_source_eligible": winner["source_eligible"],
        "fallback_no_valid_source": not eligible,
    }


def set_metrics(ids, truth, successful=True):
    predicted = set(ids) if successful else set()
    truth = set(truth)
    tp = len(predicted & truth)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(truth)
    return {
        "exact": successful and predicted == truth,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "tp": tp,
        "fp": len(predicted - truth),
        "fn": len(truth - predicted),
        "set_size": len(predicted),
    }


def availability(slots, chosen, truth, removed_ranks):
    wanted = set(truth)
    originals = [s for s in slots[:4] if s["source_eligible"]]
    eligible = [s for s in slots if s["source_eligible"]]
    original_exact = any(set(s["set_ids"]) == wanted for s in originals)
    available = any(set(s["set_ids"]) == wanted for s in eligible)
    selected = chosen["selected_source_eligible"] and set(chosen["selected_set_ids"]) == wanted
    require(not selected or available, "selected exact absent from pool")
    union = set().union(*(set(s["set_ids"]) for s in originals))
    covered = wanted <= union
    return {
        "original_exact_available": original_exact,
        "all_exact_available": available,
        "selected_exact": selected,
        "recoverable_exact_miss": available and not selected,
        "source_union_contains_truth": covered,
        "source_union_ids": sorted(union),
        "eligible_source_count": len(originals),
        "source_subject_removed": [rank in removed_ranks for rank in range(1, 5)],
        "state": "selected_exact"
        if selected
        else "exact_available_but_missed"
        if available
        else "exact_unavailable",
    }


def rebuild_row(control, treatment):
    keys = ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_ids")
    exact([control[k] for k in keys], [treatment[k] for k in keys], "paired query identity")
    pools, mapping, diag = {}, [], {}
    for arm, row in zip(ARMS, (control, treatment), strict=True):
        slots, pairs = pool(row)
        pools[arm] = slots
        mapping.extend(pairs)
        diag[arm] = choose(slots, row["symmetric_relation_logits"])
        exact(
            row["composition"],
            {"source_paths": row["composition"]["source_paths"], **diag[arm], "policy": POLICY},
            "diagonal full composition",
        )
        exact(
            row["selected_metrics"],
            set_metrics(
                diag[arm]["selected_set_ids"],
                row["expected_set_ids"],
                diag[arm]["selected_source_eligible"],
            ),
            "diagonal selected metrics",
        )
    cells = {}
    for pool_arm, prefix in zip(ARMS, ("c", "t"), strict=True):
        for score_arm, suffix, score_row in zip(
            ARMS, ("c", "t"), (control, treatment), strict=True
        ):
            selection = choose(pools[pool_arm], score_row["symmetric_relation_logits"])
            cells[prefix + suffix] = {
                "pool_arm": pool_arm,
                "score_arm": score_arm,
                "slot_scores": [s["score"] for s in selection["slots"]],
                **{k: v for k, v in selection.items() if k != "slots"},
                "metrics": set_metrics(
                    selection["selected_set_ids"],
                    control["expected_set_ids"],
                    selection["selected_source_eligible"],
                ),
            }
    availability_records = {}
    for arm, row in zip(ARMS, (control, treatment), strict=True):
        removed = [
            rank
            for rank, raw in enumerate(row["composition"]["source_paths"], 1)
            if raw["terminated"]
            and raw["protocol_valid"]
            and row["prompt_ids"][1] in raw["token_ids"][5:-1]
        ]
        availability_records[arm] = availability(
            pools[arm], diag[arm], control["expected_set_ids"], removed
        )
    return {
        **{k: control[k] for k in keys},
        "pools": {
            arm: {
                "slots": slots,
                "source_subject_removed": availability_records[arm]["source_subject_removed"],
            }
            for arm, slots in pools.items()
        },
        "availability": availability_records,
        "cells": cells,
    }, mapping


def bind_old_evidence():
    directory = ROOT / "runs/learning/weak-margin-screen-v1"
    campaign = read(directory / "summary.json", WEAK_SHA)
    audit = read(directory / "independent-audit.json", WEAK_AUDIT_SHA)
    decision = read(directory / "decision.json", DECISION_SHA)
    require(
        campaign["complete"] is True and campaign["gate"]["numerical_gates_passed"] is False,
        "old screen scope",
    )
    require(
        audit["audit_passed"] is True
        and audit["all_recomputed_outputs_equal"] is True
        and audit["summary_sha256"] == WEAK_SHA,
        "old independent audit",
    )
    require(
        decision["evidence_accepted"] is True and decision["audit_sha256"] == WEAK_AUDIT_SHA,
        "separate rejection binding",
    )
    require(
        decision["summary_sha256"] == WEAK_SHA
        and decision["stage_one_accepted"] is False
        and decision["replication_authorized"] is False,
        "rejected stage remains closed",
    )
    bind(directory / "independent-audit.py", audit["script_sha256"])
    bind(directory / "record-decision.py", decision["decision_script_sha256"])
    reports = []
    for i, arm in enumerate(ARMS):
        name = f"evaluation-{arm}.json"
        report = read(directory / name, campaign["report_sha256"][name])
        require(
            report["complete"] is True
            and report["arm"] == arm
            and report["query_count"] == len(report["responses"]) == 222,
            "complete evaluation coverage",
        )
        require(
            report["checkpoint_hash"] == campaign["training"][i]["checkpoint_hash"],
            "checkpoint identity",
        )
        exact(report["training_identity"], campaign["training"][i]["identity"], "training identity")
        exact([r["index"] for r in report["responses"]], list(range(222)), "original query order")
        exact(report["product_token_ids"], list(range(1024, 2049)), "product ID columns")
        reports.append(report)
    for key in ("product_token_ids", "corpus_identity", "split_hash", "numerical_settings"):
        exact(reports[0][key], reports[1][key], "matched " + key)
    return campaign, audit, decision, reports


def cell_totals(cells):
    require(cells, "empty cell partition")
    metrics = [cell["metrics"] for cell in cells]
    return {
        "query_count": len(cells),
        "exact_set_count": sum(m["exact"] for m in metrics),
        "failure_count": sum(
            not cell["selected_source_eligible"] or cell["fallback_no_valid_source"]
            for cell in cells
        ),
        "mean_set_size": math.fsum(m["set_size"] for m in metrics) / len(cells),
        **{
            key: math.fsum(m[key] for m in metrics) / len(cells)
            for key in ("precision", "recall", "f1")
        },
    }


def decomposition(values):
    baseline = values["cc"]
    pool_effect = values["tc"] - baseline
    subsequent_score = values["tt"] - values["tc"]
    score_effect = values["ct"] - baseline
    subsequent_pool = values["tt"] - values["ct"]
    return {
        "diagonal_change": values["tt"] - baseline,
        "pool_then_score": {
            "pool_effect": pool_effect,
            "score_effect": subsequent_score,
            "sum": pool_effect + subsequent_score,
        },
        "score_then_pool": {
            "score_effect": score_effect,
            "pool_effect": subsequent_pool,
            "sum": score_effect + subsequent_pool,
        },
        "interaction": values["tt"] - values["tc"] - values["ct"] + baseline,
    }


def summarize(rows):
    totals = {
        cell: cell_totals([row["cells"][cell] for row in rows]) for cell in ("cc", "ct", "tc", "tt")
    }
    arm_availability = {}
    for arm, diagonal in zip(ARMS, ("cc", "tt"), strict=True):
        records = [row["availability"][arm] for row in rows]
        arm_availability[arm] = {
            "query_count": len(rows),
            **{
                key + "_count": sum(record[key] for record in records)
                for key in (
                    "original_exact_available",
                    "all_exact_available",
                    "selected_exact",
                    "recoverable_exact_miss",
                    "source_union_contains_truth",
                )
            },
            "state_counts": {
                state: sum(record["state"] == state for record in records) for state in STATES
            },
            "selected_f1": totals[diagonal]["f1"],
        }
    table = {old: dict.fromkeys(STATES, 0) for old in STATES}
    for row in rows:
        table[row["availability"]["control"]["state"]][
            row["availability"]["treatment"]["state"]
        ] += 1
    return {
        "query_count": len(rows),
        "cell_metrics": totals,
        "arm_availability": arm_availability,
        "transitions": table,
        "decompositions": {
            metric: decomposition({cell: values[metric] for cell, values in totals.items()})
            for metric in ("exact_set_count", "f1")
        },
    }


def checked_name_map(pairs, forward, reverse):
    for token, name in pairs:
        require(type(name) is str and name.startswith("PKM_"), "invalid product key")
        require(token not in forward or forward[token] == name, "inconsistent ID-to-name mapping")
        require(name not in reverse or reverse[name] == token, "inconsistent name-to-ID mapping")
        forward[token], reverse[name] = name, token


def test_evidence(own_sha):
    directory = ROOT / "runs/learning/candidate-score-auditor-tests-v1"
    receipt = read(directory / "test-receipt.json", sha(directory / "test-receipt.json"))
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["model_execution"] is False,
        "synthetic test receipt",
    )
    require(receipt["tested_auditor_sha256"] == own_sha, "auditor differs from tested bytes")
    bind(directory / "test_auditor.py", receipt["tests_sha256"])
    bind(directory / "stdout.txt", receipt["stdout_sha256"])
    require(
        f"{receipt['passed_count']} passed" in (directory / "stdout.txt").read_text(),
        "test stdout outcome",
    )
    return sha(directory / "test-receipt.json")


def runner_test_evidence():
    directory = ROOT / "runs/learning/candidate-score-runner-tests-v1"
    receipt = read(directory / "test-receipt.json", RUNNER_TEST_SHA)
    require(
        receipt["passed"] is True
        and receipt["new_diagnostic_executed"] is False
        and receipt["tested_runner_sha256"] == RUNNER_SHA
        and all(c["exit_code"] == 0 for c in receipt["commands"]),
        "focused primary checks",
    )
    for name, digest in receipt["tested_files_sha256"].items():
        bind(ROOT / name, digest)
    bind(directory / "stdout.txt", receipt["stdout_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    require("torch" not in sys.modules, "stdlib-only audit")
    path = args.summary.resolve()
    output = path.parent / "independent-audit.json"
    require(not output.exists(), "immutable audit already exists")
    require(path.is_file(), "complete primary summary missing")
    own_sha = sha(__file__)
    summary = read(path, sha(path))
    require(
        summary["complete"] is True
        and not summary.get("error")
        and summary["diagnostic_version"] == "candidate-score-decomposition-v1",
        "complete diagnostic required",
    )
    require(
        summary["diagnostic_accepted"] is False
        and summary["independent_audit_status"] == "required",
        "acceptance boundary",
    )
    require(
        summary["plan_sha256"] == PLAN_SHA and summary["script_sha256"] == RUNNER_SHA,
        "frozen plan/runner",
    )
    bind(ROOT / "docs/experiments/2026-09-25-candidate-score-decomposition-plan.md", PLAN_SHA)
    bind(ROOT / "scripts/diagnose_candidate_scores.py", RUNNER_SHA)
    require(summary["query_count"] == 222 and summary["seed"] == 1729, "declared coverage")
    exact(summary["state_order"], list(STATES), "three-state order")
    require(
        summary["cell_key_order"] == "pool then score; c=control,t=treatment",
        "cell-axis convention",
    )
    for name, digest in summary["input_sha256"].items():
        bind(name, digest)
    exact(
        summary["snapshot_sha256"],
        {"script.py": RUNNER_SHA, "plan.md": PLAN_SHA},
        "snapshot inventory",
    )
    for suffix, digest in summary["snapshot_sha256"].items():
        bind(path.with_suffix("." + suffix), digest)
    campaign, _, _, reports = bind_old_evidence()
    identities = {
        arm: {
            key: report[key]
            for key in (
                "checkpoint_hash",
                "training_identity",
                "corpus_identity",
                "split_hash",
                "product_token_ids",
                "numerical_settings",
            )
        }
        for arm, report in zip(ARMS, reports, strict=True)
    }
    exact(summary["input_identities"], identities, "input identity receipts")
    require(len(summary["rows"]) == 222, "complete per-query diagnostic rows")
    reconstructed, names, reverse_names, seen = [], {}, {}, set()
    for index, (control, treatment, observed) in enumerate(
        zip(reports[0]["responses"], reports[1]["responses"], summary["rows"], strict=True)
    ):
        query = (control["subject"], control["dimension"])
        require(query not in seen, "duplicate query")
        seen.add(query)
        for row in (control, treatment):
            require(
                row["index"] == index
                and row["batch_index"] == index // 8
                and row["batch_shape"] == [8 if index < 216 else 6, 5],
                "query order/batch identity",
            )
        rebuilt, pairs = rebuild_row(control, treatment)
        checked_name_map(pairs, names, reverse_names)
        exact(observed, rebuilt, f"independent row {index}")
        reconstructed.append(rebuilt)
    overall = summarize(reconstructed)
    grouped = {
        group: summarize([row for row in reconstructed if row["group"] == group])
        for group in GROUPS
    }
    exact(summary["overall"], overall, "overall 2x2/availability/transition/decomposition")
    exact(summary["groups"], grouped, "group 2x2/availability/transition/decomposition")
    for arm, diagonal, report in zip(ARMS, ("cc", "tt"), reports, strict=True):
        for group in (None, *GROUPS):
            values = overall if group is None else grouped[group]
            original_metrics = (
                report["metrics"] if group is None else report["groups"][group]["metrics"]
            )
            exact(
                values["cell_metrics"][diagonal],
                {k: original_metrics[k] for k in values["cell_metrics"][diagonal]},
                "original diagonal aggregate " + arm,
            )
    changes = []
    for row in reconstructed:
        if row["cells"]["cc"]["metrics"]["exact"] != row["cells"]["tt"]["metrics"]["exact"]:
            changes.append(
                {
                    **{k: row[k] for k in ("index", "subject", "dimension", "group")},
                    "change": "gain" if row["cells"]["tt"]["metrics"]["exact"] else "loss",
                    "control_state": row["availability"]["control"]["state"],
                    "treatment_state": row["availability"]["treatment"]["state"],
                }
            )
    exact(summary["changed_exact"], changes, "changed-exact query table")
    gained = sum(row["change"] == "gain" for row in changes)
    lost = sum(row["change"] == "loss" for row in changes)
    require(
        gained == summary["gained_exact_count"] == campaign["gate"]["gained_exact_count"] == 11
        and lost == summary["lost_exact_count"] == campaign["gate"]["lost_exact_count"] == 8,
        "changed-exact coverage",
    )
    for kind in ("gain", "loss"):
        original = campaign["gate"]["gained_exact" if kind == "gain" else "lost_exact"]
        exact(
            [
                {key: row[key] for key in ("index", "subject", "dimension", "group")}
                for row in changes
                if row["change"] == kind
            ],
            original,
            "paired prior changes",
        )
    receipt_sha = test_evidence(own_sha)
    runner_test_evidence()
    require(all(sha(name) == digest for name, digest in INPUTS.items()), "input/script drift")
    require(sha(__file__) == own_sha and "torch" not in sys.modules, "auditor drift/import")
    result = {
        "complete": True,
        "audit_passed": True,
        "all_recomputed_outputs_equal": True,
        "diagnostic_accepted": False,
        "summary_sha256": sha(path),
        "script_sha256": own_sha,
        "plan_sha256": PLAN_SHA,
        "test_receipt_sha256": receipt_sha,
        "runner_test_receipt_sha256": RUNNER_TEST_SHA,
        "distinct_queries": 222,
        "cells_checked": 888,
        "slot_scores_checked": 8880,
        "source_paths_checked": 1776,
        "overall": overall,
        "groups": grouped,
        "gained_exact_count": gained,
        "lost_exact_count": lost,
        "input_sha256": dict(sorted(INPUTS.items())),
        "limitations": [
            "Fixed saved-pool/head-vector substitutions do not regenerate guidance or outputs.",
            "Fraction sums round once to binary64 and match every stored canonical "
            "fsum score exactly.",
            "Floating decompositions preserve the declared subtraction order; "
            "no exact reassociation claim.",
            "222 aligned validation queries across four cells are not 888 independent samples.",
            "Prior accepted audit binds neural outputs and training identities; "
            "no neural/autograd proof or retraining audit is claimed.",
            "The failed weak-margin gate remains closed; no causal module attribution "
            "or deployment selection.",
        ],
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit_passed": True, "audit_sha256": sha(output), "cells_checked": 888}))


if __name__ == "__main__":
    main()
