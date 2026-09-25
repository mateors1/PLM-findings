"""Saved-only candidate availability and fixed-pool/head-vector decomposition.

This post-hoc diagnostic never generates candidates, loads a model or selects a
deployment policy. Raw generated subjects remain valid; only the existing SAME
postprocess removes them from valid complete candidate sets, recorded explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

_PLAN = "cafbe8df1ae0507cee417b8ebf56f49fab18dbaec8bffca090c1dc2f1300b952"
_SUMMARY = "3d535d1e298497609169dda8791aef3920d3d8993afab9189df359644b8c7a93"
_AUDIT = "5cf9d5b556f06f068bcfda4709d737f351260f21144bc70ec42c305feef9caab"
_DECISION = "607b576806a618d9b658cc3213d4108d99f8eefaa61030e7b3588151018c7797"
_COLUMNS = list(range(1024, 2049))
_ARMS = ("control", "treatment")
_GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
_STATES = ("selected_exact", "exact_available_but_missed", "exact_unavailable")
_ORDER = ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
_BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"
_POLICY = _BASE + "+first4-pair6-symmetric-set-logit-sum-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_bound(path, digest, inputs):
    path = Path(path).resolve()
    _require(path.is_file() and _sha(path) == digest, f"input identity mismatch: {path}")
    inputs[str(path)] = digest
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _refuse(output):
    _require(
        not output.exists() and not list(output.parent.glob(output.stem + ".*")),
        "immutable diagnostic output exists",
    )


def _fp32(values):
    _require(len(values) == 1025, "FP32 vector column count")
    for value in values:
        _require(
            type(value) in (int, float) and math.isfinite(value),
            "nonfinite or nonnumeric FP32 value",
        )
        try:
            rounded = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError("value outside FP32 range") from exc
        _require(value == rounded, "value is not exactly FP32")


def _metrics(selected, truth, success=True):
    predicted, expected = set(selected) if success else set(), set(truth)
    _require(expected, "empty teacher set")
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected)
    return {
        "exact": bool(success and predicted == expected),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": len(predicted - expected),
        "fn": len(expected - predicted),
        "set_size": len(predicted),
    }


def _name(token_id, name, names):
    _require(type(name) is str and name.startswith("PKM_"), "invalid product name")
    _require(token_id not in names or names[token_id] == name, "ID/name conflict")
    _require(name not in names.values() or names.get(token_id) == name, "name/ID conflict")
    names[token_id] = name


def _pool(row, names):
    prompt, truth, value = row["prompt_ids"], row["expected_set_ids"], row["composition"]
    _require(
        len(prompt) == 5
        and all(type(i) is int for i in prompt)
        and prompt[0] == 1
        and prompt[1] in _COLUMNS
        and prompt[2] == {"TYPE": 32, "COLOR": 33}.get(row["dimension"])
        and prompt[3:] == [34, 5],
        "saved prompt grammar",
    )
    _require(
        truth
        and all(type(i) is int and i in _COLUMNS and i != prompt[1] for i in truth)
        and truth == sorted(set(truth)),
        "canonical subject-excluded teacher set",
    )
    _require(
        row["group"] in _GROUPS and (row["dimension"] == "COLOR") == (row["group"] == "COLOR"),
        "analysis group",
    )
    _name(prompt[1], row["subject"], names)
    _require(
        value["policy"] == _POLICY
        and len(value["source_paths"]) == 4
        and len(value["slots"]) == 10,
        "policy/source/slot count",
    )
    sets, eligible, removed = [], [], []
    for rank, path in enumerate(value["source_paths"], 1):
        ids = path["token_ids"]
        _require(
            6 <= len(ids) <= 512 and ids[:5] == prompt and all(type(i) is int for i in ids),
            "raw path prompt/budget",
        )
        terminated = ids[-1] == 2
        members = ids[5:-1] if terminated else ids[5:]
        _require(
            all(i in _COLUMNS for i in members) and (terminated or len(ids) == 512),
            "raw path product grammar",
        )
        valid = terminated and len(ids) >= 7
        error = (
            None
            if valid
            else (
                "generated response is shorter than the protocol grammar"
                if len(ids) < 7
                else "response must terminate with EOS"
            )
        )
        _require(
            type(path["terminated"]) is bool
            and path["terminated"] == terminated
            and type(path["protocol_valid"]) is bool
            and path["protocol_valid"] == valid
            and path["error"] == error
            and len(path["targets"]) == len(members)
            and path["decoding"] == _BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            "raw path fields",
        )
        for entity, token in zip(members, path["targets"], strict=True):
            _name(entity, token, names)
        eligible.append(valid and terminated and len(members) == len(set(members)))
        remove_subject = valid and terminated and prompt[1] in members
        removed.append(remove_subject)
        sets.append(set(members) - {prompt[1]} if valid and terminated else set(members))
    slots = []
    for index, ranks in enumerate(_ORDER, 1):
        slot = {
            "slot": index,
            "kind": "original" if index <= 4 else "pair_composition",
            "source_ranks": list(ranks),
            "set_ids": sorted(set().union(*(sets[r - 1] for r in ranks))),
            "source_eligible": all(eligible[r - 1] for r in ranks),
        }
        saved = value["slots"][index - 1]
        _require(
            type(saved["slot"]) is int
            and all(type(i) is int for i in saved["source_ranks"])
            and all(type(i) is int for i in saved["set_ids"])
            and type(saved["score"]) in (int, float)
            and math.isfinite(saved["score"])
            and type(saved["source_eligible"]) is bool
            and {k: v for k, v in saved.items() if k != "score"} == slot,
            "canonical source/union slot mismatch",
        )
        slots.append(slot)
    return {"slots": slots, "source_subject_removed": removed}


def _cell(pool, vector, truth, pool_arm, score_arm):
    # This selector does not inspect teacher labels; metrics are computed only afterward.
    scores = [math.fsum(vector[i - 1024] for i in slot["set_ids"]) for slot in pool["slots"]]
    _require(all(math.isfinite(score) for score in scores), "nonfinite set score")
    candidates = [i for i, slot in enumerate(pool["slots"]) if slot["source_eligible"]]
    index = min(candidates, key=lambda i: (-scores[i], i)) if candidates else 0
    chosen = pool["slots"][index]
    return {
        "pool_arm": pool_arm,
        "score_arm": score_arm,
        "slot_scores": scores,
        "selected_slot": chosen["slot"],
        "selected_kind": chosen["kind"],
        "selected_source_ranks": chosen["source_ranks"],
        "selected_set_ids": chosen["set_ids"],
        "selected_source_eligible": chosen["source_eligible"],
        "fallback_no_valid_source": not candidates,
        "metrics": _metrics(
            chosen["set_ids"], truth, chosen["source_eligible"] and bool(candidates)
        ),
    }


def _diagonal(row, pool, arm):
    result = _cell(pool, row["symmetric_relation_logits"], row["expected_set_ids"], arm, arm)
    saved = row["composition"]
    _require(
        result["slot_scores"] == [s["score"] for s in saved["slots"]], "diagonal score mismatch"
    )
    _require(
        all(
            result[k] == saved[k]
            for k in (
                "selected_slot",
                "selected_kind",
                "selected_source_ranks",
                "selected_set_ids",
                "selected_source_eligible",
                "fallback_no_valid_source",
            )
        )
        and result["metrics"] == row["selected_metrics"],
        "diagonal choice/metrics mismatch",
    )
    return result


def _align(control, treatment):
    keys = (
        "index",
        "subject",
        "dimension",
        "group",
        "prompt_ids",
        "expected_set_ids",
        "batch_index",
        "batch_shape",
    )
    _require(all(control[k] == treatment[k] for k in keys), "paired query alignment mismatch")


def _totals(cells):
    _require(cells, "empty cell partition")
    metrics = [cell["metrics"] for cell in cells]
    return {
        "query_count": len(cells),
        "exact_set_count": sum(m["exact"] for m in metrics),
        "failure_count": sum(
            not c["selected_source_eligible"] or c["fallback_no_valid_source"] for c in cells
        ),
        "mean_set_size": math.fsum(m["set_size"] for m in metrics) / len(cells),
        **{k: math.fsum(m[k] for m in metrics) / len(cells) for k in ("precision", "recall", "f1")},
    }


def _preflight(root, plan, output):
    _refuse(output)
    inputs = {}
    _require(_sha(plan) == _PLAN, "declared plan identity mismatch")
    inputs[str(plan.resolve())] = _PLAN
    directory = root / "runs/learning/weak-margin-screen-v1"
    summary = _read_bound(directory / "summary.json", _SUMMARY, inputs)
    audit = _read_bound(directory / "independent-audit.json", _AUDIT, inputs)
    decision = _read_bound(directory / "decision.json", _DECISION, inputs)
    _require(
        summary["complete"] is True
        and summary["seed"] == 1729
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["all_recomputed_outputs_equal"] is True
        and audit["summary_sha256"] == _SUMMARY
        and decision["summary_sha256"] == _SUMMARY
        and decision["audit_sha256"] == _AUDIT
        and decision["evidence_accepted"] is True
        and decision["stage_one_accepted"] is False
        and decision["replication_authorized"] is False
        and decision["gate"] == summary["gate"] == audit["gate"],
        "prior summary/audit/rejection chain",
    )
    auditor = directory / "independent-audit.py"
    _require(_sha(auditor) == audit["script_sha256"], "prior auditor identity")
    inputs[str(auditor.resolve())] = audit["script_sha256"]
    reports = [
        _read_bound(
            directory / f"evaluation-{arm}.json",
            summary["report_sha256"][f"evaluation-{arm}.json"],
            inputs,
        )
        for arm in _ARMS
    ]
    for key in ("corpus_identity", "split_hash", "product_token_ids", "numerical_settings"):
        _require(reports[0][key] == reports[1][key], f"cross-arm {key}")
    _require(
        reports[0]["config"]["eval"] == reports[1]["config"]["eval"],
        "cross-arm inference configuration",
    )
    _require(reports[0]["product_token_ids"] == _COLUMNS, "product column mapping")
    names, pools, diagonal = {}, {}, {}
    for arm, report in zip(_ARMS, reports, strict=True):
        _require(
            report["arm"] == arm
            and report["complete"] is True
            and report["query_count"] == len(report["responses"]) == 222,
            "saved report coverage",
        )
        _require(
            report["checkpoint_hash"] == summary["training"][_ARMS.index(arm)]["checkpoint_hash"]
            and report["training_identity"] == summary["training"][_ARMS.index(arm)]["identity"],
            "checkpoint identity",
        )
        pools[arm], diagonal[arm] = [], []
        seen = set()
        for index, row in enumerate(report["responses"]):
            _require(
                row["index"] == index
                and (row["subject"], row["dimension"]) not in seen
                and row["batch_index"] == index // 8
                and row["batch_shape"] == [8 if index < 216 else 6, 5],
                "row order/batch shape",
            )
            seen.add((row["subject"], row["dimension"]))
            _fp32(row["symmetric_relation_logits"])
            pool = _pool(row, names)
            pools[arm].append(pool)
            diagonal[arm].append(_diagonal(row, pool, arm))
        for group in (None, *_GROUPS):
            cells = [
                c
                for c, r in zip(diagonal[arm], report["responses"], strict=True)
                if group is None or r["group"] == group
            ]
            saved = report["metrics"] if group is None else report["groups"][group]["metrics"]
            _require(
                all(saved[k] == v for k, v in _totals(cells).items()),
                "reported diagonal metrics mismatch",
            )
    for a, b in zip(reports[0]["responses"], reports[1]["responses"], strict=True):
        _align(a, b)
    _require("torch" not in sys.modules, "saved-only preflight imported Torch")
    return {
        "inputs": inputs,
        "reports": dict(zip(_ARMS, reports, strict=True)),
        "pools": pools,
        "diagonal": diagonal,
        "prior_summary": summary,
        "id_to_name": names,
    }


def _availability(pool, cell, truth):
    exact_slots = [
        s["slot"] for s in pool["slots"] if s["source_eligible"] and s["set_ids"] == truth
    ]
    selected = cell["metrics"]["exact"]
    union = sorted(
        set().union(*(set(s["set_ids"]) for s in pool["slots"][:4] if s["source_eligible"]))
    )
    _require(not selected or bool(exact_slots), "exact selection without exact availability")
    return {
        "original_exact_available": any(i <= 4 for i in exact_slots),
        "all_exact_available": bool(exact_slots),
        "selected_exact": selected,
        "recoverable_exact_miss": bool(exact_slots) and not selected,
        "state": "selected_exact"
        if selected
        else "exact_available_but_missed"
        if exact_slots
        else "exact_unavailable",
        "source_union_contains_truth": set(truth) <= set(union),
        "source_union_ids": union,
        "eligible_source_count": sum(s["source_eligible"] for s in pool["slots"][:4]),
        "source_subject_removed": pool["source_subject_removed"],
    }


def _decomposition(values):
    cc, ct, tc, tt = (values[key] for key in ("cc", "ct", "tc", "tt"))
    return {
        "diagonal_change": tt - cc,
        "pool_then_score": {
            "pool_effect": tc - cc,
            "score_effect": tt - tc,
            "sum": (tc - cc) + (tt - tc),
        },
        "score_then_pool": {
            "score_effect": ct - cc,
            "pool_effect": tt - ct,
            "sum": (ct - cc) + (tt - ct),
        },
        "interaction": tt - tc - ct + cc,
    }


def _summarize(rows):
    totals = {cell: _totals([r["cells"][cell] for r in rows]) for cell in ("cc", "ct", "tc", "tt")}
    availability = {}
    for arm in _ARMS:
        values = [r["availability"][arm] for r in rows]
        availability[arm] = {
            "query_count": len(rows),
            **{
                key + "_count": sum(v[key] for v in values)
                for key in (
                    "original_exact_available",
                    "all_exact_available",
                    "selected_exact",
                    "recoverable_exact_miss",
                    "source_union_contains_truth",
                )
            },
            "state_counts": {state: sum(v["state"] == state for v in values) for state in _STATES},
            "selected_f1": totals["cc" if arm == "control" else "tt"]["f1"],
        }
    transitions = {
        a: {
            b: sum(
                r["availability"]["control"]["state"] == a
                and r["availability"]["treatment"]["state"] == b
                for r in rows
            )
            for b in _STATES
        }
        for a in _STATES
    }
    return {
        "query_count": len(rows),
        "cell_metrics": totals,
        "arm_availability": availability,
        "transitions": transitions,
        "decompositions": {
            metric: _decomposition({key: value[metric] for key, value in totals.items()})
            for metric in ("exact_set_count", "f1")
        },
    }


def _diagnose(context):
    rows = []
    for index, record in enumerate(context["reports"]["control"]["responses"]):
        truth = record["expected_set_ids"]
        cells = {}
        for pool_arm in _ARMS:
            for score_arm in _ARMS:
                key = pool_arm[0] + score_arm[0]
                cells[key] = _cell(
                    context["pools"][pool_arm][index],
                    context["reports"][score_arm]["responses"][index]["symmetric_relation_logits"],
                    truth,
                    pool_arm,
                    score_arm,
                )
        _require(
            cells["cc"] == context["diagonal"]["control"][index]
            and cells["tt"] == context["diagonal"]["treatment"][index],
            "diagonal changed",
        )
        rows.append(
            {
                **{
                    key: record[key]
                    for key in (
                        "index",
                        "subject",
                        "dimension",
                        "group",
                        "prompt_ids",
                        "expected_set_ids",
                    )
                },
                "pools": {arm: context["pools"][arm][index] for arm in _ARMS},
                "cells": cells,
                "availability": {
                    arm: _availability(context["pools"][arm][index], cells[arm[0] * 2], truth)
                    for arm in _ARMS
                },
            }
        )
    changed = [
        {
            **{k: r[k] for k in ("index", "subject", "dimension", "group")},
            "change": "gain" if r["cells"]["tt"]["metrics"]["exact"] else "loss",
            "control_state": r["availability"]["control"]["state"],
            "treatment_state": r["availability"]["treatment"]["state"],
        }
        for r in rows
        if r["cells"]["cc"]["metrics"]["exact"] != r["cells"]["tt"]["metrics"]["exact"]
    ]
    _require(
        len(changed) == 19 and sum(r["change"] == "gain" for r in changed) == 11,
        "declared changed-exact coverage",
    )
    return {
        "rows": rows,
        "overall": _summarize(rows),
        "groups": {g: _summarize([r for r in rows if r["group"] == g]) for g in _GROUPS},
        "changed_exact": changed,
        "gained_exact_count": 11,
        "lost_exact_count": 8,
    }


def main():
    script = Path(__file__).resolve()
    executing = script.read_bytes()
    root = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "runs/learning/candidate-score-decomposition-v1/summary.json",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=root / "docs/experiments/2026-09-25-candidate-score-decomposition-plan.md",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    _require("torch" not in sys.modules, "standalone stdlib-only launch required")
    output = args.out.resolve()
    context = _preflight(root, args.plan, output)
    context["inputs"][str(script)] = hashlib.sha256(executing).hexdigest()
    _require(
        all(_sha(Path(p)) == digest for p, digest in context["inputs"].items()),
        "input/script drift",
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "torch_imported": False,
                    "new_diagnostic_executed": False,
                    "diagonal_reconstruction_exact": True,
                    "query_count": 222,
                }
            )
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, content in (("script.py", executing), ("plan.md", args.plan.read_bytes())):
        with output.with_suffix("." + name).open("xb") as stream:
            stream.write(content)
    result = {
        "diagnostic_version": "candidate-score-decomposition-v1",
        "complete": False,
        "diagnostic_accepted": False,
        "independent_audit_status": "required",
        "plan_sha256": _PLAN,
        "script_sha256": hashlib.sha256(executing).hexdigest(),
        "input_sha256": context["inputs"],
        "snapshot_sha256": {k: _sha(output.with_suffix("." + k)) for k in ("script.py", "plan.md")},
        "query_count": 222,
        "seed": 1729,
        "cell_key_order": "pool then score; c=control,t=treatment",
        "state_order": list(_STATES),
        "input_identities": {
            a: {
                k: r[k]
                for k in (
                    "checkpoint_hash",
                    "training_identity",
                    "corpus_identity",
                    "split_hash",
                    "product_token_ids",
                    "numerical_settings",
                )
            }
            for a, r in context["reports"].items()
        },
        "limitations": [
            (
                "Post-hoc saved-output factorial analysis, "
                "not causal module attribution or a deployed policy."
            ),
            "222 distinct validation queries; four cells do not create 888 independent samples.",
            (
                "The head also participates in guidance; "
                "a real head swap may regenerate different pools."
            ),
            (
                "No neural replay, training, protected-test measurement "
                "or reopening of the rejected weak-margin gate."
            ),
        ],
    }
    try:
        result.update(_diagnose(context))
        _require(
            all(_sha(Path(p)) == digest for p, digest in context["inputs"].items()),
            "input/script drift",
        )
        _require(
            all(
                _sha(output.with_suffix("." + k)) == v for k, v in result["snapshot_sha256"].items()
            ),
            "snapshot drift",
        )
        _require("torch" not in sys.modules, "diagnostic imported Torch")
        result["complete"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        _write(output, result)
    print(
        json.dumps({"complete": True, "diagnostic_accepted": False, "summary_sha256": _sha(output)})
    )


if __name__ == "__main__":
    main()
