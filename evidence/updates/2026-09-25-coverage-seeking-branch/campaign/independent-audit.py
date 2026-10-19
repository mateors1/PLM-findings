"""Independent saved-output coverage-seeking audit; no model or evaluator imports.

Exact rational accumulation cross-checks saved FP32 set logits. Old paths are
reused; new greedy transitions remain source/receipt evidence, not a second
independent neural execution. A passing audit does not itself accept the policy.
"""

import argparse
import hashlib
import itertools
import json
import math
import struct
import sys
import zipfile
from fractions import Fraction
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN_SHA = "4d718ca6b19d77d8387ef7b90089e2692058bfc99577ac9affcafe765e5a1551"
REFERENCE_SHA = "1422f7ae69f012683e01e7a2e3299019e61c163aa3b675606843cc8ef9bb2183"
REFERENCE_AUDIT_SHA = "eb7413d30e92519616c74e0541399a90d50d6d1556529ea06c30954d26fb1e95"
ACCEPTANCE_SHA = "70196a4bcbf1cb8175eccda3956c652fa7d6357c8638dd71196777bf09457601"
SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
SPLIT_SHA = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
SEEDS = (1729, 1730, 1731)
CHECKPOINTS = {
    1729: "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1",
    1730: "5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2",
    1731: "ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d",
}
BASE = "greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1+first-relation-logsigmoid-alpha16-v1"
POLICY4 = BASE + "+first4-pair6-symmetric-set-logit-sum-v1"
POLICY8 = BASE + "+first8-pair28-symmetric-set-logit-sum-v1"
ORDER4 = ((1,), (2,), (3,), (4,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
ORDER8 = (
    ORDER4
    + tuple((i,) for i in range(5, 9))
    + tuple(pair for pair in itertools.combinations(range(1, 9), 2) if pair not in ORDER4)
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


def source_set(raw, prompt, descriptor):
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
        raw["decoding"] == descriptor,
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


def anchor(union, subject, values):
    """No labels, query groups, teacher cardinalities or error masks enter."""
    vector(values)
    require(type(subject) is int and 1024 <= subject <= 2048, "subject ID")
    require(
        union == sorted(set(union)) and all(type(i) is int and 1024 <= i <= 2048 for i in union),
        "canonical old union",
    )
    outside = [
        i for i in range(1024, 2049) if i != subject and i not in union and values[i - 1024] > 0
    ]
    chosen = max(outside, key=lambda i: (values[i - 1024], -i)) if outside else None
    return {
        "active": chosen is not None,
        "token_id": chosen,
        "logit": values[chosen - 1024] if chosen is not None else None,
        "uncovered_positive_count": len(outside),
    }


def pool(source_sets, eligible, values):
    """Keep the 36 old slots first, then one new original and eight unions."""
    vector(values)
    count = len(source_sets)
    require(
        count in (8, 9) and len(eligible) == count and all(type(ok) is bool for ok in eligible),
        "pool width/eligibility",
    )
    order = ORDER8 + (((9,), *((i, 9) for i in range(1, 9))) if count == 9 else ())
    result = []
    for index, ranks in enumerate(order, 1):
        members = sorted(set().union(*(set(source_sets[r - 1]) for r in ranks)))
        result.append(
            {
                "slot": index,
                "kind": "original" if len(ranks) == 1 else "pair_composition",
                "source_ranks": list(ranks),
                "set_ids": members,
                "source_eligible": all(eligible[r - 1] for r in ranks),
                "score": canonical_sum(members, values),
            }
        )
    return result


def select(slots):
    valid = [s for s in slots if s["source_eligible"]]
    winner = max(valid, key=lambda s: (s["score"], -s["slot"])) if valid else slots[0]
    return {
        f"selected_{key}": winner[key]
        for key in ("slot", "kind", "source_ranks", "set_ids", "source_eligible")
    } | {"fallback_no_valid_source": not valid}


def baseline(row, vocabulary):
    prompt = row["prompt_ids"]
    require(len(prompt) == 5 and all(type(i) is int for i in prompt), "five-token prompt")
    require(row["dimension"] in ("TYPE", "COLOR") and row["group"] in GROUPS, "dimension/group")
    require((row["dimension"] == "COLOR") == (row["group"] == "COLOR"), "dimension/group agreement")
    require(
        prompt == [1, prompt[1], 32 if row["dimension"] == "TYPE" else 33, 34, 5]
        and 1024 <= prompt[1] <= 2048,
        "protocol prompt",
    )
    require(vocabulary[prompt[1]] == row["subject"], "subject vocabulary binding")
    truth = row["expected_set_ids"]
    require(
        truth
        and truth == sorted(set(truth))
        and all(type(i) is int and 1024 <= i <= 2048 and i != prompt[1] for i in truth),
        "teacher canonical subject exclusion",
    )
    exact(sorted(vocabulary[i] for i in truth), row["expected"], "teacher keys")
    comp = row["composition"]
    paths = comp["source_paths"]
    require(len(paths) == 8, "baseline eight sources")
    sets = []
    eligibility = []
    first = []
    for rank, path in enumerate(paths, 1):
        members, valid, mapping = source_set(
            path, prompt, BASE + (f"+first-rank{rank}-v1" if rank > 1 else "")
        )
        require(valid, "baseline valid terminated unique source")
        require(
            all(vocabulary[token] == key for token, key in mapping),
            "baseline raw ID/name alignment",
        )
        sets.append(members)
        eligibility.append(valid)
        first.append(path["token_ids"][5])
    require(len(set(first)) == 8, "baseline first-rank distinctness")
    slots = pool(sets, eligibility, row["symmetric_relation_logits"])
    chosen = select(slots)
    exact(
        comp,
        {"source_paths": paths, "slots": slots, **chosen, "policy": POLICY8},
        "baseline full36 composition",
    )
    return sets, eligibility, sorted(set().union(*(set(s) for s in sets)))


def control_parity(actual, saved, prompt, descriptor, vocabulary):
    _, valid, mapping = source_set(actual, prompt, descriptor)
    require(
        valid and all(vocabulary[token] == key for token, key in mapping),
        "fresh control valid/ID mapping",
    )
    exact(
        {k: v for k, v in actual.items() if k != "decoding"},
        {k: v for k, v in saved.items() if k != "decoding"},
        "fresh forced-rank1 full path parity",
    )


def new_path_membership(path, prompt, descriptor, forced, vocabulary):
    require(type(forced) is int and 1024 <= forced <= 2048, "forced first product")
    members, valid, mapping = source_set(path, prompt, descriptor)
    require(path["token_ids"][5] == forced, "forced first ID mismatch")
    require(all(vocabulary[token] == key for token, key in mapping), "new/padding raw ID mapping")
    return members, valid


def expansion(old_union, new_ids, eligible, chosen_anchor):
    if chosen_anchor["active"] and eligible:
        require(
            chosen_anchor["token_id"] in new_ids and chosen_anchor["token_id"] not in old_union,
            "valid active source must expand coverage by its anchor",
        )
        return sorted(set(new_ids) - set(old_union))
    return []


def safe_archive(path, digest, runtime_root=None):
    bind(path, digest)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len({n.casefold() for n in names}), "duplicate archive member")
        hashes = {}
        for name in names:
            require(archive.getinfo(name).orig_filename == name, "unsafe archive member spelling")
            posix = PurePosixPath(name)
            require(
                not posix.is_absolute()
                and posix.parts
                and ".." not in posix.parts
                and posix.as_posix() == name
                and ":" not in name
                and "\\" not in name,
                "unsafe archive member",
            )
            require(not archive.getinfo(name).is_dir(), "unexpected archive directory entry")
            require(
                (archive.getinfo(name).external_attr >> 16) & 0o170000 != 0o120000,
                "archive symlink",
            )
            hashes[name] = hashlib.sha256(archive.read(name)).hexdigest()
            if runtime_root is not None:
                bind(Path(runtime_root) / name, hashes[name])
    return hashes


def check_manifest(manifest):
    require(isinstance(manifest, dict) and bool(manifest), "empty hash manifest")
    for path, digest in manifest.items():
        bind(path, digest)


def runtime_origins(origins, inventory, runtime):
    require(
        {"plm", "plm.serving.runtime", "plm.serving.set_reranking"} <= set(origins),
        "missing required runtime origins",
    )
    for name, observation in origins.items():
        require(name == "plm" or name.startswith("plm."), "non-plm origin")
        path = Path(observation["path"]).resolve()
        require(path.is_relative_to(runtime / "src"), "runtime module outside isolation")
        require(inventory.get(str(path)) == observation["sha256"], "runtime module hash/inventory")
        relative = path.relative_to(runtime / "src")
        expected = name.replace(".", "/")
        require(
            relative.as_posix() in (expected + ".py", expected + "/__init__.py"),
            "module name/file mismatch",
        )


def set_to_list(mapping):
    return sorted(str(Path(p).resolve()) for p in mapping)


DIAGNOSIS_SHA = "47bf9d54967009a2be6e46bedc28f7949231be9bc14e18798fd66ad776c59805"
DIAGNOSIS_AUDIT_SHA = "dc4bda47ac580f2c49d8cdfdd4e8cecc6c17d75c421ba970a1383268403ee16a"
DIAGNOSIS_DECISION_SHA = "ffd804c63618d79f1de986305bf31ec7cc9699a0bb965abbc1399af52261c422"
ARCHIVED_RUNNER_SHA = "2ab48674b526ec6641a6c2366b669ac76bdeac1e8633331e01c772d4393edd39"
ARCHIVED_PLAN_SHA = "02faaedcf88c94c65e70228488431a85e6687e34ea6d6f530f90ec51f43e1461"
SETTINGS = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "deterministic_algorithms": False,
    "autocast_cuda_enabled": False,
    "autocast_cpu_enabled": False,
}


def upstream():
    folder = ROOT / "runs/learning/wide-first-choice-v1"
    summary = read(folder / "summary.json", REFERENCE_SHA)
    audit = read(folder / "independent-audit.json", REFERENCE_AUDIT_SHA)
    accepted = read(folder / "decision.json", ACCEPTANCE_SHA)
    require(
        summary["complete"] is True and summary["final_identity_check"] is True,
        "incomplete eight-source reference",
    )
    require(
        audit["audit_passed"] is True
        and audit["complete"] is True
        and audit["summary_sha256"] == REFERENCE_SHA,
        "upstream audit binding",
    )
    require(
        accepted["evidence_accepted"] is True
        and accepted["fixed_quality_gate_passed"] is True
        and accepted["summary_sha256"] == REFERENCE_SHA
        and accepted["audit_sha256"] == REFERENCE_AUDIT_SHA,
        "upstream owner decision",
    )
    bind(folder / "independent-audit.py", audit["script_sha256"])
    bind(folder / "summary.script.py", ARCHIVED_RUNNER_SHA)
    bind(folder / "summary.plan.md", ARCHIVED_PLAN_SHA)
    bind(folder / "summary.source.zip", SOURCE_SHA)
    bind(folder / "summary.configs.zip", CONFIGS_SHA)
    require([r["seed"] for r in summary["reports"]] == list(SEEDS), "ordered reference seeds")
    reports = []
    for descriptor in summary["reports"]:
        report = read(descriptor["path"], descriptor["sha256"])
        seed = descriptor["seed"]
        require(
            report["complete"] is True
            and report["seed"] == seed
            and report["query_count"] == len(report["responses"]) == 222,
            "reference coverage",
        )
        require(
            report["checkpoint_hash"] == CHECKPOINTS[seed] and report["split_hash"] == SPLIT_SHA,
            "reference checkpoint/split",
        )
        exact(report["product_token_ids"], list(range(1024, 2049)), "product columns")
        reports.append(report)
    inherited = {str(Path(p).resolve()): h for p, h in summary["input_sha256"].items()}
    vocab = ROOT / "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json"
    vocabulary = read(vocab, inherited[str(vocab.resolve())])["tokens"]
    require(len(vocabulary) == len(set(vocabulary)) == 2049, "vocabulary shape")
    diagnosis_folder = ROOT / "runs/learning/missing-source-membership-v1"
    diagnosis = read(diagnosis_folder / "summary.json", DIAGNOSIS_SHA)
    diagnosis_audit = read(diagnosis_folder / "independent-audit.json", DIAGNOSIS_AUDIT_SHA)
    diagnosis_decision = read(diagnosis_folder / "decision.json", DIAGNOSIS_DECISION_SHA)
    require(
        diagnosis["complete"] is True
        and diagnosis_audit["complete"] is True
        and diagnosis_audit["audit_passed"] is True,
        "diagnosis evidence incomplete",
    )
    require(
        diagnosis_audit["summary_sha256"] == DIAGNOSIS_SHA
        and diagnosis_decision["summary_sha256"] == DIAGNOSIS_SHA
        and diagnosis_decision["audit_sha256"] == DIAGNOSIS_AUDIT_SHA,
        "diagnosis chain binding",
    )
    require(diagnosis_decision["evidence_accepted"] is True, "diagnosis not accepted")
    return summary, audit, reports, vocabulary


def answer_metrics(comp, truth):
    successful = comp["selected_source_eligible"] and not comp["fallback_no_valid_source"]
    measures = set_metrics(comp["selected_set_ids"], truth, successful)
    return {k: measures[k] for k in ("exact", "precision", "recall", "f1", "set_size")} | {
        "success": successful
    }


def availability(comp, truth):
    eligible = [s for s in comp["slots"] if s["source_eligible"]]
    exists = any(s["set_ids"] == truth for s in eligible)
    originals = [s for s in eligible if s["kind"] == "original"]
    union = sorted(set().union(*(set(s["set_ids"]) for s in originals)))
    return {
        "exact_available": exists,
        "available_exact_miss": exists and not answer_metrics(comp, truth)["exact"],
        "source_union_contains_truth": set(truth) <= set(union),
        "source_union_ids": union,
        "distinct_eligible_sets": len({tuple(s["set_ids"]) for s in eligible}),
    }


def answer_errors(comp, truth):
    predicted = (
        set(comp["selected_set_ids"])
        if comp["selected_source_eligible"] and not comp["fallback_no_valid_source"]
        else set()
    )
    return {
        "false_positive_ids": sorted(predicted - set(truth)),
        "false_negative_ids": sorted(set(truth) - predicted),
    }


def coverage_changes(old_union, new_ids, eligible, anchor_decision, truth):
    added = expansion(old_union, new_ids, eligible, anchor_decision)
    return {
        "anchor_correct": anchor_decision["token_id"] in truth
        if anchor_decision["active"]
        else None,
        "newly_covered_true_ids": sorted(set(added) & set(truth)),
        "newly_covered_false_ids": sorted(set(added) - set(truth)),
    }


FORCED = "forced-first-explicit-greedy-protocol-mask-v1+unique-v1+kv-v1+batch-v1"
POLICY = "first8-pair28-plus-positive-uncovered-anchor9-unions8-v1"
RUNNER_SHA = "ff41f533bd20e004391ed0337c1d7e8a5ecf24b159331fe1db27945d06609e7f"
RUNNER_RECEIPT_SHA = "0659157a47c8c49e99270aee5eecf959d45543f21d055090203b3db7fa0f082a"


def rebuild_row(old, row, control, vocabulary, index):
    sets, eligible, union = baseline(old, vocabulary)
    values, truth = old["symmetric_relation_logits"], old["expected_set_ids"]
    decision = anchor(union, old["prompt_ids"][1], values)
    forced = (
        decision["token_id"]
        if decision["active"]
        else old["composition"]["source_paths"][0]["token_ids"][5]
    )
    exact(control["index"], index, "control query index")
    exact(control["symmetric_relation_logits"], values, "fresh head exact replay")
    control_parity(
        control["control_path"],
        old["composition"]["source_paths"][0],
        old["prompt_ids"],
        FORCED,
        vocabulary,
    )
    members, valid = new_path_membership(
        row["new_path"], old["prompt_ids"], FORCED, forced, vocabulary
    )
    expansion(union, members, valid, decision)
    paths = list(old["composition"]["source_paths"])
    if decision["active"]:
        sets = [*sets, members]
        eligible = [*eligible, valid]
        paths.append(row["new_path"])
    slots = pool(sets, eligible, values)
    composed = {"source_paths": paths, "slots": slots, **select(slots), "policy": POLICY}
    result = {
        k: old[k]
        for k in (
            "index",
            "subject",
            "dimension",
            "group",
            "prompt_ids",
            "expected",
            "expected_set_ids",
            "symmetric_relation_logits",
        )
    }
    exact(result["index"], index, "query index")
    result.update(
        {
            "baseline": old["composition"],
            "control_path": control["control_path"],
            "anchor": decision,
            "forced_first_id": forced,
            "new_path": row["new_path"],
            "new_source_set_ids": members,
            "new_source_eligible": valid,
            "composition": composed,
        }
    )
    result["metrics"] = {
        "baseline": answer_metrics(old["composition"], truth),
        "new": answer_metrics(composed, truth),
    }
    result["availability"] = {
        "baseline": availability(old["composition"], truth),
        "new": availability(composed, truth),
    }
    result.update(coverage_changes(union, members, valid, decision, truth))
    result.update(answer_errors(composed, truth))
    result["gained_exact"] = (
        result["metrics"]["new"]["exact"] and not result["metrics"]["baseline"]["exact"]
    )
    result["lost_exact"] = (
        result["metrics"]["baseline"]["exact"] and not result["metrics"]["new"]["exact"]
    )
    exact(row, result, "reconstructed complete query row")
    return result


def totals(rows):
    result = {
        "query_count": len(rows),
        "active_anchors": sum(r["anchor"]["active"] for r in rows),
        "inactive_anchors": sum(not r["anchor"]["active"] for r in rows),
        "correct_anchors": sum(r["anchor_correct"] is True for r in rows),
        "false_anchors": sum(r["anchor_correct"] is False for r in rows),
        "gains": sum(r["gained_exact"] for r in rows),
        "losses": sum(r["lost_exact"] for r in rows),
    }
    for key in ("newly_covered_true", "newly_covered_false", "false_positive", "false_negative"):
        result[key + "_occurrences"] = sum(len(r[key + "_ids"]) for r in rows)
    for arm, key in (("baseline", "baseline"), ("new", "composition")):
        result[arm] = {
            "exact_count": sum(r["metrics"][arm]["exact"] for r in rows),
            **{
                m: math.fsum(r["metrics"][arm][m] for r in rows) / len(rows) if rows else 0.0
                for m in ("precision", "recall", "f1", "set_size")
            },
            **{
                m: sum(r["availability"][arm][m] for r in rows)
                for m in ("exact_available", "available_exact_miss", "source_union_contains_truth")
            },
            "added_slot_selections": sum(r[key]["selected_slot"] > 36 for r in rows),
            "failures": sum(not r["metrics"][arm]["success"] for r in rows),
        }
    return result


def aggregate(rows):
    return {
        "overall": totals(rows),
        "groups": {g: totals([r for r in rows if r["group"] == g]) for g in GROUPS},
    }


def gate(reports, pooled):
    checks = {
        "complete_exact_replay": len(reports) == 3
        and all(r["complete"] and r["replay_exact"] and r["query_count"] == 222 for r in reports),
        "active_sources_valid": all(
            row["new_source_eligible"]
            for r in reports
            for row in r["responses"]
            if row["anchor"]["active"]
        ),
        "all_selected_eligible": pooled["overall"]["new"]["failures"] == 0,
        "per_seed_exact_nonregression": all(
            r["overall"]["new"]["exact_count"] >= r["overall"]["baseline"]["exact_count"]
            for r in reports
        ),
        "per_seed_f1_nonregression": all(
            r["overall"]["new"]["f1"] >= r["overall"]["baseline"]["f1"] for r in reports
        ),
        "pooled_exact_improvement": pooled["overall"]["new"]["exact_count"] > 603,
        "pooled_dual_improvement": pooled["groups"]["TYPE_dual"]["new"]["exact_count"] > 155,
        "pooled_group_nonregression": all(
            g["new"]["exact_count"] >= g["baseline"]["exact_count"]
            for g in pooled["groups"].values()
        ),
    }
    return {"checks": checks, "quality_passed": all(checks.values())}


def work_record(paths, active, offset, phase, seconds):
    lengths = [len(p["token_ids"]) - 5 for p in paths]
    calls, count = max(lengths), sum(active)
    useful = sum(n for n, flag in zip(lengths, active, strict=True) if flag)
    return {
        "phase": phase,
        "offset": offset,
        "batch_size": len(paths),
        "decode_calls": calls,
        "wall_seconds": seconds,
        "padded_decode_positions": len(paths) * (calls + 4),
        "total_emitted_tokens": sum(lengths),
        "useful_emitted_tokens": useful,
        "guidance_forwards": 1,
        "active_rows": count,
        "padding_rows": len(paths) - count,
        "active_decode_positions": count * (calls + 4),
        "padding_decode_positions": (len(paths) - count) * (calls + 4),
        "active_emitted_tokens": useful,
        "padding_emitted_tokens": sum(lengths) - useful,
    }


def finite_nonnegative(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid timing")


def audit_work(report):
    require(len(report["work"]) == 56, "work coverage")
    expected = []
    for phase in ("control", "anchor"):
        for offset in range(0, 222, 8):
            rows = report["responses"][offset : offset + 8]
            paths = [r["control_path" if phase == "control" else "new_path"] for r in rows]
            active = [True if phase == "control" else r["anchor"]["active"] for r in rows]
            seconds = report["work"][len(expected)]["wall_seconds"]
            finite_nonnegative(seconds)
            expected.append(work_record(paths, active, offset, phase, seconds))
    exact(report["work"], expected, "independent active/padding work arithmetic")
    seconds = {
        p: math.fsum(w["wall_seconds"] for w in expected if w["phase"] == p)
        for p in ("control", "anchor")
    }
    exact(report["generation_seconds"], seconds, "timing reduction")
    for key in ("head_seconds", "set_scoring_seconds", "wall_seconds"):
        finite_nonnegative(report[key])
    require(
        type(report["peak_gpu_allocated_bytes"]) is int and report["peak_gpu_allocated_bytes"] > 0,
        "peak allocation",
    )
    return {
        "generation_seconds": seconds,
        "head_seconds": report["head_seconds"],
        "set_scoring_seconds": report["set_scoring_seconds"],
        "peak_gpu_allocated_bytes": report["peak_gpu_allocated_bytes"],
        "head_forwards": 28,
        **{
            k: sum(w[k] for w in expected)
            for k in (
                "decode_calls",
                "padded_decode_positions",
                "total_emitted_tokens",
                "useful_emitted_tokens",
                "active_emitted_tokens",
                "padding_emitted_tokens",
                "guidance_forwards",
                "active_decode_positions",
                "padding_decode_positions",
            )
        },
    }


def audit_seed(report, old, vocabulary, folder):
    seed = old["seed"]
    require(
        report["seed"] == seed and report["complete"] is True and report["replay_exact"] is True,
        "incomplete seed",
    )
    require(
        report["query_count"] == len(report["responses"]) == len(report["controls"]) == 222,
        "seed coverage",
    )
    require(
        report["cleanup_errors"] == []
        and not any(k in report for k in ("error", "failed_observation")),
        "seed execution failure",
    )
    for name in ("config", "checkpoint_hash", "training_identity", "corpus_identity", "split_hash"):
        exact(report[name], old[name], "unchanged " + name)
    exact(report["numerical_settings"], SETTINGS, "numerical settings")
    receipt = report["control_receipt"]
    require(
        Path(receipt["path"]).resolve() == folder / f"controls-{seed}.json",
        "control receipt location",
    )
    control = read(receipt["path"], receipt["sha256"])
    exact(
        control,
        {"seed": seed, "replay_exact": True, "responses": report["controls"]},
        "sealed controls",
    )
    rows = [
        rebuild_row(prior, row, ctl, vocabulary, i)
        for i, (prior, row, ctl) in enumerate(
            zip(old["responses"], report["responses"], report["controls"], strict=True)
        )
    ]
    require(len({(r["subject"], r["dimension"]) for r in rows}) == 222, "duplicate query")
    measured = aggregate(rows)
    for key in ("overall", "groups"):
        exact(report[key], measured[key], "seed aggregate " + key)
    for scope in ("overall", *GROUPS):
        current = measured["overall"] if scope == "overall" else measured["groups"][scope]
        previous = old["overall"] if scope == "overall" else old["groups"][scope]
        for key in ("exact_count", "precision", "recall", "f1", "set_size"):
            exact(current["baseline"][key], previous["wide"][key], "accepted baseline metric")
    return rows, audit_work(report)


def authenticate(summary, folder, inherited):
    require(summary["campaign_version"] == "coverage-seeking-branch-v1", "campaign identity")
    require(
        summary["complete"] is True and summary["final_identity_check"] is True,
        "incomplete campaign",
    )
    require(
        summary["acceptance"] is False
        and summary["training_executed"] is False
        and summary["protected_test_used"] is False,
        "scope",
    )
    require(not any(k in summary for k in ("error", "identity_error")), "execution error")
    exact(summary["plan_sha256"], PLAN_SHA, "plan binding")
    exact(summary["script_sha256"], RUNNER_SHA, "runner binding")
    exact(summary["helper_sha256"], ARCHIVED_RUNNER_SHA, "helper binding")
    exact(
        summary["archived_rank_source_sha256"],
        "b5fc41463695c6758ff98471e6c1af5a2cb5ddfb7ca9a85a545b6616908629da",
        "rank source binding",
    )
    bind(ROOT / "scripts/evaluate_coverage_seeking_branch.py", RUNNER_SHA)
    bind(ROOT / "docs/experiments/2026-09-25-coverage-seeking-branch-plan.md", PLAN_SHA)
    for path, digest in inherited["input_sha256"].items():
        relative = Path(path).resolve().relative_to(ROOT)
        if relative.parts[0] == "data" or any(
            f"national_dex_continuation_control_s{s}_v1" in relative.parts for s in SEEDS
        ):
            bind(path, digest)
    check_manifest(summary["input_sha256"])
    normalized = {str(Path(p).resolve()): h for p, h in summary["input_sha256"].items()}
    # Equivalent immutable archive/plan copies are independently authenticated above;
    # the primary consumes the original paths, so bind their exact content digests.
    required_digests = {
        REFERENCE_SHA,
        REFERENCE_AUDIT_SHA,
        ACCEPTANCE_SHA,
        DIAGNOSIS_SHA,
        DIAGNOSIS_AUDIT_SHA,
        DIAGNOSIS_DECISION_SHA,
        SOURCE_SHA,
        CONFIGS_SHA,
        ARCHIVED_RUNNER_SHA,
        ARCHIVED_PLAN_SHA,
        PLAN_SHA,
        RUNNER_SHA,
        *CHECKPOINTS.values(),
        *(r["sha256"] for r in inherited["reports"]),
    }
    require(required_digests <= set(normalized.values()), "required immutable input missing")
    for path, digest in inherited["input_sha256"].items():
        relative = Path(path).resolve().relative_to(ROOT)
        if relative.parts[0] == "data" or any(
            f"national_dex_continuation_control_s{s}_v1" in relative.parts for s in SEEDS
        ):
            require(
                normalized.get(str(Path(path).resolve())) == digest,
                "consumed data/checkpoint recipe binding",
            )
    names = (
        "script.py",
        "helper.py",
        "plan.md",
        "source.zip",
        "configs.zip",
        "test-receipt.json",
        "test-stdout.txt",
        "inputs.json",
    )
    exact(
        set_to_list(summary["snapshot_sha256"]),
        sorted(str(folder / ("summary." + n)) for n in names),
        "snapshot inventory",
    )
    check_manifest(summary["snapshot_sha256"])
    for name, digest in (
        ("script.py", RUNNER_SHA),
        ("helper.py", ARCHIVED_RUNNER_SHA),
        ("plan.md", PLAN_SHA),
    ):
        bind(folder / ("summary." + name), digest)
    receipt = read(folder / "summary.test-receipt.json", RUNNER_RECEIPT_SHA)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["tested_script_sha256"] == RUNNER_SHA
        and receipt["new_branch_measured"] is False
        and receipt["synthetic_cpu_model_only"] is True
        and receipt["test_count"] == 28,
        "runner test receipt",
    )
    require(RUNNER_RECEIPT_SHA in normalized.values(), "runner test receipt not input-bound")
    for path, digest in (
        (receipt["test_source"], receipt["tested_test_sha256"]),
        (receipt["stdout"]["path"], receipt["stdout"]["sha256"]),
    ):
        bind(path, digest)
        require(normalized.get(str(Path(path).resolve())) == digest, "focused test input binding")
    bind(folder / "summary.test-stdout.txt", receipt["stdout"]["sha256"])
    exact(
        json.loads((folder / "summary.inputs.json").read_text(encoding="utf-8")),
        summary["input_sha256"],
        "input snapshot",
    )
    runtime = folder / "runtime"
    source = safe_archive(folder / "summary.source.zip", SOURCE_SHA, runtime)
    configs = safe_archive(folder / "summary.configs.zip", CONFIGS_SHA, runtime)
    require(
        not ({n.casefold() for n in source} & {n.casefold() for n in configs}),
        "archive member collision",
    )
    expected = {str((runtime / n).resolve()): h for n, h in (source | configs).items()}
    exact(summary["runtime_files_sha256"], expected, "isolated runtime inventory")
    require(
        {
            str(p.resolve())
            for p in runtime.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        == set(expected),
        "unexpected runtime file",
    )
    runtime_origins(summary["imported_modules"], expected, runtime)
    exact(summary["environment"], inherited["environment"], "same runtime environment")
    exact(summary["numerical_settings"], SETTINGS, "same numerical settings")
    finite_nonnegative(summary["wall_seconds"])


def audit(summary_path):
    folder = summary_path.parent
    summary = read(summary_path, sha(summary_path))
    inherited, _, previous, vocabulary = upstream()
    authenticate(summary, folder, inherited)
    require([r["seed"] for r in summary["reports"]] == list(SEEDS), "ordered three seed reports")
    rows, measured_reports, work = [], [], []
    for descriptor, old in zip(summary["reports"], previous, strict=True):
        require(
            Path(descriptor["path"]).resolve() == folder / f"seed-{old['seed']}.json",
            "seed report path",
        )
        report = read(descriptor["path"], descriptor["sha256"])
        rebuilt, performed = audit_seed(report, old, vocabulary, folder)
        rows.extend(rebuilt)
        measured_reports.append({**report, **aggregate(rebuilt), "responses": rebuilt})
        work.append({"seed": old["seed"], **performed})
    pooled = aggregate(rows)
    for key in ("overall", "groups"):
        exact(summary[key], pooled[key], "pooled " + key)
    fixed_gate = gate(measured_reports, pooled)
    exact(summary["gate"], fixed_gate, "fixed quality gate")
    accounting = {
        "query_seed_observations": 666,
        "distinct_queries": 222,
        "reused_source_paths": 5328,
        "fresh_control_paths": 666,
        "fresh_active_paths": sum(r["anchor"]["active"] for r in rows),
        "fresh_padding_paths": sum(not r["anchor"]["active"] for r in rows),
        "candidate_slots": sum(len(r["composition"]["slots"]) for r in rows),
    }
    exact(summary["accounting"], accounting, "work coverage accounting")
    tests_dir = ROOT / "runs/learning/coverage-seeking-branch-auditor-tests-v1"
    receipt = read(tests_dir / "test-receipt.json", sha(tests_dir / "test-receipt.json"))
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["model_execution"] is False
        and receipt["new_diagnostic_executed"] is False
        and receipt["tested_auditor_sha256"] == sha(__file__)
        and receipt["passed_count"] >= 25,
        "auditor test receipt",
    )
    bind(tests_dir / "test_auditor.py", receipt["tests_sha256"])
    bind(tests_dir / "stdout.txt", receipt["stdout_sha256"])
    check_manifest(dict(INPUTS))
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(summary_path),
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN_SHA,
        "test_receipt_sha256": sha(tests_dir / "test-receipt.json"),
        "input_sha256": dict(INPUTS),
        **pooled,
        "gate": fixed_gate,
        "accounting": accounting,
        "work": work,
        "per_seed": [
            {"seed": r["seed"], "overall": r["overall"], "groups": r["groups"]}
            for r in measured_reports
        ],
        "limitations": [
            "Independent saved-path parsing and rational score/metric reductions; "
            "no independent neural execution or later-argmax proof.",
            "Fresh heads and forced-first controls are compared exactly; "
            "eight historical source paths are reused, not freshly generated.",
            "Chronological control sealing and synchronization are frozen runner and "
            "receipt evidence; hashes alone do not prove chronology.",
            "Timing reductions are descriptive and not a throughput, energy, "
            "or concurrency benchmark.",
            "set_scoring_seconds includes set scoring, postprocessing and diagnostic analysis.",
            "Evidence audit does not grant policy acceptance or default promotion.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()
    require("torch" not in sys.modules, "Torch-free audit")
    summary_path = args.summary.resolve()
    output = summary_path.parent / "independent-audit.json"
    require(not output.exists(), "immutable audit output exists")
    result = audit(summary_path)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "audit_sha256": sha(output),
                "summary_sha256": result["summary_sha256"],
                "overall": result["overall"],
                "gate": result["gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
