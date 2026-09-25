"""Independent saved-output width-eight audit; no model or evaluator imports.

Exact rational accumulation cross-checks saved FP32 set logits. Neural top-eight
ranking and greedy transitions remain source/receipt evidence, not a second
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
PLAN_SHA = "02faaedcf88c94c65e70228488431a85e6687e34ea6d6f530f90ec51f43e1461"
REFERENCE_SHA = "77f9d9bba8eb9c5e0cf5df86f2eb29f522cf00ad50600ee9eea76267ec041c8f"
REFERENCE_AUDIT_SHA = "40eba8d9dac72dbe762eb4b54e7c053128b08580f905dd7539a89961c144ebde"
ACCEPTANCE_SHA = "1d5835709646a9dbf7f8db2727e3adabef2c451ff75be932e2eb22833acdd9d4"
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


def composition(raw_paths, prompt, values):
    """Rebuild fixed slots and label-free winner from saved neural outputs."""
    vector(values)
    width = len(raw_paths)
    require(width in (4, 8), "source width must be four or eight")
    order = ORDER4 if width == 4 else ORDER8
    sources, eligible, mapping, removed = [], [], [], []
    for rank, raw in enumerate(raw_paths, 1):
        ids, ok, pairs = source_set(raw, prompt, rank)
        sources.append(set(ids))
        eligible.append(ok)
        mapping.extend(pairs)
        removed.append(
            raw["protocol_valid"] and raw["terminated"] and prompt[1] in raw["token_ids"][5:]
        )
    first = [raw["token_ids"][5] for raw in raw_paths]
    require(
        len(set(first)) == width and all(1024 <= i <= 2048 for i in first),
        "first-rank masks/distinctness",
    )
    slots = []
    for index, ranks in enumerate(order, 1):
        ids = sorted(set().union(*(sources[r - 1] for r in ranks)))
        slots.append(
            {
                "slot": index,
                "kind": "original" if len(ranks) == 1 else "pair_composition",
                "source_ranks": list(ranks),
                "set_ids": ids,
                "source_eligible": all(eligible[r - 1] for r in ranks),
                "score": canonical_sum(ids, values),
            }
        )
    valid = [s for s in slots if s["source_eligible"]]
    winner = max(valid, key=lambda s: (s["score"], -s["slot"])) if valid else slots[0]
    return (
        {
            "source_paths": raw_paths,
            "slots": slots,
            "selected_slot": winner["slot"],
            "selected_kind": winner["kind"],
            "selected_source_ranks": winner["source_ranks"],
            "selected_set_ids": winner["set_ids"],
            "selected_source_eligible": winner["source_eligible"],
            "fallback_no_valid_source": not valid,
            "policy": POLICY4 if width == 4 else POLICY8,
        },
        removed,
        mapping,
    )


def query_identity(row, index, vocabulary):
    prompt = row["prompt_ids"]
    require(row["index"] == index and type(row["index"]) is int, "query index")
    require(row["dimension"] in ("TYPE", "COLOR"), "query dimension")
    require(
        row["group"] in GROUPS and ((row["group"] == "COLOR") == (row["dimension"] == "COLOR")),
        "query group",
    )
    require(len(prompt) == 5 and all(type(i) is int for i in prompt), "five-token prompt")
    require(
        prompt == [1, prompt[1], 32 if row["dimension"] == "TYPE" else 33, 34, 5], "protocol prompt"
    )
    require(
        1024 <= prompt[1] <= 2048 and vocabulary[prompt[1]] == row["subject"], "subject identity"
    )
    truth = row["expected_set_ids"]
    require(
        truth
        and truth == sorted(set(truth))
        and all(type(i) is int and 1024 <= i <= 2048 and i != prompt[1] for i in truth),
        "teacher IDs",
    )
    require(sorted(vocabulary[i] for i in truth) == sorted(row["expected"]), "teacher key mapping")


def available(comp, truth):
    eligible = [s for s in comp["slots"] if s["source_eligible"]]
    originals = [s for s in eligible if s["kind"] == "original"]
    exact_available = any(s["set_ids"] == truth for s in eligible)
    selected_exact = comp["selected_source_eligible"] and comp["selected_set_ids"] == truth
    union = set().union(*(set(s["set_ids"]) for s in originals))
    return {
        "exact_available": exact_available,
        "available_exact_miss": exact_available and not selected_exact,
        "source_union_contains_truth": set(truth) <= union,
        "source_union_ids": sorted(union),
        "distinct_eligible_sets": len({tuple(s["set_ids"]) for s in eligible}),
    }


def metric_mean(metrics):
    count = len(metrics)
    require(count > 0, "empty metric aggregate")
    return {
        "query_count": count,
        "exact_count": sum(m["exact"] for m in metrics),
        **{
            name: math.fsum(m[name] for m in metrics) / count
            for name in ("precision", "recall", "f1", "set_size")
        },
    }


def gate(seed_summaries, pooled, grouped, valid=True, replay=True):
    per_seed = [
        {
            "seed": seed,
            "exact_nonregression": summary["wide"]["exact_count"]
            >= summary["baseline"]["exact_count"],
            "f1_nonregression": summary["wide"]["f1"] >= summary["baseline"]["f1"],
        }
        for seed, summary in seed_summaries
    ]
    result = {
        "baseline_replay_exact": replay,
        "all_sources_valid_terminated_unique_and_selection_eligible": valid,
        "per_seed": per_seed,
        "no_per_seed_regression": all(
            s["exact_nonregression"] and s["f1_nonregression"] for s in per_seed
        ),
        "strict_pooled_exact_gain": pooled["wide"]["exact_count"]
        > pooled["baseline"]["exact_count"],
        "strict_pooled_dual_exact_gain": grouped["TYPE_dual"]["wide"]["exact_count"]
        > grouped["TYPE_dual"]["baseline"]["exact_count"],
        "pooled_group_exact_nonregression": all(
            g["wide"]["exact_count"] >= g["baseline"]["exact_count"] for g in grouped.values()
        ),
    }
    result["eligible_for_further_research"] = all(v for k, v in result.items() if k != "per_seed")
    return result


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


def historical_references():
    directory = ROOT / "runs/learning/pair-composition-integration-v2"
    summary = read(directory / "summary.json", REFERENCE_SHA)
    audit = read(directory / "independent-audit.json", REFERENCE_AUDIT_SHA)
    accepted = read(directory / "acceptance.json", ACCEPTANCE_SHA)
    require(
        accepted["integration_accepted"] is True and summary["verification_complete"] is True,
        "unaccepted reference",
    )
    require(
        accepted["summary_sha256"] == REFERENCE_SHA
        and accepted["independent_audit_sha256"] == REFERENCE_AUDIT_SHA,
        "acceptance binding",
    )
    # Consume the certified immutable report chain, without redoing training or HTTP.
    manifest = {str(Path(p).resolve()): h for p, h in summary["input_sha256"].items()}
    references = {}
    for seed, record in zip(SEEDS, summary["offline_reuse"], strict=True):
        require(record["seed"] == seed, "reference seed order")
        offline = read(record["report"], record["report_sha256"])
        head_path = ROOT / f"runs/learning/pair-unions-v1/seed-{seed}.json"
        require(str(head_path.resolve()) in manifest, "head report absent from accepted chain")
        heads = read(head_path, manifest[str(head_path.resolve())])
        require(
            offline["checkpoint_hash"] == heads["checkpoint_hash"] == CHECKPOINTS[seed],
            "reference checkpoint",
        )
        require(offline["split_hash"] == heads["split_hash"] == SPLIT_SHA, "reference split")
        for name in ("training_identity", "corpus_identity"):
            exact(offline[name], heads[name], f"reference {name}")
        require(
            len(offline["responses"]) == len(heads["responses"]) == 222, "reference query count"
        )
        references[seed] = (offline, heads)
    vocab_path = ROOT / "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json"
    vocabulary = read(vocab_path, manifest[str(vocab_path.resolve())])["tokens"]
    require(len(vocabulary) == 2049 and len(set(vocabulary)) == 2049, "vocabulary shape")
    require(audit.get("audit_passed") is True, "reference audit not passed")
    return references, vocabulary, summary


def answer_metrics(comp, truth):
    successful = comp["selected_source_eligible"] and not comp["fallback_no_valid_source"]
    metrics = set_metrics(comp["selected_set_ids"], truth, successful)
    return {k: metrics[k] for k in ("exact", "precision", "recall", "f1", "set_size")} | {
        "success": successful
    }


def rebuild_row(base, row, vocabulary, index):
    query_identity(base, index, vocabulary)
    for key, value in base.items():
        if key != "composition":
            exact(row[key], value, f"wider row changed baseline field {key}")
    exact(row["baseline"], base["composition"], "retained baseline composition")
    head = base["symmetric_relation_logits"]
    baseline, _, base_mapping = composition(
        base["composition"]["source_paths"], base["prompt_ids"], head
    )
    exact(base["composition"], baseline, "reconstructed baseline")
    wide, removed, mapping = composition(
        row["composition"]["source_paths"], base["prompt_ids"], head
    )
    exact(row["composition"], wide, "reconstructed wide composition")
    exact(wide["source_paths"][:4], baseline["source_paths"], "legacy paths changed")
    exact(wide["slots"][:10], baseline["slots"], "legacy slots changed")
    for product, key in base_mapping + mapping:
        require(vocabulary[product] == key, "raw token/target key mapping")
    metrics = {
        arm: answer_metrics(comp, base["expected_set_ids"])
        for arm, comp in (("baseline", baseline), ("wide", wide))
    }
    result = {
        **{k: v for k, v in base.items() if k != "composition"},
        "baseline": baseline,
        "composition": wide,
        "metrics": metrics,
        "availability": {
            arm: available(comp, base["expected_set_ids"])
            for arm, comp in (("baseline", baseline), ("wide", wide))
        },
        "subject_removed": removed,
        "gained_exact": metrics["wide"]["exact"] and not metrics["baseline"]["exact"],
        "lost_exact": metrics["baseline"]["exact"] and not metrics["wide"]["exact"],
    }
    exact(row, result, "full reconstructed response")
    return result


def totals(rows):
    require(bool(rows), "empty aggregate")
    result = {
        "query_count": len(rows),
        "gains": sum(r["gained_exact"] for r in rows),
        "losses": sum(r["lost_exact"] for r in rows),
    }
    for arm, key in (("baseline", "baseline"), ("wide", "composition")):
        metrics = metric_mean([r["metrics"][arm] for r in rows])
        metrics.pop("query_count")
        availability = {
            name: sum(r["availability"][arm][name] for r in rows)
            for name in ("exact_available", "available_exact_miss", "source_union_contains_truth")
        }
        chosen = [r[key] for r in rows if r["metrics"][arm]["success"]]
        result[arm] = {
            **metrics,
            **availability,
            "mean_distinct_eligible_sets": math.fsum(
                r["availability"][arm]["distinct_eligible_sets"] for r in rows
            )
            / len(rows),
            "original_selections": sum(c["selected_kind"] == "original" for c in chosen),
            "pair_selections": sum(c["selected_kind"] == "pair_composition" for c in chosen),
            "added_slot_selections": sum(c["selected_slot"] > 10 for c in chosen),
            "failures": len(rows) - len(chosen),
        }
    return result


def aggregate(rows):
    return {
        "overall": totals(rows),
        "groups": {g: totals([r for r in rows if r["group"] == g]) for g in GROUPS},
    }


def all_sources_valid(rows):
    return all(
        p["protocol_valid"]
        and p["terminated"]
        and p["error"] is None
        and len(p["token_ids"][5:-1]) == len(set(p["token_ids"][5:-1]))
        for r in rows
        for p in r["composition"]["source_paths"]
    )


def reported_gate(reports, pooled):
    pairs = [(r["seed"], r["overall"]) for r in reports]
    valid = all_sources_valid([r for report in reports for r in report["responses"]])
    replay = len(reports) == 3 and all(
        r["complete"] is True and r["baseline_replay_exact"] is True and r["query_count"] == 222
        for r in reports
    )
    facts = gate(pairs, pooled["overall"], pooled["groups"], valid=valid, replay=replay)
    require(pooled["overall"]["baseline"]["exact_count"] == 569, "baseline exact count changed")
    checks = {
        "complete_exact_replay": replay,
        "all_sources_valid_terminated_unique": valid,
        "all_selected_eligible": pooled["overall"]["wide"]["failures"] == 0,
        "per_seed_exact_nonregression": all(r["exact_nonregression"] for r in facts["per_seed"]),
        "per_seed_f1_nonregression": all(r["f1_nonregression"] for r in facts["per_seed"]),
        "pooled_exact_improvement": facts["strict_pooled_exact_gain"],
        "pooled_group_exact_nonregression": facts["pooled_group_exact_nonregression"],
        "pooled_dual_exact_improvement": facts["strict_pooled_dual_exact_gain"],
    }
    return {"checks": checks, "quality_passed": all(checks.values())}


def work_audit(report, baseline):
    records = report["batch_work"]
    ordered = [
        (offset, rank)
        for ranks in (range(1, 5), range(5, 9))
        for offset in range(0, 222, 8)
        for rank in ranks
    ]
    require(len(records) == 224, "rank/batch work coverage")
    exact(baseline["batch_work"], records[:112], "baseline work before expansion")
    for record, (offset, rank) in zip(records, ordered, strict=True):
        rows = report["responses"][offset : offset + 8]
        lengths = [len(r["composition"]["source_paths"][rank - 1]["token_ids"]) - 5 for r in rows]
        expected = {
            "offset": offset,
            "rank": rank,
            "batch_size": len(rows),
            "decode_calls": max(lengths),
            "padded_decode_positions": len(rows) * (max(lengths) + 4),
            "useful_emitted_tokens": sum(lengths),
            "guidance_forward_calls": 1,
        }
        exact(
            {k: v for k, v in record.items() if k != "wall_seconds"},
            expected,
            "derived decoder work",
        )
        require(
            type(record["wall_seconds"]) is float
            and math.isfinite(record["wall_seconds"])
            and record["wall_seconds"] >= 0,
            "invalid measured generation time",
        )
    for name in ("wall_seconds", "head_seconds", "set_scoring_seconds"):
        require(
            type(report[name]) is float and math.isfinite(report[name]) and report[name] >= 0,
            f"invalid {name}",
        )
    require(
        type(report["peak_gpu_allocated_bytes"]) is int and report["peak_gpu_allocated_bytes"] > 0,
        "GPU allocation receipt",
    )
    exact(baseline["head_seconds"], report["head_seconds"], "single unchanged head pass")
    exact(
        report["generation_seconds"],
        {
            "ranks1_4": math.fsum(w["wall_seconds"] for w in records if w["rank"] <= 4),
            "ranks5_8": math.fsum(w["wall_seconds"] for w in records if w["rank"] > 4),
        },
        "generation timing reductions",
    )
    return {
        "rank_1_to_4_seconds": math.fsum(w["wall_seconds"] for w in records if w["rank"] <= 4),
        "rank_5_to_8_seconds": math.fsum(w["wall_seconds"] for w in records if w["rank"] > 4),
        "decode_calls": sum(w["decode_calls"] for w in records),
        "padded_decode_positions": sum(w["padded_decode_positions"] for w in records),
        "useful_emitted_tokens": sum(w["useful_emitted_tokens"] for w in records),
        "guidance_forward_calls": 224,
        "prompt_head_forward_calls": 28,
    }


SETTINGS = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "deterministic_algorithms": False,
    "autocast_cuda_enabled": False,
    "autocast_cpu_enabled": False,
}


def audit_seed(report, old, heads, vocabulary, output_dir):
    seed = report["seed"]
    require(
        seed in SEEDS and report["complete"] is True and report["baseline_replay_exact"] is True,
        "seed incomplete",
    )
    require(
        report["query_count"] == 222
        and len(report["responses"]) == 222
        and len(report["baseline_observations"]) == 222,
        "seed query coverage",
    )
    require(
        report["cleanup_errors"] == []
        and "error" not in report
        and "failed_observation" not in report,
        "seed execution failure",
    )
    require(
        report["checkpoint_hash"] == CHECKPOINTS[seed] and report["split_hash"] == SPLIT_SHA,
        "seed identities",
    )
    for name in ("config", "training_identity", "corpus_identity"):
        exact(report[name], old[name], f"reference {name} changed")
    exact(report["numerical_settings"], SETTINGS, "seed numerical settings")
    exact(report["product_token_ids"], list(range(1024, 2049)), "head column mapping")
    receipt = report["baseline_receipt"]
    require(
        Path(receipt["path"]).resolve() == (output_dir / f"baseline-{seed}.json").resolve(),
        "baseline report location",
    )
    baseline = read(receipt["path"], receipt["sha256"])
    require(baseline["seed"] == seed and baseline["exact_replay"] is True, "baseline seal")
    exact(baseline["responses"], report["baseline_observations"], "sealed baseline observations")
    exact(baseline["product_token_ids"], report["product_token_ids"], "baseline columns")
    rows = []
    for index, (base, row, reference, head) in enumerate(
        zip(
            baseline["responses"],
            report["responses"],
            old["responses"],
            heads["responses"],
            strict=True,
        )
    ):
        for name in ("subject", "dimension", "group"):
            exact(base[name], reference[name], f"historical query {name}")
            exact(base[name], head[name], f"historical head query {name}")
        exact(base["expected"], sorted(reference["expected"]), "historical teacher")
        exact(
            base["symmetric_relation_logits"],
            head["symmetric_relation_logits"],
            "exact historical head replay",
        )
        exact(
            base["composition"], reference["composition"], "exact historical raw/composition replay"
        )
        rows.append(rebuild_row(base, row, vocabulary, index))
    require(
        len({(r["subject"], r["dimension"]) for r in rows}) == 222, "duplicate validation query"
    )
    derived = aggregate(rows)
    for name in ("overall", "groups"):
        exact(report[name], derived[name], f"seed {name} aggregate")
    for name in ("exact_count", "precision", "recall", "f1", "set_size"):
        exact(
            derived["overall"]["baseline"][name],
            old["metrics"][name],
            f"historical baseline metric {name}",
        )
        for group in GROUPS:
            exact(
                derived["groups"][group]["baseline"][name],
                old["groups"][group][name],
                f"historical group metric {name}",
            )
    return rows, work_audit(report, baseline)


RUNNER_SHA = "2ab48674b526ec6641a6c2366b669ac76bdeac1e8633331e01c772d4393edd39"
RUNNER_RECEIPT_SHA = "0249dbde598d33063888598792193b25e9b46f509e99172200a8afe990c9a807"


def authenticate_execution(summary, output_dir, inherited):
    require(summary["campaign_version"] == "wide-first-choice-v1", "campaign version")
    require(
        summary["complete"] is True and summary["final_identity_check"] is True,
        "campaign incomplete",
    )
    require(
        summary["acceptance"] is False
        and summary["training_executed"] is False
        and summary["protected_test_used"] is False,
        "scope/acceptance",
    )
    require("error" not in summary and "identity_error" not in summary, "execution error")
    require(
        summary["plan_sha256"] == PLAN_SHA and summary["script_sha256"] == RUNNER_SHA,
        "frozen declaration/runner",
    )
    bind(ROOT / "docs/experiments/2026-09-25-wide-first-choice-plan.md", PLAN_SHA)
    bind(ROOT / "scripts/evaluate_wide_first_choices.py", RUNNER_SHA)
    # Authenticate consumed immutable data and the three old training receipts;
    # no training or protected-split measurement is repeated.
    for path, digest in inherited["input_sha256"].items():
        path = Path(path).resolve()
        relative = path.relative_to(ROOT)
        if relative.parts[0] == "data" or any(
            f"national_dex_continuation_control_s{s}_v1" in relative.parts for s in SEEDS
        ):
            bind(path, digest)
    expected_inputs = {p: h for p, h in INPUTS.items() if Path(p) != output_dir / "summary.json"}
    check_manifest(summary["input_sha256"])
    normalized = {str(Path(p).resolve()): h for p, h in summary["input_sha256"].items()}
    for path, digest in expected_inputs.items():
        require(normalized.get(path) == digest, f"required input absent or changed: {path}")
    snapshots = summary["snapshot_sha256"]
    names = {
        "script.py",
        "plan.md",
        "source.zip",
        "configs.zip",
        "test-receipt.json",
        "test-stdout.txt",
        "inputs.json",
    }
    exact(
        set_to_list(snapshots),
        sorted(str((output_dir / f"summary.{n}").resolve()) for n in names),
        "snapshot inventory",
    )
    check_manifest(snapshots)
    bind(output_dir / "summary.script.py", RUNNER_SHA)
    bind(output_dir / "summary.plan.md", PLAN_SHA)
    receipt = read(output_dir / "summary.test-receipt.json", RUNNER_RECEIPT_SHA)
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["tested_script_sha256"] == RUNNER_SHA
        and receipt["real_inference_executed"] is False
        and receipt["test_count"] == 29,
        "runner test receipt",
    )
    require(
        any(h == RUNNER_RECEIPT_SHA for h in normalized.values()),
        "runner test receipt not input-bound",
    )
    bind(ROOT / "tests/unit/test_wide_first_choices.py", receipt["tested_test_sha256"])
    bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"])
    bind(output_dir / "summary.test-stdout.txt", receipt["stdout"]["sha256"])
    frozen_inputs = json.loads((output_dir / "summary.inputs.json").read_text(encoding="utf-8"))
    exact(frozen_inputs, summary["input_sha256"], "frozen input manifest")
    runtime = output_dir / "runtime"
    source = safe_archive(output_dir / "summary.source.zip", SOURCE_SHA, runtime)
    configs = safe_archive(output_dir / "summary.configs.zip", CONFIGS_SHA, runtime)
    require(
        not ({n.casefold() for n in source} & {n.casefold() for n in configs}),
        "duplicate merged runtime member",
    )
    members = source | configs
    expected_runtime = {str((runtime / n).resolve()): h for n, h in members.items()}
    exact(summary["runtime_files_sha256"], expected_runtime, "isolated archive byte inventory")
    check_manifest(summary["runtime_files_sha256"])
    runtime_origins(summary["imported_modules"], expected_runtime, runtime)
    actual_inventory = {
        str(p.resolve()) for p in runtime.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    require(actual_inventory == set(expected_runtime), "extra or missing runtime file")
    require(
        type(summary["wall_seconds"]) is float
        and math.isfinite(summary["wall_seconds"])
        and summary["wall_seconds"] > 0,
        "campaign wall time",
    )
    exact(summary["numerical_settings"], SETTINGS, "fixed numerical settings")
    env = summary["environment"]
    exact(
        {k: env[k] for k in ("python", "torch", "cuda", "gpu")},
        {
            "python": "3.12.13",
            "torch": "2.11.0+cu128",
            "cuda": "12.8",
            "gpu": "NVIDIA GeForce RTX 5070 Ti",
        },
        "historical dependency/device settings",
    )
    require(
        isinstance(env["platform"], str) and env["platform"].startswith("Windows-"),
        "native platform",
    )
    return runtime


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


def own_test_receipt():
    directory = ROOT / "runs/learning/wide-first-choice-auditor-tests-v1"
    receipt_path = directory / "test-receipt.json"
    receipt = read(receipt_path, sha(receipt_path))
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["model_execution"] is False
        and receipt["new_diagnostic_executed"] is False,
        "independent test receipt",
    )
    require(
        receipt["tested_auditor_sha256"] == sha(__file__) and receipt["passed_count"] >= 26,
        "auditor was not tested",
    )
    bind(directory / "test_auditor.py", receipt["tests_sha256"])
    bind(directory / "stdout.txt", receipt["stdout_sha256"])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    output = summary_path.parent / "independent-audit.json"
    require(not output.exists(), "refusing to overwrite independent audit")
    require(summary_path.exists(), "complete primary summary required")
    summary = read(summary_path, sha(summary_path))
    require(summary.get("complete") is True, "incomplete primary cannot receive successful audit")
    refs, vocabulary, historical = historical_references()
    authenticate_execution(summary, summary_path.parent, historical)
    own_test_receipt()
    bind(__file__, sha(__file__))
    records = summary["reports"]
    require([r["seed"] for r in records] == list(SEEDS), "three ordered seed reports required")
    reports, rows, work = [], [], []
    reference_query_order = None
    for record in records:
        seed = record["seed"]
        require(
            Path(record["path"]).resolve() == summary_path.parent / f"seed-{seed}.json", "seed path"
        )
        report = read(record["path"], record["sha256"])
        require(report["seed"] == seed, "seed/report mismatch")
        seed_rows, seed_work = audit_seed(report, *refs[seed], vocabulary, summary_path.parent)
        order = [(r["subject"], r["dimension"], r["expected_set_ids"]) for r in seed_rows]
        if reference_query_order is None:
            reference_query_order = order
        else:
            exact(order, reference_query_order, "query partition/order changed across seeds")
        reports.append(report)
        rows.extend(seed_rows)
        work.append({"seed": seed, **seed_work})
    pooled = aggregate(rows)
    exact(summary["overall"], pooled["overall"], "pooled aggregate")
    exact(summary["groups"], pooled["groups"], "pooled groups")
    require(
        pooled["overall"]["baseline"]["exact_count"] == 569
        and pooled["overall"]["baseline"]["f1"] == 0.9668960529174291
        and pooled["overall"]["baseline"]["exact_available"] == 570,
        "declared historical quality/availability",
    )
    result_gate = reported_gate(reports, pooled)
    exact(summary["gate"], result_gate, "declared quality gate")
    changed = [
        {
            "seed": report["seed"],
            **{k: r[k] for k in ("index", "subject", "dimension", "gained_exact", "lost_exact")},
        }
        for report in reports
        for r in report["responses"]
        if r["gained_exact"] or r["lost_exact"]
    ]
    exact(summary["changed_queries"], changed, "paired changed-query inventory")
    accounting = {
        "query_seed_observations": 666,
        "distinct_queries": 222,
        "source_paths": 5328,
        "candidate_slots": 23976,
    }
    exact(summary["accounting"], accounting, "full fixed coverage")
    require(
        "torch" not in sys.modules
        and not any(k == "plm" or k.startswith("plm.") for k in sys.modules),
        "auditor imported model/runtime",
    )
    for path, digest in INPUTS.items():
        require(sha(path) == digest, f"input changed during audit: {path}")
    result = {
        "audit_version": "wide-first-choice-independent-v1",
        "audit_passed": True,
        "complete": True,
        "acceptance": False,
        "summary_sha256": sha(summary_path),
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN_SHA,
        "accounting": accounting,
        **pooled,
        "gate": result_gate,
        "work": work,
        "per_seed": [
            {"seed": r["seed"], "overall": r["overall"], "groups": r["groups"]} for r in reports
        ],
        "input_sha256": dict(sorted(INPUTS.items())),
        "limitations": [
            (
                "Independent arithmetic and protocol audit of authenticated saved neural "
                "outputs; no neural replay or autograd proof."
            ),
            (
                "Saved paths alone do not prove top-eight learned first-logit ordering or "
                "later greedy choices; frozen source and runner receipts document those "
                "operations."
            ),
            (
                "Measured wall times and GPU allocation are authenticated receipts; "
                "derived decode work is independently reconstructed, not a serving "
                "throughput benchmark."
            ),
            (
                "222 distinct validation queries across three seeds; no protected-test "
                "measurement, new training or default-policy promotion."
            ),
            (
                "A scientific quality-gate pass or failure is separate from faithful "
                "execution; owner acceptance remains separate."
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
                "summary_sha256": sha(summary_path),
                "accounting": accounting,
                "overall": pooled["overall"],
                "gate": result_gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
