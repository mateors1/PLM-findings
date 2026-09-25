"""Independent stdlib audit of the declared membership separability diagnosis.

Authentication follows the prior direct-membership audit. All measurement math
is independently implemented; no evaluator or custom helper is imported.
Run only after the primary diagnostic has finished and the owner authorizes audit.
"""

import argparse
import hashlib
import json
import math
import re
import sqlite3
import struct
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = (1729, 1730, 1731)
PAIR_SHA = "4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992"
PAIR_AUDIT_SHA = "13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057"
DIAG_SHA = "f5c826295bfc30466dc435c5a995134fe198686f69f2890cac2e3f2435053523"
DIAG_AUDIT_SHA = "1639c4b6fd2a45600436fbbd4d9a566e43be9016ed4e618750604c3315b7f895"
DIRECT_PLAN_SHA = "4004639897b88d54090ce94bfecf26adf5b2a75686be4b8cdf97da2155d19da0"
INPUTS = {}


DIRECT_DIR = ROOT / "runs/learning/direct-membership-v1"
PLAN_SHA = "b189638ba7c89e380934a0239537fd8a8f6a4f34f68058c619fe86fa5e7bca87"
DIRECT_SHA = "2cbe82aad296b6ff4a920a951de38a5999a23e0eb8507d40444254e5d404ceab"
DIRECT_AUDIT_SHA = "8111692647153787e095da94c64ec497ebc41537b6f6c3ea8ad1d5b046b8df42"
HELPER_SHA = "4f9baef52a7f365bfafdb46041ede98cf4d04ff27c130e27c053bb4726f4e4c6"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path, digest):
    path = Path(path).resolve()
    require(sha(path) == digest, f"input hash mismatch: {path}")
    require(str(path) not in INPUTS or INPUTS[str(path)] == digest, f"input drift: {path}")
    INPUTS[str(path)] = digest
    return path


def read(path, digest=None):
    path = bind(path, sha(path) if digest is None else digest)
    return json.loads(path.read_text(encoding="utf-8"))


def members(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate archive member")
        return {name: archive.read(name) for name in names}


def remove_default(payload, field):
    lines = payload.splitlines(keepends=True)
    selected = [line for line in lines if line.startswith(field.encode() + b":")]
    require(
        len(selected) == 1
        and re.fullmatch(
            re.escape(field.encode()) + rb":[ \t]+false[ \t]*(?:#[^\r\n]*)?\r?\n?", selected[0]
        ),
        "unrecognized default migration",
    )
    return b"".join(line for line in lines if line != selected[0])


def authenticate_direct(summary):
    pair_dir = ROOT / "runs/learning/pair-unions-v1"
    diagnosis_dir = ROOT / "runs/learning/pair-error-diagnosis-v1"
    pair = read(pair_dir / "summary.json", PAIR_SHA)
    audit = read(pair_dir / "independent-audit.json", PAIR_AUDIT_SHA)
    diagnosis = read(diagnosis_dir / "summary.json", DIAG_SHA)
    diagnosis_audit = read(diagnosis_dir / "independent-audit.json", DIAG_AUDIT_SHA)
    require(
        audit["summary_sha256"] == PAIR_SHA and audit["all_arithmetic_and_provenance_equal"],
        "pair audit binding",
    )
    require(
        diagnosis_audit["summary_sha256"] == DIAG_SHA
        and diagnosis_audit["all_recomputed_counts_equal"],
        "diagnosis audit binding",
    )
    for directory, manifest in ((pair_dir, pair),):
        for suffix, key in (
            ("script.py", "script_sha256"),
            ("plan.md", "plan_sha256"),
            ("helper.py", "helper_sha256"),
            ("source.zip", "analysis_source_archive_sha256"),
        ):
            bind(directory / f"summary.{suffix}", manifest[key])
    bind(pair_dir / "independent-audit.py", audit["script_sha256"])
    bind(diagnosis_dir / "diagnose.py", diagnosis["script_sha256"])
    bind(diagnosis_dir / "independent-audit.py", diagnosis_audit["script_sha256"])
    bind(DIRECT_DIR / "summary.script.py", summary["script_sha256"])
    bind(DIRECT_DIR / "summary.plan.md", DIRECT_PLAN_SHA)
    require(summary["plan_sha256"] == DIRECT_PLAN_SHA, "undeclared plan change")
    for name, digest in summary["input_sha256"].items():
        bind(ROOT / name, digest)
    # Validate every historical input using its frozen archive when live source/config changed.
    historical = summary["historical_archived_inputs"]
    archive_cache = {}
    for record in historical:
        archived = (ROOT / record["archive"]).resolve()
        require(str(archived) in INPUTS, "unbound historical archive")
        if str(archived) not in archive_cache:
            archive_cache[str(archived)] = members(archived)
        relative = (ROOT / record["path"]).resolve().relative_to(ROOT).as_posix()
        require(record["member"] == relative, "historical member identity differs")
        payload = archive_cache[str(archived)][relative]
        if record["transformation"] != "none":
            require(
                relative == "configs/eval/ranking.yaml"
                and record["transformation"] == "remove symmetric_set_reranking: false",
                "unrecognized historical transformation",
            )
            payload = remove_default(payload, "symmetric_set_reranking")
        require(
            hashlib.sha256(payload).hexdigest() == record["sha256"],
            "historical member bytes differ",
        )
    for manifest in (pair, audit, diagnosis, diagnosis_audit):
        hashes = {**manifest.get("input_sha256", {}), **manifest.get("validated_input_sha256", {})}
        for name, digest in hashes.items():
            path = (ROOT / name).resolve()
            if path.exists() and sha(path) == digest:
                bind(path, digest)
                continue
            candidates = [
                r
                for r in historical
                if (ROOT / r["path"]).resolve() == path and r["sha256"] == digest
            ]
            require(candidates, f"missing historical bridge: {path}")
            relative = path.relative_to(ROOT).as_posix()
            matched = False
            for record in candidates:
                archived = (ROOT / record["archive"]).resolve()
                require(str(archived) in INPUTS, "unbound historical archive")
                contents = archive_cache.setdefault(str(archived), members(archived))
                require(relative in contents, "historical archive member absent")
                payload = contents[relative]
                if (
                    hashlib.sha256(payload).hexdigest() != digest
                    and relative == "configs/eval/ranking.yaml"
                ):
                    for field in ("pair_set_composition", "symmetric_set_reranking"):
                        if any(
                            line.startswith(field.encode() + b":") for line in payload.splitlines()
                        ):
                            payload = remove_default(payload, field)
                            if hashlib.sha256(payload).hexdigest() == digest:
                                break
                if hashlib.sha256(payload).hexdigest() == digest:
                    matched = True
                    break
            require(matched, "historical archive bytes mismatch")
    # Three reference reports are authenticated before any prediction is derived.
    references = [
        read(pair_dir / f"seed-{seed}.json", pair["report_sha256"][f"seed-{seed}.json"])
        for seed in SEEDS
    ]
    vocab_paths = {Path(p).resolve() for p in INPUTS if Path(p).name == "vocabulary.json"}
    require(len(vocab_paths) == 1, "ambiguous vocabulary")
    tokens = read(next(iter(vocab_paths)))["tokens"]
    require(
        len(tokens) == 2049 and all(t.startswith("PKM_") for t in tokens[1024:]),
        "product vocabulary",
    )
    return pair, references, tokens


def groups_from_graph():
    paths = {Path(p) for p in INPUTS if p.endswith(".snapshot.db")}
    require(len(paths) == 1, "ambiguous diagnostic graph")
    graph = next(iter(paths))
    with sqlite3.connect(graph.as_uri() + "?mode=ro", uri=True) as database:
        counts = dict(
            database.execute(
                "SELECT n.key,COUNT(DISTINCT e.dst) FROM nodes n JOIN edges e ON e.src=n.id "
                "JOIN relation_types r ON r.id=e.rel WHERE n.kind='product' "
                "AND r.name='HAS_TYPE' GROUP BY n.key"
            )
        )
    require(all(n in (1, 2) for n in counts.values()), "unexpected TYPE cardinality")
    return counts


def set_metric(ids, expected):
    found, truth = set(ids), set(expected)
    correct = len(found.intersection(truth))
    return {
        "tp": correct,
        "fp": len(found.difference(truth)),
        "fn": len(truth.difference(found)),
        "precision": correct / len(found) if found else float(not truth),
        "recall": correct / len(truth) if truth else 1.0,
        "f1": 2 * correct / (len(found) + len(truth)) if found or truth else 1.0,
        "exact": found == truth,
        "set_size": len(found),
    }


def examine(columns, values, subject, truth_ids):
    """Pure synthetic-testable mathematics, independent of all evaluator functions."""
    require(isinstance(columns, list) and isinstance(values, list), "column/value lists required")
    require(len(columns) == len(values) and len(set(columns)) == len(columns), "head alignment")
    require(all(type(i) is int and i >= 1024 for i in columns), "invalid product column")
    require(columns == sorted(columns), "noncanonical product column order")
    require(type(subject) is int and subject in columns, "subject outside vocabulary")
    require(
        isinstance(truth_ids, list) and len(set(truth_ids)) == len(truth_ids), "duplicate truth"
    )
    require(all(type(i) is int for i in truth_ids), "noninteger truth")
    for value in values:
        require(type(value) in (float, int) and math.isfinite(value), "nonfinite/numeric score")
        require(struct.unpack("!f", struct.pack("!f", value))[0] == value, "non-FP32 score")
    z = {i: v for i, v in zip(columns, values, strict=True) if i != subject}
    truth = set(truth_ids)
    require(truth <= z.keys(), "truth outside eligible products or contains subject")
    negatives = set(z).difference(truth)
    positive_min = min((z[i] for i in truth), default=None)
    negative_max = max((z[i] for i in negatives), default=None)
    smallest_true = sorted(i for i in truth if z[i] == positive_min)
    largest_false = sorted(i for i in negatives if z[i] == negative_max)
    gap = None if positive_min is None or negative_max is None else positive_min - negative_max
    separable = gap is None or gap > 0
    order = sorted(z, key=lambda i: (-z[i], i))
    k = len(truth)
    top = order[:k]
    cut_tied = 0 < k < len(order) and z[order[k - 1]] == z[order[k]]
    boundary_score = z[order[k - 1]] if 0 < k < len(order) else None
    at_boundary = [i for i in order if z[i] == boundary_score] if cut_tied else []
    direct = sorted(i for i, value in z.items() if value > 0)
    top_metrics = set_metric(top, truth)
    require(not separable or top_metrics["exact"], "strict separation implies top-K exact")
    return {
        "allowed_ids": sorted(z),
        "truth_ids": sorted(truth),
        "negative_ids": sorted(negatives),
        "positive_min": positive_min,
        "negative_max": negative_max,
        "gap": gap,
        "smallest_true_ids": smallest_true,
        "largest_false_ids": largest_false,
        "separable": separable,
        "gap_zero": gap == 0 if gap is not None else False,
        "overlap": gap < 0 if gap is not None else False,
        "ranked_ids": order,
        "top_ids_ranked": top,
        "top_ids": sorted(top),
        "top_metrics": top_metrics,
        "direct_ids": direct,
        "direct_metrics": set_metric(direct, truth),
        "boundary_tied": cut_tied,
        "boundary_score": boundary_score,
        "boundary_ids_ranked": at_boundary,
        "boundary_selected_ids": sorted(set(at_boundary).intersection(top)),
        "boundary_excluded_ids": sorted(set(at_boundary).difference(top)),
        "separable_wrong_zero": separable and not set_metric(direct, truth)["exact"],
    }


def exact(actual, expected, context):
    """No numeric tolerance: both sides use saved FP32 values and canonical fsum."""
    require(
        type(actual) is type(expected)
        or (type(actual) in (float, int) and type(expected) in (float, int)),
        context + " type",
    )
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), context + " keys")
        for key in expected:
            exact(actual[key], expected[key], context + "." + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), context + " length")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            exact(left, right, f"{context}[{index}]")
    else:
        require(actual == expected, context + " differs")


def raw_material(reference, tokens):
    """Read authenticated original B8 head columns, not any serial replay outputs."""
    lookup = {token: i for i, token in enumerate(tokens)}
    subject = lookup[reference["subject"]]
    truth = sorted(lookup[token] for token in reference["expected"])
    core = examine(list(range(1024, 2049)), reference["symmetric_relation_logits"], subject, truth)
    require(len(core["allowed_ids"]) == 1024, "eligible universe size")
    selection = reference["selection"]
    baseline = selection["selected_set_ids"]
    require(
        selection["selected_source_eligible"] is True
        and selection["fallback_no_valid_source"] is False,
        "baseline failed source",
    )
    require(
        baseline == sorted(set(baseline)) and set(baseline) <= set(core["allowed_ids"]),
        "baseline canonical membership",
    )
    require(reference["selected_set_keys"] == [tokens[i] for i in baseline], "baseline token keys")
    baseline_metrics = set_metric(baseline, truth)
    require(
        all(reference["selected_set_metrics"][k] == v for k, v in baseline_metrics.items()),
        "baseline raw membership/metrics",
    )
    union = set()
    require(len(reference["source_paths"]) == 4, "source count")
    for source in reference["source_paths"]:
        require(
            source["eligible"] and source["terminated"] and source["protocol_valid"],
            "invalid source",
        )
        ids = source["token_ids"]
        prompt = [
            lookup[t]
            for t in ("BOS", reference["subject"], reference["dimension"], "SAME", "ANSWER")
        ]
        require(ids[:5] == prompt and ids[-1] == lookup["EOS"], "source protocol")
        products = ids[5:-1]
        require(
            products
            and len(products) == len(set(products))
            and all(type(i) is int and 1024 <= i < 2049 for i in products),
            "source products",
        )
        processed = [i for i in products if i != subject]
        require(source["processed_targets"] == [tokens[i] for i in processed], "source postprocess")
        union.update(processed)
    return core, baseline_metrics, sorted(set(truth) - union)


def preflight(summary, summary_path):
    """Authenticate inputs only; callers explicitly invoke real measurement afterward."""
    direct = read(DIRECT_DIR / "summary.json", DIRECT_SHA)
    direct_audit = read(DIRECT_DIR / "independent-audit.json", DIRECT_AUDIT_SHA)
    require(
        direct_audit["summary_sha256"] == DIRECT_SHA
        and direct_audit["all_recomputed_outputs_equal"] is True,
        "direct audit binding",
    )
    bind(DIRECT_DIR / "independent-audit.py", direct_audit["script_sha256"])
    bind(DIRECT_DIR / "summary.script.py", HELPER_SHA)
    bind(DIRECT_DIR / "summary.plan.md", DIRECT_PLAN_SHA)
    pair, references, tokens = authenticate_direct(direct)
    require(len(tokens) == 2049 and len(set(tokens)) == 2049, "vocabulary uniqueness")
    direct_reports = [
        read(DIRECT_DIR / f"seed-{seed}.json", direct["report_sha256"][f"seed-{seed}.json"])
        for seed in SEEDS
    ]
    for path, digest in summary["input_sha256"].items():
        bind(ROOT / path, digest)
    require(summary["plan_sha256"] == PLAN_SHA, "declaration changed")
    bind(summary_path.with_suffix(".plan.md"), PLAN_SHA)
    bind(ROOT / "docs/experiments/2026-09-25-membership-separability-plan.md", PLAN_SHA)
    bind(summary_path.with_suffix(".script.py"), summary["snapshot_sha256"]["script.py"])
    bind(summary_path.with_suffix(".helper.py"), HELPER_SHA)
    bind(
        ROOT / "scripts/diagnose_membership_separability.py",
        summary["snapshot_sha256"]["script.py"],
    )
    exact(
        summary["historical_archived_inputs"],
        direct["historical_archived_inputs"],
        "historical archive bridge",
    )
    return pair, references, tokens, direct, direct_reports


CROSS_FIELDS = (
    "pair_exact",
    "direct_exact",
    "strict_separable",
    "separable_with_wrong_zero_threshold",
    "boundary_tie",
    "strict_rank_overlap",
    "oracle_top_k_exact",
    "source_union_lacks_truth",
)
FLAGS = (*CROSS_FIELDS, "oracle_top_k_gained_exact", "oracle_top_k_lost_exact")
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
CASES = ("mixed", "empty_truth", "full_truth", "empty_universe")


def geometry(core, columns, values):
    allowed = set(core["allowed_ids"])
    truth = set(core["truth_ids"])
    if not allowed:
        case, semantics = "empty_universe", "any finite t"
    elif not truth:
        case, semantics = "empty_truth", "t >= lower; upper unbounded"
    elif truth == allowed:
        case, semantics = "full_truth", "lower unbounded; t < upper"
    else:
        case = "mixed"
        semantics = "lower <= t < upper" if core["separable"] else "empty interval"
    interval = {
        "exists": core["separable"],
        "lower": core["negative_max"],
        "upper": core["positive_min"],
        "lower_inclusive": True if core["negative_max"] is not None else None,
        "upper_inclusive": False if core["positive_min"] is not None else None,
        "semantics": semantics,
    }
    separation = {
        "case": case,
        "strict_separable": core["separable"],
        "min_true_logit": core["positive_min"],
        "max_negative_logit": core["negative_max"],
        "gap": core["gap"],
        "min_true_ids": core["smallest_true_ids"],
        "max_negative_ids": core["largest_false_ids"],
        "threshold_interval": interval,
    }
    scores = dict(zip(columns, values, strict=True))
    k = len(truth)
    ranked = core["ranked_ids"]
    selected = core["top_ids_ranked"]
    tied = sorted(core["boundary_ids_ranked"])
    top = {
        "k": k,
        "selected_ranked_ids": selected,
        "selected_set_ids": core["top_ids"],
        "metrics": core["top_metrics"],
        "boundary": {
            "has_cut": 0 < k < len(ranked),
            "selected_min_logit": scores[selected[-1]] if selected else None,
            "excluded_max_logit": scores[ranked[k]] if k < len(ranked) else None,
            "tie": core["boundary_tied"],
            "tied_product_ids": tied,
            "tied_true_ids": sorted(truth.intersection(tied)),
            "tied_negative_ids": sorted(set(tied).difference(truth)),
            "selected_tied_ids": core["boundary_selected_ids"],
            "excluded_tied_ids": core["boundary_excluded_ids"],
        },
    }
    return separation, top


def row_result(seed, index, reference, tokens, saved_direct):
    core, pair_metrics, missing = raw_material(reference, tokens)
    separation, top = geometry(
        core, list(range(1024, 2049)), reference["symmetric_relation_logits"]
    )
    baseline = reference["selection"]["selected_set_ids"]
    direct_metrics = core["direct_metrics"]
    # Check accepted zero-threshold predictions independently before the new diagnosis.
    for key, expected in {
        "subject": reference["subject"],
        "subject_id": tokens.index(reference["subject"]),
        "dimension": reference["dimension"],
        "group": reference["group"],
        "expected_set_ids": core["truth_ids"],
        "baseline_set_ids": baseline,
        "baseline_selection": reference["selection"],
        "predicted_set_ids": core["direct_ids"],
        "baseline_metrics": pair_metrics,
        "direct_metrics": direct_metrics,
        "source_union_missing_true_ids": missing,
        "gained_exact": direct_metrics["exact"] and not pair_metrics["exact"],
        "lost_exact": pair_metrics["exact"] and not direct_metrics["exact"],
    }.items():
        exact(saved_direct[key], expected, f"saved direct.{key}")
    pair_exact, direct_exact = pair_metrics["exact"], direct_metrics["exact"]
    top_exact = top["metrics"]["exact"]
    return {
        "seed": seed,
        "index": index,
        "subject": reference["subject"],
        "subject_id": tokens.index(reference["subject"]),
        "dimension": reference["dimension"],
        "group": reference["group"],
        "expected_set_ids": core["truth_ids"],
        "baseline_set_ids": baseline,
        "source_union_missing_true_ids": missing,
        "allowed_product_count": 1024,
        "direct_set_ids": core["direct_ids"],
        "baseline_selection": {
            k: reference["selection"][k]
            for k in ("selected_slot", "selected_kind", "selected_source_ranks")
        },
        "pair_metrics": pair_metrics,
        "direct_metrics": direct_metrics,
        "separation": separation,
        "oracle_top_k": top,
        "pair_exact": pair_exact,
        "direct_exact": direct_exact,
        "strict_separable": core["separable"],
        "separable_with_wrong_zero_threshold": core["separable_wrong_zero"],
        "boundary_tie": core["gap_zero"],
        "strict_rank_overlap": core["overlap"],
        "oracle_top_k_exact": top_exact,
        "oracle_top_k_gained_exact": top_exact and not pair_exact,
        "oracle_top_k_lost_exact": pair_exact and not top_exact,
        "source_union_lacks_truth": bool(missing),
    }


def aggregate(rows):
    result = {"query_count": len(rows)}
    for label in ("pair_metrics", "direct_metrics", "oracle_top_k_metrics"):
        scores = [
            r["oracle_top_k"]["metrics"] if label == "oracle_top_k_metrics" else r[label]
            for r in rows
        ]
        result[label] = {
            "exact_count": sum(int(s["exact"]) for s in scores),
            **{key: sum(s[key] for s in scores) for key in ("tp", "fp", "fn")},
            **{
                key: math.fsum(s[key] for s in scores) / len(scores) if scores else None
                for key in ("precision", "recall", "f1", "set_size")
            },
        }
    result["counts"] = {key: sum(int(row[key]) for row in rows) for key in FLAGS}
    result["separation_cases"] = {
        label: sum(r["separation"]["case"] == label for r in rows) for label in CASES
    }
    combinations = {}
    for row in rows:
        key = tuple(row[field] for field in CROSS_FIELDS)
        combinations[key] = combinations.get(key, 0) + 1
    result["cross_tab"] = [
        {**dict(zip(CROSS_FIELDS, key, strict=True)), "query_count": combinations[key]}
        for key in sorted(combinations)
    ]
    return result


def summarize(rows):
    failed = [row for row in rows if not row["pair_exact"]]
    return {
        "totals": aggregate(rows),
        "groups": {group: aggregate([r for r in rows if r["group"] == group]) for group in GROUPS},
        "pair_failures": {
            "totals": aggregate(failed),
            "groups": {g: aggregate([r for r in failed if r["group"] == g]) for g in GROUPS},
        },
    }


def live_inventory(expected):
    paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")}
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").rglob("*.yaml"))
    paths.update(("pyproject.toml", "uv.lock"))
    require(set(expected) == paths, "live source/config inventory changed")
    for name, digest in expected.items():
        bind(ROOT / name, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    path = args.summary.resolve()
    output = path.parent / "independent-audit.json"
    require(not output.exists(), "refusing existing independent audit")
    require(path.is_file(), "primary summary missing; no measurement or audit output")
    require("torch" not in sys.modules, "independent audit must remain stdlib-only")
    script_path = Path(__file__).resolve()
    script_digest = sha(script_path)
    summary = read(path)
    require(
        summary["diagnostic_version"] == "plm-membership-separability-v1"
        and summary["diagnostic_complete"] is True
        and summary["diagnostic_accepted"] is False
        and summary["independent_audit_status"] == "required",
        "incomplete/self-accepted diagnosis",
    )
    require(
        summary["query_count"] == 666
        and summary["distinct_validation_queries"] == 222
        and summary["training_seed_replicas"] == 3,
        "observation/replica counts",
    )
    exact(summary["product_token_ids"], list(range(1024, 2049)), "product columns")
    exact(
        summary["saved_logit_batch_shape"],
        {"batch_size": 8, "final_batch_size": 6},
        "B8 input shape",
    )
    exact(summary["cross_tab_fields"], list(CROSS_FIELDS), "declared cross-tab fields")
    for key, digest in {
        "plan_sha256": PLAN_SHA,
        "helper_sha256": HELPER_SHA,
        "pair_summary_sha256": PAIR_SHA,
        "pair_audit_sha256": PAIR_AUDIT_SHA,
        "direct_summary_sha256": DIRECT_SHA,
        "direct_audit_sha256": DIRECT_AUDIT_SHA,
    }.items():
        require(summary[key] == digest, "frozen identity: " + key)
    require(
        set(summary["snapshot_sha256"]) == {"script.py", "plan.md", "helper.py", "helper-plan.md"},
        "snapshot inventory",
    )
    for suffix, digest in summary["snapshot_sha256"].items():
        bind(path.with_suffix("." + suffix), digest)
    require(
        summary["snapshot_sha256"]["plan.md"] == PLAN_SHA
        and summary["snapshot_sha256"]["helper.py"] == HELPER_SHA
        and summary["snapshot_sha256"]["helper-plan.md"] == DIRECT_PLAN_SHA,
        "snapshot pins",
    )
    require(set(summary["report_sha256"]) == {f"seed-{s}.json" for s in SEEDS}, "report inventory")
    for name, digest in summary["report_sha256"].items():
        bind(path.parent / name, digest)
    require([r["seed"] for r in summary["runs"]] == list(SEEDS), "seed summary inventory")
    live_inventory(summary["live_source_config_sha256"])
    pair, references, tokens, direct_summary, direct_reports = preflight(summary, path)
    exact(
        summary["historical_model_environment"],
        pair["historical_model_environment"],
        "model origin",
    )
    type_counts = groups_from_graph()
    all_rows, partition, results = [], None, []
    for seed, reference, saved_direct, top in zip(
        SEEDS, references, direct_reports, summary["runs"], strict=True
    ):
        report = read(
            path.parent / f"seed-{seed}.json", summary["report_sha256"][f"seed-{seed}.json"]
        )
        require(
            report["seed"] == reference["seed"] == saved_direct["seed"] == seed, "seed identity"
        )
        require(
            report["query_count"]
            == saved_direct["query_count"]
            == len(report["responses"])
            == len(reference["responses"])
            == len(saved_direct["responses"])
            == 222,
            "seed coverage",
        )
        exact({k: v for k, v in report.items() if k != "responses"}, top, "summary/full report")
        for key in ("checkpoint_hash", "training_identity", "corpus_identity", "split_hash"):
            exact(report[key], reference[key], "pair identity." + key)
            exact(report[key], saved_direct[key], "direct identity." + key)
        rows, identities = [], []
        for index, (ref, direct, observed) in enumerate(
            zip(reference["responses"], saved_direct["responses"], report["responses"], strict=True)
        ):
            row = row_result(seed, index, ref, tokens, direct)
            expected_group = (
                "COLOR"
                if row["dimension"] == "COLOR"
                else "TYPE_single"
                if type_counts[row["subject"]] == 1
                else "TYPE_dual"
            )
            require(row["group"] == expected_group, "graph-derived group")
            exact(observed, row, f"seed {seed} row {index}")
            rows.append(row)
            identities.append(
                (row["subject"], row["dimension"], row["group"], row["expected_set_ids"])
            )
        require(len({(x[0], x[1]) for x in identities}) == 222, "duplicate query")
        if partition is None:
            partition = identities
        require(identities == partition, "shared ordered validation partition changed")
        reconstructed = summarize(rows)
        for key, value in reconstructed.items():
            exact(report[key], value, f"seed {seed} {key}")
        results.append({"seed": seed, "query_count": 222, **reconstructed})
        all_rows.extend(rows)
        print(f"seed {seed}: 222 independent row calculations and all aggregates agree", flush=True)
    reconstructed = summarize(all_rows)
    for key, value in reconstructed.items():
        exact(summary[key], value, "pooled " + key)
    baseline = {
        "query_count": len(all_rows),
        "pair_exact": sum(r["pair_exact"] for r in all_rows),
        "direct_exact": sum(r["direct_exact"] for r in all_rows),
        "direct_gains": sum(r["direct_exact"] and not r["pair_exact"] for r in all_rows),
        "direct_losses": sum(r["pair_exact"] and not r["direct_exact"] for r in all_rows),
    }
    exact(
        baseline,
        {
            "query_count": 666,
            "pair_exact": 569,
            "direct_exact": 339,
            "direct_gains": 3,
            "direct_losses": 233,
        },
        "accepted baseline reconstruction",
    )
    exact(summary["baseline_reconstruction"], baseline, "recorded baseline reconstruction")
    for section, count in (("baseline", 569), ("direct", 339)):
        require(
            direct_summary["comparison"][section]["exact_count"] == count, "direct baseline receipt"
        )
    require(
        summary["analysis_environment"]["torch_imported"] is False
        and summary["analysis_environment"]["model_execution"] is False
        and "torch" not in sys.modules,
        "neural/Torch execution",
    )
    require(
        summary["empty_metric_convention"]
        == "empty/empty precision=recall=F1=1; empty truth recall=1; "
        "empty prediction precision=0 when truth nonempty",
        "metric convention",
    )
    live_inventory(summary["live_source_config_sha256"])
    require(
        all(sha(name) == digest for name, digest in INPUTS.items()), "input changed during audit"
    )
    require(sha(script_path) == script_digest, "auditor changed during execution")
    result = {
        "complete": True,
        "all_recomputed_outputs_equal": True,
        "diagnostic_accepted": False,
        "acceptance_authority": "campaign owner after reviewing this independent receipt",
        "summary_sha256": sha(path),
        "script_sha256": script_digest,
        "plan_sha256": PLAN_SHA,
        "query_count": 666,
        "distinct_validation_queries": 222,
        "training_seed_replicas": 3,
        "baseline_reconstruction": baseline,
        **reconstructed,
        "runs": results,
        "input_sha256": dict(sorted(INPUTS.items())),
        "limitations": [
            "Authenticated saved batch-eight FP32 logits; no independent neural regeneration.",
            "Expected labels and cardinalities are privileged diagnostic information.",
            "Exact-answer attainability applies only to the declared rule families; "
            "top-K precision/recall/F1 are not threshold-family upper bounds.",
            "Three training seeds repeat the same 222 validation queries.",
            "No threshold fitting, alternate policy selection, protected test, "
            "or model/application promotion.",
        ],
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "diagnostic_accepted": False,
                "summary_sha256": sha(path),
                "audit_sha256": sha(output),
            }
        )
    )


if __name__ == "__main__":
    main()
