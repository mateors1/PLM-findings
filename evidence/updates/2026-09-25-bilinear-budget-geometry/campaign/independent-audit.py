"""Independent paired saved-score audit: no primary arithmetic or neural execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PLAN = "9dc2186cb9cf19ee6972c78d136e0680f05ca4ce4f85fc7a8f407217500477e2"
RUNNER = "28a3e5f8491d1f3a903e3026d94d70643bfdcf4dd04b2033673c80f120d7f92d"
TEST_RECEIPT = "2563e38baa69414d3736782afbc6c24f2696aed1623eb66335fa488bf3c5b88c"
OLD_AUDITOR = "6a7321f0ec0d0ddb644f04b617667e40b349170ab34dc6822a8ebbde45ac0cad"
PRIMARY_HELPER = "66093bfbd06c8622b61535eeaffd0b09886c0cee05ec9eae71292a6e85925789"
PINS = {
    2000: (
        "bilinear-seed-replication-v1",
        "338fb6b7ea08128952ce9abc02b3506639e712a9cff6ce63468ed2e7980f79a0",
        "ddbecaceabdacc0808f9b4f122e0309aa3e233fe76b0d75694e31833a40083a8",
        "62a8dcbb7cbe5d038984f9d57aecacec4b0ba612ba55c2a9f6fe335977304978",
        "independent-audit.json",
    ),
    8000: (
        "bilinear-budget8000-replication-v1",
        "77cc2ee2ebf4dc1f8671e71833712302160190c85477eb6e0dd0b5494bb3cde6",
        "49fc5d7edc7763870ac0e9310295385d8e6a3af6e67d3cc6b5a9f0e44bc2ae77",
        "01a9faefb90ee3d3e7525c0d5191d41741131843dc79ea2b44d860c5c8e21d5e",
        "independent-audit-v2.json",
    ),
}
GEOMETRY_PINS = (
    "1a38b93a27a77eb597637798be4f8b24ef858c546d6a7c25aed8ba8f85a99606",
    "1b10b77db77ebb79b5f1327bcc1538331eb876216c424036a516a0ac04cdba96",
    "4f2d624341cf03d4af8e7c6976135a1b6766071899e745ea4071f9f1cf04e652",
)
SEEDS = (1729, 1730, 1731)
BUDGETS = (2000, 8000)
GROUPS = ("COLOR", "TYPE_dual", "TYPE_single")
CATEGORIES = ("exact_zero", "separated_missing", "separated_extra", "overlap_or_tie")
IDENTITY = ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_sha256")
MEASUREMENTS = ("a", "b", "g", "category", "exact", "strict_separation")


def require(condition, message):
    if not condition:
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
    require(str(path) not in inputs or inputs[str(path)] == digest, "conflicting input")
    inputs[str(path)] = digest
    return read(path) if path.suffix == ".json" else path.read_bytes()


def helper(root, inputs):
    path = root / "runs/learning/bilinear-error-geometry-v1/independent-audit.py"
    bound(path, OLD_AUDITOR, inputs)
    spec = importlib.util.spec_from_file_location("independent_geometry_frozen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accepted(summary, audit, decision, summary_hash, audit_hash):
    require(summary["complete"] is True, "completed upstream summary")
    require(audit["complete"] is True and audit["audit_passed"] is True, "completed upstream audit")
    same(audit["summary_sha256"], summary_hash, "upstream audit binding")
    same(decision["summary_sha256"], summary_hash, "upstream decision summary")
    same(decision["audit_sha256"], audit_hash, "upstream decision audit")
    require(decision["evidence_accepted"] is True, "upstream evidence acceptance")


def child_binding(summary, audit, child, digest):
    require(summary["complete"] is True and audit["complete"] is True, "child completion")
    require(audit["audit_passed"] is True, "child audited")
    same(summary["artifact_sha256"]["child.json"], digest, "saved child identity")
    same(
        summary["artifact_sha256"]["checkpoint-final.pt"],
        audit["checkpoint"]["child_checkpoint_sha256"],
        "checkpoint JSON identity",
    )
    same(child["aggregate"], audit["child"]["aggregate"], "audited child aggregate")
    same(child["groups"], audit["child"]["groups"], "audited child groups")
    same(child["aggregate"], summary["child"]["aggregate"], "summarized child aggregate")
    same(child["groups"], summary["child"]["groups"], "summarized child groups")


def authenticate_budget(root, budget, inputs):
    name, summary_hash, audit_hash, decision_hash, audit_name = PINS[budget]
    folder = root / "runs/learning" / name
    summary = bound(folder / "aggregate-v2.json", summary_hash, inputs)
    audit = bound(folder / audit_name, audit_hash, inputs)
    decision = bound(folder / "decision.json", decision_hash, inputs)
    accepted(summary, audit, decision, summary_hash, audit_hash)
    history = summary["historical1729"]
    old = root / "runs/learning" / f"bilinear-budget{budget}-v1"
    hs, ha, hd = [
        bound(old / filename, history[key], inputs)
        for filename, key in (
            ("summary.json", "summary_sha256"),
            ("independent-audit.json", "audit_sha256"),
            ("decision.json", "decision_sha256"),
        )
    ]
    accepted(hs, ha, hd, history["summary_sha256"], history["audit_sha256"])
    digests = {1729: history["child_sha256"]}
    children = {1729: bound(old / "child.json", digests[1729], inputs)}
    child_binding(hs, ha, children[1729], digests[1729])
    same(
        ha["input_sha256"][str((old / "child.json").resolve())],
        digests[1729],
        "historical audit child",
    )
    summaries = {1729: hs}
    parents = {1729: ha["checkpoint"]["parent_checkpoint_sha256"]}
    same([e["seed"] for e in summary["reports"]], [1730, 1731], "fresh coverage")
    same([e["seed"] for e in audit["per_seed"]], [1730, 1731], "audit coverage")
    for entry, proof in zip(summary["reports"], audit["per_seed"], strict=True):
        seed = entry["seed"]
        own = folder / f"seed-{seed}"
        require(Path(entry["path"]).resolve() == (own / "summary.json").resolve(), "seed path")
        ss = bound(own / "summary.json", entry["sha256"], inputs)
        aa = bound(own / audit_name, proof["audit_sha256"], inputs)
        same(ss["seed"], seed, "summary seed")
        same(aa["seed"], seed, "audit seed")
        same(aa["summary_sha256"], entry["sha256"], "seed audit summary")
        same(proof["summary_sha256"], entry["sha256"], "aggregate seed summary")
        digests[seed] = ss["artifact_sha256"]["child.json"]
        same(
            aa["input_sha256"][str((own / "child.json").resolve())],
            digests[seed],
            "child audit hash",
        )
        children[seed] = bound(own / "child.json", digests[seed], inputs)
        child_binding(ss, aa, children[seed], digests[seed])
        same(proof["checkpoint"], aa["checkpoint"], "aggregate checkpoint binding")
        summaries[seed] = ss
        parents[seed] = aa["checkpoint"]["parent_checkpoint_sha256"]
    return children, digests, summaries, parents


def authenticate_history(root, inputs):
    folder = root / "runs/learning/bilinear-error-geometry-v1"
    summary, audit, decision = [
        bound(folder / name, digest, inputs)
        for name, digest in zip(
            ("summary.json", "independent-audit.json", "decision.json"), GEOMETRY_PINS, strict=True
        )
    ]
    accepted(summary, audit, decision, GEOMETRY_PINS[0], GEOMETRY_PINS[1])
    same(
        [e["seed"] for e in summary["seed_artifacts"]],
        list(SEEDS),
        "historical reductions coverage",
    )
    reports = {}
    for entry in summary["seed_artifacts"]:
        seed = entry["seed"]
        path = folder / f"seed-{seed}.json"
        require(Path(entry["path"]).resolve() == path.resolve(), "historical report path")
        reports[seed] = bound(path, entry["sha256"], inputs)
    return summary, reports


def pair_row(before, after):
    same({k: before[k] for k in IDENTITY}, {k: after[k] for k in IDENTITY}, "paired query identity")
    return {
        **{k: before[k] for k in IDENTITY},
        "budget2000": {k: before[k] for k in MEASUREMENTS},
        "budget8000": {k: after[k] for k in MEASUREMENTS},
        "delta": {k: after[k] - before[k] for k in ("a", "b", "g")},
    }


def transitions(rows):
    require(rows, "nonempty paired aggregation")
    matrix = [[0 for _ in CATEGORIES] for _ in CATEGORIES]
    exact_gains = exact_losses = separation_gains = separation_losses = 0
    for row in rows:
        old, new = row["budget2000"], row["budget8000"]
        matrix[CATEGORIES.index(old["category"])][CATEGORIES.index(new["category"])] += 1
        exact_gains += int(new["exact"] and not old["exact"])
        exact_losses += int(old["exact"] and not new["exact"])
        separation_gains += int(new["strict_separation"] and not old["strict_separation"])
        separation_losses += int(old["strict_separation"] and not new["strict_separation"])
    return {
        "query_count": len(rows),
        "category_order": list(CATEGORIES),
        "matrix": matrix,
        "exact_gains": exact_gains,
        "exact_losses": exact_losses,
        "separation_gains": separation_gains,
        "separation_losses": separation_losses,
    }


def paired_report(seed, before, after):
    require(len(before["rows"]) == len(after["rows"]), "paired row coverage")
    rows = [pair_row(a, b) for a, b in zip(before["rows"], after["rows"], strict=True)]
    return {
        "complete": True,
        "seed": seed,
        "query_count": len(rows),
        "rows": rows,
        "aggregate": transitions(rows),
        "groups": {g: transitions([r for r in rows if r["group"] == g]) for g in GROUPS},
    }


def pooled_transitions(reports, seeds):
    rows = [r for seed in seeds for r in reports[seed]["rows"]]
    return {
        "aggregate": transitions(rows),
        "groups": {g: transitions([r for r in rows if r["group"] == g]) for g in GROUPS},
    }


def reconcile_pairs(report, accepted_pairing):
    same(report["aggregate"]["exact_gains"], accepted_pairing["gains"], "accepted gains")
    same(report["aggregate"]["exact_losses"], accepted_pairing["losses"], "accepted losses")
    for group in GROUPS:
        same(
            report["groups"][group]["exact_gains"],
            accepted_pairing["groups"][group]["gains"],
            "accepted group gains",
        )
        same(
            report["groups"][group]["exact_losses"],
            accepted_pairing["groups"][group]["losses"],
            "accepted group losses",
        )


def dependency_inventory(actual, expected):
    require(isinstance(actual, dict), "dependency mapping")
    require(set(actual) == set(expected), "exact dependency names")
    same(actual, expected, "dependency digests")


def preflight(root, h, inputs):
    plan = root / "docs/experiments/2026-09-25-bilinear-budget-geometry-plan.md"
    bound(plan, PLAN, inputs)
    bound(root / "scripts/diagnose_bilinear_errors.py", PRIMARY_HELPER, inputs)
    children, digests, summaries, parents = {}, {}, {}, {}
    for budget in BUDGETS:
        children[budget], digests[budget], summaries[budget], parents[budget] = authenticate_budget(
            root, budget, inputs
        )
    all_children = dict(children[2000])
    all_children.update({s + 10000: c for s, c in children[8000].items()})
    h.align(all_children)
    for seed in SEEDS:
        same(
            parents[2000][seed],
            parents[8000][seed],
            "same original parent",
        )
        old_folder = (
            root
            / "runs/learning"
            / (
                "bilinear-budget2000-v1"
                if seed == 1729
                else f"bilinear-seed-replication-v1/seed-{seed}"
            )
        )
        comparator_binding(
            summaries[8000][seed],
            inputs[str((old_folder / "summary.json").resolve())],
            digests[2000][seed],
        )
        for budget in BUDGETS:
            child = children[budget][seed]
            require(
                child["complete"] is True
                and child["query_count"] == 222
                and len(child["responses"]) == 222,
                "complete ordered child",
            )
            same(child["product_token_ids"], list(range(1024, 2049)), "full aligned columns")
    historical_summary, historical_reports = authenticate_history(root, inputs)
    for seed in SEEDS:
        same(
            historical_reports[seed]["child_sha256"],
            digests[2000][seed],
            "historical reduction child identity",
        )
    return children, digests, summaries, historical_summary, historical_reports


def comparator_binding(current, historical_summary_hash, historical_child_hash):
    same(
        current["historical2000_summary_sha256"],
        historical_summary_hash,
        "matched historical summary",
    )
    if "historical2000_child_sha256" in current:
        same(
            current["historical2000_child_sha256"],
            historical_child_hash,
            "matched historical child",
        )


def validate_test_receipt(root, receipt, inputs):
    require(
        type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0, "test process exit"
    )
    require(receipt.get("terminal_completion_observed") is True, "test observed completion")
    require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["diagnostic_executed"] is False,
        "synthetic CPU receipt",
    )
    require(type(receipt["test_count"]) is int and receipt["test_count"] > 0, "test count")
    require(type(receipt["skipped"]) is int and receipt["skipped"] == 0, "zero skipped tests")
    same(receipt["tested_script_sha256"], RUNNER, "tested runner")
    same(receipt["tested_helper_sha256"], PRIMARY_HELPER, "tested primary helper")
    test = root / "tests/unit/test_bilinear_budget_geometry.py"
    same(receipt["test_sha256"], sha(test), "tested test source")
    dependencies = {
        str((root / name).resolve()): sha(root / name)
        for name in (
            "scripts/diagnose_bilinear_budget_geometry.py",
            "scripts/diagnose_bilinear_errors.py",
            "tests/unit/test_bilinear_budget_geometry.py",
            "tests/unit/test_bilinear_error_geometry.py",
        )
    }
    dependency_inventory(receipt["test_dependencies"], dependencies)
    for path, digest in dependencies.items():
        bound(path, digest, inputs)
    stdout = Path(receipt["stdout"]["path"]).resolve()
    require(stdout.suffix == ".txt", "test stdout suffix")
    bound(stdout, receipt["stdout"]["sha256"], inputs)
    return stdout


def provenance(root, folder, summary, inputs):
    require(
        summary["complete"] is True and summary["acceptance"] is False,
        "complete unaccepted primary",
    )
    same(summary["campaign_version"], "bilinear-budget-geometry-v1", "campaign")
    same(summary["script_sha256"], RUNNER, "frozen primary")
    same(summary["helper_sha256"], PRIMARY_HELPER, "primary helper")
    same(summary["plan_sha256"], PLAN, "frozen plan")
    script = root / "scripts/diagnose_bilinear_budget_geometry.py"
    bound(script, RUNNER, inputs)
    receipt_path = root / "runs/learning/bilinear-budget-geometry-runner-tests-v1/receipt.json"
    receipt = bound(receipt_path, TEST_RECEIPT, inputs)
    stdout = validate_test_receipt(root, receipt, inputs)
    same(summary["input_sha256"], inputs, "exact primary input inventory")
    snapshots = {
        "script.py": script.read_bytes(),
        "helper.py": (root / "scripts/diagnose_bilinear_errors.py").read_bytes(),
        "plan.md": (
            root / "docs/experiments/2026-09-25-bilinear-budget-geometry-plan.md"
        ).read_bytes(),
        "test-source.py": (root / "tests/unit/test_bilinear_budget_geometry.py").read_bytes(),
        "test-receipt.json": receipt_path.read_bytes(),
        "test-stdout.txt": stdout.read_bytes(),
        "inputs.json": encoded(inputs),
    }
    hashes = {
        str(folder / ("summary." + suffix)): hashlib.sha256(data).hexdigest()
        for suffix, data in snapshots.items()
    }
    same(summary["snapshot_sha256"], hashes, "exact snapshot inventory")
    for path, digest in hashes.items():
        require(Path(path).stat().st_size <= 512 * 1024, "snapshot publication bound")
        bound(path, digest, inputs)


def audit(summary_path):
    root = Path(__file__).resolve().parents[3]
    folder = root / "runs/learning/bilinear-budget-geometry-v1"
    require(summary_path.resolve() == (folder / "summary.json").resolve(), "campaign location")
    independent_inputs = {}
    h = helper(root, independent_inputs)
    inputs = {}
    children, digests, sources, historical_summary, historical_reports = preflight(root, h, inputs)
    summary = read(summary_path)
    provenance(root, folder, summary, inputs)
    inputs.update(independent_inputs)
    summary_hash = sha(summary_path)
    bound(summary_path, summary_hash, inputs)
    terminal_path = folder / "execution-receipt.json"
    terminal = bound(terminal_path, sha(terminal_path), inputs)
    actual_stdout = folder / "primary-stdout.txt"
    bound(actual_stdout, terminal["stdout_sha256"], inputs)
    h.execution_receipt(terminal, summary_hash, sha(actual_stdout))
    same([e["seed"] for e in summary["per_seed"]], list(SEEDS), "output seed coverage")
    reports = {b: {} for b in BUDGETS}
    paired, per_seed = {}, []
    for entry in summary["per_seed"]:
        seed = entry["seed"]
        same(sorted(entry["budgets"]), ["2000", "8000"], "output budget coverage")
        rebuilt_entry = {"seed": seed, "budgets": {}}
        for budget in BUDGETS:
            report = h.seed_report(seed, children[budget][seed], digests[budget][seed])
            if budget == 2000:
                same(report, historical_reports[seed], "historical full reductions")
            reports[budget][seed] = report
            path = folder / f"seed-{seed}-budget-{budget}.json"
            saved_entry = entry["budgets"][str(budget)]
            require(Path(saved_entry["path"]).resolve() == path.resolve(), "budget output path")
            require(path.stat().st_size <= 512 * 1024, "budget report publication bound")
            saved = bound(path, saved_entry["sha256"], inputs)
            same(saved, report, "independent full budget report")
            rebuilt_entry["budgets"][str(budget)] = {
                "path": str(path),
                "sha256": saved_entry["sha256"],
                "checkpoint_sha256": sources[budget][seed]["artifact_sha256"][
                    "checkpoint-final.pt"
                ],
                "aggregate": report["aggregate"],
                "groups": report["groups"],
            }
        paired[seed] = paired_report(seed, reports[2000][seed], reports[8000][seed])
        reconcile_pairs(paired[seed], sources[8000][seed]["child"]["paired_vs_historical2000"])
        path = folder / f"seed-{seed}-paired.json"
        require(Path(entry["paired"]["path"]).resolve() == path.resolve(), "paired output path")
        require(path.stat().st_size <= 512 * 1024, "paired report publication bound")
        saved = bound(path, entry["paired"]["sha256"], inputs)
        same(saved, paired[seed], "independent full paired report")
        rebuilt_entry["paired"] = {
            "path": str(path),
            "sha256": entry["paired"]["sha256"],
            "aggregate": paired[seed]["aggregate"],
            "groups": paired[seed]["groups"],
        }
        same(entry, rebuilt_entry, "summarized per seed reductions")
        per_seed.append(rebuilt_entry)
    budgets = {
        str(b): {
            "fresh_two": h.pool(reports[b], (1730, 1731)),
            "all_three": h.pool(reports[b], SEEDS),
            "overlap": h.overlap(reports[b]),
        }
        for b in BUDGETS
    }
    for key in ("fresh_two", "all_three", "overlap"):
        same(budgets["2000"][key], historical_summary[key], "historical pooled reductions")
    transition_summary = {
        "fresh_two": pooled_transitions(paired, (1730, 1731)),
        "all_three": pooled_transitions(paired, SEEDS),
    }
    same(summary["budgets"], budgets, "budget aggregate reductions")
    same(summary["transitions"], transition_summary, "pooled transitions")
    same(
        summary["checks"],
        {
            "historical2000_exact": True,
            "paired_comparisons_exact": True,
            "query_alignment": True,
            "final_identity_check": True,
        },
        "reported complete checks",
    )
    require(all(sha(p) == digest for p, digest in inputs.items()), "final evidence identity")
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": summary_hash,
        "script_sha256": sha(__file__),
        "plan_sha256": PLAN,
        "primary_script_sha256": RUNNER,
        "independent_helper_sha256": OLD_AUDITOR,
        "input_sha256": inputs,
        "per_seed": per_seed,
        "budgets": budgets,
        "transitions": transition_summary,
        "limitations": [
            "Independent saved-score reductions; no independent model execution.",
            "444/666 observations repeat 222 validation questions; 1729 is development-selected.",
            "Intervals use labels but do not choose thresholds or generate alternative answers.",
            "Score deltas are descriptive within each parent, not confidence or causal effects.",
            "No checkpoint payload, training data, graph, Torch, GPU or protected examples opened.",
        ],
    }


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        root = Path(__file__).resolve().parents[3]
        inputs = {}
        h = helper(root, inputs)
        preflight(root, h, inputs)
        require("torch" not in sys.modules, "stdlib-only preflight")
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "input_count": len(inputs),
                    "reductions_executed": False,
                    "torch_imported": False,
                }
            )
        )
        return
    require(args.summary is not None, "summary path required")
    out = args.summary.parent / "independent-audit.json"
    require(not out.exists(), "immutable output already exists")
    result = audit(args.summary)
    require("torch" not in sys.modules, "stdlib-only auditor")
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
