"""Paired saved-score geometry across fixed budgets; stdlib and no new predictions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PLAN_SHA = "9dc2186cb9cf19ee6972c78d136e0680f05ca4ce4f85fc7a8f407217500477e2"
HELPER_SHA = "66093bfbd06c8622b61535eeaffd0b09886c0cee05ec9eae71292a6e85925789"
SEEDS = (1729, 1730, 1731)
BUDGETS = (2000, 8000)
CATEGORIES = ("exact_zero", "separated_missing", "separated_extra", "overlap_or_tie")
CHAIN_PINS = {
    2000: (
        "bilinear-seed-replication-v1",
        "independent-audit.json",
        "338fb6b7ea08128952ce9abc02b3506639e712a9cff6ce63468ed2e7980f79a0",
        "ddbecaceabdacc0808f9b4f122e0309aa3e233fe76b0d75694e31833a40083a8",
        "62a8dcbb7cbe5d038984f9d57aecacec4b0ba612ba55c2a9f6fe335977304978",
    ),
    8000: (
        "bilinear-budget8000-replication-v1",
        "independent-audit-v2.json",
        "77cc2ee2ebf4dc1f8671e71833712302160190c85477eb6e0dd0b5494bb3cde6",
        "49fc5d7edc7763870ac0e9310295385d8e6a3af6e67d3cc6b5a9f0e44bc2ae77",
        "01a9faefb90ee3d3e7525c0d5191d41741131843dc79ea2b44d860c5c8e21d5e",
    ),
}
GEOMETRY_PINS = (
    "1a38b93a27a77eb597637798be4f8b24ef858c546d6a7c25aed8ba8f85a99606",
    "1b10b77db77ebb79b5f1327bcc1538331eb876216c424036a516a0ac04cdba96",
    "4f2d624341cf03d4af8e7c6976135a1b6766071899e745ea4071f9f1cf04e652",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helper(root):
    path = root / "scripts/diagnose_bilinear_errors.py"
    require(sha(path) == HELPER_SHA, "frozen scientific helper")
    spec = importlib.util.spec_from_file_location("frozen_geometry_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accepted_chain(summary, audit, decision, summary_sha, audit_sha):
    require(
        summary["complete"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == summary_sha
        and decision["summary_sha256"] == summary_sha
        and decision["audit_sha256"] == audit_sha
        and decision["evidence_accepted"] is True,
        "accepted complete evidence chain",
    )


def child_contract(child):
    require(
        child["complete"] is True
        and child["query_count"] == 222
        and child["product_token_ids"] == list(range(1024, 2049))
        and len(child["responses"]) == 222
        and [r["index"] for r in child["responses"]] == list(range(222)),
        "ordered query and product-column contract",
    )


def bind_child(h, folder, summary, audit, inputs):
    digest = summary["artifact_sha256"]["child.json"]
    path = (folder / "child.json").resolve()
    checkpoint = summary["artifact_sha256"]["checkpoint-final.pt"]
    require(
        audit["input_sha256"][str(path)] == digest
        and audit["checkpoint"]["child_checkpoint_sha256"] == checkpoint,
        "audited child and checkpoint identity",
    )
    child = h._read_bound(path, digest, inputs)
    child_contract(child)
    require(
        child["aggregate"] == summary["child"]["aggregate"] == audit["child"]["aggregate"]
        and child["groups"] == summary["child"]["groups"] == audit["child"]["groups"],
        "accepted child metrics identity",
    )
    return {
        "child": child,
        "child_sha256": digest,
        "checkpoint_sha256": checkpoint,
        "summary": summary,
        "parent_checkpoint_sha256": audit["checkpoint"]["parent_checkpoint_sha256"],
    }


def budget_chain(root, budget, h, inputs):
    folder_name, audit_name, ss, aa, dd = CHAIN_PINS[budget]
    folder = root / "runs/learning" / folder_name
    summary = h._read_bound(folder / "aggregate-v2.json", ss, inputs)
    audit = h._read_bound(folder / audit_name, aa, inputs)
    decision = h._read_bound(folder / "decision.json", dd, inputs)
    accepted_chain(summary, audit, decision, ss, aa)
    historical = summary["historical1729"]
    old = root / f"runs/learning/bilinear-budget{budget}-v1"
    hs = h._read_bound(old / "summary.json", historical["summary_sha256"], inputs)
    ha = h._read_bound(old / "independent-audit.json", historical["audit_sha256"], inputs)
    hd = h._read_bound(old / "decision.json", historical["decision_sha256"], inputs)
    accepted_chain(hs, ha, hd, historical["summary_sha256"], historical["audit_sha256"])
    result = {1729: bind_child(h, old, hs, ha, inputs)}
    result[1729]["summary_sha256"] = historical["summary_sha256"]
    require(result[1729]["child_sha256"] == historical["child_sha256"], "historical child binding")
    require([r["seed"] for r in summary["reports"]] == [1730, 1731], "ordered fresh summaries")
    require([r["seed"] for r in audit["per_seed"]] == [1730, 1731], "ordered fresh audits")
    for entry, accepted in zip(summary["reports"], audit["per_seed"], strict=True):
        seed = entry["seed"]
        own = folder / f"seed-{seed}"
        require(
            Path(entry["path"]).resolve() == (own / "summary.json").resolve(), "seed summary path"
        )
        report = h._read_bound(own / "summary.json", entry["sha256"], inputs)
        checked = h._read_bound(own / audit_name, accepted["audit_sha256"], inputs)
        require(
            report["complete"] is True
            and checked["complete"] is True
            and checked["audit_passed"] is True
            and report["seed"] == checked["seed"] == seed
            and checked["summary_sha256"] == accepted["summary_sha256"] == entry["sha256"],
            "completed fresh seed audit chain",
        )
        result[seed] = bind_child(h, own, report, checked, inputs)
        result[seed]["summary_sha256"] = entry["sha256"]
        require(
            accepted["checkpoint"]["child_checkpoint_sha256"] == result[seed]["checkpoint_sha256"],
            "aggregate checkpoint identity",
        )
    return result


def matched_historical(before, after):
    require(
        before["parent_checkpoint_sha256"] == after["parent_checkpoint_sha256"]
        and after["summary"]["historical2000_summary_sha256"] == before["summary_sha256"],
        "matched historical parent and summary",
    )
    if "historical2000_child_sha256" in after["summary"]:
        require(
            after["summary"]["historical2000_child_sha256"] == before["child_sha256"],
            "matched historical child",
        )


def preflight(root, plan, h):
    require(sha(plan) == PLAN_SHA, "frozen plan")
    inputs = {
        str(plan.resolve()): PLAN_SHA,
        str((root / "scripts/diagnose_bilinear_errors.py").resolve()): HELPER_SHA,
    }
    chains = {budget: budget_chain(root, budget, h, inputs) for budget in BUDGETS}
    all_children = {seed: chains[2000][seed]["child"] for seed in SEEDS}
    all_children.update({seed + 10000: chains[8000][seed]["child"] for seed in SEEDS})
    h._alignment(all_children)
    for seed in SEEDS:
        matched_historical(chains[2000][seed], chains[8000][seed])
    old = root / "runs/learning/bilinear-error-geometry-v1"
    gs, ga, gd = [
        h._read_bound(old / name, pin, inputs)
        for name, pin in zip(
            ("summary.json", "independent-audit.json", "decision.json"), GEOMETRY_PINS, strict=True
        )
    ]
    accepted_chain(gs, ga, gd, GEOMETRY_PINS[0], GEOMETRY_PINS[1])
    require(
        [r["seed"] for r in gs["seed_artifacts"]] == list(SEEDS), "historical geometry seed order"
    )
    old_reports = {}
    for entry in gs["seed_artifacts"]:
        path = old / f"seed-{entry['seed']}.json"
        require(Path(entry["path"]).resolve() == path.resolve(), "historical geometry path")
        old_reports[entry["seed"]] = h._read_bound(path, entry["sha256"], inputs)
        require(
            old_reports[entry["seed"]]["child_sha256"]
            == chains[2000][entry["seed"]]["child_sha256"],
            "historical geometry child binding",
        )
    return chains, old_reports, gs, inputs


def paired_rows(before, after):
    require(len(before) == len(after), "paired query count")
    identity = ("index", "subject", "dimension", "group", "prompt_ids", "expected_set_sha256")
    values = ("a", "b", "g", "category", "exact", "strict_separation")
    rows = []
    for left, right in zip(before, after, strict=True):
        require(all(left[k] == right[k] for k in identity), "paired query identity")
        rows.append(
            {
                **{k: left[k] for k in identity},
                "budget2000": {k: left[k] for k in values},
                "budget8000": {k: right[k] for k in values},
                "delta": {k: right[k] - left[k] for k in ("a", "b", "g")},
            }
        )
    return rows


def transition(rows):
    require(bool(rows), "nonempty transition group")
    matrix = [[0] * 4 for _ in range(4)]
    counts = {"exact_gains": 0, "exact_losses": 0, "separation_gains": 0, "separation_losses": 0}
    for row in rows:
        left, right = row["budget2000"], row["budget8000"]
        matrix[CATEGORIES.index(left["category"])][CATEGORIES.index(right["category"])] += 1
        for metric, prefix in (("exact", "exact"), ("strict_separation", "separation")):
            counts[prefix + "_gains"] += not left[metric] and right[metric]
            counts[prefix + "_losses"] += left[metric] and not right[metric]
    require(sum(map(sum, matrix)) == len(rows), "transition coverage")
    return {
        "query_count": len(rows),
        "category_order": list(CATEGORIES),
        "matrix": matrix,
        **counts,
    }


def paired_report(seed, before, after, accepted):
    rows = paired_rows(before["rows"], after["rows"])
    aggregate = transition(rows)
    groups = {g: transition([r for r in rows if r["group"] == g]) for g in sorted(before["groups"])}
    require(
        {r["group"] for r in rows} == set(groups) == set(after["groups"]), "paired group coverage"
    )
    reconstructed = {
        "gains": aggregate["exact_gains"],
        "losses": aggregate["exact_losses"],
        "groups": {
            g: {"gains": v["exact_gains"], "losses": v["exact_losses"]} for g, v in groups.items()
        },
    }
    require(reconstructed == accepted, "accepted paired gains/losses reconciliation")
    return {
        "complete": True,
        "seed": seed,
        "query_count": len(rows),
        "rows": rows,
        "aggregate": aggregate,
        "groups": groups,
    }


def pooled_transitions(reports, seeds):
    rows = [r for seed in seeds for r in reports[seed]["rows"]]
    return {
        "aggregate": transition(rows),
        "groups": {
            g: transition([r for r in rows if r["group"] == g])
            for g in sorted({r["group"] for r in rows})
        },
    }


def historical_equal(rebuilt, historical):
    require(rebuilt == historical, "historical 2000 reduction equality")


def test_dependencies(root):
    return {
        str((root / name).resolve()): sha(root / name)
        for name in (
            "scripts/diagnose_bilinear_budget_geometry.py",
            "scripts/diagnose_bilinear_errors.py",
            "tests/unit/test_bilinear_budget_geometry.py",
            "tests/unit/test_bilinear_error_geometry.py",
        )
    }


def test_receipt(root, h, path, digest, inputs):
    receipt = h._read_bound(path, digest, inputs)
    require(
        receipt["passed"] is True
        and type(receipt.get("exit_code")) is int
        and receipt["exit_code"] == 0
        and receipt.get("terminal_completion_observed") is True
        and receipt["gpu_used"] is False
        and receipt["diagnostic_executed"] is False
        and type(receipt["skipped"]) is int
        and receipt["skipped"] == 0
        and type(receipt["test_count"]) is int
        and receipt["test_count"] > 0
        and receipt["tested_script_sha256"]
        == sha(root / "scripts/diagnose_bilinear_budget_geometry.py")
        and receipt["tested_helper_sha256"] == HELPER_SHA
        and receipt["test_sha256"] == sha(root / "tests/unit/test_bilinear_budget_geometry.py"),
        "completed synthetic test receipt",
    )
    dependencies = test_dependencies(root)
    require(receipt["test_dependencies"] == dependencies, "exact tested dependency inventory")
    inputs.update(dependencies)
    stdout = Path(receipt["stdout"]["path"]).resolve()
    require(
        stdout.suffix == ".txt" and sha(stdout) == receipt["stdout"]["sha256"],
        "observed test stdout",
    )
    inputs[str(stdout)] = receipt["stdout"]["sha256"]
    return stdout


def execute(root, h, chains, old_reports, old_summary, inputs, plan, receipt_path, stdout):
    out = root / "runs/learning/bilinear-budget-geometry-v1"
    script = Path(__file__).resolve()
    snapshots = {
        "summary.script.py": script.read_bytes(),
        "summary.helper.py": (root / "scripts/diagnose_bilinear_errors.py").read_bytes(),
        "summary.plan.md": plan.read_bytes(),
        "summary.test-source.py": (
            root / "tests/unit/test_bilinear_budget_geometry.py"
        ).read_bytes(),
        "summary.test-receipt.json": receipt_path.read_bytes(),
        "summary.test-stdout.txt": stdout.read_bytes(),
        "summary.inputs.json": h._bytes(inputs),
    }
    names = [f"seed-{seed}-budget-{b}.json" for seed in SEEDS for b in BUDGETS]
    names += [f"seed-{seed}-paired.json" for seed in SEEDS] + ["summary.json"] + list(snapshots)
    require(all(not (out / name).exists() for name in names), "immutable output namespace")
    require(
        all(len(data) <= 512 * 1024 for data in snapshots.values()), "snapshot publication bound"
    )
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, data in snapshots.items():
        h._write(out / name, data)
        hashes[str(out / name)] = sha(out / name)
    summary = {
        "campaign_version": "bilinear-budget-geometry-v1",
        "complete": False,
        "acceptance": False,
        "script_sha256": sha(script),
        "helper_sha256": HELPER_SHA,
        "plan_sha256": PLAN_SHA,
        "input_sha256": inputs,
        "snapshot_sha256": hashes,
        "per_seed": [],
        "limitations": [
            "Label-informed descriptive bounds; no threshold or alternative predictions.",
            "444/666 model-query observations share 222 questions; 1729 is development-selected.",
            "No Torch, checkpoint payload, training, protected data, serving or quality gate.",
        ],
    }
    try:
        reduced = {b: {} for b in BUDGETS}
        paired = {}
        for seed in SEEDS:
            entry = {"seed": seed, "budgets": {}}
            for budget in BUDGETS:
                source = chains[budget][seed]
                report = h._seed(seed, source["child"], source["child_sha256"])
                if budget == 2000:
                    historical_equal(report, old_reports[seed])
                reduced[budget][seed] = report
                path = out / f"seed-{seed}-budget-{budget}.json"
                h._write(path, report)
                entry["budgets"][str(budget)] = {
                    "path": str(path),
                    "sha256": sha(path),
                    "checkpoint_sha256": source["checkpoint_sha256"],
                    "aggregate": report["aggregate"],
                    "groups": report["groups"],
                }
            comparison = chains[8000][seed]["summary"]["child"]["paired_vs_historical2000"]
            paired[seed] = paired_report(seed, reduced[2000][seed], reduced[8000][seed], comparison)
            path = out / f"seed-{seed}-paired.json"
            h._write(path, paired[seed])
            entry["paired"] = {
                "path": str(path),
                "sha256": sha(path),
                "aggregate": paired[seed]["aggregate"],
                "groups": paired[seed]["groups"],
            }
            summary["per_seed"].append(entry)
        summary["budgets"] = {
            str(b): {
                "fresh_two": h._pool(reduced[b], [1730, 1731]),
                "all_three": h._pool(reduced[b], list(SEEDS)),
                "overlap": h._overlap(reduced[b]),
            }
            for b in BUDGETS
        }
        for key in ("fresh_two", "all_three", "overlap"):
            historical_equal(summary["budgets"]["2000"][key], old_summary[key])
        summary["transitions"] = {
            "fresh_two": pooled_transitions(paired, [1730, 1731]),
            "all_three": pooled_transitions(paired, list(SEEDS)),
        }
        require(all(sha(p) == v for p, v in (inputs | hashes).items()), "final input identity")
        summary["checks"] = {
            "historical2000_exact": True,
            "paired_comparisons_exact": True,
            "query_alignment": True,
            "final_identity_check": True,
        }
        summary["complete"] = True
    except BaseException as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        h._write(out / "summary.json", summary)
    require("torch" not in sys.modules, "stdlib execution boundary")
    return sha(out / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    require(
        Path.cwd().resolve() == root and "torch" not in sys.modules, "stdlib repository boundary"
    )
    h = helper(root)
    plan = root / "docs/experiments/2026-09-25-bilinear-budget-geometry-plan.md"
    chains, old_reports, old_summary, inputs = preflight(root, plan, h)
    inputs[str(Path(__file__).resolve())] = sha(Path(__file__))
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "input_count": len(inputs),
                    "torch_imported": False,
                    "reductions_executed": False,
                }
            )
        )
        return
    require(
        args.test_receipt is not None and args.test_receipt_sha256,
        "pinned synthetic tests required",
    )
    stdout = test_receipt(root, h, args.test_receipt, args.test_receipt_sha256, inputs)
    result = execute(
        root, h, chains, old_reports, old_summary, inputs, plan, args.test_receipt, stdout
    )
    print(json.dumps({"complete": True, "summary_sha256": result}))


if __name__ == "__main__":
    main()
