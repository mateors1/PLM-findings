"""Aggregate audited 8000-update replications; no training entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

_RUNNER = "5da5ad02365f72f5e8a7123183d0b5d2f56e50c3538e64e2ac984e2b491230dd"

_REPAIR_PLAN = "130fd639ecc9b1d771f44f87e1997d1690eb6255b39f85c81f959616f84a89eb"
_REPAIR_BINDINGS = {
    "repair-declaration.md": (
        "docs/experiments/2026-09-25-bilinear-budget8000-replication-evidence-repair.md",
        _REPAIR_PLAN,
    ),
    "failed-auditor-script.py": (
        "runs/learning/bilinear-budget8000-replication-v1/independent-audit.py",
        "495db4e3d65e3893fdc68579cad62792c3ab73fd614738b7e97263935af00daa",
    ),
    "failed-auditor-test-receipt.json": (
        "runs/learning/bilinear-budget8000-replication-auditor-tests-v1/test-receipt.json",
        "e130b1dee6d8cedae18af6abdf3cdaabeb5ac78668c8e72802357bde6b8e1ad1",
    ),
    "failed-audit-stdout.txt": (
        "runs/learning/bilinear-budget8000-replication-v1/seed-1730/audit-stdout.txt",
        "8c5773b049fc57518f2c389bbd616046d70743704c90a8eba7f9c343732cd8fb",
    ),
    "failed-audit-execution-receipt.json": (
        "runs/learning/bilinear-budget8000-replication-v1/seed-1730/audit-execution-receipt.json",
        "d25163d1315aafbc98d9a2cc2bd7f1c7e9c8bdd1355ba27a5d4e3fab2b2cbbb6",
    ),
    "original-aggregation-script.py": (
        "scripts/aggregate_bilinear_budget8000_replication.py",
        "78cae68bdf741ec35c6deff6a62b57673092be0cf61968e88bbf40274878789f",
    ),
    "original-aggregation-test-receipt.json": (
        "runs/learning/bilinear-budget8000-replication-runner-tests-v1/aggregate-receipt.json",
        "0dc5c87d41db7ae3e1a3d9cd591717d0941501df201e836abcad765ead2e3f36",
    ),
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _instant(value):
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})", value
    )
    _require(match is not None, "timezone-aware ISO launch/terminal timestamps required")
    base = datetime.fromisoformat(match[1] + match[3])
    delta = base - datetime(1970, 1, 1, tzinfo=UTC)
    fraction = Fraction(int(match[2]), 10 ** len(match[2])) if match[2] else Fraction(0)
    return Fraction(delta.days * 86400 + delta.seconds) + fraction


def _same_instant(left, right):
    return _instant(left) == _instant(right)


def _terminal(receipt, seed, summary_sha, stdout_sha, previous=None):
    _require(
        receipt["seed"] == seed
        and receipt["execution_order"] == seed - 1729
        and type(receipt["exit_code"]) is int
        and receipt["exit_code"] == 0
        and receipt["terminal_completion_observed_by_primary"] is True
        and receipt["summary_sha256"] == summary_sha
        and receipt["stdout_sha256"] == stdout_sha
        and receipt["previous_terminal_receipt_sha256"] == previous,
        "ordered terminal receipt",
    )
    begin, finish = _instant(receipt["started_at_utc"]), _instant(receipt["finished_at_utc"])
    _require(finish >= begin, "reversed terminal chronology")
    return begin, finish


def _chronology(begin, previous_finish):
    _require(
        previous_finish is None or begin >= previous_finish,
        "overlapping/out-of-order seed processes",
    )


def _fresh_audit(folder, seed, summary_sha, helper, inputs, auditor_sha, test_receipt_sha):
    audit_path = folder / "independent-audit-v2.json"
    audit = helper._read(helper._bind(audit_path, helper._sha(audit_path), inputs))
    receipt_path = folder / "audit-v2-execution-receipt.json"
    receipt = helper._read(helper._bind(receipt_path, helper._sha(receipt_path), inputs))
    stdout = helper._bind(folder / "audit-v2-stdout.txt", receipt["stdout_sha256"], inputs)
    _audit_contract(audit, receipt, seed, summary_sha, helper._sha(audit_path), helper._sha(stdout))
    _require(
        audit["script_sha256"] == auditor_sha and audit["test_receipt_sha256"] == test_receipt_sha,
        "frozen auditor and test receipt identity",
    )
    return audit, receipt


def _audit_contract(audit, receipt, seed, summary_sha, audit_sha, stdout_sha):
    _require(
        audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["seed"] == seed
        and audit["summary_sha256"] == summary_sha,
        "matching completed independent audit",
    )
    _require(
        receipt["seed"] == seed
        and receipt["audit_sha256"] == audit_sha
        and receipt["summary_sha256"] == summary_sha
        and receipt["auditor_sha256"] == audit["script_sha256"]
        and receipt["test_receipt_sha256"] == audit["test_receipt_sha256"]
        and receipt["stdout_sha256"] == stdout_sha
        and type(receipt["exit_code"]) is int
        and receipt["exit_code"] == 0
        and receipt["terminal_completion_observed_by_primary"] is True,
        "observed matching audit execution",
    )


def _campaign_gate(reports, audits, historical_accepted, runner):
    _require(set(audits) == {1730, 1731}, "exact audited seed coverage")
    gate = runner._campaign_gate(reports, historical_accepted)
    gate["checks"].update(
        {
            f"fresh_seed{seed}_audited": audits[seed]["complete"] is True
            and audits[seed]["audit_passed"] is True
            for seed in (1730, 1731)
        }
    )
    gate["primary_checks_passed"] = all(gate["checks"].values())
    return gate


def _load_runner(root):
    path = root / "scripts/refit_bilinear_budget8000_replication.py"
    _require(
        hashlib.sha256(path.read_bytes()).hexdigest() == _RUNNER, "frozen seed runner identity"
    )
    spec = importlib.util.spec_from_file_location("budget8000_replication_seed_helper", path)
    _require(spec is not None and spec.loader is not None, "seed helper loader")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner, path


def _aggregate(out, context, mean, helper, summary, runner):
    started = time.perf_counter()
    reports, children, entries, audits = {}, {}, [], {}
    previous, previous_finish = None, None
    try:
        for seed in (1730, 1731):
            folder = out.parent / f"seed-{seed}"
            path = folder / "summary.json"
            report = helper._read(helper._bind(path, helper._sha(path), context["inputs"]))
            _require(
                report["complete"] is True
                and report["final_identity_check"] is True
                and report["seed"] == seed
                and report["model_seed"] == seed
                and report["data_split_seed"] == 1729
                and report["campaign_version"] == runner._CAMPAIGN
                and report["evaluator"] == runner._EVALUATOR
                and report["script_sha256"] == summary["script_sha256"]
                and report["recipe_sha256"] == summary["recipe_sha256"]
                and report["scorer_sha256"] == runner._SCORER,
                "complete matching fresh execution",
            )
            _require(
                report["parent_checkpoint_sha256"]
                == runner._SEED_CONTRACTS[seed]["parent_checkpoint_sha256"]
                and report["width8_sha256"] == runner._SEED_CONTRACTS[seed]["width8_sha256"]
                and report["comparator_floors"] == runner._SEED_CONTRACTS[seed]["floors"],
                "fresh seed comparator/parent identity",
            )
            for manifest in (
                report["input_sha256"],
                report["snapshot_sha256"],
                report["runtime_files_sha256"],
            ):
                helper._unchanged(manifest)
                for name, digest in manifest.items():
                    _require(
                        name not in context["inputs"] or context["inputs"][name] == digest,
                        "input identity conflict",
                    )
                    context["inputs"][name] = digest
            for name, digest in report["artifact_sha256"].items():
                helper._bind(folder / name, digest, context["inputs"])
            child = helper._read(folder / "child.json")
            _require(
                child["complete"] is True
                and report["gate"]
                == runner._seed_gate(
                    child, report["invariants"], runner._SEED_CONTRACTS[seed]["floors"]
                ),
                "fresh gate reconstruction",
            )
            _require(
                all(value is True for value in report["invariants"].values()),
                "execution invariant failure",
            )
            comparisons = helper._read(folder / "comparisons.json")
            _require(
                comparisons["paired_vs_historical2000"]
                == context["extended"]._historical_pair(
                    child, context["seeds"][seed]["historical2000"], mean
                ),
                "matched2000 pairing",
            )
            _require(
                report["child"]
                == {
                    k: child[k]
                    for k in ("aggregate", "groups", "paired_vs_parent_dense", "paired_vs_width8")
                }
                | {"paired_vs_historical2000": comparisons["paired_vs_historical2000"]},
                "fresh child summary binding",
            )
            receipt_path = folder / "execution-receipt.json"
            receipt = helper._read(
                helper._bind(receipt_path, helper._sha(receipt_path), context["inputs"])
            )
            stdout = helper._bind(
                folder / "primary-stdout.txt", receipt["stdout_sha256"], context["inputs"]
            )
            begin, finish = _terminal(
                receipt, seed, helper._sha(path), helper._sha(stdout), previous
            )
            _chronology(begin, previous_finish)
            launch = folder / "launch.json"
            _require(launch.is_file(), "explicit launch receipt required")
            if launch.exists():
                launch_value = helper._read(
                    helper._bind(launch, helper._sha(launch), context["inputs"])
                )
                _require(
                    launch_value["seed"] == seed
                    and _same_instant(launch_value["started_at_utc"], receipt["started_at_utc"]),
                    "launch/terminal chronology binding",
                )
            previous, previous_finish = helper._sha(receipt_path), finish
            seed_audit, audit_receipt = _fresh_audit(
                folder,
                seed,
                helper._sha(path),
                helper,
                context["inputs"],
                context["auditor_sha256"],
                context["auditor_test_receipt_sha256"],
            )
            _require(seed_audit["gate"] == report["gate"], "audited seed gate agrees")
            audits[seed] = seed_audit
            reports[seed], children[seed] = report, child
            entries.append(
                {
                    "seed": seed,
                    "role": "fresh replication",
                    "path": str(path),
                    "sha256": helper._sha(path),
                    "execution_receipt": str(receipt_path),
                    "execution_receipt_sha256": previous,
                    "audit_sha256": helper._sha(folder / "independent-audit-v2.json"),
                    "audit_path": str(folder / "independent-audit-v2.json"),
                    "audit_execution_receipt_sha256": helper._sha(
                        folder / "audit-v2-execution-receipt.json"
                    ),
                    "audit_execution_receipt": str(folder / "audit-v2-execution-receipt.json"),
                    "audit_stdout_sha256": audit_receipt["stdout_sha256"],
                }
            )
        summary.update(
            reports=entries,
            **runner._aggregate_reports(children, context["historical1729"], mean),
            gate=_campaign_gate(reports, audits, True, runner),
        )
        summary["historical1729"] = {
            "role": "historical development/selection seed",
            "fresh_execution": False,
            "summary_sha256": runner._SUMMARY,
            "audit_sha256": runner._AUDIT,
            "decision_sha256": runner._DECISION,
            "child_sha256": context["historical1729_child_sha256"],
            "aggregate": context["historical1729"]["aggregate"],
            "groups": context["historical1729"]["groups"],
        }
        summary["observation_scope"] = {
            "fresh_seed_order": [1730, 1731],
            "fresh_query_seed_observations": 444,
            "all_query_seed_observations": 666,
            "shared_validation_queries": 222,
            "new_checkpoints": 2,
            "historical_seed_retrained": False,
        }
        _require(
            reports[1730]["environment"]
            == reports[1731]["environment"]
            == context["historical1729_summary"]["environment"]
            and reports[1730]["numerical_settings"]
            == reports[1731]["numerical_settings"]
            == context["historical1729_summary"]["numerical_settings"],
            "shared dependency/device/numerical environment",
        )
        summary["environment"] = reports[1730]["environment"]
        summary["numerical_settings"] = reports[1730]["numerical_settings"]
        helper._unchanged(context["inputs"])
        helper._unchanged(summary["snapshot_sha256"])
        summary.update(final_identity_check=True, complete=True)
    except BaseException as exc:
        summary["error"] = repr(exc)
        summary["reports"] = entries
        raise
    finally:
        # Dynamic aggregate inputs are sealed separately after the immutable preflight snapshot.
        manifest = out.with_suffix(".aggregate-inputs.json")
        mean._write(manifest, context["inputs"])
        summary["aggregate_inputs_path"] = str(manifest)
        summary["aggregate_inputs_sha256"] = helper._sha(manifest)
        summary["wall_seconds"] = time.perf_counter() - started
        mean._write(out, helper._failure_safe(summary))


def _test_dependency_inventory(dependencies, required):
    _require(
        isinstance(dependencies, dict)
        and set(dependencies) == {str(p.resolve()) for p in required},
        "aggregate test dependency inventory",
    )


def _bind_repair(root, helper, inputs):
    paths = {
        suffix: helper._bind(root / relative, digest, inputs)
        for suffix, (relative, digest) in _REPAIR_BINDINGS.items()
    }
    failed = helper._read(paths["failed-audit-execution-receipt.json"])
    _require(
        type(failed["exit_code"]) is int
        and failed["exit_code"] == 1
        and failed["seed"] == 1730
        and failed["audit_sha256"] is None
        and failed["training_executed"] is False
        and failed["terminal_completion_observed_by_primary"] is True
        and failed["stdout_sha256"] == _REPAIR_BINDINGS["failed-audit-stdout.txt"][1]
        and failed["auditor_sha256"] == _REPAIR_BINDINGS["failed-auditor-script.py"][1]
        and failed["test_receipt_sha256"]
        == _REPAIR_BINDINGS["failed-auditor-test-receipt.json"][1],
        "preserved failed audit attempt",
    )
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--seed-test-receipt", type=Path)
    parser.add_argument("--seed-test-receipt-sha256")
    parser.add_argument("--auditor-sha256")
    parser.add_argument("--auditor-test-receipt", type=Path)
    parser.add_argument("--auditor-test-receipt-sha256")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/learning/bilinear-budget8000-replication-v1/aggregate-v2.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    _require(
        Path.cwd().resolve() == root and "torch" not in sys.modules, "Torch-free root preflight"
    )
    runner, runner_path = _load_runner(root)
    budget, budget_path = runner._load_budget(root)
    bilinear, bilinear_path = budget._load_bilinear(root)
    mean, mean_path = bilinear._load_mean(root)
    helper, helper_path = mean._load_helper(root)
    helper._modules()
    out = args.out.resolve()
    runner._refuse(out, mean, aggregate=True)
    plan = root / "docs/experiments/2026-09-25-bilinear-budget8000-replication-plan.md"
    recipe = root / "configs/experiments/bilinear_budget8000_replication_v1.json"
    context = runner._preflight(
        root,
        plan,
        recipe,
        budget,
        budget_path,
        bilinear,
        bilinear_path,
        mean,
        mean_path,
        helper,
        helper_path,
    )
    helper._bind(runner_path, _RUNNER, context["inputs"])
    helper._bind(script, digest, context["inputs"])
    repair_paths = _bind_repair(root, helper, context["inputs"])
    if args.preflight_only:
        _require("torch" not in sys.modules, "preflight imported Torch")
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "torch_imported": False,
                    "input_count": len(context["inputs"]),
                    "script_sha256": _RUNNER,
                    "aggregation_runner_sha256": digest,
                    "auditor_sha256": args.auditor_sha256,
                    "auditor_test_receipt_sha256": args.auditor_test_receipt_sha256,
                }
            )
        )
        return
    _require(
        args.test_receipt
        and args.test_receipt_sha256
        and args.seed_test_receipt
        and args.seed_test_receipt_sha256,
        "both pinned test receipts required",
    )
    _require(
        args.auditor_sha256 and args.auditor_test_receipt and args.auditor_test_receipt_sha256,
        "explicit frozen auditor and test receipt pins required",
    )
    auditor_path = helper._bind(
        out.parent / "independent-audit-v2.py", args.auditor_sha256, context["inputs"]
    )
    auditor_receipt_path = helper._bind(
        args.auditor_test_receipt, args.auditor_test_receipt_sha256, context["inputs"]
    )
    auditor_receipt = helper._read(auditor_receipt_path)
    _require(
        auditor_receipt["passed"] is True
        and auditor_receipt["tested_auditor_sha256"] == args.auditor_sha256,
        "passing frozen auditor test receipt",
    )
    context["auditor_sha256"] = args.auditor_sha256
    context["auditor_test_receipt_sha256"] = args.auditor_test_receipt_sha256
    receipt_paths = []
    for path, pinned, expected in (
        (args.seed_test_receipt, args.seed_test_receipt_sha256, _RUNNER),
        (args.test_receipt, args.test_receipt_sha256, digest),
    ):
        path = helper._bind(path, pinned, context["inputs"])
        value = helper._read(path)
        _require(
            value["passed"] is True
            and value["gpu_used"] is False
            and value["experiment_training_executed"] is False
            and value["tested_script_sha256"] == expected
            and value["tested_recipe_sha256"] == helper._sha(recipe),
            "matched test receipt",
        )
        if expected == digest:
            required = [
                root / "tests/unit/test_bilinear_budget8000_replication_aggregate_v2.py",
                runner_path,
                root / "tests/unit/test_bilinear_replication.py",
                mean_path,
                root / "scripts/aggregate_bilinear_budget8000_replication.py",
                root / "tests/unit/test_bilinear_budget8000_replication_aggregate.py",
            ]
            _test_dependency_inventory(value["test_dependencies"], required)
            _require(value["tested_runner_sha256"] == _RUNNER, "tested seed runner identity")
        for dependency, dependency_sha in value["test_dependencies"].items():
            helper._bind(dependency, dependency_sha, context["inputs"])
        stdout = helper._bind(value["stdout"]["path"], value["stdout"]["sha256"], context["inputs"])
        receipt_paths.append((path, stdout))
    helper._unchanged(context["inputs"])
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "script.py": runner_path.read_bytes(),
        "aggregation-script.py": script.read_bytes(),
        "auditor-script.py": auditor_path.read_bytes(),
        "auditor-test-receipt.json": auditor_receipt_path.read_bytes(),
        "budget-helper.py": budget_path.read_bytes(),
        "replication-helper.py": context["replication_path"].read_bytes(),
        "budget8000-helper.py": context["extended_path"].read_bytes(),
        "bilinear-helper.py": bilinear_path.read_bytes(),
        "mean-helper.py": mean_path.read_bytes(),
        "helper.py": helper_path.read_bytes(),
        "plan.md": plan.read_bytes(),
        "recipe.json": recipe.read_bytes(),
        "source.zip": context["source"].read_bytes(),
        "configs.zip": context["configs"].read_bytes(),
        "test-receipt.json": receipt_paths[0][0].read_bytes(),
        "test-stdout.txt": receipt_paths[0][1].read_bytes(),
        "aggregation-test-receipt.json": receipt_paths[1][0].read_bytes(),
        "aggregation-test-stdout.txt": receipt_paths[1][1].read_bytes(),
        "inputs.json": mean._bytes(context["inputs"]),
    }
    snapshots.update({suffix: path.read_bytes() for suffix, path in repair_paths.items()})
    hashes = {}
    for suffix, content in snapshots.items():
        path = out.with_suffix("." + suffix)
        with path.open("xb") as stream:
            stream.write(content)
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
    summary = {
        "campaign_version": runner._CAMPAIGN,
        "report_kind": "aggregate",
        "aggregation_version": 2,
        "evidence_repair": {
            suffix: {"path": str(path), "sha256": helper._sha(path)}
            for suffix, path in repair_paths.items()
        },
        "complete": False,
        "acceptance": False,
        "architecture": bilinear._architecture(),
        "objective": runner._OBJECTIVE,
        "evaluator": runner._EVALUATOR,
        "script_sha256": _RUNNER,
        "aggregation_runner_sha256": digest,
        "auditor_sha256": args.auditor_sha256,
        "auditor_test_receipt_sha256": args.auditor_test_receipt_sha256,
        "plan_sha256": runner._PLAN,
        "recipe_sha256": helper._sha(recipe),
        "scorer_sha256": runner._SCORER,
        "mean_helper_sha256": runner._MEAN,
        "budget_helper_sha256": runner._BUDGET,
        "replication_helper_sha256": runner._REPLICATION,
        "budget8000_helper_sha256": runner._EXTENDED,
        "helper_sha256": mean._HELPER,
        "snapshot_sha256": hashes,
        "preflight_input_sha256": dict(context["inputs"]),
        "historical1729_summary_sha256": runner._SUMMARY,
        "historical1729_audit_sha256": runner._AUDIT,
        "historical1729_decision_sha256": runner._DECISION,
        "protected_test_used": False,
        "standard_serving_supported": False,
        "limitations": [
            "444 fresh and 666 total observations reuse 222 validation queries.",
            "Historical1729 is development-selected, never freshly executed here.",
            "Independent evidence audits do not repeat CUDA training.",
            "No protected evaluation, promotion or remote weight-archive claim.",
        ],
    }
    context["inputs"] = dict(context["inputs"])
    summary["input_sha256"] = context["inputs"]
    _aggregate(out, context, mean, helper, summary, runner)


if __name__ == "__main__":
    main()
