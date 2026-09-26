"""Aggregate-only recovery for equivalent aware timestamps; never executes training.

The frozen aggregator body is deliberately copied with only explicit frozen-helper
qualification and aware-time equality changed. Keep failed evidence immutable.
"""

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

_FROZEN = "24fc43fc7d31ec1bf7aa84b475af79198717fbb6e34f41453939a3da49e9fdde"
_FAILED_SUMMARY = "c3d79ed3effe7504b67121b25274603b2a5870f9150a5e8043b975f9dc478c2c"
_FAILED_STDOUT = "68736e31ddca6d572c4384273d7317dbfb02bdbb37896de8b555df40296fbce3"
_ORIGINAL_TEST = "b786a28b9970843ccc14186748757b25ac187285a40775ee6a2cd8a3d8622229"


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


def _load_frozen(root):
    path = root / "scripts/refit_bilinear_replication.py"
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == _FROZEN, "frozen runner identity")
    spec = importlib.util.spec_from_file_location("frozen_replication_aggregate_helper", path)
    _require(spec is not None and spec.loader is not None, "frozen loader")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result, path


def _aggregate(out, context, mean, helper, summary, frozen):
    started = time.perf_counter()
    reports, children, entries = {}, {}, []
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
                and report["campaign_version"] == frozen._CAMPAIGN
                and report["evaluator"] == frozen._EVALUATOR
                and report["script_sha256"] == summary["script_sha256"]
                and report["recipe_sha256"] == summary["recipe_sha256"]
                and report["scorer_sha256"] == frozen._SCORER,
                "complete matching fresh execution",
            )
            _require(
                report["parent_checkpoint_sha256"]
                == frozen._SEED_CONTRACTS[seed]["parent_checkpoint_sha256"]
                and report["width8_sha256"] == frozen._SEED_CONTRACTS[seed]["width8_sha256"]
                and report["comparator_floors"] == frozen._SEED_CONTRACTS[seed]["floors"],
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
                == frozen._seed_gate(
                    child, report["invariants"], frozen._SEED_CONTRACTS[seed]["floors"]
                ),
                "fresh gate reconstruction",
            )
            _require(
                all(value is True for value in report["invariants"].values()),
                "execution invariant failure",
            )
            _require(
                report["child"]
                == {
                    k: child[k]
                    for k in ("aggregate", "groups", "paired_vs_parent_dense", "paired_vs_width8")
                },
                "fresh child summary binding",
            )
            receipt_path = folder / "execution-receipt.json"
            receipt = helper._read(
                helper._bind(receipt_path, helper._sha(receipt_path), context["inputs"])
            )
            stdout = helper._bind(
                folder / "primary-stdout.txt", receipt["stdout_sha256"], context["inputs"]
            )
            begin, finish = frozen._terminal(
                receipt, seed, helper._sha(path), helper._sha(stdout), previous
            )
            _require(
                previous_finish is None or begin >= previous_finish,
                "overlapping/out-of-order seed processes",
            )
            launch = folder / "launch.json"
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
            reports[seed], children[seed] = report, child
            entries.append(
                {
                    "seed": seed,
                    "role": "fresh replication",
                    "path": str(path),
                    "sha256": helper._sha(path),
                    "execution_receipt": str(receipt_path),
                    "execution_receipt_sha256": previous,
                }
            )
        summary.update(
            reports=entries,
            **frozen._aggregate_reports(children, context["historical1729"], mean),
            gate=frozen._campaign_gate(reports, True),
        )
        summary["historical1729"] = {
            "role": "historical development/selection seed",
            "fresh_execution": False,
            "summary_sha256": frozen._SUMMARY,
            "audit_sha256": frozen._AUDIT,
            "decision_sha256": frozen._DECISION,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--test-receipt-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    _require(Path.cwd().resolve() == root and "torch" not in sys.modules, "Torch-free root")
    frozen, frozen_path = _load_frozen(root)
    budget, budget_path = frozen._load_budget(root)
    bilinear, bilinear_path = budget._load_bilinear(root)
    mean, mean_path = bilinear._load_mean(root)
    helper, helper_path = mean._load_helper(root)
    folder = root / "runs/learning/bilinear-seed-replication-v1"
    out = folder / "aggregate-v2.json"
    frozen._refuse(out, mean, True)
    for suffix in (
        "aggregation-script.py",
        "aggregation-test-receipt.json",
        "aggregation-test-stdout.txt",
    ):
        _require(not out.with_suffix("." + suffix).exists(), "immutable aggregation snapshot")
    plan = root / "docs/experiments/2026-09-25-bilinear-seed-replication-plan.md"
    recipe = root / "configs/experiments/bilinear_seed_replication_v1.json"
    context = frozen._preflight(
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
    repair = root / "docs/experiments/2026-09-25-bilinear-replication-evidence-repair.md"
    helper._bind(
        repair,
        "0020f6e9136669109101af048a830690eaa716d9a3607dd1b612e95ab0a57e4f",
        context["inputs"],
    )
    script = Path(__file__).resolve()
    digest = helper._sha(script)
    helper._bind(frozen_path, _FROZEN, context["inputs"])
    helper._bind(script, digest, context["inputs"])
    helper._bind(folder / "summary.json", _FAILED_SUMMARY, context["inputs"])
    helper._bind(folder / "primary-stdout.txt", _FAILED_STDOUT, context["inputs"])
    failed_receipt = folder / "execution-receipt.json"
    failed = helper._read(
        helper._bind(failed_receipt, helper._sha(failed_receipt), context["inputs"])
    )
    _require(
        failed["exit_code"] == 1
        and failed["terminal_completion_observed_by_primary"] is True
        and failed["summary_sha256"] == _FAILED_SUMMARY
        and failed["stdout_sha256"] == _FAILED_STDOUT,
        "failed aggregate terminal chain",
    )
    old_receipt = root / "runs/learning/bilinear-seed-replication-runner-tests-v1/receipt.json"
    helper._bind(old_receipt, _ORIGINAL_TEST, context["inputs"])
    old_tests = helper._read(old_receipt)
    for path, pin in old_tests["test_dependencies"].items():
        helper._bind(path, pin, context["inputs"])
    old_stdout = helper._bind(
        old_tests["stdout"]["path"], old_tests["stdout"]["sha256"], context["inputs"]
    )
    if args.preflight_only:
        helper._unchanged(context["inputs"])
        _require("torch" not in sys.modules, "preflight imported Torch")
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "torch_imported": False,
                    "input_count": len(context["inputs"]),
                    "aggregation_runner_sha256": digest,
                }
            )
        )
        return
    _require(
        args.test_receipt is not None and args.test_receipt_sha256,
        "pinned aggregation tests required",
    )
    receipt_path = helper._bind(
        args.test_receipt.resolve(), args.test_receipt_sha256, context["inputs"]
    )
    receipt = helper._read(receipt_path)
    _require(
        receipt["passed"] is True
        and receipt["gpu_used"] is False
        and receipt["experiment_training_executed"] is False
        and receipt["tested_script_sha256"] == digest
        and receipt["frozen_runner_sha256"] == _FROZEN,
        "aggregation CPU receipt",
    )
    required = {
        str((root / "tests/unit/test_bilinear_replication_aggregation.py").resolve()),
        str((root / "tests/unit/test_bilinear_replication.py").resolve()),
    }
    _require(set(receipt["test_dependencies"]) == required, "aggregation test inventory")
    for path, pin in receipt["test_dependencies"].items():
        helper._bind(path, pin, context["inputs"])
    stdout = helper._bind(receipt["stdout"]["path"], receipt["stdout"]["sha256"], context["inputs"])
    helper._unchanged(context["inputs"])
    snapshots = {
        "script.py": frozen_path.read_bytes(),
        "budget-helper.py": budget_path.read_bytes(),
        "bilinear-helper.py": bilinear_path.read_bytes(),
        "mean-helper.py": mean_path.read_bytes(),
        "helper.py": helper_path.read_bytes(),
        "plan.md": plan.read_bytes(),
        "recipe.json": recipe.read_bytes(),
        "source.zip": context["source"].read_bytes(),
        "configs.zip": context["configs"].read_bytes(),
        "test-receipt.json": old_receipt.read_bytes(),
        "test-stdout.txt": old_stdout.read_bytes(),
        "inputs.json": mean._bytes(context["inputs"]),
        "aggregation-script.py": script.read_bytes(),
        "aggregation-test-receipt.json": receipt_path.read_bytes(),
        "aggregation-test-stdout.txt": stdout.read_bytes(),
    }
    hashes = {}
    for suffix, content in snapshots.items():
        path = out.with_suffix("." + suffix)
        with path.open("xb") as stream:
            stream.write(content)
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
    summary = {
        "campaign_version": frozen._CAMPAIGN,
        "report_kind": "aggregate",
        "complete": False,
        "acceptance": False,
        "architecture": bilinear._architecture(),
        "objective": frozen._OBJECTIVE,
        "evaluator": frozen._EVALUATOR,
        "plan_sha256": frozen._PLAN,
        "script_sha256": _FROZEN,
        "aggregation_runner_sha256": digest,
        "aggregation_revision": "aware-time-v2",
        "failed_aggregate_summary_sha256": _FAILED_SUMMARY,
        "failed_aggregate_stdout_sha256": _FAILED_STDOUT,
        "failed_aggregate_execution_receipt_sha256": helper._sha(failed_receipt),
        "scorer_sha256": frozen._SCORER,
        "budget_helper_sha256": frozen._BUDGET,
        "mean_helper_sha256": frozen._MEAN,
        "helper_sha256": mean._HELPER,
        "recipe_sha256": helper._sha(recipe),
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "historical1729_summary_sha256": frozen._SUMMARY,
        "historical1729_audit_sha256": frozen._AUDIT,
        "historical1729_decision_sha256": frozen._DECISION,
        "protected_test_used": False,
        "standard_serving_supported": False,
        "limitations": [
            "Aggregate-only recovery; no retraining or new predictions.",
            "Equivalent aware timestamps compare by instant, not spelling.",
            "666 model-query observations share 222 validation queries; "
            "seed1729 is historical selection evidence.",
            "No serving promotion, protected-test result or durable weight archive is established.",
        ],
    }
    summary["preflight_input_sha256"] = dict(context["inputs"])
    context["inputs"] = dict(context["inputs"])
    summary["input_sha256"] = context["inputs"]
    _aggregate(out, context, mean, helper, summary, frozen)
    _require("torch" not in sys.modules, "aggregation imported Torch")


if __name__ == "__main__":
    main()
