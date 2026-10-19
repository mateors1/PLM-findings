"""Metadata-only aggregation of three audited 32000-update seeds; never promote."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

CAMPAIGN = "bilinear-budget32000-v1"
EVALUATOR = "plm-bilinear-budget32000-screen-v1"
ARCHITECTURE = "plm-frozen-symmetric-bilinear-residual-v1"
OBJECTIVE = "plm-bilinear-residual-balanced-bce-v1"
INFERENCE = "dense-positive-bilinear-residual-subject-excluded-ascending-fp32-b8-last6-v1"
INPUT_ADAPTER = "plm-authenticated-partition-membership-v1"
AMENDMENT_SHA = "bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f"
PLAN_SHA = "5f1e0243b373ad1287c31dcef8a3c1cfa310cac9701795469a68677506e6f444"
MEAN_SHA = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
SCORER_SHA = "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
SPLIT = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
SEEDS = (1729, 1730, 1731)
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")
GROUP_COUNTS = {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 68}
FLOORS = {
    1729: (216, 0.9998843626799785, (103, 51, 62)),
    1730: (220, 0.9999717006519728, (103, 51, 66)),
    1731: (214, 0.9998584743646364, (103, 51, 60)),
}
INVARIANTS = {
    "parent_replay_exact",
    "zero_A_replay_exact",
    "initial_train_loss_exact",
    "completed_32000_updates",
    "original_93_tensors_unchanged",
    "residual_changed",
    "reload_exact",
    "optimizer_reload_exact",
    "train_only_supervision",
}
BASE = "runs/learning/bilinear-budget32000-v1"
READINESS = "runs/learning/bilinear-budget32000-readiness-v1/readiness.json"
PLAN = "docs/experiments/2026-09-26-bilinear-budget32000-plan.md"
AMENDMENT = "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md"
RECIPE = "configs/experiments/bilinear_budget32000_v1.json"
RUNNER = "scripts/refit_bilinear_budget32000.py"
SCORER = "scripts/refit_bilinear_residual.py"
MEAN = "scripts/refit_membership_projection.py"
AGGREGATOR = "scripts/aggregate_bilinear_budget32000.py"
AUDITOR = BASE + "/independent-audit.py"
PRIMARY_TEST = "tests/unit/test_bilinear_budget32000.py"
AUDITOR_TEST = "tests/unit/test_bilinear_budget32000_audit.py"
AGGREGATE_TEST = "tests/unit/test_bilinear_budget32000_aggregate.py"
TEST_DEPENDENCIES = {
    "runner": {
        RUNNER,
        PRIMARY_TEST,
        SCORER,
        MEAN,
        "runs/learning/wide-first-choice-v1/summary.script.py",
        "scripts/refit_symmetric_affine_v3.py",
        "tests/unit/test_bilinear_residual.py",
        "tests/fixtures/projection_refit_runtime_source.zip",
        PLAN,
        AMENDMENT,
    },
    "auditor": {
        AUDITOR,
        AUDITOR_TEST,
        "runs/learning/projection-only-refit-v1/independent-audit.py",
        PLAN,
        AMENDMENT,
    },
    "aggregate": {AGGREGATOR, AGGREGATE_TEST, MEAN, PLAN},
}
TEST_DIRECTORIES = {
    role: f"runs/learning/bilinear-budget32000-{role}-tests-v1" for role in TEST_DEPENDENCIES
}
READINESS_FILES = (
    {RECIPE}
    | set().union(*TEST_DEPENDENCIES.values())
    | {
        directory + "/" + filename
        for directory in TEST_DIRECTORIES.values()
        for filename in ("test-receipt.json", "stdout.txt")
    }
)
SEED_FILES = {
    f"{BASE}/seed-{seed}/{name}"
    for seed in SEEDS
    for name in (
        "summary.json",
        "child.json",
        "independent-audit.json",
        "execution-receipt.json",
        "audit-execution-receipt.json",
        "primary-stdout.txt",
        "audit-stdout.txt",
    )
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(actual, expected, message):
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        message,
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class Reader:
    """Every path is admitted before opening, including hashing."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}

    def admit(self, name):
        require(
            type(name) is str and name in READINESS_FILES | SEED_FILES | {READINESS},
            "path not on explicit metadata allowlist",
        )
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), "path outside repository")
        return path

    def bind(self, name, expected=None):
        path = self.admit(name)
        require(expected is None or valid_hash(expected), "SHA256 required")
        actual = digest(path)
        require(expected is None or actual == expected, "hash mismatch: " + name)
        require(name not in self.inputs or self.inputs[name] == actual, "changed input: " + name)
        self.inputs[name] = actual
        return path

    def read(self, name, expected=None):
        return json.loads(self.bind(name, expected).read_text(encoding="utf-8"))

    def unchanged(self):
        for name, expected in list(self.inputs.items()):
            self.bind(name, expected)


def test_receipt(receipt, role, files):
    require(
        receipt.get("passed") is True
        and receipt.get("gpu_used") is False
        and receipt.get("experiment_training_executed") is False
        and type(receipt.get("exit_code")) is int
        and receipt["exit_code"] == 0
        and receipt.get("terminal_completion_observed") is True
        and type(receipt.get("tests_passed")) is int
        and receipt["tests_passed"] > 0
        and type(receipt.get("tests_skipped")) is int
        and receipt["tests_skipped"] == 0,
        "observed passing CPU tests",
    )
    dependencies = receipt["test_dependencies"]
    require(set(dependencies) == TEST_DEPENDENCIES[role], "exact test dependency inventory")
    require(
        all(files[name] == value for name, value in dependencies.items()), "test dependency hashes"
    )
    source = {"runner": RUNNER, "auditor": AUDITOR, "aggregate": AGGREGATOR}[role]
    require(receipt["tested_script_sha256"] == files[source], "tested exact source")
    stdout = TEST_DIRECTORIES[role] + "/stdout.txt"
    exact(receipt["stdout"], {"path": stdout, "sha256": files[stdout]}, "test stdout binding")
    if role == "runner":
        require(receipt["tested_recipe_sha256"] == files[RECIPE], "tested recipe")


def readiness(reader, pin):
    require(valid_hash(pin), "pinned readiness required")
    value = reader.read(READINESS, pin)
    require(
        type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("ready") is True
        and value.get("campaign_version") == CAMPAIGN
        and value.get("plan_sha256") == PLAN_SHA,
        "campaign readiness identity",
    )
    files = value["files"]
    require(set(files) == READINESS_FILES, "exact readiness file inventory")
    require(all(valid_hash(v) for v in files.values()), "readiness digest format")
    # Validate the entire inventory before opening any referenced file.
    for name in files:
        reader.admit(name)
    require(
        files[PLAN] == PLAN_SHA
        and files[MEAN] == MEAN_SHA
        and files[SCORER] == SCORER_SHA
        and files[AMENDMENT] == AMENDMENT_SHA,
        "frozen plan and arithmetic identities",
    )
    for name, pin in files.items():
        reader.bind(name, pin)
    for role, directory in TEST_DIRECTORIES.items():
        name = directory + "/test-receipt.json"
        test_receipt(reader.read(name, files[name]), role, files)
    return value


def load_mean(reader):
    path = reader.bind(MEAN, MEAN_SHA)
    spec = importlib.util.spec_from_file_location("budget32000_aggregate_metrics", path)
    require(spec is not None and spec.loader is not None, "metric helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instant(value):
    require(type(value) is str, "timestamp string required")
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})", value
    )
    require(match is not None, "timezone-aware timestamp required")
    if match[3] != "Z":
        require(int(match[3][1:3]) < 24 and int(match[3][4:6]) < 60, "invalid UTC offset")
    delta = datetime.fromisoformat(match[1] + match[3]) - datetime(1970, 1, 1, tzinfo=UTC)
    return Fraction(delta.days * 86400 + delta.seconds) + (
        Fraction(int(match[2]), 10 ** len(match[2])) if match[2] else Fraction(0)
    )


def terminal(receipt, seed, summary_sha, stdout_sha):
    require(
        type(receipt["seed"]) is int
        and receipt["seed"] == seed
        and type(receipt["exit_code"]) is int
        and receipt["exit_code"] == 0
        and receipt["terminal_completion_observed_by_primary"] is True
        and receipt["summary_sha256"] == summary_sha
        and receipt["stdout_sha256"] == stdout_sha,
        "observed matching terminal receipt",
    )
    start, finish = instant(receipt["started_utc"]), instant(receipt["completed_utc"])
    require(finish >= start, "reversed terminal chronology")
    return start, finish


def chronology(execution, audit_receipt, seed, summary_sha, execution_sha, previous):
    begin, finish = terminal(execution, seed, summary_sha, execution["stdout_sha256"])
    audit_begin, audit_finish = terminal(
        audit_receipt, seed, summary_sha, audit_receipt["stdout_sha256"]
    )
    require(audit_begin >= finish, "audit before execution completion")
    require(
        audit_receipt["execution_receipt_sha256"] == execution_sha
        and audit_receipt["cuda_visible_devices"] == "-1",
        "audit execution binding",
    )
    if previous is None:
        require(
            seed == SEEDS[0]
            and execution["prior_seed"] is None
            and execution["prior_execution_receipt_sha256"] is None
            and execution["prior_audit_receipt_sha256"] is None,
            "first seed predecessor",
        )
    else:
        require(previous["seed"] + 1 == seed, "seed order")
        prior = execution["prior_seed"]
        require(
            set(prior) == {"seed", "summary_sha256", "completed_utc"}
            and type(prior["seed"]) is int
            and prior["seed"] == previous["seed"]
            and prior["summary_sha256"] == previous["summary_sha256"]
            and instant(prior["completed_utc"]) == previous["execution_finish"],
            "prior seed binding",
        )
        require(
            execution["prior_execution_receipt_sha256"] == previous["execution_receipt_sha256"]
            and execution["prior_audit_receipt_sha256"] == previous["audit_receipt_sha256"],
            "prior receipt hash chain",
        )
        require(
            begin >= previous["execution_finish"] and begin >= previous["audit_finish"],
            "next seed before previous audit completed",
        )
    return finish, audit_finish


def rebuild_child(child, mean):
    require(
        child["complete"] is True
        and child["query_count"] == 222
        and child["product_token_ids"] == list(range(1024, 2049))
        and len(child["responses"]) == 222,
        "complete validation coverage",
    )
    rows, queries = [], set()
    for index, row in enumerate(child["responses"]):
        prompt, truth = row["prompt_ids"], row["expected_set_ids"]
        require(
            type(row["index"]) is int
            and row["index"] == index
            and row["group"] in GROUPS
            and type(row["subject"]) is str,
            "validation index/group/subject",
        )
        require(
            len(prompt) == 5
            and all(type(v) is int for v in prompt)
            and prompt[0] == 1
            and 1024 <= prompt[1] < 2049
            and prompt[2] in (32, 33)
            and prompt[3:] == [34, 5],
            "validation prompt",
        )
        require(
            row["dimension"] == ("TYPE" if prompt[2] == 32 else "COLOR")
            and (row["group"] == "COLOR") == (prompt[2] == 33),
            "dimension group",
        )
        key = tuple(prompt)
        require(key not in queries, "duplicate validation query")
        queries.add(key)
        require(
            bool(truth)
            and all(type(v) is int and 1024 <= v < 2049 for v in truth)
            and truth == sorted(set(truth))
            and prompt[1] not in truth,
            "teacher membership",
        )
        scores = row["symmetric_relation_logits"]
        require(len(scores) == 1025, "score columns")
        selected = mean._prediction(scores, prompt[1])
        exact(row["selected_set_ids"], selected, "zero-threshold prediction")
        metrics = mean._metrics(selected, truth)
        exact(row["metrics"], metrics, "recomputed row metrics")
        excluded = set(truth) | {prompt[1]}
        negatives = [z for i, z in enumerate(scores, 1024) if i not in excluded]
        require(bool(negatives), "nonempty negative class")
        separation = min(scores[i - 1024] for i in truth) > max(negatives)
        require(row["strict_separation"] is separation, "strict separation")
        rows.append({**row, "metrics": metrics, "strict_separation": separation})
    groups = {g: mean._totals([r for r in rows if r["group"] == g]) for g in GROUPS}
    require(
        {g: v["query_count"] for g, v in groups.items()} == GROUP_COUNTS,
        "fixed validation group coverage",
    )
    aggregate = mean._totals(rows)
    exact(child["aggregate"], aggregate, "recomputed seed aggregate")
    exact(child["groups"], groups, "recomputed seed groups")
    return rows, aggregate, groups


def seed_gate(seed, aggregate, groups, invariants):
    floor, f1, counts = FLOORS[seed]
    checks = {
        "execution_invariants": set(invariants) == INVARIANTS
        and all(v is True for v in invariants.values()),
        "exact_above_matching8000": aggregate["exact_count"] > floor,
        "f1_at_least_matching8000": aggregate["f1"] >= f1,
        "group_exact_nonregression": all(
            groups[g]["exact_count"] >= c for g, c in zip(GROUPS, counts, strict=True)
        ),
        "serialization_compatible": aggregate["serialization_compatible"] == 222,
    }
    return {
        "accepted": False,
        "independent_audit_required": True,
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
    }


def seed_report(reader, ready, seed, mean, previous):
    folder = f"{BASE}/seed-{seed}"
    report = reader.read(folder + "/summary.json")
    summary_sha = reader.inputs[folder + "/summary.json"]
    files = ready["files"]
    require(
        report["complete"] is True
        and report["final_identity_check"] is True
        and report["acceptance"] is False
        and report["protected_test_used"] is False
        and report["standard_serving_supported"] is False
        and "error" not in report,
        "completed unpromoted seed",
    )
    for key, value in {
        "seed": seed,
        "data_split_seed": 1729,
        "campaign_version": CAMPAIGN,
        "evaluator": EVALUATOR,
        "architecture": ARCHITECTURE,
        "objective": OBJECTIVE,
        "inference": INFERENCE,
        "input_adapter_identity": INPUT_ADAPTER,
        "input_amendment_sha256": AMENDMENT_SHA,
        "plan_sha256": PLAN_SHA,
        "script_sha256": files[RUNNER],
        "recipe_sha256": files[RECIPE],
        "scorer_sha256": SCORER_SHA,
        "split_hash": SPLIT,
    }.items():
        exact(report[key], value, "seed identity: " + key)
    admission = {"path": READINESS, "sha256": reader.inputs[READINESS]}
    exact(report["shared_readiness"], admission, "seed shared admission")
    child = reader.read(folder + "/child.json", report["artifact_sha256"]["child.json"])
    rows, aggregate, groups = rebuild_child(child, mean)
    exact(report["child"]["aggregate"], aggregate, "summary child aggregate")
    exact(report["child"]["groups"], groups, "summary child groups")
    exact(
        report["child"]["paired_vs_parent_dense"],
        child["paired_vs_parent_dense"],
        "summary parent pairing",
    )
    gate = seed_gate(seed, aggregate, groups, report["invariants"])
    require(gate["checks"]["execution_invariants"], "execution invariant failure")
    exact(report["gate"], gate, "seed gate reconstruction")
    execution = reader.read(folder + "/execution-receipt.json")
    execution_sha = reader.inputs[folder + "/execution-receipt.json"]
    reader.bind(folder + "/primary-stdout.txt", execution["stdout_sha256"])
    audit = reader.read(folder + "/independent-audit.json")
    audit_sha = reader.inputs[folder + "/independent-audit.json"]
    audit_receipt = reader.read(folder + "/audit-execution-receipt.json")
    reader.bind(folder + "/audit-stdout.txt", audit_receipt["stdout_sha256"])
    require(
        audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["acceptance"] is False,
        "completed independent audit",
    )
    for key, value in {
        "seed": seed,
        "summary_sha256": summary_sha,
        "script_sha256": files[AUDITOR],
        "plan_sha256": PLAN_SHA,
        "execution_receipt_sha256": execution_sha,
        "test_receipt_sha256": files[TEST_DIRECTORIES["auditor"] + "/test-receipt.json"],
    }.items():
        exact(audit[key], value, "auditor identity: " + key)
    exact(audit["child"], report["child"], "audited child summary")
    exact(audit["gate"], gate, "audited gate")
    exact(audit["shared_readiness"], admission, "audit shared admission")
    require(audit_receipt["audit_sha256"] == audit_sha, "audit terminal hash")
    require(
        audit_receipt["auditor_sha256"] == files[AUDITOR]
        and audit_receipt["test_receipt_sha256"]
        == files[TEST_DIRECTORIES["auditor"] + "/test-receipt.json"],
        "audit terminal implementation identity",
    )
    finish, audit_finish = chronology(
        execution, audit_receipt, seed, summary_sha, execution_sha, previous
    )
    state = {
        "seed": seed,
        "summary_sha256": summary_sha,
        "execution_receipt_sha256": execution_sha,
        "audit_receipt_sha256": reader.inputs[folder + "/audit-execution-receipt.json"],
        "execution_finish": finish,
        "audit_finish": audit_finish,
    }
    entry = {
        "seed": seed,
        "summary_sha256": summary_sha,
        "audit_sha256": audit_sha,
        "execution_receipt_sha256": execution_sha,
        "audit_receipt_sha256": state["audit_receipt_sha256"],
        "child": report["child"],
        "gate": gate,
    }
    return rows, entry, state


def aggregate(reader, ready, mean):
    entries, all_rows, identity_rows = [], [], None
    previous = None
    for seed in SEEDS:
        rows, entry, previous = seed_report(reader, ready, seed, mean, previous)
        identities = [
            {
                k: r[k]
                for k in (
                    "index",
                    "subject",
                    "dimension",
                    "prompt_ids",
                    "expected_set_ids",
                    "group",
                )
            }
            for r in rows
        ]
        if identity_rows is not None:
            exact(identities, identity_rows, "cross-seed query/teacher alignment")
        identity_rows = identities
        entries.append(entry)
        all_rows.extend(rows)
    reader.unchanged()
    checks = {str(e["seed"]): e["gate"]["primary_checks_passed"] for e in entries}
    return {
        "complete": True,
        "acceptance": False,
        "campaign_version": CAMPAIGN,
        "evaluator": EVALUATOR,
        "plan_sha256": PLAN_SHA,
        "per_seed": entries,
        "all_three": {
            "aggregate": mean._totals(all_rows),
            "groups": {g: mean._totals([r for r in all_rows if r["group"] == g]) for g in GROUPS},
        },
        "gate": {
            "checks": checks,
            "primary_checks_passed": all(checks.values()),
            "accepted": False,
            "independent_audit_required": True,
        },
        "protected_test_used": False,
        "standard_serving_supported": False,
        "limitations": [
            "666 observations reuse 222 development queries and three parents.",
            "Metadata-only aggregation; seed audits establish payload contracts.",
            "Aggregate remains provisional until independent audit and owner decision.",
        ],
    }


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--readiness-sha256", required=True)
    parser.add_argument("--out", type=Path, default=Path(BASE + "/aggregate.json"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    require(
        Path.cwd().resolve() == root and "torch" not in sys.modules,
        "Torch-free repository-root aggregation",
    )
    require(args.readiness.resolve() == root / READINESS, "fixed readiness path")
    out = args.out.resolve()
    require(out == root / BASE / "aggregate.json", "fixed aggregate path")
    reader = Reader(root)
    ready = readiness(reader, args.readiness_sha256)
    require(ready["files"][AGGREGATOR] == digest(__file__), "frozen aggregate source")
    mean = load_mean(reader)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "readiness_passed": True,
                    "seed_artifacts_read": False,
                    "torch_imported": "torch" in sys.modules,
                }
            )
        )
        return
    snapshots = [
        out.with_suffix(".script.py"),
        out.with_suffix(".readiness.json"),
        out.with_suffix(".inputs.json"),
    ]
    require(not any(p.exists() for p in [out, *snapshots]), "immutable aggregate namespace")
    out.parent.mkdir(parents=True, exist_ok=True)
    with snapshots[0].open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    with snapshots[1].open("xb") as stream:
        stream.write((root / READINESS).read_bytes())
    result = {"complete": False, "acceptance": False, "campaign_version": CAMPAIGN}
    try:
        result.update(aggregate(reader, ready, mean))
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        write(snapshots[2], reader.inputs)
        result.update(
            input_sha256=reader.inputs,
            script_sha256=digest(__file__),
            readiness_sha256=args.readiness_sha256,
            snapshot_sha256={str(p.relative_to(root)): digest(p) for p in snapshots},
        )
        write(out, result)


if __name__ == "__main__":
    main()
