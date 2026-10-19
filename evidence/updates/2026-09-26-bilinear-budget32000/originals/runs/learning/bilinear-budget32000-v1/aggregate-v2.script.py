"""Versioned metadata aggregation of original and repaired 32000 campaign evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

BASE_SCRIPT = "scripts/aggregate_bilinear_budget32000.py"
BASE_SCRIPT_SHA = "41bc30b8e0b29742b4446a3b4a776c91ea7264f656b8dc01ab60966f1df5a09a"
BASE_READINESS = "runs/learning/bilinear-budget32000-readiness-v1/readiness.json"
BASE_READINESS_SHA = "5266fba8ab636590c0654db31b337ba8fd806361d9703c41a1b7260fcb5d766d"
AMENDMENT = "docs/experiments/2026-09-26-bilinear-budget32000-evidence-repair-v2.md"
AMENDMENT_SHA = "90aa30088980aeffd999fe6fe4e7349220d163b196ad4eb28f06193b9ce2991c"
REVISION = "bilinear-budget32000-evidence-v2"
EXECUTION_REVISION = "bilinear-budget32000-continuation-v2"
READINESS = "runs/learning/bilinear-budget32000-readiness-v2/readiness.json"
CAMPAIGN = "bilinear-budget32000-v1"
BASE = "runs/learning/" + CAMPAIGN
SOURCES = {
    "runner": "scripts/refit_bilinear_budget32000_continuation_v2.py",
    "auditor": BASE + "/independent-audit-v2.py",
    "aggregate": "scripts/aggregate_bilinear_budget32000_v2.py",
}
TESTS = {
    "runner": "tests/unit/test_bilinear_budget32000_continuation_v2.py",
    "auditor": "tests/unit/test_bilinear_budget32000_audit_v2.py",
    "aggregate": "tests/unit/test_bilinear_budget32000_aggregate_v2.py",
}
DIRECTORIES = {role: f"runs/learning/bilinear-budget32000-{role}-tests-v2" for role in SOURCES}
DEPENDENCIES = {
    role: {source, TESTS[role], AMENDMENT, BASE_READINESS} for role, source in SOURCES.items()
}
FILES = (
    {AMENDMENT, BASE_READINESS}
    | set(SOURCES.values())
    | set(TESTS.values())
    | {
        directory + "/" + name
        for directory in DIRECTORIES.values()
        for name in ("test-receipt.json", "stdout.txt")
    }
)
ORIGINAL_SUMMARIES = {
    1729: "b41fec77f92855db5753c97add0255e4593fa328192eb34fe3ab6ad567bbacd1",
    1730: "1237d6e266fa57b5284f76ab76d50069ec9f161f5e23add84e06d769a47b5e11",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_base(root):
    path = root / BASE_SCRIPT
    require(sha(path) == BASE_SCRIPT_SHA, "frozen base aggregator")
    spec = importlib.util.spec_from_file_location("frozen_budget32000_aggregate_v1", path)
    require(spec is not None and spec.loader is not None, "base source loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route(seed):
    require(type(seed) is int and seed in (1729, 1730, 1731), "declared seed")
    suffix = "" if seed == 1729 else "-v2"
    return {
        "audit": f"independent-audit{suffix}.json",
        "terminal": f"audit-execution-receipt{suffix}.json",
        "stdout": f"audit-stdout{suffix}.txt",
    }


class Reader:
    """Fixed metadata/source allowlist; never follow input or artifact manifests."""

    def __init__(self, root, base):
        self.root = Path(root).resolve()
        self.base = base
        self.inputs = {}
        seed_files = {
            f"{BASE}/seed-{seed}/{name}"
            for seed in base.SEEDS
            for name in (
                "summary.json",
                "child.json",
                "execution-receipt.json",
                "primary-stdout.txt",
                *route(seed).values(),
            )
        }
        self.allowed = (
            base.READINESS_FILES
            | {BASE_READINESS, READINESS}
            | FILES
            | seed_files
            | {BASE + "/seed-1731/continuation-complete.json"}
            | {BASE + "/seed-1731/continuation/" + name for name in FILES | {READINESS}}
        )

    def admit(self, name):
        require(
            type(name) is str and name in self.allowed, "path not on explicit metadata allowlist"
        )
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), "path outside repository")
        return path

    def bind(self, name, expected=None):
        path = self.admit(name)
        require(expected is None or self.base.valid_hash(expected), "SHA256 required")
        actual = sha(path)
        require(expected is None or actual == expected, "hash mismatch: " + name)
        require(name not in self.inputs or self.inputs[name] == actual, "changed input: " + name)
        self.inputs[name] = actual
        return path

    def read(self, name, expected=None):
        return json.loads(self.bind(name, expected).read_text(encoding="utf-8"))

    def unchanged(self):
        for name, digest in list(self.inputs.items()):
            self.bind(name, digest)


def cpu_receipt(base, value, role, files):
    require(
        value.get("passed") is True
        and value.get("gpu_used") is False
        and value.get("experiment_training_executed") is False
        and type(value.get("exit_code")) is int
        and value["exit_code"] == 0
        and value.get("terminal_completion_observed") is True
        and type(value.get("tests_passed")) is int
        and value["tests_passed"] > 0
        and type(value.get("tests_skipped")) is int
        and value["tests_skipped"] == 0,
        "observed passing v2 CPU tests",
    )
    base.exact(value["evidence_revision"], REVISION, "test evidence revision")
    base.exact(value["amendment_sha256"], AMENDMENT_SHA, "test amendment")
    base.exact(
        value["test_dependencies"],
        {n: files[n] for n in DEPENDENCIES[role]},
        "four exact direct dependencies",
    )
    base.exact(value["tested_script_sha256"], files[SOURCES[role]], "tested v2 source")
    stdout = DIRECTORIES[role] + "/stdout.txt"
    base.exact(value["stdout"], {"path": stdout, "sha256": files[stdout]}, "v2 test stdout")


def readiness(reader, pin):
    base = reader.base
    require(base.valid_hash(pin), "pinned v2 readiness")
    ready = reader.read(READINESS, pin)
    require(
        type(ready.get("schema_version")) is int
        and ready["schema_version"] == 1
        and ready.get("ready") is True
        and ready.get("campaign_version") == CAMPAIGN
        and ready.get("evidence_revision") == REVISION
        and ready.get("amendment_sha256") == AMENDMENT_SHA,
        "v2 readiness identity",
    )
    base.exact(
        ready["base_readiness"],
        {"path": BASE_READINESS, "sha256": BASE_READINESS_SHA},
        "original readiness identity",
    )
    files = ready["files"]
    require(type(files) is dict and set(files) == FILES, "exact fourteen-file readiness inventory")
    require(all(base.valid_hash(h) for h in files.values()), "readiness SHA format")
    for name in files:
        reader.admit(name)
    require(
        files[AMENDMENT] == AMENDMENT_SHA and files[BASE_READINESS] == BASE_READINESS_SHA,
        "fixed repair identities",
    )
    for name, digest in files.items():
        reader.bind(name, digest)
    original = base.readiness(reader, BASE_READINESS_SHA)
    require(original["files"][BASE_SCRIPT] == BASE_SCRIPT_SHA, "base arithmetic source admission")
    for role, directory in DIRECTORIES.items():
        cpu_receipt(base, reader.read(directory + "/test-receipt.json"), role, files)
    return ready, original


def continuation(reader, ready, summary, summary_sha, execution):
    base = reader.base
    admission = {"path": READINESS, "sha256": reader.inputs[READINESS]}
    values = {
        "execution_wrapper_sha256": ready["files"][SOURCES["runner"]],
        "execution_revision": EXECUTION_REVISION,
        "amendment_sha256": AMENDMENT_SHA,
        "continuation_readiness": admission,
    }
    for key, expected in values.items():
        base.exact(summary[key], expected, "continuation summary " + key)
    snapshots = {
        BASE + "/seed-1731/continuation/" + name: digest
        for name, digest in (ready["files"] | {READINESS: reader.inputs[READINESS]}).items()
    }
    base.exact(
        summary["continuation_snapshots"], snapshots, "exact continuation snapshot inventory"
    )
    for name in snapshots:
        reader.admit(name)
    for name, digest in snapshots.items():
        reader.bind(name, digest)
    name = BASE + "/seed-1731/continuation-complete.json"
    complete = reader.read(name)
    require(
        complete["complete"] is True and type(complete["seed"]) is int and complete["seed"] == 1731,
        "completed continuation wrapper",
    )
    for key, expected in {**values, "summary_sha256": summary_sha}.items():
        base.exact(complete[key], expected, "continuation completion " + key)
    require(
        complete["root_observed_terminal_required"] is True, "separate root observation required"
    )
    base.exact(
        execution["continuation_complete_sha256"],
        reader.inputs[name],
        "observed continuation completion hash",
    )
    return reader.inputs[name]


def seed_report(reader, ready, original, seed, mean, previous):
    base = reader.base
    folder = f"{BASE}/seed-{seed}"
    report = reader.read(folder + "/summary.json", ORIGINAL_SUMMARIES.get(seed))
    summary_sha = reader.inputs[folder + "/summary.json"]
    files = original["files"]
    require(
        report["complete"] is True
        and report["final_identity_check"] is True
        and report["acceptance"] is False
        and report["protected_test_used"] is False
        and report["standard_serving_supported"] is False
        and "error" not in report,
        "completed unpromoted seed",
    )
    for key, expected in {
        "seed": seed,
        "data_split_seed": 1729,
        "campaign_version": CAMPAIGN,
        "evaluator": base.EVALUATOR,
        "architecture": base.ARCHITECTURE,
        "objective": base.OBJECTIVE,
        "inference": base.INFERENCE,
        "input_adapter_identity": base.INPUT_ADAPTER,
        "input_amendment_sha256": base.AMENDMENT_SHA,
        "plan_sha256": base.PLAN_SHA,
        "script_sha256": files[base.RUNNER],
        "recipe_sha256": files[base.RECIPE],
        "scorer_sha256": base.SCORER_SHA,
        "split_hash": base.SPLIT,
    }.items():
        base.exact(report[key], expected, "seed identity " + key)
    original_admission = {"path": BASE_READINESS, "sha256": BASE_READINESS_SHA}
    repaired_admission = {"path": READINESS, "sha256": reader.inputs[READINESS]}
    base.exact(report["shared_readiness"], original_admission, "original scientific admission")
    child = reader.read(folder + "/child.json", report["artifact_sha256"]["child.json"])
    rows, aggregate, groups = base.rebuild_child(child, mean)
    base.exact(report["child"]["aggregate"], aggregate, "summary aggregate")
    base.exact(report["child"]["groups"], groups, "summary groups")
    base.exact(
        report["child"]["paired_vs_parent_dense"], child["paired_vs_parent_dense"], "parent pairing"
    )
    gate = base.seed_gate(seed, aggregate, groups, report["invariants"])
    require(gate["checks"]["execution_invariants"], "execution invariant failure")
    base.exact(report["gate"], gate, "fixed per-seed gate")
    execution = reader.read(folder + "/execution-receipt.json")
    execution_sha = reader.inputs[folder + "/execution-receipt.json"]
    reader.bind(folder + "/primary-stdout.txt", execution["stdout_sha256"])
    paths = route(seed)
    audit_path = folder + "/" + paths["audit"]
    terminal_path = folder + "/" + paths["terminal"]
    audit = reader.read(audit_path)
    audit_sha = reader.inputs[audit_path]
    terminal = reader.read(terminal_path)
    reader.bind(folder + "/" + paths["stdout"], terminal["stdout_sha256"])
    require(
        audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["acceptance"] is False,
        "successful routed independent audit",
    )
    auditor_sha = files[base.AUDITOR] if seed == 1729 else ready["files"][SOURCES["auditor"]]
    test_sha = (
        files[base.TEST_DIRECTORIES["auditor"] + "/test-receipt.json"]
        if seed == 1729
        else ready["files"][DIRECTORIES["auditor"] + "/test-receipt.json"]
    )
    for key, expected in {
        "seed": seed,
        "summary_sha256": summary_sha,
        "script_sha256": auditor_sha,
        "plan_sha256": base.PLAN_SHA,
        "execution_receipt_sha256": execution_sha,
        "test_receipt_sha256": test_sha,
    }.items():
        base.exact(audit[key], expected, "actual audit identity " + key)
    base.exact(audit["child"], report["child"], "audited child")
    base.exact(audit["gate"], gate, "audited gate")
    base.exact(
        audit["shared_readiness"],
        original_admission if seed == 1729 else repaired_admission,
        "actual audit admission",
    )
    if seed != 1729:
        for key, expected in {
            "audit_revision": REVISION,
            "amendment_sha256": AMENDMENT_SHA,
            "base_auditor_sha256": files[base.AUDITOR],
            "base_readiness": original_admission,
        }.items():
            base.exact(audit[key], expected, "repaired audit " + key)
    for key, expected in {
        "audit_sha256": audit_sha,
        "auditor_sha256": auditor_sha,
        "test_receipt_sha256": test_sha,
    }.items():
        base.exact(terminal[key], expected, "actual audit terminal " + key)
    completion_sha = (
        continuation(reader, ready, report, summary_sha, execution) if seed == 1731 else None
    )
    if seed == 1731:
        base.exact(
            audit["continuation_complete_sha256"], completion_sha, "audited continuation completion"
        )
    finish, audit_finish = base.chronology(
        execution, terminal, seed, summary_sha, execution_sha, previous
    )
    state = {
        "seed": seed,
        "summary_sha256": summary_sha,
        "execution_receipt_sha256": execution_sha,
        "audit_receipt_sha256": reader.inputs[terminal_path],
        "execution_finish": finish,
        "audit_finish": audit_finish,
    }
    entry = {
        "seed": seed,
        "summary_sha256": summary_sha,
        "audit_sha256": audit_sha,
        "audit_path": audit_path,
        "audit_receipt_path": terminal_path,
        "auditor_sha256": auditor_sha,
        "test_receipt_sha256": test_sha,
        "execution_receipt_sha256": execution_sha,
        "audit_receipt_sha256": state["audit_receipt_sha256"],
        "child": report["child"],
        "gate": gate,
    }
    if seed == 1731:
        entry.update(
            continuation_complete_sha256=completion_sha,
            execution_wrapper_sha256=report["execution_wrapper_sha256"],
        )
    return rows, entry, state


def aggregate(reader, ready, original, mean):
    base = reader.base
    entries, all_rows, identity_rows = [], [], None
    previous = None
    for seed in base.SEEDS:
        rows, entry, previous = seed_report(reader, ready, original, seed, mean, previous)
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
            base.exact(identities, identity_rows, "cross-seed teacher/query alignment")
        identity_rows = identities
        entries.append(entry)
        all_rows.extend(rows)
    reader.unchanged()
    checks = {str(entry["seed"]): entry["gate"]["primary_checks_passed"] for entry in entries}
    return {
        "complete": True,
        "acceptance": False,
        "campaign_version": CAMPAIGN,
        "evaluator": base.EVALUATOR,
        "plan_sha256": base.PLAN_SHA,
        "evidence_revision": REVISION,
        "amendment_sha256": AMENDMENT_SHA,
        "base_aggregator_sha256": BASE_SCRIPT_SHA,
        "base_readiness": {"path": BASE_READINESS, "sha256": BASE_READINESS_SHA},
        "shared_readiness": {"path": READINESS, "sha256": reader.inputs[READINESS]},
        "per_seed": entries,
        "all_three": {
            "aggregate": mean._totals(all_rows),
            "groups": {
                g: mean._totals([r for r in all_rows if r["group"] == g]) for g in base.GROUPS
            },
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
            "Mixed original and repaired audit routes; original failed audit is preserved.",
            "666 observations reuse 222 development queries and three parents.",
            "Metadata aggregation; scientific payload checks belong to seed audits.",
            "Provisional until independent aggregate audit and owner decision.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--readiness-sha256", required=True)
    parser.add_argument("--out", type=Path, default=Path(BASE + "/aggregate-v2.json"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    require(
        Path.cwd().resolve() == root and "torch" not in sys.modules,
        "Torch-free repository-root aggregation",
    )
    require(args.readiness.resolve() == root / READINESS, "fixed repaired readiness path")
    out = args.out.resolve()
    require(out == root / BASE / "aggregate-v2.json", "fixed repaired aggregate path")
    base = load_base(root)
    reader = Reader(root, base)
    ready, original = readiness(reader, args.readiness_sha256)
    require(ready["files"][SOURCES["aggregate"]] == sha(__file__), "actual aggregator source")
    mean = base.load_mean(reader)
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
    require(not any(p.exists() for p in [out, *snapshots]), "immutable aggregate v2 namespace")
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, data in (
        (snapshots[0], Path(__file__).read_bytes()),
        (snapshots[1], (root / READINESS).read_bytes()),
    ):
        with path.open("xb") as stream:
            stream.write(data)
    result = {
        "complete": False,
        "acceptance": False,
        "campaign_version": CAMPAIGN,
        "evidence_revision": REVISION,
    }
    try:
        result.update(aggregate(reader, ready, original, mean))
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        base.write(snapshots[2], reader.inputs)
        result.update(
            input_sha256=reader.inputs,
            script_sha256=sha(__file__),
            readiness_sha256=args.readiness_sha256,
            snapshot_sha256={p.relative_to(root).as_posix(): sha(p) for p in snapshots},
        )
        base.write(out, result)


if __name__ == "__main__":
    main()
