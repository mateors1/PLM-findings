"""Independent stdlib verification of saved 32000-update error geometry.

No primary arithmetic imports, checkpoint reads, training data or neural execution.
Historical independent arithmetic executes only after authenticating its bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import types
from pathlib import Path

CAMPAIGN = "bilinear-budget32000-error-geometry-v1"
BASE = f"runs/learning/{CAMPAIGN}"
PREP = "runs/learning/bilinear-budget32000-error-geometry-preparation-v1"
UPSTREAM = "runs/learning/bilinear-budget32000-v1"
PLAN_PATH = "docs/experiments/2026-09-26-bilinear-budget32000-error-geometry-plan.md"
PLAN_SHA = "d7208191cfdc49308d5e4e4c6bfc3473ed440a686fc5556caeb628a8f5ab52b4"
HELPER = "runs/learning/bilinear-error-geometry-v1/independent-audit.py"
HELPER_SHA = "6a7321f0ec0d0ddb644f04b617667e40b349170ab34dc6822a8ebbde45ac0cad"
PRIMARY_HELPER = "scripts/diagnose_bilinear_errors.py"
PRIMARY_HELPER_SHA = "66093bfbd06c8622b61535eeaffd0b09886c0cee05ec9eae71292a6e85925789"
PRIMARY = "scripts/diagnose_bilinear_budget32000_errors.py"
PRIMARY_TEST = "tests/unit/test_bilinear_budget32000_error_geometry.py"
PRIMARY_HELPER_TEST = "tests/unit/test_bilinear_error_geometry.py"
AUDITOR = f"{BASE}/independent-audit.py"
AUDITOR_TEST = "tests/unit/test_bilinear_budget32000_error_geometry_audit.py"
PINS = {
    "aggregate-v2.json": "8f4539583080275efe53409ce6881af3b50f2b89a092e0a21801ceafe6fd0708",
    "aggregate-audit-v2.json": "bacc8614e11afceee2080a3faf9850c1b70de92cf5cbd3ac912025da4edb0b3c",
    "decision-v2.json": "4a508abaf3fac7e2e678cd5604f73636241aabcb18c876076c134468257e44af",
}
READINESS = {
    "runs/learning/bilinear-budget32000-readiness-v1/readiness.json": (
        "5266fba8ab636590c0654db31b337ba8fd806361d9703c41a1b7260fcb5d766d"
    ),
    "runs/learning/bilinear-budget32000-readiness-v2/readiness.json": (
        "9df87cdafc02a3449b3b78d9bf48be93f0840da4ba64b35c21c09c286c96a69a"
    ),
}
SEEDS = (1729, 1730, 1731)
GROUPS = ("COLOR", "TYPE_dual", "TYPE_single")
AUDITOR_DEPS = {AUDITOR, AUDITOR_TEST, HELPER, PLAN_PATH, PRIMARY, PRIMARY_HELPER}
LIMIT = 512 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def same(actual, expected, message):
    require(encoded(actual) == encoded(expected), message)


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


class Reader:
    """An exact named-path allowlist; JSON manifests never authorize traversal."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.allowed = set()
        self.inputs = {}

    def admit(self, relative):
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "relative admitted path")
        resolved = (self.root / path).resolve()
        require(resolved.is_relative_to(self.root), "contained admitted path")
        self.allowed.add(path.as_posix())
        return path.as_posix()

    def read(self, relative, expected=None, json_value=True):
        require(relative in self.allowed, "unadmitted read: " + relative)
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), "contained read")
        payload = path.read_bytes()
        digest = sha_bytes(payload)
        if expected is not None:
            require(
                type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected), "SHA256 syntax"
            )
            same(digest, expected, "bound input: " + relative)
        require(relative not in self.inputs or self.inputs[relative] == digest, "changed input")
        self.inputs[relative] = digest
        return json.loads(payload) if json_value else payload

    def bound(self, relative, expected=None, json_value=True):
        self.admit(relative)
        return self.read(relative, expected, json_value)

    def unchanged(self):
        for relative, digest in list(self.inputs.items()):
            self.read(relative, digest, False)


def load_helper(reader):
    payload = reader.bound(HELPER, HELPER_SHA, False)
    module = types.ModuleType("authenticated_independent_geometry")
    module.__file__ = str(reader.root / HELPER)
    exec(compile(payload, module.__file__, "exec"), module.__dict__)
    return module


def accepted(aggregate, audit, decision):
    require(aggregate["complete"] is True, "completed aggregate")
    require(audit["complete"] is True and audit["audit_passed"] is True, "passed aggregate audit")
    same(audit["summary_sha256"], PINS["aggregate-v2.json"], "aggregate audit binding")
    same(decision["summary_sha256"], PINS["aggregate-v2.json"], "owner summary binding")
    same(decision["audit_sha256"], PINS["aggregate-audit-v2.json"], "owner audit binding")
    require(decision["evidence_accepted"] is True, "owner evidence acceptance")
    require(decision["fixed_quality_gate_passed"] is True, "owner quality gate")
    require(
        aggregate["acceptance"] is False and audit["acceptance"] is False,
        "raw acceptance preserved",
    )
    require(decision["model_promoted"] is False, "no promotion")
    for value in (aggregate, audit, decision):
        require(value["protected_test_used"] is False, "no protected partition")
    same(aggregate["all_three"], audit["all_three"], "audited aggregate metrics")
    same(aggregate["per_seed"], audit["per_seed"], "audited seed metadata")
    same([e["seed"] for e in aggregate["per_seed"]], list(SEEDS), "aggregate seed coverage")
    same([e["seed"] for e in decision["children"]], list(SEEDS), "decision seed coverage")


def input_digest(value, reader, relative):
    """Read an existing metadata digest without opening any indexed dependency."""
    keys = (relative, str((reader.root / relative).resolve()))
    found = [value[k] for k in keys if k in value]
    require(found and len(set(found)) == 1, "audited input identity: " + relative)
    return found[0]


def authenticate_children(reader):
    aggregate, audit, decision = [reader.bound(f"{UPSTREAM}/{n}", h) for n, h in PINS.items()]
    accepted(aggregate, audit, decision)
    admissions = [{"path": p, "sha256": h} for p, h in READINESS.items()]
    ready = [reader.bound(p, h) for p, h in READINESS.items()]
    for value in (aggregate, audit):
        same(value["base_readiness"], admissions[0], "original readiness identity")
        same(value["shared_readiness"], admissions[1], "repaired readiness identity")
    children, digests, provenance = {}, {}, []
    for entry, owner in zip(aggregate["per_seed"], decision["children"], strict=True):
        seed = entry["seed"]
        folder = f"{UPSTREAM}/seed-{seed}"
        audit_path = folder + (
            "/independent-audit.json" if seed == 1729 else "/independent-audit-v2.json"
        )
        same(entry["audit_path"], audit_path, "actual original/v2 audit route")
        summary = reader.bound(folder + "/summary.json", entry["summary_sha256"])
        checked = reader.bound(audit_path, entry["audit_sha256"])
        require(summary["complete"] is True, "complete seed summary")
        require(
            checked["complete"] is True and checked["audit_passed"] is True, "passed seed audit"
        )
        same(summary["seed"], seed, "summary seed")
        same(checked["seed"], seed, "audit seed")
        same(checked["summary_sha256"], entry["summary_sha256"], "seed audit summary binding")
        same(owner["summary_sha256"], entry["summary_sha256"], "owner seed summary")
        same(owner["audit_sha256"], entry["audit_sha256"], "owner seed audit")
        same(summary["child"], checked["child"], "summary versus seed-audit child metrics")
        same(summary["child"], entry["child"], "aggregate child metrics")
        same(summary["gate"], entry["gate"], "aggregate seed gate")
        same(checked["gate"], entry["gate"], "audited seed gate")
        require(entry["gate"]["primary_checks_passed"] is True, "seed gate passed")
        require(summary["final_identity_check"] is True, "final identity verified")
        revision = 0 if seed == 1729 else 1
        suffix = "" if seed == 1729 else "-v2"
        same(summary["shared_readiness"], admissions[0], "scientific base readiness")
        same(checked["shared_readiness"], admissions[revision], "audit revision readiness")
        auditor_path = f"{UPSTREAM}/independent-audit{suffix}.py"
        test_path = (
            f"runs/learning/bilinear-budget32000-auditor-tests-v{revision + 1}/test-receipt.json"
        )
        same(checked["script_sha256"], entry["auditor_sha256"], "admitted seed auditor")
        same(
            checked["script_sha256"], ready[revision]["files"][auditor_path], "sealed seed auditor"
        )
        same(checked["test_receipt_sha256"], entry["test_receipt_sha256"], "admitted seed tests")
        same(
            checked["test_receipt_sha256"], ready[revision]["files"][test_path], "sealed seed tests"
        )
        checkpoint = summary["artifact_sha256"]["checkpoint-final.pt"]
        same(
            checkpoint,
            checked["checkpoint"]["child_checkpoint_sha256"],
            "checkpoint audit identity",
        )
        same(checkpoint, owner["checkpoint_sha256"], "checkpoint owner identity")
        same(
            owner["checkpoint_path"],
            folder + "/checkpoint-final.pt",
            "checkpoint metadata location",
        )
        terminal_path = folder + f"/audit-execution-receipt{suffix}.json"
        same(entry["audit_receipt_path"], terminal_path, "audit terminal route")
        execution = reader.bound(
            folder + "/execution-receipt.json", entry["execution_receipt_sha256"]
        )
        terminal = reader.bound(terminal_path, entry["audit_receipt_sha256"])
        for record in (execution, terminal):
            require(
                type(record["exit_code"]) is int and record["exit_code"] == 0,
                "upstream terminal exit",
            )
            require(
                record["terminal_completion_observed_by_primary"] is True,
                "observed upstream terminal",
            )
            same(record["seed"], seed, "terminal seed")
            same(record["summary_sha256"], entry["summary_sha256"], "terminal seed summary")
        for key in ("audit_sha256", "execution_receipt_sha256", "test_receipt_sha256"):
            same(terminal[key], entry[key], "upstream terminal " + key)
        same(terminal["auditor_sha256"], entry["auditor_sha256"], "upstream auditor terminal")
        same(
            terminal["execution_receipt_sha256"],
            checked["execution_receipt_sha256"],
            "seed audit execution binding",
        )
        same(terminal["cuda_visible_devices"], "-1", "upstream CPU audit")
        continuation = None
        if seed == 1731:
            complete_hash = entry["continuation_complete_sha256"]
            complete = reader.bound(folder + "/continuation-complete.json", complete_hash)
            require(
                complete["complete"] is True
                and complete["root_observed_terminal_required"] is True,
                "continuation completion",
            )
            same(complete["seed"], seed, "continuation seed")
            same(complete["summary_sha256"], entry["summary_sha256"], "continuation summary")
            same(
                execution["continuation_complete_sha256"],
                complete_hash,
                "observed continuation hash",
            )
            same(
                checked["continuation_complete_sha256"], complete_hash, "audited continuation hash"
            )
            for key in (
                "execution_wrapper_sha256",
                "execution_revision",
                "amendment_sha256",
                "continuation_readiness",
            ):
                same(complete[key], summary[key], "continuation " + key)
            same(summary["continuation_readiness"], admissions[1], "continuation readiness")
            same(
                summary["execution_wrapper_sha256"],
                ready[1]["files"]["scripts/refit_bilinear_budget32000_continuation_v2.py"],
                "sealed continuation source",
            )
            continuation = {"completion_sha256": complete_hash, **complete}
        child_path = folder + "/child.json"
        child_hash = summary["artifact_sha256"]["child.json"]
        same(
            input_digest(checked["input_sha256"], reader, child_path),
            child_hash,
            "audited child hash",
        )
        same(
            input_digest(aggregate["input_sha256"], reader, child_path),
            child_hash,
            "aggregated child hash",
        )
        child = reader.bound(child_path, child_hash)
        require(
            child["complete"] is True
            and type(child["query_count"]) is int
            and child["query_count"] == 222,
            "complete child",
        )
        same(child["product_token_ids"], list(range(1024, 2049)), "full product columns")
        require(len(child["responses"]) == 222, "full ordered validation rows")
        same(child["aggregate"], summary["child"]["aggregate"], "child aggregate identity")
        same(child["groups"], summary["child"]["groups"], "child group identity")
        same(owner["exact_count"], child["aggregate"]["exact_count"], "owner exact count")
        children[seed], digests[seed] = child, child_hash
        provenance.append(
            {
                "seed": seed,
                "summary_sha256": entry["summary_sha256"],
                "audit_path": audit_path,
                "audit_sha256": entry["audit_sha256"],
                "child_sha256": child_hash,
                "checkpoint_sha256": checkpoint,
                "audit_receipt_sha256": entry["audit_receipt_sha256"],
                "execution_receipt_sha256": entry["execution_receipt_sha256"],
                "checkpoint_opened": False,
                "audit_readiness": checked["shared_readiness"],
                "continuation": continuation,
            }
        )
    return (
        children,
        digests,
        {
            "accepted_campaign": "bilinear-budget32000-v1",
            "aggregate_sha256": PINS["aggregate-v2.json"],
            "audit_sha256": PINS["aggregate-audit-v2.json"],
            "decision_sha256": PINS["decision-v2.json"],
            "all_three": aggregate["all_three"],
            "per_seed": provenance,
        },
    )


def execution_receipt(value, summary_hash, primary_hash, receipt_hash, stdout_hash):
    require(type(value["exit_code"]) is int and value["exit_code"] == 0, "successful terminal exit")
    require(
        value["terminal_completion_observed_by_primary"] is True, "observed terminal completion"
    )
    for key, expected in (
        ("summary_sha256", summary_hash),
        ("script_sha256", primary_hash),
        ("test_receipt_sha256", receipt_hash),
        ("stdout_sha256", stdout_hash),
    ):
        same(value[key], expected, "terminal " + key)


def compare_results(helper, children, digests, saved_reports, summary):
    helper.align(children)
    reports, checks = {}, []
    for seed in SEEDS:
        rows = children[seed]["responses"]
        require(
            len({(r["prompt_ids"][1], r["dimension"]) for r in rows}) == 222, "unique token queries"
        )
        for index, row in enumerate(rows):
            require(type(row["index"]) is int and row["index"] == index, "typed sequential index")
            require(type(row["subject"]) is str and bool(row["subject"]), "subject label")
            for key in ("expected_set_ids", "selected_set_ids"):
                require(
                    type(row[key]) is list and all(type(i) is int for i in row[key]),
                    "typed memberships",
                )
        same(
            {g: sum(r["group"] == g for r in rows) for g in GROUPS},
            {"COLOR": 103, "TYPE_dual": 68, "TYPE_single": 51},
            "validation group coverage",
        )
        reports[seed] = helper.seed_report(seed, children[seed], digests[seed])
        same(saved_reports[seed], reports[seed], "independent full seed reductions")
        checks.append(
            {
                "seed": seed,
                "child_sha256": digests[seed],
                "aggregate": reports[seed]["aggregate"],
                "groups": reports[seed]["groups"],
            }
        )
    fresh = helper.pool(reports, (1730, 1731))
    pooled = helper.pool(reports, SEEDS)
    shared = helper.overlap(reports)
    same(summary["fresh_two"], fresh, "two-seed descriptive subset")
    same(summary["all_three"], pooled, "three-seed pooled counts")
    same(summary["overlap"], shared, "aligned failure overlap")
    return checks, fresh, pooled, shared


def test_receipt(reader, role, digest):
    primary = role == "primary"
    path = f"{PREP}/" + ("test-receipt.json" if primary else "auditor-test-receipt.json")
    stdout = f"{PREP}/" + ("stdout.txt" if primary else "auditor-stdout.txt")
    script = PRIMARY if primary else AUDITOR
    dependencies = (
        {PRIMARY, PRIMARY_TEST, PRIMARY_HELPER, PRIMARY_HELPER_TEST, PLAN_PATH}
        if primary
        else AUDITOR_DEPS
    )
    value = reader.bound(path, digest)
    require(
        value["passed"] is True
        and value["gpu_used"] is False
        and value["diagnostic_executed"] is False,
        "synthetic test scope",
    )
    require(
        type(value["exit_code"]) is int
        and value["exit_code"] == 0
        and value["terminal_completion_observed"] is True,
        "observed test terminal",
    )
    require(type(value["tests_passed"]) is int and value["tests_passed"] > 0, "positive test count")
    require(type(value["tests_skipped"]) is int and value["tests_skipped"] == 0, "no skipped tests")
    same(sorted(value["test_dependencies"]), sorted(dependencies), "exact test dependencies")
    same(
        value["tested_script_sha256"], value["test_dependencies"][script], "tested script identity"
    )
    same(value["test_dependencies"][PLAN_PATH], PLAN_SHA, "tested plan")
    helper_path, helper_sha = (
        (PRIMARY_HELPER, PRIMARY_HELPER_SHA) if primary else (HELPER, HELPER_SHA)
    )
    same(value["test_dependencies"][helper_path], helper_sha, "tested scientific helper")
    for relative in sorted(dependencies):
        reader.bound(relative, value["test_dependencies"][relative], False)
    same(value["stdout"]["path"], stdout, "test stdout location")
    reader.bound(stdout, value["stdout"]["sha256"], False)
    return value


def audit(root, primary_receipt_hash, auditor_receipt_hash):
    root = Path(root).resolve()
    require(not (root / BASE / "failure.json").exists(), "primary failure artifact present")
    reader = Reader(root)
    reader.bound(PLAN_PATH, PLAN_SHA, False)
    children, digests, upstream = authenticate_children(reader)
    primary_receipt = test_receipt(reader, "primary", primary_receipt_hash)
    primary_inputs = dict(reader.inputs)
    summary_path = f"{BASE}/summary.json"
    summary = reader.bound(summary_path)
    summary_hash = reader.inputs[summary_path]
    require(
        summary["complete"] is True and summary["acceptance"] is False,
        "completed unaccepted primary",
    )
    same(summary["campaign_version"], CAMPAIGN, "campaign identity")
    same(summary["plan_sha256"], PLAN_SHA, "declared plan")
    same(summary["helper_sha256"], PRIMARY_HELPER_SHA, "unchanged primary scientific helper")
    same(summary["script_sha256"], primary_receipt["tested_script_sha256"], "tested primary script")
    same(summary["test_receipt_sha256"], primary_receipt_hash, "primary receipt admission")
    same(summary["input_sha256"], primary_inputs, "exact primary input inventory")
    same(summary["upstream"], upstream, "accepted upstream chain")
    source_snapshots = {
        "script.py": PRIMARY,
        "helper.py": PRIMARY_HELPER,
        "plan.md": PLAN_PATH,
        "test.py": PRIMARY_TEST,
        "helper-test.py": PRIMARY_HELPER_TEST,
        "test-receipt.json": f"{PREP}/test-receipt.json",
        "test-stdout.txt": f"{PREP}/stdout.txt",
    }
    expected_snapshots = {
        f"{BASE}/summary.{suffix}": primary_inputs[path]
        for suffix, path in source_snapshots.items()
    }
    expected_snapshots[f"{BASE}/summary.inputs.json"] = sha_bytes(encoded(primary_inputs))
    same(summary["snapshot_sha256"], expected_snapshots, "exact primary snapshots")
    for path, digest in expected_snapshots.items():
        reader.bound(path, digest, False)
    terminal_path = f"{BASE}/execution-receipt.json"
    terminal = reader.bound(terminal_path)
    reader.bound(f"{BASE}/primary-stdout.txt", terminal["stdout_sha256"], False)
    execution_receipt(
        terminal,
        summary_hash,
        summary["script_sha256"],
        primary_receipt_hash,
        reader.inputs[f"{BASE}/primary-stdout.txt"],
    )
    auditor_receipt = test_receipt(reader, "auditor", auditor_receipt_hash)
    helper = load_helper(reader)
    same([e["seed"] for e in summary["seed_artifacts"]], list(SEEDS), "output seed coverage")
    saved = {}
    for entry in summary["seed_artifacts"]:
        seed = entry["seed"]
        path = f"{BASE}/seed-{seed}.json"
        same(entry["path"], path, "fixed seed result location")
        saved[seed] = reader.bound(path, entry["sha256"])
        require((root / path).stat().st_size <= LIMIT, "bounded seed report")
    per_seed, fresh, pooled, overlap = compare_results(helper, children, digests, saved, summary)
    rows = [row for seed in SEEDS for row in children[seed]["responses"]]
    same(
        upstream["all_three"],
        {
            "aggregate": helper.source_totals(rows),
            "groups": {
                g: helper.source_totals([r for r in rows if r["group"] == g]) for g in GROUPS
            },
        },
        "recomputed upstream aggregate and group metrics",
    )
    require((root / summary_path).stat().st_size <= LIMIT, "bounded primary summary")
    reader.unchanged()
    require(not (root / BASE / "failure.json").exists(), "primary failed during audit")
    return {
        "campaign_version": CAMPAIGN,
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": summary_hash,
        "script_sha256": auditor_receipt["tested_script_sha256"],
        "independent_helper_sha256": HELPER_SHA,
        "plan_sha256": PLAN_SHA,
        "primary_script_sha256": summary["script_sha256"],
        "primary_test_receipt_sha256": primary_receipt_hash,
        "test_receipt_sha256": auditor_receipt_hash,
        "execution_receipt_sha256": reader.inputs[terminal_path],
        "input_sha256": reader.inputs,
        "upstream": upstream,
        "per_seed": per_seed,
        "fresh_two": fresh,
        "all_three": pooled,
        "overlap": overlap,
        "protected_test_used": False,
        "new_checkpoint_created": False,
        "threshold_selected": False,
        "limitations": [
            "Independent reconstruction from saved validation scores, not a neural replay.",
            "The 666 observations reuse 222 queries across existing parents; "
            "no generalization claim.",
            "Label-informed intervals describe saved geometry; "
            "no threshold or new predictions selected.",
            "The fresh_two field is the descriptive 1730/1731 subset; "
            "all three campaign fits were fresh residual fits.",
            "Checkpoint identities are metadata only; "
            "no weights, graph, training or protected payloads read.",
        ],
    }


def write(path, value):
    payload = encoded(value)
    require(len(payload) <= LIMIT, "bounded output")
    with Path(path).open("xb") as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--primary-test-receipt-sha256", required=True)
    parser.add_argument("--audit-test-receipt-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    require(args.summary.resolve() == root / BASE / "summary.json", "fixed summary path")
    out = root / BASE / "independent-audit.json"
    require(not out.exists(), "immutable audit output")
    result = audit(root, args.primary_test_receipt_sha256, args.audit_test_receipt_sha256)
    require("torch" not in sys.modules, "stdlib-only audit process")
    write(out, result)
    print(
        json.dumps(
            {
                "audit_passed": True,
                "audit_sha256": sha_bytes(out.read_bytes()),
                "summary_sha256": result["summary_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
