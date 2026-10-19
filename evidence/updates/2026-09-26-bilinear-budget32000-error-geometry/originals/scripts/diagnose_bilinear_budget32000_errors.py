"""Authenticated current-only saved-score geometry; no neural execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

CAMPAIGN = "bilinear-budget32000-error-geometry-v1"
PLAN_PATH = "docs/experiments/2026-09-26-bilinear-budget32000-error-geometry-plan.md"
PLAN = "d7208191cfdc49308d5e4e4c6bfc3473ed440a686fc5556caeb628a8f5ab52b4"
SOURCE = "scripts/diagnose_bilinear_budget32000_errors.py"
HELPER_PATH = "scripts/diagnose_bilinear_errors.py"
HELPER = "66093bfbd06c8622b61535eeaffd0b09886c0cee05ec9eae71292a6e85925789"
TEST = "tests/unit/test_bilinear_budget32000_error_geometry.py"
HELPER_TEST = "tests/unit/test_bilinear_error_geometry.py"
PREPARATION = "runs/learning/bilinear-budget32000-error-geometry-preparation-v1"
RECEIPT = PREPARATION + "/test-receipt.json"
STDOUT = PREPARATION + "/stdout.txt"
DEPENDENCIES = {SOURCE, HELPER_PATH, TEST, HELPER_TEST, PLAN_PATH}
UPSTREAM = "runs/learning/bilinear-budget32000-v1"
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
GROUPS = {"COLOR": 103, "TYPE_single": 51, "TYPE_dual": 68}
LIMIT = 512 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def evidence_paths():
    paths = {PLAN_PATH, SOURCE, HELPER_PATH, TEST, HELPER_TEST, RECEIPT, STDOUT, *READINESS}
    paths.update(UPSTREAM + "/" + n for n in PINS)
    for seed in SEEDS:
        folder = f"{UPSTREAM}/seed-{seed}/"
        suffix = "" if seed == 1729 else "-v2"
        paths.update(
            folder + n
            for n in (
                "summary.json",
                "child.json",
                "execution-receipt.json",
                f"independent-audit{suffix}.json",
                f"audit-execution-receipt{suffix}.json",
            )
        )
    paths.add(UPSTREAM + "/seed-1731/continuation-complete.json")
    return paths


class Reader:
    def __init__(self, root):
        self.root, self.inputs = root.resolve(), {}

    def bind(self, relative, digest):
        require(
            type(relative) is str and relative in evidence_paths(), "explicit input path required"
        )
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), "input path escape")
        require(
            type(digest) is str
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            "SHA256 required",
        )
        require(
            relative not in self.inputs or self.inputs[relative] == digest,
            "conflicting input identity",
        )
        require(sha(path) == digest, "input identity: " + relative)
        self.inputs[relative] = digest
        return path

    def read(self, relative, digest):
        require(relative.endswith(".json"), "JSON input required")
        value = json.loads(self.bind(relative, digest).read_text(encoding="utf-8"))
        require(type(value) is dict, "JSON object required")
        return value

    def unchanged(self):
        require(
            all(sha(self.root / name) == digest for name, digest in self.inputs.items()),
            "final input identity drift",
        )


def load_helper(reader):
    path = reader.bind(HELPER_PATH, HELPER)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == HELPER, "helper bytes before execution")
    helper = ModuleType("authenticated_bilinear_geometry")
    helper.__file__ = str(path)
    exec(compile(data, str(path), "exec"), helper.__dict__)
    return helper


def test_receipt(reader, path, digest):
    require(path.resolve() == reader.root / RECEIPT, "fixed test receipt path")
    receipt = reader.read(RECEIPT, digest)
    require(
        receipt.get("passed") is True
        and receipt.get("gpu_used") is False
        and receipt.get("diagnostic_executed") is False
        and type(receipt.get("exit_code")) is int
        and receipt["exit_code"] == 0
        and receipt.get("terminal_completion_observed") is True
        and type(receipt.get("tests_passed")) is int
        and receipt["tests_passed"] > 0
        and type(receipt.get("tests_skipped")) is int
        and receipt["tests_skipped"] == 0,
        "observed synthetic test receipt",
    )
    deps = receipt.get("test_dependencies")
    require(type(deps) is dict and set(deps) == DEPENDENCIES, "exact test dependency inventory")
    require(
        receipt.get("tested_script_sha256") == deps[SOURCE]
        and deps[HELPER_PATH] == HELPER
        and deps[PLAN_PATH] == PLAN,
        "tested source and declarations",
    )
    stdout = receipt.get("stdout")
    require(
        type(stdout) is dict and set(stdout) == {"path", "sha256"} and stdout["path"] == STDOUT,
        "fixed test stdout",
    )
    for name, pin in deps.items():
        reader.bind(name, pin)
    reader.bind(STDOUT, stdout["sha256"])
    return receipt


def child_schema(child, helper):
    require(
        child.get("complete") is True
        and type(child.get("query_count")) is int
        and child["query_count"] == 222
        and child.get("product_token_ids") == list(range(1024, 2049))
        and all(type(i) is int for i in child["product_token_ids"])
        and len(child["responses"]) == 222,
        "complete ordered product columns",
    )
    queries = set()
    for index, row in enumerate(child["responses"]):
        require(type(row["index"]) is int and row["index"] == index, "typed ordered indexes")
        prompt = row["prompt_ids"]
        require(
            type(prompt) is list
            and len(prompt) == 5
            and all(type(i) is int for i in prompt)
            and prompt[0] == 1
            and 1024 <= prompt[1] <= 2048
            and prompt[3:] == [34, 5],
            "typed protocol prompt",
        )
        require(
            row["dimension"] in ("TYPE", "COLOR")
            and prompt[2] == {"TYPE": 32, "COLOR": 33}[row["dimension"]]
            and row["group"] in GROUPS
            and (row["group"] == "COLOR") == (row["dimension"] == "COLOR")
            and type(row["subject"]) is str
            and bool(row["subject"]),
            "query identity",
        )
        key = (prompt[1], prompt[2])
        require(key not in queries, "unique query identities")
        queries.add(key)
        for name in ("expected_set_ids", "selected_set_ids"):
            ids = row[name]
            require(
                type(ids) is list
                and all(type(i) is int and 1024 <= i <= 2048 for i in ids)
                and ids == sorted(set(ids))
                and prompt[1] not in ids,
                "typed nonself memberships",
            )
        require(
            bool(row["expected_set_ids"]) and len(row["expected_set_ids"]) < 1024,
            "nonempty membership sides",
        )
        require(len(row["symmetric_relation_logits"]) == 1025, "score columns")
        for score in row["symmetric_relation_logits"]:
            helper._finite_fp32(score)
    require(
        dict(Counter(row["group"] for row in child["responses"])) == GROUPS
        and set(child["groups"]) == set(GROUPS),
        "fixed validation group coverage",
    )


def preflight(reader, helper):
    reader.bind(PLAN_PATH, PLAN)
    aggregate, audit, decision = [
        reader.read(UPSTREAM + "/" + name, pin) for name, pin in PINS.items()
    ]
    require(
        aggregate["complete"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == PINS["aggregate-v2.json"]
        and decision["summary_sha256"] == PINS["aggregate-v2.json"]
        and decision["audit_sha256"] == PINS["aggregate-audit-v2.json"]
        and decision["evidence_accepted"] is True,
        "accepted upstream chain",
    )
    require(
        aggregate["campaign_version"] == decision["campaign_version"] == "bilinear-budget32000-v1"
        and aggregate["all_three"] == audit["all_three"]
        and aggregate["gate"] == audit["gate"],
        "audited aggregate identity",
    )
    require(
        [e["seed"] for e in aggregate["per_seed"]]
        == [e["seed"] for e in audit["per_seed"]]
        == [e["seed"] for e in decision["children"]]
        == list(SEEDS),
        "ordered three-seed evidence",
    )
    ready = [reader.read(path, pin) for path, pin in READINESS.items()]
    admissions = [{"path": path, "sha256": pin} for path, pin in READINESS.items()]
    require(
        aggregate["shared_readiness"] == audit["shared_readiness"] == admissions[1]
        and aggregate["base_readiness"] == audit["base_readiness"] == admissions[0],
        "upstream readiness identity",
    )
    children, hashes, provenance = {}, {}, []
    for entry, audited, owner in zip(
        aggregate["per_seed"], audit["per_seed"], decision["children"], strict=True
    ):
        seed = entry["seed"]
        require(type(seed) is int, "typed seed")
        folder = f"{UPSTREAM}/seed-{seed}/"
        suffix = "" if seed == 1729 else "-v2"
        audit_path, terminal_path = (
            folder + f"independent-audit{suffix}.json",
            folder + f"audit-execution-receipt{suffix}.json",
        )
        require(
            entry["audit_path"] == audit_path and entry["audit_receipt_path"] == terminal_path,
            "fixed audit routing",
        )
        summary = reader.read(folder + "summary.json", entry["summary_sha256"])
        checked = reader.read(audit_path, entry["audit_sha256"])
        require(
            summary["complete"] is True
            and summary["final_identity_check"] is True
            and checked["complete"] is True
            and checked["audit_passed"] is True
            and summary["seed"] == checked["seed"] == seed
            and checked["summary_sha256"]
            == audited["summary_sha256"]
            == owner["summary_sha256"]
            == entry["summary_sha256"]
            and audited["audit_sha256"] == owner["audit_sha256"] == entry["audit_sha256"],
            "completed audited seed chain",
        )
        require(
            summary["child"] == checked["child"] == entry["child"] == audited["child"]
            and summary["gate"] == checked["gate"] == entry["gate"] == audited["gate"],
            "audited child metrics",
        )
        version = 0 if seed == 1729 else 1
        auditor_name = UPSTREAM + f"/independent-audit{suffix}.py"
        test_name = (
            f"runs/learning/bilinear-budget32000-auditor-tests-v{version + 1}/test-receipt.json"
        )
        require(
            summary["shared_readiness"] == admissions[0]
            and checked["shared_readiness"] == admissions[version]
            and checked["script_sha256"]
            == entry["auditor_sha256"]
            == ready[version]["files"][auditor_name]
            and checked["test_receipt_sha256"]
            == entry["test_receipt_sha256"]
            == ready[version]["files"][test_name],
            "actual audited source admission",
        )
        checkpoint = summary["artifact_sha256"]["checkpoint-final.pt"]
        require(
            owner["checkpoint_path"] == folder + "checkpoint-final.pt"
            and owner["checkpoint_sha256"]
            == checked["checkpoint"]["child_checkpoint_sha256"]
            == checkpoint,
            "metadata-only checkpoint identity",
        )
        execution = reader.read(
            folder + "execution-receipt.json", entry["execution_receipt_sha256"]
        )
        terminal = reader.read(terminal_path, entry["audit_receipt_sha256"])
        for observed in (execution, terminal):
            require(
                type(observed["exit_code"]) is int
                and observed["exit_code"] == 0
                and observed["terminal_completion_observed_by_primary"] is True
                and observed["seed"] == seed
                and observed["summary_sha256"] == entry["summary_sha256"],
                "observed successful upstream terminal",
            )
        require(
            terminal["audit_sha256"] == entry["audit_sha256"]
            and terminal["execution_receipt_sha256"]
            == checked["execution_receipt_sha256"]
            == entry["execution_receipt_sha256"]
            and terminal["auditor_sha256"] == entry["auditor_sha256"]
            and terminal["test_receipt_sha256"] == entry["test_receipt_sha256"]
            and terminal["cuda_visible_devices"] == "-1",
            "bound CPU audit terminal",
        )
        continuation = None
        if seed == 1731:
            completion_sha = entry["continuation_complete_sha256"]
            completion = reader.read(folder + "continuation-complete.json", completion_sha)
            require(
                completion["complete"] is True
                and completion["seed"] == seed
                and completion["summary_sha256"] == entry["summary_sha256"]
                and completion["root_observed_terminal_required"] is True
                and execution["continuation_complete_sha256"]
                == checked["continuation_complete_sha256"]
                == completion_sha,
                "observed continuation completion",
            )
            for key in (
                "execution_wrapper_sha256",
                "execution_revision",
                "amendment_sha256",
                "continuation_readiness",
            ):
                require(completion[key] == summary[key], "continuation identity: " + key)
            require(
                summary["execution_wrapper_sha256"]
                == ready[1]["files"]["scripts/refit_bilinear_budget32000_continuation_v2.py"]
                and summary["continuation_readiness"] == admissions[1],
                "actual continuation source",
            )
            continuation = {"completion_sha256": completion_sha, **completion}
        child_hash = summary["artifact_sha256"]["child.json"]
        child_path = folder + "child.json"
        require(
            checked["input_sha256"].get(str((reader.root / child_path).resolve())) == child_hash,
            "audited saved child hash",
        )
        child = reader.read(child_path, child_hash)
        require(
            child["aggregate"] == summary["child"]["aggregate"]
            and child["groups"] == summary["child"]["groups"],
            "saved child metrics identity",
        )
        child_schema(child, helper)
        children[seed], hashes[seed] = child, child_hash
        provenance.append(
            {
                "seed": seed,
                "summary_sha256": entry["summary_sha256"],
                "audit_path": audit_path,
                "audit_sha256": entry["audit_sha256"],
                "audit_receipt_sha256": entry["audit_receipt_sha256"],
                "execution_receipt_sha256": entry["execution_receipt_sha256"],
                "child_sha256": child_hash,
                "checkpoint_sha256": checkpoint,
                "checkpoint_opened": False,
                "audit_readiness": checked["shared_readiness"],
                "continuation": continuation,
            }
        )
    helper._alignment(children)
    reader.unchanged()
    return (
        children,
        hashes,
        {
            "accepted_campaign": "bilinear-budget32000-v1",
            "aggregate_sha256": PINS["aggregate-v2.json"],
            "audit_sha256": PINS["aggregate-audit-v2.json"],
            "decision_sha256": PINS["decision-v2.json"],
            "all_three": aggregate["all_three"],
            "per_seed": provenance,
        },
    )


def write_exclusive(path, content):
    data = content if isinstance(content, bytes) else encode(content)
    require(len(data) <= LIMIT, "output exceeds 512 KiB")
    with path.open("xb") as stream:
        stream.write(data)


def run(root, receipt_path, receipt_sha, preflight_only=False):
    require(Path.cwd().resolve() == root and "torch" not in sys.modules, "stdlib root boundary")
    reader = Reader(root)
    helper = load_helper(reader)
    test_receipt(reader, receipt_path, receipt_sha)
    children, child_hashes, provenance = preflight(reader, helper)
    if preflight_only:
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "new_reductions_executed": False,
                    "input_count": len(reader.inputs),
                }
            )
        )
        return
    out = root / "runs/learning" / CAMPAIGN
    snapshot_sources = {
        "summary.script.py": SOURCE,
        "summary.helper.py": HELPER_PATH,
        "summary.plan.md": PLAN_PATH,
        "summary.test.py": TEST,
        "summary.helper-test.py": HELPER_TEST,
        "summary.test-receipt.json": RECEIPT,
        "summary.test-stdout.txt": STDOUT,
    }
    snapshots = {
        name: (root / relative).read_bytes() for name, relative in snapshot_sources.items()
    }
    require(
        all(
            hashlib.sha256(snapshots[name]).hexdigest() == reader.inputs[relative]
            for name, relative in snapshot_sources.items()
        ),
        "snapshot tested input identity",
    )
    snapshots["summary.inputs.json"] = encode(reader.inputs)
    names = ["summary.json", "failure.json", *(f"seed-{s}.json" for s in SEEDS), *snapshots]
    require(all(not (out / name).exists() for name in names), "immutable diagnostic namespace")
    summary = {
        "campaign_version": CAMPAIGN,
        "complete": False,
        "acceptance": False,
        "script_sha256": reader.inputs[SOURCE],
        "helper_sha256": HELPER,
        "plan_sha256": PLAN,
        "test_receipt_sha256": receipt_sha,
        "input_sha256": reader.inputs,
        "snapshot_sha256": {
            (out / n).relative_to(root).as_posix(): hashlib.sha256(v).hexdigest()
            for n, v in snapshots.items()
        },
        "seed_artifacts": [],
        "upstream": provenance,
        "limitations": [
            "Posthoc descriptive saved-score reductions; "
            "no threshold selection or alternative predictions.",
            "666 observations reuse 222 development queries; fresh_two groups seeds1730/1731. "
            "All three are current 32000 fits.",
            "No weights, training records, graph, protected partition, Torch or GPU accessed.",
            "No model promotion or change to the 41-checkpoint inventory; "
            "independent audit and owner acceptance remain required.",
            "Exclusive batch output writes are not atomic; "
            "any failure blocks publication until separately audited.",
        ],
    }
    try:
        reports = {seed: helper._seed(seed, children[seed], child_hashes[seed]) for seed in SEEDS}
        summary.update(
            fresh_two=helper._pool(reports, [1730, 1731]),
            all_three=helper._pool(reports, list(SEEDS)),
            overlap=helper._overlap(reports),
        )
        helper._reconcile(summary["all_three"]["aggregate"], provenance["all_three"]["aggregate"])
        for group in GROUPS:
            helper._reconcile(
                summary["all_three"]["groups"][group], provenance["all_three"]["groups"][group]
            )
        outputs = dict(snapshots)
        for seed, report in reports.items():
            name = f"seed-{seed}.json"
            data = encode(report)
            outputs[name] = data
            summary["seed_artifacts"].append(
                {
                    "seed": seed,
                    "path": (out / name).relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        reader.unchanged()
        require("torch" not in sys.modules, "stdlib execution boundary")
        summary["complete"] = True
        outputs["summary.json"] = encode(summary)
        require(
            all(len(data) <= LIMIT for data in outputs.values()), "prevalidated output file bounds"
        )
        out.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            write_exclusive(out / name, data)
        reader.unchanged()
        require(
            all(
                sha(out / name) == hashlib.sha256(data).hexdigest()
                for name, data in outputs.items()
            ),
            "written output identity",
        )
    except BaseException as exc:
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "campaign_version": CAMPAIGN,
            "complete": False,
            "acceptance": False,
            "script_sha256": reader.inputs[SOURCE],
            "plan_sha256": PLAN,
            "error": repr(exc),
            "primary_success_invalidated": True,
        }
        write_exclusive(out / "failure.json", failure)
        raise
    print(
        json.dumps(
            {"complete": True, "acceptance": False, "summary_sha256": sha(out / "summary.json")}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path, required=True)
    parser.add_argument("--test-receipt-sha256", required=True)
    args = parser.parse_args()
    run(
        Path(__file__).resolve().parents[1],
        args.test_receipt.resolve(),
        args.test_receipt_sha256,
        args.preflight_only,
    )


if __name__ == "__main__":
    main()
