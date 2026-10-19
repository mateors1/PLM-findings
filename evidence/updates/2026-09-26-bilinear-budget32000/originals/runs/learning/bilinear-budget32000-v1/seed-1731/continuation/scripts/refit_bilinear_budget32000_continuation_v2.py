"""Explicit evidence-v2 admission for the never-run seed1731; unchanged science."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

CAMPAIGN = "bilinear-budget32000-v1"
REVISION = "bilinear-budget32000-evidence-v2"
EXECUTION = "bilinear-budget32000-continuation-v2"
AMENDMENT_PATH = "docs/experiments/2026-09-26-bilinear-budget32000-evidence-repair-v2.md"
AMENDMENT = "90aa30088980aeffd999fe6fe4e7349220d163b196ad4eb28f06193b9ce2991c"
BASE_SOURCE = "scripts/refit_bilinear_budget32000.py"
BASE_SHA = "ee1543bcd39fb1d6f5bfd93bfbbcd1989198acb350cf057d2e4e6be8314e21c1"
BASE_READY = "runs/learning/bilinear-budget32000-readiness-v1/readiness.json"
BASE_READY_SHA = "5266fba8ab636590c0654db31b337ba8fd806361d9703c41a1b7260fcb5d766d"
READINESS = "runs/learning/bilinear-budget32000-readiness-v2/readiness.json"
SOURCES = {
    "runner": "scripts/refit_bilinear_budget32000_continuation_v2.py",
    "auditor": "runs/learning/bilinear-budget32000-v1/independent-audit-v2.py",
    "aggregate": "scripts/aggregate_bilinear_budget32000_v2.py",
}
TESTS = {
    "runner": "tests/unit/test_bilinear_budget32000_continuation_v2.py",
    "auditor": "tests/unit/test_bilinear_budget32000_audit_v2.py",
    "aggregate": "tests/unit/test_bilinear_budget32000_aggregate_v2.py",
}
DIRECTORIES = {role: f"runs/learning/bilinear-budget32000-{role}-tests-v2" for role in SOURCES}
DEPENDENCIES = {role: {SOURCES[role], TESTS[role], AMENDMENT_PATH, BASE_READY} for role in SOURCES}
FILES = set().union(*DEPENDENCIES.values()) | {
    directory + "/" + name
    for directory in DIRECTORIES.values()
    for name in ("test-receipt.json", "stdout.txt")
}
PREVIOUS_PINS = {
    "summary.json": "1237d6e266fa57b5284f76ab76d50069ec9f161f5e23add84e06d769a47b5e11",
    "execution-receipt.json": "97f36f6d2b1bee433082b49c5204376d4f70dd78e3abba52431d5be32781b1b8",
    "independent-audit.json": "85b193ad473d77ac10d84aa2e9138ba4d56efd7e9694ef52eeab4ed6e069c188",
    "audit-execution-receipt.json": (
        "2195369180305bb04f7967b5877e41006f965fd8499d411229193415662fc5ce"
    ),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_base(root):
    path = root / BASE_SOURCE
    require(sha(path) == BASE_SHA, "frozen scientific runner identity")
    spec = importlib.util.spec_from_file_location("bilinear32000_frozen_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def readiness(root, path, pin, base, helper, inputs):
    require(path.resolve() == (root / READINESS).resolve(), "fixed v2 readiness path")
    require(
        type(pin) is str and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin),
        "required v2 readiness SHA256",
    )
    old = base._readiness(root, root / BASE_READY, BASE_READY_SHA, helper, inputs)
    value = helper._read(helper._bind(path, pin, inputs))
    require(
        type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("campaign_version") == CAMPAIGN
        and value.get("evidence_revision") == REVISION
        and value.get("amendment_sha256") == AMENDMENT
        and value.get("base_readiness") == {"path": BASE_READY, "sha256": BASE_READY_SHA}
        and value.get("ready") is True,
        "v2 readiness identity",
    )
    files = value.get("files")
    require(type(files) is dict and set(files) == FILES, "exact fourteen-file readiness inventory")
    require(
        all(
            type(h) is str and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
            for h in files.values()
        ),
        "v2 readiness hash format",
    )
    require(
        files[AMENDMENT_PATH] == AMENDMENT and files[BASE_READY] == BASE_READY_SHA,
        "declaration pins",
    )
    paths = {name: base._resolve(root, name) for name in files}
    for name, reference in paths.items():
        helper._bind(reference, files[name], inputs)
    for role, directory in DIRECTORIES.items():
        receipt = helper._read(paths[directory + "/test-receipt.json"])
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
            "observed CPU receipt",
        )
        require(
            receipt.get("evidence_revision") == REVISION
            and receipt.get("amendment_sha256") == AMENDMENT
            and receipt.get("tested_script_sha256") == files[SOURCES[role]],
            "tested v2 identity",
        )
        require(
            receipt.get("test_dependencies") == {n: files[n] for n in DEPENDENCIES[role]},
            "four direct dependencies",
        )
        require(
            receipt.get("stdout")
            == {"path": directory + "/stdout.txt", "sha256": files[directory + "/stdout.txt"]},
            "tested stdout",
        )
    return old, value


def same_predecessor(actual, expected, base):
    require(
        type(actual) is dict and set(actual) == {"seed", "summary_sha256", "completed_utc"},
        "exact predecessor keys",
    )
    require(
        type(actual["seed"]) is int
        and actual["seed"] == expected["seed"]
        and actual["summary_sha256"] == expected["summary_sha256"],
        "predecessor identity",
    )
    require(
        base._instant(actual["completed_utc"]) == base._instant(expected["completed_utc"]),
        "exact predecessor instant",
    )


def predecessor(root, ready, ready_sha, base, helper, inputs, admission_utc):
    folder = root / f"runs/learning/{CAMPAIGN}/seed-1730"

    def read(name, expected=None):
        path = folder / name
        return helper._read(helper._bind(path, expected or helper._sha(path), inputs))

    original = {name: read(name, pin) for name, pin in PREVIOUS_PINS.items()}
    summary, execution = original["summary.json"], original["execution-receipt.json"]
    checked = read("independent-audit-v2.json")
    terminal = read("audit-execution-receipt-v2.json")
    files = ready["files"]
    require(
        summary["seed"] == 1730
        and summary["complete"] is True
        and summary["final_identity_check"] is True
        and summary["gate"]["checks"]["execution_invariants"] is True
        and summary["script_sha256"] == BASE_SHA
        and summary["shared_readiness"] == {"path": BASE_READY, "sha256": BASE_READY_SHA},
        "fixed completed seed1730",
    )
    require(
        original["independent-audit.json"]["audit_passed"] is False,
        "preserved original audit failure",
    )
    require(
        type(original["audit-execution-receipt.json"]["exit_code"]) is int
        and original["audit-execution-receipt.json"]["exit_code"] == 1,
        "preserved failed terminal",
    )
    require(
        checked["complete"] is True
        and checked["audit_passed"] is True
        and checked["seed"] == 1730
        and checked["summary_sha256"] == PREVIOUS_PINS["summary.json"]
        and checked["execution_receipt_sha256"] == PREVIOUS_PINS["execution-receipt.json"]
        and checked["script_sha256"] == files[SOURCES["auditor"]]
        and checked["test_receipt_sha256"] == files[DIRECTORIES["auditor"] + "/test-receipt.json"]
        and checked["audit_revision"] == REVISION
        and checked["amendment_sha256"] == AMENDMENT
        and checked["plan_sha256"] == base._PLAN
        and checked["base_readiness"] == {"path": BASE_READY, "sha256": BASE_READY_SHA}
        and checked["shared_readiness"] == {"path": READINESS, "sha256": ready_sha},
        "actual repaired scientific audit",
    )
    require(
        checked["child"] == summary["child"] and checked["gate"] == summary["gate"],
        "audited outcome identity",
    )
    for receipt in (execution, terminal):
        require(
            type(receipt["exit_code"]) is int
            and receipt["exit_code"] == 0
            and receipt["terminal_completion_observed_by_primary"] is True
            and receipt["seed"] == 1730
            and receipt["summary_sha256"] == PREVIOUS_PINS["summary.json"],
            "observed successful terminal",
        )
    require(
        terminal["auditor_sha256"] == files[SOURCES["auditor"]]
        and terminal["test_receipt_sha256"] == files[DIRECTORIES["auditor"] + "/test-receipt.json"]
        and terminal["cuda_visible_devices"] == "-1"
        and terminal["audit_sha256"] == helper._sha(folder / "independent-audit-v2.json")
        and terminal["execution_receipt_sha256"] == PREVIOUS_PINS["execution-receipt.json"],
        "repaired audit terminal binding",
    )
    require(
        base._instant(execution["started_utc"])
        <= base._instant(execution["completed_utc"])
        <= base._instant(terminal["started_utc"])
        <= base._instant(terminal["completed_utc"])
        <= base._instant(admission_utc),
        "repaired audit chronology",
    )
    prior_folder = root / f"runs/learning/{CAMPAIGN}/seed-1729"
    prior_path = prior_folder / "execution-receipt.json"
    prior = helper._read(
        helper._bind(prior_path, execution["prior_execution_receipt_sha256"], inputs)
    )
    same_predecessor(
        execution["prior_seed"],
        {
            "seed": 1729,
            "summary_sha256": prior["summary_sha256"],
            "completed_utc": prior["completed_utc"],
        },
        base,
    )
    helper._bind(folder / "primary-stdout.txt", execution["stdout_sha256"], inputs)
    helper._bind(folder / "audit-stdout-v2.txt", terminal["stdout_sha256"], inputs)
    return {
        "seed": 1730,
        "summary_sha256": PREVIOUS_PINS["summary.json"],
        "completed_utc": execution["completed_utc"],
        "execution_receipt_sha256": PREVIOUS_PINS["execution-receipt.json"],
        "audit_execution_receipt_sha256": helper._sha(folder / "audit-execution-receipt-v2.json"),
    }


def snapshot(path, data, root, relative=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return (
        path.relative_to(root).as_posix() if relative else str(path),
        hashlib.sha256(data).hexdigest(),
    )


def execute(root, ready_path, ready_sha, preflight_only=False):
    require(
        Path.cwd().resolve() == root and "torch" not in sys.modules, "Torch-free root admission"
    )
    base = load_base(root)
    bilinear, bilinear_path = base._load_bilinear(root)
    mean, mean_path = bilinear._load_mean(root)
    helper, helper_path = mean._load_helper(root)
    adapter = base._adapter(root)
    helper._modules()
    inputs = {}
    old, ready = readiness(root, ready_path, ready_sha, base, helper, inputs)
    wrapper = root / SOURCES["runner"]
    helper._bind(wrapper, ready["files"][SOURCES["runner"]], inputs)
    out = root / f"runs/learning/{CAMPAIGN}/seed-1731/summary.json"
    base._admit_output(root, 1731, out, mean)
    require(
        not (out.parent / "continuation").exists()
        and not (out.parent / "continuation-complete.json").exists(),
        "immutable continuation namespace",
    )
    plan = root / "docs/experiments/2026-09-26-bilinear-budget32000-plan.md"
    recipe = root / "configs/experiments/bilinear_budget32000_v1.json"
    context = base._preflight(root, 1731, plan, recipe, bilinear, mean, helper, adapter)
    context["inputs"].update(inputs)
    prior = predecessor(
        root, ready, ready_sha, base, helper, context["inputs"], datetime.now(UTC).isoformat()
    )
    helper._unchanged(context["inputs"])
    if preflight_only:
        require("torch" not in sys.modules, "preflight imported torch")
        print(
            json.dumps(
                {
                    "preflight_passed": True,
                    "seed": 1731,
                    "execution_revision": EXECUTION,
                    "torch_imported": False,
                }
            )
        )
        return
    base_snapshots = {
        "script.py": (root / BASE_SOURCE).read_bytes(),
        "bilinear-helper.py": bilinear_path.read_bytes(),
        "mean-helper.py": mean_path.read_bytes(),
        "helper.py": helper_path.read_bytes(),
        "input-adapter.py": (root / "scripts/refit_symmetric_affine_v3.py").read_bytes(),
        "plan.md": plan.read_bytes(),
        "input-amendment.md": (
            root / "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md"
        ).read_bytes(),
        "recipe.json": recipe.read_bytes(),
        "source.zip": context["source"].read_bytes(),
        "configs.zip": context["configs"].read_bytes(),
        "readiness.json": (root / BASE_READY).read_bytes(),
        "inputs.json": mean._bytes(context["inputs"]),
    }
    hashes = dict(
        snapshot(out.with_suffix("." + suffix), data, root)
        for suffix, data in base_snapshots.items()
    )
    for relative, expected in old["files"].items():
        data = (root / relative).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, "base readiness copy")
        hashes.update([snapshot(out.parent / "readiness" / relative, data, root)])
    continuation = {}
    for relative, expected in {**ready["files"], READINESS: ready_sha}.items():
        data = (root / relative).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, "continuation copy identity")
        continuation.update([snapshot(out.parent / "continuation" / relative, data, root, True)])
    summary = {
        "campaign_version": CAMPAIGN,
        "seed": 1731,
        "data_split_seed": 1729,
        "complete": False,
        "acceptance": False,
        "architecture": base._ARCHITECTURE,
        "objective": base._OBJECTIVE,
        "inference": base._INFERENCE,
        "evaluator": base._EVALUATOR,
        "plan_sha256": base._PLAN,
        "script_sha256": BASE_SHA,
        "scorer_sha256": base._SCORER,
        "mean_helper_sha256": base._MEAN,
        "helper_sha256": mean._HELPER,
        "input_adapter_identity": base._INPUT_ADAPTER,
        "input_adapter_source_sha256": base._ADAPTER_SHA,
        "input_amendment_sha256": base._AMENDMENT,
        "recipe_sha256": helper._sha(recipe),
        "shared_readiness": {"path": BASE_READY, "sha256": BASE_READY_SHA},
        "prior_seed_execution": prior,
        "input_sha256": context["inputs"],
        "snapshot_sha256": hashes,
        "protected_test_used": False,
        "standard_serving_supported": False,
        "execution_wrapper_sha256": ready["files"][SOURCES["runner"]],
        "execution_revision": EXECUTION,
        "amendment_sha256": AMENDMENT,
        "continuation_readiness": {"path": READINESS, "sha256": ready_sha},
        "continuation_snapshots": continuation,
        **{
            k: context[k]
            for k in (
                "historical8000_summary_sha256",
                "historical8000_audit_sha256",
                "historical8000_decision_sha256",
                "historical8000_child_sha256",
            )
        },
        "limitations": [
            "Continuation changes evidence admission and provenance only; "
            "scientific execution is unchanged.",
            "Three existing parent seeds reuse 222 development queries.",
            "Diagnostic serialization is included in refit timing; no serving promotion.",
        ],
    }
    helper._unchanged(context["inputs"])
    helper._unchanged(hashes)
    helper._unchanged({str(root / p): h for p, h in continuation.items()})
    base._execute(root, out, context, bilinear, mean, helper, summary)
    helper._unchanged(context["inputs"])
    helper._unchanged(hashes)
    helper._unchanged({str(root / p): h for p, h in continuation.items()})
    saved = helper._read(out)
    require(
        saved == summary and saved["complete"] is True and saved["final_identity_check"] is True,
        "base execution completed exactly",
    )
    completion = {
        "complete": True,
        "seed": 1731,
        "summary_sha256": helper._sha(out),
        "execution_wrapper_sha256": summary["execution_wrapper_sha256"],
        "execution_revision": EXECUTION,
        "amendment_sha256": AMENDMENT,
        "continuation_readiness": summary["continuation_readiness"],
        "root_observed_terminal_required": True,
    }
    snapshot(out.parent / "continuation-complete.json", mean._bytes(completion), root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1731,), required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--readiness-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    execute(root, args.readiness.resolve(), args.readiness_sha256, args.preflight_only)


if __name__ == "__main__":
    main()
