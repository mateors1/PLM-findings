"""Synthetic mixed-revision aggregation; no campaign artifacts, weights or GPU."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def subject():
    return load(ROOT / "scripts/aggregate_bilinear_budget32000_v2.py", "aggregate_v2_test")


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def read(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def environment(root, subject, errors=None):
    """Copy only the frozen readiness's explicit source/doc/test inventory."""
    base = subject.load_base(ROOT)
    for name in base.READINESS_FILES | {subject.BASE_READINESS}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    original = read(root, subject.BASE_READINESS)
    assert subject.sha(root / subject.BASE_READINESS) == subject.BASE_READINESS_SHA
    for name in subject.FILES - {subject.BASE_READINESS}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in (subject.SOURCES["aggregate"], subject.TESTS["aggregate"], subject.AMENDMENT):
            path.write_bytes((ROOT / name).read_bytes())
        else:
            path.write_bytes(b"synthetic v2 source or stdout\n")
    for role, directory in subject.DIRECTORIES.items():
        write(
            root,
            directory + "/test-receipt.json",
            {
                "passed": True,
                "gpu_used": False,
                "experiment_training_executed": False,
                "exit_code": 0,
                "terminal_completion_observed": True,
                "tests_passed": 7,
                "tests_skipped": 0,
                "evidence_revision": subject.REVISION,
                "amendment_sha256": subject.AMENDMENT_SHA,
                "tested_script_sha256": subject.sha(root / subject.SOURCES[role]),
                "test_dependencies": {
                    name: subject.sha(root / name) for name in subject.DEPENDENCIES[role]
                },
                "stdout": {
                    "path": directory + "/stdout.txt",
                    "sha256": subject.sha(root / directory / "stdout.txt"),
                },
            },
        )
    ready = {
        "schema_version": 1,
        "campaign_version": subject.CAMPAIGN,
        "evidence_revision": subject.REVISION,
        "amendment_sha256": subject.AMENDMENT_SHA,
        "base_readiness": {"path": subject.BASE_READINESS, "sha256": subject.BASE_READINESS_SHA},
        "ready": True,
        "files": {name: subject.sha(root / name) for name in subject.FILES},
    }
    write(root, subject.READINESS, ready)
    pin = subject.sha(root / subject.READINESS)
    helper = load(ROOT / base.MEAN, "frozen_metric_test")
    fixtures = load(ROOT / base.AGGREGATE_TEST, "original_aggregate_test_fixtures")
    original_admission = {"path": subject.BASE_READINESS, "sha256": subject.BASE_READINESS_SHA}
    repaired_admission = {"path": subject.READINESS, "sha256": pin}
    previous = None
    for order, seed in enumerate(base.SEEDS):
        folder = f"{subject.BASE}/seed-{seed}"
        child = fixtures.child_fixture(base, helper, (errors or {}).get(seed, 0))
        write(root, folder + "/child.json", child)
        child_summary = {k: child[k] for k in ("aggregate", "groups", "paired_vs_parent_dense")}
        child_summary["paired_vs_historical8000"] = {"gains": 0, "losses": 0}
        invariants = dict.fromkeys(base.INVARIANTS, True)
        summary = {
            "complete": True,
            "final_identity_check": True,
            "acceptance": False,
            "protected_test_used": False,
            "standard_serving_supported": False,
            "seed": seed,
            "data_split_seed": 1729,
            "campaign_version": base.CAMPAIGN,
            "evaluator": base.EVALUATOR,
            "architecture": base.ARCHITECTURE,
            "objective": base.OBJECTIVE,
            "inference": base.INFERENCE,
            "input_adapter_identity": base.INPUT_ADAPTER,
            "input_amendment_sha256": base.AMENDMENT_SHA,
            "plan_sha256": base.PLAN_SHA,
            "script_sha256": original["files"][base.RUNNER],
            "recipe_sha256": original["files"][base.RECIPE],
            "scorer_sha256": base.SCORER_SHA,
            "split_hash": base.SPLIT,
            "shared_readiness": original_admission,
            "child": child_summary,
            "invariants": invariants,
            "gate": base.seed_gate(seed, child["aggregate"], child["groups"], invariants),
            "artifact_sha256": {
                "child.json": subject.sha(root / folder / "child.json"),
                "checkpoint-final.pt": "f" * 64,
            },
            "input_sha256": {"data/protected.jsonl": "f" * 64},
        }
        extra = {
            "execution_wrapper_sha256": ready["files"][subject.SOURCES["runner"]],
            "execution_revision": subject.EXECUTION_REVISION,
            "amendment_sha256": subject.AMENDMENT_SHA,
            "continuation_readiness": repaired_admission,
        }
        if seed == 1731:
            snapshots = {}
            for name, digest in (ready["files"] | {subject.READINESS: pin}).items():
                snap = folder + "/continuation/" + name
                path = root / snap
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((root / name).read_bytes())
                snapshots[snap] = digest
            summary.update(**extra, continuation_snapshots=snapshots)
        write(root, folder + "/summary.json", summary)
        summary_sha = subject.sha(root / folder / "summary.json")
        if seed in subject.ORIGINAL_SUMMARIES:
            subject.ORIGINAL_SUMMARIES[seed] = summary_sha  # Synthetic fixture pins, wrapper only.
        complete_sha = None
        if seed == 1731:
            write(
                root,
                folder + "/continuation-complete.json",
                {
                    "complete": True,
                    "seed": seed,
                    "summary_sha256": summary_sha,
                    "root_observed_terminal_required": True,
                    **extra,
                },
            )
            complete_sha = subject.sha(root / folder / "continuation-complete.json")
        paths = subject.route(seed)
        for name in ("primary-stdout.txt", paths["stdout"]):
            (root / folder / name).write_text("synthetic observed output", encoding="utf-8")

        # Distinct 4-second slots, with equivalent +01 timezone predecessor spelling.
        def stamp(second):
            return f"2026-09-26T00:00:{order * 4 + second:02d}.5359481Z"

        execution = {
            "seed": seed,
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "started_utc": stamp(0),
            "completed_utc": stamp(1),
            "summary_sha256": summary_sha,
            "stdout_sha256": subject.sha(root / folder / "primary-stdout.txt"),
            "prior_seed": previous["reference"] if previous else None,
            "prior_execution_receipt_sha256": previous["execution"] if previous else None,
            "prior_audit_receipt_sha256": previous["audit"] if previous else None,
        }
        if seed == 1731:
            execution["continuation_complete_sha256"] = complete_sha
        write(root, folder + "/execution-receipt.json", execution)
        execution_sha = subject.sha(root / folder / "execution-receipt.json")
        auditor_sha = (
            original["files"][base.AUDITOR]
            if seed == 1729
            else ready["files"][subject.SOURCES["auditor"]]
        )
        test_sha = (
            original["files"][base.TEST_DIRECTORIES["auditor"] + "/test-receipt.json"]
            if seed == 1729
            else ready["files"][subject.DIRECTORIES["auditor"] + "/test-receipt.json"]
        )
        audit = {
            "complete": True,
            "audit_passed": True,
            "acceptance": False,
            "seed": seed,
            "summary_sha256": summary_sha,
            "script_sha256": auditor_sha,
            "test_receipt_sha256": test_sha,
            "plan_sha256": base.PLAN_SHA,
            "execution_receipt_sha256": execution_sha,
            "child": child_summary,
            "gate": summary["gate"],
            "shared_readiness": original_admission if seed == 1729 else repaired_admission,
        }
        if seed != 1729:
            audit.update(
                audit_revision=subject.REVISION,
                amendment_sha256=subject.AMENDMENT_SHA,
                base_auditor_sha256=original["files"][base.AUDITOR],
                base_readiness=original_admission,
            )
        if seed == 1731:
            audit["continuation_complete_sha256"] = complete_sha
        write(root, folder + "/" + paths["audit"], audit)
        terminal = {
            "seed": seed,
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "started_utc": stamp(2),
            "completed_utc": stamp(3),
            "summary_sha256": summary_sha,
            "audit_sha256": subject.sha(root / folder / paths["audit"]),
            "execution_receipt_sha256": execution_sha,
            "stdout_sha256": subject.sha(root / folder / paths["stdout"]),
            "auditor_sha256": auditor_sha,
            "test_receipt_sha256": test_sha,
            "cuda_visible_devices": "-1",
        }
        write(root, folder + "/" + paths["terminal"], terminal)
        previous = {
            "reference": {
                "seed": seed,
                "summary_sha256": summary_sha,
                "completed_utc": stamp(1).replace("T00:", "T01:").replace("Z", "+01:00"),
            },
            "execution": execution_sha,
            "audit": subject.sha(root / folder / paths["terminal"]),
        }
        if seed == 1730:
            write(
                root,
                folder + "/independent-audit.json",
                {"complete": False, "error": "preserved original failure"},
            )
            write(root, folder + "/audit-execution-receipt.json", {"exit_code": 1})
    return base, ready, original, pin, helper


def run(root, subject, pin, helper):
    base = subject.load_base(root)
    reader = subject.Reader(root, base)
    ready, original = subject.readiness(reader, pin)
    return subject.aggregate(reader, ready, original, helper), reader


@pytest.mark.parametrize("negative", [False, True])
def test_complete_mixed_routes_and_negative_no_pooled_rescue(subject, tmp_path, negative):
    _, _, _, pin, helper = environment(tmp_path, subject, {1729: 6} if negative else {})
    result, reader = run(tmp_path, subject, pin, helper)
    assert result["complete"] and result["acceptance"] is False
    assert result["gate"]["primary_checks_passed"] is (not negative)
    assert result["all_three"]["aggregate"]["query_count"] == 666
    assert [entry["audit_path"].split("/")[-1] for entry in result["per_seed"]] == [
        "independent-audit.json",
        "independent-audit-v2.json",
        "independent-audit-v2.json",
    ]
    assert not any(name.endswith((".pt", ".jsonl", ".db")) for name in reader.inputs)
    assert subject.BASE + "/seed-1730/independent-audit.json" not in reader.inputs
    assert (
        len(read(tmp_path, subject.BASE + "/seed-1731/summary.json")["continuation_snapshots"])
        == 15
    )
    assert "torch" not in sys.modules


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("summary.json", "execution_wrapper_sha256", "0" * 64),
        ("summary.json", "continuation_readiness", {}),
        ("summary.json", "continuation_snapshots", {}),
        ("continuation-complete.json", "complete", False),
        ("continuation-complete.json", "complete", 1),
        ("continuation-complete.json", "summary_sha256", "0" * 64),
        ("continuation-complete.json", "execution_wrapper_sha256", "0" * 64),
        ("continuation-complete.json", "root_observed_terminal_required", False),
        ("execution-receipt.json", "continuation_complete_sha256", "0" * 64),
        ("independent-audit-v2.json", "script_sha256", "0" * 64),
        ("independent-audit-v2.json", "test_receipt_sha256", "0" * 64),
        ("independent-audit-v2.json", "shared_readiness", {}),
        ("independent-audit-v2.json", "base_readiness", {}),
        ("independent-audit-v2.json", "base_auditor_sha256", "0" * 64),
        ("independent-audit-v2.json", "audit_revision", "old"),
        ("audit-execution-receipt-v2.json", "exit_code", 1),
        ("audit-execution-receipt-v2.json", "exit_code", False),
        ("audit-execution-receipt-v2.json", "cuda_visible_devices", "0"),
    ],
)
def test_bad_continuation_or_actual_audit_identity(subject, tmp_path, target, field, value):
    _, _, _, pin, helper = environment(tmp_path, subject)
    name = subject.BASE + "/seed-1731/" + target
    data = read(tmp_path, name)
    data[field] = value
    write(tmp_path, name, data)
    with pytest.raises(ValueError):
        run(tmp_path, subject, pin, helper)


@pytest.mark.parametrize(
    "mutation",
    ["100ns", "extra_key", "missing_key", "wrong_seed", "wrong_summary", "overlap", "prior_hash"],
)
def test_exact_predecessor_equality_and_order(subject, tmp_path, mutation):
    _, _, _, pin, helper = environment(tmp_path, subject)
    folder = subject.BASE + "/seed-1730"
    execution = read(tmp_path, folder + "/execution-receipt.json")
    if mutation == "100ns":
        execution["prior_seed"]["completed_utc"] = execution["prior_seed"]["completed_utc"].replace(
            "5359481", "5359482"
        )
    elif mutation == "extra_key":
        execution["prior_seed"]["extra"] = 0
    elif mutation == "missing_key":
        execution["prior_seed"].pop("seed")
    elif mutation == "wrong_seed":
        execution["prior_seed"]["seed"] = 1731
    elif mutation == "wrong_summary":
        execution["prior_seed"]["summary_sha256"] = "0" * 64
    elif mutation == "overlap":
        execution["started_utc"] = "2026-09-26T00:00:03.5359480Z"
    else:
        execution["prior_audit_receipt_sha256"] = "0" * 64
    write(tmp_path, folder + "/execution-receipt.json", execution)
    execution_sha = subject.sha(tmp_path / folder / "execution-receipt.json")
    audit = read(tmp_path, folder + "/independent-audit-v2.json")
    audit["execution_receipt_sha256"] = execution_sha
    write(tmp_path, folder + "/independent-audit-v2.json", audit)
    terminal = read(tmp_path, folder + "/audit-execution-receipt-v2.json")
    terminal.update(
        execution_receipt_sha256=execution_sha,
        audit_sha256=subject.sha(tmp_path / folder / "independent-audit-v2.json"),
    )
    write(tmp_path, folder + "/audit-execution-receipt-v2.json", terminal)
    with pytest.raises(ValueError, match=r"prior seed binding|next seed|prior receipt"):
        run(tmp_path, subject, pin, helper)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_seed",
        "missing_repaired_audit",
        "snapshot_tamper",
        "child_tamper",
        "original_summary_tamper",
    ],
)
def test_missing_and_tampered_evidence(subject, tmp_path, mutation):
    _, _, _, pin, helper = environment(tmp_path, subject)
    folder = subject.BASE + "/seed-1731"
    if mutation == "missing_seed":
        (tmp_path / folder / "summary.json").unlink()
    elif mutation == "missing_repaired_audit":
        (tmp_path / subject.BASE / "seed-1730/independent-audit-v2.json").unlink()
    elif mutation == "snapshot_tamper":
        (tmp_path / folder / "continuation" / subject.AMENDMENT).write_text("changed")
    elif mutation == "child_tamper":
        (tmp_path / folder / "child.json").write_text("{}")
    else:
        (tmp_path / subject.BASE / "seed-1729/summary.json").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        run(tmp_path, subject, pin, helper)


@pytest.mark.parametrize(
    "mutation",
    ["unknown_path", "not_ready", "base_pin", "tests_skipped", "test_exit", "test_dependencies"],
)
def test_readiness_closed_inventory_and_actual_receipts(subject, tmp_path, mutation):
    base, ready, _, _, _ = environment(tmp_path, subject)
    if mutation == "unknown_path":
        ready["files"]["data/protected.jsonl"] = "0" * 64
    elif mutation == "not_ready":
        ready["ready"] = False
    elif mutation == "base_pin":
        ready["base_readiness"]["sha256"] = "0" * 64
    else:
        name = subject.DIRECTORIES["aggregate"] + "/test-receipt.json"
        value = read(tmp_path, name)
        if mutation == "tests_skipped":
            value["tests_skipped"] = 1
        elif mutation == "test_exit":
            value["exit_code"] = False
        else:
            value["test_dependencies"].pop(subject.AMENDMENT)
        write(tmp_path, name, value)
        ready["files"][name] = subject.sha(tmp_path / name)
    write(tmp_path, subject.READINESS, ready)
    reader = subject.Reader(tmp_path, base)
    with pytest.raises(ValueError):
        subject.readiness(reader, subject.sha(tmp_path / subject.READINESS))
    assert "data/protected.jsonl" not in reader.inputs


@pytest.mark.parametrize("mode", ["success", "preflight", "failure"])
def test_real_main_cli_and_immutable_evidence(subject, tmp_path, monkeypatch, capsys, mode):
    _, _, _, pin, _ = environment(tmp_path, subject)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subject, "__file__", str(tmp_path / subject.SOURCES["aggregate"]))
    argv = [
        subject.__file__,
        "--readiness",
        str(tmp_path / subject.READINESS),
        "--readiness-sha256",
        pin,
    ]
    if mode == "preflight":
        argv.append("--preflight-only")
    monkeypatch.setattr(sys, "argv", argv)
    if mode == "failure":
        (tmp_path / subject.BASE / "seed-1731/independent-audit-v2.json").unlink()
    out = tmp_path / subject.BASE / "aggregate-v2.json"
    if mode == "failure":
        with pytest.raises(FileNotFoundError):
            subject.main()
        assert read(tmp_path, str(out.relative_to(tmp_path)))["complete"] is False
    else:
        subject.main()
        if mode == "preflight":
            assert json.loads(capsys.readouterr().out)["seed_artifacts_read"] is False
            assert not out.exists()
            return
        assert read(tmp_path, str(out.relative_to(tmp_path)))["complete"] is True
    assert out.with_suffix(".script.py").is_file() and out.with_suffix(".inputs.json").is_file()
    before = out.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        subject.main()
    assert out.read_bytes() == before


def test_frozen_gate_boundaries_and_no_monkeypatch(subject):
    base = subject.load_base(ROOT)
    for seed, (floor, f1, counts) in base.FLOORS.items():
        for exact in (floor, floor + 1):
            for quality in (f1 - 1e-6, f1):
                for valid in (221, 222):
                    aggregate = {
                        "exact_count": exact,
                        "f1": quality,
                        "serialization_compatible": valid,
                    }
                    groups = {
                        g: {"exact_count": n} for g, n in zip(base.GROUPS, counts, strict=True)
                    }
                    gate = base.seed_gate(
                        seed, aggregate, groups, dict.fromkeys(base.INVARIANTS, True)
                    )
                    assert gate["primary_checks_passed"] is (
                        exact > floor and quality >= f1 and valid == 222
                    )
    assert len(subject.FILES) == 14 and all(len(v) == 4 for v in subject.DEPENDENCIES.values())
    assert subject.sha(ROOT / subject.BASE_SCRIPT) == subject.BASE_SCRIPT_SHA
