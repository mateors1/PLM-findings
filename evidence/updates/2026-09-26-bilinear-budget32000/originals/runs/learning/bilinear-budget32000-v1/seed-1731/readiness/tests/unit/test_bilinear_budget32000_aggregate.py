"""Synthetic metadata-only campaign integration; no real run inputs or tensor payloads."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def subject():
    return load(ROOT / "scripts/aggregate_bilinear_budget32000.py", "aggregate32000_test")


@pytest.fixture(scope="module")
def mean():
    return load(ROOT / "scripts/refit_membership_projection.py", "aggregate32000_mean_test")


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def read(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def child_fixture(subject, mean, errors=0):
    rows = []
    for index in range(222):
        group = "COLOR" if index < 103 else "TYPE_single" if index < 154 else "TYPE_dual"
        dimension = "COLOR" if group == "COLOR" else "TYPE"
        scores = [-1.0] * 1025
        scores[-1] = 1.0
        if index >= 222 - errors:
            scores[-1], scores[-2] = -1.0, 1.0
        selected = mean._prediction(scores, 1024 + index)
        rows.append(
            {
                "index": index,
                "subject": f"PKM_SYNTHETIC_{index}",
                "dimension": dimension,
                "prompt_ids": [1, 1024 + index, 33 if group == "COLOR" else 32, 34, 5],
                "group": group,
                "expected_set_ids": [2048],
                "selected_set_ids": selected,
                "symmetric_relation_logits": scores,
                "metrics": mean._metrics(selected, [2048]),
                "strict_separation": index < 222 - errors,
            }
        )
    child = {
        "complete": True,
        "query_count": 222,
        "product_token_ids": list(range(1024, 2049)),
        "responses": rows,
        "aggregate": mean._totals(rows),
        "groups": {g: mean._totals([r for r in rows if r["group"] == g]) for g in subject.GROUPS},
        "paired_vs_parent_dense": {"gains": 0, "losses": 0},
    }
    return child


def environment(root, subject, mean, errors=None):
    errors = errors or {}
    for name in subject.READINESS_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic source or stdout fixture\n")
    for name in (subject.PLAN, subject.AMENDMENT, subject.MEAN, subject.SCORER, subject.AGGREGATOR):
        (root / name).write_bytes((ROOT / name).read_bytes())
    for role, directory in subject.TEST_DIRECTORIES.items():
        source = {
            "runner": subject.RUNNER,
            "auditor": subject.AUDITOR,
            "aggregate": subject.AGGREGATOR,
        }[role]
        receipt = {
            "passed": True,
            "gpu_used": False,
            "experiment_training_executed": False,
            "exit_code": 0,
            "terminal_completion_observed": True,
            "tests_passed": 7,
            "tests_skipped": 0,
            "tested_script_sha256": subject.digest(root / source),
            "test_dependencies": {
                n: subject.digest(root / n) for n in subject.TEST_DEPENDENCIES[role]
            },
            "stdout": {
                "path": directory + "/stdout.txt",
                "sha256": subject.digest(root / directory / "stdout.txt"),
            },
        }
        if role == "runner":
            receipt["tested_recipe_sha256"] = subject.digest(root / subject.RECIPE)
        write(root, directory + "/test-receipt.json", receipt)
    files = {n: subject.digest(root / n) for n in subject.READINESS_FILES}
    ready = {
        "schema_version": 1,
        "campaign_version": subject.CAMPAIGN,
        "plan_sha256": subject.PLAN_SHA,
        "ready": True,
        "files": files,
    }
    write(root, subject.READINESS, ready)
    admission = {"path": subject.READINESS, "sha256": subject.digest(root / subject.READINESS)}
    previous = None
    for order, seed in enumerate(subject.SEEDS):
        folder = f"{subject.BASE}/seed-{seed}"
        child = child_fixture(subject, mean, errors.get(seed, 0))
        write(root, folder + "/child.json", child)
        invariants = dict.fromkeys(subject.INVARIANTS, True)
        child_summary = {k: child[k] for k in ("aggregate", "groups", "paired_vs_parent_dense")}
        child_summary["paired_vs_historical8000"] = {"gains": 0, "losses": 0}
        summary = {
            "complete": True,
            "final_identity_check": True,
            "acceptance": False,
            "protected_test_used": False,
            "standard_serving_supported": False,
            "seed": seed,
            "data_split_seed": 1729,
            "campaign_version": subject.CAMPAIGN,
            "evaluator": subject.EVALUATOR,
            "architecture": subject.ARCHITECTURE,
            "objective": subject.OBJECTIVE,
            "inference": subject.INFERENCE,
            "input_adapter_identity": subject.INPUT_ADAPTER,
            "input_amendment_sha256": subject.AMENDMENT_SHA,
            "shared_readiness": admission,
            "plan_sha256": subject.PLAN_SHA,
            "script_sha256": files[subject.RUNNER],
            "recipe_sha256": files[subject.RECIPE],
            "scorer_sha256": subject.SCORER_SHA,
            "split_hash": subject.SPLIT,
            "artifact_sha256": {
                "child.json": subject.digest(root / folder / "child.json"),
                "checkpoint-final.pt": "a" * 64,
            },
            # These deliberately nonexistent paths MUST NOT be traversed, even for hashing.
            "input_sha256": {"data/protected.jsonl": "b" * 64},
            "child": child_summary,
            "invariants": invariants,
            "gate": subject.seed_gate(seed, child["aggregate"], child["groups"], invariants),
        }
        write(root, folder + "/summary.json", summary)
        summary_sha = subject.digest(root / folder / "summary.json")
        for name in ("primary-stdout.txt", "audit-stdout.txt"):
            (root / folder / name).write_text("synthetic terminal output", encoding="utf-8")

        def stamp(offset):
            return f"2026-09-26T00:00:{order * 4 + offset:02d}.0000001Z"

        execution = {
            "seed": seed,
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "started_utc": stamp(0),
            "completed_utc": stamp(1),
            "summary_sha256": summary_sha,
            "stdout_sha256": subject.digest(root / folder / "primary-stdout.txt"),
            "prior_seed": previous["reference"] if previous else None,
            "prior_execution_receipt_sha256": previous["execution"] if previous else None,
            "prior_audit_receipt_sha256": previous["audit"] if previous else None,
        }
        write(root, folder + "/execution-receipt.json", execution)
        execution_sha = subject.digest(root / folder / "execution-receipt.json")
        audit = {
            "complete": True,
            "audit_passed": True,
            "acceptance": False,
            "seed": seed,
            "summary_sha256": summary_sha,
            "script_sha256": files[subject.AUDITOR],
            "plan_sha256": subject.PLAN_SHA,
            "execution_receipt_sha256": execution_sha,
            "shared_readiness": admission,
            "test_receipt_sha256": files[
                subject.TEST_DIRECTORIES["auditor"] + "/test-receipt.json"
            ],
            "child": child_summary,
            "gate": summary["gate"],
        }
        write(root, folder + "/independent-audit.json", audit)
        audit_receipt = {
            "seed": seed,
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "started_utc": stamp(2),
            "completed_utc": stamp(3),
            "summary_sha256": summary_sha,
            "stdout_sha256": subject.digest(root / folder / "audit-stdout.txt"),
            "audit_sha256": subject.digest(root / folder / "independent-audit.json"),
            "execution_receipt_sha256": execution_sha,
            "auditor_sha256": files[subject.AUDITOR],
            "test_receipt_sha256": files[
                subject.TEST_DIRECTORIES["auditor"] + "/test-receipt.json"
            ],
            "cuda_visible_devices": "-1",
        }
        write(root, folder + "/audit-execution-receipt.json", audit_receipt)
        previous = {
            "reference": {
                "seed": seed,
                "summary_sha256": summary_sha,
                "completed_utc": execution["completed_utc"],
            },
            "execution": execution_sha,
            "audit": subject.digest(root / folder / "audit-execution-receipt.json"),
        }
    return subject.digest(root / subject.READINESS)


def execute(root, subject, pin):
    reader = subject.Reader(root)
    ready = subject.readiness(reader, pin)
    return subject.aggregate(reader, ready, subject.load_mean(reader)), reader


@pytest.mark.parametrize("negative", [False, True])
def test_full_metadata_campaign_including_audited_negative(subject, mean, tmp_path, negative):
    pin = environment(tmp_path, subject, mean, {1729: 6} if negative else {})
    result, reader = execute(tmp_path, subject, pin)
    assert result["complete"] and result["acceptance"] is False
    assert result["gate"]["primary_checks_passed"] is (not negative)
    assert result["gate"]["accepted"] is False
    assert result["all_three"]["aggregate"]["query_count"] == 666
    assert result["all_three"]["aggregate"]["exact_count"] == (660 if negative else 666)
    assert len(result["per_seed"]) == 3
    assert not any(n.endswith(".pt") or n.startswith("data/") for n in reader.inputs)
    assert all(e["gate"]["primary_checks_passed"] for e in result["per_seed"][1:])


@pytest.mark.parametrize(
    "target",
    [
        "child.json",
        "summary.json",
        "primary-stdout.txt",
        "audit-stdout.txt",
        "independent-audit.json",
    ],
)
def test_content_hash_mutation_rejected(subject, mean, tmp_path, target):
    pin = environment(tmp_path, subject, mean)
    path = tmp_path / subject.BASE / "seed-1730" / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        execute(tmp_path, subject, pin)


def test_missing_seed_cannot_form_campaign(subject, mean, tmp_path):
    pin = environment(tmp_path, subject, mean)
    (tmp_path / subject.BASE / "seed-1731/summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        execute(tmp_path, subject, pin)


@pytest.mark.parametrize(
    "field",
    [
        "architecture",
        "objective",
        "inference",
        "input_adapter_identity",
        "input_amendment_sha256",
        "shared_readiness",
    ],
)
def test_seed_semantic_identity_and_shared_admission(subject, mean, tmp_path, field):
    pin = environment(tmp_path, subject, mean)
    name = subject.BASE + "/seed-1729/summary.json"
    value = read(tmp_path, name)
    value[field] = "different"
    write(tmp_path, name, value)
    with pytest.raises(ValueError, match=r"seed identity|seed shared admission"):
        execute(tmp_path, subject, pin)


def test_audit_must_bind_same_admission(subject, mean, tmp_path):
    pin = environment(tmp_path, subject, mean)
    name = subject.BASE + "/seed-1729/independent-audit.json"
    value = read(tmp_path, name)
    value["shared_readiness"]["sha256"] = "0" * 64
    write(tmp_path, name, value)
    with pytest.raises(ValueError, match="audit shared admission"):
        execute(tmp_path, subject, pin)


@pytest.mark.parametrize("field", ["auditor_sha256", "test_receipt_sha256"])
def test_audit_terminal_binds_frozen_implementation(subject, mean, tmp_path, field):
    pin = environment(tmp_path, subject, mean)
    name = subject.BASE + "/seed-1729/audit-execution-receipt.json"
    value = read(tmp_path, name)
    value[field] = "0" * 64
    write(tmp_path, name, value)
    with pytest.raises(ValueError, match="audit terminal implementation"):
        execute(tmp_path, subject, pin)


@pytest.mark.parametrize(
    "change",
    ["metrics", "selection", "separation", "duplicate", "truth", "columns", "group", "nan"],
)
def test_rebuild_rejects_row_corruption(subject, mean, change):
    child = child_fixture(subject, mean)
    row = child["responses"][0]
    if change == "metrics":
        row["metrics"]["f1"] = 0.5
    elif change == "selection":
        row["selected_set_ids"] = [2047]
    elif change == "separation":
        row["strict_separation"] = False
    elif change == "duplicate":
        child["responses"][1]["prompt_ids"] = row["prompt_ids"]
    elif change == "truth":
        row["expected_set_ids"] = [2048, 2048]
    elif change == "columns":
        child["product_token_ids"].reverse()
    elif change == "group":
        row["group"] = "TYPE_dual"
    else:
        row["symmetric_relation_logits"][0] = float("nan")
    with pytest.raises(ValueError):
        subject.rebuild_child(child, mean)


def test_alignment_checked_after_individually_valid_reports(subject, mean, tmp_path, monkeypatch):
    pin = environment(tmp_path, subject, mean)
    original = subject.seed_report

    def changed(*args):
        rows, entry, state = original(*args)
        if entry["seed"] == 1731:
            rows[0]["subject"] = "PKM_DIFFERENT"
        return rows, entry, state

    monkeypatch.setattr(subject, "seed_report", changed)
    with pytest.raises(ValueError, match="cross-seed"):
        execute(tmp_path, subject, pin)


def test_exact_fractional_time_and_equivalent_offsets(subject):
    assert subject.instant("2026-09-26T00:00:00.1234567Z") == subject.instant(
        "2026-09-25T19:00:00.1234567-05:00"
    )
    assert subject.instant("2026-09-26T00:00:00.1234567Z") < subject.instant(
        "2026-09-26T00:00:00.1234568Z"
    )


@pytest.mark.parametrize(
    "value", [None, "2026-09-26", "2026-09-26T00:00:00", "2026-09-26T24:00:00Z"]
)
def test_reject_invalid_time(subject, value):
    with pytest.raises(ValueError):
        subject.instant(value)


@pytest.mark.parametrize(
    "change",
    [
        "overlap",
        "prior_exec",
        "prior_audit",
        "prior_seed",
        "exit_bool",
        "unobserved",
        "audit_early",
        "audit_gpu",
    ],
)
def test_terminal_and_audit_order_contract(subject, change):
    before = "2026-09-26T00:00:00.1234567Z"
    after = "2026-09-26T00:00:00.1234568Z"
    previous = {
        "seed": 1729,
        "summary_sha256": "s0",
        "execution_receipt_sha256": "e0",
        "audit_receipt_sha256": "a0",
        "execution_finish": subject.instant(before),
        "audit_finish": subject.instant(after),
    }
    execution = {
        "seed": 1730,
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "summary_sha256": "s",
        "stdout_sha256": "o",
        "started_utc": after,
        "completed_utc": after,
        "prior_execution_receipt_sha256": "e0",
        "prior_audit_receipt_sha256": "a0",
        "prior_seed": {"seed": 1729, "summary_sha256": "s0", "completed_utc": before},
    }
    audit = {
        k: execution[k]
        for k in (
            "seed",
            "exit_code",
            "terminal_completion_observed_by_primary",
            "summary_sha256",
            "stdout_sha256",
            "started_utc",
            "completed_utc",
        )
    }
    audit.update(execution_receipt_sha256="e", cuda_visible_devices="-1")
    subject.chronology(execution, audit, 1730, "s", "e", previous)
    if change == "overlap":
        execution["started_utc"] = before
    elif change == "prior_exec":
        execution["prior_execution_receipt_sha256"] = "bad"
    elif change == "prior_audit":
        execution["prior_audit_receipt_sha256"] = "bad"
    elif change == "prior_seed":
        execution["prior_seed"]["seed"] = 1731
    elif change == "exit_bool":
        execution["exit_code"] = False
    elif change == "unobserved":
        execution["terminal_completion_observed_by_primary"] = False
    elif change == "audit_early":
        audit["started_utc"] = before
    else:
        audit["cuda_visible_devices"] = "0"
    with pytest.raises(ValueError):
        subject.chronology(execution, audit, 1730, "s", "e", previous)


def test_unknown_readiness_path_rejected_before_references_open(
    subject, mean, tmp_path, monkeypatch
):
    environment(tmp_path, subject, mean)
    value = read(tmp_path, subject.READINESS)
    value["files"]["data/protected.jsonl"] = "0" * 64
    write(tmp_path, subject.READINESS, value)
    pin = subject.digest(tmp_path / subject.READINESS)
    opened = []
    original = subject.digest

    def tracking(path):
        opened.append(Path(path).relative_to(tmp_path).as_posix())
        return original(path)

    monkeypatch.setattr(subject, "digest", tracking)
    with pytest.raises(ValueError, match="inventory"):
        subject.readiness(subject.Reader(tmp_path), pin)
    assert opened == [subject.READINESS]


@pytest.mark.parametrize("role", ["runner", "auditor", "aggregate"])
def test_success_receipt_requires_observed_zero_skip_exact_dependencies(
    subject, mean, tmp_path, role
):
    environment(tmp_path, subject, mean)
    ready = read(tmp_path, subject.READINESS)
    receipt = read(tmp_path, subject.TEST_DIRECTORIES[role] + "/test-receipt.json")
    for key, value in (
        ("exit_code", False),
        ("tests_skipped", 1),
        ("terminal_completion_observed", False),
        ("tested_script_sha256", "0" * 64),
    ):
        changed = copy.deepcopy(receipt)
        changed[key] = value
        with pytest.raises(ValueError):
            subject.test_receipt(changed, role, ready["files"])
    receipt["test_dependencies"]["data/protected.jsonl"] = "0" * 64
    with pytest.raises(ValueError, match="inventory"):
        subject.test_receipt(receipt, role, ready["files"])


@pytest.mark.parametrize("failure", [False, True])
def test_cli_fresh_process_snapshots_failures_and_immutable_outputs(
    subject, mean, tmp_path, failure
):
    pin = environment(tmp_path, subject, mean)
    if failure:
        (tmp_path / subject.BASE / "seed-1731/summary.json").unlink()
    command = [
        sys.executable,
        str(tmp_path / subject.AGGREGATOR),
        "--readiness",
        subject.READINESS,
        "--readiness-sha256",
        pin,
    ]
    preflight = subprocess.run(
        [*command, "--preflight-only"], cwd=tmp_path, capture_output=True, text=True
    )
    assert preflight.returncode == 0, preflight.stderr
    assert json.loads(preflight.stdout)["torch_imported"] is False
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == (1 if failure else 0), result.stderr
    out = tmp_path / subject.BASE / "aggregate.json"
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["complete"] is (not failure) and saved["acceptance"] is False
    assert ("error" in saved) is failure
    for name, expected in saved["snapshot_sha256"].items():
        assert subject.digest(tmp_path / name) == expected
    old_bytes = out.read_bytes()
    again = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode != 0 and "immutable" in again.stderr
    assert out.read_bytes() == old_bytes
