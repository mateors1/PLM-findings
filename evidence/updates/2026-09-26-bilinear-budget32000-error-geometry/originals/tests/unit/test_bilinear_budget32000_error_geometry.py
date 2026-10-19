"""Synthetic accepted-chain and CLI checks; no real score reductions."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location(
        "current_geometry", ROOT / "scripts/diagnose_bilinear_budget32000_errors.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def totals(count):
    return {
        "query_count": count,
        "exact_count": count,
        "strict_separation_count": count,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "f1": 1.0,
    }


def setup(tmp_path, monkeypatch, mutation=None):
    m = load()

    def put(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value if isinstance(value, bytes) else (json.dumps(value) + "\n").encode())
        return m.sha(p)

    for name in (m.SOURCE, m.HELPER_PATH, m.TEST, m.HELPER_TEST):
        put(name, (ROOT / name).read_bytes())
    monkeypatch.setattr(m, "PLAN", put(m.PLAN_PATH, b"synthetic plan"))
    ready_values = [
        {
            "files": {
                m.UPSTREAM + "/independent-audit.py": "a" * 64,
                "runs/learning/bilinear-budget32000-auditor-tests-v1/test-receipt.json": "b" * 64,
            }
        },
        {
            "files": {
                m.UPSTREAM + "/independent-audit-v2.py": "c" * 64,
                "runs/learning/bilinear-budget32000-auditor-tests-v2/test-receipt.json": "d" * 64,
                "scripts/refit_bilinear_budget32000_continuation_v2.py": "e" * 64,
            }
        },
    ]
    monkeypatch.setattr(
        m,
        "READINESS",
        {name: put(name, value) for name, value in zip(m.READINESS, ready_values, strict=True)},
    )
    admissions = [{"path": n, "sha256": h} for n, h in m.READINESS.items()]
    entries, audits, owners = [], [], []
    for seed in m.SEEDS:
        folder = f"{m.UPSTREAM}/seed-{seed}/"
        groups = [g for g, n in m.GROUPS.items() for _ in range(n)]
        rows = []
        for i, g in enumerate(groups):
            dimension = "COLOR" if g == "COLOR" else "TYPE"
            scores = [-1.0] * 1025
            scores[-1] = 1.0
            rows.append(
                {
                    "index": i,
                    "subject": f"PKM_{i}",
                    "dimension": dimension,
                    "group": g,
                    "prompt_ids": [1, 1024 + i, 33 if dimension == "COLOR" else 32, 34, 5],
                    "expected_set_ids": [2048],
                    "selected_set_ids": [2048],
                    "symmetric_relation_logits": scores,
                    "strict_separation": True,
                    "metrics": {
                        "exact": True,
                        "false_positive_count": 0,
                        "false_negative_count": 0,
                    },
                }
            )
        child = {
            "complete": True,
            "query_count": 222,
            "product_token_ids": list(range(1024, 2049)),
            "responses": rows,
            "aggregate": totals(222),
            "groups": {g: totals(n) for g, n in m.GROUPS.items()},
        }
        if seed == 1729:
            if mutation == "index_bool":
                rows[0]["index"] = False
            if mutation == "duplicate_index":
                rows[1]["index"] = 0
            if mutation == "prompt_bool":
                rows[0]["prompt_ids"][0] = True
            if mutation == "columns":
                child["product_token_ids"].reverse()
            if mutation == "truth_bool":
                rows[0]["expected_set_ids"] = [True]
            if mutation == "self":
                rows[0]["expected_set_ids"] = [1024]
            if mutation == "duplicate_query":
                rows[1]["prompt_ids"] = rows[0]["prompt_ids"][:]
            if mutation == "group":
                rows[0]["group"] = "TYPE_single"
            if mutation == "nonfinite":
                rows[0]["symmetric_relation_logits"][2] = float("nan")
            if mutation == "nonfp32":
                rows[0]["symmetric_relation_logits"][2] = 0.1
            if mutation == "selection":
                rows[0]["selected_set_ids"] = []
            if mutation == "alignment":
                rows[0]["subject"] = "PKM_OTHER"
        child_sha = put(folder + "child.json", child)
        outcome = {"aggregate": child["aggregate"], "groups": child["groups"]}
        summary = {
            "complete": True,
            "final_identity_check": True,
            "seed": seed,
            "child": outcome,
            "gate": {"pass": True},
            "shared_readiness": admissions[0],
            "artifact_sha256": {"checkpoint-final.pt": "f" * 64, "child.json": child_sha},
        }
        fields = {
            "execution_wrapper_sha256": "e" * 64,
            "execution_revision": "continuation",
            "amendment_sha256": "repair",
            "continuation_readiness": admissions[1],
        }
        if seed == 1731:
            summary.update(fields)
        summary_sha = put(folder + "summary.json", summary)
        completion_sha = None
        if seed == 1731:
            completion_sha = put(
                folder + "continuation-complete.json",
                {
                    **fields,
                    "complete": True,
                    "seed": 1731,
                    "summary_sha256": summary_sha,
                    "root_observed_terminal_required": True,
                },
            )
        execution = {
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "seed": seed,
            "summary_sha256": summary_sha,
        }
        if seed == 1731:
            execution["continuation_complete_sha256"] = completion_sha
        execution_sha = put(folder + "execution-receipt.json", execution)
        suffix = "" if seed == 1729 else "-v2"
        auditor_sha = "a" * 64 if seed == 1729 else "c" * 64
        test_sha = "b" * 64 if seed == 1729 else "d" * 64
        checked = {
            "complete": True,
            "audit_passed": True,
            "seed": seed,
            "summary_sha256": summary_sha,
            "child": copy.deepcopy(outcome),
            "gate": summary["gate"],
            "shared_readiness": admissions[0 if seed == 1729 else 1],
            "script_sha256": auditor_sha,
            "test_receipt_sha256": test_sha,
            "execution_receipt_sha256": execution_sha,
            "checkpoint": {"child_checkpoint_sha256": "f" * 64},
            "input_sha256": {str((tmp_path / (folder + "child.json")).resolve()): child_sha},
        }
        if seed == 1731:
            checked["continuation_complete_sha256"] = completion_sha
        if mutation == "audit_incomplete" and seed == 1729:
            checked["complete"] = False
        if mutation == "metric_drift" and seed == 1729:
            checked["child"]["aggregate"]["f1"] = 0.5
        if mutation == "child_pin" and seed == 1729:
            checked["input_sha256"][str((tmp_path / (folder + "child.json")).resolve())] = "0" * 64
        audit_path = folder + f"independent-audit{suffix}.json"
        audit_sha = put(audit_path, checked)
        terminal = {
            **execution,
            "audit_sha256": audit_sha,
            "execution_receipt_sha256": execution_sha,
            "auditor_sha256": auditor_sha,
            "test_receipt_sha256": test_sha,
            "cuda_visible_devices": "-1",
        }
        terminal_path = folder + f"audit-execution-receipt{suffix}.json"
        if mutation == "terminal_bool" and seed == 1729:
            terminal["exit_code"] = False
        terminal_sha = put(terminal_path, terminal)
        entry = {
            "seed": seed,
            "summary_sha256": summary_sha,
            "audit_sha256": audit_sha,
            "child": outcome,
            "gate": summary["gate"],
            "audit_path": audit_path,
            "audit_receipt_path": terminal_path,
            "auditor_sha256": auditor_sha,
            "test_receipt_sha256": test_sha,
            "execution_receipt_sha256": execution_sha,
            "audit_receipt_sha256": terminal_sha,
        }
        if seed == 1731:
            entry["continuation_complete_sha256"] = completion_sha
        if mutation == "route" and seed == 1730:
            entry["audit_path"] = folder + "independent-audit.json"
        entries.append(entry)
        audits.append(copy.deepcopy(entry))
        owner = {
            "seed": seed,
            "summary_sha256": summary_sha,
            "audit_sha256": audit_sha,
            "checkpoint_path": folder + "checkpoint-final.pt",
            "checkpoint_sha256": "f" * 64,
        }
        if mutation == "checkpoint" and seed == 1729:
            owner["checkpoint_sha256"] = "0" * 64
        owners.append(owner)
    pooled = {"aggregate": totals(666), "groups": {g: totals(n * 3) for g, n in m.GROUPS.items()}}
    if mutation == "pooled":
        pooled["aggregate"]["exact_count"] -= 1
    aggregate = {
        "complete": True,
        "campaign_version": "bilinear-budget32000-v1",
        "per_seed": entries,
        "all_three": pooled,
        "gate": {"pass": True},
        "base_readiness": admissions[0],
        "shared_readiness": admissions[1],
    }
    agg_sha = put(m.UPSTREAM + "/aggregate-v2.json", aggregate)
    audit = {
        "complete": True,
        "audit_passed": True,
        "summary_sha256": agg_sha,
        "per_seed": audits,
        "all_three": pooled,
        "gate": aggregate["gate"],
        "base_readiness": admissions[0],
        "shared_readiness": admissions[1],
    }
    audit_sha = put(m.UPSTREAM + "/aggregate-audit-v2.json", audit)
    decision = {
        "campaign_version": "bilinear-budget32000-v1",
        "evidence_accepted": mutation != "owner_reject",
        "summary_sha256": agg_sha,
        "audit_sha256": audit_sha,
        "children": owners,
    }
    decision_sha = put(m.UPSTREAM + "/decision-v2.json", decision)
    monkeypatch.setattr(
        m, "PINS", dict(zip(m.PINS, (agg_sha, audit_sha, decision_sha), strict=True))
    )
    stdout_sha = put(m.STDOUT, b"synthetic passed")
    receipt = {
        "passed": True,
        "gpu_used": False,
        "diagnostic_executed": False,
        "exit_code": 0,
        "terminal_completion_observed": True,
        "tests_passed": 1,
        "tests_skipped": 0,
        "tested_script_sha256": m.sha(tmp_path / m.SOURCE),
        "test_dependencies": {p: m.sha(tmp_path / p) for p in m.DEPENDENCIES},
        "stdout": {"path": m.STDOUT, "sha256": stdout_sha},
    }
    receipt_sha = put(m.RECEIPT, receipt)
    return m, receipt_sha, put


@pytest.mark.parametrize(
    "mutation",
    [
        "index_bool",
        "duplicate_index",
        "prompt_bool",
        "columns",
        "truth_bool",
        "self",
        "duplicate_query",
        "group",
        "nonfinite",
        "nonfp32",
        "alignment",
        "audit_incomplete",
        "metric_drift",
        "child_pin",
        "terminal_bool",
        "route",
        "checkpoint",
        "owner_reject",
    ],
)
def test_bad_accepted_chain_or_schema_rejected(tmp_path, monkeypatch, mutation):
    m, _, _ = setup(tmp_path, monkeypatch, mutation)
    reader = m.Reader(tmp_path)
    helper = m.load_helper(reader)
    with pytest.raises(ValueError):
        m.preflight(reader, helper)


@pytest.mark.parametrize(
    "mode", ["success", "preflight", "selection", "pooled", "snapshot_drift", "overwrite"]
)
def test_real_entrypoint_synthetic(tmp_path, monkeypatch, mode):
    m, pin, _ = setup(tmp_path, monkeypatch, mode)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "runs/learning" / m.CAMPAIGN
    if mode == "preflight":
        real = m.load_helper

        def forbidden_reductions(reader):
            h = real(reader)
            h._seed = lambda *args: pytest.fail("preflight performed reductions")
            h._reduce = lambda *args: pytest.fail("preflight performed reductions")
            return h

        monkeypatch.setattr(m, "load_helper", forbidden_reductions)
    if mode == "snapshot_drift":
        original = m.preflight

        def drift(reader, helper):
            result = original(reader, helper)
            (tmp_path / m.SOURCE).write_text("changed after admission")
            return result

        monkeypatch.setattr(m, "preflight", drift)
    if mode == "overwrite":
        out.mkdir(parents=True)
        (out / "seed-1729.json").write_text("preserve")
    if mode in {"selection", "pooled", "snapshot_drift", "overwrite"}:
        with pytest.raises(ValueError):
            m.run(tmp_path, tmp_path / m.RECEIPT, pin)
        assert not (out / "summary.json").exists()
        if mode in {"selection", "pooled"}:
            assert json.loads((out / "failure.json").read_text())["complete"] is False
        if mode == "overwrite":
            assert (out / "seed-1729.json").read_text() == "preserve"
    else:
        m.run(tmp_path, tmp_path / m.RECEIPT, pin, mode == "preflight")
        if mode == "preflight":
            assert not out.exists()
        else:
            summary = json.loads((out / "summary.json").read_text())
            assert summary["complete"] and summary["acceptance"] is False
            assert summary["all_three"]["aggregate"]["exact_count"] == 666
            assert summary["fresh_two"]["aggregate"]["query_count"] == 444
            assert "shared_threshold" not in summary["all_three"]["aggregate"]
            assert len(summary["seed_artifacts"]) == 3 and len(summary["snapshot_sha256"]) == 8
            assert all(p.stat().st_size <= m.LIMIT for p in out.iterdir())
            assert not any("checkpoint-final.pt" in p for p in summary["input_sha256"])


@pytest.mark.parametrize(
    "mutation", ["exit_bool", "unobserved", "extra_dependency", "stdout_escape"]
)
def test_receipt_fail_closed_before_extra_path_open(tmp_path, monkeypatch, mutation):
    m, _, put = setup(tmp_path, monkeypatch)
    receipt = json.loads((tmp_path / m.RECEIPT).read_text())
    if mutation == "exit_bool":
        receipt["exit_code"] = False
    if mutation == "unobserved":
        receipt["terminal_completion_observed"] = False
    if mutation == "extra_dependency":
        receipt["test_dependencies"]["data/records.jsonl"] = "0" * 64
    if mutation == "stdout_escape":
        receipt["stdout"]["path"] = "../protected.json"
    pin = put(m.RECEIPT, receipt)
    reader = m.Reader(tmp_path)
    with pytest.raises(ValueError):
        m.test_receipt(reader, tmp_path / m.RECEIPT, pin)
    assert set(reader.inputs) == {m.RECEIPT}


def test_untrusted_helper_never_executed(tmp_path, monkeypatch):
    m, _, _ = setup(tmp_path, monkeypatch)
    sentinel = tmp_path / "executed"
    (tmp_path / m.HELPER_PATH).write_text(
        "from pathlib import Path\nPath(" + repr(str(sentinel)) + ").touch()"
    )
    with pytest.raises(ValueError):
        m.load_helper(m.Reader(tmp_path))
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "path",
    [
        "data/records.jsonl",
        "runs/model/checkpoint-final.pt",
        "data/graph.sqlite",
        "../private.json",
    ],
)
def test_forbidden_path_rejected_before_open(tmp_path, path):
    m = load()
    with pytest.raises(ValueError, match="explicit input path"):
        m.Reader(tmp_path).bind(path, "0" * 64)
