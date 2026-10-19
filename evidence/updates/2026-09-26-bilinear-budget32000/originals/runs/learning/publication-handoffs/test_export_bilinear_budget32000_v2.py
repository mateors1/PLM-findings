"""Self-contained synthetic exporter checks; no campaign evidence is opened."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, quality=True):
    path = Path(__file__).with_name("export-bilinear-budget32000-v2.py")
    spec = importlib.util.spec_from_file_location("exporter", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def put(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else m.encode(value))
        return digest(path)

    put(
        "scripts/inventory_model_versions.py",
        (ROOT / "scripts/inventory_model_versions.py").read_bytes(),
    )
    monkeypatch.setattr(m, "AMENDMENT", put(m.AMENDMENT_PATH, b"synthetic amendment"))
    original = {
        "files": {
            "runs/learning/bilinear-budget32000-v1/independent-audit.py": "base-auditor",
            "runs/learning/bilinear-budget32000-auditor-tests-v1/test-receipt.json": "base-tests",
        }
    }
    monkeypatch.setattr(m, "BASE_READY_SHA", put(m.BASE_READY_PATH, original))
    names = [
        m.AUDITOR_PATH,
        m.WRAPPER_PATH,
        "runs/learning/bilinear-budget32000-auditor-tests-v2/test-receipt.json",
    ]
    names += [f"synthetic-approved-source-{i}.py" for i in range(11)]
    files = {n: put(n, n.encode()) for n in names}
    monkeypatch.setattr(m, "AUDITOR_SHA", files[m.AUDITOR_PATH])
    old_admission = {"path": m.BASE_READY_PATH, "sha256": m.BASE_READY_SHA}
    ready = {
        "ready": True,
        "evidence_revision": m.REVISION,
        "amendment_sha256": m.AMENDMENT,
        "base_readiness": old_admission,
        "files": files,
    }
    monkeypatch.setattr(m, "READY_SHA", put(m.READY_PATH, ready))
    new_admission = {"path": m.READY_PATH, "sha256": m.READY_SHA}
    prefix = "runs/learning/" + m.CAMPAIGN
    monkeypatch.setattr(
        m,
        "FAILED_AUDIT",
        put(prefix + "/seed-1730/independent-audit.json", {"audit_passed": False}),
    )
    monkeypatch.setattr(
        m,
        "FAILED_TERMINAL",
        put(
            prefix + "/seed-1730/audit-execution-receipt.json",
            {"exit_code": 1, "terminal_completion_observed_by_primary": True},
        ),
    )
    prior = {
        "registered_final_count": 38,
        "verified_final_checkpoint_bytes": {
            f"runs/prior-{i}/checkpoint-final.pt": f"{i:064x}" for i in range(38)
        },
    }
    monkeypatch.setattr(m, "PRIOR_SHA", put(m.PRIOR, prior))
    entries, children = [], []
    gate = {"primary_checks_passed": quality}
    for seed in m.SEEDS:
        folder = f"{prefix}/seed-{seed}"
        artifacts = {}
        checkpoint_path = folder + "/checkpoint-final.pt"
        artifacts["checkpoint-final.pt"] = put(checkpoint_path, b"synthetic weight bytes")
        sidecar = {
            "checkpoint_hash": artifacts["checkpoint-final.pt"],
            "global_step": 32000,
            "experiment_identity": {"seeds": [seed]},
            "training_metadata": {
                "residual_updates": 32000,
                "parent_training_steps": 2000,
                "parent_checkpoint_sha256": "parent",
            },
        }
        artifacts["checkpoint-final.pt.json"] = put(folder + "/checkpoint-final.pt.json", sidecar)
        endpoints = {}
        for point in (8000, 32000):
            score = f"train-scores-post{point}.json"
            artifacts[score] = (
                "0" * 64
            )  # Matrices deliberately absent: exporter must not read them.
            endpoints[str(point)] = {
                "post_update": point,
                "query_count": 1637,
                "aggregate": {"exact_count": 1637},
                "groups": {},
                "minimum_signed_margin": 1.0,
                "minimum_signed_margin_index": 0,
                "score_artifact": score,
                "score_artifact_sha256": artifacts[score],
                "rows": [{"index": i, "zero_margin": 1.0} for i in range(1637)],
            }
        diagnostics = {
            "complete": True,
            "descriptive_only": True,
            "used_for_selection_or_updates": False,
            "endpoints": endpoints,
        }
        artifacts["train-diagnostics.json"] = put(folder + "/train-diagnostics.json", diagnostics)
        training = {
            "complete": True,
            "completed_updates": 32000,
            "history": [
                {
                    "update": i,
                    "pre_update_loss": 1.0 / i,
                    "gradient_finite": True,
                    "parameters_finite": True,
                }
                for i in range(1, 32001)
            ],
            "initial_pre_update_loss": 1.0,
            "final_post_update_loss": 0.00001,
            "training_prefix_diagnostic": {"descriptive_only": True},
        }
        artifacts["training.json"] = put(folder + "/training.json", training)
        summary = {
            "complete": True,
            "final_identity_check": True,
            "seed": seed,
            "plan_sha256": m.PLAN,
            "child": {"exact_count": 222},
            "gate": gate,
            "shared_readiness": old_admission,
            "artifact_sha256": artifacts,
            **{
                k: {}
                for k in (
                    "environment",
                    "numerical_settings",
                    "peak_memory",
                    "parent",
                    "invariants",
                )
            },
            **dict.fromkeys(
                (
                    "architecture",
                    "objective",
                    "inference",
                    "evaluator",
                    "recipe_sha256",
                    "script_sha256",
                ),
                "synthetic",
            ),
            "wall_seconds": 1.0,
        }
        continuation_fields = {
            "execution_wrapper_sha256": files[m.WRAPPER_PATH],
            "execution_revision": m.EXECUTION,
            "amendment_sha256": m.AMENDMENT,
            "continuation_readiness": new_admission,
        }
        if seed == 1731:
            snapshots = {}
            for name, sha in (files | {m.READY_PATH: m.READY_SHA}).items():
                target = folder + "/continuation/" + name
                assert put(target, (tmp_path / name).read_bytes()) == sha
                snapshots[target] = sha
            summary.update(**continuation_fields, continuation_snapshots=snapshots)
        summary_sha = put(folder + "/summary.json", summary)
        completion_sha = None
        if seed == 1731:
            completion_sha = put(
                folder + "/continuation-complete.json",
                {
                    **continuation_fields,
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
            "stdout_sha256": put(folder + "/primary-stdout.txt", b"synthetic"),
        }
        if seed == 1731:
            execution["continuation_complete_sha256"] = completion_sha
        execution_sha = put(folder + "/execution-receipt.json", execution)
        routed = m.route(seed)
        auditor_sha = "base-auditor" if seed == 1729 else m.AUDITOR_SHA
        test_sha = (
            "base-tests"
            if seed == 1729
            else files["runs/learning/bilinear-budget32000-auditor-tests-v2/test-receipt.json"]
        )
        checked = {
            "complete": True,
            "audit_passed": True,
            "seed": seed,
            "summary_sha256": summary_sha,
            "child": summary["child"],
            "gate": gate,
            "script_sha256": auditor_sha,
            "test_receipt_sha256": test_sha,
            "shared_readiness": old_admission if seed == 1729 else new_admission,
            "checkpoint": {},
            "training_diagnostics": {
                "endpoints": {
                    p: {k: v for k, v in e.items() if k != "rows"} for p, e in endpoints.items()
                }
            },
        }
        if seed != 1729:
            checked.update(
                audit_revision=m.REVISION,
                amendment_sha256=m.AMENDMENT,
                base_readiness=old_admission,
            )
        if seed == 1731:
            checked["continuation_complete_sha256"] = completion_sha
        audit_sha = put(folder + "/" + routed["audit"], checked)
        terminal = {
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "seed": seed,
            "summary_sha256": summary_sha,
            "audit_sha256": audit_sha,
            "execution_receipt_sha256": execution_sha,
            "auditor_sha256": auditor_sha,
            "test_receipt_sha256": test_sha,
            "cuda_visible_devices": "-1",
            "stdout_sha256": put(folder + "/" + routed["stdout"], b"synthetic audit"),
        }
        terminal_sha = put(folder + "/" + routed["terminal"], terminal)
        entry = {
            "seed": seed,
            "summary_sha256": summary_sha,
            "audit_sha256": audit_sha,
            "audit_path": folder + "/" + routed["audit"],
            "audit_receipt_path": folder + "/" + routed["terminal"],
            "auditor_sha256": auditor_sha,
            "test_receipt_sha256": test_sha,
            "execution_receipt_sha256": execution_sha,
            "audit_receipt_sha256": terminal_sha,
            "child": summary["child"],
            "gate": gate,
        }
        if seed == 1731:
            entry.update(
                continuation_complete_sha256=completion_sha,
                execution_wrapper_sha256=files[m.WRAPPER_PATH],
            )
        entries.append(entry)
        children.append(
            {
                "seed": seed,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": artifacts["checkpoint-final.pt"],
                "sidecar_sha256": artifacts["checkpoint-final.pt.json"],
                "summary_sha256": summary_sha,
                "audit_sha256": audit_sha,
            }
        )
    aggregate = {
        "complete": True,
        "campaign_version": m.CAMPAIGN,
        "plan_sha256": m.PLAN,
        "evidence_revision": m.REVISION,
        "amendment_sha256": m.AMENDMENT,
        "base_readiness": old_admission,
        "shared_readiness": new_admission,
        "per_seed": entries,
        "gate": gate,
        "all_three": {"synthetic": True},
    }
    agg_sha = put(prefix + "/aggregate-v2.json", aggregate)
    audit = {
        "complete": True,
        "audit_passed": True,
        "summary_sha256": agg_sha,
        "plan_sha256": m.PLAN,
        "gate": gate,
        "audit_revision": m.REVISION,
        "amendment_sha256": m.AMENDMENT,
        "base_readiness": old_admission,
        "shared_readiness": new_admission,
        "script_sha256": m.AUDITOR_SHA,
    }
    audit_sha = put(prefix + "/aggregate-audit-v2.json", audit)
    decision = {
        "campaign_version": m.CAMPAIGN,
        "evidence_revision": m.REVISION,
        "amendment_sha256": m.AMENDMENT,
        "evidence_accepted": True,
        "summary_sha256": agg_sha,
        "audit_sha256": audit_sha,
        "fixed_quality_gate_passed": quality,
        "new_checkpoint_count": 3,
        "fresh_seeds": list(m.SEEDS),
        "model_promoted": False,
        "children": children,
    }
    decision_sha = put(prefix + "/decision-v2.json", decision)
    return (
        m,
        {"aggregate": agg_sha, "audit": audit_sha, "decision": decision_sha},
        put,
        aggregate,
        decision,
    )


@pytest.mark.parametrize("quality", [True, False])
def test_full_synthetic_build_all_children_even_negative(tmp_path, monkeypatch, quality):
    m, pins, _, _, _ = fixture(tmp_path, monkeypatch, quality)
    outputs = m.build(tmp_path, pins)
    assert all(len(v) <= m.LIMIT for v in outputs.values())
    assert len([p for p in outputs if "-training-" in p.name]) == 48
    portable = json.loads(outputs[tmp_path / f"docs/experiments/{m.STEM}.json"])
    inventory = json.loads(
        outputs[tmp_path / f"docs/experiments/{m.STEM}-model-version-inventory.json"]
    )
    assert inventory["registered_final_count"] == 41 and len(inventory["new_artifacts"]) == 3
    assert all(e["quality_gate_passed"] is quality for e in inventory["new_artifacts"])
    assert inventory["verification_scope"]["prior_weights_rehashed"] is False
    assert portable["per_seed"][0]["continuation"] is None
    assert portable["per_seed"][2]["continuation"]["execution_revision"] == m.EXECUTION
    assert not any(p.exists() for p in outputs)  # build itself is read-only.


@pytest.mark.parametrize(
    "mutation",
    ["missing_seed", "rejected_evidence", "wrong_route", "missing_completion", "oversize_row"],
)
def test_synthetic_exporter_rejects_incomplete_or_wrong_evidence(tmp_path, monkeypatch, mutation):
    m, pins, put, aggregate, decision = fixture(tmp_path, monkeypatch)
    prefix = "runs/learning/" + m.CAMPAIGN
    if mutation == "missing_seed":
        decision["children"].pop()
    if mutation == "rejected_evidence":
        decision["evidence_accepted"] = False
    if mutation == "wrong_route":
        aggregate["per_seed"][0]["audit_path"] = prefix + "/seed-1729/independent-audit-v2.json"
        pins["aggregate"] = put(prefix + "/aggregate-v2.json", aggregate)
        audit = json.loads((tmp_path / prefix / "aggregate-audit-v2.json").read_text())
        audit["summary_sha256"] = pins["aggregate"]
        pins["audit"] = put(prefix + "/aggregate-audit-v2.json", audit)
        decision.update(summary_sha256=pins["aggregate"], audit_sha256=pins["audit"])
    if mutation == "missing_completion":
        (tmp_path / prefix / "seed-1731/continuation-complete.json").unlink()
    pins["decision"] = put(prefix + "/decision-v2.json", decision)
    if mutation == "oversize_row":
        with pytest.raises(ValueError):
            m.bounded_rows([{"value": "x" * 600000}], {})
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            m.build(tmp_path, pins)


def test_helper_rejected_before_execution(tmp_path, monkeypatch):
    m, pins, _, _, _ = fixture(tmp_path, monkeypatch)
    helper = tmp_path / "scripts/inventory_model_versions.py"
    sentinel = tmp_path / "untrusted-executed"
    helper.write_text("from pathlib import Path\nPath(" + repr(str(sentinel)) + ").touch()\n")
    with pytest.raises(ValueError, match="before execution"):
        m.build(tmp_path, pins)
    assert not sentinel.exists()


def test_helper_rehashed_after_build(tmp_path, monkeypatch):
    m, pins, _, _, _ = fixture(tmp_path, monkeypatch)
    helper = tmp_path / "scripts/inventory_model_versions.py"
    original_encode = m.encode

    def mutate_after_preparation(value):
        if value.get("status") == "completed audited three-seed screen":
            helper.write_text("changed after helper import")
        return original_encode(value)

    monkeypatch.setattr(m, "encode", mutate_after_preparation)
    with pytest.raises(ValueError, match="bound input changed"):
        m.build(tmp_path, pins)
