"""Export completed, audited, owner-accepted 32000 evidence revision v2.

Preparation is inert on import. Execution requires all three explicit final pins.
No Torch imports, raw score matrix reads, or prior checkpoint byte reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

CAMPAIGN = "bilinear-budget32000-v1"
STEM = "2026-09-26-bilinear-budget32000"
SEEDS = (1729, 1730, 1731)
PLAN = "5f1e0243b373ad1287c31dcef8a3c1cfa310cac9701795469a68677506e6f444"
PRIOR = "docs/experiments/2026-09-26-symmetric-affine8000-v3-model-version-inventory.json"
PRIOR_SHA = "cac9497ecef4e0e89236a8818404b3159ed9b8e65cdc23b50b2153f6a54820a4"
LIMIT = 512 * 1024
REVISION = "bilinear-budget32000-evidence-v2"
EXECUTION = "bilinear-budget32000-continuation-v2"
AMENDMENT_PATH = "docs/experiments/2026-09-26-bilinear-budget32000-evidence-repair-v2.md"
AMENDMENT = "90aa30088980aeffd999fe6fe4e7349220d163b196ad4eb28f06193b9ce2991c"
READY_PATH = "runs/learning/bilinear-budget32000-readiness-v2/readiness.json"
READY_SHA = "9df87cdafc02a3449b3b78d9bf48be93f0840da4ba64b35c21c09c286c96a69a"
BASE_READY_PATH = "runs/learning/bilinear-budget32000-readiness-v1/readiness.json"
BASE_READY_SHA = "5266fba8ab636590c0654db31b337ba8fd806361d9703c41a1b7260fcb5d766d"
AUDITOR_PATH = "runs/learning/bilinear-budget32000-v1/independent-audit-v2.py"
AUDITOR_SHA = "a04f6e137131c05973f15aa0b3f2e5c07169648e8df95f1d98a9313dc3ba418c"
WRAPPER_PATH = "scripts/refit_bilinear_budget32000_continuation_v2.py"
FAILED_AUDIT = "85b193ad473d77ac10d84aa2e9138ba4d56efd7e9694ef52eeab4ed6e069c188"
FAILED_TERMINAL = "2195369180305bb04f7967b5877e41006f965fd8499d411229193415662fc5ce"
INVENTORY_HELPER_SHA = "8c8955d3b0aa785f5282cc3ba978fdafc75f1e70257786d8298efbb89de7c2d7"


def route(seed):
    require(seed in SEEDS, "declared seed")
    suffix = "" if seed == 1729 else "-v2"
    return {
        "audit": f"independent-audit{suffix}.json",
        "terminal": f"audit-execution-receipt{suffix}.json",
        "stdout": f"audit-stdout{suffix}.txt",
    }


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_rows(rows, header, limit=LIMIT):
    """Keep every reduced row intact; bound encoded bytes, not estimated row size."""
    chunks, current = [], []
    for row in rows:
        candidate = {**header, "rows": [*current, row]}
        if len(current) == 100 or len(encode(candidate)) > limit:
            require(bool(current), "one diagnostic row exceeds publication bound")
            chunks.append({**header, "rows": current})
            current = [row]
            require(len(encode({**header, "rows": current})) <= limit, "oversized diagnostic row")
        else:
            current.append(row)
    if current:
        chunks.append({**header, "rows": current})
    return chunks


def build(root, pins):
    """Read an explicit evidence inventory and prepare outputs; never write here."""
    helper_path = root / "scripts/inventory_model_versions.py"
    helper_bytes = helper_path.read_bytes()
    require(
        hashlib.sha256(helper_bytes).hexdigest() == INVENTORY_HELPER_SHA,
        "authenticated inventory helper before execution",
    )
    inventory = ModuleType("authenticated_inventory_helper")
    inventory.__file__ = str(helper_path)
    # Execute only the bytes just authenticated, avoiding a second source read.
    exec(compile(helper_bytes, str(helper_path), "exec"), inventory.__dict__)
    sha, read = inventory._digest, inventory._read
    run = root / "runs/learning" / CAMPAIGN
    documents = root / "docs/experiments"
    inputs = {helper_path: INVENTORY_HELPER_SHA, Path(__file__): sha(Path(__file__))}
    outputs = {}

    def bound(path, expected):
        path = Path(path)
        require(path.resolve().is_relative_to(root.resolve()), "evidence path escape")
        require(type(expected) is str and len(expected) == 64, "required SHA256")
        require(sha(path) == expected, "evidence identity: " + str(path))
        inputs[path] = expected
        return path

    def read_bound(path, expected):
        return read(bound(path, expected))

    def add(name, value):
        path = documents / name
        require(path not in outputs and not path.exists(), "immutable export: " + name)
        payload = encode(value)
        require(len(payload) <= LIMIT, "bounded publication file: " + name)
        outputs[path] = payload
        return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}

    aggregate = read_bound(run / "aggregate-v2.json", pins["aggregate"])
    audit = read_bound(run / "aggregate-audit-v2.json", pins["audit"])
    decision = read_bound(run / "decision-v2.json", pins["decision"])
    require(
        aggregate["complete"] is True
        and audit["complete"] is True
        and audit["audit_passed"] is True,
        "completed aggregate audit",
    )
    require(
        aggregate["campaign_version"] == CAMPAIGN
        and aggregate["plan_sha256"] == audit["plan_sha256"] == PLAN,
        "campaign identity",
    )
    require(
        audit["summary_sha256"] == pins["aggregate"] and audit["gate"] == aggregate["gate"],
        "audited aggregate",
    )
    require(
        decision["campaign_version"] == CAMPAIGN and decision["evidence_accepted"] is True,
        "owner accepted evidence",
    )
    require(
        decision["evidence_revision"] == REVISION and decision["amendment_sha256"] == AMENDMENT,
        "owner evidence revision",
    )
    require(
        decision["summary_sha256"] == pins["aggregate"]
        and decision["audit_sha256"] == pins["audit"],
        "owner evidence binding",
    )
    require(
        type(decision["fixed_quality_gate_passed"]) is bool
        and decision["fixed_quality_gate_passed"] == aggregate["gate"]["primary_checks_passed"],
        "owner fixed gate",
    )
    require(
        decision["new_checkpoint_count"] == 3
        and decision["fresh_seeds"] == list(SEEDS)
        and decision["model_promoted"] is False,
        "three completed unpromoted children",
    )
    require(
        [e["seed"] for e in aggregate["per_seed"]]
        == [e["seed"] for e in decision["children"]]
        == list(SEEDS),
        "ordered seed coverage",
    )
    ready = read_bound(root / READY_PATH, READY_SHA)
    original = read_bound(root / BASE_READY_PATH, BASE_READY_SHA)
    bound(root / AMENDMENT_PATH, AMENDMENT)
    require(
        ready["ready"] is True
        and ready["evidence_revision"] == REVISION
        and ready["amendment_sha256"] == AMENDMENT
        and len(ready["files"]) == 14,
        "repaired readiness identity",
    )
    repaired_admission = {"path": READY_PATH, "sha256": READY_SHA}
    original_admission = {"path": BASE_READY_PATH, "sha256": BASE_READY_SHA}
    require(
        ready["base_readiness"] == original_admission
        and ready["files"][AUDITOR_PATH] == AUDITOR_SHA,
        "actual repaired auditor",
    )
    for report, revision_key in ((aggregate, "evidence_revision"), (audit, "audit_revision")):
        require(
            report[revision_key] == REVISION
            and report["amendment_sha256"] == AMENDMENT
            and report["base_readiness"] == original_admission
            and report["shared_readiness"] == repaired_admission,
            "v2 report identities",
        )
    require(audit["script_sha256"] == AUDITOR_SHA, "actual aggregate auditor")
    failed = read_bound(run / "seed-1730/independent-audit.json", FAILED_AUDIT)
    failed_terminal = read_bound(run / "seed-1730/audit-execution-receipt.json", FAILED_TERMINAL)
    require(
        failed["audit_passed"] is False
        and type(failed_terminal["exit_code"]) is int
        and failed_terminal["exit_code"] == 1
        and failed_terminal["terminal_completion_observed_by_primary"] is True,
        "preserved original audit rejection",
    )
    failure_provenance = {
        "seed": 1730,
        "training_rerun": False,
        "original_audit_path": "runs/learning/" + CAMPAIGN + "/seed-1730/independent-audit.json",
        "original_audit_sha256": FAILED_AUDIT,
        "original_terminal_sha256": FAILED_TERMINAL,
        "failure": (
            "Equivalent timezone instants rejected by exact predecessor timestamp text comparison."
        ),
        "repair": "Versioned audit compares exact Fraction instants; original evidence unchanged.",
        "amendment_path": AMENDMENT_PATH,
        "amendment_sha256": AMENDMENT,
    }
    prior = read_bound(root / PRIOR, PRIOR_SHA)
    require(prior["registered_final_count"] == 38, "prior inventory count")
    inherited = prior["verified_final_checkpoint_bytes"]
    require(type(inherited) is dict and len(inherited) == 38, "38 prior checkpoint entries")
    expected, added, portable_seeds = dict(inherited), [], []
    for entry, child_decision in zip(aggregate["per_seed"], decision["children"], strict=True):
        seed = entry["seed"]
        folder = run / f"seed-{seed}"
        routed = route(seed)
        summary = read_bound(folder / "summary.json", entry["summary_sha256"])
        checked = read_bound(folder / routed["audit"], entry["audit_sha256"])
        require(
            entry["audit_path"] == (folder / routed["audit"]).relative_to(root).as_posix()
            and entry["audit_receipt_path"]
            == (folder / routed["terminal"]).relative_to(root).as_posix(),
            "explicit per-seed audit routing",
        )
        require(
            summary["complete"] is True
            and summary["final_identity_check"] is True
            and checked["complete"] is True
            and checked["audit_passed"] is True,
            "completed audited child",
        )
        require(
            summary["seed"] == checked["seed"] == seed and summary["plan_sha256"] == PLAN,
            "seed declaration",
        )
        require(
            checked["summary_sha256"] == entry["summary_sha256"]
            and checked["child"] == summary["child"] == entry["child"]
            and checked["gate"] == summary["gate"] == entry["gate"],
            "seed result identity",
        )
        require(
            summary["shared_readiness"] == original_admission,
            "original scientific admission snapshot",
        )
        auditor_sha = (
            original["files"]["runs/learning/bilinear-budget32000-v1/independent-audit.py"]
            if seed == 1729
            else AUDITOR_SHA
        )
        tests_path = (
            f"runs/learning/bilinear-budget32000-auditor-tests-v{1 if seed == 1729 else 2}"
            "/test-receipt.json"
        )
        tests_sha = (original if seed == 1729 else ready)["files"][tests_path]
        require(
            checked["script_sha256"] == entry["auditor_sha256"] == auditor_sha
            and checked["test_receipt_sha256"] == entry["test_receipt_sha256"] == tests_sha
            and checked["shared_readiness"]
            == (original_admission if seed == 1729 else repaired_admission),
            "actual tested audit provenance",
        )
        if seed != 1729:
            require(
                checked["audit_revision"] == REVISION
                and checked["amendment_sha256"] == AMENDMENT
                and checked["base_readiness"] == original_admission,
                "repaired per-seed audit identity",
            )
        terminals = {}
        for kind, name, key in (
            ("execution", "execution-receipt.json", "execution_receipt_sha256"),
            ("audit", routed["terminal"], "audit_receipt_sha256"),
        ):
            terminal = read_bound(folder / name, entry[key])
            terminals[kind] = terminal
            require(
                type(terminal["exit_code"]) is int
                and terminal["exit_code"] == 0
                and terminal["terminal_completion_observed_by_primary"] is True,
                "observed " + kind + " terminal",
            )
            require(
                terminal["seed"] == seed and terminal["summary_sha256"] == entry["summary_sha256"],
                "terminal identity",
            )
            stdout = "primary-stdout.txt" if kind == "execution" else routed["stdout"]
            bound(folder / stdout, terminal["stdout_sha256"])
            if kind == "audit":
                require(
                    terminal["audit_sha256"] == entry["audit_sha256"]
                    and terminal["execution_receipt_sha256"] == entry["execution_receipt_sha256"],
                    "audit terminal binding",
                )
                require(
                    terminal["auditor_sha256"] == auditor_sha
                    and terminal["test_receipt_sha256"] == tests_sha
                    and terminal["cuda_visible_devices"] == "-1",
                    "actual CPU audit terminal",
                )
        continuation = None
        if seed == 1731:
            fields = {
                "execution_wrapper_sha256": ready["files"][WRAPPER_PATH],
                "execution_revision": EXECUTION,
                "amendment_sha256": AMENDMENT,
                "continuation_readiness": repaired_admission,
            }
            require(all(summary[k] == v for k, v in fields.items()), "actual continuation identity")
            require(
                entry["execution_wrapper_sha256"] == fields["execution_wrapper_sha256"],
                "aggregate wrapper identity",
            )
            expected_snapshots = {
                f"runs/learning/{CAMPAIGN}/seed-1731/continuation/" + name: digest
                for name, digest in (ready["files"] | {READY_PATH: READY_SHA}).items()
            }
            require(
                summary["continuation_snapshots"] == expected_snapshots,
                "exact continuation snapshot inventory",
            )
            for name, digest in expected_snapshots.items():
                bound(root / name, digest)
            complete_sha = entry["continuation_complete_sha256"]
            complete = read_bound(folder / "continuation-complete.json", complete_sha)
            require(
                complete["complete"] is True
                and complete["seed"] == 1731
                and complete["root_observed_terminal_required"] is True
                and complete["summary_sha256"] == entry["summary_sha256"]
                and all(complete[k] == v for k, v in fields.items())
                and checked["continuation_complete_sha256"] == complete_sha
                and terminals["execution"]["continuation_complete_sha256"] == complete_sha,
                "observed and audited wrapper completion",
            )
            continuation = {
                **fields,
                "completion_sha256": complete_sha,
                "snapshot_sha256": expected_snapshots,
            }
        artifacts = summary["artifact_sha256"]
        # Explicit named outputs only: no recursive input manifests or raw score reads.
        training = read_bound(folder / "training.json", artifacts["training.json"])
        diagnostics = read_bound(
            folder / "train-diagnostics.json", artifacts["train-diagnostics.json"]
        )
        sidecar = read_bound(
            folder / "checkpoint-final.pt.json", artifacts["checkpoint-final.pt.json"]
        )
        require(
            training["complete"] is True and training["completed_updates"] == 32000,
            "completed 32000 updates",
        )
        history = training["history"]
        require(
            [r["update"] for r in history] == list(range(1, 32001)), "exact full training order"
        )
        parts = []
        for part in range(1, 17):
            start, end = (part - 1) * 2000, part * 2000
            value = {
                "seed": seed,
                "source_training_sha256": artifacts["training.json"],
                "update_start": start + 1,
                "update_end": end,
                "history": history[start:end],
            }
            parts.append(
                {
                    **add(f"{STEM}-training-{seed}-part-{part:02}.json", value),
                    "update_start": start + 1,
                    "update_end": end,
                }
            )
        require(
            [r for p in parts for r in json.loads(outputs[documents / p["path"]])["history"]]
            == history,
            "exact trace reconstruction",
        )
        require(
            diagnostics["complete"] is True
            and diagnostics["descriptive_only"] is True
            and diagnostics["used_for_selection_or_updates"] is False
            and set(diagnostics["endpoints"]) == {"8000", "32000"},
            "train-only diagnostic endpoints",
        )
        diagnostic_summaries = {}
        for point in (8000, 32000):
            endpoint = diagnostics["endpoints"][str(point)]
            rows = endpoint["rows"]
            require(
                endpoint["post_update"] == point
                and endpoint["query_count"] == len(rows) == 1637
                and [r["index"] for r in rows] == list(range(1637)),
                "diagnostic row coverage",
            )
            require(
                {k: v for k, v in endpoint.items() if k != "rows"}
                == checked["training_diagnostics"]["endpoints"][str(point)],
                "independently audited diagnostic summary",
            )
            score_name = f"train-scores-post{point}.json"
            require(
                endpoint["score_artifact"] == score_name
                and endpoint["score_artifact_sha256"] == artifacts[score_name],
                "local score identity",
            )
            row_parts = []
            header = {
                "seed": seed,
                "post_update": point,
                "source_diagnostics_sha256": artifacts["train-diagnostics.json"],
                "raw_score_sha256": artifacts[score_name],
                "descriptive_only": True,
            }
            for part, value in enumerate(bounded_rows(rows, header), start=1):
                ref = add(f"{STEM}-diagnostics-{seed}-post{point}-part-{part:02}.json", value)
                row_parts.append(
                    {
                        **ref,
                        "index_start": value["rows"][0]["index"],
                        "index_end": value["rows"][-1]["index"],
                        "row_count": len(value["rows"]),
                    }
                )
            require(
                [r for p in row_parts for r in json.loads(outputs[documents / p["path"]])["rows"]]
                == rows,
                "exact diagnostic reconstruction",
            )
            diagnostic_summaries[str(point)] = {
                **{k: v for k, v in endpoint.items() if k != "rows"},
                "row_artifacts": row_parts,
                "raw_score_publication_status": (
                    "Hash-bound local only; not copied or reread by exporter."
                ),
            }
        checkpoint = folder / "checkpoint-final.pt"
        relative = checkpoint.relative_to(root).as_posix()
        require(
            relative not in expected and child_decision["checkpoint_path"] == relative,
            "new final checkpoint path",
        )
        for key, value in {
            "checkpoint_sha256": artifacts["checkpoint-final.pt"],
            "sidecar_sha256": artifacts["checkpoint-final.pt.json"],
            "summary_sha256": entry["summary_sha256"],
            "audit_sha256": entry["audit_sha256"],
        }.items():
            require(child_decision[key] == value, "owner child identity: " + key)
        bound(checkpoint, artifacts["checkpoint-final.pt"])
        require(
            sidecar["checkpoint_hash"] == artifacts["checkpoint-final.pt"]
            and sidecar["global_step"] == 32000
            and sidecar["experiment_identity"]["seeds"] == [seed],
            "final checkpoint metadata",
        )
        metadata = sidecar["training_metadata"]
        require(
            metadata["residual_updates"] == 32000 and metadata["parent_training_steps"] == 2000,
            "parent and derivative budgets",
        )
        expected[relative] = artifacts["checkpoint-final.pt"]
        added.append(
            {
                **child_decision,
                "sidecar": sidecar,
                "parent_checkpoint_sha256": metadata["parent_checkpoint_sha256"],
                "parent_training_steps": 2000,
                "residual_updates": 32000,
                "quality_gate_passed": entry["gate"]["primary_checks_passed"],
                "promoted": False,
                "standard_serving_supported": False,
                "durable_weight_archive": None,
                "checkpoint_verification": "fresh byte SHA256; no payload deserialization",
            }
        )
        portable_seeds.append(
            {
                "seed": seed,
                **{
                    k: entry[k]
                    for k in (
                        "summary_sha256",
                        "audit_sha256",
                        "execution_receipt_sha256",
                        "audit_receipt_sha256",
                        "child",
                        "gate",
                    )
                },
                **{
                    k: summary[k]
                    for k in (
                        "architecture",
                        "objective",
                        "inference",
                        "evaluator",
                        "recipe_sha256",
                        "script_sha256",
                        "shared_readiness",
                        "environment",
                        "numerical_settings",
                        "peak_memory",
                        "wall_seconds",
                        "parent",
                        "invariants",
                    )
                },
                "checkpoint": checked["checkpoint"],
                "initial_pre_update_loss": training["initial_pre_update_loss"],
                "final_post_update_loss": training["final_post_update_loss"],
                "training_prefix_diagnostic": training["training_prefix_diagnostic"],
                "source_training_sha256": artifacts["training.json"],
                "training_history_artifacts": parts,
                "training_diagnostics": diagnostic_summaries,
                "independent_training_diagnostics": checked["training_diagnostics"],
                "audit_provenance": {
                    "path": entry["audit_path"],
                    "terminal_path": entry["audit_receipt_path"],
                    "auditor_sha256": auditor_sha,
                    "test_receipt_sha256": tests_sha,
                    "readiness": checked["shared_readiness"],
                },
                "continuation": continuation,
            }
        )
    require(
        {p: expected[p] for p in inherited} == inherited and len(expected) == 41,
        "preserved 38 plus three new finals",
    )
    inventory_ref = add(
        f"{STEM}-model-version-inventory.json",
        {
            "schema_version": 1,
            "inventory_kind": "additive-three-seed-bilinear32000-inherited38-fresh3-v1",
            "observed_at_utc": datetime.now(UTC).isoformat(),
            "prior_inventory_path": PRIOR,
            "prior_inventory_sha256": PRIOR_SHA,
            "prior_registered_count": 38,
            "added_count": 3,
            "registered_final_count": 41,
            "verified_final_checkpoint_bytes": expected,
            "verification_scope": {
                "inherited_prior_entries": 38,
                "prior_weights_rehashed": False,
                "fresh_new_checkpoint_hashes": {
                    e["checkpoint_path"]: e["checkpoint_sha256"] for e in added
                },
                "payloads_deserialized": False,
            },
            "new_artifacts": added,
            "decision_sha256": pins["decision"],
            "exporter_sha256": sha(Path(__file__)),
            "inventory_helper_sha256": sha(root / "scripts/inventory_model_versions.py"),
            "limitations": [
                "The legacy map name includes inherited verification: prior 38 hashes "
                "were carried forward unchanged, not reverified in this export.",
                "All three completed children are registered regardless of quality gate outcome.",
                "Local ignored weights are not published; "
                "hashes do not establish a durable backup.",
            ],
        },
    )
    add(
        f"{STEM}.json",
        {
            "schema_version": 1,
            "date": "2026-09-26",
            "status": "completed audited three-seed screen",
            "campaign_version": CAMPAIGN,
            "evidence_revision": REVISION,
            "amendment_sha256": AMENDMENT,
            "base_readiness": original_admission,
            "shared_readiness": repaired_admission,
            "preserved_failure_provenance": failure_provenance,
            "plan_sha256": PLAN,
            "aggregate_sha256": pins["aggregate"],
            "aggregate_audit_sha256": pins["audit"],
            "decision_sha256": pins["decision"],
            "decision": decision,
            "gate": aggregate["gate"],
            "all_three": aggregate["all_three"],
            "per_seed": portable_seeds,
            "model_version_inventory": {**inventory_ref, "registered_final_count": 41},
            "exporter_sha256": sha(Path(__file__)),
            "protected_test_used": False,
            "standard_serving_supported": False,
            "limitations": [
                "666 observations reuse the same 222 development queries; "
                "they are not 666 independent held-out samples.",
                "Fresh fits start from the three existing original parents; "
                "these are not newly selected parent seeds.",
                "Training diagnostics are descriptive; "
                "no threshold selection or new candidate evaluation.",
                "Instrumented refit timing includes detached diagnostic reduction and "
                "serialization; do not compare it directly with historical uninstrumented time.",
                "Raw training matrices and final weights remain local; reduced rows retain "
                "audited summaries and score hashes, not independently replayable matrices.",
                "Prior 38 checkpoint identities are inherited; "
                "only the three new checkpoint files are freshly hashed.",
                "Batch output creation is exclusive but not atomic; a failed write may leave "
                "partial files. Publication handoff requires verification of every output.",
            ],
        },
    )
    require(
        all(sha(path) == digest for path, digest in inputs.items()),
        "bound input changed during export preparation",
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aggregate", "audit", "decision"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    require(Path.cwd().resolve() == root, "repository root invocation")
    pins = {name: getattr(args, name + "_sha256") for name in ("aggregate", "audit", "decision")}
    outputs = build(root, pins)
    # All inputs, reconstruction and bounds pass before any publication file exists.
    require(all(not path.exists() for path in outputs), "immutable export outputs")
    for path, payload in outputs.items():
        with path.open("xb") as stream:
            stream.write(payload)
    print(
        json.dumps(
            {
                p.relative_to(root).as_posix(): hashlib.sha256(v).hexdigest()
                for p, v in outputs.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
