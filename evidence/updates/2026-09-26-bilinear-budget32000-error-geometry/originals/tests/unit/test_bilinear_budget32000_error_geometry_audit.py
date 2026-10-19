"""Synthetic saved-score contracts; no actual campaign scores or neural runtime."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDITOR = ROOT / "runs/learning/bilinear-budget32000-error-geometry-v1/independent-audit.py"


@pytest.fixture
def auditor():
    spec = importlib.util.spec_from_file_location("geometry_audit_test", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def helper(auditor):
    return auditor.load_helper(auditor.Reader(ROOT))


def small_row(helper, scores):
    columns = list(range(1024, 1030))
    truth = [1025, 1026]
    selected = [i for i, z in zip(columns, scores, strict=True) if i != 1024 and z > 0]
    return {
        "index": 0,
        "subject": "PKM_SYNTHETIC",
        "dimension": "TYPE",
        "group": "TYPE_dual",
        "prompt_ids": [1, 1024, 32, 34, 5],
        "expected_set_ids": truth,
        "selected_set_ids": selected,
        "symmetric_relation_logits": scores,
        "metrics": helper.metrics(selected, truth),
        "strict_separation": min(scores[1:3]) > max(scores[3:]),
    }


@pytest.mark.parametrize(
    "scores,category",
    [
        ([99, 1, 2, 0, -1, -2], "exact_zero"),
        ([99, 0, 1, -1, -2, -3], "separated_missing"),
        ([99, 2, 3, 1, 0, -1], "separated_extra"),
        ([99, 0, 2, 0, -1, -2], "overlap_or_tie"),
        ([99, -1, 2, 1, -2, -3], "overlap_or_tie"),
        ([99, 1, 2, -0.0, 0.0, -2], "exact_zero"),
    ],
)
def test_independent_categories_and_zero_boundary(helper, scores, category):
    reduced = helper.reduce_row(small_row(helper, scores), list(range(1024, 1030)))
    assert reduced["category"] == category
    assert reduced["exact"] == (category == "exact_zero")


def test_lowest_id_ties_and_signed_zero(helper):
    reduced = helper.reduce_row(
        small_row(helper, [99, 1, 1, -0.0, 0.0, -1]), list(range(1024, 1030))
    )
    assert reduced["minimum_true_id"] == 1025
    assert reduced["maximum_false_id"] == 1027
    assert reduced["false_positive"] == reduced["false_negative"] == []


def test_shared_interval_need_not_exist_for_separated_rows(helper):
    first = helper.reduce_row(small_row(helper, [99, 0, 1, -1, -2, -3]), list(range(1024, 1030)))
    second = helper.reduce_row(small_row(helper, [99, 2, 3, 1, 0, -1]), list(range(1024, 1030)))
    second["index"] = 1
    assert first["strict_separation"] and second["strict_separation"]
    interval = helper.totals([first, second])["shared_threshold"]
    assert interval == {
        "A": 0,
        "B": 1,
        "exists": False,
        "lower_inclusive": True,
        "upper_exclusive": True,
        "A_query_index": 0,
        "B_query_index": 1,
    }
    assert "shared_threshold" not in helper.totals([first, second], False)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), 0.1, 1e100, True, "1"]
)
def test_invalid_scores_rejected(helper, value):
    row = small_row(helper, [99, 1, 2, 0, -1, -2])
    row["symmetric_relation_logits"][0] = value
    with pytest.raises(ValueError):
        helper.reduce_row(row, list(range(1024, 1030)))


@pytest.mark.parametrize(
    "field", ["selected_set_ids", "expected_set_ids", "strict_separation", "metrics"]
)
def test_saved_row_drift_rejected(helper, field):
    row = small_row(helper, [99, 1, 2, 0, -1, -2])
    if field == "selected_set_ids":
        row[field] = [1025]
    elif field == "expected_set_ids":
        row[field] = [1024, 1025]
    elif field == "strict_separation":
        row[field] = False
    else:
        row[field]["f1"] = 0.5
    with pytest.raises(ValueError):
        helper.reduce_row(row, list(range(1024, 1030)))


def synthetic_child(helper, seed):
    rows = []
    for i in range(222):
        group = "COLOR" if i < 103 else "TYPE_dual" if i < 171 else "TYPE_single"
        dimension = "COLOR" if group == "COLOR" else "TYPE"
        subject = 1024 + i
        truth = [subject + 1]
        scores = [-1.0] * 1025
        scores[i] = 99.0
        scores[i + 1] = 1.0
        if seed == 1729 and i == 104:
            scores[i + 1] = 0.0
        if seed == 1731 and i == 104:
            scores[i + 2] = 1.0
        selected = [1024 + k for k, z in enumerate(scores) if 1024 + k != subject and z > 0]
        rows.append(
            {
                "index": i,
                "subject": f"PKM_SYNTHETIC_{i}",
                "dimension": dimension,
                "group": group,
                "prompt_ids": [1, subject, 33 if dimension == "COLOR" else 32, 34, 5],
                "expected_set_ids": truth,
                "selected_set_ids": selected,
                "symmetric_relation_logits": scores,
                "metrics": helper.metrics(selected, truth),
                "strict_separation": scores[i + 1]
                > max(z for k, z in enumerate(scores) if k not in (i, i + 1)),
            }
        )
    return {
        "complete": True,
        "query_count": 222,
        "product_token_ids": list(range(1024, 2049)),
        "responses": rows,
        "aggregate": helper.source_totals(rows),
        "groups": {
            g: helper.source_totals([r for r in rows if r["group"] == g]) for g in helper.GROUPS
        },
    }


def build_fixture(tmp, a, helper, monkeypatch):
    """Real auditor orchestration over synthetic JSONs; only fixed upstream pins replaced."""

    def put(rel, value):
        data = value if isinstance(value, bytes) else a.encoded(value)
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return a.sha_bytes(data)

    for rel in [a.AUDITOR, a.AUDITOR_TEST, a.HELPER, a.PLAN_PATH, a.PRIMARY_HELPER, a.PRIMARY]:
        put(rel, (ROOT / rel).read_bytes())
    for rel in [a.PRIMARY_TEST, a.PRIMARY_HELPER_TEST]:
        put(rel, b"# synthetic contract fixture; never executed\n")
    readiness = {}
    for version, path in enumerate(a.READINESS, 1):
        suffix = "" if version == 1 else "-v2"
        files = {
            f"{a.UPSTREAM}/independent-audit{suffix}.py": str(version) * 64,
            f"runs/learning/bilinear-budget32000-auditor-tests-v{version}/test-receipt.json": str(
                version + 2
            )
            * 64,
        }
        if version == 2:
            files["scripts/refit_bilinear_budget32000_continuation_v2.py"] = "a" * 64
        readiness[path] = put(path, {"ready": True, "files": files})
    monkeypatch.setattr(a, "READINESS", readiness)
    admissions = [{"path": p, "sha256": h} for p, h in readiness.items()]
    children = {s: synthetic_child(helper, s) for s in a.SEEDS}
    entries, owners = [], []
    for seed, child in children.items():
        folder = f"{a.UPSTREAM}/seed-{seed}"
        suffix = "" if seed == 1729 else "-v2"
        version = 0 if seed == 1729 else 1
        ch = put(folder + "/child.json", child)
        ck = str(seed)[-1] * 64
        metrics = {"aggregate": child["aggregate"], "groups": child["groups"]}
        gate = {"primary_checks_passed": True}
        summary = {
            "complete": True,
            "final_identity_check": True,
            "seed": seed,
            "shared_readiness": admissions[0],
            "child": metrics,
            "gate": gate,
            "artifact_sha256": {"child.json": ch, "checkpoint-final.pt": ck},
        }
        if seed == 1731:
            summary.update(
                execution_wrapper_sha256="a" * 64,
                execution_revision="synthetic-continuation",
                amendment_sha256="b" * 64,
                continuation_readiness=admissions[1],
            )
        sh = put(folder + "/summary.json", summary)
        execution = {
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "seed": seed,
            "summary_sha256": sh,
        }
        completion = None
        if seed == 1731:
            completion = {
                "complete": True,
                "seed": seed,
                "summary_sha256": sh,
                "root_observed_terminal_required": True,
                **{
                    k: summary[k]
                    for k in (
                        "execution_wrapper_sha256",
                        "execution_revision",
                        "amendment_sha256",
                        "continuation_readiness",
                    )
                },
            }
            completion_hash = put(folder + "/continuation-complete.json", completion)
            execution["continuation_complete_sha256"] = completion_hash
        eh = put(folder + "/execution-receipt.json", execution)
        checked = {
            "complete": True,
            "audit_passed": True,
            "seed": seed,
            "summary_sha256": sh,
            "child": metrics,
            "gate": gate,
            "checkpoint": {"child_checkpoint_sha256": ck},
            "input_sha256": {str((tmp / folder / "child.json").resolve()): ch},
            "shared_readiness": admissions[version],
            "script_sha256": str(version + 1) * 64,
            "test_receipt_sha256": str(version + 3) * 64,
            "execution_receipt_sha256": eh,
        }
        if completion:
            checked["continuation_complete_sha256"] = completion_hash
        ah = put(folder + f"/independent-audit{suffix}.json", checked)
        terminal = {
            **execution,
            "audit_sha256": ah,
            "execution_receipt_sha256": eh,
            "auditor_sha256": checked["script_sha256"],
            "test_receipt_sha256": checked["test_receipt_sha256"],
            "cuda_visible_devices": "-1",
        }
        th = put(folder + f"/audit-execution-receipt{suffix}.json", terminal)
        entry = {
            "seed": seed,
            "summary_sha256": sh,
            "audit_path": folder + f"/independent-audit{suffix}.json",
            "audit_sha256": ah,
            "audit_receipt_path": folder + f"/audit-execution-receipt{suffix}.json",
            "audit_receipt_sha256": th,
            "execution_receipt_sha256": eh,
            "auditor_sha256": checked["script_sha256"],
            "test_receipt_sha256": checked["test_receipt_sha256"],
            "child": metrics,
            "gate": gate,
        }
        if completion:
            entry["continuation_complete_sha256"] = completion_hash
        entries.append(entry)
        owners.append(
            {
                "seed": seed,
                "summary_sha256": sh,
                "audit_sha256": ah,
                "checkpoint_path": folder + "/checkpoint-final.pt",
                "checkpoint_sha256": ck,
                "exact_count": child["aggregate"]["exact_count"],
            }
        )
    flat = [r for c in children.values() for r in c["responses"]]
    all_three = {
        "aggregate": helper.source_totals(flat),
        "groups": {
            g: helper.source_totals([r for r in flat if r["group"] == g]) for g in helper.GROUPS
        },
    }
    aggregate = {
        "campaign_version": "bilinear-budget32000-v1",
        "gate": {"primary_checks_passed": True},
        "complete": True,
        "acceptance": False,
        "protected_test_used": False,
        "all_three": all_three,
        "per_seed": entries,
        "input_sha256": {
            f"{a.UPSTREAM}/seed-{s}/child.json": a.sha_bytes(a.encoded(children[s]))
            for s in a.SEEDS
        },
        "base_readiness": admissions[0],
        "shared_readiness": admissions[1],
    }
    aggregate_hash = put(a.UPSTREAM + "/aggregate-v2.json", aggregate)
    aggregate_audit = {**aggregate, "audit_passed": True, "summary_sha256": aggregate_hash}
    audit_hash = put(a.UPSTREAM + "/aggregate-audit-v2.json", aggregate_audit)
    decision = {
        "campaign_version": "bilinear-budget32000-v1",
        "summary_sha256": aggregate_hash,
        "audit_sha256": audit_hash,
        "evidence_accepted": True,
        "fixed_quality_gate_passed": True,
        "model_promoted": False,
        "protected_test_used": False,
        "children": owners,
    }
    decision_hash = put(a.UPSTREAM + "/decision-v2.json", decision)
    monkeypatch.setattr(
        a,
        "PINS",
        {
            "aggregate-v2.json": aggregate_hash,
            "aggregate-audit-v2.json": audit_hash,
            "decision-v2.json": decision_hash,
        },
    )

    def receipt(role):
        primary = role == "primary"
        script = a.PRIMARY if primary else a.AUDITOR
        deps = (
            {a.PRIMARY, a.PRIMARY_TEST, a.PRIMARY_HELPER, a.PRIMARY_HELPER_TEST, a.PLAN_PATH}
            if primary
            else a.AUDITOR_DEPS
        )
        stdout = a.PREP + ("/stdout.txt" if primary else "/auditor-stdout.txt")
        stdout_hash = put(stdout, b"synthetic test fixture completed\n")
        hashes = {p: a.sha_bytes((tmp / p).read_bytes()) for p in deps}
        value = {
            "passed": True,
            "gpu_used": False,
            "diagnostic_executed": False,
            "exit_code": 0,
            "terminal_completion_observed": True,
            "tests_passed": 1,
            "tests_skipped": 0,
            "tested_script_sha256": hashes[script],
            "test_dependencies": hashes,
            "stdout": {"path": stdout, "sha256": stdout_hash},
        }
        return put(
            a.PREP + ("/test-receipt.json" if primary else "/auditor-test-receipt.json"), value
        )

    primary_pin, audit_pin = receipt("primary"), receipt("auditor")
    reader = a.Reader(tmp)
    reader.bound(a.PLAN_PATH, a.PLAN_SHA, False)
    admitted, digests, upstream = a.authenticate_children(reader)
    a.test_receipt(reader, "primary", primary_pin)
    reports = {s: helper.seed_report(s, admitted[s], digests[s]) for s in a.SEEDS}
    sources = {
        "script.py": a.PRIMARY,
        "helper.py": a.PRIMARY_HELPER,
        "plan.md": a.PLAN_PATH,
        "test.py": a.PRIMARY_TEST,
        "helper-test.py": a.PRIMARY_HELPER_TEST,
        "test-receipt.json": a.PREP + "/test-receipt.json",
        "test-stdout.txt": a.PREP + "/stdout.txt",
    }
    snapshots = {
        a.BASE + "/summary." + suffix: put(a.BASE + "/summary." + suffix, (tmp / path).read_bytes())
        for suffix, path in sources.items()
    }
    snapshots[a.BASE + "/summary.inputs.json"] = put(a.BASE + "/summary.inputs.json", reader.inputs)
    summary = {
        "campaign_version": a.CAMPAIGN,
        "complete": True,
        "acceptance": False,
        "script_sha256": reader.inputs[a.PRIMARY],
        "helper_sha256": a.PRIMARY_HELPER_SHA,
        "plan_sha256": a.PLAN_SHA,
        "test_receipt_sha256": primary_pin,
        "input_sha256": reader.inputs,
        "snapshot_sha256": snapshots,
        "upstream": upstream,
        "seed_artifacts": [
            {
                "seed": s,
                "path": f"{a.BASE}/seed-{s}.json",
                "sha256": put(f"{a.BASE}/seed-{s}.json", reports[s]),
            }
            for s in a.SEEDS
        ],
        "fresh_two": helper.pool(reports, (1730, 1731)),
        "all_three": helper.pool(reports, a.SEEDS),
        "overlap": helper.overlap(reports),
    }
    sh = put(a.BASE + "/summary.json", summary)
    stdout_hash = put(a.BASE + "/primary-stdout.txt", b"synthetic primary result\n")
    put(
        a.BASE + "/execution-receipt.json",
        {
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "summary_sha256": sh,
            "script_sha256": summary["script_sha256"],
            "test_receipt_sha256": primary_pin,
            "stdout_sha256": stdout_hash,
        },
    )
    return {
        "put": put,
        "primary_pin": primary_pin,
        "audit_pin": audit_pin,
        "summary": summary,
        "children": children,
        "reports": reports,
    }


def test_complete_synthetic_audit_lifecycle(tmp_path, auditor, helper, monkeypatch):
    fixture = build_fixture(tmp_path, auditor, helper, monkeypatch)
    checked = auditor.audit(tmp_path, fixture["primary_pin"], fixture["audit_pin"])
    assert checked["audit_passed"] and not checked["acceptance"]
    assert checked["all_three"]["aggregate"]["exact_count"] == 664
    assert checked["overlap"]["failure_seed_histogram"] == {"1729,1731": 1, "none": 221}
    assert not any("checkpoint-final" in p for p in checked["input_sha256"])
    assert not any("training" in p or "train-scores" in p for p in checked["input_sha256"])


def test_actual_cli_in_fresh_process_uses_real_stages(tmp_path, auditor, helper, monkeypatch):
    f = build_fixture(tmp_path, auditor, helper, monkeypatch)
    launcher = tmp_path / "synthetic_cli.py"
    launcher.write_text(
        "import importlib.util,sys,contextlib,io,json\nfrom pathlib import Path\n"
        "s=importlib.util.spec_from_file_location('auditor',"
        + repr(str(tmp_path / auditor.AUDITOR))
        + ")\nm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nm.PINS="
        + repr(auditor.PINS)
        + "\nm.READINESS="
        + repr(auditor.READINESS)
        + "\nroot=Path.cwd()\nfolder=root/m.BASE\n"
        "if not (folder/'independent-audit.json').exists():\n"
        " s=importlib.util.spec_from_file_location('primary',root/m.PRIMARY)\n"
        " p=importlib.util.module_from_spec(s);s.loader.exec_module(p)\n"
        " p.PINS=m.PINS;p.READINESS=m.READINESS\n"
        " names=['summary.json',*[f'seed-{seed}.json' for seed in m.SEEDS],"
        "*[f'summary.{suffix}' for suffix in ['script.py','helper.py','plan.md',"
        "'test.py','helper-test.py','test-receipt.json','test-stdout.txt','inputs.json']]]\n"
        " for name in names:\n"
        "  target=(folder/name).resolve();assert target.is_relative_to(root);target.unlink()\n"
        " captured=io.StringIO()\n"
        " with contextlib.redirect_stdout(captured):\n"
        "  p.run(root,root/p.RECEIPT," + repr(f["primary_pin"]) + ")\n"
        " output=captured.getvalue().encode();(folder/'primary-stdout.txt').write_bytes(output)\n"
        " terminal=json.loads((folder/'execution-receipt.json').read_text())\n"
        " terminal['summary_sha256']=m.sha_bytes((folder/'summary.json').read_bytes())\n"
        " terminal['stdout_sha256']=m.sha_bytes(output)\n"
        " (folder/'execution-receipt.json').write_bytes(m.encoded(terminal))\n"
        + "\nsys.argv=['auditor','--summary',"
        + repr(str(tmp_path / auditor.BASE / "summary.json"))
        + ",'--primary-test-receipt-sha256',"
        + repr(f["primary_pin"])
        + ",'--audit-test-receipt-sha256',"
        + repr(f["audit_pin"])
        + "]\nm.main()\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(launcher)], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    out = tmp_path / auditor.BASE / "independent-audit.json"
    before = out.read_bytes()
    second = subprocess.run(
        [sys.executable, str(launcher)], capture_output=True, text=True, cwd=tmp_path
    )
    assert second.returncode != 0 and "immutable audit output" in second.stderr
    assert out.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        "row",
        "interval",
        "pool",
        "overlap",
        "output_route",
        "inputs",
        "snapshot",
        "terminal",
        "failure",
    ],
)
def test_tampered_saved_outputs_rejected(tmp_path, auditor, helper, monkeypatch, change):
    f = build_fixture(tmp_path, auditor, helper, monkeypatch)
    put = f["put"]
    summary = copy.deepcopy(f["summary"])
    if change in ("row", "interval"):
        saved = copy.deepcopy(f["reports"][1729])
        if change == "row":
            saved["rows"][104]["a"] = 2.0
        else:
            saved["aggregate"]["shared_threshold"]["exists"] = False
        summary["seed_artifacts"][0]["sha256"] = put(auditor.BASE + "/seed-1729.json", saved)
    elif change == "pool":
        summary["all_three"]["aggregate"]["exact_count"] += 1
    elif change == "overlap":
        summary["overlap"]["failure_seed_histogram"]["none"] += 1
    elif change == "output_route":
        summary["seed_artifacts"][0]["path"] = "forbidden/child.json"
    elif change == "inputs":
        summary["input_sha256"]["forbidden/checkpoint.pt"] = "a" * 64
    elif change == "snapshot":
        summary["snapshot_sha256"][auditor.BASE + "/summary.script.py"] = "a" * 64
    elif change == "failure":
        put(auditor.BASE + "/failure.json", {"failed": True})
    sh = put(auditor.BASE + "/summary.json", summary)
    terminal_path = tmp_path / auditor.BASE / "execution-receipt.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["summary_sha256"] = sh
    if change == "terminal":
        terminal["terminal_completion_observed_by_primary"] = False
    put(auditor.BASE + "/execution-receipt.json", terminal)
    with pytest.raises(ValueError):
        auditor.audit(tmp_path, f["primary_pin"], f["audit_pin"])


def test_helper_authenticated_before_execution(tmp_path, auditor):
    path = tmp_path / auditor.HELPER
    path.parent.mkdir(parents=True)
    path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="bound input"):
        auditor.load_helper(auditor.Reader(tmp_path))


def test_allowlist_and_final_drift(tmp_path, auditor):
    reader = auditor.Reader(tmp_path)
    with pytest.raises(ValueError, match="unadmitted"):
        reader.read("checkpoint.pt")
    with pytest.raises(ValueError):
        reader.admit("../escape.json")
    (tmp_path / "fixture.json").write_text("{}")
    reader.bound("fixture.json")
    (tmp_path / "fixture.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="bound input"):
        reader.unchanged()


def test_output_bound_and_exclusive_creation(tmp_path, auditor):
    p = tmp_path / "audit.json"
    auditor.write(p, {"ok": True})
    with pytest.raises(FileExistsError):
        auditor.write(p, {"ok": False})
    with pytest.raises(ValueError, match="bounded"):
        auditor.write(tmp_path / "large.json", {"text": "x" * auditor.LIMIT})


@pytest.mark.parametrize(
    "change", ["incomplete", "failed_audit", "unaccepted", "audit_binding", "seed_drift"]
)
def test_forged_accepted_chain_rejected(auditor, change):
    aggregate = {
        "complete": True,
        "acceptance": False,
        "protected_test_used": False,
        "all_three": {},
        "per_seed": [{"seed": s} for s in auditor.SEEDS],
    }
    audit = {
        **copy.deepcopy(aggregate),
        "audit_passed": True,
        "summary_sha256": auditor.PINS["aggregate-v2.json"],
    }
    decision = {
        "evidence_accepted": True,
        "fixed_quality_gate_passed": True,
        "model_promoted": False,
        "protected_test_used": False,
        "summary_sha256": auditor.PINS["aggregate-v2.json"],
        "audit_sha256": auditor.PINS["aggregate-audit-v2.json"],
        "children": [{"seed": s} for s in auditor.SEEDS],
    }
    if change == "incomplete":
        aggregate["complete"] = False
    elif change == "failed_audit":
        audit["audit_passed"] = False
    elif change == "unaccepted":
        decision["evidence_accepted"] = False
    elif change == "audit_binding":
        decision["audit_sha256"] = "0" * 64
    else:
        audit["per_seed"][0]["seed"] = 999
    with pytest.raises(ValueError):
        auditor.accepted(aggregate, audit, decision)


@pytest.mark.parametrize("change", ["audit_route", "checkpoint_binding", "child_metrics"])
def test_seed_chain_semantics_reject_decoded_fixture_tampering(
    tmp_path, auditor, helper, monkeypatch, change
):
    build_fixture(tmp_path, auditor, helper, monkeypatch)
    reader = auditor.Reader(tmp_path)
    real_read = reader.read

    def mutated_read(relative, expected=None, json_value=True):
        value = real_read(relative, expected, json_value)
        # Injection after byte admission tests semantic defenses independently of hash rejection.
        if change == "audit_route" and relative.endswith(
            ("aggregate-v2.json", "aggregate-audit-v2.json")
        ):
            value["per_seed"][1]["audit_path"] = (
                auditor.UPSTREAM + "/seed-1730/independent-audit.json"
            )
        if change == "checkpoint_binding" and relative.endswith("decision-v2.json"):
            value["children"][0]["checkpoint_sha256"] = "f" * 64
        if change == "child_metrics" and relative.endswith("seed-1729/child.json"):
            value["aggregate"]["exact_count"] -= 1
        return value

    monkeypatch.setattr(reader, "read", mutated_read)
    with pytest.raises(ValueError):
        auditor.authenticate_children(reader)
