"""CPU contract checks with synthetic rows; no neural forward or GPU imports."""

from __future__ import annotations

import copy
import importlib.util
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "pair_composition_shapes", ROOT / "scripts/verify_pair_composition_shapes.py"
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.fixture(scope="module")
def archived():
    directory = ROOT / "runs/learning/pair-composition-integration-v1"
    required = {
        directory / "independent-audit.py": (
            "1d4f1fbe070e06b841cf695c0fb673bfbca79b7390537f13cd496464950bb98a"
        ),
        directory / "summary.script.py": (
            "e64b4d309c4cffee09106df46acc7f2546f423b8c1c51da674166524045c7036"
        ),
    }
    for path, digest in required.items():
        if path.exists():
            verifier._bind(path, digest, {})
    if any(not path.is_file() for path in required):
        pytest.skip("Optional authenticated local archive helpers are absent from this checkout")
    audit = verifier._load(
        directory / "independent-audit.py",
        "1d4f1fbe070e06b841cf695c0fb673bfbca79b7390537f13cd496464950bb98a",
        {},
        "_shape_test_audit",
    )
    transport = verifier._load(
        directory / "summary.script.py",
        "e64b4d309c4cffee09106df46acc7f2546f423b8c1c51da674166524045c7036",
        {},
        "_shape_test_transport",
    )
    return audit, transport


def test_absent_optional_archive_skips_but_existing_mismatch_never_skips(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        archived.__wrapped__()
    directory = tmp_path / "runs/learning/pair-composition-integration-v1"
    directory.mkdir(parents=True)
    (directory / "summary.script.py").write_bytes(b"changed archived helper")
    with pytest.raises(ValueError, match="identity_mismatch"):
        archived.__wrapped__()


def composed(value=1.0, eligible=True):
    originals = [[1024], [1025], [1026], [1027]]
    slots = []
    for index, ranks in enumerate(verifier._ORDER, 1):
        ids = sorted({token for rank in ranks for token in originals[rank - 1]})
        slots.append(
            {
                "slot": index,
                "kind": "original" if len(ranks) == 1 else "pair_composition",
                "source_ranks": list(ranks),
                "set_ids": ids,
                "source_eligible": eligible,
                "score": value * len(ids),
            }
        )
    logits = [value] * 4 + [-1.0] * 1021
    baseline = {
        "slots": slots,
        "source_paths": [{"token_ids": [1, 2], "error": None}] * 4,
        "policy": "synthetic",
    }
    return verifier._serial_composition(baseline, logits, verifier._COLUMNS), logits


def test_same_shape_exact_and_cross_shape_difference():
    offline, _ = composed(2.0)
    serial, _ = composed(1.0)
    verifier._same_shape(serial, offline, serial)
    altered = copy.deepcopy(serial)
    altered["slots"][0]["score"] = math.nextafter(altered["slots"][0]["score"], math.inf)
    with pytest.raises(ValueError, match="same_shape_score_mismatch"):
        verifier._same_shape(altered, offline, serial)


@pytest.mark.parametrize(
    "field",
    [
        "source_paths",
        "set_ids",
        "source_ranks",
        "source_eligible",
        "policy",
        "selected_slot",
        "error",
    ],
)
def test_non_score_fields_cannot_hide_behind_score_projection(field):
    original, _ = composed()
    altered = copy.deepcopy(original)
    if field == "source_paths":
        altered[field][0]["token_ids"] = [9, 2]
    elif field == "error":
        altered["source_paths"][0]["error"] = "failure"
    elif field in ("set_ids", "source_ranks", "source_eligible"):
        altered["slots"][0][field] = False if field == "source_eligible" else [1028]
    else:
        altered[field] = 6 if field == "selected_slot" else "different"
    with pytest.raises(ValueError, match="cross_shape_output_mismatch"):
        verifier._same_shape(altered, original, original)


def test_canonical_duplicate_tie_and_all_invalid_rules():
    original, logits = composed(0.0)
    assert original["selected_slot"] == 1
    invalid, _ = composed(-1.0, False)
    assert invalid["selected_slot"] == 1 and invalid["fallback_no_valid_source"]
    original["slots"][0]["set_ids"] *= 2
    with pytest.raises(ValueError, match="selection_arithmetic_mismatch"):
        verifier._serial_composition(original, logits, verifier._COLUMNS)
    original, logits = composed()
    original["slots"][4]["set_ids"] = [1024]
    with pytest.raises(ValueError, match="selection_arithmetic_mismatch"):
        verifier._serial_composition(original, logits, verifier._COLUMNS)


def test_identical_set_does_not_allow_selected_slot_drift():
    original, logits = composed(0.0)
    for slot in original["slots"]:
        slot["set_ids"] = [1024]
    original = verifier._serial_composition(original, logits, verifier._COLUMNS)
    changed = copy.deepcopy(original)
    changed.update(selected_slot=2, selected_source_ranks=[2])
    assert changed["selected_set_ids"] == original["selected_set_ids"]
    with pytest.raises(ValueError, match="cross_shape_output_mismatch"):
        verifier._same_shape(changed, original, original)


@pytest.mark.parametrize("value", [math.nan, math.inf, 0.1, True, 1e100])
def test_invalid_logit_precision_and_values(value):
    logits = [0.0] * 1025
    logits[0] = value
    with pytest.raises(ValueError, match="invalid_serial_reference"):
        verifier._logits(logits, verifier._COLUMNS)


def test_columns_partition_count_order_and_seed():
    rows = [{"subject": "A", "dimension": "TYPE"}, {"subject": "B", "dimension": "COLOR"}]
    verifier._coverage(rows, rows, 1729, 2)
    for wrong in (rows[:1], rows[::-1], [rows[0], rows[0]]):
        with pytest.raises(ValueError, match="coverage_mismatch"):
            verifier._coverage(wrong, rows, 1729, 2)
    with pytest.raises(ValueError, match="coverage_mismatch"):
        verifier._coverage(rows, rows, 99, 2)
    columns = verifier._COLUMNS.copy()
    columns[:2] = columns[:2][::-1]
    with pytest.raises(ValueError, match="coverage_mismatch"):
        verifier._logits([0.0] * 1025, columns)


def test_failed_repeats_diagnosis_and_numerical_settings():
    verifier._repeated([1.0], [1.0], [1.0])
    for repeat, diagnosis in (([2.0], [1.0]), ([1.0], [2.0])):
        with pytest.raises(ValueError, match="invalid_serial_reference"):
            verifier._repeated([1.0], repeat, diagnosis)
    with pytest.raises(ValueError, match="identity_mismatch"):
        verifier._settings(
            {"float32_matmul_precision": "high"}, {"float32_matmul_precision": "highest"}
        )


def test_identity_and_archive_member_drift(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    files = {
        "src/a.py": b"pass",
        "pyproject.toml": b"project",
        "uv.lock": b"lock",
        "configs/a.yaml": b"a: true",
    }
    for name, payload in files.items():
        (tmp_path / name).write_bytes(payload)
    source, configs = tmp_path / "source.zip", tmp_path / "configs.zip"
    for path, members in ((source, list(files)[:3]), (configs, list(files)[3:])):
        with zipfile.ZipFile(path, "w") as archive:
            for name in members:
                archive.writestr(name, files[name])
    verifier._inventory(tmp_path, source, configs)
    pinned = verifier._sha(tmp_path / "src/a.py")
    (tmp_path / "src/a.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity_mismatch"):
        verifier._bind(tmp_path / "src/a.py", pinned, {})
    with pytest.raises(ValueError, match="identity_mismatch"):
        verifier._inventory(tmp_path, source, configs)
    (tmp_path / "src/a.py").write_bytes(b"pass")
    (tmp_path / "src/new.py").write_bytes(b"pass")
    with pytest.raises(ValueError, match="identity_mismatch"):
        verifier._inventory(tmp_path, source, configs)
    (tmp_path / "src/new.py").unlink()
    (tmp_path / "configs/a.yaml").write_bytes(b"a: false")
    with pytest.raises(ValueError, match="identity_mismatch"):
        verifier._inventory(tmp_path, source, configs)


@pytest.mark.parametrize("completed", [0, 1])
def test_forward_exception_preserves_partial_rows_and_original_error(tmp_path, completed):
    composition, logits = composed()
    queries = [{"subject": str(i), "dimension": "TYPE"} for i in range(2)]
    report = {
        "checkpoint_hash": "checkpoint",
        "config_hash": "config",
        "runtime_environment_sha256": "runtime",
        "responses": [],
        "repeat_logits": [],
        "query_count": 0,
        "reference_forward_count": 0,
        "repeat_forward_count": 0,
        "complete": False,
    }
    item = {"seed": 1730, "offline": {"responses": [{"composition": composition}] * 2}}
    calls = 0

    def forward(query):
        nonlocal calls
        if calls == completed:
            raise RuntimeError("original forward failure")
        calls += 1
        return [1, 2, 3, 4, 5], logits

    def cleanup():
        raise RuntimeError("secondary cleanup failure")

    output = tmp_path / "partial.json"
    with pytest.raises(RuntimeError, match="original forward failure"):
        verifier._collect_reference(report, queries, forward, item, {}, output, cleanup)
    saved = verifier._read(output)
    assert saved["complete"] is False and saved["query_count"] == completed
    assert len(saved["responses"]) == completed
    assert "incomplete_execution" in saved["error"]
    assert "secondary cleanup failure" in saved["cleanup_error"]


def test_shutdown_failure_stops_before_next_job():
    reached = []
    with pytest.raises(ValueError, match="owned_server_shutdown_failure"):
        for seed in (1730, 1731):
            reached.append(seed)
            verifier._stopped(
                {
                    "server_stopped": True,
                    "disabled_check": {"server_stopped": True},
                    "failure_check": {"server_stopped": False},
                }
            )
    assert reached == [1730]


def test_release_requires_pinned_strict_booleans_and_no_output_overwrite(tmp_path):
    path = tmp_path / "release.json"
    verifier._write(path, {"gpu_released": 1, "owned_servers_stopped": True})
    with pytest.raises(ValueError, match="identity_mismatch"):
        verifier._release(path, verifier._sha(path), {})
    with pytest.raises(ValueError, match="identity_mismatch"):
        verifier._release(None, None, {})
    with pytest.raises(FileExistsError):
        verifier._write(path, {})
    output = tmp_path / "summary.json"
    (tmp_path / "independent-audit.py").write_text("# independently owned")
    verifier._refuse(output)
    verifier._write(tmp_path / "serial-1729.json", {})
    with pytest.raises(ValueError, match="incomplete_execution"):
        verifier._refuse(output)


def test_accounting_never_promotes_reuse_or_missing_audit():
    receipts = [
        {
            "seed": seed,
            "complete": True,
            "query_count": 222,
            "evidence_origin": "reused" if seed == 1729 else "fresh",
        }
        for seed in verifier._SEEDS
    ]
    references = [
        {"seed": seed, "complete": True, "query_count": 222, "repeat_forward_count": 8}
        for seed in verifier._SEEDS
    ]
    result = verifier._accounting(receipts, references)
    assert result["http_queries"] == {"reused": 222, "fresh": 444}
    assert result["integration_accepted"] is False
    receipts[0]["evidence_origin"] = "fresh"
    with pytest.raises(ValueError, match="evidence_accounting_failure"):
        verifier._accounting(receipts, references)
    receipts[0]["evidence_origin"] = "reused"
    references[0]["complete"] = False
    with pytest.raises(ValueError, match="evidence_accounting_failure"):
        verifier._accounting(receipts, references, True)


def test_seal_bound_before_exclusive_launch_receipt(tmp_path):
    references = []
    for seed in verifier._SEEDS:
        path = tmp_path / f"serial-{seed}.json"
        verifier._write(path, {"seed": seed})
        references.append({"seed": seed, "path": str(path), "sha256": verifier._sha(path)})
    seal = tmp_path / "references-seal.json"
    verifier._write(seal, {"references": references})
    script = tmp_path / "script.py"
    script.write_bytes(b"pass")
    inputs = {str(script): verifier._sha(script)}
    item = {"seed": 1730, "reference": {"checkpoint_hash": "checkpoint"}}
    receipt = verifier._launch_receipt(
        tmp_path / "summary.json", item, seal, verifier._sha(seal), references, inputs, script
    )
    assert verifier._read(receipt)["serial_reference_sha256"] == references[1]["sha256"]
    assert inputs[str(receipt)] == verifier._sha(receipt)
    with pytest.raises(FileExistsError):
        verifier._launch_receipt(
            tmp_path / "summary.json", item, seal, verifier._sha(seal), references, inputs, script
        )
    Path(references[0]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="identity_mismatch"):
        verifier._launch_receipt(
            tmp_path / "summary.json", item, seal, verifier._sha(seal), references, inputs, script
        )


@pytest.fixture
def http_fixture(archived):
    audit, transport = archived
    tokens = [f"RESERVED_{i}" for i in range(1024)] + [f"PKM_{i}" for i in range(1025)]
    tokens[:6] = ["BOS", "EOS", "TYPE", "COLOR", "SAME", "ANSWER"]
    products = {t: {"key": t, "label": t} for t in tokens[1024:]}
    identity = {
        "checkpoint_hash": "checkpoint",
        "training_identity": {"x": "training"},
        "corpus_identity": {"graph_hash": "graph"},
        "split_hash": "split",
    }
    rows, offline, serial, observed = [], [], [], []

    def body(composition, row, **directives):
        targets, counts = audit.process(
            [tokens[i] for i in composition["selected_set_ids"]],
            row["subject"],
            directives.get("ignore", ()),
            directives.get("limit"),
        )
        return {
            **copy.deepcopy(composition),
            "result": {
                "subject": products[row["subject"]],
                "dimension": row["dimension"],
                "mode": "SAME",
                "targets": [products[t] for t in targets],
            },
            "postprocess": counts,
            **{k: identity[k] for k in ("checkpoint_hash", "corpus_identity", "split_hash")},
            "graph_hash": "graph",
            "inference_config_hash": "config",
            "timing": {"wall": 0.1},
        }

    logits = [1.0] * 4 + [-1.0] * 1021
    for i in range(222):
        subject = tokens[1028 + i]
        paths = [
            audit.raw_evidence(
                [0, 1028 + i, 2, 4, 5, 1024 + r, 1],
                subject,
                "TYPE",
                tokens,
                audit.BASE + (f"+first-rank{r + 1}-v1" if r else ""),
            )
            for r in range(4)
        ]
        composed_serial = audit.composition(paths, logits, subject, "TYPE", tokens)
        composed_offline = audit.composition(
            paths, [2.0] * 4 + [-1.0] * 1021, subject, "TYPE", tokens
        )
        row = {
            "subject": subject,
            "dimension": "TYPE",
            "selection": {
                k: v for k, v in composed_offline.items() if k not in ("source_paths", "policy")
            },
            "selected_set_keys": [tokens[j] for j in composed_serial["selected_set_ids"]],
            "source_paths": [{**p, "predicted": p["targets"]} for p in paths],
        }
        rows.append(row)
        offline.append({"composition": composed_offline})
        serial.append(
            {
                "subject": subject,
                "dimension": "TYPE",
                "logits": logits,
                "composition": composed_serial,
            }
        )
        observed.append(
            {
                "request": {"subject": subject, "dimension": "TYPE"},
                "status": 200,
                "response": body(composed_serial, row),
            }
        )
    first, request = rows[0], observed[0]["request"]
    metadata = []
    for directive in ({"limit": 0}, {"limit": 1, "ignore": [tokens[1024]]}):
        metadata.append(
            {
                "kind": "filtered",
                "status": 200,
                "request": {**request, **directive},
                "response": body(serial[0]["composition"], first, **directive),
            }
        )
    metadata.append(
        {"kind": "legacy", "status": 200, "request": request, "reference_index": 0, "response": {}}
    )
    for directive in (
        {"subject": "ATTR_RED"},
        {"ignore": ["ATTR_RED"]},
        {"limit": -1},
        {"dimension": "BIOME"},
        {"max_new_tokens": 1},
    ):
        metadata.append(
            {
                "kind": "invalid",
                "status": 422,
                "request": {**request, **directive},
                "response": {"detail": "invalid"},
            }
        )
    for endpoint in ("/v1/predict-set", "/v1/predict"):
        metadata.append(
            {
                "kind": "admission",
                "status": 503,
                "endpoint": endpoint,
                "response": {"detail": "in-flight request limit reached; retry later"},
            }
        )
    # Incomplete failure slots retain SUBJECT; the first source deliberately emits it.
    failure_paths = [
        audit.raw_evidence(
            [0, 1028, 2, 4, 5, product],
            first["subject"],
            "TYPE",
            tokens,
            audit.BASE + (f"+first-rank{r + 1}-v1" if r else ""),
            1,
        )
        for r, product in enumerate((1028, 1025, 1026, 1027))
    ]
    failure_composition = audit.composition(
        failure_paths, logits, first["subject"], "TYPE", tokens, 1
    )
    report = {
        **identity,
        "seed": 1730,
        "query_count": 222,
        "exact_parity": True,
        "mismatches": [],
        "responses": observed,
        "metadata_checks": metadata,
        "server_stopped": True,
        "disabled_check": {
            "server_stopped": True,
            "status": 409,
            "request": request,
            "direct_composition": None,
            "response": {"detail": {"code": "set_composition_disabled"}},
        },
        "failure_check": {
            "server_stopped": True,
            "status": 502,
            "request": request,
            "direct_composition": failure_composition,
            "response": {
                "detail": {"code": "set_composition_no_valid_source", **failure_composition}
            },
        },
    }
    item = {
        "seed": 1730,
        "reference": {**identity, "responses": rows},
        "offline": {"responses": offline},
        "tokens": tokens,
        "run": {},
        "old_http": {
            "responses": [{"response": {"raw": {}}}] * 222,
            "failure_check": {"direct_candidates": {"candidates": failure_paths}},
        },
    }
    adapter = SimpleNamespace(
        deployment=lambda *args: (products, "config"),
        http_body=audit.http_body,
        composition=audit.composition,
        verify_http_body=lambda *args: None,
    )
    context = {"arithmetic": adapter, "v1": {"evaluator_environment": {}}}
    return report, item, {"responses": serial}, context, products, transport


def test_full_transport_and_serial_reference_bridge(http_fixture):
    report, item, reference, context, products, transport = http_fixture
    result = verifier._validate_http(report, item, reference, context, "fresh")
    assert result["slot_scores_checked"] == 2220 and result["owned_servers_stopped"] == 3
    original = copy.deepcopy(item["reference"])
    revised = verifier._transport_reference(item, reference)
    assert item["reference"] == original
    assert revised["responses"][0]["source_paths"] == original["responses"][0]["source_paths"]
    assert (
        transport._expected_composition(revised["responses"][0])
        == reference["responses"][0]["composition"]
    )
    assert (
        transport._check_http(
            200,
            report["responses"][0]["response"],
            revised["responses"][0],
            products,
            item["reference"],
            "config",
        )
        == []
    )
    assert report["failure_check"]["direct_composition"]["slots"][0]["set_ids"] == [1028]


@pytest.mark.parametrize("mutation", ["metadata", "failure", "status", "hydration", "synthetic"])
def test_transport_schema_auxiliary_and_hydration_fail_closed(http_fixture, mutation):
    report, item, reference, context, _, _ = http_fixture
    if mutation == "metadata":
        report["metadata_checks"].pop()
    elif mutation == "failure":
        report["failure_check"]["response"] = {"detail": {}}
    elif mutation == "status":
        report["responses"][0]["status"] = 500
    elif mutation == "hydration":
        report["responses"][0]["response"]["result"]["targets"] = []
    else:
        report["responses"][0]["response"].update(
            token_ids=[1], terminated=True, protocol_valid=True, sum_logprob=0.0
        )
    with pytest.raises(ValueError, match="transport_contract_mismatch"):
        verifier._validate_http(report, item, reference, context, "fresh")


def test_module_import_and_help_are_torch_free_in_fresh_interpreter():
    code = (
        "import runpy, sys; runpy.run_path('scripts/verify_pair_composition_shapes.py', "
        "run_name='cpu_import'); assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_reference_seal_rejects_checkpoint_config_runtime_prompt_and_source_drift(http_fixture):
    _, item, reference, _, _, _ = http_fixture
    environment = {"python": "synthetic"}
    config = {"batch_size": 1}
    context = {
        "arithmetic": SimpleNamespace(
            config=lambda *args: config, canonical=lambda value: "config"
        ),
        "v1": {"evaluator_environment": environment},
        "diagnosis": {"torch_settings": {}},
    }
    report = {
        **{k: item["reference"][k] for k in verifier._IDENTITY},
        **reference,
        "seed": 1730,
        "complete": True,
        "query_count": 222,
        "reference_forward_count": 222,
        "repeat_forward_count": 8,
        "repeat_logits": [r["logits"] for r in reference["responses"][:8]],
        "config": config,
        "config_hash": "config",
        "batch_shape": [1, 5],
        "environment": environment,
        "source_archive_sha256": verifier._SOURCE_SHA,
        "runtime_environment_sha256": verifier._value_sha(environment),
        "evidence_origin": "fresh",
        "product_token_ids": verifier._COLUMNS,
        "numerical_settings": {"autocast_cuda_enabled": False, "autocast_cpu_enabled": False},
    }
    for index, row in enumerate(report["responses"]):
        row.update(
            prompt_ids=[0, 1028 + index, 2, 4, 5],
            batch_shape=[1, 5],
            seed=1730,
            checkpoint_hash="checkpoint",
            config_hash="config",
            source_archive_sha256=verifier._SOURCE_SHA,
            runtime_environment_sha256=verifier._value_sha(environment),
            cross_shape_score_differences=[
                a["score"] - b["score"]
                for a, b in zip(
                    row["composition"]["slots"],
                    item["offline"]["responses"][index]["composition"]["slots"],
                    strict=True,
                )
            ],
        )
    verifier._seal_reference(report, item, context)
    for target, key, changed in (
        (report, "config", {}),
        (report, "checkpoint_hash", "other"),
        (report, "environment", {}),
        (report["responses"][0], "prompt_ids", [0, 1, 2, 3, 4]),
        (report["responses"][0], "batch_shape", [8, 5]),
        (report["responses"][0], "source_archive_sha256", "other"),
    ):
        original = target[key]
        target[key] = changed
        with pytest.raises(ValueError, match="identity_mismatch"):
            verifier._seal_reference(report, item, context)
        target[key] = original
    report["complete"] = False
    with pytest.raises(ValueError, match="invalid_serial_reference"):
        verifier._seal_reference(report, item, context)


def test_cpu_test_receipt_binds_final_script_and_tests(tmp_path):
    tests = tmp_path / "tests.py"
    tests.write_bytes(b"synthetic test")
    receipt = tmp_path / "receipt.json"
    verifier._write(
        receipt,
        {
            "passed": True,
            "model_execution": False,
            "verifier_sha256": "script",
            "tests_sha256": verifier._sha(tests),
        },
    )
    inputs = {}
    verifier._test_receipt(receipt, verifier._sha(receipt), "script", tests, inputs)
    assert str(tests) in inputs
    tests.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity_mismatch"):
        verifier._test_receipt(receipt, verifier._sha(receipt), "script", tests, inputs)
