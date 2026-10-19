"""Synthetic independent width-eight checks; no campaign result measurement."""

import copy
import importlib.util
import math
import zipfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3] / "runs/learning/wide-first-choice-v1/independent-audit.py"
)
spec = importlib.util.spec_from_file_location("wide_audit", SCRIPT)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
PROMPT = [1, 1024, 32, 34, 5]


def sources(width=8):
    return [
        {
            "token_ids": [*PROMPT, 1025 + i, 2],
            "targets": [f"PKM_{1025 + i}"],
            "terminated": True,
            "protocol_valid": True,
            "error": None,
            "decoding": a.BASE + (f"+first-rank{i + 1}-v1" if i else ""),
        }
        for i in range(width)
    ]


def test_slot_order_legacy_then_originals_then_lexicographic_pairs():
    assert len(a.ORDER8) == len(set(a.ORDER8)) == 36
    assert a.ORDER8[:10] == a.ORDER4
    assert a.ORDER8[10:14] == ((5,), (6,), (7,), (8,))
    assert a.ORDER8[14:19] == ((1, 5), (1, 6), (1, 7), (1, 8), (2, 5))
    assert a.ORDER8[-1] == (7, 8)


def test_rational_sum_survives_cancellation_and_rejects_uncanonical():
    values = [0.0] * 1025
    values[1:4] = [2.0**80, 1.0, -(2.0**80)]
    assert a.canonical_sum([1025, 1026, 1027], values) == math.fsum(values[1:4]) == 1.0
    for ids in ([1025, 1025], [1026, 1025], [1023], [True]):
        with pytest.raises(ValueError):
            a.canonical_sum(ids, values)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.1, True, 1])
def test_vector_rejects_nonfinite_nonfp32_and_nonfloats(bad):
    values = [0.0] * 1025
    values[4] = bad
    with pytest.raises(ValueError):
        a.vector(values)


def test_same_subject_removed_but_raw_stays_intact_and_equal_tie_keeps_legacy():
    raw = sources()
    raw[0]["token_ids"] = [*PROMPT, 1025, 1024, 2]
    raw[0]["targets"] = ["PKM_1025", "PKM_1024"]
    before = copy.deepcopy(raw)
    comp, removed, _ = a.composition(raw, PROMPT, [0.0] * 1025)
    assert raw == before
    assert comp["slots"][0]["set_ids"] == [1025]
    assert removed == [True] + [False] * 7
    assert comp["selected_slot"] == 1
    assert comp["slots"][10]["kind"] == "original"


def test_new_equal_set_cannot_displace_legacy_tie():
    raw = sources()
    raw[0]["token_ids"] = [*PROMPT, 1025, 1029, 2]
    raw[0]["targets"] = ["PKM_1025", "PKM_1029"]
    raw[4]["token_ids"] = [*PROMPT, 1029, 1025, 2]
    raw[4]["targets"] = ["PKM_1029", "PKM_1025"]
    values = [-10.0] * 1025
    values[1] = values[5] = 1.0
    comp, _, _ = a.composition(raw, PROMPT, values)
    assert comp["slots"][0]["set_ids"] == comp["slots"][10]["set_ids"]
    assert comp["selected_slot"] == 1


def test_first_product_distinctness_and_forbidden_eos():
    raw = sources()
    raw[1]["token_ids"][5] = raw[0]["token_ids"][5]
    with pytest.raises(ValueError, match="distinctness"):
        a.composition(raw, PROMPT, [0.0] * 1025)
    raw = sources()
    raw[0].update(
        token_ids=[*PROMPT, 2],
        targets=[],
        protocol_valid=False,
        error="generated response is shorter than the protocol grammar",
    )
    with pytest.raises(ValueError, match="distinctness"):
        a.composition(raw, PROMPT, [0.0] * 1025)


def test_raw_repeat_is_ineligible_before_subject_removal():
    raw = sources()
    raw[0]["token_ids"] = [*PROMPT, 1025, 1024, 1024, 2]
    raw[0]["targets"] = ["PKM_1025", "PKM_1024", "PKM_1024"]
    comp, removed, _ = a.composition(raw, PROMPT, [1.0] * 1025)
    assert not comp["slots"][0]["source_eligible"]
    assert not comp["slots"][4]["source_eligible"]
    assert removed[0]


def test_bound_failure_keeps_subject_membership_and_no_answer_credit():
    raw = sources()
    for i, item in enumerate(raw):
        products = [1025 + i, 1024, *list(range(1033, 1538))]
        assert len(products) == 507
        item.update(
            token_ids=PROMPT + products,
            targets=[f"PKM_{p}" for p in products],
            terminated=False,
            protocol_valid=False,
            error="response must terminate with EOS",
        )
    comp, removed, _ = a.composition(raw, PROMPT, [1.0] * 1025)
    assert comp["fallback_no_valid_source"] and comp["selected_slot"] == 1
    assert not any(removed)
    assert 1024 in comp["selected_set_ids"]
    assert a.set_metrics(comp["selected_set_ids"], [1025], False)["f1"] == 0
    assert a.available(comp, [1025])["distinct_eligible_sets"] == 0


def test_truncated_early_or_overbudget_flags_rejected():
    raw = sources()[0]
    raw.update(
        token_ids=[*PROMPT, 1025],
        terminated=False,
        protocol_valid=False,
        error="response must terminate with EOS",
    )
    with pytest.raises(ValueError, match="stopped before bound"):
        a.source_set(raw, PROMPT, 1)
    raw["token_ids"] = PROMPT + [1025] * 508 + [2]
    with pytest.raises(ValueError, match="budget"):
        a.source_set(raw, PROMPT, 1)


def test_selection_has_no_teacher_input_and_availability_can_gain_while_quality_loses():
    raw = sources()
    values = [-1.0] * 1025
    values[1] = 1.0
    values[5] = 2.0
    narrow, _, _ = a.composition(raw[:4], PROMPT, values)
    wide, _, _ = a.composition(raw, PROMPT, values)
    assert narrow["selected_set_ids"] == [1025]
    assert wide["selected_set_ids"] == [1025, 1029]
    assert a.available(wide, [1025])["exact_available"]
    assert a.available(wide, [1025])["available_exact_miss"]
    assert a.set_metrics(wide["selected_set_ids"], [1025])["exact"] is False
    assert a.composition(raw, PROMPT, values)[0] == wide


def passing_gate_inputs():
    seed = [
        (s, {"baseline": {"exact_count": 10, "f1": 0.8}, "wide": {"exact_count": 11, "f1": 0.81}})
        for s in a.SEEDS
    ]
    pool = {"baseline": {"exact_count": 30}, "wide": {"exact_count": 33}}
    groups = {g: {"baseline": {"exact_count": 10}, "wide": {"exact_count": 11}} for g in a.GROUPS}
    return seed, pool, groups


def test_declared_gate_requires_selected_gain_not_just_availability():
    seed, pool, groups = passing_gate_inputs()
    assert a.gate(seed, pool, groups)["eligible_for_further_research"]
    pool["wide"]["exact_count"] = 30
    assert not a.gate(seed, pool, groups)["eligible_for_further_research"]


@pytest.mark.parametrize("mutation", ["seed_exact", "seed_f1", "dual", "color", "valid", "replay"])
def test_gate_fails_each_material_regression(mutation):
    seed, pool, groups = passing_gate_inputs()
    if mutation == "seed_exact":
        seed[0][1]["wide"]["exact_count"] = 9
    if mutation == "seed_f1":
        seed[0][1]["wide"]["f1"] = 0.79
    if mutation == "dual":
        groups["TYPE_dual"]["wide"]["exact_count"] = 10
    if mutation == "color":
        groups["COLOR"]["wide"]["exact_count"] = 9
    result = a.gate(seed, pool, groups, valid=mutation != "valid", replay=mutation != "replay")
    assert not result["eligible_for_further_research"]


@pytest.mark.parametrize("name", ["../escape.py", "/abs.py", "C:/bad.py", "src\\bad.py"])
def test_archive_unsafe_paths_rejected(tmp_path, name):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("safe.py")
        info.filename = name
        archive.writestr(info, b"data")
    with pytest.raises(ValueError, match="unsafe"):
        a.safe_archive(path, a.sha(path))


def test_archive_bytes_and_runtime_are_exact(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("src/plm/a.py", b"original")
    runtime = tmp_path / "runtime"
    target = runtime / "src/plm/a.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        a.safe_archive(path, a.sha(path), runtime)


def full_row():
    vocabulary = [f"reserved_{i}" for i in range(1024)] + [f"PKM_{i}" for i in range(1024, 2049)]
    raw = sources()
    values = [-1.0] * 1025
    values[1] = 1.0
    values[5] = 2.0
    baseline, _, _ = a.composition(raw[:4], PROMPT, values)
    wide, removed, _ = a.composition(raw, PROMPT, values)
    base = {
        "index": 0,
        "subject": "PKM_1024",
        "dimension": "TYPE",
        "group": "TYPE_dual",
        "prompt_ids": PROMPT,
        "expected": ["PKM_1025"],
        "expected_set_ids": [1025],
        "symmetric_relation_logits": values,
        "composition": baseline,
    }
    metrics = {
        "baseline": a.answer_metrics(baseline, [1025]),
        "wide": a.answer_metrics(wide, [1025]),
    }
    row = {
        **{k: v for k, v in base.items() if k != "composition"},
        "baseline": baseline,
        "composition": wide,
        "metrics": metrics,
        "availability": {
            "baseline": a.available(baseline, [1025]),
            "wide": a.available(wide, [1025]),
        },
        "subject_removed": removed,
        "gained_exact": False,
        "lost_exact": True,
    }
    return base, row, vocabulary


def test_full_row_reconstruction_and_macro_arithmetic():
    base, row, vocab = full_row()
    assert a.rebuild_row(base, row, vocab, 0) == row
    total = a.totals([row, copy.deepcopy(row)])
    assert total["losses"] == 2 and total["gains"] == 0
    assert total["baseline"]["exact_count"] == 2
    assert total["wide"]["exact_count"] == 0
    assert total["wide"]["precision"] == 0.5
    assert total["wide"]["recall"] == 1.0
    assert total["wide"]["f1"] == 2 / 3
    assert total["wide"]["added_slot_selections"] == 2
    assert total["wide"]["available_exact_miss"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "score",
        "legacy_path",
        "target_key",
        "choice",
        "metric",
        "removed",
        "availability",
        "truth_subject",
    ],
)
def test_full_row_rejects_saved_corruption(mutation):
    base, row, vocab = copy.deepcopy(full_row())
    if mutation == "score":
        row["composition"]["slots"][35]["score"] += 0.1
    if mutation == "legacy_path":
        row["composition"]["source_paths"][0]["token_ids"][5] = 1034
    if mutation == "target_key":
        row["composition"]["source_paths"][7]["targets"][0] = "PKM_wrong"
    if mutation == "choice":
        row["composition"]["selected_slot"] = 1
    if mutation == "metric":
        row["metrics"]["wide"]["f1"] = 1.0
    if mutation == "removed":
        row["subject_removed"][0] = True
    if mutation == "availability":
        row["availability"]["wide"]["distinct_eligible_sets"] += 1
    if mutation == "truth_subject":
        base["expected_set_ids"] = [1024]
    with pytest.raises(ValueError):
        a.rebuild_row(base, row, vocab, 0)


def work_fixture():
    _, row, _ = full_row()
    rows = [copy.deepcopy(row) for _ in range(222)]
    records = []
    for ranks in (range(1, 5), range(5, 9)):
        for offset in range(0, 222, 8):
            for rank in ranks:
                count = min(8, 222 - offset)
                records.append(
                    {
                        "offset": offset,
                        "rank": rank,
                        "batch_size": count,
                        "decode_calls": 2,
                        "padded_decode_positions": count * 6,
                        "useful_emitted_tokens": count * 2,
                        "guidance_forward_calls": 1,
                        "wall_seconds": 0.01,
                    }
                )
    report = {
        "responses": rows,
        "batch_work": records,
        "wall_seconds": 5.0,
        "head_seconds": 0.1,
        "set_scoring_seconds": 0.1,
        "peak_gpu_allocated_bytes": 1,
        "generation_seconds": {"ranks1_4": 1.12, "ranks5_8": 1.12},
    }
    baseline = {"batch_work": copy.deepcopy(records[:112]), "head_seconds": 0.1}
    return report, baseline


def test_work_counts_prompt_and_eos_and_last_six_rows():
    report, baseline = work_fixture()
    result = a.work_audit(report, baseline)
    assert result["decode_calls"] == 224 * 2
    assert result["useful_emitted_tokens"] == 222 * 8 * 2
    assert result["padded_decode_positions"] == 222 * 8 * 6
    assert result["guidance_forward_calls"] == 224
    assert result["prompt_head_forward_calls"] == 28


@pytest.mark.parametrize("mutation", ["rank", "last_batch", "eos_omitted", "guidance", "order"])
def test_work_corruption_rejected(mutation):
    report, baseline = work_fixture()
    if mutation == "rank":
        report["batch_work"][-1]["rank"] = 9
    if mutation == "last_batch":
        report["batch_work"][-1]["batch_size"] = 8
    if mutation == "eos_omitted":
        report["batch_work"][-1]["useful_emitted_tokens"] -= 6
    if mutation == "guidance":
        report["batch_work"][-1]["guidance_forward_calls"] = 2
    if mutation == "order":
        report["batch_work"][-1], report["batch_work"][-2] = (
            report["batch_work"][-2],
            report["batch_work"][-1],
        )
    with pytest.raises(ValueError):
        a.work_audit(report, baseline)


@pytest.mark.parametrize("mutation", [None, "outside", "hash", "module", "missing"])
def test_runtime_origins_bound_to_module_name_and_extracted_inventory(tmp_path, mutation):
    runtime = tmp_path / "runtime"
    names = {
        "plm": "plm/__init__.py",
        "plm.serving.runtime": "plm/serving/runtime.py",
        "plm.serving.set_reranking": "plm/serving/set_reranking.py",
    }
    inventory = {str((runtime / "src" / path).resolve()): "f" * 64 for path in names.values()}
    origins = {
        name: {"path": str((runtime / "src" / path).resolve()), "sha256": "f" * 64}
        for name, path in names.items()
    }
    if mutation == "outside":
        origins["plm"]["path"] = str(tmp_path / "wrong.py")
    if mutation == "hash":
        origins["plm"]["sha256"] = "e" * 64
    if mutation == "module":
        origins["plm"]["path"] = origins["plm.serving.runtime"]["path"]
    if mutation == "missing":
        del origins["plm.serving.runtime"]
    if mutation is None:
        a.runtime_origins(origins, inventory, runtime)
    else:
        with pytest.raises(ValueError):
            a.runtime_origins(origins, inventory, runtime)


def test_incomplete_summary_rejected_before_any_reference_read(tmp_path, monkeypatch):
    import json

    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"complete": False}))
    monkeypatch.setattr(a.sys, "argv", [str(SCRIPT), "--summary", str(path)])
    monkeypatch.setattr(
        a, "historical_references", lambda: pytest.fail("must not read historical results")
    )
    with pytest.raises(ValueError, match="incomplete"):
        a.main()
    assert not (tmp_path / "independent-audit.json").exists()


def test_prior_independent_audit_refuses_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    output = tmp_path / "independent-audit.json"
    output.write_text("original receipt")
    monkeypatch.setattr(a.sys, "argv", [str(SCRIPT), "--summary", str(path)])
    with pytest.raises(ValueError, match="overwrite"):
        a.main()
    assert output.read_text() == "original receipt"
