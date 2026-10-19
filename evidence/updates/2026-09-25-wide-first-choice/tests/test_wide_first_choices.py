"""Focused synthetic width-eight selection/isolation tests; no real model forwards."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "wide_first_choices", ROOT / "scripts/evaluate_wide_first_choices.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@dataclass
class PathResult:
    token_ids: tuple
    targets: tuple
    terminated: bool = True
    protocol_valid: bool = True
    decoding: str = "synthetic"
    error: str | None = None


class Vocabulary:
    eos_id = 2

    def encode(self, targets):
        return [int(t.removeprefix("PKM_")) for t in targets]


def source(ids, rank=1, *, valid=True):
    return PathResult(
        (1, 1024, 32, 34, 5, *ids, *((2,) if valid else ())),
        tuple(f"PKM_{i}" for i in ids),
        valid,
        valid,
        runner._BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
        None if valid else "truncated",
    )


def candidate_sets(paths, logits, vocabulary, query):
    """Small synthetic stand-in; actual experiment calls authenticated archived helper."""
    sets, eligible = [], []
    for p in paths:
        raw = vocabulary.encode(p.targets)
        members = set(raw)
        if p.protocol_valid and p.terminated:
            members.discard(1024)
        sets.append(tuple(sorted(members)))
        eligible.append(p.protocol_valid and p.terminated and len(raw) == len(set(raw)))
    return sets, [score(s, logits) for s in sets], eligible


def score(ids, logits):
    return math.fsum(logits[i - 1024] for i in ids)


def compose(sets, *, valid=True, logits=None):
    paths = tuple(source(ids, i, valid=valid) for i, ids in enumerate(sets, 1))
    return runner._compose(
        paths, logits or [1.0] * 1025, Vocabulary(), ("PKM_1024", "TYPE"), candidate_sets, score
    )


def row(group="TYPE_dual"):
    base = compose([[1025], [1026], [1027], [1028]])
    wide = compose([[1025], [1026], [1027], [1028], [1029], [1030], [1031], [1032]])
    return runner._row(
        {
            "index": 0,
            "subject": "PKM_1024",
            "dimension": "TYPE",
            "group": group,
            "expected_set_ids": [1025, 1026],
            "composition": base,
        },
        wide,
    )


def test_legacy_priority_and_all_36_provenances():
    value = compose([[1025 + i] for i in range(8)])
    assert len(value["slots"]) == 36
    assert tuple(tuple(s["source_ranks"]) for s in value["slots"][:10]) == runner._ORDER10
    assert [s["source_ranks"] for s in value["slots"][10:14]] == [[5], [6], [7], [8]]
    assert [s["source_ranks"] for s in value["slots"][-2:]] == [[6, 8], [7, 8]]
    assert len({tuple(s["source_ranks"]) for s in value["slots"]}) == 36
    assert value["selected_slot"] == 5  # First pair keeps tie priority.


def test_duplicate_sets_retained_and_new_equal_slot_cannot_displace_legacy():
    value = compose([[1025]] * 8)
    assert len(value["slots"]) == 36
    assert value["selected_slot"] == 1
    assert len({tuple(s["set_ids"]) for s in value["slots"]}) == 1


def test_negative_union_does_not_force_pair_selection():
    values = [-1.0] * 1025
    value = compose([[1025 + i] for i in range(8)], logits=values)
    assert value["selected_slot"] == 1
    assert value["slots"][4]["score"] == -2


def test_raw_subject_removed_only_for_completed_sources():
    complete = compose([[1024, 1025]] * 4)
    incomplete = compose([[1024, 1025]] * 4, valid=False)
    assert complete["slots"][0]["set_ids"] == [1025]
    assert incomplete["slots"][0]["set_ids"] == [1024, 1025]
    assert complete["source_paths"][0]["targets"] == ["PKM_1024", "PKM_1025"]


def test_fallback_preserves_rank_one_and_receives_no_credit():
    value = compose([[1025]] * 8, valid=False)
    assert value["fallback_no_valid_source"] and value["selected_set_ids"] == [1025]
    metrics = runner._metrics(value, [1025])
    assert metrics == {
        "exact": False,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "set_size": 0,
        "success": False,
    }
    assert not runner._availability(value, [1025])["exact_available"]


def test_ineligible_source_excludes_all_its_pairs():
    paths = (source([1025], 1, valid=False), *(source([1025 + i], i + 1) for i in range(1, 8)))
    value = runner._compose(
        paths, [1.0] * 1025, Vocabulary(), ("PKM_1024", "TYPE"), candidate_sets, score
    )
    assert all(not s["source_eligible"] for s in value["slots"] if 1 in s["source_ranks"])
    assert 1 not in value["selected_source_ranks"]


def test_truth_only_changes_diagnostics_not_selection():
    r = row()
    different_truth = runner._row(
        {**r, "expected_set_ids": [1031, 1032], "composition": r["baseline"]}, r["composition"]
    )
    assert different_truth["composition"] == r["composition"]
    assert r["metrics"]["wide"]["exact"] and not different_truth["metrics"]["wide"]["exact"]
    assert different_truth["availability"]["wide"]["available_exact_miss"]


@pytest.mark.parametrize("field", ["head", "score", "raw", "choice"])
def test_exact_legacy_corruption_rejected(field):
    actual = compose([[1025]] * 4)
    expected = copy.deepcopy(actual)
    logits = [1.0] * 1025
    expected_logits = list(logits)
    if field == "head":
        expected_logits[0] += 1e-7
    elif field == "score":
        expected["slots"][0]["score"] += 1e-14
    elif field == "raw":
        expected["source_paths"][0]["token_ids"][5] += 1
    else:
        expected["selected_slot"] = 2
    with pytest.raises(ValueError, match="replay failed"):
        runner._legacy(
            actual,
            logits,
            {"composition": expected},
            {"symmetric_relation_logits": expected_logits},
        )


def test_source_mask_first_choice_and_prefix_checks():
    vocab = Vocabulary()
    prompt = [1, 1024, 32, 34, 5]
    good = tuple(source([1025 + i], i + 1) for i in range(8))
    runner._sources(good, prompt, vocab)
    with pytest.raises(ValueError, match="distinct first"):
        runner._sources(tuple(source([1025], i + 1) for i in range(8)), prompt, vocab)
    with pytest.raises(ValueError, match="product mask"):
        runner._sources((source([32]),), prompt, vocab)
    with pytest.raises(ValueError, match="uniqueness"):
        runner._sources((source([1025, 1025]),), prompt, vocab)


def test_fp32_and_canonical_fsum():
    logits = [0.0] * 1025
    logits[:3] = [2.0**60, 1.0, -(2.0**60)]
    assert score([1024, 1025, 1026], logits) == 1.0
    runner._fp32(logits)
    logits[3] = 0.1
    with pytest.raises(ValueError, match="FP32"):
        runner._fp32(logits)


def passing_gate_data():
    r = row()
    pooled = runner._aggregate([row(g) for g in runner._GROUPS])
    pooled["overall"]["wide"]["exact_count"] = 570
    pooled["groups"]["TYPE_dual"]["wide"]["exact_count"] += 1
    reports = [
        {
            "complete": True,
            "baseline_replay_exact": True,
            "query_count": 222,
            "responses": [r],
            "overall": copy.deepcopy(pooled["overall"]),
        }
        for _ in range(3)
    ]
    return reports, pooled


@pytest.mark.parametrize(
    "regression", ["seed_exact", "seed_f1", "pooled", "group", "dual", "source"]
)
def test_fixed_quality_gate_rejects_each_regression(regression):
    reports, pooled = passing_gate_data()
    assert runner._gate(reports, pooled)["quality_passed"]
    if regression == "seed_exact":
        reports[0]["overall"]["wide"]["exact_count"] = 0
    elif regression == "seed_f1":
        reports[0]["overall"]["wide"]["f1"] = 0
    elif regression == "pooled":
        pooled["overall"]["wide"]["exact_count"] = 569
    elif regression == "group":
        pooled["groups"]["COLOR"]["wide"]["exact_count"] = 0
    elif regression == "dual":
        pooled["groups"]["TYPE_dual"]["wide"]["exact_count"] -= 1
    else:
        reports[0]["responses"][0]["composition"]["source_paths"][4]["terminated"] = False
    assert not runner._gate(reports, pooled)["quality_passed"]


def test_work_counts_include_prefill_and_padded_finished_rows():
    paths = (source([1025]), source([1026, 1027, 1028], 2))
    assert runner._work(paths, 216, 5, 1.0) == {
        "offset": 216,
        "batch_size": 2,
        "rank": 5,
        "wall_seconds": 1.0,
        "decode_calls": 4,
        "padded_decode_positions": 16,
        "useful_emitted_tokens": 6,
        "guidance_forward_calls": 1,
    }


def archive(path, extra=None, configs=True):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("src/plm/__init__.py", "")
        if configs:
            z.writestr("configs/config.yaml", "{}")
            z.writestr("configs/protocol/pokemon_v1.yaml", "{}")
        if extra:
            z.writestr(extra, "unsafe")


@pytest.mark.parametrize("name", ["../escape.py", "C:/absolute.py", "src/PLM/__init__.py"])
def test_archive_unsafe_or_case_colliding_paths(tmp_path, name):
    path = tmp_path / "x.zip"
    archive(path, name)
    with pytest.raises(ValueError, match="unsafe or duplicate"):
        runner._archive_members([path])


def test_missing_config_and_output_reuse_fail(tmp_path):
    path = tmp_path / "x.zip"
    archive(path, configs=False)
    with pytest.raises(ValueError, match="missing archived"):
        runner._archive_members([path])
    out = tmp_path / "summary.json"
    (tmp_path / "independent-audit.py").write_text("# permitted")
    runner._refuse(out)
    (tmp_path / "baseline-1729.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        runner._refuse(out)


def test_runtime_extraction_refuses_reuse_and_detects_edit_or_added_source(tmp_path):
    path = tmp_path / "x.zip"
    archive(path)
    members = runner._archive_members([path])
    destination = tmp_path / "runtime"
    hashes = runner._extract(members, destination)
    runner._runtime_unchanged(destination, hashes)
    with pytest.raises(FileExistsError):
        runner._extract(members, destination)
    (destination / "src/plm/injected.py").write_text("# new")
    with pytest.raises(ValueError, match="inventory changed"):
        runner._runtime_unchanged(destination, hashes)


def test_module_contamination_rejected_without_removing_shared_modules(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "plm.injected_wide_test",
        types.SimpleNamespace(__file__=str(tmp_path / "bad.py")),
    )
    with pytest.raises(ValueError, match="before runtime isolation"):
        runner._modules()
    with pytest.raises(ValueError, match="contamination"):
        runner._modules(tmp_path / "runtime")


def test_partial_failure_summary_preserves_completed_reports(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    summary = {
        "complete": False,
        "acceptance": False,
        "input_sha256": {},
        "snapshot_sha256": {},
        "reports": [{"seed": 1729, "sha256": "prior"}],
    }

    def fail():
        raise RuntimeError("injected isolation failure")

    monkeypatch.setattr(runner, "_modules", fail)
    with pytest.raises(RuntimeError, match="injected"):
        runner._execute(tmp_path, output, {}, summary)
    saved = runner._read(output)
    assert not saved["complete"] and not saved["acceptance"]
    assert saved["reports"] == summary["reports"]
    assert "injected isolation failure" in saved["error"]


def test_bound_input_drift_rejected(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"original")
    inputs = {}
    runner._bind(path, runner._sha(path), inputs)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        runner._unchanged(inputs)


def test_nonfinite_failed_head_preserves_raw_evidence_and_original_error(tmp_path):
    report = {
        "complete": False,
        "responses": [{"completed": 1}],
        "failed_observation": {
            "rank_outputs": [[1025, 2]],
            "symmetric_relation_logits": [math.nan, math.inf] + [0.0] * 1023,
        },
    }
    output = tmp_path / "seed-1729.json"
    with pytest.raises(ValueError, match="nonfinite head"):
        try:
            runner._fp32(report["failed_observation"]["symmetric_relation_logits"])
        finally:
            runner._write(output, runner._failure_safe(report))
    saved = runner._read(output)
    assert saved["responses"] == [{"completed": 1}]
    assert saved["failed_observation"]["rank_outputs"] == [[1025, 2]]
    assert saved["failed_observation"]["symmetric_relation_logits"][:2] == [
        {"nonfinite_float": "nan"},
        {"nonfinite_float": "inf"},
    ]
