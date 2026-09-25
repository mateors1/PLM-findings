"""Synthetic stdlib diagnostic cases; no measurement of saved validation rankings."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "membership_separability", ROOT / "scripts/diagnose_membership_separability.py"
)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def row(logits, truth, baseline=None):
    baseline = truth if baseline is None else baseline
    return {
        "subject": "PKM_SUBJECT",
        "subject_id": 1024,
        "dimension": "TYPE",
        "group": "TYPE_single",
        "expected_set_ids": truth,
        "baseline_set_ids": baseline,
        "source_union_missing_true_ids": [],
        "logits": logits,
        "baseline_selection": {
            "selected_slot": 1,
            "selected_kind": "original",
            "selected_source_ranks": [1],
        },
    }


def test_strict_separation_with_misplaced_zero_and_subject_exclusion():
    result = diagnostic._measure(
        1729, 0, row([99.0, 4.0, 1.0, 3.0], [1025, 1027]), [1024, 1025, 1026, 1027]
    )
    assert result["strict_separable"] and result["separable_with_wrong_zero_threshold"]
    assert result["direct_set_ids"] == [1025, 1026, 1027]
    assert result["oracle_top_k"]["selected_ranked_ids"] == [1025, 1027]
    assert result["oracle_top_k_exact"]
    assert result["separation"]["gap"] == 2
    assert result["separation"]["threshold_interval"] == {
        "exists": True,
        "lower": 1.0,
        "upper": 3.0,
        "lower_inclusive": True,
        "upper_inclusive": False,
        "semantics": "lower <= t < upper",
    }


def test_exact_zero_threshold_is_strict_not_nonnegative():
    result = diagnostic._measure(1729, 0, row([99.0, 0.0, -1.0], [1025]), [1024, 1025, 1026])
    assert result["direct_set_ids"] == []
    assert result["separable_with_wrong_zero_threshold"]
    assert result["separation"]["threshold_interval"]["upper"] == 0.0


@pytest.mark.parametrize("truth,exact", [([1025], True), ([1026], False)])
def test_zero_gap_boundary_tie_depends_on_token_id(truth, exact):
    result = diagnostic._measure(1729, 0, row([99.0, 1.0, 1.0], truth), [1024, 1025, 1026])
    assert not result["strict_separable"] and result["boundary_tie"]
    assert not result["strict_rank_overlap"]
    assert result["oracle_top_k_exact"] is exact
    assert result["oracle_top_k"]["selected_set_ids"] == [1025]
    boundary = result["oracle_top_k"]["boundary"]
    assert boundary["tie"] and boundary["tied_product_ids"] == [1025, 1026]
    assert boundary["selected_tied_ids"] == [1025] and boundary["excluded_tied_ids"] == [1026]
    assert boundary["tied_true_ids"] == truth
    assert result["separation"]["threshold_interval"]["exists"] is False


def test_strict_rank_inversion_is_not_calibration_only():
    result = diagnostic._measure(1729, 0, row([99.0, 2.0, 1.0], [1026]), [1024, 1025, 1026])
    assert result["strict_rank_overlap"] and not result["boundary_tie"]
    assert result["separation"]["gap"] == -1
    assert result["oracle_top_k_lost_exact"]
    assert result["oracle_top_k"]["metrics"]["fp"] == 1
    assert result["oracle_top_k"]["metrics"]["fn"] == 1


def test_extremal_witnesses_include_every_tied_product():
    separation, _ = diagnostic._geometry(
        [9.0, 2.0, 2.0, -1.0, -1.0], [1024, 1025, 1026, 1027, 1028], 1024, [1025, 1026]
    )
    assert separation["min_true_ids"] == [1025, 1026]
    assert separation["max_negative_ids"] == [1027, 1028]


@pytest.mark.parametrize(
    "ids,logits,truth,case,lower,upper",
    [
        ([1024, 1025, 1026], [99.0, 4.0, 2.0], [], "empty_truth", 4.0, None),
        ([1024, 1025, 1026], [99.0, 4.0, 2.0], [1025, 1026], "full_truth", None, 2.0),
        ([1024], [99.0], [], "empty_universe", None, None),
    ],
)
def test_empty_full_and_vacuous_cases_use_finite_json(ids, logits, truth, case, lower, upper):
    separation, top_k = diagnostic._geometry(logits, ids, 1024, truth)
    assert separation["case"] == case and separation["strict_separable"]
    assert separation["gap"] is None
    assert separation["threshold_interval"]["lower"] == lower
    assert separation["threshold_interval"]["upper"] == upper
    assert not top_k["boundary"]["has_cut"] and not top_k["boundary"]["tie"]
    assert top_k["metrics"]["exact"] and top_k["metrics"]["f1"] == 1.0
    json.dumps([separation, top_k], allow_nan=False)


@pytest.mark.parametrize(
    "ids,subject,truth",
    [
        ([1024, 1025, 1025], 1024, [1025]),
        ([1025, 1024, 1026], 1024, [1025]),
        ([1024, 1025, 1026], 1024, [1024]),
        ([1024, 1025, 1026], 1024, [1025, 1025]),
        ([1024, 1025, 1026], 1030, [1025]),
        ([1024, 1025, True], 1024, [1025]),
    ],
)
def test_malformed_product_truth_and_subject_ids_fail(ids, subject, truth):
    with pytest.raises(ValueError, match=r"IDs|ID"):
        diagnostic._geometry([1.0, 2.0, 3.0], ids, subject, truth)


@pytest.mark.parametrize("value", [math.nan, math.inf, True, 0.1, 1e100])
def test_nonfinite_or_non_fp32_logits_fail(value):
    with pytest.raises(ValueError, match="logit"):
        diagnostic._geometry([1.0, value], [1024, 1025], 1024, [1025])


def test_logit_column_alignment_fails():
    with pytest.raises(ValueError, match="alignment"):
        diagnostic._geometry([1.0], [1024, 1025], 1024, [1025])


def test_aggregate_crosstabs_cover_all_rows_and_pair_failures():
    rows = [
        diagnostic._measure(1729, 0, row([99.0, 4.0, 1.0], [1025], []), [1024, 1025, 1026]),
        diagnostic._measure(1729, 1, row([99.0, 1.0, 2.0], [1025]), [1024, 1025, 1026]),
    ]
    rows[0]["source_union_lacks_truth"] = True
    rows[0]["source_union_missing_true_ids"] = [1025]
    summary = diagnostic._summarize(rows)
    assert summary["totals"]["query_count"] == 2
    assert summary["totals"]["counts"]["oracle_top_k_gained_exact"] == 1
    assert summary["totals"]["counts"]["oracle_top_k_lost_exact"] == 1
    assert summary["pair_failures"]["totals"]["query_count"] == 1
    cross = summary["totals"]["cross_tab"]
    assert sum(r["query_count"] for r in cross) == 2
    assert cross[0]["pair_exact"] is False and cross[0]["source_union_lacks_truth"] is True
    assert summary["groups"]["COLOR"]["pair_metrics"]["f1"] is None


@pytest.fixture
def alignment_rows():
    prepared, saved = [], []
    for seed_index, seed in enumerate(diagnostic._SEEDS):
        rows, direct_rows = [], []
        for index in range(222):
            global_index = seed_index * 222 + index
            subject = 1024 + index
            truth = [1025 if subject == 1024 else 1024]
            pair_exact = global_index < 569
            direct_exact = global_index < 336 or 569 <= global_index < 572
            logits = [-1.0] * 1025
            if direct_exact:
                logits[truth[0] - 1024] = 1.0
            baseline, direct = (truth if pair_exact else []), (truth if direct_exact else [])
            value = {
                "subject": f"PKM_{subject}",
                "subject_id": subject,
                "dimension": "COLOR",
                "group": "COLOR",
                "expected_set_ids": truth,
                "baseline_set_ids": baseline,
                "baseline_selection": {},
                "source_union_missing_true_ids": [],
                "logits": logits,
                "baseline_metrics": diagnostic._metrics(baseline, truth),
            }
            rows.append(value)
            direct_rows.append(
                {
                    **{k: v for k, v in value.items() if k != "logits"},
                    "predicted_set_ids": direct,
                    "direct_metrics": diagnostic._metrics(direct, truth),
                    "gained_exact": direct_exact and not pair_exact,
                    "lost_exact": pair_exact and not direct_exact,
                }
            )
        identity = {k: f"{seed}:{k}" for k in diagnostic._IDENTITY}
        prepared.append(({"seed": seed, **identity}, rows))
        saved.append({"seed": seed, "query_count": 222, **identity, "responses": direct_rows})
    return prepared, saved, list(range(1024, 2049))


def test_independent_zero_reconstruction_matches_frozen_counts(alignment_rows):
    assert diagnostic._alignment(*alignment_rows) == {
        "query_count": 666,
        "pair_exact": 569,
        "direct_exact": 339,
        "direct_gains": 3,
        "direct_losses": 233,
    }


@pytest.mark.parametrize("change", ["order", "identity", "set", "count"])
def test_saved_row_or_identity_alignment_drift_fails(alignment_rows, change):
    prepared, saved, ids = alignment_rows
    if change == "order":
        saved[0]["responses"][:2] = saved[0]["responses"][:2][::-1]
    elif change == "identity":
        saved[0]["checkpoint_hash"] = "changed"
    elif change == "set":
        saved[0]["responses"][0]["predicted_set_ids"] = []
    else:
        saved[0]["query_count"] = 221
    with pytest.raises(ValueError, match=r"alignment|identity|coverage|reconstruction"):
        diagnostic._alignment(prepared, saved, ids)


def test_immutable_outputs_and_inputs_fail_closed(tmp_path):
    output = tmp_path / "summary.json"
    (tmp_path / "independent-audit.py").write_bytes(b"# separately owned")
    diagnostic._refuse(output)
    path = tmp_path / "plan.md"
    path.write_bytes(b"declared")
    inputs = {}
    diagnostic._bind(path, diagnostic._sha(path), inputs)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        diagnostic._bind(path, next(iter(inputs.values())), {})
    with pytest.raises(ValueError, match="changed"):
        diagnostic._unchanged(inputs)
    diagnostic._write(tmp_path / "seed-1729.json", {})
    with pytest.raises(ValueError, match="immutable"):
        diagnostic._refuse(output)
    with pytest.raises(FileExistsError):
        diagnostic._write(tmp_path / "seed-1729.json", {})


def test_live_inventory_detects_changes_and_new_members(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    for relative in ("src/a.py", "configs/a.yaml", "pyproject.toml", "uv.lock"):
        (tmp_path / relative).write_bytes(b"original")
    inventory = diagnostic._live_inventory(tmp_path)
    (tmp_path / "configs/a.yaml").write_bytes(b"changed")
    assert diagnostic._live_inventory(tmp_path) != inventory
    (tmp_path / "configs/a.yaml").write_bytes(b"original")
    (tmp_path / "src/new.py").write_bytes(b"new")
    assert diagnostic._live_inventory(tmp_path) != inventory


def test_import_is_torch_free_in_fresh_interpreter():
    code = (
        "import runpy,sys; runpy.run_path('scripts/diagnose_membership_separability.py', "
        "run_name='cpu_import'); assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, check=True)


def test_script_edit_during_preflight_cannot_be_recorded_as_executed(tmp_path):
    script = tmp_path / "scripts/diagnose_membership_separability.py"
    script.parent.mkdir()
    script.write_bytes((ROOT / "scripts/diagnose_membership_separability.py").read_bytes())
    program = """
import importlib.util, sys
from pathlib import Path
path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('isolated', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def authenticate(root, plan):
    path.write_bytes(path.read_bytes() + b'\\n# changed during authentication\\n')
    return {'inputs': {}}
module._preflight = authenticate
sys.argv = [str(path), '--preflight-only', '--out', str(path.parent / 'summary.json')]
try:
    module.main()
except ValueError as exc:
    assert 'executing script/plan changed' in str(exc), str(exc)
else:
    raise AssertionError('changed executing script accepted')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", program, str(script)], capture_output=True, check=True
    )
