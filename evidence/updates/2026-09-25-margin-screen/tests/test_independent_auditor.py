"""Synthetic-only audit checks; no new model result or experiment measurement."""

import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "runs/learning/symmetric-margin-screen-v1/independent-audit.py"
spec = importlib.util.spec_from_file_location("margin_independent_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    tokens = [f"reserved{i}" for i in range(1024)] + [f"PKM_{i}" for i in range(1025)]
    for index, name in (
        (1, "BOS"),
        (2, "EOS"),
        (5, "ANSWER"),
        (32, "TYPE"),
        (33, "COLOR"),
        (64, "SAME"),
    ):
        tokens[index] = name
    head = [-1.0] * 1025
    head[:3] = [99.0, 2.0, 3.0]
    return tokens, head


def paths(tokens, products=None, *, bound=507, ended=True):
    products = products or [[1025], [1026], [1024, 1025], [1027]]
    return [
        audit.parse_path(
            [1, 1024, 32, 64, 5] + ids + ([2] if ended else []),
            tokens[1024],
            "TYPE",
            tokens,
            audit.BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            bound,
        )
        for rank, ids in enumerate(products, 1)
    ]


def test_pool_math_subject_postprocess_and_exact_ties():
    tokens, head = fixture()
    result = audit.build_composition(paths(tokens), head, tokens[1024], "TYPE", tokens)
    assert [s["set_ids"] for s in result["slots"]] == [
        [1025],
        [1026],
        [1025],
        [1027],
        [1025, 1026],
        [1025],
        [1025, 1027],
        [1025, 1026],
        [1026, 1027],
        [1025, 1027],
    ]
    assert [s["score"] for s in result["slots"]] == [
        2.0,
        3.0,
        2.0,
        -1.0,
        5.0,
        2.0,
        1.0,
        5.0,
        2.0,
        1.0,
    ]
    assert result["selected_slot"] == 5
    assert result["selected_source_ranks"] == [1, 2]


def test_invalid_sources_preserve_raw_membership_and_cannot_win():
    tokens, head = fixture()
    sources = paths(tokens, [[1024], [1025], [1026], [1027]], bound=1, ended=False)
    result = audit.build_composition(sources, head, tokens[1024], "TYPE", tokens, bound=1)
    assert result["fallback_no_valid_source"] is True
    assert result["selected_set_ids"] == [1024]
    assert result["slots"][0]["score"] == 99.0
    assert not any(s["source_eligible"] for s in result["slots"])
    sources = paths(tokens, [[1025, 1025], [1026], [1024, 1025], [1027]])
    result = audit.build_composition(sources, head, tokens[1024], "TYPE", tokens)
    assert result["selected_slot"] == 8
    assert result["slots"][0]["source_eligible"] is False


@pytest.mark.parametrize(
    "mutation", ["flag", "early", "budget", "control", "bool", "nan", "float64"]
)
def test_source_and_head_corruption_rejected(mutation):
    tokens, head = fixture()
    source = paths(tokens)
    if mutation == "flag":
        source[0]["terminated"] = False
    elif mutation == "early":
        source[0]["token_ids"].pop()
    elif mutation == "budget":
        source[0]["token_ids"] = source[0]["token_ids"][:5] + [1025] * 507 + [2]
    elif mutation == "control":
        source[0]["token_ids"][5] = 5
    elif mutation == "bool":
        source[0]["token_ids"][5] = True
    elif mutation == "nan":
        head[1] = math.nan
    else:
        head[1] = 0.1
    with pytest.raises(ValueError):
        audit.build_composition(source, head, tokens[1024], "TYPE", tokens)


def test_separation_not_zero_threshold_accuracy_and_tied_boundary():
    _, head = fixture()
    head[1:3] = [-0.5, -0.25]
    value = audit.membership(head, 1024, [1025, 1026])
    assert value["strict_separable"] is True
    assert value["gap"] == 0.5
    assert value["direct_set_ids"] == []
    assert value["direct_metrics"]["fn"] == 2
    head[3] = -0.5
    value = audit.membership(head, 1024, [1025, 1026])
    assert value["strict_separable"] is False
    assert value["gap"] == 0
    assert value["min_true_ids"] == [1025]
    assert value["max_negative_ids"] == [1027]


def synthetic_report():
    tokens, head = fixture()
    original = json.loads(
        (ROOT / "runs/national_dex_continuation_control_s1729_v1/run.json").read_text()
    )
    config = audit.evaluation_recipe(audit.effective_recipe(original, "control"))
    answer = audit.build_composition(paths(tokens), head, tokens[1024], "TYPE", tokens)
    exact = {
        "exact": True,
        "tp": 2,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "set_size": 2,
    }
    rows, validation = [], []
    for i in range(222):
        group = audit.GROUPS[i % 3]
        validation.append(
            {
                "subject": tokens[1024],
                "dimension": "TYPE",
                "group": group,
                "expected": tokens[1025:1027],
            }
        )
        rows.append(
            {
                "index": i,
                "subject": tokens[1024],
                "dimension": "TYPE",
                "group": group,
                "expected_set_ids": [1025, 1026],
                "prompt_ids": [1, 1024, 32, 64, 5],
                "batch_index": i // 8,
                "batch_shape": [8 if i < 216 else 6, 5],
                "composition": copy.deepcopy(answer),
                "symmetric_relation_logits": head.copy(),
                "selected_metrics": exact.copy(),
                "direct_set_ids": [1025, 1026],
                "direct_metrics": exact.copy(),
                "strict_separable": True,
                "min_true_logit": 2.0,
                "max_negative_logit": -1.0,
                "gap": 3.0,
            }
        )
    training = {"checkpoint_hash": "synthetic", "identity": {"synthetic": True}}
    report = {
        "complete": True,
        "arm": "control",
        "query_count": 222,
        "responses": rows,
        "cleanup_errors": [],
        "config": config,
        "config_hash": audit.canonical(config),
        "checkpoint_hash": "synthetic",
        "training_identity": training["identity"],
        "corpus_identity": {
            k: original["identity"][k] for k in ("graph_hash", "records_hash", "vocabulary_hash")
        },
        "split_hash": original["identity"]["split_hash"],
        "product_token_ids": list(range(1024, 2049)),
        "head_reference_batch_count": 28,
        "numerical_settings": {"parameter_dtype": "torch.float32", "autocast_cuda_enabled": False},
        "evaluation_wall_seconds": 1.0,
        "metrics": audit.aggregate_compositions(rows),
        "head_diagnostics": audit.head_totals(rows),
        "groups": {
            g: {
                "metrics": audit.aggregate_compositions(rows[i::3]),
                "head_diagnostics": audit.head_totals(rows[i::3]),
            }
            for i, g in enumerate(audit.GROUPS)
        },
    }
    return report, training, original, tokens, validation


def test_complete_synthetic_report_and_aggregates():
    report, training, original, tokens, validation = synthetic_report()
    actual = audit.evaluation_evidence(report, "control", training, original, tokens, validation)
    assert actual["metrics"]["exact_set_count"] == 222
    assert actual["metrics"]["source_path_count"] == 888
    assert actual["metrics"]["pair_selection_count"] == 222
    assert actual["head_diagnostics"]["strict_separation_count"] == 222
    assert all(g["metrics"]["exact_set_count"] == 74 for g in actual["groups"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        "score",
        "tie",
        "shape",
        "label",
        "metric",
        "group",
        "config",
        "hash",
        "dtype",
        "source",
        "column",
    ],
)
def test_report_mutations_fail(mutation):
    report, training, original, tokens, validation = synthetic_report()
    row = report["responses"][0]
    if mutation == "score":
        row["composition"]["slots"][0]["score"] += 0.0001
    elif mutation == "tie":
        row["composition"]["selected_slot"] = 8
    elif mutation == "shape":
        report["responses"][-1]["batch_shape"] = [8, 5]
    elif mutation == "label":
        row["expected_set_ids"] = [1025]
    elif mutation == "metric":
        report["metrics"]["f1"] = 0.999
    elif mutation == "group":
        report["groups"]["COLOR"]["head_diagnostics"]["strict_separation_count"] = 73
    elif mutation == "config":
        report["config"]["model"]["symmetric_margin_loss_weight"] = 0.1
    elif mutation == "hash":
        report["config_hash"] = "wrong"
    elif mutation == "dtype":
        report["numerical_settings"]["autocast_cuda_enabled"] = True
    elif mutation == "source":
        row["composition"]["source_paths"][0]["targets"] = [tokens[1026]]
    else:
        report["product_token_ids"].reverse()
    with pytest.raises(ValueError):
        audit.evaluation_evidence(report, "control", training, original, tokens, validation)


def test_paired_gates_require_every_predeclared_condition():
    control = {
        "all_valid": True,
        "exact_count": 10,
        "f1": 0.8,
        "groups": dict.fromkeys(audit.GROUPS, 3),
        "separation_count": 5,
    }
    treatment = {
        "all_valid": True,
        "exact_count": 11,
        "f1": 0.8,
        "groups": dict.fromkeys(audit.GROUPS, 3),
        "separation_count": 6,
    }
    assert all(audit.paired_gate(control, treatment).values())
    for name, value in (
        ("all_valid", False),
        ("exact_count", 10),
        ("f1", 0.799),
        ("separation_count", 5),
        ("groups", dict.fromkeys(audit.GROUPS, 2)),
    ):
        corrupted = copy.deepcopy(treatment)
        corrupted[name] = value
        assert sum(not p for p in audit.paired_gate(control, corrupted).values()) == 1


def test_entry_point_missing_or_partial_summary_writes_nothing(tmp_path):
    for value in (None, {"complete": False}):
        path = tmp_path / "summary.json"
        if value is not None:
            path.write_text(json.dumps(value))
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--summary", str(path)], capture_output=True, text=True
        )
        assert result.returncode != 0
        assert not (tmp_path / "independent-audit.json").exists()


def test_final_gate_negative_result_still_has_honest_paired_receipts():
    report, _, _, _, _ = synthetic_report()
    control = copy.deepcopy(report)
    treatment = copy.deepcopy(report)
    control["responses"][0]["selected_metrics"]["exact"] = False
    treatment["responses"][1]["selected_metrics"]["exact"] = False
    treatment["groups"]["COLOR"]["metrics"]["exact_set_count"] -= 1
    result = audit.final_gate(control, treatment)
    assert result["numerical_gates_passed"] is False
    assert result["gained_exact_count"] == result["lost_exact_count"] == 1
    assert result["gained_exact"][0]["index"] == 0
    assert result["lost_exact"][0]["index"] == 1
    assert result["checks"]["groups_exact_nondecreased"] is False
    assert result["stage_one_accepted"] is result["replication_authorized"] is False
    treatment["responses"][0]["subject"] = "different"
    with pytest.raises(ValueError, match="paired validation identity"):
        audit.final_gate(control, treatment)
