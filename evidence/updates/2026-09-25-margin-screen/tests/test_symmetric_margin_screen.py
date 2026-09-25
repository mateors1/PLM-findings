"""CPU-only synthetic checks for the fixed, immutable margin screen."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from plm.config import RootConfig
from plm.configuration import config_hash

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "margin_screen", ROOT / "scripts/run_symmetric_margin_screen.py"
)
assert SPEC is not None and SPEC.loader is not None
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


def recipes():
    base = RootConfig(seed=1729).model_dump(mode="json")
    base["model"]["symmetric_relation_loss_weight"] = 1.0
    arms = []
    for name, weight in zip(screen._NAMES, (0.0, 0.1), strict=True):
        raw = copy.deepcopy(base)
        raw["run_name"] = name
        raw["model"]["symmetric_margin_loss_weight"] = weight
        arms.append(raw)
    return base, arms


def test_recipe_preserves_every_other_field_and_eval_does_not_mutate_training():
    base, arms = recipes()
    screen._comparability(arms, base)
    before = copy.deepcopy(arms)
    evaluation = screen._evaluation_config(arms[1])
    assert arms == before
    assert evaluation.eval.generation_batch_size == 8
    assert evaluation.eval.max_new_tokens == 507
    assert evaluation.eval.first_target_guidance_alpha == 16
    assert evaluation.eval.pair_set_composition
    assert evaluation.model == RootConfig.model_validate(arms[1]).model


@pytest.mark.parametrize("change", ["seed", "run_name", "learning_rate", "margin", "weight"])
def test_recipe_drift_is_rejected(change):
    base, arms = recipes()
    if change == "seed":
        arms[1]["seed"] = 1730
    elif change == "run_name":
        arms[1]["run_name"] = "other"
    elif change == "learning_rate":
        arms[1]["train"]["learning_rate"] = 0.5
    else:
        arms[1]["model"][
            "symmetric_margin" if change == "margin" else "symmetric_margin_loss_weight"
        ] = 2.0
    with pytest.raises(ValueError, match="recipe"):
        screen._comparability(arms, base)


def test_outputs_run_directories_and_snapshot_are_immutable(tmp_path):
    _, arms = recipes()
    out = tmp_path / "results/summary.json"
    out.parent.mkdir()
    screen._refuse(out, tmp_path, arms)
    screen._write(out, {"complete": False})
    with pytest.raises(FileExistsError):
        screen._write(out, {"complete": True})
    with pytest.raises(ValueError, match="output exists"):
        screen._refuse(out, tmp_path, arms)
    other = tmp_path / "other/summary.json"
    (tmp_path / arms[0]["paths"]["run_root"] / arms[0]["run_name"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="run directory"):
        screen._refuse(other, tmp_path, arms)


def composition():
    tokens = [f"TOKEN_{i}" for i in range(2049)]
    tokens[2] = "EOS"
    prompt = [1, 1024, 32, 34, 3]
    logits = [0.0] * 1025
    logits[1:5] = [3.0, 2.0, -1.0, 0.0]
    sources = []
    for rank in range(1, 5):
        entity = 1024 + rank
        sources.append(
            {
                "token_ids": [*prompt, entity, 2],
                "targets": [tokens[entity]],
                "terminated": True,
                "protocol_valid": True,
                "error": None,
                "decoding": screen._BASE + (f"+first-rank{rank}-v1" if rank > 1 else ""),
            }
        )
    slots = [
        {
            "slot": i,
            "kind": "original" if i <= 4 else "pair_composition",
            "source_ranks": list(ranks),
            "set_ids": [1024 + r for r in ranks],
            "source_eligible": True,
            "score": math.fsum(logits[r] for r in ranks),
        }
        for i, ranks in enumerate(screen._ORDER, 1)
    ]
    selected = slots[4]
    value = {
        "source_paths": sources,
        "slots": slots,
        "policy": screen._POLICY,
        "selected_slot": 5,
        "selected_kind": selected["kind"],
        "selected_source_ranks": [1, 2],
        "selected_set_ids": [1025, 1026],
        "selected_source_eligible": True,
        "fallback_no_valid_source": False,
    }
    return value, logits, prompt, tokens


def test_exact_slot_score_reconstruction_detects_one_ulp():
    value, logits, prompt, tokens = composition()
    screen._verify_composition(value, logits, prompt, tokens)
    value["slots"][4]["score"] = math.nextafter(value["slots"][4]["score"], math.inf)
    with pytest.raises(ValueError, match="exact same-shape score"):
        screen._verify_composition(value, logits, prompt, tokens)


def test_failed_observation_retains_raw_vector_and_completed_rows(tmp_path):
    value, logits, prompt, tokens = composition()
    record = SimpleNamespace(subject=tokens[1024], dimension="TYPE", targets=[tokens[1025]])
    report = {"complete": False, "query_count": 0, "responses": []}
    screen._record_observation(
        report, 0, record, {"group": "TYPE_single"}, value, logits, prompt, tokens, 0, 8
    )
    assert "failed_observation" not in report and report["query_count"] == 1
    bad = copy.deepcopy(value)
    bad["slots"][4]["score"] = math.nextafter(bad["slots"][4]["score"], math.inf)
    with pytest.raises(ValueError, match="exact same-shape score"):
        screen._record_observation(
            report, 1, record, {"group": "TYPE_single"}, bad, logits, prompt, tokens, 0, 8
        )
    screen._write(tmp_path / "partial.json", report)
    saved = screen._read(tmp_path / "partial.json")
    assert saved["query_count"] == len(saved["responses"]) == 1
    assert saved["complete"] is False
    assert saved["failed_observation"]["index"] == 1
    assert saved["failed_observation"]["composition"] == bad
    assert saved["failed_observation"]["symmetric_relation_logits"] == logits


@pytest.mark.parametrize("flag", ["passed", "synthetic", "real_corpus_used"])
def test_smoke_receipt_flags_are_strict_booleans(tmp_path, flag):
    raw_path = tmp_path / "raw.json"
    screen._write(
        raw_path,
        {"complete": True, "steps": [{"step": i, "loss": 1.0, "margin": 0.5} for i in (1, 2, 3)]},
    )
    receipt = {
        "passed": True,
        "source_archive_sha256": "source",
        "synthetic": True,
        "real_corpus_used": False,
        "steps": 3,
        "config_files_sha256": {},
        "implementation_source_sha256": {"src/model.py": "model"},
        "raw_receipt": {"path": "raw.json", "sha256": screen._sha(raw_path)},
    }
    context = {"root": tmp_path, "inputs": {}, "live_inventory": {"src/model.py": "model"}}
    good = tmp_path / "good.json"
    screen._write(good, receipt)
    screen._receipt(good, screen._sha(good), context, "GPU smoke", "source")
    receipt[flag] = int(receipt[flag])
    bad = tmp_path / "bad.json"
    screen._write(bad, receipt)
    with pytest.raises(ValueError, match=r"verification/source|scope"):
        screen._receipt(bad, screen._sha(bad), context, "GPU smoke", "source")


def test_identical_score_tie_requires_earliest_slot():
    value, logits, prompt, tokens = composition()
    logits[2] = 0.0
    for slot in value["slots"]:
        slot["score"] = math.fsum(logits[i - 1024] for i in slot["set_ids"])
    with pytest.raises(ValueError, match="selection/tie"):
        screen._verify_composition(value, logits, prompt, tokens)


def test_head_analysis_subject_zero_tie_and_fp32():
    values = [-1.0] * 1025
    values[0:3] = [100.0, 1.0, 0.0]
    result = screen._head_analysis(values, [1025], 1024)
    assert result["direct_set_ids"] == [1025]
    assert result["strict_separable"] is True
    values[2] = 1.0
    assert screen._head_analysis(values, [1025], 1024)["strict_separable"] is False
    values[2] = 0.1
    with pytest.raises(ValueError, match="FP32"):
        screen._head_analysis(values, [1025], 1024)


def reports():
    value = composition()[0]
    rows = [
        {
            "index": i,
            "subject": f"PKM_{i}",
            "dimension": "COLOR" if i % 3 == 0 else "TYPE",
            "group": screen._GROUPS[i % 3],
            "expected_set_ids": [1025],
            "composition": copy.deepcopy(value),
            "selected_metrics": {"exact": i < 200},
        }
        for i in range(222)
    ]
    control = {
        "complete": True,
        "responses": rows,
        "metrics": {"exact_set_count": 200, "f1": 0.9},
        "groups": {g: {"metrics": {"exact_set_count": 60}} for g in screen._GROUPS},
        "head_diagnostics": {"strict_separation_count": 180},
    }
    treatment = copy.deepcopy(control)
    treatment["responses"][200]["selected_metrics"]["exact"] = True
    treatment["metrics"]["exact_set_count"] = 201
    treatment["head_diagnostics"]["strict_separation_count"] = 181
    return control, treatment


def test_fixed_screen_gates_do_not_self_accept():
    result = screen._gate(*reports())
    assert result["numerical_gates_passed"]
    assert result["gained_exact_count"] == 1 and result["lost_exact_count"] == 0
    assert result["stage_one_accepted"] is False and result["replication_authorized"] is False


@pytest.mark.parametrize("regression", ["exact", "f1", "group", "separation", "source"])
def test_each_fixed_gate_can_stop_replication(regression):
    control, treatment = reports()
    if regression == "exact":
        treatment["metrics"]["exact_set_count"] = 200
    elif regression == "f1":
        treatment["metrics"]["f1"] = 0.89
    elif regression == "group":
        treatment["groups"]["TYPE_dual"]["metrics"]["exact_set_count"] = 59
    elif regression == "separation":
        treatment["head_diagnostics"]["strict_separation_count"] = 180
    else:
        treatment["responses"][0]["composition"]["source_paths"][0]["terminated"] = False
    assert screen._gate(control, treatment)["numerical_gates_passed"] is False


def test_paired_alignment_failure_is_not_scored():
    control, treatment = reports()
    treatment["responses"].reverse()
    with pytest.raises(ValueError, match="paired alignment"):
        screen._gate(control, treatment)


def test_cleanup_cannot_hide_original_training_error():
    def train(config):
        raise RuntimeError("original training failure")

    def cleanup():
        raise ValueError("cleanup failed")

    errors = []
    with pytest.raises(RuntimeError, match="original training failure"):
        screen._train_once(train, None, cleanup, errors)
    assert errors == ["ValueError('cleanup failed')"]


def test_script_and_source_inventory_drift(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    for name in ("src/model.py", "configs/config.yaml", "pyproject.toml", "uv.lock", "screen.py"):
        (tmp_path / name).write_text("original", encoding="utf-8")
    context = {
        "root": tmp_path,
        "inputs": {str(tmp_path / "screen.py"): screen._sha(tmp_path / "screen.py")},
        "live_inventory": screen._inventory(tmp_path),
    }
    screen._unchanged(context)
    (tmp_path / "screen.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="executing script"):
        screen._unchanged(context)
    (tmp_path / "src/model.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source/config"):
        screen._unchanged(context)


def test_actual_training_receipt_shape_and_config_mismatch(tmp_path):
    config = recipes()[1][0]
    directory = tmp_path / config["paths"]["run_root"] / config["run_name"]
    directory.mkdir(parents=True)
    (directory / "source.zip").write_bytes(b"archive")
    (directory / "checkpoint-final.pt").write_bytes(b"weights")
    environment = {"source_archive_sha256": screen._sha(directory / "source.zip")}
    corpus = {"graph_hash": "graph", "records_hash": "records", "vocabulary_hash": "vocab"}
    identity = {
        "resolved_config_hash": config_hash(RootConfig.model_validate(config)),
        "environment": environment,
        "split_hash": "split",
        "seeds": [1729],
        **corpus,
    }
    history = [{"step": 2000.0, "last_batch_token_loss": 0.4}]
    checkpoint = screen._sha(directory / "checkpoint-final.pt")
    screen._write(directory / "run.json", {"config": config, "identity": identity})
    screen._write(
        directory / "training-result.json",
        {
            "identity": identity,
            "global_step": 2000,
            "checkpoint_hash": checkpoint,
            "history": history,
            "training_wall_seconds": 1.0,
        },
    )
    screen._write(
        directory / "checkpoint-final.pt.json",
        {
            "experiment_identity": identity,
            "global_step": 2000,
            "checkpoint_hash": checkpoint,
            "training_metadata": {
                "objective": "causal-next-token-v1",
                "model_config": config["model"],
            },
        },
    )
    (directory / "metrics.jsonl").write_text(json.dumps(history[0]) + "\n", encoding="utf-8")
    context = {"root": tmp_path, "inputs": {}, "split_hash": "split", "corpus_identity": corpus}
    receipt = screen._training_receipt("control", config, context, environment)
    assert receipt["training_history"] == history and receipt["global_step"] == 2000
    altered = copy.deepcopy(config)
    altered["model"]["symmetric_margin"] = 2.0
    with pytest.raises(ValueError, match="fresh training identity"):
        screen._training_receipt("control", altered, context, environment)


def test_standalone_import_is_torch_free():
    code = (
        "import runpy,sys; runpy.run_path('scripts/run_symmetric_margin_screen.py',"
        "run_name='screen_test'); assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)
