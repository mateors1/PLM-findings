"""Synthetic weak-screen lifecycle checks; no model, GPU or experiment execution."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from plm.config import RootConfig

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "weak_margin", ROOT / "scripts/run_weak_margin_screen.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def recipes():
    base = RootConfig(seed=1729).model_dump(mode="json")
    base["model"]["symmetric_relation_loss_weight"] = 1.0
    arms = []
    for name, weight in zip(runner._NAMES, (0.0, 0.001), strict=True):
        arm = copy.deepcopy(base)
        arm["run_name"] = name
        arm["model"]["symmetric_margin_loss_weight"] = weight
        arms.append(arm)
    return base, arms


def test_exact_weak_recipe_preserves_every_other_field():
    base, arms = recipes()
    original = copy.deepcopy(arms)
    runner._comparability(arms, base)
    assert arms == original
    assert arms[1]["model"]["symmetric_margin_loss_weight"] == 0.001


@pytest.mark.parametrize("change", ["coefficient", "old_name", "lr", "seed", "margin", "boolean"])
def test_wrong_recipe_cannot_run(change):
    base, arms = recipes()
    if change == "coefficient":
        arms[1]["model"]["symmetric_margin_loss_weight"] = 0.1
    elif change == "old_name":
        arms[0]["run_name"] = "national_dex_rank_margin_control_s1729_v1"
    elif change == "lr":
        arms[1]["train"]["learning_rate"] = 0.1
    elif change == "seed":
        arms[1]["seed"] = 1730
    elif change == "margin":
        arms[1]["model"]["symmetric_margin"] = 2.0
    else:
        arms[0]["model"]["symmetric_margin_loss_weight"] = False
    with pytest.raises(ValueError, match="recipe"):
        runner._comparability(arms, base)


def test_helper_authentication_precedes_execution(tmp_path):
    helper = tmp_path / "helper.py"
    marker = tmp_path / "executed"
    helper.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="helper identity"):
        runner._load_helper(helper)
    assert not marker.exists()


@pytest.fixture
def archived():
    path = ROOT / "runs/learning/symmetric-margin-screen-v1/summary.script.py"
    if not path.is_file():
        pytest.skip("optional authenticated local experiment helper absent")
    return runner._load_helper(path)


def test_archived_helper_remains_unchanged_and_rejects_output_collision(tmp_path, archived):
    before = (archived._NAMES, archived._PLAN_SHA, archived._sha(Path(archived.__file__)))
    base, arms = recipes()
    runner._comparability(arms, base)
    out = tmp_path / "results/summary.json"
    out.parent.mkdir()
    (out.parent / "independent-audit.py").write_text("# independent", encoding="utf-8")
    archived._refuse(out, tmp_path, arms)
    archived._write(out, {"complete": False})
    with pytest.raises(ValueError, match="immutable"):
        archived._refuse(out, tmp_path, arms)
    assert (archived._NAMES, archived._PLAN_SHA, archived._sha(Path(archived.__file__))) == before


def lifecycle(tmp_path, archived, *, failure=None):
    events = []
    environment = {"source_archive_sha256": runner._SOURCE}
    config = recipes()[1]
    result = {
        "complete": False,
        "stage_one_accepted": False,
        "script_sha256": "new-runner",
        "training": [],
        "evaluation": [],
        "cleanup_errors": [],
        "snapshot_sha256": {},
    }

    def training_receipt(arm, raw, context, env):
        assert len([e for e in events if e.startswith("train")]) == (1 if arm == "control" else 2)
        return {"arm": arm, "config": raw}

    def evaluate(arm, training, context, output):
        assert events[:2] == ["train_control", "train_treatment"]
        assert (tmp_path / "training-seal.json").is_file()
        events.append("evaluate_" + arm)
        if failure == "evaluation":
            archived._write(
                output,
                {
                    "complete": False,
                    "query_count": 1,
                    "responses": [{"index": 0}],
                    "error": "synthetic evaluation failure",
                },
            )
            raise RuntimeError("synthetic evaluation failure")
        report = {"complete": True, "numerical_settings": {"FP32": True}, "responses": []}
        archived._write(output, report)
        return report

    # A new test double wraps immutable I/O helpers; archived globals are never patched.
    helper = SimpleNamespace(
        _write=archived._write,
        _read=archived._read,
        _sha=archived._sha,
        _bind=archived._bind,
        _train_once=archived._train_once,
        _unchanged=lambda context: None,
        _training_receipt=training_receipt,
        _evaluate=evaluate,
        _gate=lambda *reports: {"stage_one_accepted": False},
    )
    context = {
        "helper": helper,
        "configs": config,
        "inputs": {},
        "cpu_receipt": {"environment": environment},
        "gpu_receipt": {"environment": environment},
        "failed": {"environment": environment},
    }

    def train(raw):
        arm = "control" if raw["run_name"] == runner._NAMES[0] else "treatment"
        events.append("train_" + arm)
        if failure == "training" and arm == "treatment":
            raise RuntimeError("synthetic training failure")

    def cleanup():
        if failure == "training" and events[-1] == "train_treatment":
            raise ValueError("synthetic cleanup failure")

    def capture():
        if failure == "capture":
            raise ImportError("synthetic runtime import failure")
        return "revision", environment

    return context, result, train, cleanup, capture, events


def test_both_fresh_trainings_finish_and_seal_before_evaluation(tmp_path, archived):
    context, result, train, cleanup, capture, events = lifecycle(tmp_path, archived)
    out = tmp_path / "summary.json"
    runner._execute(context, out, result, train, lambda raw: raw, cleanup, capture)
    assert events == ["train_control", "train_treatment", "evaluate_control", "evaluate_treatment"]
    saved = json.loads(out.read_text())
    assert saved["complete"] is True and saved["stage_one_accepted"] is False
    seal = json.loads((tmp_path / "training-seal.json").read_text())
    assert seal["helper_sha256"] == runner._HELPER and seal["script_sha256"] == "new-runner"


@pytest.mark.parametrize("failure", ["training", "evaluation", "capture"])
def test_failed_lifecycle_preserves_prior_evidence_and_primary_exception(
    tmp_path, archived, failure
):
    context, result, train, cleanup, capture, events = lifecycle(
        tmp_path, archived, failure=failure
    )
    out = tmp_path / "summary.json"
    with pytest.raises((RuntimeError, ImportError), match="synthetic"):
        runner._execute(context, out, result, train, lambda raw: raw, cleanup, capture)
    saved = json.loads(out.read_text())
    assert saved["complete"] is False and "synthetic" in saved["error"]
    if failure == "training":
        assert saved["cleanup_errors"] == ["ValueError('synthetic cleanup failure')"]
        assert saved["training"][0]["arm"] == "control"
        assert not (tmp_path / "training-seal.json").exists()
        assert not any(e.startswith("evaluate") for e in events)
    elif failure == "evaluation":
        assert len(saved["training"]) == 2
        assert "evaluation-control.json" in saved["report_sha256"]
        assert json.loads((tmp_path / "evaluation-control.json").read_text())["query_count"] == 1
        assert "evaluate_treatment" not in events
    else:
        assert events == []


def test_import_is_torch_free():
    code = (
        "import runpy,sys; runpy.run_path('scripts/run_weak_margin_screen.py',"
        "run_name='test_import'); assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)
