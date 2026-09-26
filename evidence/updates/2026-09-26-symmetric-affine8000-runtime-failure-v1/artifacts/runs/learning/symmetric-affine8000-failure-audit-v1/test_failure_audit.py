"""Synthetic mutation tests for the failure-only contract."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "failure_auditor", Path(__file__).with_name("audit_failure.py")
)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def fixtures():
    summary = {
        "campaign_version": "symmetric-affine8000-v1",
        "complete": False,
        "acceptance": False,
        "standard_serving_supported": False,
        "error": A.ERROR,
        "artifact_sha256": {"training.json": A.TRAINING_SHA},
        "failure_state_availability": {"model_available": False, "optimizer_available": False},
    }
    training = {"complete": False, "completed_updates": 0, "history": [], "error": A.ERROR}
    receipt = {
        "exit_code": 1,
        "terminal_completion_observed_by_primary": True,
        "terminal_chunk_id": "50ba53",
        "summary_sha256": A.SUMMARY_SHA,
        "training_sha256": A.TRAINING_SHA,
        "stdout_sha256": A.STDOUT_SHA,
        "candidate_quality_result": False,
    }
    stdout = (
        "line 965, in _execute\nline 466, in _load_selected_runtime\n"
        "ValueError: parent checkpoint identity\n"
    )
    return summary, training, receipt, stdout


def test_valid_failure():
    A.contract(*fixtures())


@pytest.mark.parametrize(
    "kind",
    [
        "complete",
        "accepted",
        "result",
        "error",
        "updates",
        "bool_updates",
        "history",
        "exit",
        "bool_exit",
        "unobserved",
        "hash",
        "model",
        "checkpoint",
        "stdout",
    ],
)
def test_corruption(kind):
    summary, training, receipt, stdout = fixtures()
    if kind == "complete":
        summary["complete"] = True
    elif kind == "accepted":
        summary["acceptance"] = True
    elif kind == "result":
        summary["child"] = {}
    elif kind == "error":
        training["error"] = "different"
    elif kind == "updates":
        training["completed_updates"] = 1
    elif kind == "bool_updates":
        training["completed_updates"] = False
    elif kind == "history":
        training["history"] = [1]
    elif kind == "exit":
        receipt["exit_code"] = 0
    elif kind == "bool_exit":
        receipt["exit_code"] = True
    elif kind == "unobserved":
        receipt["terminal_completion_observed_by_primary"] = False
    elif kind == "hash":
        receipt["summary_sha256"] = "wrong"
    elif kind == "model":
        summary["failure_state_availability"]["model_available"] = True
    elif kind == "checkpoint":
        summary["artifact_sha256"]["checkpoint-final.pt"] = "0" * 64
    else:
        stdout = "another error"
    with pytest.raises(ValueError):
        A.contract(summary, training, receipt, stdout)


def test_config_diagnosis():
    train = {"batch_size": 4}
    result = A.config_diagnosis({"config": train}, {"config": {"model": {}, "train": train}})
    assert result["sidecar_config_equals_run_train_config"]
    assert not result["later_short_circuited_predicates_independently_executed"]


def test_config_diagnosis_rejects_different_train_config():
    with pytest.raises(ValueError):
        A.config_diagnosis({"config": {"batch_size": 4}}, {"config": {"train": {"batch_size": 8}}})
