"""Synthetic signature and diagnostic-state audit tests."""

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "failure_v2", Path(__file__).with_name("audit_failure.py")
)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)

SOURCE = """
def _evaluate(model, records, wide_rows, mean, helper, path, affine, bilinear,
              parent=None, batch_size=8, phase="child"):
    raise AssertionError("must never execute")
def _execute():
    parent = _evaluate(model, records, rows, helper, path, affine, bilinear, phase="parent")
"""


def test_missing_mean_signature_is_independently_reproduced():
    report = A.call_diagnosis(SOURCE)
    assert not report["helper_body_entered"]
    assert "bilinear" in report["independent_binding_error"]


def test_corrected_call_cannot_be_labeled_same_failure():
    with pytest.raises(ValueError):
        A.call_diagnosis(SOURCE.replace("records, rows, helper", "records, rows, mean, helper"))


def payload():
    state = {f"parameter_{i}": torch.tensor([float(i)]) for i in range(93)}
    expected = {
        n: {
            "sha256": hashlib.sha256(v.numpy().tobytes()).hexdigest(),
            "shape": [1],
            "dtype": "torch.float32",
        }
        for n, v in state.items()
    }
    return {
        "model_state": state,
        "optimizer_state": None,
        "completed_updates": 0,
        "optimizer_steps_attempted": 0,
        "diagnostic_only": True,
        "resume_allowed": False,
    }, expected


def test_original_parent_diagnostic_state():
    data, expected = payload()
    report = A.inspect_failure_payload(data, expected)
    assert report["unchanged_original_tensors"] == 93
    assert not report["new_trained_model_checkpoint"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "changed",
        "nan",
        "dtype",
        "shape",
        "optimizer",
        "trained",
        "resume",
        "not_diagnostic",
    ],
)
def test_state_corruption(mutation):
    data, expected = payload()
    if mutation == "missing":
        data["model_state"].pop("parameter_0")
    elif mutation == "extra":
        data["model_state"]["symmetric_affine_bias"] = torch.zeros(2)
    elif mutation == "changed":
        data["model_state"]["parameter_0"][0] = 1
    elif mutation == "nan":
        data["model_state"]["parameter_0"][0] = float("nan")
    elif mutation == "dtype":
        data["model_state"]["parameter_0"] = torch.zeros(1, dtype=torch.float64)
    elif mutation == "shape":
        data["model_state"]["parameter_0"] = torch.zeros(1, 1)
    elif mutation == "optimizer":
        data["optimizer_state"] = {}
    elif mutation == "trained":
        data["completed_updates"] = 1
    elif mutation == "resume":
        data["resume_allowed"] = True
    else:
        data["diagnostic_only"] = False
    with pytest.raises(ValueError):
        A.inspect_failure_payload(data, expected)
