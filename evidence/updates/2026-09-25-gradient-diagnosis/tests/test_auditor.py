"""Small synthetic-only gradient auditor cases, without Torch imports."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "runs/learning/margin-gradient-diagnosis-v1/independent-audit.py"
)
spec = importlib.util.spec_from_file_location("gradient_auditor", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_reductions_weighting_and_undefined_cases():
    a = np.array([3, 4], dtype=np.float32)
    b = np.array([-4, 3], dtype=np.float32)
    assert audit.magnitude(a) == 5.0
    assert audit.dot(a, b) == 0.0
    assert audit.directions(a, b) == {"cosine": 0.0, "norm_ratio": 1.0, "reason": None}
    weighted = a.astype(np.float64) * 0.1
    assert audit.magnitude(weighted) == 0.5
    assert audit.directions(weighted, a)["norm_ratio"] == 0.1
    assert audit.directions(a, -a)["cosine"] == -1.0
    zero = np.zeros_like(a)
    for first, second, reason in (
        (zero, a, "left_zero"),
        (a, zero, "right_zero"),
        (zero, zero, "both_zero"),
    ):
        assert audit.directions(first, second) == {
            "cosine": None,
            "norm_ratio": None,
            "reason": reason,
        }


def arrays():
    ids = np.array([[1, 1024, 32, 64, 5, 1025, 2, 0]], dtype=np.int64)
    labels = np.array([[-100, -100, -100, -100, -100, 1025, 2, 1026]], dtype=np.int64)
    attention = np.array([[True] * 7 + [False]])
    logits = np.full((1, 1025), -2, dtype=np.float32)
    logits[0, :3] = [100, 1, 2]
    return logits, labels, ids, attention


def test_padding_and_subject_are_excluded_from_membership():
    logits, labels, ids, attention = arrays()
    result = audit.margin_masks(logits, labels, ids, attention)
    assert np.flatnonzero(result["positive"]).tolist() == [1]
    assert not result["negative"][0, 0]
    assert result["negative"][0, 2]
    assert result["margin"] == 2
    assert result["radial_rhs"] == 1


@pytest.mark.parametrize(
    "positive,hinge,active,radial",
    [(4, 0, False, 0), (3, 0, False, 0), (2, 1, True, 0), (1, 2, True, 1)],
)
def test_radial_boundary_and_inactive_rows(positive, hinge, active, radial):
    logits, labels, ids, attention = arrays()
    logits[0, 1] = positive
    result = audit.margin_masks(logits, labels, ids, attention)
    assert result["margin"] == hinge
    assert bool(result["active"][0]) is active
    assert result["radial_rhs"] == radial


def test_fp32_boundary_activity_and_fp64_radial_reduction():
    logits, labels, ids, attention = arrays()
    logits[0, 1] = np.float32(3.0)
    logits[0, 2] = np.nextafter(np.float32(2), np.float32(0))
    result = audit.margin_masks(logits, labels, ids, attention)
    assert not result["active"].any()
    assert result["radial_rhs"] == 0


def test_fixed_tolerances_reject_outside_boundary():
    audit.close(1 + 1e-6, 1.0, rtol=1e-5, atol=1e-6, label="loss")
    with pytest.raises(ValueError, match="loss"):
        audit.close(1.0001, 1.0, rtol=1e-5, atol=1e-6, label="loss")
    with pytest.raises(ValueError):
        audit.close(float("nan"), 1.0, rtol=1e-5, atol=1e-6, label="loss")


def test_nonfinite_shape_dtype_and_subject_teacher_rejected():
    with pytest.raises(ValueError):
        audit.checked_array(np.zeros(3, dtype=np.float64), (3,), np.float32, "gradient")
    with pytest.raises(ValueError):
        audit.checked_array(np.array([np.nan], dtype=np.float32), (1,), np.float32, "gradient")
    logits, labels, ids, attention = arrays()
    labels[0, 5] = 1024
    with pytest.raises(ValueError, match="subject"):
        audit.margin_masks(logits, labels, ids, attention)


def test_unused_is_not_the_same_as_a_measured_zero_derivative():
    parameter = np.array([2, 4], dtype=np.float32)
    raw = {
        name: np.zeros(2, dtype=np.float32)
        for name in ("ce", "prompt_bce", "symmetric_bce", "margin")
    }
    raw["symmetric_bce"][:] = [1, 3]
    raw["margin"][:] = [2, -4]
    unused = {"ce": True, "prompt_bce": True, "symmetric_bce": False, "margin": False}
    result = audit.gradient_vectors(raw, unused, parameter)
    np.testing.assert_array_equal(result["control_sum"], [1, 3])
    np.testing.assert_array_equal(result["hypothetical_sum"], [1.2, 2.6])
    assert result["control_sum"].dtype == np.float64
    raw["ce"][0] = 1
    with pytest.raises(ValueError, match="unused"):
        audit.gradient_vectors(raw, unused, parameter)
    raw["ce"][:] = 0
    unused["ce"] = "true"
    with pytest.raises(ValueError, match="boolean"):
        audit.gradient_vectors(raw, unused, parameter)


def test_balanced_bce_excludes_subject_and_deduplicates_teacher_products():
    logits, labels, ids, attention = arrays()
    logits[:] = 0
    logits[0, 0] = 1000
    masks = audit.margin_masks(logits, labels, ids, attention)
    assert audit.balanced_symmetric_bce(logits, masks) == pytest.approx(np.log(2), abs=1e-15)
    attention[0, -1] = True
    labels[0, -1] = 1025
    duplicate_masks = audit.margin_masks(logits, labels, ids, attention)
    np.testing.assert_array_equal(duplicate_masks["positive"], masks["positive"])
    assert audit.balanced_symmetric_bce(logits, duplicate_masks) == audit.balanced_symmetric_bce(
        logits, masks
    )


def observation_fixture():
    record = {
        "subject": "PKM_subject",
        "dimension": "TYPE",
        "targets": ["PKM_target"],
        "input_ids": [1, 1024, 32, 64, 5, 1025, 2],
        "labels": [-100, -100, -100, -100, -100, 1025, 2],
    }
    batch = {
        "records": [copy.deepcopy(record) for _ in range(32)],
        "input_ids": np.tile(record["input_ids"], (32, 1)).astype(np.int64),
        "labels": np.tile(record["labels"], (32, 1)).astype(np.int64),
        "attention_mask": np.ones((32, 7), dtype=bool),
    }
    arrays = {k: v.copy() for k, v in batch.items() if k != "records"}
    arrays["shifted_labels"] = arrays["labels"][:, 1:].copy()
    arrays["symmetric_logits"] = np.zeros((32, 1025), dtype=np.float32)
    arrays["symmetric_logits"][:, 1] = 2
    arrays["prompt_logits"] = np.zeros((32, 1025), dtype=np.float32)
    masks = audit.margin_masks(
        arrays["symmetric_logits"], arrays["labels"], arrays["input_ids"], arrays["attention_mask"]
    )
    arrays["positive_mask"] = masks["positive"]
    arrays["negative_mask"] = masks["negative"]
    arrays["margin_hinges"] = masks["hinge"]
    arrays["token_ce_unreduced"] = (arrays["shifted_labels"] != -100).astype(np.float32)
    unused = {}
    for block, shape in (("embedding", (2049, 256)), ("projection", (256, 256))):
        arrays[f"parameter__{block}"] = np.ones(shape, dtype=np.float32)
        for objective in ("ce", "prompt_bce", "symmetric_bce", "margin"):
            arrays[f"gradient__{objective}__{block}"] = np.zeros(shape, dtype=np.float32)
            unused.setdefault(objective, {})[block] = block == "projection" and objective in (
                "ce",
                "prompt_bce",
            )
    for objective, scalar in (
        ("ce", 1),
        ("prompt_bce", 1),
        ("symmetric_bce", audit.balanced_symmetric_bce(arrays["symmetric_logits"], masks)),
        ("margin", 0),
    ):
        arrays[f"losses__{objective}"] = np.array(scalar, dtype=np.float32)
    arrays["losses__total"] = np.array(
        np.float32(2) + arrays["losses__symmetric_bce"], dtype=np.float32
    )
    report = {
        "complete": True,
        "state": "initial",
        "train_slice": [0, 32],
        "queries": audit.expected_queries(batch, 0),
        "dimension_ids": [32],
        "dimension_counts": {"TYPE": 32, "COLOR": 0},
        "unused": unused,
        "state_unchanged": True,
        "all_parameter_grad_fields_none": True,
        "state_before_sha256": "synthetic",
        "state_after_sha256": "synthetic",
        "margin_query_count": 32,
        "margin_active_count": 0,
        "total_loss_residual": 0.0,
        "losses": {
            name: float(arrays[f"losses__{name}"])
            for name in ("ce", "prompt_bce", "symmetric_bce", "margin", "total")
        },
        "array_manifest": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "groups": audit.group_statistics(arrays, unused, [32]),
        "radial": {
            "gradient_parameter_dot": 0.0,
            "active_extrema_mean": 0.0,
            "residual": 0.0,
            "relative_tolerance": 1e-4,
            "absolute_tolerance": 1e-5,
            "passed": True,
            "active_count": 0,
            "query_count": 32,
        },
    }
    return report, arrays, batch


def test_complete_saved_observation_and_defined_aggregates():
    report, arrays, batch = observation_fixture()
    result = audit.observation_evidence(report, arrays, "initial", 0, batch)
    aggregate = audit.aggregate_statistics([result["groups"]] * 3)
    undefined = aggregate["/projection/comparisons/weighted_margin_vs_control_sum/cosine/value"]
    assert undefined == {"mean_of_defined_batch_values": None, "defined_count": 0, "batch_count": 3}
    defined = aggregate["/projection/vectors/ce/l2_norm"]
    assert defined == {"mean_of_defined_batch_values": 0.0, "defined_count": 3, "batch_count": 3}
    assert result["groups"]["steering"]["parameter_shape"] == [1, 256]
    assert result["groups"]["products"]["parameter_shape"] == [1025, 256]


@pytest.mark.parametrize(
    "mutation",
    [
        "unused",
        "nonzero_unused",
        "mask",
        "hinge",
        "loss",
        "labels",
        "shape",
        "norm",
        "state",
        "radial",
    ],
)
def test_corrupt_saved_observation_rejected(mutation):
    report, arrays, batch = observation_fixture()
    if mutation == "unused":
        report["unused"]["ce"]["projection"] = False
    elif mutation == "nonzero_unused":
        arrays["gradient__ce__projection"][0, 0] = 1
    elif mutation == "mask":
        arrays["negative_mask"][0, 0] = True
    elif mutation == "hinge":
        arrays["margin_hinges"][0] = 1
    elif mutation == "loss":
        arrays["losses__symmetric_bce"] += np.float32(0.01)
        report["losses"]["symmetric_bce"] = float(arrays["losses__symmetric_bce"])
    elif mutation == "labels":
        arrays["shifted_labels"][0, 0] = 1026
    elif mutation == "shape":
        report["array_manifest"]["parameter__projection"]["shape"] = [65536]
    elif mutation == "norm":
        report["groups"]["projection"]["vectors"]["margin"]["l2_norm"] = 1.0
    elif mutation == "state":
        report["state_after_sha256"] = "changed"
    else:
        report["radial"]["residual"] = 1e-10
    with pytest.raises(ValueError):
        audit.observation_evidence(report, arrays, "initial", 0, batch)
