"""Small CPU fixtures for a fixed diagnostic; never load real model weights."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gradient_diagnosis", ROOT / "scripts/diagnose_margin_gradients.py"
)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_fp64_norm_dot_and_weighting():
    a = np.array([3, 4], dtype=np.float32)
    b = np.array([-4, 3], dtype=np.float32)
    assert diagnostic._norm(a) == 5
    assert diagnostic._dot(a, b) == 0
    assert diagnostic._comparison(a, b) == {
        "cosine": {"value": 0.0, "reason": None},
        "norm_ratio": {"value": 1.0, "reason": None},
    }
    assert diagnostic._norm(a.astype(np.float64) * 0.1) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("a", "b", "reason"),
    [([0], [0], "both_zero_norm"), ([0], [1], "left_zero_norm"), ([1], [0], "right_zero_norm")],
)
def test_zero_norm_comparisons_are_explicitly_undefined(a, b, reason):
    assert diagnostic._comparison(a, b) == {
        "cosine": {"value": None, "reason": reason},
        "norm_ratio": {"value": None, "reason": reason},
    }


def arrays_and_unused():
    arrays = {
        "parameter__embedding": np.ones((1027, 2), dtype=np.float32),
        "parameter__projection": np.ones((2, 2), dtype=np.float32),
    }
    unused = {}
    for k in diagnostic._OBJECTIVES:
        unused[k] = {"embedding": False, "projection": k in ("ce", "prompt_bce")}
        for block in diagnostic._BLOCKS:
            arrays[f"gradient__{k}__{block}"] = np.full_like(
                arrays[f"parameter__{block}"], 0 if unused[k][block] else 2
            )
    return arrays, unused


def test_groups_are_subsets_and_unused_projection_is_distinct_from_zero():
    arrays, unused = arrays_and_unused()
    groups = diagnostic._analysis(arrays, unused, [32, 33])
    assert groups["products"]["parameter_shape"] == [3, 2]
    assert groups["steering"]["parameter_shape"] == [2, 2]
    projection = groups["projection"]["vectors"]
    assert projection["ce"] == {"l2_norm": 0.0, "parameter_dot": 0.0, "unused": True}
    assert projection["control_sum"] == projection["symmetric_bce"]
    assert projection["weighted_margin"]["l2_norm"] == pytest.approx(
        0.1 * projection["margin"]["l2_norm"]
    )
    unused["ce"]["projection"] = False
    with pytest.raises(ValueError, match="unused by CE"):
        diagnostic._analysis(arrays, unused, [32, 33])


def test_unused_zero_fill_cannot_contain_a_fabricated_derivative():
    arrays, unused = arrays_and_unused()
    arrays["gradient__ce__projection"][0, 0] = 1.0
    with pytest.raises(ValueError, match="zero fill"):
        diagnostic._analysis(arrays, unused, [32])


def test_aggregate_means_keep_defined_counts_and_do_not_average_vectors():
    records = [
        {"groups": {"x": {"cosine": {"value": v, "reason": None}, "l2_norm": n}}}
        for v, n in ((1.0, 1.0), (None, 2.0), (-0.5, 3.0))
    ]
    summary = diagnostic._aggregate(records)
    assert summary["/x/cosine/value"] == {
        "mean_of_defined_batch_values": 0.25,
        "defined_count": 2,
        "batch_count": 3,
    }
    assert summary["/x/l2_norm"]["mean_of_defined_batch_values"] == 2.0


def test_radial_derivative_active_inactive_boundary_and_tied_extrema():
    torch = pytest.importorskip("torch")
    from plm.model.layers import _prompt_set_masks, _symmetric_margin_loss

    parameter = torch.tensor([[1.0]], requires_grad=True)
    base = torch.tensor(
        [
            [99.0, 1.0, 3.0, 2.0, -2.0],
            [99.0, 4.0, 5.0, 0.0, -1.0],
            [99.0, 1.0, 2.0, 0.0, -1.0],
            [99.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    logits = parameter[0, 0] * base
    labels = torch.tensor([[1025, 1026, 1025, 2, -100]] * 4)
    excluded = torch.zeros(4, dtype=torch.long)
    loss, _, active = _symmetric_margin_loss(logits, labels, excluded_ids=excluded, margin=1.0)
    positive, negative, _, _ = _prompt_set_masks(logits, labels, excluded_ids=excluded)
    gradient = torch.autograd.grad(loss, (parameter,))[0]
    hinges = torch.relu(
        1
        + logits.masked_fill(~negative.bool(), -torch.inf).amax(-1)
        - logits.masked_fill(~positive.bool(), torch.inf).amin(-1)
    )
    arrays = {
        "parameter__projection": parameter.detach().numpy(),
        "gradient__margin__projection": gradient.numpy(),
        "symmetric_logits": logits.detach().numpy(),
        "positive_mask": positive.bool().numpy(),
        "negative_mask": negative.bool().numpy(),
        "margin_hinges": hinges.detach().numpy(),
    }
    result = diagnostic._radial(arrays)
    assert result["passed"] is True and result["active_count"] == active == 2
    assert result["gradient_parameter_dot"] == result["active_extrema_mean"] == 0.5
    assert not arrays["positive_mask"][:, 0].any() and not arrays["negative_mask"][:, 0].any()
    assert parameter.grad is None


def small_model(weight):
    pytest.importorskip("torch")
    from plm.config import ModelConfig
    from plm.model.architecture import build_model

    return build_model(
        ModelConfig(
            vocab_size=1029,
            max_seq_len=16,
            dim=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            ffn_multiple_of=8,
            dtype="fp32",
            prompt_set_loss_weight=1.0,
            symmetric_relation_loss_weight=1.0,
            symmetric_margin_loss_weight=weight,
        )
    ).eval()


def records():
    from plm.corpus.compiler import CorpusRecord

    a = (1, 1024, 32, 64, 5, 1025, 1026, 2)
    b = (1, 1025, 33, 64, 5, 1026, 2)
    return [
        CorpusRecord("PKM_A", "TYPE", ("PKM_B", "PKM_C"), a, (-100,) * 5 + a[5:]),
        CorpusRecord("PKM_B", "COLOR", ("PKM_C",), b, (-100,) * 5 + b[5:]),
    ]


@pytest.mark.parametrize("weight", [0.0, 0.1])
def test_complete_synthetic_observation_masks_ownership_and_no_updates(tmp_path, weight):
    model = small_model(weight)
    before = diagnostic._state_identity(model)
    report = diagnostic._measure(model, records(), 0, "synthetic", 0, tmp_path, [], device="cpu")
    assert report["complete"] and report["state_unchanged"]
    assert report["all_parameter_grad_fields_none"] and report["radial"]["passed"]
    assert diagnostic._state_identity(model) == before
    with np.load(tmp_path / "synthetic-batch-0.npz", allow_pickle=False) as arrays:
        assert arrays["attention_mask"][1, -1] == np.False_
        assert arrays["shifted_labels"][1, -1] == -100
        assert not arrays["positive_mask"][0, 0] and not arrays["negative_mask"][0, 0]
        assert arrays["losses__margin"].dtype == np.float32
        expected = sum(
            float(arrays["losses__" + k]) for k in diagnostic._OBJECTIVES[:3]
        ) + weight * float(arrays["losses__margin"])
        assert math.isclose(float(arrays["losses__total"]), expected, rel_tol=1e-5, abs_tol=1e-6)
        assert np.isfinite(arrays["gradient__margin__embedding"]).all()


def test_partial_raw_arrays_survive_late_failure(tmp_path, monkeypatch):
    model = small_model(0.0)

    def fail(*args):
        raise ValueError("injected analysis failure")

    monkeypatch.setattr(diagnostic, "_analysis", fail)
    with pytest.raises(ValueError, match="injected analysis failure"):
        diagnostic._measure(model, records(), 0, "synthetic", 0, tmp_path, [], device="cpu")
    saved = json.loads((tmp_path / "synthetic-batch-0.json").read_text())
    assert saved["complete"] is False and saved["state_unchanged"] is True
    assert "injected analysis failure" in saved["error"]
    with np.load(tmp_path / "synthetic-batch-0.npz", allow_pickle=False) as arrays:
        assert "gradient__margin__projection" in arrays
    with pytest.raises(ValueError, match="immutable"):
        diagnostic._refuse(tmp_path / "summary.json")


def test_separate_seeded_builds_have_exact_state_identity():
    torch = pytest.importorskip("torch")
    torch.manual_seed(1729)
    first = small_model(0.0)
    torch.manual_seed(1729)
    second = small_model(0.1)
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    assert diagnostic._state_identity(first) == diagnostic._state_identity(second)
    with torch.no_grad():
        second.token_embedding.weight[0, 0] += 1
    assert diagnostic._state_identity(first) != diagnostic._state_identity(second)


def test_input_identity_and_output_refusal(tmp_path):
    path = tmp_path / "plan.md"
    path.write_text("declared", encoding="utf-8")
    digest = diagnostic._sha(path)
    inputs = {}
    diagnostic._bound(path, digest, inputs)
    path.write_text("edited", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        diagnostic._bound(path, digest, inputs)
    (tmp_path / "summary.script.py").write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        diagnostic._refuse(tmp_path / "summary.json")


def test_standalone_import_is_torch_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy,sys; "
                "runpy.run_path('scripts/diagnose_margin_gradients.py',run_name='import_test'); "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
