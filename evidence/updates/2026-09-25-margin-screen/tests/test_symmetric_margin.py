"""CPU arithmetic and gradient contracts for optional hardest-boundary supervision."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from plm.config import ModelConfig  # noqa: E402
from plm.model.architecture import build_model  # noqa: E402
from plm.model.interfaces import SYMMETRIC_MARGIN_OBJECTIVE, ModelOutput  # noqa: E402
from plm.model.layers import _prompt_set_loss, _symmetric_margin_loss  # noqa: E402


def _config(weight=0.0, margin=1.0):
    return ModelConfig(
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
        symmetric_margin=margin,
    )


def _batch():
    tokens = torch.tensor(
        [[1, 1024, 32, 64, 5, 1025, 1026, 2], [1, 1025, 33, 64, 5, 1026, 1027, 2]]
    )
    labels = tokens.clone()
    labels[:, :5] = -100
    return tokens, labels


def test_hinge_query_mean_includes_satisfied_rows_and_zero_boundary_gradient():
    scores = torch.tensor(
        [[99.0, 1.0, 3.0, 2.0, -2.0], [99.0, 4.0, 5.0, 0.0, -1.0], [99.0, 1.0, 2.0, 0.0, -1.0]],
        requires_grad=True,
    )
    labels = torch.tensor([[1025, 1026, 2]] * 3)
    loss, count, active = _symmetric_margin_loss(
        scores, labels, excluded_ids=torch.zeros(3, dtype=torch.long), margin=1.0
    )
    torch.testing.assert_close(loss, torch.tensor(2.0 / 3), rtol=0, atol=0)
    assert count == 3 and active == 1
    loss.backward()
    expected = torch.zeros_like(scores)
    expected[0, 1], expected[0, 3] = -1 / 3, 1 / 3
    torch.testing.assert_close(scores.grad, expected, rtol=0, atol=0)


def test_tied_extrema_split_positive_and_negative_gradients_evenly():
    scores = torch.tensor([[99.0, 0.0, 0.0, 1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([[1025, 1026, 2]])
    loss, count, active = _symmetric_margin_loss(
        scores, labels, excluded_ids=torch.tensor([0]), margin=1.0
    )
    assert loss.item() == 2.0 and count == active == 1
    loss.backward()
    torch.testing.assert_close(
        scores.grad, torch.tensor([[0.0, -0.5, -0.5, 0.5, 0.5]]), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "labels",
    [
        [[1025, 1026, 2]],
        [[-100, 1, 32, 64, 5, 1025, 1026, 1025, 2, 0]],
    ],
)
def test_controls_padding_and_duplicates_do_not_change_membership_masks(labels):
    scores = torch.tensor([[100.0, 1.0, 3.0, 2.0, -2.0]], requires_grad=True)
    result = _symmetric_margin_loss(
        scores, torch.tensor(labels), excluded_ids=torch.tensor([0]), margin=1.0
    )
    assert result[0].item() == 2.0 and result[1:] == (1, 1)
    result[0].backward()
    torch.testing.assert_close(
        scores.grad, torch.tensor([[0.0, -1.0, 0.0, 1.0, 0.0]]), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "labels,message",
    [
        ([[1024, 1025, 2]], "exclude the SAME subject"),
        ([[-100, 1, 32, 64, 5, 2, 0]], "positive and negative"),
        ([[1025, 1026, 1027, 1028, 2]], "positive and negative"),
    ],
)
def test_margin_and_bce_share_subject_and_empty_class_rejection(labels, message):
    scores, targets, subject = torch.zeros(1, 5), torch.tensor(labels), torch.tensor([0])
    with pytest.raises(ValueError, match=message):
        _prompt_set_loss(scores, targets, excluded_ids=subject)
    with pytest.raises(ValueError, match=message):
        _symmetric_margin_loss(scores, targets, excluded_ids=subject, margin=1.0)


def test_shared_mask_refactor_preserves_bce_arithmetic_and_gradients_exactly():
    scores = torch.tensor(
        [[1.25, -0.5, 0.75, 2.0, -3.0], [-1.25, 0.5, -0.75, -2.0, 3.0]], requires_grad=True
    )
    labels = torch.tensor([[1025, 1026, 1025, 2, -100], [1026, 1027, 2, 0, -100]])
    positive = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]])
    negative = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0]])
    # Frozen original operation order, with independently stated masks.
    expected = (
        (
            (torch.nn.functional.softplus(-scores) * positive).sum(-1) / positive.sum(-1)
            + (torch.nn.functional.softplus(scores) * negative).sum(-1) / negative.sum(-1)
        )
        * 0.5
    ).mean()
    actual = _prompt_set_loss(scores, labels, excluded_ids=torch.tensor([0, 1]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual, scores, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, scores)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_optional_loss_preserves_parameters_rng_scores_and_existing_objectives():
    torch.manual_seed(59)
    disabled = build_model(_config()).eval()
    disabled_rng = torch.get_rng_state().clone()
    torch.manual_seed(59)
    enabled = build_model(_config(0.75, 1.5)).eval()
    assert torch.equal(disabled_rng, torch.get_rng_state())
    assert disabled.state_dict().keys() == enabled.state_dict().keys()
    for key, value in disabled.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[key])
    tokens, labels = _batch()
    plain, extra = disabled(tokens, labels=labels), enabled(tokens, labels=labels)
    assert plain.symmetric_margin_loss is None
    assert plain.symmetric_margin_query_count == plain.symmetric_margin_active_count == 0
    assert extra.symmetric_margin_query_count == 2
    for name in (
        "logits",
        "task_loss",
        "first_target_loss",
        "prompt_set_logits",
        "prompt_set_loss",
        "symmetric_relation_logits",
        "symmetric_relation_loss",
    ):
        torch.testing.assert_close(getattr(plain, name), getattr(extra, name), rtol=0, atol=0)
    legacy_total = extra.task_loss + extra.prompt_set_loss + extra.symmetric_relation_loss
    torch.testing.assert_close(plain.loss, legacy_total, rtol=0, atol=0)
    torch.testing.assert_close(
        extra.loss, plain.loss + 0.75 * extra.symmetric_margin_loss, rtol=0, atol=0
    )
    plain_gradients = torch.autograd.grad(plain.loss, tuple(disabled.parameters()))
    legacy_gradients = torch.autograd.grad(legacy_total, tuple(enabled.parameters()))
    for actual, expected in zip(plain_gradients, legacy_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_weight_never_calls_margin_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled auxiliary executed")

    monkeypatch.setattr("plm.model.layers._symmetric_margin_loss", forbidden)
    model = build_model(_config(0.0, 7.0)).eval()
    tokens, labels = _batch()
    output = model(tokens, labels=labels)
    assert output.symmetric_margin_loss is None


def test_unlabeled_and_cached_outputs_are_unchanged():
    torch.manual_seed(67)
    disabled = build_model(_config()).eval()
    torch.manual_seed(67)
    enabled = build_model(_config(1.0)).eval()
    tokens, _ = _batch()
    with torch.inference_mode():
        plain, extra = disabled(tokens), enabled(tokens)
        torch.testing.assert_close(plain.logits, extra.logits, rtol=0, atol=0)
        torch.testing.assert_close(
            disabled.decode(tokens[:, :5]).logits,
            enabled.decode(tokens[:, :5]).logits,
            rtol=0,
            atol=0,
        )
    assert extra.loss is None and extra.symmetric_margin_loss is None
    assert extra.symmetric_margin_query_count == extra.symmetric_margin_active_count == 0


def test_attention_mask_excludes_product_valued_padding_and_future_inputs():
    model = build_model(_config(1.0)).eval()
    tokens, labels = _batch()
    padded_tokens = torch.cat((tokens, torch.tensor([[1028, 1028], [1024, 1024]])), dim=1)
    padded_labels = torch.cat((labels, padded_tokens[:, -2:]), dim=1)
    mask = torch.ones_like(padded_tokens, dtype=torch.bool)
    mask[:, -2:] = False
    short = model(tokens, labels=labels)
    padded = model(padded_tokens, labels=padded_labels, attention_mask=mask)
    changed = tokens.clone()
    changed[:, 5:] = 1028
    changed_output = model(changed, labels=labels)
    for output in (padded, changed_output):
        torch.testing.assert_close(
            output.symmetric_margin_loss, short.symmetric_margin_loss, rtol=0, atol=0
        )
        assert output.symmetric_margin_query_count == short.symmetric_margin_query_count == 2
        assert output.symmetric_margin_active_count == short.symmetric_margin_active_count


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_margin_reduction_is_fp32_under_cpu_bfloat16_autocast(dtype):
    scores = torch.tensor([[99.0, 1.0, 3.0, 2.0, -2.0]], dtype=dtype, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, count, active = _symmetric_margin_loss(
            scores, torch.tensor([[1025, 1026, 2]]), excluded_ids=torch.tensor([0]), margin=1.0
        )
    assert loss.dtype == torch.float32 and loss.item() == 2.0 and count == active == 1
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_margin_backpropagates_only_through_shared_relational_components():
    model = build_model(_config(1.0, 100.0)).train()
    tokens, labels = _batch()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(tokens, labels=labels)
    assert (
        output.symmetric_margin_loss.dtype
        == output.symmetric_relation_logits.dtype
        == torch.float32
    )
    assert output.symmetric_margin_active_count == 2
    score_gradient = torch.autograd.grad(
        output.symmetric_margin_loss, output.symmetric_relation_logits, retain_graph=True
    )[0]
    assert score_gradient[0, 0] == score_gradient[1, 1] == 0
    output.symmetric_margin_loss.backward()
    for parameter in (model.token_embedding.weight, model.symmetric_relation_projection.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model.token_embedding.weight.grad[32:34].abs().sum() > 0
    assert torch.count_nonzero(model.token_embedding.weight.grad[:32]) == 0
    assert model.prompt_set_projection.weight.grad is None
    assert all(parameter.grad is None for parameter in model.blocks.parameters())


def test_public_objective_and_default_output_are_torch_light():
    assert SYMMETRIC_MARGIN_OBJECTIVE == "hard-boundary-hinge-fp32-amin-amax-query-mean-v1"
    output = ModelOutput(logits=None)
    assert output.symmetric_margin_loss is None
    assert output.symmetric_margin_query_count == output.symmetric_margin_active_count == 0
    source = Path(__file__).parents[2] / "src"
    program = """
import sys
sys.path.insert(0, sys.argv[1])
from plm.model.interfaces import SYMMETRIC_MARGIN_OBJECTIVE
assert 'torch' not in sys.modules
assert SYMMETRIC_MARGIN_OBJECTIVE == 'hard-boundary-hinge-fp32-amin-amax-query-mean-v1'
"""
    subprocess.run(
        [sys.executable, "-I", "-c", program, str(source)], check=True, capture_output=True
    )
