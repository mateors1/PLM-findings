"""Synthetic CPU checks only; no real model or dataset access."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def case():
    torch = pytest.importorskip("torch")
    helper = module("symmetric_affine_residual")
    bilinear = module("refit_bilinear_residual")._residual
    torch.manual_seed(34)
    model = torch.nn.Module()
    model.config = SimpleNamespace(dim=3)
    model.parent = torch.nn.Linear(3, 3)
    model.register_buffer("counter", torch.tensor(7))
    helper.attach(model)
    e = torch.randn(5, 3)
    subjects = torch.tensor([0, 1, 2, 3])
    dims = torch.tensor([32, 32, 33, 33])
    parent = torch.randn(4, 5)
    labels = torch.tensor([[1025, 2], [1026, 2], [1027, 2], [1028, 2]])
    params = [getattr(model, n) for n in helper.PARAMETER_NAMES]
    optimizer = torch.optim.AdamW(params, lr=0.0003, weight_decay=0, fused=False)

    def loss(scores, labels, excluded_ids):
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(1, excluded_ids[:, None], False)
        targets = torch.zeros_like(scores)
        targets.scatter_(1, (labels[:, :1] - 1024), 1)
        return torch.nn.functional.binary_cross_entropy_with_logits(scores[mask], targets[mask])

    return SimpleNamespace(
        t=torch,
        h=helper,
        bilinear=bilinear,
        model=model,
        e=e,
        s=subjects,
        d=dims,
        parent=parent,
        labels=labels,
        params=params,
        opt=optimizer,
        loss=loss,
    )


def scoring(c, **overrides):
    arguments = {
        "parent_scores": c.parent,
        "model": c.model,
        "entities": c.e,
        "subjects": c.s,
        "dimension_tokens": c.d,
        "scale": 16,
        "bilinear_fn": c.bilinear,
    }
    return c.h.score(**(arguments | overrides))


def fitting(c, receipt, **overrides):
    arguments = {
        "model": c.model,
        "parent_scores": c.parent,
        "entities": c.e,
        "subjects": c.s,
        "dimension_tokens": c.d,
        "labels": c.labels,
        "loss_fn": c.loss,
        "optimizer": c.opt,
        "updates": 3,
        "receipt": receipt,
        "bilinear_fn": c.bilinear,
    }
    return c.h.fit(**(arguments | overrides))


def test_import_without_torch():
    script = (
        "import sys; sys.path.insert(0, 'scripts'); import symmetric_affine_residual; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_explicit_independent_formula(case):
    c = case
    with c.t.no_grad():
        for p in c.params:
            p.normal_()
    actual = scoring(c)
    expected = c.parent.double().clone()
    a, u, bias = (p.double() for p in c.params)
    e = c.e.double()
    for row, (s, token) in enumerate(zip(c.s.tolist(), c.d.tolist(), strict=True)):
        d = token - 32
        for i in range(len(e)):
            pair = sum(
                e[s, j] * (a[d, j, k] + a[d, k, j]) / 2 * e[i, k]
                for j in range(3)
                for k in range(3)
            )
            linear = sum(u[d, j] * (e[s, j] + e[i, j]) for j in range(3))
            expected[row, i] += (pair + linear) / 16 + bias[d]
    c.t.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)


def test_zero_reduction_and_autocast(case):
    c = case
    assert c.t.equal(scoring(c), c.parent)
    with c.t.no_grad():
        c.params[0].normal_()
    expected = c.parent + c.bilinear(c.params[0], c.e, c.s, c.d, 16)
    assert c.t.equal(scoring(c), expected)
    with c.t.autocast("cpu", dtype=c.t.bfloat16):
        assert c.t.equal(scoring(c), expected)


@pytest.mark.parametrize("token", [32, 33])
def test_symmetry(case, token):
    c = case
    c.e.clamp_(-1, 1)
    with c.t.no_grad():
        for p in c.params:
            p.uniform_(-1, 1)
    output = scoring(
        c,
        parent_scores=c.t.zeros(5, 5),
        subjects=c.t.arange(5),
        dimension_tokens=c.t.full((5,), token),
    )
    c.t.testing.assert_close(output, output.T, rtol=1e-5, atol=2e-5)


def test_dimension_isolation(case):
    c = case
    with c.t.no_grad():
        for p in c.params:
            p[0].fill_(1)
    output = scoring(c)
    assert c.t.equal(output[2:], c.parent[2:])
    assert not c.t.equal(output[:2], c.parent[:2])


@pytest.mark.parametrize(
    "field,kind",
    [
        ("entities", "double"),
        ("entities", "nan"),
        ("entities", "rank"),
        ("entities", "gradient"),
        ("parent_scores", "double"),
        ("parent_scores", "nan"),
        ("parent_scores", "rank"),
        ("parent_scores", "gradient"),
        ("subjects", "double"),
        ("subjects", "rank"),
        ("subjects", "negative"),
        ("subjects", "large"),
        ("dimension_tokens", "double"),
        ("dimension_tokens", "rank"),
        ("dimension_tokens", "negative"),
    ],
)
def test_invalid_inputs(case, field, kind):
    c = case
    x = {"entities": c.e, "parent_scores": c.parent, "subjects": c.s, "dimension_tokens": c.d}[
        field
    ].clone()
    if kind == "double":
        x = x.double()
    elif kind == "nan":
        x.flatten()[0] = float("nan")
    elif kind == "rank":
        x = x.unsqueeze(0)
    elif kind == "gradient":
        x.requires_grad_(True)
    elif kind == "negative":
        x[0] = -1
    else:
        x[0] = 999
    with pytest.raises(ValueError):
        scoring(c, **{field: x})


@pytest.mark.parametrize("value", [True, "16", 0, -16, 1, float("nan"), float("inf")])
def test_scale_contract(case, value):
    with pytest.raises(ValueError, match="scale"):
        scoring(case, scale=value)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_nonfinite_parameters(case, index):
    with case.t.no_grad():
        case.params[index].flatten()[0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        scoring(case)


def test_fit_and_exact_cpu_reload(case, tmp_path):
    c = case
    before = {k: v.clone() for k, v in c.model.state_dict().items()}
    initial = c.loss(scoring(c), c.labels, excluded_ids=c.s).item()
    receipt = {}
    fitting(c, receipt)
    assert receipt["complete"] and receipt["completed_updates"] == 3
    assert receipt["initial_pre_update_loss"] == initial
    assert (
        receipt["final_post_update_loss"] == c.loss(scoring(c), c.labels, excluded_ids=c.s).item()
    )
    assert [r["update"] for r in receipt["history"]] == [1, 2, 3]
    assert all(int(s["step"]) == 3 for s in c.opt.state.values())
    for n, value in c.model.state_dict().items():
        if n in c.h.PARAMETER_NAMES:
            assert not c.t.equal(value, before[n])
        else:
            assert c.t.equal(value, before[n])
    assert all(
        p.grad is None for n, p in c.model.named_parameters() if n not in c.h.PARAMETER_NAMES
    )
    path = tmp_path / "synthetic.pt"
    c.t.save({"model": c.model.state_dict(), "optimizer": c.opt.state_dict()}, path)
    state = c.t.load(path, weights_only=True)
    restored = copy.deepcopy(c.model)
    restored.load_state_dict(state["model"])
    opt = c.t.optim.AdamW(
        [getattr(restored, n) for n in c.h.PARAMETER_NAMES], lr=0.0003, weight_decay=0
    )
    opt.load_state_dict(state["optimizer"])
    assert c.t.equal(scoring(c, model=restored), scoring(c))
    for old, new in zip(c.opt.state.values(), opt.state.values(), strict=True):
        assert all(c.t.equal(old[k], new[k]) for k in old)
    assert opt.state_dict()["param_groups"] == c.opt.state_dict()["param_groups"]
    json.dumps(receipt, allow_nan=False)


def test_loss_receives_self_mask_and_has_zero_self_gradient(case):
    c = case

    def callback(scores, labels, excluded_ids):
        assert c.t.equal(excluded_ids, c.s)
        assert c.t.equal(labels, c.labels)
        value = c.loss(scores, labels, excluded_ids)
        if scores.requires_grad:
            grad = c.t.autograd.grad(value, scores, retain_graph=True)[0]
            assert c.t.count_nonzero(grad[c.t.arange(4), c.s]) == 0
        return value

    fitting(c, {}, loss_fn=callback)


@pytest.mark.parametrize("at", [1, 2, 4])
def test_partial_failure_receipt(case, at):
    c = case
    calls = 0

    def fail(scores, labels, excluded_ids):
        nonlocal calls
        calls += 1
        if calls == at:
            raise RuntimeError("synthetic callback failure")
        return c.loss(scores, labels, excluded_ids)

    receipt = {}
    with pytest.raises(RuntimeError, match="synthetic"):
        fitting(c, receipt, loss_fn=fail)
    assert not receipt["complete"] and receipt["completed_updates"] == at - 1
    assert receipt["optimizer_steps_attempted"] == at - 1
    assert len(receipt["history"]) == at - 1 and "error" in receipt
    assert receipt["phase"] == ("final_loss" if at == 4 else "forward")
    json.dumps(receipt, allow_nan=False)


def test_post_step_failure_is_not_rolled_back(case):
    c = case
    original = c.opt.step

    def failing_step(*args, **kwargs):
        original(*args, **kwargs)
        with c.t.no_grad():
            c.params[1][0, 0] = float("nan")

    c.opt.step = failing_step
    receipt = {}
    with pytest.raises(ValueError, match="nonfinite parameter"):
        fitting(c, receipt)
    assert receipt["optimizer_steps_attempted"] == 1 and receipt["completed_updates"] == 0
    assert receipt["phase"] == "post_update" and receipt["failed_update"] == 1
    assert c.t.isnan(c.params[1][0, 0])
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_update_rejection_receipted(case, bad):
    receipt = {}
    with pytest.raises(ValueError):
        fitting(case, receipt, updates=bad)
    assert not receipt["complete"] and receipt["phase"] == "preflight"


def test_attach_preflight_does_not_mutate(case):
    c = case
    before = {n: p.requires_grad for n, p in c.model.named_parameters()}
    with pytest.raises(ValueError, match="already attached"):
        c.h.attach(c.model)
    assert before == {n: p.requires_grad for n, p in c.model.named_parameters()}


def test_optimizer_selection_rejection(case):
    c = case
    c.opt = c.t.optim.AdamW(c.params[:2], lr=0.0003, weight_decay=0)
    with pytest.raises(ValueError, match="optimizer only"):
        fitting(c, {})


def test_frozen_mutation_rejected(case):
    c = case

    def mutate(scores, labels, excluded_ids):
        with c.t.no_grad():
            c.model.parent.weight[0, 0] += 1
        return c.loss(scores, labels, excluded_ids)

    receipt = {}
    with pytest.raises(ValueError, match="frozen state"):
        fitting(c, receipt, loss_fn=mutate)
    assert receipt["completed_updates"] == 3
    assert receipt["frozen_state_exact"] is False


@pytest.mark.parametrize(
    "name",
    [
        "symmetric_bilinear_residual",
        "symmetric_affine_linear",
        "symmetric_affine_bias",
    ],
)
def test_parameter_shape_rejection(case, name):
    setattr(case.model, name, case.t.nn.Parameter(case.t.zeros(1)))
    with pytest.raises(ValueError, match="parameter shape"):
        scoring(case)


def test_same_device_rejection_without_gpu(case):
    with pytest.raises(ValueError, match="same device"):
        scoring(case, parent_scores=case.t.empty(4, 5, device="meta"))


@pytest.mark.parametrize("bad", [None, [0], 32])
def test_non_tensor_rejection(case, bad):
    with pytest.raises(ValueError, match="tensor inputs"):
        scoring(case, subjects=bad)


@pytest.mark.parametrize("which", ["scores", "loss", "gradient"])
def test_nonfinite_fit_failure(case, which):
    c = case
    if which == "scores":
        c.parent[0, 0] = float("nan")
    if which == "gradient":
        c.params[2].register_hook(lambda grad: grad * float("nan"))

    def loss(scores, labels, excluded_ids):
        value = c.loss(scores, labels, excluded_ids)
        return value * float("nan") if which == "loss" else value

    receipt = {}
    with pytest.raises(ValueError, match="nonfinite"):
        fitting(c, receipt, loss_fn=loss)
    assert receipt["optimizer_steps_attempted"] == 0
    assert receipt["frozen_state_exact"]
    json.dumps(receipt, allow_nan=False)


def test_existing_parent_gradient_rejected(case):
    case.model.parent.weight.grad = case.t.ones_like(case.model.parent.weight)
    with pytest.raises(ValueError, match="frozen gradient"):
        fitting(case, {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", 0.001),
        ("weight_decay", 0.01),
        ("maximize", True),
        ("amsgrad", True),
        ("capturable", True),
        ("differentiable", True),
        ("foreach", True),
        ("foreach", False),
    ],
)
def test_optimizer_settings(case, field, value):
    case.opt.param_groups[0][field] = value
    with pytest.raises(ValueError):
        fitting(case, {})


def test_fresh_optimizer_required(case):
    fitting(case, {})
    with pytest.raises(ValueError, match="fresh optimizer"):
        fitting(case, {})


def test_full_dimension_bounded_symmetry(case):
    c = case
    model = c.t.nn.Module()
    model.config = SimpleNamespace(dim=256)
    model.parent = c.t.nn.Linear(1, 1)
    c.h.attach(model)
    entities = c.t.empty(7, 256).uniform_(-1, 1)
    with c.t.no_grad():
        for name in c.h.PARAMETER_NAMES:
            getattr(model, name).uniform_(-1, 1)
    for token in (32, 33):
        scores = c.h.score(
            c.t.zeros(7, 7), model, entities, c.t.arange(7), c.t.full((7,), token), 16, c.bilinear
        )
        c.t.testing.assert_close(scores, scores.T, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("bad", ["dtype", "complex", "nonfinite", "gradient", "dimension"])
def test_attach_invalid_parent_is_unchanged(case, bad):
    c = case
    model = c.t.nn.Module()
    model.config = SimpleNamespace(dim=3)
    model.parent = c.t.nn.Linear(3, 3)
    if bad == "dtype":
        model.double()
    elif bad == "complex":
        model.register_buffer("complex_state", c.t.tensor([1 + 2j]))
    elif bad == "nonfinite":
        with c.t.no_grad():
            model.parent.weight[0, 0] = float("inf")
    elif bad == "gradient":
        model.parent.weight.grad = c.t.ones_like(model.parent.weight)
    else:
        model.config.dim = True
    before = set(model.state_dict())
    with pytest.raises(ValueError):
        c.h.attach(model)
    assert set(model.state_dict()) == before
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("index", [0, 1, 2])
def test_initial_parameters_must_be_zero(case, index):
    with case.t.no_grad():
        case.params[index].flatten()[0] = 1
    with pytest.raises(ValueError, match="zero A/u/b"):
        fitting(case, {})
