"""Experimental train-only symmetric affine head; no eager Torch import.

Callers authenticate and inject the frozen bilinear implementation and BCE loss.
These helpers neither load data/checkpoints nor choose evaluation endpoints.
"""

import math

PARAMETER_NAMES = (
    "symmetric_bilinear_residual",
    "symmetric_affine_linear",
    "symmetric_affine_bias",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(tensor, name):
    import torch

    _require(bool(torch.isfinite(tensor).all()), f"nonfinite {name}")


def attach(model):
    """Freeze an FP32 parent and attach fresh, zero A/u/b on its device."""
    import torch

    _require(isinstance(model, torch.nn.Module), "model must be a module")
    _require(not any(hasattr(model, n) for n in PARAMETER_NAMES), "already attached")
    dimension = getattr(getattr(model, "config", None), "dim", None)
    _require(type(dimension) is int and dimension > 0, "positive integer model dimension")
    parameters = list(model.parameters())
    _require(bool(parameters), "empty parent")
    device = parameters[0].device
    for tensor in model.state_dict().values():
        _require(tensor.device == device, "mixed parent devices")
        _require(
            not tensor.is_complex()
            and (not tensor.is_floating_point() or tensor.dtype == torch.float32),
            "FP32 parent",
        )
        _finite(tensor, "parent state")
    _require(all(p.grad is None for p in parameters), "parent has existing gradients")
    shapes = ((2, dimension, dimension), (2, dimension), (2,))
    additions = [
        torch.nn.Parameter(torch.zeros(s, device=device, dtype=torch.float32)) for s in shapes
    ]
    for parameter in parameters:
        parameter.requires_grad_(False)
    for name, parameter in zip(PARAMETER_NAMES, additions, strict=True):
        model.register_parameter(name, parameter)
    return model


def score(parent_scores, model, entities, subjects, dimension_tokens, scale, bilinear_fn):
    """FP32 [B,N] scores, using only [2,N] affine projections; fixed scale 16."""
    import torch

    _require(callable(bilinear_fn), "bilinear callable required")
    _require(type(scale) in (int, float) and math.isfinite(scale) and scale == 16, "scale 16")
    tensors = (parent_scores, entities, subjects, dimension_tokens)
    _require(all(isinstance(t, torch.Tensor) for t in tensors), "tensor inputs required")
    _require(entities.ndim == 2 and all(s > 0 for s in entities.shape), "entity shape")
    n, d = entities.shape
    _require(subjects.ndim == dimension_tokens.ndim == 1, "index shape")
    _require(subjects.shape == dimension_tokens.shape and subjects.numel() > 0, "query shape")
    _require(subjects.dtype == dimension_tokens.dtype == torch.long, "integer indices")
    _require(parent_scores.shape == (len(subjects), n), "parent score shape")
    parameters = tuple(getattr(model, name, None) for name in PARAMETER_NAMES)
    _require(all(isinstance(p, torch.nn.Parameter) for p in parameters), "attached parameters")
    for p, shape in zip(parameters, ((2, d, d), (2, d), (2,)), strict=True):
        _require(p.shape == shape, "parameter shape")
    floating = (parent_scores, entities, *parameters)
    _require(all(t.dtype == torch.float32 for t in floating), "FP32 inputs")
    _require(all(t.device == entities.device for t in (*tensors, *parameters)), "same device")
    _require(
        not parent_scores.requires_grad and not entities.requires_grad, "detached parent inputs"
    )
    _require(bool(((subjects >= 0) & (subjects < n)).all()), "subject index")
    _require(bool(((dimension_tokens == 32) | (dimension_tokens == 33)).all()), "dimension token")
    for tensor in floating:
        _finite(tensor, "score input")
    a, u, b = parameters
    with torch.autocast(device_type=entities.device.type, enabled=False):
        residual = bilinear_fn(a, entities, subjects, dimension_tokens, scale)
        _require(isinstance(residual, torch.Tensor), "bilinear tensor output")
        _require(residual.shape == parent_scores.shape, "bilinear output shape")
        _require(residual.dtype == torch.float32, "bilinear output dtype")
        _require(residual.device == entities.device, "bilinear output device")
        _finite(residual, "bilinear output")
        base = parent_scores + residual
        projections = (u @ entities.T) / scale
        dimensions = dimension_tokens - 32
        affine = projections[dimensions, subjects, None] + projections[dimensions]
        result = base + affine + b[dimensions, None]
    _finite(result, "scores")
    return result


def fit(
    model,
    parent_scores,
    entities,
    subjects,
    dimension_tokens,
    labels,
    loss_fn,
    optimizer,
    updates,
    receipt,
    bilinear_fn,
):
    """Fit fresh A/u/b only; preserve a JSON-safe partial receipt on any failure.

    ``optimizer_steps_attempted`` can exceed ``completed_updates`` when an
    optimizer mutates state and then fails. Callers persist the receipt and
    current model/optimizer on failure; this helper never rolls back or retries.
    """
    import torch

    _require(isinstance(receipt, dict) and not receipt, "fresh receipt required")
    receipt.update(complete=False, history=[], completed_updates=0, optimizer_steps_attempted=0)
    frozen_check = None
    try:
        receipt["phase"] = "preflight"
        _require(type(updates) is int and updates > 0, "positive integer updates")
        _require(callable(loss_fn), "loss callable required")
        _require(isinstance(labels, torch.Tensor), "label tensor")
        _require(labels.ndim == 2 and labels.shape[0] == len(subjects), "label shape")
        _require(
            labels.dtype == torch.long and labels.device == entities.device, "label dtype/device"
        )
        parameters = tuple(getattr(model, n) for n in PARAMETER_NAMES)
        selected = {id(p) for p in parameters}
        _require(
            {n for n, p in model.named_parameters() if p.requires_grad} == set(PARAMETER_NAMES),
            "only A/u/b trainable",
        )
        _require(isinstance(optimizer, torch.optim.AdamW), "AdamW required")
        actual = [p for group in optimizer.param_groups for p in group["params"]]
        _require(len(actual) == 3 and {id(p) for p in actual} == selected, "A/u/b optimizer only")
        _require(not optimizer.state, "fresh optimizer required")
        for group in optimizer.param_groups:
            expected = {"lr": 0.0003, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0}
            _require(all(group[k] == v for k, v in expected.items()), "fixed AdamW settings")
            _require(
                not group.get("fused") and not group.get("maximize"), "nonfused minimizing AdamW"
            )
            _require(
                not any(group.get(k) for k in ("amsgrad", "capturable", "differentiable"))
                and group.get("foreach") is None,
                "ordinary AdamW defaults required",
            )
        _require(all(int(torch.count_nonzero(p)) == 0 for p in parameters), "zero A/u/b required")
        frozen = {
            n: t.detach().clone() for n, t in model.state_dict().items() if n not in PARAMETER_NAMES
        }

        def frozen_check(check_state=True):
            if check_state:
                state = model.state_dict()
                _require(
                    set(state) == set(frozen) | set(PARAMETER_NAMES), "state inventory changed"
                )
                _require(
                    all(
                        torch.equal(
                            state[n].reshape(-1).view(torch.uint8), t.reshape(-1).view(torch.uint8)
                        )
                        for n, t in frozen.items()
                    ),
                    "frozen state changed",
                )
            _require(
                all(
                    p.grad is None for n, p in model.named_parameters() if n not in PARAMETER_NAMES
                ),
                "frozen gradient",
            )

        def compute_loss():
            scores = score(
                parent_scores, model, entities, subjects, dimension_tokens, 16, bilinear_fn
            )
            value = loss_fn(scores, labels, excluded_ids=subjects)
            _require(isinstance(value, torch.Tensor) and value.ndim == 0, "scalar tensor loss")
            _require(
                value.dtype == torch.float32 and value.device == entities.device,
                "loss dtype/device",
            )
            _finite(value, "loss")
            return value

        frozen_check()
        for update in range(1, updates + 1):
            receipt.update(failed_update=update, phase="forward")
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss()
            value = float(loss.detach().cpu())
            receipt["failed_observation"] = {"pre_update_loss": value}
            receipt["phase"] = "backward"
            loss.backward()
            for parameter in parameters:
                _require(parameter.grad is not None, "missing residual gradient")
                _finite(parameter.grad, "gradient")
            frozen_check(check_state=False)
            receipt["phase"] = "optimizer_step"
            receipt["optimizer_steps_attempted"] += 1
            optimizer.step()
            receipt["phase"] = "post_update"
            for parameter in parameters:
                _finite(parameter, "parameter")
            frozen_check(check_state=False)
            receipt["history"].append(
                {
                    "update": update,
                    "pre_update_loss": value,
                    "gradient_finite": True,
                    "parameters_finite": True,
                }
            )
            receipt["completed_updates"] = update
            receipt.pop("failed_observation", None)
        receipt["phase"] = "final_loss"
        with torch.no_grad():
            final = compute_loss()
        frozen_check()
        receipt.update(
            complete=True,
            phase="complete",
            initial_pre_update_loss=receipt["history"][0]["pre_update_loss"],
            final_post_update_loss=float(final.cpu()),
            frozen_state_exact=True,
        )
        receipt.pop("failed_update", None)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        if frozen_check is not None:
            try:
                frozen_check()
                receipt["frozen_state_exact"] = True
            except BaseException as state_exc:
                receipt["frozen_state_exact"] = False
                receipt["frozen_state_error"] = repr(state_exc)
        raise
