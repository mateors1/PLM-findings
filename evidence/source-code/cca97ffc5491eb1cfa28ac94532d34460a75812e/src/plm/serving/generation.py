"""Bounded autoregressive decoding for the native reference model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plm.protocol.tokenizer import CLASS_ENTITY, Vocabulary
from plm.serving.guidance import (
    apply_first_target_guidance,
    guidance_decoding_suffix,
    validate_guidance_alpha,
)
from plm.serving.parser import ProtocolParseError, constrain_logits, parse_generated, prompt_ids

__all__ = ["GenerationResult", "generate_response"]


def _prepare_model_for_inference(model: Any, device: str | Any) -> Any:
    """Move only when needed; compiled parameters cannot be swapped on repeat calls."""
    import torch

    parameter_fn = getattr(model, "parameters", None)
    parameter = next(iter(parameter_fn()), None) if callable(parameter_fn) else None
    target = torch.device(device)
    target_index = (
        torch.cuda.current_device()
        if target.type == "cuda" and target.index is None
        else target.index
    )
    if (
        parameter is None
        or parameter.device.type != target.type
        or (target_index is not None and parameter.device.index != target_index)
    ):
        model = model.to(device)
    return model.eval()


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    targets: tuple[str, ...]
    terminated: bool
    protocol_valid: bool
    decoding: str
    error: str | None = None


def generate_response(
    model: Any,
    vocabulary: Vocabulary,
    subject: str,
    dimension: str,
    *,
    max_new_tokens: int,
    device: str | Any = "cpu",
    constrained: bool = True,
    use_cache: bool = False,
    prevent_repeated_targets: bool = False,
    first_target_guidance_alpha: float = 0.0,
    symmetric_set_reranking: bool = False,
) -> GenerationResult:
    """Greedily generate until EOS or the length bound, without oracle facts.

    The default protocol mask restricts token classes only. The opt-in target
    uniqueness policy masks entities already emitted in this answer; it does
    not exclude the prompt subject, consult relations, or manufacture EOS.
    Positive first-target guidance biases only the cached prompt's first
    product decision using the model's learned symmetric relation head.
    """
    import torch

    if symmetric_set_reranking:
        if not (constrained and use_cache and prevent_repeated_targets):
            raise ValueError("symmetric set reranking requires constrained, unique cached decoding")
        from plm.serving.set_reranking import generate_reranked_responses

        return generate_reranked_responses(
            model,
            vocabulary,
            [(subject, dimension)],
            max_new_tokens=max_new_tokens,
            device=device,
            first_target_guidance_alpha=first_target_guidance_alpha,
        )[0].selected

    if isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if prevent_repeated_targets and not constrained:
        raise ValueError("prevent_repeated_targets requires constrained decoding")
    validate_guidance_alpha(first_target_guidance_alpha)
    if first_target_guidance_alpha and not (constrained and use_cache):
        raise ValueError("first target guidance requires constrained decoding and use_cache")
    prefix = list(prompt_ids(subject, dimension, vocabulary))
    if vocabulary.class_of(subject) != CLASS_ENTITY or dimension not in {"TYPE", "COLOR"}:
        raise ValueError("generation requires a product subject and TYPE or COLOR")
    context_limit = getattr(getattr(model, "config", None), "max_seq_len", None)
    if context_limit is not None and len(prefix) + max_new_tokens > context_limit:
        raise ValueError("prompt plus max_new_tokens exceeds model context")
    model = _prepare_model_for_inference(model, device)
    if use_cache and not callable(getattr(model, "decode", None)):
        raise ValueError("cached generation requires a model.decode implementation")
    terminated = False
    cache = None  # Always fresh; never carry another request's prefix state.
    seen_targets: set[int] = set()
    with torch.inference_mode():
        for step in range(max_new_tokens):
            chunk = prefix if cache is None else prefix[-1:]
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            if use_cache:
                output = model.decode(input_ids, cache=cache)
                if step == 0 and first_target_guidance_alpha:
                    output = apply_first_target_guidance(
                        model, input_ids, output, first_target_guidance_alpha
                    )
                cache = output.cache
            else:
                output = model(input_ids)
            logits = output.logits[0, -1]
            if logits.shape[-1] != len(vocabulary) or not torch.isfinite(logits).all():
                raise ValueError("generation logits must be finite and match vocabulary")
            if constrained:
                logits = constrain_logits(logits, prefix, vocabulary)
            if prevent_repeated_targets and seen_targets:
                # The protocol mask above owns a fresh vector. Never mask EOS
                # or the prompt subject merely because it occurs in the prompt.
                logits[list(seen_targets)] = -float("inf")
            next_id = int(logits.argmax().item())
            prefix.append(next_id)
            if prevent_repeated_targets and vocabulary.class_of(next_id) == CLASS_ENTITY:
                seen_targets.add(next_id)
            if next_id == vocabulary.eos_id:
                terminated = True
                break
    return _generation_result(
        prefix,
        vocabulary,
        terminated,
        ("greedy-protocol-mask-v1" if constrained else "greedy-unconstrained-v1")
        + ("+unique-v1" if prevent_repeated_targets else "")
        + ("+kv-v1" if use_cache else "")
        + guidance_decoding_suffix(first_target_guidance_alpha),
    )


def _generation_result(
    prefix: list[int], vocabulary: Vocabulary, terminated: bool, decoding: str
) -> GenerationResult:
    """Apply one shared parser/result contract to serial and batched decoding."""
    error = None
    try:
        parse_generated(prefix, vocabulary)
        valid = True
    except (ProtocolParseError, KeyError, ValueError) as exc:
        valid = False
        error = str(exc)
    targets = tuple(
        vocabulary.token_of(token_id)
        for token_id in prefix[5:]
        if vocabulary.class_of(token_id) == CLASS_ENTITY
    )
    return GenerationResult(
        tuple(prefix),
        targets,
        terminated,
        valid,
        decoding,
        error,
    )
