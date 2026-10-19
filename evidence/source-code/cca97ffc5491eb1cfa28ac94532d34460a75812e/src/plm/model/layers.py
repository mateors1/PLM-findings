"""Dense reference and optional sparse-expert decoder components.

This module is imported lazily by :func:`plm.model.build_model`; the core
Phase-A/B environment can import ``plm.model`` without torch installed.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from plm.config import ModelConfig
from plm.model.interfaces import DecodeOutput, KVCache, ModelInput, ModelOutput
from plm.protocol.tokenizer import ENTITY_BASE

__all__ = [
    "CausalSelfAttention",
    "DecoderBlock",
    "PLMDecoder",
    "RMSNorm",
    "RoPE",
    "SparseMoE",
    "SwiGLU",
]


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by the reference decoder."""

    def __init__(self, dim: int, eps: float = 1.0e-5) -> None:
        if dim < 1:
            raise ValueError("RMSNorm dim must be positive")
        if eps <= 0.0:
            raise ValueError("RMSNorm eps must be positive")
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.eps).to(dtype=hidden.dtype)
        return normalized * self.weight.to(dtype=hidden.dtype)


class RoPE(nn.Module):
    """Rotary position embeddings with a cached reference frequency table."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        if head_dim < 2 or head_dim % 2:
            raise ValueError("RoPE requires a positive even head_dim")
        if max_seq_len < 1:
            raise ValueError("RoPE max_seq_len must be positive")
        if theta <= 0.0:
            raise ValueError("RoPE theta must be positive")
        super().__init__()
        inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inverse)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(values: Tensor) -> Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        *,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError(
                "RoPE query and key must have shape (batch, heads, sequence, head_dim)"
            )
        cos_table = cast(Tensor, self.cos)
        sin_table = cast(Tensor, self.sin)
        rotary_dim = cos_table.shape[-1] * 2
        if query.shape[-1] != rotary_dim or key.shape[-1] != rotary_dim:
            raise ValueError("RoPE head dimension does not match the cached table")
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("RoPE query and key sequence lengths must match")
        if position_offset < 0:
            raise ValueError("RoPE position_offset must be non-negative")
        length = query.shape[-2]
        if position_offset + length > cos_table.shape[0]:
            raise ValueError("RoPE sequence exceeds the cached max_seq_len")
        cos = cos_table[position_offset : position_offset + length]
        sin = sin_table[position_offset : position_offset + length]
        # _rotate_half pairs the first and second halves, so both coordinates
        # in each pair must share one angle (not adjacent repeated angles).
        cos = torch.cat((cos, cos), dim=-1).to(dtype=query.dtype, device=query.device)
        sin = torch.cat((sin, sin), dim=-1).to(dtype=query.dtype, device=query.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (
            query * cos + self._rotate_half(query) * sin,
            key * cos + self._rotate_half(key) * sin,
        )


class SwiGLU(nn.Module):
    """Gated feed-forward network used in the minimal dense model."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        if dim < 1 or hidden_dim < 1:
            raise ValueError("SwiGLU dimensions must be positive")
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)))


class SparseMoE(nn.Module):
    """Inspectable top-k SwiGLU experts, without token dropping or fused kernels.

    Selected experts use their full-softmax probabilities (not renormalized),
    retaining a task-loss gradient to the router even for top-1 routing.
    Padding is excluded from routing and the load-balancing auxiliary loss.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.moe.num_experts
        self.top_k = config.moe.experts_per_token
        self.router = nn.Linear(config.dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(config.dim, config.ffn_hidden_dim) for _ in range(self.num_experts)
        )
        self.shared = nn.ModuleList(
            SwiGLU(config.dim, config.ffn_hidden_dim) for _ in range(config.moe.shared_experts)
        )

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        flat = hidden.reshape(-1, hidden.shape[-1])
        if attention_mask is None:
            active_indices = torch.arange(flat.shape[0], device=flat.device)
        else:
            active_indices = (
                attention_mask.reshape(-1)
                .to(device=hidden.device, dtype=torch.bool)
                .nonzero()
                .flatten()
            )
        active = flat.index_select(0, active_indices)
        if not active.shape[0]:
            return (
                torch.zeros_like(hidden),
                hidden.new_zeros(()),
                hidden.new_zeros(self.num_experts),
            )
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            probabilities = F.linear(active.float(), self.router.weight.float()).softmax(dim=-1)
        weights, choices = probabilities.topk(self.top_k, dim=-1)
        combined = torch.zeros_like(active)
        for index, expert in enumerate(self.experts):
            positions, slots = torch.where(choices == index)
            if positions.numel():
                expert_output = expert(active.index_select(0, positions))
                weighted = expert_output * weights[positions, slots, None].to(expert_output.dtype)
                combined = combined.index_add(0, positions, weighted.to(combined.dtype))
        for expert in self.shared:
            combined = combined + expert(active)
        # Fraction of selected routes per expert; sums to one for top-k too.
        load = F.one_hot(choices, self.num_experts).float().mean(dim=(0, 1))
        auxiliary = self.num_experts * (load.detach() * probabilities.mean(dim=0)).sum()
        output = torch.zeros_like(flat).index_copy(0, active_indices, combined)
        return output.reshape_as(hidden), auxiliary, load


class CausalSelfAttention(nn.Module):
    """Correct reference MHA/GQA attention backed by PyTorch SDPA."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.resolved_head_dim
        self.kv_groups = config.kv_groups
        self.dropout = config.attention.dropout
        self.sliding_window = config.attention.sliding_window
        projection_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.dim, projection_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(projection_dim, config.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps) if config.attention.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps) if config.attention.qk_norm else None
        self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope.theta)
        self.logit_soft_cap = config.attention.logit_soft_cap

    def _causal_mask(self, length: int, device: torch.device, position_offset: int = 0) -> Tensor:
        keys = torch.arange(position_offset + length, device=device)
        queries = torch.arange(position_offset, position_offset + length, device=device)
        mask = keys[None, :] <= queries[:, None]
        if self.sliding_window is not None:
            distance = keys[None, :] - queries[:, None]
            mask = mask & (distance >= -self.sliding_window + 1)
        return mask

    @staticmethod
    def _validate_attention_mask(attention_mask: Tensor, batch: int, length: int) -> Tensor:
        if attention_mask.ndim != 2 or attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must have shape (batch, sequence)")
        return attention_mask.to(dtype=torch.bool)

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        attended, _ = self._attend(hidden, attention_mask)
        return attended

    def decode(
        self, hidden: Tensor, past: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend an unpadded inference chunk using cached, already rotated keys."""
        if self.training or torch.is_grad_enabled():
            raise ValueError("KV decoding requires eval mode and disabled autograd")
        return self._attend(hidden, None, past)

    def _attend(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
        past: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = (
            self.k_proj(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )
        value = (
            self.v_proj(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        offset = 0
        if past is not None:
            if len(past) != 2:
                raise ValueError("KV cache needs a key/value pair")
            for cached in past:
                if (
                    cached.ndim != 4
                    or cached.shape[:2] != (batch, self.n_kv_heads)
                    or cached.shape[-1] != self.head_dim
                    or cached.device != key.device
                    or cached.dtype != key.dtype
                ):
                    raise ValueError("KV cache shape/device/dtype does not match this request")
            if past[0].shape != past[1].shape:
                raise ValueError("cached keys and values must have the same shape")
            offset = past[0].shape[-2]
        query, key = self.rope(query, key, position_offset=offset)
        if past is not None:
            key = torch.cat((past[0], key), dim=-2)
            value = torch.cat((past[1], value), dim=-2)
        present = (key, value)  # Before GQA repetition: store only KV heads.
        if self.kv_groups > 1:
            key = key.repeat_interleave(self.kv_groups, dim=1)
            value = value.repeat_interleave(self.kv_groups, dim=1)
        mask: Tensor | None = None
        if attention_mask is not None:
            key_valid = self._validate_attention_mask(attention_mask, batch, length).to(
                device=hidden.device
            )
            padding_mask = key_valid[:, None, None, :]
            mask = self._causal_mask(length, hidden.device)[None, None, :, :] & padding_mask
        elif self.sliding_window is not None or (offset and length > 1):
            mask = self._causal_mask(length, hidden.device, offset)[None, None, :, :]

        dropout = self.dropout if self.training else 0.0
        use_sdpa = self.logit_soft_cap is None and not torch.are_deterministic_algorithms_enabled()
        if use_sdpa:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout,
                # A cached single query is at the END of the key sequence.
                # There are no future keys; a top-left causal mask is incorrect.
                is_causal=mask is None and self.sliding_window is None and past is None,
            )
        else:
            # SDPA does not expose logits, so use the reference math path when
            # logit soft-capping is requested. The cap belongs before softmax,
            # never on the attended values. Deterministic mode also uses this
            # path because some CUDA SDPA kernels are nondeterministic.
            if mask is None:
                mask = self._causal_mask(length, hidden.device, offset)[None, None, :, :]
            scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5
            if self.logit_soft_cap is not None:
                scores = self.logit_soft_cap * torch.tanh(scores / self.logit_soft_cap)
            scores = scores.masked_fill(~mask, -float("inf"))
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=dropout, training=self.training)
            attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return cast(Tensor, self.out_proj(attended)), present


class DecoderBlock(nn.Module):
    """Pre-norm attention + SwiGLU decoder block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = (
            SparseMoE(config) if config.moe.enabled else SwiGLU(config.dim, config.ffn_hidden_dim)
        )
        self.resid_dropout = nn.Dropout(config.resid_dropout)

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = hidden + self.resid_dropout(
            self.attention(self.attention_norm(hidden), attention_mask)
        )
        return self._feed_forward(hidden, attention_mask)

    def decode(
        self, hidden: Tensor, past: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, present = self.attention.decode(self.attention_norm(hidden), past)
        hidden, _, _ = self._feed_forward(hidden + self.resid_dropout(attended), None)
        return hidden, present

    def _feed_forward(
        self, hidden: Tensor, attention_mask: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized = self.ffn_norm(hidden)
        if isinstance(self.ffn, SparseMoE):
            transformed, auxiliary, load = self.ffn(normalized, attention_mask)
        else:
            transformed = self.ffn(normalized)
            auxiliary, load = hidden.new_zeros(()), hidden.new_zeros((0,))
        return hidden + self.resid_dropout(transformed), auxiliary, load


class PLMDecoder(nn.Module):
    """Dense reference decoder with explicit architecture experiments."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        _validate_reference_config(config)
        self.config = config
        self._cache_owner = object()
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._init_weights(config.init_std)
        # Initialize after the reference stack, preserving identical common
        # weights for the same seed when the auxiliary objective is enabled.
        self.prompt_set_projection: nn.Linear | None = None
        if config.prompt_set_loss_weight:
            if config.vocab_size <= ENTITY_BASE:
                raise ValueError("prompt set supervision requires product tokens at IDs 1024+")
            self.prompt_set_projection = nn.Linear(config.dim, config.dim, bias=False)
            nn.init.normal_(self.prompt_set_projection.weight, mean=0.0, std=config.init_std)
        self.symmetric_relation_projection: nn.Linear | None = None
        if config.symmetric_relation_loss_weight:
            if config.vocab_size < ENTITY_BASE + 3:
                raise ValueError("symmetric supervision requires at least three product tokens")
            self.symmetric_relation_projection = nn.Linear(config.dim, config.dim, bias=False)
            nn.init.normal_(
                self.symmetric_relation_projection.weight, mean=0.0, std=config.init_std
            )

    def _init_weights(self, init_std: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

    def decode(self, input_ids: Tensor, *, cache: KVCache | None = None) -> DecodeOutput:
        """Prefill or append an unpadded chunk without recomputing earlier tokens.

        This path scores tokens only, bypassing training losses and the optional
        prompt-set head. State belongs to this model and this one unchanged prefix.
        Discard it after a request or any change to model weights/device/dtype.
        """
        if self.training or torch.is_grad_enabled():
            raise ValueError("KV decoding requires eval mode and disabled autograd")
        if input_ids.ndim != 2 or not input_ids.shape[0] or not input_ids.shape[1]:
            raise ValueError("decode input must be a nonempty (batch, chunk) tensor")
        position = 0 if cache is None else cache.position
        if cache is not None:
            if cache.owner is not self._cache_owner:
                raise ValueError("KV cache belongs to a different model")
            if (
                len(cache.layers) != len(self.blocks)
                or not isinstance(position, int)
                or isinstance(position, bool)
                or position < 1
            ):
                raise ValueError("KV cache has invalid layer count or position")
            for layer in cache.layers:
                if len(layer) != 2 or any(
                    not isinstance(t, Tensor) or t.ndim != 4 or t.shape[-2] != position
                    for t in layer
                ):
                    raise ValueError("KV cache layers must agree on the cached prefix length")
        if position + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("cached prefix plus chunk exceeds max_seq_len")
        hidden = self.token_embedding(input_ids)
        present: list[tuple[Tensor, Tensor]] = []
        for index, block in enumerate(self.blocks):
            hidden, layer = cast(DecoderBlock, block).decode(
                hidden, None if cache is None else cache.layers[index]
            )
            present.append(layer)
        return DecodeOutput(
            logits=self.lm_head(self.final_norm(hidden)),
            cache=KVCache(tuple(present), position + input_ids.shape[1], self._cache_owner),
        )

    def forward(
        self,
        input_ids: Tensor | ModelInput,
        *,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> ModelOutput:
        if isinstance(input_ids, ModelInput):
            if labels is not None or attention_mask is not None:
                raise ValueError("explicit labels/attention_mask cannot accompany ModelInput")
            labels = input_ids.labels
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids
        input_ids = cast(Tensor, input_ids)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds configured max_seq_len")
        hidden = self.embed_dropout(self.token_embedding(input_ids))
        auxiliary_losses: list[Tensor] = []
        expert_loads: list[Tensor] = []
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden, auxiliary, load = checkpoint(
                    block, hidden, attention_mask, use_reentrant=False, preserve_rng_state=True
                )
            else:
                hidden, auxiliary, load = block(hidden, attention_mask)
            if self.config.moe.enabled:
                auxiliary_losses.append(auxiliary)
                expert_loads.append(load)
        auxiliary_loss = torch.stack(auxiliary_losses).mean() if auxiliary_losses else None
        expert_load = torch.stack(expert_loads) if expert_loads else None
        normalized = self.final_norm(hidden)
        logits = self.lm_head(normalized)
        set_logits = None
        if self.prompt_set_projection is not None:
            if input_ids.shape[1] < 5:
                raise ValueError("prompt set supervision requires the five-token protocol prompt")
            # ANSWER is position four. Causal attention prevents this state from
            # seeing answer tokens even in a teacher-forced training batch.
            # Shared entity embeddings receive direct relational supervision
            # when they occur as targets as well as when they occur as subjects.
            set_logits = F.linear(
                self.prompt_set_projection(normalized[:, 4]),
                self.token_embedding.weight[ENTITY_BASE:],
            )
        loss = None
        task_loss = None
        first_target_loss = None
        set_loss = None
        symmetric_loss = None
        symmetric_logits = None
        continuation_loss = None
        continuation_query_count = 0
        symmetric_margin_loss = None
        symmetric_margin_query_count = 0
        symmetric_margin_active_count = 0
        if self.symmetric_relation_projection is not None:
            if input_ids.shape[1] < 5:
                raise ValueError("symmetric supervision requires the five-token protocol prompt")
            subjects = input_ids[:, 1] - ENTITY_BASE
            dimensions = input_ids[:, 2]
            if ((subjects < 0) | (subjects >= self.config.vocab_size - ENTITY_BASE)).any():
                raise ValueError("symmetric supervision requires a product subject")
            if ((dimensions < 32) | (dimensions >= 128)).any():
                raise ValueError("symmetric supervision requires a steering-band dimension")
            # Shared entity vectors on both sides make the score symmetric.
            # Use FP32 for this small auxiliary path, including under BF16 training.
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                scale = self.config.dim**0.5
                entities = (
                    F.normalize(self.token_embedding.weight[ENTITY_BASE:].float(), dim=-1) * scale
                )
                steering = (
                    F.normalize(self.token_embedding.weight[dimensions].float(), dim=-1) * scale
                )
                relation = F.linear(steering, self.symmetric_relation_projection.weight.float())
                symmetric_logits = F.linear(entities[subjects] * relation, entities) / scale
        if labels is not None:
            if labels.ndim != 2 or labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            labels = labels.to(dtype=torch.long)
            if attention_mask is not None:
                if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                    raise ValueError("attention_mask must have the same shape as input_ids")
                labels = labels.masked_fill(
                    ~attention_mask.to(dtype=torch.bool, device=labels.device), -100
                )
            # Corpus labels align with input tokens. Position t predicts t+1;
            # shift here exactly once, retaining full logits for inference.
            prediction_logits = logits[:, :-1, :]
            next_labels = labels[:, 1:]
            valid_labels = next_labels != -100
            if not torch.any(valid_labels):
                raise ValueError("labels must contain at least one unmasked next-token target")
            loss = F.cross_entropy(
                prediction_logits.reshape(-1, logits.shape[-1]),
                next_labels.reshape(-1),
                ignore_index=-100,
            )
            task_loss = loss
            row_has_target = valid_labels.any(dim=1)
            first_positions = valid_labels.to(torch.int64).argmax(dim=1)
            row_indices = torch.arange(next_labels.shape[0], device=next_labels.device)
            first_logits = prediction_logits[row_indices, first_positions][row_has_target]
            first_labels = next_labels[row_indices, first_positions][row_has_target]
            first_target_loss = F.cross_entropy(first_logits, first_labels)
            if self.config.first_target_loss_weight != 1.0:
                extra = self.config.first_target_loss_weight - 1.0
                token_count = valid_labels.sum()
                first_count = row_has_target.sum()
                loss = (task_loss * token_count + extra * first_target_loss * first_count) / (
                    token_count + extra * first_count
                )
            if set_logits is not None:
                if not row_has_target.all() or not (first_positions == 4).all():
                    raise ValueError("prompt set labels must begin after the five-token prompt")
                set_loss = _prompt_set_loss(set_logits, next_labels)
                loss = loss + self.config.prompt_set_loss_weight * set_loss
            if symmetric_logits is not None:
                if not row_has_target.all() or not (first_positions == 4).all():
                    raise ValueError("symmetric labels must begin after the five-token prompt")
                symmetric_loss = _prompt_set_loss(
                    symmetric_logits, next_labels, excluded_ids=input_ids[:, 1] - ENTITY_BASE
                )
                loss = loss + self.config.symmetric_relation_loss_weight * symmetric_loss
                if self.config.symmetric_margin_loss_weight:
                    (
                        symmetric_margin_loss,
                        symmetric_margin_query_count,
                        symmetric_margin_active_count,
                    ) = _symmetric_margin_loss(
                        symmetric_logits,
                        next_labels,
                        excluded_ids=input_ids[:, 1] - ENTITY_BASE,
                        margin=self.config.symmetric_margin,
                    )
                    loss = loss + self.config.symmetric_margin_loss_weight * symmetric_margin_loss
            if self.config.continuation_set_loss_weight:
                if self.prompt_set_projection is None:
                    raise ValueError("continuation set supervision requires the prompt set head")
                # States 5, 6, 7 have consumed one, two, three teacher products.
                # Causal attention keeps their future-derived labels out of hidden states.
                continuation_logits = F.linear(
                    self.prompt_set_projection(normalized[:, 5:8]),
                    self.token_embedding.weight[ENTITY_BASE:],
                )
                continuation_loss, continuation_query_count = _continuation_set_loss(
                    continuation_logits, input_ids, labels
                )
                loss = loss + self.config.continuation_set_loss_weight * continuation_loss
            if self.config.z_loss_weight:
                log_z = torch.logsumexp(prediction_logits.float(), dim=-1)
                loss = (
                    loss
                    + self.config.z_loss_weight * log_z.square().masked_select(valid_labels).mean()
                )
            if auxiliary_loss is not None:
                loss = loss + self.config.moe.aux_loss_weight * auxiliary_loss
        return ModelOutput(
            logits=logits,
            loss=loss,
            aux_loss=auxiliary_loss,
            expert_load=expert_load,
            task_loss=task_loss,
            first_target_loss=first_target_loss,
            prompt_set_loss=set_loss,
            prompt_set_logits=set_logits,
            symmetric_relation_loss=symmetric_loss,
            symmetric_relation_logits=symmetric_logits,
            continuation_set_loss=continuation_loss,
            continuation_set_query_count=continuation_query_count,
            symmetric_margin_loss=symmetric_margin_loss,
            symmetric_margin_query_count=symmetric_margin_query_count,
            symmetric_margin_active_count=symmetric_margin_active_count,
        )


def _continuation_set_loss(
    set_logits: Tensor, input_ids: Tensor, labels: Tensor
) -> tuple[Tensor, int]:
    """Balanced remaining-set BCE, averaged by stage within each eligible query.

    Labels are input-aligned and already attention-masked. Product IDs through
    the current state and the subject are negatives even if repeated later.
    Empty remaining sets are left to the ordinary EOS/token objective.
    """
    batch_size, stages, product_count = set_logits.shape
    query_totals = set_logits.float().sum(dim=(1, 2)) * 0.0
    stage_counts = torch.zeros(batch_size, device=set_logits.device, dtype=torch.long)
    for stage in range(stages):
        position = 5 + stage
        suffix = labels[:, position + 1 :]
        products = (suffix >= ENTITY_BASE) & (suffix < ENTITY_BASE + product_count)
        targets = torch.zeros_like(set_logits[:, stage], dtype=torch.float32)
        targets.scatter_add_(
            1, (suffix - ENTITY_BASE).clamp(0, product_count - 1), products.float()
        )
        targets = (targets > 0).float()
        consumed = torch.cat((input_ids[:, 1:2], input_ids[:, 5 : position + 1]), dim=1)
        consumed_products = (consumed >= ENTITY_BASE) & (consumed < ENTITY_BASE + product_count)
        seen = torch.zeros_like(targets)
        seen.scatter_add_(
            1, (consumed - ENTITY_BASE).clamp(0, product_count - 1), consumed_products.float()
        )
        targets = targets.masked_fill(seen > 0, 0)
        positive_count = targets.sum(-1)
        negative_count = product_count - positive_count
        prefix = labels[:, 5 : position + 1]
        prefix_products = (prefix >= ENTITY_BASE) & (prefix < ENTITY_BASE + product_count)
        eligible = prefix_products.all(dim=1) & (positive_count > 0)
        scores = set_logits[:, stage].float()
        positive = (F.softplus(-scores) * targets).sum(-1) / positive_count.clamp_min(1)
        negative = (F.softplus(scores) * (1 - targets)).sum(-1) / negative_count.clamp_min(1)
        query_totals = query_totals + torch.where(eligible, (positive + negative) * 0.5, 0.0)
        stage_counts = stage_counts + eligible.long()
    participating = stage_counts > 0
    query_count = int(participating.sum().item())
    if not query_count:
        return set_logits.sum() * 0.0, 0
    return (query_totals / stage_counts.clamp_min(1))[participating].mean(), query_count


def _prompt_set_masks(
    set_logits: Tensor, labels: Tensor, *, excluded_ids: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Deduplicate supervised products and preserve BCE's subject/count contract."""
    product_labels = (labels >= ENTITY_BASE) & (labels < ENTITY_BASE + set_logits.shape[-1])
    targets = torch.zeros_like(set_logits, dtype=torch.float32)
    indices = (labels - ENTITY_BASE).clamp(0, set_logits.shape[-1] - 1)
    targets.scatter_add_(1, indices, product_labels.float())
    targets = (targets > 0).float()  # Repeated IDs cannot add positive weight.
    negatives = 1 - targets
    if excluded_ids is not None:
        if targets.gather(1, excluded_ids[:, None]).any():
            raise ValueError("symmetric labels must exclude the SAME subject")
        negatives = negatives.scatter(1, excluded_ids[:, None], 0)
    positive_count = targets.sum(dim=-1)
    negative_count = (
        targets.shape[-1] - positive_count if excluded_ids is None else negatives.sum(dim=-1)
    )
    if (positive_count == 0).any() or (negative_count == 0).any():
        raise ValueError("prompt set supervision needs positive and negative products per query")
    return targets, negatives, positive_count, negative_count


def _prompt_set_loss(
    set_logits: Tensor, labels: Tensor, *, excluded_ids: Tensor | None = None
) -> Tensor:
    """Query-balanced binary loss over product IDs; never includes control tokens."""
    targets, negatives, positive_count, negative_count = _prompt_set_masks(
        set_logits, labels, excluded_ids=excluded_ids
    )
    scores = set_logits.float()
    positive = (F.softplus(-scores) * targets).sum(-1) / positive_count
    negative = (F.softplus(scores) * negatives).sum(-1) / negative_count
    return ((positive + negative) * 0.5).mean()


def _symmetric_margin_loss(
    set_logits: Tensor, labels: Tensor, *, excluded_ids: Tensor, margin: float
) -> tuple[Tensor, int, int]:
    """Mean hardest-boundary hinge over all queries, including satisfied zeros.

    FP32 amin/amax split gradients evenly across tied extrema. Labels define
    membership only; the prompt-only head has already computed every score.
    """
    with torch.autocast(device_type=set_logits.device.type, enabled=False):
        targets, negatives, _, _ = _prompt_set_masks(set_logits, labels, excluded_ids=excluded_ids)
        scores = set_logits.float()
        weakest_true = scores.masked_fill(~targets.bool(), float("inf")).amin(dim=-1)
        strongest_negative = scores.masked_fill(~negatives.bool(), float("-inf")).amax(dim=-1)
        hinge = F.relu(margin + strongest_negative - weakest_true)
        return hinge.mean(), scores.shape[0], int((hinge > 0).sum().item())


def _validate_reference_config(config: ModelConfig) -> None:
    unsupported: list[str] = []
    if config.norm != "rmsnorm":
        unsupported.append("norm != rmsnorm")
    if config.activation != "swiglu":
        unsupported.append("activation != swiglu")
    if config.use_mup:
        unsupported.append("use_mup")
    if config.rope.scaling_type != "none":
        unsupported.append("rope scaling")
    if not config.attention.causal:
        unsupported.append("non-causal attention")
    if config.attention.backend != "sdpa":
        unsupported.append(f"attention.backend={config.attention.backend}")
    if config.attention.logit_soft_cap is not None and config.attention.logit_soft_cap <= 0.0:
        raise ValueError("attention.logit_soft_cap must be positive when set")
    if unsupported:
        raise NotImplementedError(
            "minimal reference decoder does not implement: " + ", ".join(unsupported)
        )
