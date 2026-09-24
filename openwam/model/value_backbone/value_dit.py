"""Independent copy of OpenWAM ActionDiT's joint-attention value expert.

The value stream owns the same projection, 1-D RoPE, timestep AdaLN, text
cross-attention and per-layer self-attention/FFN topology as ActionDiT. It is
not an ActionDiT subclass; the final norm/projection and scheduler are value
specific. Its attention geometry is exactly the action expert's geometry.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from openwam.model.action_backbone.components import (
    TimestepEmbedding,
    TimestepModulation,
    precompute_freqs_cis_1d,
    rope_apply_1d,
)
from openwam.model.action_backbone.separate_action_dit import ActionDiTState, SelfAttnActionDiTBlock
from openwam.model.value_backbone.scheduler import ValueScheduler


class ValueDiT(nn.Module):
    """Independent ActionDiT-shaped expert which predicts value velocity."""

    DEFAULT_VALUE_DIM = 1
    DEFAULT_SHIFT_VALUE = ValueScheduler.DEFAULT_SHIFT

    def __init__(
        self,
        value_dim: int,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        num_layers: int,
        video_dim: int,
        bridge_layers: Tuple[int, ...],
        *,
        attn_head_dim: Optional[int] = None,
        text_dim: int = 4096,
        freq_dim: int = 256,
        rope_base_length: int = 57,
        eps: float = 1e-6,
        shift_value: float = DEFAULT_SHIFT_VALUE,
    ):
        super().__init__()
        if int(value_dim) <= 0:
            raise ValueError(f"value_dim must be > 0, got {value_dim}")
        if len(bridge_layers) != int(num_layers):
            raise ValueError(
                f"bridge_layers ({len(bridge_layers)}) must equal num_layers ({num_layers}); "
                "each value block corresponds to one video layer"
            )
        if int(num_heads) <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if attn_head_dim is None:
            attn_head_dim = int(dim) // int(num_heads)
            if attn_head_dim * int(num_heads) != int(dim):
                raise ValueError("Cannot infer attn_head_dim from dim / num_heads; specify it explicitly")
        attn_head_dim = int(attn_head_dim)
        if attn_head_dim <= 0 or attn_head_dim % 2:
            raise ValueError(f"attn_head_dim must be a positive even int, got {attn_head_dim}")
        if int(rope_base_length) <= 0:
            raise ValueError(f"rope_base_length must be > 0, got {rope_base_length}")
        try:
            shift_value = float(shift_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"shift_value must be a positive finite number, got {shift_value!r}") from exc
        if not math.isfinite(shift_value) or shift_value <= 0.0:
            raise ValueError(f"shift_value must be a positive finite number, got {shift_value!r}")
        self.value_dim = int(value_dim)
        self.dim = int(dim)
        self._num_heads = int(num_heads)
        self._head_dim = attn_head_dim
        self._num_layers = int(num_layers)
        self._bridge_layers = tuple(bridge_layers)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.rope_base_length = int(rope_base_length)
        self.max_value_len = self.rope_base_length
        self._shift_value = shift_value
        self.scheduler = ValueScheduler()
        # Copy ActionDiT's joint-self-attn expert modules, with a value-width
        # input/output. The same SelfAttnActionDiTBlock supplies local self-
        # attention, video/text context cross-attention, AdaLN, and FFN.
        self.value_proj_in = nn.Linear(int(value_dim), int(dim))
        self.text_embedding = nn.Sequential(
            nn.Linear(int(text_dim), int(dim)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(dim), int(dim)),
        )
        self.freqs = precompute_freqs_cis_1d(attn_head_dim, int(rope_base_length))
        self.time_embedding = TimestepEmbedding(int(freq_dim), int(dim))
        self.time_projection = TimestepModulation(int(dim), 6)
        self.blocks = nn.ModuleList(
            [
                SelfAttnActionDiTBlock(
                    int(dim), int(num_heads), attn_head_dim, int(ffn_dim),
                    kv_hidden_dim=int(dim), eps=float(eps),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.value_norm_out = nn.LayerNorm(int(dim), eps=float(eps), elementwise_affine=False)
        self.value_proj_out = nn.Linear(int(dim), int(value_dim))


    @property
    def shift_value(self):
        return self._shift_value

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def set_dtype_device(self, dtype, device) -> None:
        self.to(dtype=dtype, device=device)

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """All value weights are saved with the architecture checkpoint."""

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse=recurse)
        ref = next(self.parameters(), None)
        if ref is not None and self.freqs.device != ref.device:
            self.freqs = self.freqs.to(device=ref.device)
        return module

    @property
    def value_time_embedding(self):
        return self.time_embedding

    @property
    def value_time_projection(self):
        return self.time_projection

    @property
    def value_decoder(self):
        return self.value_proj_out

    def _embed_values(self, value_tokens):
        if value_tokens.ndim != 3 or value_tokens.shape[-1] != self.value_dim:
            raise ValueError(f"values must be [B, T, {self.value_dim}], got {tuple(value_tokens.shape)}")
        if value_tokens.shape[1] > self.rope_base_length:
            raise ValueError(
                f"Value sequence length {value_tokens.shape[1]} exceeds rope_base_length {self.rope_base_length}"
            )
        return self.value_proj_in(value_tokens)

    def _prepare_timestep(self, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
        if timestep.ndim != 1 or timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"value timestep must be [B] or [1], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, value timestep length must match batch_size")
            timestep = timestep.expand(batch_size)
        return timestep

    def _prepare_context(self, context, context_mask, *, batch_size, seq_len, dtype, device):
        if context is None:
            raise ValueError("ValueDiT.prepare_value_state requires raw text/proprio context")
        if context.ndim != 3 or context.shape[0] != batch_size or context.shape[-1] != self.text_dim:
            raise ValueError(
                f"context must have shape [{batch_size}, L, {self.text_dim}], got {tuple(context.shape)}"
            )
        context = context.to(device=device, dtype=dtype)
        context_emb = self.text_embedding(context)
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
        else:
            if context_mask.shape != context.shape[:2]:
                raise ValueError(
                    f"context_mask must have shape {tuple(context.shape[:2])}, got {tuple(context_mask.shape)}"
                )
            context_mask = context_mask.to(device=device, dtype=torch.bool)
        return context_emb, context_mask.unsqueeze(1).expand(-1, seq_len, -1)

    def prepare_value_state(
        self,
        noisy_values,
        timestep,
        *,
        context=None,
        context_mask=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        """Copy ActionDiT.prepare_state for the value-owned modules."""
        from openwam.model.architectures.base import ActionState

        x = self._embed_values(noisy_values)
        timestep = self._prepare_timestep(timestep, noisy_values.shape[0])
        t = self.time_embedding(timestep)
        t_mod = self.time_projection(t)
        freqs = self.freqs[: x.shape[1]].to(device=x.device)
        value_context, value_context_mask = self._prepare_context(
            context, context_mask,
            batch_size=noisy_values.shape[0], seq_len=x.shape[1], dtype=x.dtype, device=x.device,
        )
        payload = ActionDiTState(
            x_action=x, t_mod=t_mod, t_embed=t, action_freqs=freqs,
            context=value_context, context_mask=value_context_mask,
        )
        return ActionState(action_latents=noisy_values, timestep=timestep, payload=payload)

    def pre_attn_at_layer(self, layer_id: int, state):
        """Copy ActionDiT's AdaLN + local Q/K/V + 1-D RoPE computation."""
        payload = state.payload
        block = self.blocks[layer_id]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.to(dtype=payload.t_mod.dtype, device=payload.t_mod.device) + payload.t_mod
        ).chunk(6, dim=1)
        residual_x = payload.x_action
        attn_input = block.self_attn_norm(residual_x) * (1 + scale_msa) + shift_msa
        sa = block.self_attn
        q = rearrange(sa.norm_q(sa.q(attn_input)), "b s (n d) -> b n s d", n=self._num_heads)
        k = rearrange(sa.norm_k(sa.k(attn_input)), "b s (n d) -> b n s d", n=self._num_heads)
        v = rearrange(sa.v(attn_input), "b s (n d) -> b n s d", n=self._num_heads)
        q = rope_apply_1d(q, payload.action_freqs)
        k = rope_apply_1d(k, payload.action_freqs)
        q_out = rearrange(q, "b n s d -> b s (n d)", n=self._num_heads)
        k_out = rearrange(k, "b n s d -> b s (n d)", n=self._num_heads)
        v_out = rearrange(v, "b n s d -> b s (n d)", n=self._num_heads)
        return q_out, k_out, v_out, (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attn_at_layer(self, layer_id: int, state, attn_out, post_state):
        """Copy ActionDiT's residual gate, context cross-attention and FFN."""
        payload = state.payload
        block = self.blocks[layer_id]
        residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state
        x = block.gate(residual_x, gate_msa, block.self_attn.o(attn_out))
        if payload.context is not None:
            x = x + block.cross_attn(
                block.context_attn_norm(x), payload.context, ctx_mask=payload.context_mask
            )
        mlp_input = block.ffn_norm(x) * (1 + scale_mlp) + shift_mlp
        payload.x_action = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return state

    def extract_value_prediction(self, state):
        return self.value_proj_out(self.value_norm_out(state.payload.x_action))

class ValueBackbone(ValueDiT):
    """Public semantic name for the ActionDiT-shaped WAV value expert."""


__all__ = ["ValueBackbone", "ValueDiT"]
