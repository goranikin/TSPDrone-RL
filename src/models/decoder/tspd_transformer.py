"""TSP-D decoder: causal Transformer over action history + additive pointer.

Drop-in alternative to ``tspd_lstm``: same ``PointerAttention`` (and optional
dynamics fusion in the pointer energy); only the recurrent block differs.

Each decode step only attends the **new** token to the KV cache (equivalent to
full causal self-attention for the last position, but ``O(B·T)`` per step
instead of ``O(B·T²)``). Full-sequence recompute was OOM at LSTM-sized batches.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.models.decoder.base import StepDecoder
from src.models.initialization import initialize_kool_linear, kool_uniform_
from src.models.layers.pointer import PointerAttention
from src.models.types import EncoderOutput


def _sinusoidal_position(
    index: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``[1, 1, H]`` sinusoidal encoding for a single timestep."""
    position = torch.tensor([[index]], device=device, dtype=dtype)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(1, 1, dim, device=device, dtype=dtype)
    pe[0, 0, 0::2] = torch.sin(position * div_term)
    pe[0, 0, 1::2] = torch.cos(position * div_term)
    return pe


class CausalTransformerLayer(nn.Module):
    """Causal MHA + FFN with incremental (KV-cache) last-token forward."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.feed_forward_in = nn.Linear(d_model, d_ff)
        self.feed_forward_out = nn.Linear(d_ff, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        for projection in (self.query_proj, self.key_proj, self.value_proj):
            initialize_kool_linear(projection)
        kool_uniform_(self.output_proj.weight, self.head_dim)
        initialize_kool_linear(self.feed_forward_in)
        initialize_kool_linear(self.feed_forward_out)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = values.shape
        return values.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward_last(
        self,
        token: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process one new token ``[B, 1, H]``; return output and updated KV cache."""
        query = self._split_heads(self.query_proj(token))
        key = self._split_heads(self.key_proj(token))
        value = self._split_heads(self.value_proj(token))
        if cache is not None:
            cached_key, cached_value = cache
            key = torch.cat([cached_key, key], dim=2)
            value = torch.cat([cached_value, value], dim=2)
        # query: [B, heads, 1, d], key/value: [B, heads, T, d]
        compatibility = torch.matmul(query, key.transpose(-2, -1))
        compatibility = compatibility / math.sqrt(self.head_dim)
        attention = torch.softmax(compatibility, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        heads = torch.matmul(attention, value)
        attended = (
            heads.transpose(1, 2)
            .contiguous()
            .reshape(token.shape[0], 1, self.d_model)
        )
        attended = self.output_proj(attended)
        intermediate = self.attention_norm(token + self.attention_dropout(attended))
        transformed = self.feed_forward_out(
            torch.relu(self.feed_forward_in(intermediate))
        )
        output = self.feed_forward_norm(
            intermediate + self.feed_forward_dropout(transformed)
        )
        return output, (key, value)


class TSPDTransformerDecoder(StepDecoder):
    """Causal Transformer over chosen-node embeddings → additive pointer query."""

    def __init__(
        self,
        hidden_size: int,
        *,
        use_dynamics: bool,
        num_layers: int = 1,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        use_tanh: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.hidden_size = hidden_size
        self.use_dynamics = use_dynamics
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                CausalTransformerLayer(hidden_size, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.encoder_attn = PointerAttention(
            hidden_size,
            use_dynamics=use_dynamics,
            use_tanh=use_tanh,
        )
        self.input_dropout = nn.Dropout(p=dropout)

    def reset(self, encoder_output: EncoderOutput, batch_size: int) -> Any:
        del encoder_output, batch_size
        return {"step": 0, "caches": [None] * self.num_layers}

    def step(
        self,
        encoder_output: EncoderOutput,
        *,
        prev_embed: torch.Tensor,
        dynamic_hidden: torch.Tensor | None,
        state: Any,
        avail_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        del avail_actions
        step_index = int(state["step"])
        position = _sinusoidal_position(
            step_index,
            self.hidden_size,
            device=prev_embed.device,
            dtype=prev_embed.dtype,
        )
        token = self.input_dropout(prev_embed.unsqueeze(1) + position)

        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = list(state["caches"])
        for layer_index, layer in enumerate(self.layers):
            token, caches[layer_index] = layer.forward_last(token, caches[layer_index])

        query = token.squeeze(1)
        static_hidden = encoder_output.node_embeddings.permute(0, 2, 1)
        dyn = dynamic_hidden if self.use_dynamics else None
        logits = self.encoder_attn(static_hidden, query, dyn)
        return logits, {"step": step_index + 1, "caches": caches}
