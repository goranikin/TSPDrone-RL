"""TSP-D decoder: causal Transformer over action history + additive pointer.

Drop-in alternative to ``tspd_lstm``: same ``PointerAttention`` (and optional
dynamics fusion in the pointer energy); only the recurrent block differs.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.models.decoder.base import StepDecoder
from src.models.initialization import initialize_kool_linear
from src.models.layers.multi_head_self_attention import MultiHeadSelfAttention
from src.models.layers.pointer import PointerAttention
from src.models.types import EncoderOutput


def _sinusoidal_positions(
    seq_len: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``[1, T, H]`` sinusoidal positional encodings."""
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def _causal_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Boolean allow-mask ``[B, T, T]`` (True = attend)."""
    base = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return base.unsqueeze(0).expand(batch_size, -1, -1)


class CausalTransformerLayer(nn.Module):
    """Pre-norm style residual MHA + FFN with a causal self-attention mask."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward_in = nn.Linear(d_model, d_ff)
        self.feed_forward_out = nn.Linear(d_ff, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        initialize_kool_linear(self.feed_forward_in)
        initialize_kool_linear(self.feed_forward_out)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attended = self.self_attention(tokens, attention_mask)
        intermediate = self.attention_norm(tokens + self.attention_dropout(attended))
        transformed = self.feed_forward_out(torch.relu(self.feed_forward_in(intermediate)))
        return self.feed_forward_norm(
            intermediate + self.feed_forward_dropout(transformed)
        )


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
        # history: [B, T, H] of previous-step embeddings (starts empty).
        return {"history": None}

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
        token = prev_embed.unsqueeze(1)  # [B, 1, H]
        history = state["history"]
        history = token if history is None else torch.cat([history, token], dim=1)

        batch_size, seq_len, _ = history.shape
        positions = _sinusoidal_positions(
            seq_len,
            self.hidden_size,
            device=history.device,
            dtype=history.dtype,
        )
        tokens = self.input_dropout(history + positions)
        mask = _causal_mask(batch_size, seq_len, history.device)
        for layer in self.layers:
            tokens = layer(tokens, mask)

        query = tokens[:, -1]
        static_hidden = encoder_output.node_embeddings.permute(0, 2, 1)
        dyn = dynamic_hidden if self.use_dynamics else None
        logits = self.encoder_attn(static_hidden, query, dyn)
        return logits, {"history": history}
