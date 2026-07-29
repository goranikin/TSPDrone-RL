import torch
import torch.nn.functional as F
from torch import nn


class CausalTransformerLayerCell(nn.Module):
    """Advance one pre-normalized Transformer layer by one causal token."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation_dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_history = (
            current if history is None else torch.cat([history, current], dim=1)
        )
        normalized_history = self.norm1(layer_history)
        normalized_current = normalized_history[:, -current.size(1) :]
        attention, _ = self.self_attention(
            normalized_current,
            normalized_history,
            normalized_history,
            need_weights=False,
        )
        output = current + self.attention_dropout(attention)
        feed_forward = self.linear2(
            self.activation_dropout(F.relu(self.linear1(self.norm2(output))))
        )
        return output + self.output_dropout(feed_forward), layer_history
