import torch
from src.models.initialization import initialize_kool_linear
from src.models.layers.multi_head_self_attention import MultiHeadSelfAttention
from src.models.layers.node_batch_norm import NodeBatchNorm
from torch import nn


class AttentionEncoderLayer(nn.Module):
    """Kool encoder layer with residual MHA/FF blocks and BatchNorm."""

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
        self.attention_norm = NodeBatchNorm(d_model)
        self.feed_forward_norm = NodeBatchNorm(d_model)
        initialize_kool_linear(self.feed_forward_in)
        initialize_kool_linear(self.feed_forward_out)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.self_attention(node_embeddings, attention_mask)
        intermediate = self.attention_norm(
            node_embeddings + self.attention_dropout(attended),
            node_mask,
        )
        transformed = self.feed_forward_out(
            torch.relu(self.feed_forward_in(intermediate))
        )
        return self.feed_forward_norm(
            intermediate + self.feed_forward_dropout(transformed),
            node_mask,
        )
