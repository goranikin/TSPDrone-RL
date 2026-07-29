import torch
from torch import nn

from src.models.layers.attention import MultiHeadAttentionLayer


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    n_heads, embed_dim, feed_forward_hidden, normalization
                )
                for _ in range(n_layers)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AttentionEncoder(nn.Module):
    """Static node encoder: linear embed + graph attention."""

    def __init__(
        self,
        embed_dim: int,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
    ):
        super().__init__()
        self.init_embed = nn.Linear(2, embed_dim)
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        return self.embedder(self.init_embed(static))
