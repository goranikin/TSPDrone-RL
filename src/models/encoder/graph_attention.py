import torch
from torch import nn

from src.models.layers.attention import MultiHeadAttentionLayer


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        node_dim: int | None = None,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
    ):
        super().__init__()

        self.init_embed = (
            nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        )

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    n_heads, embed_dim, feed_forward_hidden, normalization
                )
                for _ in range(n_layers)
            )
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert mask is None, "TODO mask not yet supported!"

        h = (
            self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
            if self.init_embed is not None
            else x
        )

        h = self.layers(h)

        return (
            h,
            h.mean(dim=1),
        )


class AttentionEncoder(nn.Module):
    """Static node encoder: linear embed + graph attention."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_encode_layers: int = 3,
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        normalization: str = "batch",
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        shrink_size: int | None = None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = "sampling"
        self.temp = 1.0
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        node_dim = 2
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        embeddings, _ = self.embedder(self._init_embed(static))
        return embeddings

    def _init_embed(self, input: torch.Tensor) -> torch.Tensor:
        return self.init_embed(input)
