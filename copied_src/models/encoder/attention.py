import torch
from src.models.initialization import initialize_kool_linear
from src.models.layers.attention_encoder_layer import AttentionEncoderLayer
from src.types import EncoderOutput
from torch import nn


class AttentionEncoder(nn.Module):
    """Attention Model encoder from Kool et al., equations (2), (3), (10)--(16)."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.0,
        depot_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.depot_input_dim = depot_input_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.depot_input_proj = (
            nn.Linear(depot_input_dim, d_model) if depot_input_dim is not None else None
        )
        self.layers = nn.ModuleList(
            [
                AttentionEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        initialize_kool_linear(self.input_proj)
        if self.depot_input_proj is not None:
            initialize_kool_linear(self.depot_input_proj)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        del adjacency, edge_features
        return self._encode(node_features, node_mask=node_mask)

    def _encode(
        self,
        node_features: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        features = node_features.float()
        node_embeddings = self.input_proj(features)
        if self.depot_input_proj is not None:
            if self.depot_input_dim is None:
                raise RuntimeError("depot_input_dim is not configured")
            depot = self.depot_input_proj(features[:, :1, : self.depot_input_dim])
            node_embeddings = torch.cat([depot, node_embeddings[:, 1:]], dim=1)
        for layer in self.layers:
            node_embeddings = layer(
                node_embeddings,
                attention_mask=attention_mask,
                node_mask=node_mask,
            )
        graph_embedding = self._pool_graph(node_embeddings, node_mask)
        return EncoderOutput(
            node_embeddings=node_embeddings,
            graph_embedding=graph_embedding,
        )

    @staticmethod
    def _pool_graph(
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if node_mask is None:
            return node_embeddings.mean(dim=1)
        batch_size, node_count, _ = node_embeddings.shape
        if node_mask.shape != (batch_size, node_count):
            raise ValueError(
                "node_mask must have shape "
                f"{(batch_size, node_count)}; got {tuple(node_mask.shape)}"
            )
        weights = node_mask.to(
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        ).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        return (node_embeddings * weights).sum(dim=1) / count
