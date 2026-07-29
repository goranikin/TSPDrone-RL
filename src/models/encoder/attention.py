import torch
from torch import nn

from src.models.initialization import initialize_kool_linear
from src.models.layers.attention_encoder_layer import AttentionEncoderLayer
from src.models.types import EncoderOutput


class AttentionEncoder(nn.Module):
    """Attention Model encoder from Kool et al. (ported from compare-architectures)."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, d_model)
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

    def forward(self, node_features: torch.Tensor) -> EncoderOutput:
        node_embeddings = self.input_proj(node_features.float())
        for layer in self.layers:
            node_embeddings = layer(node_embeddings)
        return EncoderOutput(
            node_embeddings=node_embeddings,
            graph_embedding=node_embeddings.mean(dim=1),
        )
