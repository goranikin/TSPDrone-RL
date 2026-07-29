import torch
from torch import nn


class AdditivePointerAttention(nn.Module):
    """Pointer Networks equation (3): additive scores over input positions."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int,
    ) -> None:
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.score_vector = nn.Parameter(torch.empty(attention_dim))
        nn.init.uniform_(self.score_vector, -0.08, 0.08)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> torch.Tensor:
        encoder_terms = self.encoder_proj(node_embeddings)
        decoder_term = self.decoder_proj(decoder_hidden).unsqueeze(1)
        energy = torch.tanh(encoder_terms + decoder_term)
        return torch.einsum("bad,d->ba", energy, self.score_vector)
