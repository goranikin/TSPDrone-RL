import torch
from torch import nn

from src.models.decoding import append_stop_embedding


class AdditivePointerAttention(nn.Module):
    """Pointer Networks equation (3): additive scores over input positions."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int,
        *,
        has_extra_stop: bool,
    ) -> None:
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.score_vector = nn.Parameter(torch.empty(attention_dim))
        self.stop_embedding = (
            nn.Parameter(torch.empty(encoder_dim)) if has_extra_stop else None
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        decoder_hidden: torch.Tensor,
        action_count: int,
    ) -> torch.Tensor:
        candidates = append_stop_embedding(
            node_embeddings,
            self.stop_embedding,
            action_count,
        )
        encoder_terms = self.encoder_proj(candidates)
        decoder_term = self.decoder_proj(decoder_hidden).unsqueeze(1)
        energy = torch.tanh(encoder_terms + decoder_term)
        return torch.einsum("bad,d->ba", energy, self.score_vector)
