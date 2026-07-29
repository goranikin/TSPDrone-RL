import torch
import torch.nn.functional as F
from torch import nn

from src.models.layers.attention import ConvEncoder, CriticAttention


class Critic(nn.Module):
    """Estimates problem complexity / expected completion time."""

    def __init__(self, hidden_size: int, num_layers: int = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dynamic_d_ex = ConvEncoder(1, hidden_size)
        self.static_encoder = ConvEncoder(2, hidden_size)
        self.attention1 = CriticAttention(hidden_size)
        self.attention2 = CriticAttention(hidden_size)
        self.attention3 = CriticAttention(hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static: torch.Tensor, d_ex: torch.Tensor) -> torch.Tensor:
        static_hidden = self.static_encoder(static)
        batch_size, _, _ = static_hidden.size()
        dynamic_hidden = self.dynamic_d_ex(d_ex.permute(0, 2, 1))

        hy = torch.zeros(
            batch_size,
            self.hidden_size,
            device=static.device,
            dtype=static.dtype,
        )

        e, logits = self.attention1(static_hidden, dynamic_hidden, hy)
        probs = torch.softmax(logits, dim=1)
        hy = torch.matmul(probs.unsqueeze(1), e.permute(0, 2, 1)).squeeze(1)

        e, logits = self.attention2(static_hidden, dynamic_hidden, hy)
        probs = torch.softmax(logits, dim=1)
        hy = torch.matmul(probs.unsqueeze(1), e.permute(0, 2, 1)).squeeze(1)

        e, logits = self.attention3(static_hidden, dynamic_hidden, hy)
        probs = torch.softmax(logits, dim=1)
        hy = torch.matmul(probs.unsqueeze(1), e.permute(0, 2, 1)).squeeze(1)

        out = F.relu(self.fc1(hy))
        out = self.fc2(out)
        return out
