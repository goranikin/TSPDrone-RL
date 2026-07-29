import torch
from torch import nn


class DynamicEncoder(nn.Module):
    """Pointwise linear projection of per-node dynamic features."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: [B, N, C] → [B, H, N] for pointer / mean-pool consumers
        return self.proj(input).permute(0, 2, 1)


class PointerAttention(nn.Module):
    """Paper TSP-D additive pointer with optional dynamic fusion."""

    def __init__(
        self,
        hidden_size: int,
        *,
        use_dynamics: bool = True,
        use_tanh: bool = False,
        C: float = 10,
    ):
        super().__init__()
        self.use_dynamics = use_dynamics
        self.use_tanh = use_tanh
        self.C = C
        self.v = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        self.project_ref = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.project_query = nn.Linear(hidden_size, hidden_size)
        self.project_d = (
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1) if use_dynamics else None
        )

    def forward(
        self,
        static_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor,
        dynamic_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # static_hidden: [B, H, N]
        batch_size, hidden_size, n_nodes = static_hidden.size()
        e = self.project_ref(static_hidden)
        query = self.project_query(decoder_hidden)
        q = query.view(batch_size, hidden_size, 1).expand(batch_size, hidden_size, n_nodes)
        energy = e + q
        if self.use_dynamics:
            if dynamic_hidden is None or self.project_d is None:
                raise RuntimeError("Dynamic features required when use_dynamics=True")
            energy = energy + self.project_d(dynamic_hidden)
        v = self.v.expand(batch_size, 1, hidden_size)
        u = torch.bmm(v, torch.tanh(energy)).squeeze(1)
        return self.C * torch.tanh(u) if self.use_tanh else u
