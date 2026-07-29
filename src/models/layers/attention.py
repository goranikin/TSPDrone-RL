import math

import numpy as np
import torch
from torch import nn


class SkipConnection(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: int,
        val_dim: int | None = None,
        key_dim: int | None = None,
    ):
        super().__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self) -> None:
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(
        self,
        q: torch.Tensor,
        h: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h is None:
            h = q

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):
    def __init__(self, embed_dim: int, normalization: str = "batch"):
        super().__init__()

        normalizer_class = {"batch": nn.BatchNorm1d, "instance": nn.InstanceNorm1d}.get(
            normalization, None
        )

        self.normalizer = (
            normalizer_class(embed_dim, affine=True)
            if normalizer_class is not None
            else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        if isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        assert self.normalizer is None, "Unknown normalizer type"
        return input


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int = 512,
        normalization: str = "batch",
    ):
        super().__init__(
            SkipConnection(
                MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )


class ConvEncoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class PointerAttention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size: int, use_tanh: bool = False, C: float = 10):
        super().__init__()
        self.use_tanh = use_tanh
        self.v = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        self.project_d = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=1
        )
        self.project_ref = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=1
        )
        self.project_query = nn.Linear(hidden_size, hidden_size)
        self.C = C

    def forward(
        self,
        static_hidden: torch.Tensor,
        dynamic_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_ex = self.project_d(dynamic_hidden)
        batch_size, hidden_size, n_nodes = static_hidden.size()
        e = self.project_ref(static_hidden)
        decoder_hidden = self.project_query(decoder_hidden)

        v = self.v.expand(batch_size, 1, hidden_size)
        q = decoder_hidden.view(batch_size, hidden_size, 1).expand(
            batch_size, hidden_size, n_nodes
        )

        u = torch.bmm(v, torch.tanh(e + q + d_ex)).squeeze(1)
        logits = self.C * torch.tanh(u) if self.use_tanh else u
        return e, logits


class CriticAttention(nn.Module):
    """Attention used by the critic value network."""

    def __init__(self, hidden_size: int, use_tanh: bool = False, C: float = 10):
        super().__init__()
        self.use_tanh = use_tanh
        self.v = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        self.project_d_ex = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=1
        )
        self.project_ref = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=1
        )
        self.project_query = nn.Linear(hidden_size, hidden_size)
        self.C = C

    def forward(
        self,
        static_hidden: torch.Tensor,
        dynamic_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, hidden_size, n_nodes = static_hidden.size()
        d_ex = self.project_d_ex(dynamic_hidden)
        e = self.project_ref(static_hidden)
        decoder_hidden = self.project_query(decoder_hidden)

        v = self.v.expand(batch_size, 1, hidden_size)
        q = decoder_hidden.view(batch_size, hidden_size, 1).expand(
            batch_size, hidden_size, n_nodes
        )

        u = torch.bmm(v, torch.tanh(e + q + d_ex)).squeeze(1)
        logits = self.C * torch.tanh(u) if self.use_tanh else u
        return e, logits
