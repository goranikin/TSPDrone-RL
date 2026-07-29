import math

import torch
from torch import nn

from src.models.initialization import initialize_kool_linear, kool_uniform_


class MultiHeadSelfAttention(nn.Module):
    """Kool et al. Appendix A, equations (10)--(14)."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        for projection in (self.query_proj, self.key_proj, self.value_proj):
            initialize_kool_linear(projection)
        kool_uniform_(self.output_proj.weight, self.head_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, node_count, _ = node_embeddings.shape
        queries = self._split_heads(self.query_proj(node_embeddings))
        keys = self._split_heads(self.key_proj(node_embeddings))
        values = self._split_heads(self.value_proj(node_embeddings))
        compatibility = torch.matmul(queries, keys.transpose(-2, -1))
        compatibility = compatibility / math.sqrt(self.head_dim)
        if attention_mask is not None:
            expected_shape = (batch_size, node_count, node_count)
            if attention_mask.shape != expected_shape:
                raise ValueError(
                    "attention_mask must have shape "
                    f"{expected_shape}; got {tuple(attention_mask.shape)}"
                )
            allowed = attention_mask.to(device=compatibility.device, dtype=torch.bool)
            compatibility = compatibility.masked_fill(
                ~allowed.unsqueeze(1),
                float("-inf"),
            )
        attention = torch.softmax(compatibility, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        heads = torch.matmul(attention, values)
        concatenated = (
            heads.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, node_count, self.d_model)
        )
        return self.output_proj(concatenated)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, _ = values.shape
        return values.reshape(
            batch_size,
            node_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
