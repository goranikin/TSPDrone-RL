import torch
from torch import nn

from src.models.initialization import kool_uniform_


class NodeBatchNorm(nn.Module):
    """Batch-normalize the feature axis of a ``[B, N, d_h]`` node tensor."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.normalizer = nn.BatchNorm1d(d_model)
        kool_uniform_(self.normalizer.weight, d_model)
        kool_uniform_(self.normalizer.bias, d_model)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, node_count, d_model = node_embeddings.shape
        flattened = node_embeddings.reshape(-1, d_model)
        if node_mask is None:
            normalized = self.normalizer(flattened)
            return normalized.reshape(batch_size, node_count, d_model)
        if node_mask.shape != (batch_size, node_count):
            raise ValueError(
                "node_mask must have shape "
                f"{(batch_size, node_count)}; got {tuple(node_mask.shape)}"
            )
        valid = node_mask.to(device=node_embeddings.device, dtype=torch.bool).reshape(-1)
        if not bool(valid.any()):
            return torch.zeros_like(node_embeddings)
        normalized = torch.zeros_like(flattened)
        normalized[valid] = self.normalizer(flattened[valid])
        return normalized.reshape(batch_size, node_count, d_model)
