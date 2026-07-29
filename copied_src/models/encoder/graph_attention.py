"""Adjacency-aware attention encoder for sparse graph problems."""

import torch

from src.models.encoder.attention import AttentionEncoder
from src.types import EncoderOutput


class GraphAttentionEncoder(AttentionEncoder):
    """Apply Kool-style attention only to graph neighbors and each node itself."""

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        del edge_features
        if adjacency is None:
            raise ValueError("GraphAttentionEncoder requires an adjacency tensor")
        batch_size, node_count, _ = node_features.shape
        expected_shape = (batch_size, node_count, node_count)
        if adjacency.shape != expected_shape:
            raise ValueError(
                f"adjacency must have shape {expected_shape}; "
                f"got {tuple(adjacency.shape)}"
            )
        valid_nodes = (
            torch.ones(
                batch_size,
                node_count,
                dtype=torch.bool,
                device=node_features.device,
            )
            if node_mask is None
            else node_mask.to(device=node_features.device, dtype=torch.bool)
        )
        if valid_nodes.shape != (batch_size, node_count):
            raise ValueError(
                "node_mask must have shape "
                f"{(batch_size, node_count)}; got {tuple(valid_nodes.shape)}"
            )
        self_loops = torch.eye(
            node_count,
            dtype=torch.bool,
            device=node_features.device,
        ).unsqueeze(0)
        attention_mask = (
            adjacency.to(
                device=node_features.device,
                dtype=torch.bool,
            )
            | self_loops
        )
        valid_pairs = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        attention_mask &= valid_pairs
        return self._encode(
            node_features,
            attention_mask=attention_mask,
            node_mask=valid_nodes,
        )
