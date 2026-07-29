"""Shared model tensor containers."""

from pydantic import BaseModel, ConfigDict
import torch


class EncoderOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
