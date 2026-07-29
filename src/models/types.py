"""Shared model tensor containers."""

import torch
from pydantic import BaseModel, ConfigDict


class EncoderOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
