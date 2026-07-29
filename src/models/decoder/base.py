"""TSP-D step decoder interface."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from src.models.types import EncoderOutput


class StepDecoder(nn.Module, ABC):
    """One truck/drone action step; opaque recurrent state."""

    @abstractmethod
    def reset(
        self,
        encoder_output: EncoderOutput,
        batch_size: int,
    ) -> Any:
        """Initialize decoder state for a new episode."""

    @abstractmethod
    def step(
        self,
        encoder_output: EncoderOutput,
        *,
        prev_embed: torch.Tensor,
        dynamic_hidden: torch.Tensor | None,
        state: Any,
        avail_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        """Return unmasked logits `[B, N]` and updated state."""
