"""Validated tensor containers shared across models, problems, and training."""

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field


class TensorModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class FrozenTensorModel(TensorModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )


class EncoderOutput(FrozenTensorModel):
    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor


class ProblemState(TensorModel):
    batch: dict[str, Any]
    selected_mask: torch.Tensor
    done: torch.Tensor
    prev_action: torch.Tensor
    first_action: torch.Tensor
    actions: list[torch.Tensor] = Field(default_factory=list)
    step_count: int = 0
    aux: dict[str, torch.Tensor] = Field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.done.size(0))

    @property
    def device(self) -> torch.device:
        return self.done.device

    def stacked_actions(self, pad_value: int = -1) -> torch.Tensor:
        if not self.actions:
            return torch.full(
                (self.batch_size, 0),
                pad_value,
                dtype=torch.long,
                device=self.device,
            )
        return torch.stack(self.actions, dim=1)


class ProblemDecodeState(TensorModel):
    problem: Any
    batch: dict[str, Any]
    state: ProblemState
    target_actions: torch.Tensor | None = None
    target_mask: torch.Tensor | None = None


class SupervisedTarget(FrozenTensorModel):
    actions: torch.Tensor | None = None
    selected_mask: torch.Tensor | None = None


class SolutionOutput(FrozenTensorModel):
    actions: torch.Tensor
    log_probs: torch.Tensor | None
    selected_mask: torch.Tensor | None
    objective: torch.Tensor
    feasible: torch.Tensor
    logits: torch.Tensor | None = None
    reward: torch.Tensor | None = None


def stack_or_empty(
    actions: list[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if actions:
        return torch.stack(actions, dim=1)
    return torch.empty(batch_size, 0, dtype=torch.long, device=device)
