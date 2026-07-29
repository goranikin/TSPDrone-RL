from abc import ABC, abstractmethod
from typing import Any

import torch
from src.types import ProblemState, SupervisedTarget

from src.constants import ObjectiveSense, ProblemName, SupervisionKind


class Problem(ABC):
    name: ProblemName
    objective_sense: ObjectiveSense
    supervision_kind: SupervisionKind = "sequence"
    context_dim = 4
    attention_context_dim = 4
    attention_uses_first_node = True
    attention_has_extra_stop = False
    depot_input_dim: int | None = None

    @abstractmethod
    def build_features(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return node features, optional adjacency, optional edge features."""

    @abstractmethod
    def make_state(self, batch: dict[str, Any]) -> ProblemState:
        """Create the initial decoding state."""

    @abstractmethod
    def get_mask(self, state: ProblemState) -> torch.Tensor:
        """Return a boolean invalid-action mask with shape [B, action_count]."""

    @abstractmethod
    def step(self, state: ProblemState, action: torch.Tensor) -> ProblemState:
        """Apply one action to the problem state."""

    @abstractmethod
    def is_done(self, state: ProblemState) -> torch.Tensor:
        """Return a boolean done vector with shape [B]."""

    @abstractmethod
    def to_solution(self, state: ProblemState) -> dict[str, torch.Tensor]:
        """Convert a state into tensors consumed by objective/metrics code."""

    @abstractmethod
    def compute_objective(
        self,
        batch: dict[str, Any],
        solution: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return objective values. Lower is better for minimization problems."""

    @abstractmethod
    def check_feasible(
        self,
        batch: dict[str, Any],
        solution: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return a boolean feasibility vector."""

    @abstractmethod
    def get_supervised_target(self, batch: dict[str, Any]) -> SupervisedTarget:
        """Return target actions and/or target selected mask."""

    @abstractmethod
    def repair_solution(
        self,
        batch: dict[str, Any],
        scores: torch.Tensor,
        proposed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert independent node scores into a feasible solution."""

    def context_features(self, state: ProblemState) -> torch.Tensor:
        selected_ratio = state.selected_mask.float().mean(dim=1)
        done = state.done.float()
        zeros = torch.zeros_like(selected_ratio)
        return torch.stack([selected_ratio, 1.0 - selected_ratio, zeros, done], dim=1)

    def attention_context_features(self, state: ProblemState) -> torch.Tensor:
        """Return scalar state used by the Kool-style attention query.

        Recurrent decoders keep using ``context_features``. Routing problems
        override this method to match the problem-specific decoder contexts in
        Kool et al.; the generic subset problems retain the existing context.
        """
        return self.context_features(state)

    def target_value(self, batch: dict[str, Any]) -> torch.Tensor | None:
        value = batch.get("target_value")
        return value.float() if isinstance(value, torch.Tensor) else None

    def reward(self, objective: torch.Tensor) -> torch.Tensor:
        if self.objective_sense == "min":
            return -objective
        return objective

    @staticmethod
    def stack_actions(state: ProblemState) -> torch.Tensor:
        return state.stacked_actions()

    @staticmethod
    def apply_done_mask(
        mask: torch.Tensor,
        done: torch.Tensor,
        safe_action: int,
    ) -> torch.Tensor:
        if not done.any():
            return mask
        done_mask = torch.ones_like(mask)
        done_mask[:, safe_action] = False
        return torch.where(done.unsqueeze(1), done_mask, mask)

    @staticmethod
    def append_action(state: ProblemState, action: torch.Tensor) -> None:
        state.actions.append(action.long())
        state.step_count += 1
        active = ~state.done
        state.prev_action = torch.where(active, action.long(), state.prev_action)
        state.first_action = torch.where(
            (state.first_action < 0) & active,
            action.long(),
            state.first_action,
        )

    @staticmethod
    def require_tensor(batch: dict[str, Any], key: str) -> torch.Tensor:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Missing tensor batch[{key!r}], got {type(value)}")
        return value

    @staticmethod
    def target_actions(batch: dict[str, Any]) -> torch.Tensor | None:
        actions = batch.get("target_actions")
        return actions.long() if isinstance(actions, torch.Tensor) else None

    @staticmethod
    def target_mask(batch: dict[str, Any]) -> torch.Tensor | None:
        mask = batch.get("target_mask")
        return mask.float() if isinstance(mask, torch.Tensor) else None
