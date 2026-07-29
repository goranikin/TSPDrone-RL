from typing import Any

import torch
from src.problems.base import Problem
from src.types import ProblemState, SupervisedTarget


class OrienteeringProblem(Problem):
    name = "orienteering"
    objective_sense = "max"
    attention_context_dim = 1
    attention_uses_first_node = False
    depot_input_dim = 2

    def build_features(self, batch: dict[str, Any]):
        loc = self.require_tensor(batch, "loc").float()
        prizes = self.require_tensor(batch, "node_prizes").float()
        scaled = prizes / 100.0
        return torch.cat([loc, scaled.unsqueeze(-1)], dim=-1), None, None

    def make_state(self, batch: dict[str, Any]) -> ProblemState:
        loc = self.require_tensor(batch, "loc")
        batch_size, node_count, _ = loc.shape
        device = loc.device
        return ProblemState(
            batch=batch,
            selected_mask=torch.zeros(
                batch_size, node_count, dtype=torch.bool, device=device
            ),
            done=torch.zeros(batch_size, dtype=torch.bool, device=device),
            prev_action=torch.zeros(batch_size, dtype=torch.long, device=device),
            first_action=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            aux={
                "current_node": torch.zeros(
                    batch_size, dtype=torch.long, device=device
                ),
                "path_length": torch.zeros(batch_size, dtype=loc.dtype, device=device),
            },
        )

    def get_mask(self, state: ProblemState) -> torch.Tensor:
        loc = self.require_tensor(state.batch, "loc").float()
        budget = self.require_tensor(state.batch, "travel_budget").float()
        current = state.aux["current_node"]
        path_length = state.aux["path_length"]
        current_loc = loc[torch.arange(state.batch_size, device=state.device), current]
        step = (loc - current_loc.unsqueeze(1)).norm(p=2, dim=-1)
        return_home = (loc - loc[:, 0:1]).norm(p=2, dim=-1)
        feasible = (
            path_length.view(-1, 1) + step + return_home <= budget.view(-1, 1) + 1e-12
        )
        mask = state.selected_mask.clone() | (~feasible)
        mask[:, 0] = False
        no_feasible_node = mask[:, 1:].all(dim=1)
        state.done = state.done | no_feasible_node
        return self.apply_done_mask(mask, state.done, 0)

    def step(self, state: ProblemState, action: torch.Tensor) -> ProblemState:
        active = ~state.done
        loc = self.require_tensor(state.batch, "loc").float()
        node_count = loc.size(1)
        action = action.clamp(min=0, max=node_count - 1).long()
        rows = torch.arange(state.batch_size, device=state.device)
        current = state.aux["current_node"]
        distance = (loc[rows, action] - loc[rows, current]).norm(p=2, dim=-1)
        node_active = active & (action != 0)
        state.aux["path_length"] = torch.where(
            active,
            state.aux["path_length"] + distance,
            state.aux["path_length"],
        )
        if node_active.any():
            state.selected_mask[rows[node_active], action[node_active]] = True
        state.aux["current_node"] = torch.where(active, action, current)
        self.append_action(state, action)
        state.done = state.done | (active & (action == 0))
        return state

    def is_done(self, state: ProblemState) -> torch.Tensor:
        return state.done

    def context_features(self, state: ProblemState) -> torch.Tensor:
        selected_ratio = state.selected_mask[:, 1:].float().mean(dim=1)
        budget = self.require_tensor(state.batch, "travel_budget").float()
        remaining_budget = 1.0 - state.aux["path_length"] / budget.clamp_min(1e-12)
        loc = self.require_tensor(state.batch, "loc").float()
        current = state.aux["current_node"]
        current_loc = loc[torch.arange(state.batch_size, device=state.device), current]
        step = (loc - current_loc.unsqueeze(1)).norm(p=2, dim=-1)
        return_home = (loc - loc[:, 0:1]).norm(p=2, dim=-1)
        feasible = (
            state.aux["path_length"].view(-1, 1) + step + return_home
            <= budget.view(-1, 1) + 1e-12
        )
        available_ratio = (
            ((~state.selected_mask[:, 1:]) & feasible[:, 1:]).float().mean(dim=1)
        )
        return torch.stack(
            [selected_ratio, available_ratio, remaining_budget, state.done.float()],
            dim=1,
        )

    def attention_context_features(self, state: ProblemState) -> torch.Tensor:
        budget = self.require_tensor(state.batch, "travel_budget").float()
        remaining_budget = budget - state.aux["path_length"]
        return remaining_budget.unsqueeze(1)

    def to_solution(self, state: ProblemState) -> dict[str, torch.Tensor]:
        return {
            "actions": self.stack_actions(state),
            "selected_mask": state.selected_mask,
        }

    def compute_objective(
        self, batch: dict[str, Any], solution: dict[str, torch.Tensor]
    ):
        prizes = self.require_tensor(batch, "node_prizes").float()
        return (solution["selected_mask"].float() * prizes).sum(dim=1)

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        loc = self.require_tensor(batch, "loc").float()
        budget = self.require_tensor(batch, "travel_budget").float()
        actions = solution["actions"].long()
        batch_size = loc.size(0)
        rows = torch.arange(batch_size, device=loc.device)
        current = torch.zeros(batch_size, dtype=torch.long, device=loc.device)
        length = torch.zeros(batch_size, dtype=loc.dtype, device=loc.device)
        visited = torch.zeros(
            batch_size, loc.size(1), dtype=torch.bool, device=loc.device
        )
        feasible = torch.ones(batch_size, dtype=torch.bool, device=loc.device)
        active = torch.ones_like(feasible)
        for raw_action in actions.unbind(dim=1):
            in_range = (raw_action >= 0) & (raw_action < loc.size(1))
            step_active = active & (raw_action >= 0)
            feasible &= ~(step_active & ~in_range)
            action = raw_action.clamp(min=0, max=loc.size(1) - 1)
            customer = step_active & in_range & (action != 0)
            feasible &= ~(visited[rows, action] & customer)
            distance = (loc[rows, action] - loc[rows, current]).norm(p=2, dim=-1)
            length += distance * (step_active & in_range)
            visited[rows, action] |= customer
            current = torch.where(step_active & in_range, action, current)
            active &= ~(step_active & in_range & (action == 0))
        length += (loc[rows, current] - loc[:, 0]).norm(p=2, dim=-1) * active
        return feasible & (length <= budget + 1e-12)

    def get_supervised_target(self, batch: dict[str, Any]) -> SupervisedTarget:
        return SupervisedTarget(
            actions=self.target_actions(batch), selected_mask=self.target_mask(batch)
        )

    def repair_solution(
        self,
        batch: dict[str, Any],
        scores: torch.Tensor,
        proposed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        loc = self.require_tensor(batch, "loc").float()
        budget = self.require_tensor(batch, "travel_budget").float()
        action_rows = []
        selected = torch.zeros_like(scores, dtype=torch.bool)
        max_len = scores.size(1) + 1
        for row in range(scores.size(0)):
            current = 0
            length = 0.0
            row_actions: list[int] = []
            candidates = _candidate_order(
                scores[row],
                proposed_mask[row] if proposed_mask is not None else None,
                skip_zero=True,
            )
            for action in candidates:
                step = float((loc[row, action] - loc[row, current]).norm(p=2))
                return_home = float((loc[row, action] - loc[row, 0]).norm(p=2))
                if length + step + return_home <= float(budget[row]) + 1e-12:
                    row_actions.append(action)
                    selected[row, action] = True
                    length += step
                    current = action
            row_actions.append(0)
            padded = torch.full((max_len,), -1, dtype=torch.long, device=scores.device)
            padded[: len(row_actions)] = torch.tensor(
                row_actions, dtype=torch.long, device=scores.device
            )
            action_rows.append(padded)
        return {"actions": torch.stack(action_rows), "selected_mask": selected}


class KnapsackProblem(Problem):
    name = "knapsack"
    objective_sense = "max"
    supervision_kind = "set"
    attention_has_extra_stop = True

    def build_features(self, batch: dict[str, Any]):
        weights = self.require_tensor(batch, "weights").float()
        values = self.require_tensor(batch, "values").float()
        capacity = self.require_tensor(batch, "capacity").float().view(-1, 1)
        value_scale = values.max(dim=1, keepdim=True).values.clamp_min(1.0)
        features = torch.stack(
            [weights / capacity.clamp_min(1.0), values / value_scale], dim=-1
        )
        return features, None, None

    def make_state(self, batch: dict[str, Any]) -> ProblemState:
        weights = self.require_tensor(batch, "weights")
        capacity = self.require_tensor(batch, "capacity").float()
        batch_size, item_count = weights.shape
        device = weights.device
        return ProblemState(
            batch=batch,
            selected_mask=torch.zeros(
                batch_size, item_count, dtype=torch.bool, device=device
            ),
            done=torch.zeros(batch_size, dtype=torch.bool, device=device),
            prev_action=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            first_action=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            aux={"remaining_capacity": capacity.clone()},
        )

    def get_mask(self, state: ProblemState) -> torch.Tensor:
        weights = self.require_tensor(state.batch, "weights").float()
        remaining = state.aux["remaining_capacity"]
        item_mask = state.selected_mask | (weights > remaining.view(-1, 1) + 1e-12)
        stop = torch.zeros(state.batch_size, 1, dtype=torch.bool, device=state.device)
        mask = torch.cat([item_mask, stop], dim=1)
        no_item = item_mask.all(dim=1)
        state.done = state.done | no_item
        return self.apply_done_mask(mask, state.done, weights.size(1))

    def step(self, state: ProblemState, action: torch.Tensor) -> ProblemState:
        active = ~state.done
        weights = self.require_tensor(state.batch, "weights").float()
        stop = weights.size(1)
        action = action.clamp(min=0, max=stop).long()
        rows = torch.arange(state.batch_size, device=state.device)
        item_active = active & (action != stop)
        if item_active.any():
            state.selected_mask[rows[item_active], action[item_active]] = True
            state.aux["remaining_capacity"][item_active] -= weights[
                rows[item_active], action[item_active]
            ]
        self.append_action(state, action)
        state.done = state.done | (active & (action == stop))
        return state

    def is_done(self, state: ProblemState) -> torch.Tensor:
        return state.done

    def context_features(self, state: ProblemState) -> torch.Tensor:
        selected_ratio = state.selected_mask.float().mean(dim=1)
        capacity = self.require_tensor(state.batch, "capacity").float()
        remaining_ratio = state.aux["remaining_capacity"] / capacity.clamp_min(1.0)
        weights = self.require_tensor(state.batch, "weights").float()
        available = (~state.selected_mask) & (
            weights <= state.aux["remaining_capacity"].view(-1, 1) + 1e-12
        )
        available_ratio = available.float().mean(dim=1)
        return torch.stack(
            [selected_ratio, available_ratio, remaining_ratio, state.done.float()],
            dim=1,
        )

    def to_solution(self, state: ProblemState) -> dict[str, torch.Tensor]:
        return {
            "actions": self.stack_actions(state),
            "selected_mask": state.selected_mask,
        }

    def compute_objective(
        self, batch: dict[str, Any], solution: dict[str, torch.Tensor]
    ):
        values = self.require_tensor(batch, "values").float()
        return (solution["selected_mask"].float() * values).sum(dim=1)

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        weights = self.require_tensor(batch, "weights").float()
        capacity = self.require_tensor(batch, "capacity").float()
        return (solution["selected_mask"].float() * weights).sum(
            dim=1
        ) <= capacity + 1e-12

    def get_supervised_target(self, batch: dict[str, Any]) -> SupervisedTarget:
        return SupervisedTarget(
            actions=self.target_actions(batch), selected_mask=self.target_mask(batch)
        )

    def repair_solution(
        self,
        batch: dict[str, Any],
        scores: torch.Tensor,
        proposed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        weights = self.require_tensor(batch, "weights").float()
        capacity = self.require_tensor(batch, "capacity").float()
        selected = torch.zeros_like(scores, dtype=torch.bool)
        actions = []
        for row in range(scores.size(0)):
            remaining = float(capacity[row])
            row_actions: list[int] = []
            for item in _candidate_order(
                scores[row], proposed_mask[row] if proposed_mask is not None else None
            ):
                weight = float(weights[row, item])
                if weight <= remaining + 1e-12:
                    selected[row, item] = True
                    row_actions.append(item)
                    remaining -= weight
            row_actions.append(scores.size(1))
            padded = torch.full(
                (scores.size(1) + 1,), -1, dtype=torch.long, device=scores.device
            )
            padded[: len(row_actions)] = torch.tensor(
                row_actions, dtype=torch.long, device=scores.device
            )
            actions.append(padded)
        return {"actions": torch.stack(actions), "selected_mask": selected}


def _candidate_order(
    scores: torch.Tensor,
    proposed_mask: torch.Tensor | None,
    *,
    skip_zero: bool = False,
) -> list[int]:
    if proposed_mask is not None and proposed_mask.any():
        candidates = torch.nonzero(proposed_mask, as_tuple=False).flatten()
        if skip_zero:
            candidates = candidates[candidates != 0]
        return candidates[scores[candidates].argsort(descending=True)].tolist()
    start = 1 if skip_zero else 0
    return [
        int(index)
        for index in scores[start:].argsort(descending=True).add(start).tolist()
    ]
