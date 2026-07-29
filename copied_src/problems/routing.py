from typing import Any

import torch
from src.problems.base import Problem
from src.types import ProblemState, SupervisedTarget


class TSPProblem(Problem):
    name = "tsp"
    objective_sense = "min"
    attention_context_dim = 0

    def build_features(self, batch: dict[str, Any]):
        return self.require_tensor(batch, "loc").float(), None, None

    def make_state(self, batch: dict[str, Any]) -> ProblemState:
        loc = self.require_tensor(batch, "loc")
        batch_size, num_nodes, _ = loc.shape
        device = loc.device
        return ProblemState(
            batch=batch,
            selected_mask=torch.zeros(
                batch_size, num_nodes, dtype=torch.bool, device=device
            ),
            done=torch.zeros(batch_size, dtype=torch.bool, device=device),
            prev_action=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            first_action=torch.full((batch_size,), -1, dtype=torch.long, device=device),
        )

    def get_mask(self, state: ProblemState) -> torch.Tensor:
        return self.apply_done_mask(state.selected_mask.clone(), state.done, 0)

    def step(self, state: ProblemState, action: torch.Tensor) -> ProblemState:
        active = ~state.done
        node_count = state.selected_mask.size(1)
        safe_action = action.clamp(min=0, max=node_count - 1)
        rows = torch.arange(state.batch_size, device=state.device)
        state.selected_mask[rows[active], safe_action[active]] = True
        self.append_action(state, safe_action)
        state.done = state.done | state.selected_mask.all(dim=1)
        return state

    def is_done(self, state: ProblemState) -> torch.Tensor:
        return state.done

    def attention_context_features(self, state: ProblemState) -> torch.Tensor:
        return torch.empty(state.batch_size, 0, device=state.device)

    def to_solution(self, state: ProblemState) -> dict[str, torch.Tensor]:
        return {
            "actions": self.stack_actions(state),
            "selected_mask": state.selected_mask,
        }

    def compute_objective(
        self, batch: dict[str, Any], solution: dict[str, torch.Tensor]
    ):
        loc = self.require_tensor(batch, "loc").float()
        actions = solution["actions"].long()
        if actions.size(1) < loc.size(1):
            pad = torch.arange(loc.size(1), device=loc.device).expand(loc.size(0), -1)
            actions = torch.where(
                torch.arange(loc.size(1), device=loc.device).view(1, -1)
                < actions.size(1),
                torch.nn.functional.pad(actions, (0, loc.size(1) - actions.size(1))),
                pad,
            )
        order = actions[:, : loc.size(1)]
        ordered = loc.gather(1, order.unsqueeze(-1).expand(-1, -1, loc.size(-1)))
        return (ordered[:, 1:] - ordered[:, :-1]).norm(p=2, dim=-1).sum(dim=1) + (
            ordered[:, 0] - ordered[:, -1]
        ).norm(p=2, dim=-1)

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        loc = self.require_tensor(batch, "loc")
        actions = solution["actions"].long()
        if actions.size(1) < loc.size(1):
            return torch.zeros(loc.size(0), dtype=torch.bool, device=loc.device)
        expected = torch.arange(loc.size(1), device=loc.device).expand(loc.size(0), -1)
        return (actions[:, : loc.size(1)].sort(dim=1).values == expected).all(dim=1)

    def get_supervised_target(self, batch: dict[str, Any]) -> SupervisedTarget:
        actions = self.target_actions(batch)
        mask = self.target_mask(batch)
        return SupervisedTarget(actions=actions, selected_mask=mask)

    def repair_solution(
        self,
        batch: dict[str, Any],
        scores: torch.Tensor,
        proposed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del proposed_mask
        return {
            "actions": scores.argsort(dim=1, descending=True),
            "selected_mask": torch.ones_like(scores, dtype=torch.bool),
        }


class CVRPProblem(Problem):
    name = "cvrp"
    objective_sense = "min"
    attention_context_dim = 1
    attention_uses_first_node = False
    depot_input_dim = 2

    def build_features(self, batch: dict[str, Any]):
        loc = self.require_tensor(batch, "loc").float()
        demands = self.require_tensor(batch, "node_demands").float()
        capacity = self.require_tensor(batch, "capacity").float().view(-1, 1)
        features = torch.cat(
            [loc, (demands / capacity.clamp_min(1.0)).unsqueeze(-1)], dim=-1
        )
        return features, None, None

    def make_state(self, batch: dict[str, Any]) -> ProblemState:
        loc = self.require_tensor(batch, "loc")
        capacity = self.require_tensor(batch, "capacity").float()
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
                "remaining_capacity": capacity.clone(),
                "current_node": torch.zeros(
                    batch_size, dtype=torch.long, device=device
                ),
            },
        )

    def get_mask(self, state: ProblemState) -> torch.Tensor:
        demands = self.require_tensor(state.batch, "node_demands").float()
        capacity = self.require_tensor(state.batch, "capacity").float()
        remaining = state.aux["remaining_capacity"]
        served = state.selected_mask
        all_served = served[:, 1:].all(dim=1)
        mask = served.clone()
        mask[:, 0] = (remaining >= capacity - 1e-12) & (~all_served)
        mask[:, 1:] |= demands[:, 1:] > remaining.view(-1, 1) + 1e-12
        return self.apply_done_mask(mask, state.done | all_served, 0)

    def step(self, state: ProblemState, action: torch.Tensor) -> ProblemState:
        active = ~state.done
        node_count = state.selected_mask.size(1)
        action = action.clamp(min=0, max=node_count - 1).long()
        rows = torch.arange(state.batch_size, device=state.device)
        capacity = self.require_tensor(state.batch, "capacity").float()
        demands = self.require_tensor(state.batch, "node_demands").float()
        is_depot = action == 0
        customer_active = active & (~is_depot)
        depot_active = active & is_depot
        if customer_active.any():
            state.selected_mask[rows[customer_active], action[customer_active]] = True
            state.aux["remaining_capacity"][customer_active] -= demands[
                rows[customer_active],
                action[customer_active],
            ]
        if depot_active.any():
            state.aux["remaining_capacity"][depot_active] = capacity[depot_active]
        state.aux["current_node"] = torch.where(
            active, action, state.aux["current_node"]
        )
        self.append_action(state, action)
        state.done = state.done | state.selected_mask[:, 1:].all(dim=1)
        return state

    def is_done(self, state: ProblemState) -> torch.Tensor:
        return state.done

    def context_features(self, state: ProblemState) -> torch.Tensor:
        customer_mask = state.selected_mask[:, 1:]
        served_ratio = customer_mask.float().mean(dim=1)
        remaining = state.aux["remaining_capacity"]
        capacity = self.require_tensor(state.batch, "capacity").float()
        remaining_ratio = remaining / capacity.clamp_min(1.0)
        feasible_customers = (
            (
                (~customer_mask)
                & (
                    self.require_tensor(state.batch, "demands").float()
                    <= remaining.view(-1, 1) + 1e-12
                )
            )
            .float()
            .mean(dim=1)
        )
        return torch.stack(
            [served_ratio, feasible_customers, remaining_ratio, state.done.float()],
            dim=1,
        )

    def attention_context_features(self, state: ProblemState) -> torch.Tensor:
        remaining = state.aux["remaining_capacity"]
        capacity = self.require_tensor(state.batch, "capacity").float()
        return (remaining / capacity.clamp_min(1.0)).unsqueeze(1)

    def to_solution(self, state: ProblemState) -> dict[str, torch.Tensor]:
        return {
            "actions": self.stack_actions(state),
            "selected_mask": state.selected_mask,
        }

    def compute_objective(
        self, batch: dict[str, Any], solution: dict[str, torch.Tensor]
    ):
        loc = self.require_tensor(batch, "loc").float()
        actions = solution["actions"].long()
        batch_size, action_count = actions.shape
        if action_count == 0:
            return torch.zeros(batch_size, dtype=loc.dtype, device=loc.device)
        valid = (actions >= 0) & (actions < loc.size(1))
        safe = actions.clamp(min=0, max=loc.size(1) - 1)
        previous = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
                safe[:, : max(action_count - 1, 0)],
            ],
            dim=1,
        )
        rows = torch.arange(batch_size, device=loc.device).unsqueeze(1)
        distances = (loc[rows, safe] - loc[rows, previous]).norm(p=2, dim=-1)
        total = (distances * valid).sum(dim=1)
        positions = torch.arange(action_count, device=loc.device).unsqueeze(0)
        last_position = torch.where(valid, positions, -1).max(dim=1).values
        has_action = last_position >= 0
        last_action = safe.gather(1, last_position.clamp_min(0).unsqueeze(1)).squeeze(1)
        last_loc = loc[torch.arange(batch_size, device=loc.device), last_action]
        return_home = (last_loc - loc[:, 0]).norm(p=2, dim=-1)
        return total + return_home * has_action

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        loc = self.require_tensor(batch, "loc")
        demands = self.require_tensor(batch, "node_demands").float()
        capacity = self.require_tensor(batch, "capacity").float()
        actions = solution["actions"].long()
        batch_size = loc.size(0)
        rows = torch.arange(batch_size, device=loc.device)
        served = torch.zeros(
            batch_size, loc.size(1), dtype=torch.bool, device=loc.device
        )
        load = torch.zeros(batch_size, dtype=demands.dtype, device=loc.device)
        feasible = torch.ones(batch_size, dtype=torch.bool, device=loc.device)
        active = torch.ones_like(feasible)
        for raw_action in actions.unbind(dim=1):
            in_range = (raw_action >= 0) & (raw_action < loc.size(1))
            step_active = active & (raw_action >= 0)
            feasible &= ~(step_active & ~in_range)
            action = raw_action.clamp(min=0, max=loc.size(1) - 1)
            is_customer = step_active & in_range & (action != 0)
            duplicate = served[rows, action] & is_customer
            next_load = load + demands[rows, action] * is_customer
            overloaded = is_customer & (next_load > capacity + 1e-12)
            feasible &= ~(duplicate | overloaded)
            served[rows, action] |= is_customer
            load = torch.where(step_active & (action == 0), 0.0, next_load)
            active &= ~served[:, 1:].all(dim=1)
        return feasible & served[:, 1:].all(dim=1)

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
        del proposed_mask
        demands = self.require_tensor(batch, "node_demands").float()
        capacity = self.require_tensor(batch, "capacity").float()
        actions: list[torch.Tensor] = []
        max_len = scores.size(1) * 2
        for row in range(scores.size(0)):
            row_actions: list[int] = []
            remaining = float(capacity[row])
            for customer in scores[row, 1:].argsort(descending=True).tolist():
                action = int(customer) + 1
                demand = float(demands[row, action])
                if demand > remaining and row_actions and row_actions[-1] != 0:
                    row_actions.append(0)
                    remaining = float(capacity[row])
                if demand <= remaining:
                    row_actions.append(action)
                    remaining -= demand
            padded = torch.full((max_len,), -1, dtype=torch.long, device=scores.device)
            if row_actions:
                padded[: len(row_actions)] = torch.tensor(
                    row_actions, dtype=torch.long, device=scores.device
                )
            actions.append(padded)
        action_tensor = torch.stack(actions)
        selected = torch.zeros_like(scores, dtype=torch.bool)
        selected[:, 1:] = True
        return {"actions": action_tensor, "selected_mask": selected}
