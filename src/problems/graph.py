from typing import Any

import torch

from src.problems.base import Problem
from src.types import ProblemState, SupervisedTarget


def valid_node_mask(batch: dict[str, Any]) -> torch.Tensor | None:
    num_nodes = batch.get("num_nodes")
    adjacency = batch.get("adjacency")
    if not isinstance(num_nodes, torch.Tensor) or not isinstance(
        adjacency, torch.Tensor
    ):
        return None
    max_nodes = adjacency.size(1)
    positions = torch.arange(max_nodes, device=adjacency.device).unsqueeze(0)
    return positions < num_nodes.to(adjacency.device).unsqueeze(1)


class GraphSubsetProblem(Problem):
    name = "mis"
    objective_sense = "max"
    supervision_kind = "set"
    attention_has_extra_stop = True

    def build_features(self, batch: dict[str, Any]):
        adjacency = self.require_tensor(batch, "adjacency").bool()
        num_nodes = batch.get("num_nodes")
        if isinstance(num_nodes, torch.Tensor):
            denominator = (
                num_nodes.to(device=adjacency.device, dtype=torch.float32) - 1
            ).clamp_min(1.0)
        else:
            denominator = torch.full(
                (adjacency.size(0),),
                max(adjacency.size(1) - 1, 1),
                dtype=torch.float32,
                device=adjacency.device,
            )
        degree = adjacency.float().sum(dim=-1) / denominator.unsqueeze(1)
        return degree.unsqueeze(-1), adjacency.float(), None

    def make_state(self, batch: dict[str, Any]) -> ProblemState:
        adjacency = self.require_tensor(batch, "adjacency")
        batch_size, node_count, _ = adjacency.shape
        device = adjacency.device
        available = torch.ones(batch_size, node_count, dtype=torch.bool, device=device)
        valid = valid_node_mask(batch)
        if valid is not None:
            available &= valid
        return ProblemState(
            batch=batch,
            selected_mask=torch.zeros(
                batch_size, node_count, dtype=torch.bool, device=device
            ),
            done=torch.zeros(batch_size, dtype=torch.bool, device=device),
            prev_action=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            first_action=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            aux={"available": available},
        )

    def get_mask(self, state: ProblemState) -> torch.Tensor:
        stop = state.selected_mask.size(1)
        node_mask = ~state.aux["available"]
        no_available = node_mask.all(dim=1)
        state.done = state.done | no_available
        mask = torch.cat(
            [
                node_mask,
                torch.zeros(state.batch_size, 1, dtype=torch.bool, device=state.device),
            ],
            dim=1,
        )
        return self.apply_done_mask(mask, state.done, stop)

    def step(self, state: ProblemState, action: torch.Tensor) -> ProblemState:
        active = ~state.done
        node_count = state.selected_mask.size(1)
        stop = node_count
        action = action.clamp(min=0, max=stop).long()
        rows = torch.arange(state.batch_size, device=state.device)
        node_active = active & (action != stop)
        if node_active.any():
            state.selected_mask[rows[node_active], action[node_active]] = True
            self._update_available(state, node_active, action)
        self.append_action(state, action)
        state.done = state.done | (active & (action == stop))
        return state

    def _update_available(
        self,
        state: ProblemState,
        node_active: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        adjacency = self.require_tensor(state.batch, "adjacency").bool()
        rows = torch.arange(state.batch_size, device=state.device)
        active_rows = rows[node_active]
        active_actions = action[node_active]
        state.aux["available"][active_rows] &= ~adjacency[active_rows, active_actions]
        state.aux["available"][active_rows, active_actions] = False

    def is_done(self, state: ProblemState) -> torch.Tensor:
        return state.done

    def context_features(self, state: ProblemState) -> torch.Tensor:
        selected_ratio = state.selected_mask.float().mean(dim=1)
        available_ratio = state.aux["available"].float().mean(dim=1)
        return torch.stack(
            [
                selected_ratio,
                available_ratio,
                torch.zeros_like(selected_ratio),
                state.done.float(),
            ],
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
        selected = solution["selected_mask"].float()
        valid = valid_node_mask(batch)
        if valid is not None:
            selected = selected * valid.float()
        return selected.sum(dim=1)

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = solution["selected_mask"].bool()
        selected_pairs = selected.unsqueeze(-1) & selected.unsqueeze(-2)
        conflicts = torch.triu(adjacency & selected_pairs, diagonal=1).any(dim=(-2, -1))
        valid = valid_node_mask(batch)
        invalid_selection = (
            (selected & ~valid).any(dim=1)
            if valid is not None
            else torch.zeros_like(conflicts)
        )
        return ~(conflicts | invalid_selection)

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
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = torch.zeros_like(scores, dtype=torch.bool)
        actions = []
        for row in range(scores.size(0)):
            available = torch.ones(
                scores.size(1), dtype=torch.bool, device=scores.device
            )
            row_actions: list[int] = []
            for node in _candidate_order(
                scores[row], proposed_mask[row] if proposed_mask is not None else None
            ):
                if bool(available[node]):
                    selected[row, node] = True
                    row_actions.append(node)
                    available &= ~adjacency[row, node]
                    available[node] = False
            row_actions.append(scores.size(1))
            actions.append(_pad_actions(row_actions, scores.size(1) + 1, scores.device))
        return {"actions": torch.stack(actions), "selected_mask": selected}


class MaxCliqueProblem(GraphSubsetProblem):
    name = "max_clique"
    objective_sense = "max"

    def _update_available(
        self,
        state: ProblemState,
        node_active: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        adjacency = self.require_tensor(state.batch, "adjacency").bool()
        rows = torch.arange(state.batch_size, device=state.device)
        active_rows = rows[node_active]
        active_actions = action[node_active]
        state.aux["available"][active_rows] &= adjacency[active_rows, active_actions]
        state.aux["available"] &= ~state.selected_mask

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = solution["selected_mask"].bool()
        selected_pairs = selected.unsqueeze(-1) & selected.unsqueeze(-2)
        eye = torch.eye(
            selected.size(1), dtype=torch.bool, device=selected.device
        ).unsqueeze(0)
        missing_edge = selected_pairs & ~adjacency & ~eye
        valid = valid_node_mask(batch)
        invalid_selection = (
            (selected & ~valid).any(dim=1)
            if valid is not None
            else torch.zeros(selected.size(0), dtype=torch.bool, device=selected.device)
        )
        return ~(missing_edge.any(dim=(-2, -1)) | invalid_selection)

    def repair_solution(
        self,
        batch: dict[str, Any],
        scores: torch.Tensor,
        proposed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = torch.zeros_like(scores, dtype=torch.bool)
        actions = []
        for row in range(scores.size(0)):
            available = torch.ones(
                scores.size(1), dtype=torch.bool, device=scores.device
            )
            row_actions: list[int] = []
            for node in _candidate_order(
                scores[row], proposed_mask[row] if proposed_mask is not None else None
            ):
                if bool(available[node]):
                    selected[row, node] = True
                    row_actions.append(node)
                    available &= adjacency[row, node]
                    available &= ~selected[row]
            row_actions.append(scores.size(1))
            actions.append(_pad_actions(row_actions, scores.size(1) + 1, scores.device))
        return {"actions": torch.stack(actions), "selected_mask": selected}


class VertexCoverProblem(GraphSubsetProblem):
    name = "vertex_cover"
    objective_sense = "min"

    def make_state(self, batch: dict[str, Any]) -> ProblemState:
        state = super().make_state(batch)
        adjacency = self._effective_adjacency(batch)
        state.aux["remaining_edges"] = (adjacency.sum(dim=(1, 2)) // 2).long()
        return state

    def get_mask(self, state: ProblemState) -> torch.Tensor:
        node_count = state.selected_mask.size(1)
        all_covered = state.aux["remaining_edges"] <= 0
        state.done = state.done | all_covered
        node_mask = state.selected_mask.clone()
        stop_mask = (~all_covered).unsqueeze(1)
        mask = torch.cat([node_mask, stop_mask], dim=1)
        return self.apply_done_mask(mask, state.done, node_count)

    def _update_available(
        self,
        state: ProblemState,
        node_active: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        adjacency = self._effective_adjacency(state.batch)
        rows = torch.arange(state.batch_size, device=state.device)
        active_rows = rows[node_active]
        active_actions = action[node_active]
        selected_before = state.selected_mask[active_rows].clone()
        row_indices = torch.arange(active_rows.size(0), device=state.device)
        selected_before[row_indices, active_actions] = False
        newly_covered = adjacency[active_rows, active_actions, :] & ~selected_before
        state.aux["remaining_edges"][active_rows] -= newly_covered.sum(dim=1).long()
        state.aux["available"] &= ~state.selected_mask

    @staticmethod
    def _effective_adjacency(batch: dict[str, Any]) -> torch.Tensor:
        adjacency = batch["adjacency"]
        if not isinstance(adjacency, torch.Tensor):
            raise ValueError("Missing tensor batch['adjacency']")
        adjacency = adjacency.bool()
        valid = valid_node_mask(batch)
        if valid is not None:
            within = valid.unsqueeze(-1) & valid.unsqueeze(-2)
            adjacency = adjacency & within
        return adjacency

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = solution["selected_mask"].bool()
        covered = selected.unsqueeze(-1) | selected.unsqueeze(-2)
        return ~((adjacency & ~covered).any(dim=(-2, -1)))

    def repair_solution(
        self,
        batch: dict[str, Any],
        scores: torch.Tensor,
        proposed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Vectorized repair: the previous Python O(batch * n^2) loops with
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = (
            proposed_mask.clone().bool() if proposed_mask is not None else scores > 0
        )
        covered = selected.unsqueeze(-1) | selected.unsqueeze(-2)
        uncovered = torch.triu(adjacency & ~covered, diagonal=1)
        prefer_u = scores.unsqueeze(-1) >= scores.unsqueeze(-2)
        selected = selected | (uncovered & prefer_u).any(dim=-1)
        selected = selected | (uncovered & ~prefer_u).any(dim=-2)

        batch_size, node_count = selected.shape
        rows = torch.arange(batch_size, device=scores.device)
        for node in scores.argsort(dim=-1).unbind(dim=-1):
            was_selected = selected[rows, node]
            if not bool(was_selected.any()):
                continue
            candidate = selected.clone()
            candidate[rows, node] = False
            covered = candidate.unsqueeze(-1) | candidate.unsqueeze(-2)
            still_cover = ~((adjacency & ~covered).any(dim=(-2, -1)))
            drop = was_selected & still_cover
            selected[rows, node] = selected[rows, node] & ~drop

        actions = []
        stop = node_count
        for row in range(batch_size):
            row_actions = (
                torch.nonzero(selected[row], as_tuple=False).flatten().tolist()
            )
            row_actions.append(stop)
            actions.append(_pad_actions(row_actions, stop + 1, scores.device))
        return {"actions": torch.stack(actions), "selected_mask": selected}


def _candidate_order(
    scores: torch.Tensor, proposed_mask: torch.Tensor | None
) -> list[int]:
    if proposed_mask is not None and proposed_mask.any():
        nodes = torch.nonzero(proposed_mask, as_tuple=False).flatten()
        return nodes[scores[nodes].argsort(descending=True)].tolist()
    return [int(node) for node in scores.argsort(descending=True).tolist()]


def _pad_actions(actions: list[int], length: int, device: torch.device) -> torch.Tensor:
    padded = torch.full((length,), -1, dtype=torch.long, device=device)
    padded[: len(actions)] = torch.tensor(actions, dtype=torch.long, device=device)
    return padded
