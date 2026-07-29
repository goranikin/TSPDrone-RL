from abc import ABC, abstractmethod

import torch
from src.models.decoding import (
    masked_log_softmax,
    select_action,
    select_set_supervision_action,
)
from src.types import EncoderOutput, ProblemDecodeState, SolutionOutput
from torch import nn

from src.constants import DecodeType


class AutoregressiveDecoder(nn.Module, ABC):
    """Shared masked autoregressive rollout for pointer-style decoders."""

    def decode(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        decode_type: DecodeType,
    ) -> SolutionOutput:
        problem = problem_state.problem
        state = problem_state.state
        target_actions = problem_state.target_actions
        target_mask = problem_state.target_mask
        log_probabilities: list[torch.Tensor] = []
        max_steps = max(1, state.selected_mask.size(1) * 2 + 2)

        for step in range(max_steps):
            done_before = problem.is_done(state)
            if bool(done_before.all()):
                break
            mask = problem.get_mask(state)
            logits = self.step_logits(encoder_output, problem_state, mask)
            if target_mask is not None:
                action, log_probability = select_set_supervision_action(
                    logits,
                    mask,
                    state.selected_mask,
                    target_mask,
                )
            elif target_actions is not None:
                if step >= target_actions.size(1):
                    active_rows = torch.nonzero(~done_before, as_tuple=False).flatten()
                    raise ValueError(
                        "Teacher-forcing target ended before decoding completed for "
                        f"batch rows {active_rows.tolist()}"
                    )
                fallback, _ = select_action(logits, mask, "greedy")
                target = target_actions[:, step].to(logits.device)
                out_of_range = (target < 0) | (target >= logits.size(1))
                safe_target = target.clamp(min=0, max=logits.size(1) - 1).long()
                masked_target = mask.gather(1, safe_target.unsqueeze(1)).squeeze(1)
                invalid_active = (~done_before) & (out_of_range | masked_target)
                invalid_finished = done_before & (target >= 0)
                invalid = invalid_active | invalid_finished
                if bool(invalid.any()):
                    invalid_rows = torch.nonzero(invalid, as_tuple=False).flatten()
                    raise ValueError(
                        "Invalid teacher-forcing action at "
                        f"step {step} for batch rows {invalid_rows.tolist()}; "
                        "active targets must be in range and feasible, and finished "
                        "rows must be padded with -1"
                    )
                action = torch.where(done_before, fallback, safe_target)
                log_probability = masked_log_softmax(logits, mask).gather(
                    1,
                    action.unsqueeze(1),
                )
                log_probability = log_probability.squeeze(1)
            else:
                action, log_probability = select_action(logits, mask, decode_type)
            log_probability = torch.where(
                done_before,
                torch.zeros_like(log_probability),
                log_probability,
            )
            state = problem.step(state, action)
            problem_state.state = state
            log_probabilities.append(log_probability)

        solution = problem.to_solution(state)
        objective = problem.compute_objective(problem_state.batch, solution)
        feasible = problem.check_feasible(problem_state.batch, solution)
        total_log_probability = (
            torch.stack(log_probabilities, dim=1).sum(dim=1)
            if log_probabilities
            else torch.zeros_like(objective)
        )
        return SolutionOutput(
            actions=solution["actions"],
            log_probs=total_log_probability,
            selected_mask=solution.get("selected_mask"),
            objective=objective,
            feasible=feasible,
            reward=problem.reward(objective),
        )

    @abstractmethod
    def step_logits(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return unmasked logits for one decoding step."""
