import torch
import torch.nn.functional as F
from torch import nn

from src.constants import DecodeType
from src.types import EncoderOutput, ProblemDecodeState, SolutionOutput


class SigmoidSubsetDecoder(nn.Module):
    """Independent Bernoulli scores followed by problem-specific repair."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, 1)

    def logits(self, encoder_output: EncoderOutput) -> torch.Tensor:
        return self.classifier(encoder_output.node_embeddings).squeeze(-1)

    def decode(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        decode_type: DecodeType,
    ) -> SolutionOutput:
        logits = self.logits(encoder_output)
        log_probabilities = None
        proposed_mask = None
        if decode_type == "sampling":
            probabilities = torch.sigmoid(logits)
            proposed_mask = torch.bernoulli(probabilities).bool()
            log_probabilities = (
                proposed_mask.float() * F.logsigmoid(logits)
                + (~proposed_mask).float() * F.logsigmoid(-logits)
            ).sum(dim=1)
        elif decode_type == "greedy":
            proposed_mask = logits > 0

        solution = problem_state.problem.repair_solution(
            problem_state.batch,
            logits,
            proposed_mask,
        )
        objective = problem_state.problem.compute_objective(
            problem_state.batch,
            solution,
        )
        feasible = problem_state.problem.check_feasible(
            problem_state.batch,
            solution,
        )
        return SolutionOutput(
            actions=solution["actions"],
            log_probs=log_probabilities,
            selected_mask=solution.get("selected_mask"),
            objective=objective,
            feasible=feasible,
            logits=logits,
            reward=problem_state.problem.reward(objective),
        )
