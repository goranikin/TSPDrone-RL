import copy
from collections.abc import Iterable

import torch
from scipy.stats import ttest_rel

from src.utils import move_to_device


class ExponentialRewardBaseline:
    def __init__(self, beta: float = 0.8) -> None:
        self.beta = beta
        self.value: torch.Tensor | None = None

    def evaluate(self, reward: torch.Tensor) -> torch.Tensor:
        mean_reward = reward.detach().mean()
        if self.value is None:
            self.value = mean_reward
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * mean_reward
        return self.value.to(reward.device).expand_as(reward)


class RolloutRewardBaseline:
    def __init__(self, device: torch.device, alpha: float = 0.05) -> None:
        self.device = device
        self.alpha = alpha
        self.baseline_model: torch.nn.Module | None = None

    def init_from(self, model: torch.nn.Module) -> None:
        self.baseline_model = copy.deepcopy(model).to(self.device)
        self.baseline_model.eval()

    @torch.no_grad()
    def evaluate_batch(self, batch: dict) -> torch.Tensor:
        if self.baseline_model is None:
            raise RuntimeError("Rollout baseline is not initialized")
        output = self.baseline_model(batch, decode_type="greedy")
        if output.reward is None:
            raise RuntimeError("Model output did not include reward")
        if not bool(output.feasible.all()):
            raise RuntimeError("Rollout baseline produced infeasible solutions")
        return output.reward.detach()

    @torch.no_grad()
    def maybe_update(
        self,
        model: torch.nn.Module,
        val_loader: Iterable[dict],
        *,
        warmup_done: bool,
    ) -> bool:
        if not warmup_done:
            return False
        if self.baseline_model is None:
            self.init_from(model)
            return True
        candidate_rewards = []
        baseline_rewards = []
        was_training = model.training
        model.eval()
        for batch in val_loader:
            batch = move_to_device(batch, self.device)
            candidate_output = model(batch, decode_type="greedy")
            candidate = candidate_output.reward
            baseline = self.evaluate_batch(batch)
            if candidate is None:
                raise RuntimeError("Candidate output did not include reward")
            if not bool(candidate_output.feasible.all()):
                raise RuntimeError("Rollout candidate produced infeasible solutions")
            candidate_rewards.append(candidate.detach().cpu())
            baseline_rewards.append(baseline.detach().cpu())
        model.train(was_training)
        if not candidate_rewards:
            return False
        candidate = torch.cat(candidate_rewards)
        baseline = torch.cat(baseline_rewards)
        _, p_value = ttest_rel(candidate.numpy(), baseline.numpy())
        if p_value < self.alpha and candidate.mean() > baseline.mean():
            self.init_from(model)
            return True
        return False
