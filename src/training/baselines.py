"""Reward baselines for REINFORCE / policy-gradient training."""

import copy
from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import ttest_rel

from src.models.policy import Policy

GreedyRolloutFn = Callable[[Policy, np.ndarray], torch.Tensor]


class ExponentialMakespanBaseline:
    """EMA of batch-mean makespan (warmup / simple baseline)."""

    def __init__(self, beta: float = 0.8) -> None:
        self.beta = beta
        self.value: torch.Tensor | None = None

    def evaluate(self, makespan: torch.Tensor) -> torch.Tensor:
        mean_makespan = makespan.detach().mean()
        if self.value is None:
            self.value = mean_makespan
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * mean_makespan
        return self.value.to(makespan.device).expand_as(makespan)


class RolloutMakespanBaseline:
    """Frozen greedy policy copy as a per-instance baseline (Kool et al.)."""

    def __init__(self, device: torch.device, alpha: float = 0.05) -> None:
        self.device = device
        self.alpha = alpha
        self.baseline_actor: Policy | None = None

    def init_from(self, policy: Policy) -> None:
        """``policy`` must be an unwrapped ``Policy`` (not DDP)."""
        self.baseline_actor = copy.deepcopy(policy).to(self.device)
        self.baseline_actor.eval()
        self.baseline_actor.set_sample_mode(False)

    @torch.no_grad()
    def evaluate(
        self,
        data: np.ndarray,
        greedy_rollout: GreedyRolloutFn,
    ) -> torch.Tensor:
        if self.baseline_actor is None:
            raise RuntimeError("Rollout baseline is not initialized")
        return greedy_rollout(self.baseline_actor, data).detach()

    @torch.no_grad()
    def maybe_update(
        self,
        policy: Policy,
        baseline_data: np.ndarray,
        greedy_rollout: GreedyRolloutFn,
        *,
        warmup_done: bool,
    ) -> bool:
        """``policy`` must be an unwrapped ``Policy`` (not DDP)."""
        if not warmup_done:
            return False
        if self.baseline_actor is None:
            self.init_from(policy)
            return True

        was_training = policy.training
        policy.eval()
        policy.set_sample_mode(False)
        candidate = greedy_rollout(policy, baseline_data).detach().cpu()
        baseline = self.evaluate(baseline_data, greedy_rollout).detach().cpu()
        policy.train(was_training)

        _, p_value = ttest_rel(candidate.numpy(), baseline.numpy())
        if p_value < self.alpha and candidate.mean() < baseline.mean():
            self.init_from(policy)
            return True
        return False
