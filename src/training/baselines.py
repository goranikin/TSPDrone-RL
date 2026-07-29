import copy
from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import ttest_rel

from src.models.actor import Actor

# (actor, instances) -> per-instance makespan tensor [batch]
GreedyRolloutFn = Callable[[Actor, np.ndarray], torch.Tensor]


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
    """Frozen greedy actor copy as a per-instance baseline (Kool et al.)."""

    def __init__(self, device: torch.device, alpha: float = 0.05) -> None:
        self.device = device
        self.alpha = alpha
        self.baseline_actor: Actor | None = None

    def init_from(self, actor: Actor) -> None:
        self.baseline_actor = copy.deepcopy(actor).to(self.device)
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
        actor: Actor,
        baseline_data: np.ndarray,
        greedy_rollout: GreedyRolloutFn,
        *,
        warmup_done: bool,
    ) -> bool:
        if not warmup_done:
            return False
        if self.baseline_actor is None:
            self.init_from(actor)
            return True

        was_training = actor.training
        actor.eval()
        actor.set_sample_mode(False)
        candidate = greedy_rollout(actor, baseline_data).detach().cpu()
        baseline = self.evaluate(baseline_data, greedy_rollout).detach().cpu()
        actor.train(was_training)

        # Makespan: lower is better (unlike reward-maximizing NCO setups).
        _, p_value = ttest_rel(candidate.numpy(), baseline.numpy())
        if p_value < self.alpha and candidate.mean() < baseline.mean():
            self.init_from(actor)
            return True
        return False
