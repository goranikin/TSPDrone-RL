from typing import Any

import torch
from pydantic import BaseModel, ConfigDict

from src.problems.base import Problem
from src.types import SolutionOutput


class BatchMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int
    objective: float
    feasibility_rate: float
    inference_time_sec: float
    reference_gap: float | None = None
    reference_gap_pct: float | None = None
    reference_objective: float | None = None


class EvaluationMetrics(BatchMetrics):
    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        key = f"{prefix}/" if prefix else ""
        return {
            f"{key}{name}": value
            for name, value in self.model_dump(exclude_none=True).items()
        }


def batch_metrics(
    problem: Problem,
    batch: dict[str, Any],
    output: SolutionOutput,
    inference_time_sec: float,
) -> BatchMetrics:
    objective = output.objective.detach()
    feasible = output.feasible.detach().float()
    count = int(objective.numel())
    target = problem.target_value(batch)
    reference_gap = None
    reference_gap_pct = None
    reference_mean = None
    if target is not None:
        target = target.to(device=objective.device, dtype=objective.dtype)
        if problem.objective_sense == "min":
            raw_gap = objective - target
        else:
            raw_gap = target - objective
        reference_gap = float(raw_gap.mean().item())
        reference_mean = float(target.mean().item())
        if abs(reference_mean) > 1e-12:
            reference_gap_pct = 100.0 * reference_gap / abs(reference_mean)
    return BatchMetrics(
        count=count,
        objective=float(objective.mean().item()),
        feasibility_rate=float(feasible.mean().item()),
        inference_time_sec=inference_time_sec,
        reference_gap=reference_gap,
        reference_gap_pct=reference_gap_pct,
        reference_objective=reference_mean,
    )


def aggregate_metrics(items: list[BatchMetrics]) -> EvaluationMetrics:
    if not items:
        return EvaluationMetrics(
            count=0,
            objective=0.0,
            feasibility_rate=0.0,
            inference_time_sec=0.0,
        )
    count = sum(item.count for item in items)
    denom = max(count, 1)

    def avg(name: str) -> float:
        return sum(getattr(item, name) * item.count for item in items) / denom

    def optional_avg(name: str) -> float | None:
        present = [item for item in items if getattr(item, name) is not None]
        if not present:
            return None
        present_count = sum(item.count for item in present)
        return sum(float(getattr(item, name)) * item.count for item in present) / max(
            present_count, 1
        )

    reference_gap = optional_avg("reference_gap")
    reference_objective = optional_avg("reference_objective")
    reference_gap_pct = None
    if (
        reference_gap is not None
        and reference_objective is not None
        and abs(reference_objective) > 1e-12
    ):
        reference_gap_pct = 100.0 * reference_gap / abs(reference_objective)

    return EvaluationMetrics(
        count=count,
        objective=avg("objective"),
        feasibility_rate=avg("feasibility_rate"),
        inference_time_sec=sum(item.inference_time_sec for item in items),
        reference_gap=reference_gap,
        reference_gap_pct=reference_gap_pct,
        reference_objective=reference_objective,
    )


def wandb_metrics(
    metrics: BatchMetrics | EvaluationMetrics,
    prefix: str,
    *,
    include_inference_time: bool = True,
) -> dict[str, float | int]:
    base = f"{prefix}/" if prefix else ""
    payload: dict[str, float | int] = {
        f"{base}count": metrics.count,
        f"{base}objective": metrics.objective,
        f"{base}feasibility_rate": metrics.feasibility_rate,
    }
    if include_inference_time:
        payload[f"{base}inference_time_sec"] = metrics.inference_time_sec
    if metrics.reference_gap is not None:
        payload[f"{base}reference_gap"] = metrics.reference_gap
    if metrics.reference_gap_pct is not None:
        payload[f"{base}reference_gap_pct"] = metrics.reference_gap_pct
    if metrics.reference_objective is not None:
        payload[f"{base}reference_objective"] = metrics.reference_objective
    return payload


def wandb_supervised_step_metrics(
    *,
    batch_loss: float,
    window_loss: float,
    metrics: BatchMetrics,
    epoch: int,
) -> dict[str, float | int]:
    return {
        "train/sl/loss": batch_loss,
        "train/sl/loss_batch": batch_loss,
        "train/sl/loss_window": window_loss,
        "train/epoch": epoch,
        **wandb_metrics(metrics, "train/sl", include_inference_time=False),
    }


def wandb_rl_step_metrics(
    *,
    batch_policy_loss: float,
    window_policy_loss: float,
    metrics: BatchMetrics,
    reward: torch.Tensor,
    advantage: torch.Tensor,
    baseline: torch.Tensor,
    epoch: int,
) -> dict[str, float | int]:
    return {
        "train/rl/policy_loss": batch_policy_loss,
        "train/rl/policy_loss_batch": batch_policy_loss,
        "train/rl/policy_loss_window": window_policy_loss,
        "train/rl/reward": float(reward.mean().item()),
        "train/rl/advantage": float(advantage.mean().item()),
        "train/rl/advantage_std": float(advantage.std(unbiased=False).item()),
        "train/rl/baseline": float(baseline.mean().item()),
        "train/epoch": epoch,
        **wandb_metrics(metrics, "train/rl", include_inference_time=False),
    }


def seed_variance(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }
