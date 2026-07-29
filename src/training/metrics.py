from pydantic import BaseModel, ConfigDict

from src.types import EvaluationResult


class BatchMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int
    makespan: float
    makespan_std: float = 0.0


class EvaluationMetrics(BatchMetrics):
    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        key = f"{prefix}/" if prefix else ""
        return {
            f"{key}makespan": self.makespan,
            f"{key}makespan_std": self.makespan_std,
            f"{key}count": self.count,
        }


def makespan_metrics(makespans) -> EvaluationMetrics:
    import numpy as np

    arr = np.asarray(makespans, dtype=np.float64)
    return EvaluationMetrics(
        count=int(arr.size),
        makespan=float(arr.mean()) if arr.size else 0.0,
        makespan_std=float(arr.std()) if arr.size else 0.0,
    )


def to_evaluation_result(metrics: EvaluationMetrics) -> EvaluationResult:
    return EvaluationResult(
        makespan_mean=metrics.makespan,
        makespan_std=metrics.makespan_std,
        count=metrics.count,
    )


def wandb_step_metrics(
    *,
    episode: int,
    actor_loss: float,
    critic_loss: float,
    train_makespan: float,
    elapsed_sec: float,
) -> dict[str, float | int]:
    return {
        "train/episode": episode,
        "train/actor_loss": actor_loss,
        "train/critic_loss": critic_loss,
        "train/makespan": train_makespan,
        "train/elapsed_sec": elapsed_sec,
    }
