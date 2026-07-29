from pydantic import BaseModel, ConfigDict


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int
    makespan: float
    makespan_std: float = 0.0


def makespan_metrics(makespans) -> EvaluationMetrics:
    import numpy as np

    arr = np.asarray(makespans, dtype=np.float64)
    return EvaluationMetrics(
        count=int(arr.size),
        makespan=float(arr.mean()) if arr.size else 0.0,
        makespan_std=float(arr.std()) if arr.size else 0.0,
    )


def wandb_step_metrics(
    *,
    episode: int,
    actor_loss: float,
    train_makespan: float,
    baseline_makespan: float,
    elapsed_sec: float,
) -> dict[str, float | int]:
    return {
        "train/episode": episode,
        "train/actor_loss": actor_loss,
        "train/makespan": train_makespan,
        "train/baseline_makespan": baseline_makespan,
        "train/elapsed_sec": elapsed_sec,
    }
