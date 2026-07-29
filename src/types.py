"""Validated tensor containers shared across models, problems, and training."""

from pydantic import BaseModel, ConfigDict


class TensorModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class FrozenTensorModel(TensorModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )


class EvaluationResult(FrozenTensorModel):
    makespan_mean: float
    makespan_std: float
    count: int
