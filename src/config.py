from typing import Annotated, Any, Literal, Self, TypeVar

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.constants import (
    DECODER_KINDS,
    DatasetSize,
    DecoderKind,
    EncoderKind,
    EncoderSelection,
    MatrixStage,
    ProblemName,
    TrainingDataPolicy,
    TrainingMode,
)
from src.paths import LOCAL_OUTPUT_ROOT, PARAMETER_BUDGET_PATH

type PositiveInt = Annotated[int, Field(gt=0)]
type NonNegativeInt = Annotated[int, Field(ge=0)]
type PositiveFloat = Annotated[float, Field(gt=0)]
type NonNegativeFloat = Annotated[float, Field(ge=0)]
type Probability = Annotated[float, Field(ge=0, le=1)]
type BaselineKind = Literal["rollout", "exponential"]

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitConfig(StrictModel):
    instances: PositiveInt


class TrainingProfileConfig(StrictModel):
    dataset_size: DatasetSize
    train: SplitConfig
    rollout_baseline: SplitConfig | None = None
    validation: SplitConfig
    test: SplitConfig
    batch_size: PositiveInt
    steps_per_epoch: PositiveInt
    epochs: PositiveInt
    train_data_policy: TrainingDataPolicy
    shuffle: bool

    @model_validator(mode="after")
    def validate_training_budget(self) -> Self:
        updates = self.steps_per_epoch * self.epochs
        presentations = self.batch_size * updates
        if self.train_data_policy == "consume_once" and (
            presentations != self.train.instances
        ):
            raise ValueError(
                "consume_once profiles require exactly one instance per presentation: "
                f"batch_size * steps_per_epoch * epochs = {presentations}, "
                f"train.instances = {self.train.instances}"
            )
        if self.train_data_policy == "consume_once" and self.shuffle:
            raise ValueError(
                "consume_once profiles must preserve the externally generated "
                "JSONL stream order"
            )
        natural_steps = (self.train.instances + self.batch_size - 1) // self.batch_size
        if self.train_data_policy == "repeat" and natural_steps != self.steps_per_epoch:
            raise ValueError(
                "repeat profiles must define one complete data pass per epoch: "
                f"expected {natural_steps} steps, got {self.steps_per_epoch}"
            )
        return self


class ScaleConfig(StrictModel):
    name: str
    supervised: TrainingProfileConfig
    rl: TrainingProfileConfig

    def for_mode(self, mode: TrainingMode) -> TrainingProfileConfig:
        return self.supervised if mode == "supervised" else self.rl


class DataConfig(StrictModel):
    root: str
    use_default_paths: bool = True
    train_path: str | None = None
    baseline_path: str | None = None
    val_path: str | None = None
    test_path: str | None = None
    target_algorithm: str | None = None
    batch_size: PositiveInt
    eval_batch_size: PositiveInt
    num_workers: NonNegativeInt = 0
    shuffle: bool = True

    @model_validator(mode="after")
    def validate_explicit_paths(self) -> Self:
        if not self.use_default_paths and self.train_path is None:
            raise ValueError("train_path is required when use_default_paths is false")
        return self


class OutputPathsConfig(StrictModel):
    output_root: str = str(LOCAL_OUTPUT_ROOT / "training")
    output_dir: str | None = None


class ModelConfig(StrictModel):
    d_model: PositiveInt | None = None
    d_ff: PositiveInt | None = None
    num_layers: PositiveInt = 3
    num_heads: PositiveInt = 8
    transformer_pointer_layers: PositiveInt = 1
    dropout: float = Field(default=0, ge=0, lt=1)
    tanh_clip: PositiveFloat = 10

    @model_validator(mode="after")
    def validate_widths(self) -> Self:
        if (self.d_model is None) != (self.d_ff is None):
            raise ValueError("d_model and d_ff must be provided together")
        if self.d_model is not None and self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self


class ParameterSearchConfig(StrictModel):
    base_d_model: PositiveInt = 128
    d_ff: PositiveInt | None = None
    d_ff_multiplier: PositiveInt = 4
    min_d_model: PositiveInt = 16
    max_d_model: PositiveInt = 512
    d_model_step: PositiveInt = 8

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.max_d_model < self.min_d_model:
            raise ValueError("max_d_model must be greater than or equal to min_d_model")
        return self


class ParameterBudgetConfig(StrictModel):
    enabled: bool = False
    path: str = str(PARAMETER_BUDGET_PATH)
    target_params: PositiveInt | None = None
    strict: bool = False
    max_delta_pct: NonNegativeFloat = 0.1
    search: ParameterSearchConfig


class TrainerConfig(StrictModel):
    epochs: PositiveInt
    steps_per_epoch: PositiveInt
    train_data_policy: TrainingDataPolicy
    expected_train_instances: PositiveInt
    learning_rate: PositiveFloat = 1e-4
    max_grad_norm: PositiveFloat = 1
    baseline: BaselineKind = "rollout"
    baseline_alpha: Probability = 0.05
    baseline_warmup_epochs: NonNegativeInt = 1
    exp_baseline_beta: Probability = 0.8
    log_every: PositiveInt = 25
    progress_bar: bool = True
    save_checkpoints: bool = True


class WandbConfig(StrictModel):
    enabled: bool = True
    project: str = "compare-architectures"
    entity: str | None = "goranikin-my-project"
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: Literal["online", "offline", "disabled"] = "online"
    train_eval_batches: PositiveInt = 10

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        invalid = [value for value in values if not 1 <= len(value) <= 64]
        if invalid:
            raise ValueError("W&B tags must contain between 1 and 64 characters")
        return values


class RunConfig(StrictModel):
    seed: int
    device: str
    problem: ProblemName
    encoder: EncoderSelection
    decoder: DecoderKind
    mode: TrainingMode
    scale: ScaleConfig
    data: DataConfig
    paths: OutputPathsConfig
    model: ModelConfig
    parameter_budget: ParameterBudgetConfig
    trainer: TrainerConfig
    wandb: WandbConfig

    @model_validator(mode="after")
    def validate_run_config(self) -> Self:
        if self.wandb.enabled and self.wandb.name is None:
            raise ValueError(
                "wandb.name is required when W&B logging is enabled; pass "
                "wandb.name=<pipeline-name> or wandb.enabled=false"
            )
        if not self.parameter_budget.enabled and self.model.d_model is None:
            raise ValueError(
                "model d_model and d_ff are required when parameter_budget is disabled"
            )
        profile = self.scale.for_mode(self.mode)
        if self.mode == "rl" and self.trainer.baseline == "rollout":
            if self.data.use_default_paths and profile.rollout_baseline is None:
                raise ValueError("RL rollout baseline profile is missing")
            if not self.data.use_default_paths and self.data.baseline_path is None:
                raise ValueError(
                    "baseline_path is required for explicit RL rollout-baseline data"
                )
        presentations = (
            self.data.batch_size
            * self.trainer.steps_per_epoch
            * self.trainer.epochs
        )
        if self.trainer.train_data_policy == "consume_once":
            if presentations != self.trainer.expected_train_instances:
                raise ValueError(
                    "consume_once runs require data.batch_size * "
                    "trainer.steps_per_epoch * trainer.epochs to equal "
                    "trainer.expected_train_instances: "
                    f"{presentations} != {self.trainer.expected_train_instances}"
                )
            if self.data.shuffle:
                raise ValueError("consume_once runs require data.shuffle=false")
        else:
            natural_steps = (
                self.trainer.expected_train_instances + self.data.batch_size - 1
            ) // self.data.batch_size
            if self.trainer.steps_per_epoch != natural_steps:
                raise ValueError(
                    "repeat runs require one complete data pass per epoch: "
                    f"expected {natural_steps} steps for batch size "
                    f"{self.data.batch_size}, got {self.trainer.steps_per_epoch}"
                )
        return self


class MatrixDataConfig(StrictModel):
    root: str
    eval_batch_size: PositiveInt | None = None
    graph_batch_size: PositiveInt | None = None
    graph_eval_batch_size: PositiveInt | None = None
    num_workers: NonNegativeInt = 0


class MatrixParameterBudgetConfig(StrictModel):
    enabled: bool = False
    path: str = str(PARAMETER_BUDGET_PATH)
    strict: bool = False
    max_delta_pct: NonNegativeFloat = 0.1


class MatrixTrainerConfig(StrictModel):
    learning_rate: PositiveFloat = 1e-4


class MatrixConfig(StrictModel):
    stage: MatrixStage = "all"
    problems: tuple[ProblemName, ...] | None = None
    encoders: tuple[EncoderKind, ...] | None = None
    decoders: tuple[DecoderKind, ...] = DECODER_KINDS
    modes: tuple[TrainingMode, ...] = ("supervised", "rl")
    seeds: tuple[int, ...]
    device: str = "auto"
    execute: bool = False
    skip_completed: bool = True
    skip_sigmoid_routing: bool = False
    scale: ScaleConfig
    data: MatrixDataConfig
    paths: OutputPathsConfig
    parameter_budget: MatrixParameterBudgetConfig
    model: ModelConfig
    trainer: MatrixTrainerConfig
    wandb: WandbConfig

    @model_validator(mode="after")
    def validate_matrix_config(self) -> Self:
        if self.execute and self.wandb.enabled and self.wandb.name is None:
            raise ValueError(
                "wandb.name is required when executing a W&B matrix; pass "
                "wandb.name=<pipeline-name> or wandb.enabled=false"
            )
        if not self.parameter_budget.enabled and self.model.d_model is None:
            raise ValueError(
                "model d_model and d_ff are required when parameter_budget is disabled"
            )
        if self.wandb.name is not None and len(self.wandb.name) > 64:
            raise ValueError(
                "matrix wandb.name becomes an experiment tag and cannot exceed "
                "64 characters"
            )
        return self


class ResolvedModelParameters(StrictModel):
    source: str
    input_dim: PositiveInt
    d_model: PositiveInt
    d_ff: PositiveInt
    num_layers: PositiveInt
    num_heads: PositiveInt
    transformer_pointer_layers: PositiveInt
    base_d_model: PositiveInt | None
    base_d_ff: PositiveInt | None
    base_params: PositiveInt | None
    matched_params: PositiveInt
    target_params: PositiveInt
    delta: int
    delta_pct: float
    command_args: str


def parse_config(model: type[ConfigT], cfg: DictConfig) -> ConfigT:
    payload: Any = OmegaConf.to_container(cfg, resolve=True)
    return model.model_validate(payload)
