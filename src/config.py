from typing import Annotated, Any, Literal, Self

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.constants import ArchitectureKind, ProblemName, RunAction, TrainingMode
from src.paths import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, LOCAL_OUTPUT_ROOT

type PositiveInt = Annotated[int, Field(gt=0)]
type NonNegativeInt = Annotated[int, Field(ge=0)]
type PositiveFloat = Annotated[float, Field(gt=0)]
type Probability = Annotated[float, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScaleConfig(StrictModel):
    name: str
    batch_size: PositiveInt
    epochs: PositiveInt
    test_interval: PositiveInt = 200
    save_interval: PositiveInt = 1000
    test_size: PositiveInt = 100
    n_samples: PositiveInt = 5


class ProblemConfig(StrictModel):
    n_nodes: PositiveInt = 11
    R: PositiveInt = 150
    v_t: PositiveFloat = 1.0
    v_d: PositiveFloat = 2.0
    max_w: PositiveFloat = 2.5


class DataConfig(StrictModel):
    data_dir: str = str(DEFAULT_DATA_DIR)
    results_dir: str = str(DEFAULT_RESULTS_DIR)
    root: str = str(LOCAL_OUTPUT_ROOT.parent)
    load_checkpoint: bool = True
    checkpoint_dir: str | None = None


class OutputPathsConfig(StrictModel):
    output_root: str = str(LOCAL_OUTPUT_ROOT / "training")
    output_dir: str | None = None


class ModelConfig(StrictModel):
    hidden_dim: PositiveInt = 256
    num_layers: PositiveInt = 1
    dropout: Probability = 0.1
    mask_logits: bool = True
    use_tanh: bool = False
    n_heads: PositiveInt = 8
    n_encode_layers: PositiveInt = 3
    decode_len: PositiveInt = 30


class TrainerConfig(StrictModel):
    batch_size: PositiveInt
    epochs: PositiveInt
    actor_lr: PositiveFloat = 1e-4
    max_grad_norm: PositiveFloat = 2.0
    test_interval: PositiveInt = 200
    save_interval: PositiveInt = 1000
    progress_bar: bool = True
    save_checkpoints: bool = True
    log_every: PositiveInt = 25
    baseline_alpha: Probability = 0.05
    baseline_warmup_episodes: NonNegativeInt = 1
    exp_baseline_beta: Probability = 0.8


class WandbConfig(StrictModel):
    enabled: bool = True
    project: str = "tspdrone-rl"
    entity: str | None = "goranikin-my-project"
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: Literal["online", "offline", "disabled"] = "online"

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
    architecture: ArchitectureKind
    mode: TrainingMode
    action: RunAction
    scale: ScaleConfig
    physics: ProblemConfig
    data: DataConfig
    paths: OutputPathsConfig
    model: ModelConfig
    trainer: TrainerConfig
    wandb: WandbConfig
    n_samples: PositiveInt = 5

    @model_validator(mode="after")
    def validate_run_config(self) -> Self:
        if self.wandb.enabled and self.wandb.name is None:
            raise ValueError(
                "wandb.name is required when W&B logging is enabled; pass "
                "wandb.name=<pipeline-name> or wandb.enabled=false"
            )
        min_decode = max(round(self.physics.n_nodes * 1.8), 1)
        self.model.decode_len = max(self.model.decode_len, min_decode)
        return self


def parse_config[ConfigT: BaseModel](model: type[ConfigT], cfg: DictConfig) -> ConfigT:
    payload: Any = OmegaConf.to_container(cfg, resolve=True)
    return model.model_validate(payload)
