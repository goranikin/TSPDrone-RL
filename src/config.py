from typing import Annotated, Any, Literal, Self

from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from src.constants import (
    DecoderKind,
    DynamicsMode,
    ProblemName,
    RunAction,
    TrainingMode,
    architecture_name,
)
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
    """TSP-D physics. ``alpha`` is the drone/truck speed ratio (``v_d = alpha * v_t``)."""

    n_nodes: PositiveInt = 11
    R: PositiveInt = 150
    v_t: PositiveFloat = 1.0
    alpha: PositiveFloat = 2.0
    max_w: PositiveFloat = 2.5

    @computed_field  # type: ignore[prop-decorator]
    @property
    def v_d(self) -> float:
        return self.alpha * self.v_t


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
    d_ff: PositiveInt = 512
    tanh_clip: PositiveFloat = 10.0
    decode_len: PositiveInt = 30


class ParameterBudgetConfig(StrictModel):
    """Match total params to ``tspd_lstm_on`` at ``model.hidden_dim`` / ``model.d_ff``."""

    enabled: bool = True
    match_target: Literal["tspd_lstm_on"] = "tspd_lstm_on"
    max_delta_pct: PositiveFloat = 1.0
    strict: bool = True
    min_hidden_dim: PositiveInt = 128
    max_hidden_dim: PositiveInt = 384
    hidden_dim_step: PositiveInt = 8


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
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: PositiveInt = 1


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
    decoder: DecoderKind
    dynamics: DynamicsMode
    mode: TrainingMode
    action: RunAction
    scale: ScaleConfig
    physics: ProblemConfig
    data: DataConfig
    paths: OutputPathsConfig
    model: ModelConfig
    parameter_budget: ParameterBudgetConfig
    trainer: TrainerConfig
    wandb: WandbConfig
    n_samples: PositiveInt = 5

    @property
    def architecture(self) -> str:
        return architecture_name(self.decoder, self.dynamics)

    @model_validator(mode="after")
    def validate_run_config(self) -> Self:
        if self.wandb.enabled and self.wandb.name is None:
            raise ValueError(
                "wandb.name is required when W&B logging is enabled; pass "
                "wandb.name=<pipeline-name> or wandb.enabled=false"
            )
        if self.model.hidden_dim % self.model.n_heads:
            raise ValueError("model.hidden_dim must be divisible by model.n_heads")
        min_decode = max(round(self.physics.n_nodes * 1.8), 1)
        self.model.decode_len = max(self.model.decode_len, min_decode)
        return self


def parse_config[ConfigT: BaseModel](model: type[ConfigT], cfg: DictConfig) -> ConfigT:
    payload: Any = OmegaConf.to_container(cfg, resolve=True)
    return model.model_validate(payload)
