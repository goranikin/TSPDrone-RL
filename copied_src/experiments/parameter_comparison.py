import csv
import json
import math
from typing import Any, Literal, Self

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import (
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from src.models.encoder.selection import encoders_for_problem
from src.models.model import NCOModel

from src.constants import (
    DECODER_KINDS,
    PROBLEM_NAMES,
    DecoderKind,
    EncoderKind,
    ProblemName,
)
from src.logs import configure_file_logger
from src.paths import experiment_log_path, resolve_user_path

INPUT_DIM_BY_PROBLEM: dict[ProblemName, int] = {
    "tsp": 2,
    "cvrp": 3,
    "orienteering": 3,
    "knapsack": 2,
    "mis": 1,
    "max_clique": 1,
    "vertex_cover": 1,
}


class ParameterComparisonSettings(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    problems: tuple[ProblemName, ...] = Field(default=PROBLEM_NAMES, min_length=1)
    encoders: tuple[EncoderKind, ...] | None = None
    decoders: tuple[DecoderKind, ...] = Field(default=DECODER_KINDS, min_length=1)
    d_model: int = Field(gt=0, validation_alias=AliasPath("model", "d_model"))
    num_layers: int = Field(gt=0, validation_alias=AliasPath("model", "num_layers"))
    num_heads: int = Field(gt=0, validation_alias=AliasPath("model", "num_heads"))
    transformer_pointer_layers: int = Field(
        gt=0,
        validation_alias=AliasPath("model", "transformer_pointer_layers"),
    )
    d_ff: int | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasPath("model", "d_ff"),
    )
    d_ff_multiplier: int = Field(
        gt=0,
        validation_alias=AliasPath("model", "d_ff_multiplier"),
    )
    dropout: float = Field(
        ge=0,
        lt=1,
        validation_alias=AliasPath("model", "dropout"),
    )
    tanh_clip: float = Field(
        gt=0,
        validation_alias=AliasPath("model", "tanh_clip"),
    )
    target_params: int | None = Field(default=None, gt=0)
    max_delta_pct: float = Field(default=0.1, ge=0)
    match_target: Literal["attention_model", "none"]
    min_d_model: int = Field(
        gt=0,
        validation_alias=AliasPath("search", "min_d_model"),
    )
    max_d_model: int = Field(
        gt=0,
        validation_alias=AliasPath("search", "max_d_model"),
    )
    d_model_step: int = Field(
        gt=0,
        validation_alias=AliasPath("search", "d_model_step"),
    )
    format: Literal["markdown", "csv", "json"]
    output: str | None

    @field_validator("problems", mode="before")
    @classmethod
    def default_problems(cls, value: Any) -> Any:
        return PROBLEM_NAMES if value is None else value

    @field_validator("decoders", mode="before")
    @classmethod
    def default_decoders(cls, value: Any) -> Any:
        return DECODER_KINDS if value is None else value

    @model_validator(mode="after")
    def validate_widths(self) -> Self:
        if self.max_d_model < self.min_d_model:
            raise ValueError("max_d_model must be greater than or equal to min_d_model")
        candidates = range(
            self.min_d_model,
            self.max_d_model + 1,
            self.d_model_step,
        )
        if not any(width % self.num_heads == 0 for width in candidates):
            raise ValueError("search range contains no width divisible by num_heads")
        if self.d_model % self.num_heads:
            raise ValueError("base d_model must be divisible by num_heads")
        return self


class ParameterRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    problem: ProblemName
    encoder: EncoderKind
    decoder: DecoderKind
    input_dim: int
    base_d_model: int
    base_d_ff: int
    base_params: int
    target_params: int
    matched_d_model: int
    matched_d_ff: int
    matched_params: int
    delta: int
    delta_pct: float
    command_args: str


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="parameter_comparison",
)
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def run_from_config(cfg: DictConfig) -> str:
    settings = settings_from_config(cfg)
    log_path = experiment_log_path("parameter_comparison.log")
    logger = configure_file_logger("parameter_comparison", log_path)
    text = build_parameter_comparison(settings)
    if settings.output:
        output_path = resolve_user_path(settings.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        logger.info("Wrote parameter comparison to %s", output_path)
    else:
        print(text)
    logger.info("result=%s", text.replace("\n", "\\n"))
    return text


def settings_from_config(cfg: DictConfig) -> ParameterComparisonSettings:
    payload = OmegaConf.to_container(cfg, resolve=True)
    return ParameterComparisonSettings.model_validate(payload)


def build_parameter_comparison(settings: ParameterComparisonSettings) -> str:
    rows = [
        parameter_row(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            args=settings,
            target_params=resolve_target(
                settings,
                problem=problem,
                encoder=encoder,
            ),
        )
        for problem in settings.problems
        for encoder in encoders_for_problem(problem, settings.encoders)
        for decoder in settings.decoders
    ]
    if settings.match_target != "none" or settings.target_params is not None:
        validate_parameter_tolerance(rows, settings.max_delta_pct)
    return format_rows(rows, args=settings)


def validate_parameter_tolerance(
    rows: list[ParameterRow] | tuple[ParameterRow, ...],
    max_delta_pct: float,
) -> None:
    outside_tolerance = [
        row for row in rows if abs(row.delta_pct) > max_delta_pct + 1e-12
    ]
    if not outside_tolerance:
        return
    examples = ", ".join(
        f"{row.problem}/{row.encoder}/{row.decoder}={row.delta_pct:+.4f}%"
        for row in outside_tolerance[:5]
    )
    raise ValueError(
        "Parameter matching exceeded max_delta_pct="
        f"{max_delta_pct:.4f}% for {len(outside_tolerance)} row(s): {examples}"
    )


def base_parameter_count(
    *,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
    args: ParameterComparisonSettings,
) -> int:
    return count_parameters(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        d_model=args.d_model,
        d_ff=resolve_d_ff(args, args.d_model),
        args=args,
    )


def parameter_row(
    *,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
    args: ParameterComparisonSettings,
    target_params: int,
) -> ParameterRow:
    base_d_ff = resolve_d_ff(args, args.d_model)
    base_params = count_parameters(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        d_model=args.d_model,
        d_ff=base_d_ff,
        args=args,
    )
    matched_d_model, matched_d_ff, matched_params = find_closest_budget(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        target_params=target_params,
        args=args,
    )
    delta = matched_params - target_params
    delta_pct = 100.0 * delta / max(target_params, 1)
    command_args = (
        f"model.d_model={matched_d_model} "
        f"model.d_ff={matched_d_ff} "
        f"model.num_layers={args.num_layers} "
        f"model.num_heads={args.num_heads} "
        f"model.transformer_pointer_layers={args.transformer_pointer_layers}"
    )
    return ParameterRow(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        input_dim=INPUT_DIM_BY_PROBLEM[problem],
        base_d_model=args.d_model,
        base_d_ff=base_d_ff,
        base_params=base_params,
        target_params=target_params,
        matched_d_model=matched_d_model,
        matched_d_ff=matched_d_ff,
        matched_params=matched_params,
        delta=delta,
        delta_pct=delta_pct,
        command_args=command_args,
    )


def find_closest_budget(
    *,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
    target_params: int,
    args: ParameterComparisonSettings,
) -> tuple[int, int, int]:
    if args.match_target == "none" and args.target_params is None:
        d_ff = resolve_d_ff(args, args.d_model)
        params = count_parameters(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            d_model=args.d_model,
            d_ff=d_ff,
            args=args,
        )
        return args.d_model, d_ff, params

    candidates: list[tuple[int, int, int, int, int, int]] = []
    for d_model in range(args.min_d_model, args.max_d_model + 1, args.d_model_step):
        if not valid_d_model(d_model, args.num_heads):
            continue
        default_d_ff = resolve_d_ff(args, d_model)
        for d_ff in candidate_d_ffs(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            d_model=d_model,
            target_params=target_params,
            args=args,
        ):
            params = count_parameters(
                problem=problem,
                encoder=encoder,
                decoder=decoder,
                d_model=d_model,
                d_ff=d_ff,
                args=args,
            )
            candidates.append(
                (
                    abs(params - target_params),
                    abs(d_model - args.d_model),
                    abs(d_ff - default_d_ff),
                    d_model,
                    d_ff,
                    params,
                )
            )
    _, _, _, d_model, d_ff, params = min(candidates)
    return d_model, d_ff, params


def candidate_d_ffs(
    *,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
    d_model: int,
    target_params: int,
    args: ParameterComparisonSettings,
) -> tuple[int, ...]:
    if args.d_ff is not None:
        return (args.d_ff,)
    params_at_one = count_parameters(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        d_model=d_model,
        d_ff=1,
        args=args,
    )
    params_at_two = count_parameters(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        d_model=d_model,
        d_ff=2,
        args=args,
    )
    slope = params_at_two - params_at_one
    if slope <= 0:
        return (resolve_d_ff(args, d_model),)
    estimate = 1 + (target_params - params_at_one) / slope
    values = {
        resolve_d_ff(args, d_model),
        max(1, math.floor(estimate)),
        max(1, math.ceil(estimate)),
    }
    return tuple(sorted(values))


def count_parameters(
    *,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
    d_model: int,
    d_ff: int,
    args: ParameterComparisonSettings,
) -> int:
    model = NCOModel(
        problem=problem,
        encoder_kind=encoder,
        decoder_kind=decoder,
        input_dim=INPUT_DIM_BY_PROBLEM[problem],
        d_model=d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        transformer_pointer_layers=args.transformer_pointer_layers,
        dropout=args.dropout,
        tanh_clip=args.tanh_clip,
    )
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def resolve_target(
    args: ParameterComparisonSettings,
    *,
    problem: ProblemName,
    encoder: EncoderKind,
) -> int:
    if args.target_params is not None:
        return args.target_params
    if args.match_target == "none":
        return base_parameter_count(
            problem=problem,
            encoder=encoder,
            decoder=args.decoders[0],
            args=args,
        )
    return base_parameter_count(
        problem=problem,
        encoder=encoder,
        decoder="attention_model",
        args=args,
    )


def resolve_d_ff(args: ParameterComparisonSettings, d_model: int) -> int:
    if args.d_ff is not None:
        return args.d_ff
    return d_model * args.d_ff_multiplier


def valid_d_model(d_model: int, num_heads: int) -> bool:
    return d_model > 0 and d_model % num_heads == 0


def format_rows(
    rows: list[ParameterRow],
    *,
    args: ParameterComparisonSettings,
) -> str:
    if args.format == "json":
        payload = {
            "settings": {
                "target_params": args.target_params,
                "anchor_decoder": "attention_model",
                "target_scope": "per_problem",
                "base_d_model": args.d_model,
                "base_d_ff": resolve_d_ff(args, args.d_model),
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "transformer_pointer_layers": args.transformer_pointer_layers,
                "d_ff_fixed": args.d_ff,
                "d_ff_multiplier": args.d_ff_multiplier,
                "min_d_model": args.min_d_model,
                "max_d_model": args.max_d_model,
                "d_model_step": args.d_model_step,
                "max_delta_pct": args.max_delta_pct,
            },
            "summary": summary(rows),
            "rows": [row.model_dump(mode="json") for row in rows],
        }
        return json.dumps(payload, indent=2, sort_keys=True)
    if args.format == "csv":
        return format_csv(rows)
    return format_markdown(rows, args=args)


def format_markdown(
    rows: list[ParameterRow],
    *,
    args: ParameterComparisonSettings,
) -> str:
    headers = [
        "problem",
        "encoder",
        "decoder",
        "base params",
        "target",
        "d_model",
        "d_ff",
        "matched params",
        "delta %",
    ]
    lines = [
        "# Parameter Budget",
        "",
        "Parameter target: the canonical Attention Model for each problem.",
        (
            "Enforcement: each whole model must be within "
            f"`{args.max_delta_pct:.4g}%` of its target."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.problem,
                    row.encoder,
                    row.decoder,
                    f"{row.base_params:,}",
                    f"{row.target_params:,}",
                    str(row.matched_d_model),
                    str(row.matched_d_ff),
                    f"{row.matched_params:,}",
                    f"{row.delta_pct:+.2f}",
                ]
            )
            + " |"
        )
    stats = summary(rows)
    lines.extend(
        [
            "",
            "Summary:",
            f"- rows: `{stats['rows']}`",
            f"- max_abs_delta_pct: `{stats['max_abs_delta_pct']:.2f}`",
            f"- mean_abs_delta_pct: `{stats['mean_abs_delta_pct']:.2f}`",
            "",
            "Default training loads the generated JSON budget and resolves these widths "
            "automatically. For a manual row, keep budget validation enabled:",
            "",
            "```bash",
            "uv run python -m src.experiments.run ... "
            "model.d_model=<d_model> model.d_ff=<d_ff> "
            "model.transformer_pointer_layers=<layers> "
            "parameter_budget.enabled=true",
            "```",
        ]
    )
    return "\n".join(lines)


def format_csv(rows: list[ParameterRow]) -> str:
    fieldnames = list(ParameterRow.model_fields) if rows else []
    output = []
    writer = csv.DictWriter(_ListWriter(output), fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.model_dump(mode="json"))
    return "".join(output)


class _ListWriter:
    def __init__(self, output: list[str]) -> None:
        self.output = output

    def write(self, value: str) -> int:
        self.output.append(value)
        return len(value)


def summary(rows: list[ParameterRow]) -> dict[str, float | int]:
    if not rows:
        return {"rows": 0, "max_abs_delta_pct": 0.0, "mean_abs_delta_pct": 0.0}
    abs_delta = [abs(row.delta_pct) for row in rows]
    return {
        "rows": len(rows),
        "max_abs_delta_pct": max(abs_delta),
        "mean_abs_delta_pct": sum(abs_delta) / len(abs_delta),
    }


# uv run python -m src.experiments.parameter_comparison format=markdown
if __name__ == "__main__":
    main()
