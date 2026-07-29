import json
from pathlib import Path
from typing import Any, Literal, overload

import hydra
import torch
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field
from src.data import build_dataloader, collate_problem_batch
from src.experiments.parameter_comparison import (
    INPUT_DIM_BY_PROBLEM,
    ParameterComparisonSettings,
    ParameterRow,
    base_parameter_count,
    find_closest_budget,
    resolve_target,
    validate_parameter_tolerance,
)
from src.models.encoder.selection import resolve_encoder_for_problem
from src.models.model import NCOModel
from torch.utils.data import DataLoader

from src.config import ResolvedModelParameters, RunConfig, parse_config
from src.constants import (
    DECODER_KINDS,
    PROBLEM_NAMES,
    DataSplit,
    DecoderKind,
    EncoderKind,
    ProblemName,
)
from src.logs import configure_file_logger
from src.paths import experiment_log_path, problem_dataset_path, resolve_user_path
from src.training.metrics import wandb_metrics
from src.training.trainer import Trainer, TrainingConfig
from src.training.wandb_support import build_wandb_config
from src.training.wandb_support import finish as finish_wandb
from src.training.wandb_support import init_from_config as init_wandb
from src.training.wandb_support import log as wandb_log
from src.utils import resolve_device, set_seed


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def run_from_config(raw_cfg: DictConfig) -> dict[str, Any]:
    cfg = parse_config(RunConfig, raw_cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    problem = cfg.problem
    encoder = resolve_encoder_for_problem(problem, cfg.encoder)
    decoder = cfg.decoder
    target_algorithm = cfg.data.target_algorithm
    train_path = resolve_data_path(cfg, split="train")
    baseline_path = resolve_data_path(cfg, split="baseline")
    val_path = resolve_data_path(cfg, split="val")
    test_path = resolve_data_path(cfg, split="test")
    output_dir = resolve_output_dir(cfg, encoder=encoder)
    log_path = experiment_log_path("run.log", data_root=cfg.data.root)
    logger = configure_file_logger("run", log_path)
    matched_params = resolve_model_parameters(
        cfg,
        problem=problem,
        encoder=encoder,
        decoder=decoder,
    )

    train_loader = build_dataloader(
        train_path,
        problem,
        batch_size=cfg.data.batch_size,
        target_algorithm=target_algorithm,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        stream=cfg.trainer.train_data_policy == "consume_once",
        generator=_loader_generator(cfg.seed, 0),
    )
    train_eval_loader = (
        DataLoader(
            train_loader.dataset,
            batch_size=cfg.data.eval_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_problem_batch,
            generator=_loader_generator(cfg.seed, 1),
        )
        if cfg.wandb.enabled
        else None
    )
    baseline_loader = (
        build_dataloader(
            baseline_path,
            problem,
            batch_size=cfg.data.eval_batch_size,
            target_algorithm=target_algorithm,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            generator=_loader_generator(cfg.seed, 2),
        )
        if baseline_path is not None
        else None
    )
    val_loader = (
        build_dataloader(
            val_path,
            problem,
            batch_size=cfg.data.eval_batch_size,
            target_algorithm=target_algorithm,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            generator=_loader_generator(cfg.seed, 3),
        )
        if val_path is not None
        else None
    )
    test_loader = (
        build_dataloader(
            test_path,
            problem,
            batch_size=cfg.data.eval_batch_size,
            target_algorithm=target_algorithm,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            generator=_loader_generator(cfg.seed, 4),
        )
        if test_path is not None
        else None
    )
    data_provenance = {
        split: _loader_provenance(loader)
        for split, loader in {
            "train": train_loader,
            "baseline": baseline_loader,
            "validation": val_loader,
            "test": test_loader,
        }.items()
        if loader is not None
    }

    model = NCOModel(
        problem=problem,
        encoder_kind=encoder,
        decoder_kind=decoder,
        input_dim=INPUT_DIM_BY_PROBLEM[problem],
        d_model=matched_params.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=matched_params.d_ff,
        transformer_pointer_layers=cfg.model.transformer_pointer_layers,
        dropout=cfg.model.dropout,
        tanh_clip=cfg.model.tanh_clip,
    )
    wandb_config = build_wandb_config(
        cfg=cfg,
        matched_params=matched_params,
        model=model,
        train_path=train_path,
        baseline_path=baseline_path,
        val_path=val_path,
        test_path=test_path,
        target_algorithm=target_algorithm,
        output_dir=output_dir,
        resolved_device=str(device),
        data_provenance=data_provenance,
    )
    wandb_logging = init_wandb(
        cfg,
        output_dir=output_dir,
        run_name=cfg.wandb.name,
        default_tags=[problem, encoder, decoder, cfg.mode],
        config=wandb_config,
    )
    train_config = TrainingConfig(
        mode=cfg.mode,
        epochs=cfg.trainer.epochs,
        steps_per_epoch=cfg.trainer.steps_per_epoch,
        train_data_policy=cfg.trainer.train_data_policy,
        expected_train_instances=cfg.trainer.expected_train_instances,
        learning_rate=cfg.trainer.learning_rate,
        max_grad_norm=cfg.trainer.max_grad_norm,
        baseline=cfg.trainer.baseline,
        baseline_alpha=cfg.trainer.baseline_alpha,
        baseline_warmup_epochs=cfg.trainer.baseline_warmup_epochs,
        exp_baseline_beta=cfg.trainer.exp_baseline_beta,
        log_every=cfg.trainer.log_every,
        progress_bar=cfg.trainer.progress_bar,
        output_dir=output_dir,
        save_checkpoints=cfg.trainer.save_checkpoints,
        wandb_log=wandb_logging,
        wandb_train_eval_batches=cfg.wandb.train_eval_batches,
    )
    run_summary = (
        "run="
        f"problem={problem} encoder={encoder} decoder={decoder} mode={cfg.mode} "
        f"d_model={matched_params.d_model} d_ff={matched_params.d_ff} "
        f"params={matched_params.matched_params} "
        f"target_params={matched_params.target_params} device={device}"
    )
    print(run_summary)
    logger.info(run_summary)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        baseline_loader=baseline_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        logger=logger,
        train_eval_loader=train_eval_loader,
    )
    result = trainer.fit()
    payload: dict[str, Any] = {
        "config": cfg.model_dump(mode="json"),
        "resolved": {
            "encoder": encoder,
            "train_path": train_path,
            "baseline_path": baseline_path,
            "val_path": val_path,
            "test_path": test_path,
            "target_algorithm": target_algorithm,
            "data_provenance": data_provenance,
            "run_provenance": wandb_config["provenance"],
            "output_dir": output_dir,
            "log_path": str(log_path),
            "model": matched_params.model_dump(mode="json"),
        },
        "training_config": train_config.model_dump(mode="json"),
        "training_time_sec": result.training_time_sec,
        "global_steps": trainer.global_step,
        "train_presentations": result.train_presentations,
        "best_validation_objective": result.best_validation_objective,
        "best_validation_feasibility_rate": (
            result.best_validation_feasibility_rate
        ),
        "best_validation_score": result.best_validation_score,
        "best_checkpoint_path": result.best_checkpoint_path,
        "history": result.history,
    }
    if test_loader is not None:
        if result.best_checkpoint_path is not None:
            trainer.load_checkpoint(result.best_checkpoint_path)
            payload["resolved"]["test_checkpoint"] = result.best_checkpoint_path
        else:
            payload["resolved"]["test_checkpoint"] = "last_in_memory"
        test_eval = trainer.evaluate(test_loader, description="test")
        test_metrics = test_eval.to_dict("test")
        payload["test"] = test_metrics
        if train_config.wandb_log:
            wandb_log(wandb_metrics(test_eval, "test"), step=trainer.global_step)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_path = Path(output_dir) / "result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    rendered_payload = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered_payload)
    logger.info("result=%s", json.dumps(payload, sort_keys=True))
    finish_wandb()
    return payload


def _loader_provenance(loader: Any) -> dict[str, Any]:
    provenance = getattr(loader.dataset, "provenance", None)
    if not callable(provenance):
        raise TypeError("Problem dataloaders must expose dataset provenance")
    return provenance()


def _loader_generator(seed: int, offset: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed + offset)


@overload
def resolve_data_path(cfg: RunConfig, *, split: Literal["train"]) -> str: ...


@overload
def resolve_data_path(
    cfg: RunConfig,
    *,
    split: Literal["baseline", "val", "test"],
) -> str | None: ...


def resolve_data_path(cfg: RunConfig, *, split: DataSplit) -> str | None:
    configured = {
        "train": cfg.data.train_path,
        "baseline": cfg.data.baseline_path,
        "val": cfg.data.val_path,
        "test": cfg.data.test_path,
    }[split]
    if configured is not None:
        return str(resolve_user_path(configured))
    if not cfg.data.use_default_paths:
        return None
    profile = cfg.scale.for_mode(cfg.mode)
    split_config = {
        "train": profile.train,
        "baseline": profile.rollout_baseline,
        "val": profile.validation,
        "test": profile.test,
    }[split]
    if split_config is None:
        return None
    return str(
        problem_dataset_path(
            cfg.problem,
            split=split,
            instances=split_config.instances,
            dataset_size=profile.dataset_size,
            labeled=cfg.mode == "supervised",
            data_root=cfg.data.root,
        )
    )


def resolve_output_dir(cfg: RunConfig, *, encoder: EncoderKind) -> str:
    if cfg.paths.output_dir:
        return str(resolve_user_path(cfg.paths.output_dir))
    return (
        f"{cfg.paths.output_root}/seed_{cfg.seed}/{cfg.mode}/"
        f"{cfg.problem}/{encoder}/{cfg.decoder}"
    )


def resolve_model_parameters(
    cfg: RunConfig,
    *,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
) -> ResolvedModelParameters:
    num_layers = cfg.model.num_layers
    num_heads = cfg.model.num_heads
    transformer_pointer_layers = cfg.model.transformer_pointer_layers
    input_dim = INPUT_DIM_BY_PROBLEM[problem]

    if not cfg.parameter_budget.enabled:
        d_model = cfg.model.d_model or cfg.parameter_budget.search.base_d_model
        d_ff = cfg.model.d_ff or (
            cfg.parameter_budget.search.d_ff
            or d_model * cfg.parameter_budget.search.d_ff_multiplier
        )
        matched_params = count_current_params(
            cfg, problem, encoder, decoder, d_model, d_ff
        )
        resolved = finalize_resolved_parameters(
            source="explicit",
            input_dim=input_dim,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            transformer_pointer_layers=transformer_pointer_layers,
            base_d_model=d_model,
            base_d_ff=d_ff,
            base_params=matched_params,
            matched_params=matched_params,
            target_params=matched_params,
            delta=0,
            delta_pct=0.0,
        )
        return resolved
    if cfg.model.d_model is not None and cfg.model.d_ff is not None:
        d_model = cfg.model.d_model
        d_ff = cfg.model.d_ff
        matched_params = count_current_params(
            cfg, problem, encoder, decoder, d_model, d_ff
        )
        settings = parameter_budget_settings(cfg)
        target_params = cfg.parameter_budget.target_params or resolve_target(
            settings,
            problem=problem,
            encoder=encoder,
        )
        base_params = base_parameter_count(
            problem=problem,
            encoder=encoder,
            decoder=decoder,
            args=settings,
        )
        resolved = finalize_resolved_parameters(
            source="explicit_over_budget",
            input_dim=input_dim,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            transformer_pointer_layers=transformer_pointer_layers,
            base_d_model=settings.d_model,
            base_d_ff=settings.d_ff or settings.d_model * settings.d_ff_multiplier,
            base_params=base_params,
            matched_params=matched_params,
            target_params=target_params,
            delta=matched_params - target_params,
            delta_pct=100.0 * (matched_params - target_params) / max(target_params, 1),
        )
        validate_resolved_parameter_tolerance(cfg, resolved)
        return resolved

    row = find_budget_row(
        path=resolve_user_path(cfg.parameter_budget.path),
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        target_params=cfg.parameter_budget.target_params,
        num_layers=num_layers,
        num_heads=num_heads,
        transformer_pointer_layers=transformer_pointer_layers,
    )
    row_source = "file"
    if row is not None:
        current_params = count_current_params(
            cfg,
            problem,
            encoder,
            decoder,
            row.matched_d_model,
            row.matched_d_ff,
        )
        if current_params != row.matched_params:
            # The architecture changed after this generated budget file was
            # written. Recompute instead of silently logging a stale count.
            row = None
    if row is None:
        if cfg.parameter_budget.strict:
            raise FileNotFoundError(
                "No parameter budget row for "
                f"problem={problem}, encoder={encoder}, decoder={decoder} "
                f"in {cfg.parameter_budget.path}"
            )
        row = compute_budget_row(cfg, problem=problem, encoder=encoder, decoder=decoder)
        row_source = "computed"
    resolved = finalize_resolved_parameters(
        source=(
            str(resolve_user_path(cfg.parameter_budget.path))
            if row_source == "file"
            else "computed"
        ),
        input_dim=row.input_dim,
        d_model=row.matched_d_model,
        d_ff=row.matched_d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        transformer_pointer_layers=transformer_pointer_layers,
        base_d_model=row.base_d_model,
        base_d_ff=row.base_d_ff,
        base_params=row.base_params,
        matched_params=row.matched_params,
        target_params=row.target_params,
        delta=row.delta,
        delta_pct=row.delta_pct,
        command_args=row.command_args,
    )
    validate_resolved_parameter_tolerance(cfg, resolved)
    return resolved


def validate_resolved_parameter_tolerance(
    cfg: RunConfig,
    resolved: ResolvedModelParameters,
) -> None:
    row = ParameterRow(
        problem=cfg.problem,
        encoder=resolve_encoder_for_problem(cfg.problem, cfg.encoder),
        decoder=cfg.decoder,
        input_dim=resolved.input_dim,
        base_d_model=resolved.base_d_model or resolved.d_model,
        base_d_ff=resolved.base_d_ff or resolved.d_ff,
        base_params=resolved.base_params or resolved.matched_params,
        target_params=resolved.target_params,
        matched_d_model=resolved.d_model,
        matched_d_ff=resolved.d_ff,
        matched_params=resolved.matched_params,
        delta=resolved.delta,
        delta_pct=resolved.delta_pct,
        command_args=resolved.command_args,
    )
    validate_parameter_tolerance([row], cfg.parameter_budget.max_delta_pct)


def finalize_resolved_parameters(
    *,
    source: str,
    input_dim: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    transformer_pointer_layers: int,
    base_d_model: int | None,
    base_d_ff: int | None,
    base_params: int | None,
    matched_params: int,
    target_params: int,
    delta: int,
    delta_pct: float,
    command_args: str | None = None,
) -> ResolvedModelParameters:
    if command_args is None:
        command_args = build_model_command_args(
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            transformer_pointer_layers=transformer_pointer_layers,
        )
    return ResolvedModelParameters(
        source=source,
        input_dim=input_dim,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        transformer_pointer_layers=transformer_pointer_layers,
        base_d_model=base_d_model,
        base_d_ff=base_d_ff,
        base_params=base_params,
        matched_params=matched_params,
        target_params=target_params,
        delta=delta,
        delta_pct=delta_pct,
        command_args=command_args,
    )


def build_model_command_args(
    *,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    transformer_pointer_layers: int,
) -> str:
    return (
        f"model.d_model={d_model} model.d_ff={d_ff} "
        f"model.num_layers={num_layers} model.num_heads={num_heads} "
        f"model.transformer_pointer_layers={transformer_pointer_layers}"
    )


class BudgetArchitectureSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    num_layers: int | None = None
    num_heads: int | None = None
    transformer_pointer_layers: int | None = None


class ParameterBudgetFile(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    settings: BudgetArchitectureSettings = Field(
        default_factory=BudgetArchitectureSettings
    )
    rows: tuple[ParameterRow, ...] = ()


def find_budget_row(
    *,
    path: Path,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
    target_params: int | None,
    num_layers: int,
    num_heads: int,
    transformer_pointer_layers: int,
) -> ParameterRow | None:
    if not path.is_file():
        return None
    payload = ParameterBudgetFile.model_validate_json(path.read_text(encoding="utf-8"))
    settings = payload.settings
    if settings.num_layers not in (None, num_layers):
        return None
    if settings.num_heads not in (None, num_heads):
        return None
    if settings.transformer_pointer_layers not in (
        None,
        transformer_pointer_layers,
    ):
        return None
    matches = [
        row
        for row in payload.rows
        if row.problem == problem
        and row.encoder == encoder
        and row.decoder == decoder
        and (target_params is None or row.target_params == target_params)
    ]
    return matches[0] if matches else None


def compute_budget_row(
    cfg: RunConfig,
    *,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
) -> ParameterRow:
    settings = parameter_budget_settings(cfg)
    if cfg.parameter_budget.target_params is None:
        target_params = resolve_target(
            settings,
            problem=problem,
            encoder=encoder,
        )
    else:
        target_params = cfg.parameter_budget.target_params
    matched_d_model, matched_d_ff, matched_params = find_closest_budget(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        target_params=target_params,
        args=settings,
    )
    delta = matched_params - target_params
    base_params = base_parameter_count(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        args=settings,
    )
    base_d_ff = settings.d_ff or settings.d_model * settings.d_ff_multiplier
    return ParameterRow(
        problem=problem,
        encoder=encoder,
        decoder=decoder,
        input_dim=INPUT_DIM_BY_PROBLEM[problem],
        base_d_model=settings.d_model,
        base_d_ff=base_d_ff,
        base_params=base_params,
        target_params=target_params,
        matched_d_model=matched_d_model,
        matched_d_ff=matched_d_ff,
        matched_params=matched_params,
        delta=delta,
        delta_pct=100.0 * delta / max(target_params, 1),
        command_args=build_model_command_args(
            d_model=matched_d_model,
            d_ff=matched_d_ff,
            num_layers=settings.num_layers,
            num_heads=settings.num_heads,
            transformer_pointer_layers=settings.transformer_pointer_layers,
        ),
    )


def parameter_budget_settings(cfg: RunConfig) -> ParameterComparisonSettings:
    return ParameterComparisonSettings(
        problems=PROBLEM_NAMES,
        encoders=None,
        decoders=DECODER_KINDS,
        d_model=cfg.parameter_budget.search.base_d_model,
        d_ff=cfg.parameter_budget.search.d_ff,
        d_ff_multiplier=cfg.parameter_budget.search.d_ff_multiplier,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        transformer_pointer_layers=cfg.model.transformer_pointer_layers,
        dropout=cfg.model.dropout,
        tanh_clip=cfg.model.tanh_clip,
        target_params=cfg.parameter_budget.target_params,
        max_delta_pct=cfg.parameter_budget.max_delta_pct,
        match_target="attention_model",
        min_d_model=cfg.parameter_budget.search.min_d_model,
        max_d_model=cfg.parameter_budget.search.max_d_model,
        d_model_step=cfg.parameter_budget.search.d_model_step,
        format="json",
        output=None,
    )


def count_current_params(
    cfg: RunConfig,
    problem: ProblemName,
    encoder: EncoderKind,
    decoder: DecoderKind,
    d_model: int,
    d_ff: int,
) -> int:
    model = NCOModel(
        problem=problem,
        encoder_kind=encoder,
        decoder_kind=decoder,
        input_dim=INPUT_DIM_BY_PROBLEM[problem],
        d_model=d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=d_ff,
        transformer_pointer_layers=cfg.model.transformer_pointer_layers,
        dropout=cfg.model.dropout,
        tanh_clip=cfg.model.tanh_clip,
    )
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


if __name__ == "__main__":
    main()
