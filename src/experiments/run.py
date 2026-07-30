"""Hydra entrypoint for TSP-D RL training and evaluation."""

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import DictConfig

from src.config import RunConfig, parse_config
from src.logs import configure_file_logger
from src.models.parameter_budget import MatchedDimensions, resolve_matched_dimensions
from src.models.policy import Policy, build_policy
from src.paths import (
    DEFAULT_TRAINED_MODELS_DIR,
    checkpoint_dir,
    experiment_log_path,
    resolve_user_path,
)
from src.problems.tspd import DataGenerator, Env
from src.training import wandb_support
from src.training.trainer import Trainer


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def resolve_model_dimensions(cfg: RunConfig) -> MatchedDimensions:
    if not cfg.parameter_budget.enabled:
        params = sum(
            parameter.numel()
            for parameter in build_policy(
                decoder=cfg.decoder,
                dynamics=cfg.dynamics,
                hidden_dim=cfg.model.hidden_dim,
                n_heads=cfg.model.n_heads,
                n_encode_layers=cfg.model.n_encode_layers,
                d_ff=cfg.model.d_ff,
                dropout=cfg.model.dropout,
                num_layers=cfg.model.num_layers,
                use_tanh=cfg.model.use_tanh,
                tanh_clip=cfg.model.tanh_clip,
                mask_logits=cfg.model.mask_logits,
            ).parameters()
        )
        return MatchedDimensions(
            decoder=cfg.decoder,
            dynamics=cfg.dynamics,
            hidden_dim=cfg.model.hidden_dim,
            d_ff=cfg.model.d_ff,
            matched_params=params,
            target_params=params,
            base_hidden_dim=cfg.model.hidden_dim,
            base_d_ff=cfg.model.d_ff,
            delta=0,
            delta_pct=0.0,
            source="disabled",
        )
    return resolve_matched_dimensions(
        decoder=cfg.decoder,
        dynamics=cfg.dynamics,
        n_heads=cfg.model.n_heads,
        n_encode_layers=cfg.model.n_encode_layers,
        dropout=cfg.model.dropout,
        num_layers=cfg.model.num_layers,
        use_tanh=cfg.model.use_tanh,
        tanh_clip=cfg.model.tanh_clip,
        mask_logits=cfg.model.mask_logits,
        base_hidden_dim=cfg.model.hidden_dim,
        base_d_ff=cfg.model.d_ff,
        match_target=cfg.parameter_budget.match_target,
        max_delta_pct=cfg.parameter_budget.max_delta_pct,
        strict=cfg.parameter_budget.strict,
        min_hidden_dim=cfg.parameter_budget.min_hidden_dim,
        max_hidden_dim=cfg.parameter_budget.max_hidden_dim,
        hidden_dim_step=cfg.parameter_budget.hidden_dim_step,
    )


def run_from_config(raw_cfg: DictConfig) -> dict[str, Any]:
    cfg = parse_config(RunConfig, raw_cfg)
    _require_cuda()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(cfg.seed, device_specific=True)
    device = accelerator.device
    if device.type != "cuda":
        raise RuntimeError(
            f"CUDA is required, but Accelerator selected device={device}. "
            "Check CUDA_VISIBLE_DEVICES and that accelerate is launching with GPUs."
        )

    output_dir = resolve_output_dir(cfg)
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    log_path = experiment_log_path("run.log", data_root=cfg.data.root)
    logger = (
        configure_file_logger("run", log_path) if accelerator.is_main_process else None
    )

    data_gen = DataGenerator(cfg)
    test_data = data_gen.get_test_all()
    env = Env(cfg, test_data)

    matched = resolve_model_dimensions(cfg)
    if accelerator.is_main_process:
        print(
            f"parameter_budget enabled={cfg.parameter_budget.enabled} "
            f"target={matched.target_params:,} matched={matched.matched_params:,} "
            f"delta_pct={matched.delta_pct:+.3f}% "
            f"hidden_dim={matched.hidden_dim} d_ff={matched.d_ff} "
            f"source={matched.source}"
        )

    policy = build_policy(
        decoder=cfg.decoder,
        dynamics=cfg.dynamics,
        hidden_dim=matched.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_encode_layers=cfg.model.n_encode_layers,
        d_ff=matched.d_ff,
        dropout=cfg.model.dropout,
        num_layers=cfg.model.num_layers,
        use_tanh=cfg.model.use_tanh,
        tanh_clip=cfg.model.tanh_clip,
        mask_logits=cfg.model.mask_logits,
    ).to(device)

    load_dir = resolve_checkpoint_load_dir(cfg, output_dir)
    if cfg.data.load_checkpoint:
        _maybe_load_weights(policy, cfg, load_dir, device)

    wandb_logging = False
    if accelerator.is_main_process:
        wandb_config = wandb_support.build_wandb_config(
            cfg=cfg,
            actor=policy,
            output_dir=output_dir,
            resolved_device=str(device),
            matched=matched,
        )
        wandb_config["run"]["num_processes"] = accelerator.num_processes
        wandb_config["run"]["mixed_precision"] = cfg.trainer.mixed_precision
        wandb_logging = wandb_support.init_from_config(
            cfg,
            output_dir=output_dir,
            run_name=cfg.wandb.name,
            default_tags=[
                cfg.problem,
                cfg.architecture,
                cfg.decoder,
                f"dynamics-{cfg.dynamics}",
                cfg.mode,
                cfg.action,
                "param-matched" if cfg.parameter_budget.enabled else "param-unmatched",
            ],
            config=wandb_config,
        )

    trainer = Trainer(
        policy=policy,
        cfg=cfg,
        env=env,
        data_gen=data_gen,
        accelerator=accelerator,
        output_dir=output_dir,
        logger=logger,
        wandb_log=wandb_logging,
    )

    summary = (
        f"run=problem={cfg.problem} architecture={cfg.architecture} "
        f"decoder={cfg.decoder} dynamics={cfg.dynamics} "
        f"action={cfg.action} n_nodes={cfg.physics.n_nodes} "
        f"hidden_dim={matched.hidden_dim} d_ff={matched.d_ff} "
        f"params={matched.matched_params} baseline=greedy_rollout "
        f"device={device} processes={accelerator.num_processes} "
        f"mixed_precision={cfg.trainer.mixed_precision} "
        f"output_dir={output_dir}"
    )
    if accelerator.is_main_process:
        print(summary)
        if logger is not None:
            logger.info(summary)

    result: dict[str, Any] = {}
    try:
        if cfg.action == "train":
            result = trainer.train()
        elif cfg.action == "sampling":
            best_rewards, times = trainer.sampling_batch(cfg.n_samples)
            result = {
                "best_rewards_mean": float(
                    sum(best_rewards) / max(len(best_rewards), 1)
                ),
                "best_rewards": best_rewards,
                "times": times,
            }
            if accelerator.is_main_process:
                print(f"sampling mean best makespan: {result['best_rewards_mean']}")
        else:
            makespan = trainer.test()
            result = {"test_makespan": makespan}
            if accelerator.is_main_process:
                print(f"test makespan: {makespan}")

        if accelerator.is_main_process:
            result_path = Path(output_dir) / "result.json"
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(_jsonable(result), handle, indent=2, sort_keys=True)
            if logger is not None:
                logger.info("result=%s", json.dumps(_jsonable(result), sort_keys=True))
        return result
    finally:
        if wandb_logging:
            wandb_support.finish()


def _require_cuda() -> None:
    if torch.cuda.is_available():
        return
    cuda_built = getattr(torch.version, "cuda", None)
    raise RuntimeError(
        "CUDA is required for training/eval; CPU fallback is disabled. "
        f"torch.cuda.is_available() is False (torch={torch.__version__}, "
        f"torch.version.cuda={cuda_built}). "
        "Install a PyTorch wheel that matches your NVIDIA driver "
        "(e.g. cu128 when nvidia-smi reports CUDA 12.8), or upgrade the driver."
    )


def resolve_output_dir(cfg: RunConfig) -> str:
    if cfg.paths.output_dir is not None:
        return str(resolve_user_path(cfg.paths.output_dir))
    root = resolve_user_path(cfg.paths.output_root)
    name = cfg.wandb.name or f"{cfg.action}-{cfg.architecture}-n{cfg.physics.n_nodes}"
    return str(root / cfg.scale.name / name)


def resolve_checkpoint_load_dir(cfg: RunConfig, output_dir: str) -> Path:
    if cfg.data.checkpoint_dir is not None:
        return resolve_user_path(cfg.data.checkpoint_dir)
    legacy = DEFAULT_TRAINED_MODELS_DIR / f"n{cfg.physics.n_nodes}"
    if (
        cfg.architecture == "tspd_lstm_on"
        and (legacy / "best_model_actor_truck_params.pkl").exists()
    ):
        return legacy
    return checkpoint_dir(output_dir, cfg.physics.n_nodes)


def _maybe_load_weights(
    policy: Policy,
    cfg: RunConfig,
    load_dir: Path,
    device: torch.device,
) -> None:
    actor_path = load_dir / "best_model_actor_truck_params.pkl"
    if not actor_path.exists():
        print(f"No actor checkpoint found under {load_dir}")
        return
    try:
        state = torch.load(actor_path, map_location=device, weights_only=True)
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"Checkpoint under {load_dir} is incompatible with "
                f"architecture={cfg.architecture} "
                f"(missing={len(missing)} unexpected={len(unexpected)}); "
                "starting from scratch"
            )
            return
        print(f"Successfully loaded policy weights from {load_dir}")
    except Exception as exc:  # noqa: BLE001
        print(
            f"Failed to load checkpoint from {load_dir}: {exc}; starting from scratch"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    main()
