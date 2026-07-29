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


def run_from_config(raw_cfg: DictConfig) -> dict[str, Any]:
    cfg = parse_config(RunConfig, raw_cfg)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(cfg.seed, device_specific=True)
    device = accelerator.device

    output_dir = resolve_output_dir(cfg)
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    log_path = experiment_log_path("run.log", data_root=cfg.data.root)
    logger = (
        configure_file_logger("run", log_path)
        if accelerator.is_main_process
        else None
    )

    data_gen = DataGenerator(cfg)
    test_data = data_gen.get_test_all()
    env = Env(cfg, test_data)

    policy = build_policy(
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
        f"hidden_dim={cfg.model.hidden_dim} baseline=greedy_rollout "
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
    # Encoder port + decoder matrix invalidate legacy paper checkpoints.
    if cfg.architecture != "tspd_lstm_on":
        print(
            f"Skipping checkpoint load for architecture={cfg.architecture} "
            "(only tspd_lstm_on may attempt load)"
        )
        return
    actor_path = load_dir / "best_model_actor_truck_params.pkl"
    if not actor_path.exists():
        print(f"No actor checkpoint found under {load_dir}")
        return
    try:
        state = torch.load(actor_path, map_location=device, weights_only=True)
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"Checkpoint under {load_dir} is incompatible with the new encoder "
                f"(missing={len(missing)} unexpected={len(unexpected)}); "
                "starting from scratch"
            )
            return
        print(f"Successfully loaded policy weights from {load_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load checkpoint from {load_dir}: {exc}; starting from scratch")


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
