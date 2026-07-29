"""Hydra entrypoint for TSP-D RL training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from src.config import RunConfig, parse_config
from src.logs import configure_file_logger
from src.models.actor import Actor
from src.paths import (
    DEFAULT_TRAINED_MODELS_DIR,
    checkpoint_dir,
    experiment_log_path,
    resolve_user_path,
)
from src.problems.tspd import DataGenerator, Env
from src.training import wandb_support
from src.training.trainer import Trainer
from src.utils import resolve_device, set_seed


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def run_from_config(raw_cfg: DictConfig) -> dict[str, Any]:
    cfg = parse_config(RunConfig, raw_cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    output_dir = resolve_output_dir(cfg)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = experiment_log_path("run.log", data_root=cfg.data.root)
    logger = configure_file_logger("run", log_path)

    data_gen = DataGenerator(cfg)
    test_data = data_gen.get_test_all()
    env = Env(cfg, test_data)

    actor = Actor(
        hidden_size=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        mask_logits=cfg.model.mask_logits,
        use_tanh=cfg.model.use_tanh,
        n_heads=cfg.model.n_heads,
        n_encode_layers=cfg.model.n_encode_layers,
    ).to(device)

    load_dir = resolve_checkpoint_load_dir(cfg, output_dir)
    if cfg.data.load_checkpoint:
        _maybe_load_weights(actor, load_dir, device)

    wandb_config = wandb_support.build_wandb_config(
        cfg=cfg,
        actor=actor,
        output_dir=output_dir,
        resolved_device=str(device),
    )
    wandb_logging = wandb_support.init_from_config(
        cfg,
        output_dir=output_dir,
        run_name=cfg.wandb.name,
        default_tags=[cfg.problem, cfg.architecture, cfg.mode, cfg.action],
        config=wandb_config,
    )

    trainer = Trainer(
        actor=actor,
        cfg=cfg,
        env=env,
        data_gen=data_gen,
        device=device,
        output_dir=output_dir,
        logger=logger,
        wandb_log=wandb_logging,
    )

    summary = (
        f"run=problem={cfg.problem} architecture={cfg.architecture} "
        f"action={cfg.action} n_nodes={cfg.physics.n_nodes} "
        f"hidden_dim={cfg.model.hidden_dim} baseline=greedy_rollout "
        f"device={device} output_dir={output_dir}"
    )
    print(summary)
    logger.info(summary)

    result: dict[str, Any]
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
            print(f"sampling mean best makespan: {result['best_rewards_mean']}")
        else:
            makespan = trainer.test()
            result = {"test_makespan": makespan}
            print(f"test makespan: {makespan}")

        result_path = Path(output_dir) / "result.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(result), handle, indent=2, sort_keys=True)
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
    if (legacy / "best_model_actor_truck_params.pkl").exists():
        return legacy
    return checkpoint_dir(output_dir, cfg.physics.n_nodes)


def _maybe_load_weights(
    actor: Actor,
    load_dir: Path,
    device: torch.device,
) -> None:
    actor_path = load_dir / "best_model_actor_truck_params.pkl"
    if actor_path.exists():
        actor.load_state_dict(
            torch.load(actor_path, map_location=device, weights_only=True)
        )
        print(f"Successfully loaded actor weights from {load_dir}")
    else:
        print(f"No actor checkpoint found under {load_dir}")


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
