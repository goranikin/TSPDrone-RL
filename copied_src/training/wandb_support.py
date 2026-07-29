import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import wandb

from src.config import ResolvedModelParameters, RunConfig


def build_wandb_config(
    *,
    cfg: RunConfig,
    matched_params: ResolvedModelParameters,
    model: Any,
    train_path: str,
    baseline_path: str | None,
    val_path: str | None,
    test_path: str | None,
    target_algorithm: str | None,
    output_dir: str,
    resolved_device: str,
    data_provenance: dict[str, Any],
) -> dict[str, Any]:
    profile = cfg.scale.for_mode(cfg.mode)
    updates = cfg.trainer.steps_per_epoch * cfg.trainer.epochs
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    num_layers = matched_params.num_layers
    num_heads = matched_params.num_heads
    transformer_pointer_layers = matched_params.transformer_pointer_layers
    d_model = matched_params.d_model
    d_ff = matched_params.d_ff

    config: dict[str, Any] = {
        "run": {
            "problem": cfg.problem,
            "encoder": model.encoder_kind,
            "decoder": cfg.decoder,
            "mode": cfg.mode,
            "seed": cfg.seed,
            "requested_device": cfg.device,
            "resolved_device": resolved_device,
        },
        "data": {
            "scale": cfg.scale.name,
            "dataset_size": profile.dataset_size,
            "train_instances": cfg.trainer.expected_train_instances,
            "baseline_instances": (
                profile.rollout_baseline.instances
                if profile.rollout_baseline is not None
                else None
            ),
            "validation_instances": profile.validation.instances,
            "test_instances": profile.test.instances,
            "planned_updates": updates,
            "planned_presentations": updates * cfg.data.batch_size,
            "train_data_policy": cfg.trainer.train_data_policy,
            "root": cfg.data.root,
            "train_path": train_path,
            "baseline_path": baseline_path,
            "val_path": val_path,
            "test_path": test_path,
            "target_algorithm": target_algorithm,
            "batch_size": cfg.data.batch_size,
            "eval_batch_size": cfg.data.eval_batch_size,
            "num_workers": cfg.data.num_workers,
            "shuffle": cfg.data.shuffle,
            "provenance": data_provenance,
        },
        "model": {
            "input_dim": matched_params.input_dim,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "transformer_pointer_layers": transformer_pointer_layers,
            "dropout": cfg.model.dropout,
            "tanh_clip": cfg.model.tanh_clip,
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
        "trainer": cfg.trainer.model_dump(mode="json"),
        "paths": {
            "output_dir": output_dir,
            "output_root": cfg.paths.output_root,
        },
    }

    if cfg.parameter_budget.enabled:
        config["parameter_budget"] = {
            "enabled": True,
            "path": cfg.parameter_budget.path,
            "strict": cfg.parameter_budget.strict,
            "max_delta_pct": cfg.parameter_budget.max_delta_pct,
            "target_params_override": cfg.parameter_budget.target_params,
            "search": cfg.parameter_budget.search.model_dump(mode="json"),
            "base": {
                "d_model": matched_params.base_d_model,
                "d_ff": matched_params.base_d_ff,
                "params": matched_params.base_params,
            },
            "matched": {
                "params": matched_params.matched_params,
                "target_params": matched_params.target_params,
                "delta": matched_params.delta,
                "delta_pct": matched_params.delta_pct,
            },
            "source": matched_params.source,
            "command_args": matched_params.command_args,
        }
    else:
        config["parameter_budget"] = {"enabled": False}

    config["provenance"] = _run_provenance(config)

    return config


def _run_provenance(config: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain")
    code_fingerprint, source_file_count = _source_tree_fingerprint()
    return {
        "config_fingerprint": hashlib.sha256(
            f"{canonical}\n{code_fingerprint}".encode()
        ).hexdigest(),
        "code_fingerprint": code_fingerprint,
        "source_file_count": source_file_count,
        "git_commit": commit or None,
        "git_dirty": bool(status),
    }


def _source_tree_fingerprint() -> tuple[str, int]:
    root_value = _git_output("rev-parse", "--show-toplevel")
    files_value = _git_output(
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    digest = hashlib.sha256()
    file_count = 0
    if root_value and files_value:
        root = Path(root_value)
        for relative in sorted(path for path in files_value.split("\0") if path):
            path = root / relative
            if not path.is_file():
                continue
            digest.update(relative.encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1_048_576), b""):
                    digest.update(chunk)
            file_count += 1
    if file_count == 0:
        digest.update(_git_output("rev-parse", "HEAD").encode())
        digest.update(_git_output("status", "--porcelain").encode())
    return digest.hexdigest(), file_count


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def init_from_config(
    cfg: RunConfig,
    *,
    output_dir: str,
    run_name: str | None = None,
    default_tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> bool:
    if not cfg.wandb.enabled:
        return False
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    resolved_run_name = run_name or cfg.wandb.name
    tags = list(cfg.wandb.tags) if cfg.wandb.tags else (default_tags or [])
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=resolved_run_name,
        group=cfg.wandb.group,
        tags=tags,
        mode=cfg.wandb.mode,
        config=config or {},
        dir=output_dir,
    )
    return True


def log(metrics: dict[str, Any], *, step: int | None = None) -> None:
    if wandb.run is None:
        return
    wandb.log(metrics, step=step)


def update_summary(values: dict[str, Any]) -> None:
    if wandb.run is None:
        return
    for key, value in values.items():
        wandb.run.summary[key] = value


def finish() -> None:
    if wandb.run is not None:
        wandb.finish()
