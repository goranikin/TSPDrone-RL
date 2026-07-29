import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import wandb

from src.config import RunConfig


def build_wandb_config(
    *,
    cfg: RunConfig,
    actor: Any,
    critic: Any,
    output_dir: str,
    resolved_device: str,
) -> dict[str, Any]:
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    config: dict[str, Any] = {
        "run": {
            "problem": cfg.problem,
            "architecture": cfg.architecture,
            "mode": cfg.mode,
            "action": cfg.action,
            "seed": cfg.seed,
            "requested_device": cfg.device,
            "resolved_device": resolved_device,
        },
        "physics": cfg.physics.model_dump(mode="json"),
        "scale": cfg.scale.model_dump(mode="json"),
        "model": {
            **cfg.model.model_dump(mode="json"),
            "actor_params": actor_params,
            "critic_params": critic_params,
            "total_params": actor_params + critic_params,
        },
        "trainer": cfg.trainer.model_dump(mode="json"),
        "data": cfg.data.model_dump(mode="json"),
        "paths": {
            "output_dir": output_dir,
            "output_root": cfg.paths.output_root,
        },
    }
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
