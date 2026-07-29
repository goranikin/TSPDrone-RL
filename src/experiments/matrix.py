import json
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.config import MatrixConfig, ScaleConfig, WandbConfig, parse_config
from src.constants import (
    GRAPH_PROBLEMS,
    PROBLEM_NAMES,
    DecoderKind,
    EncoderKind,
    MatrixStage,
    ProblemName,
    TrainingMode,
)
from src.logs import configure_file_logger
from src.models.encoder.selection import encoders_for_problem
from src.paths import (
    experiment_log_path,
    problem_split_paths,
    resolve_data_root,
    resolve_user_path,
)

STAGES: dict[MatrixStage, tuple[ProblemName, ...]] = {
    "all": PROBLEM_NAMES,
    "routing": ("tsp", "cvrp"),
    "subset": ("knapsack", "mis", "max_clique", "vertex_cover"),
    "hybrid": ("orienteering",),
}


@hydra.main(version_base=None, config_path="../../configs", config_name="matrix")
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def run_from_config(raw_cfg: DictConfig) -> list[list[str]]:
    cfg = parse_config(MatrixConfig, raw_cfg)
    log_path = experiment_log_path("matrix.log", data_root=cfg.data.root)
    logger = configure_file_logger("matrix", log_path)
    problems = resolve_problems(cfg.problems, cfg.stage)
    commands = build_commands(
        problems=problems,
        encoders=cfg.encoders,
        decoders=cfg.decoders,
        modes=cfg.modes,
        seeds=cfg.seeds,
        data_root=resolve_data_root(cfg.data.root),
        scale=cfg.scale,
        output_root=str(resolve_user_path(cfg.paths.output_root)),
        parameter_budget=str(resolve_user_path(cfg.parameter_budget.path)),
        use_parameter_budget=cfg.parameter_budget.enabled,
        parameter_budget_strict=cfg.parameter_budget.strict,
        parameter_budget_max_delta_pct=cfg.parameter_budget.max_delta_pct,
        eval_batch_size=cfg.data.eval_batch_size,
        graph_batch_size=cfg.data.graph_batch_size,
        graph_eval_batch_size=cfg.data.graph_eval_batch_size,
        learning_rate=cfg.trainer.learning_rate,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        transformer_pointer_layers=cfg.model.transformer_pointer_layers,
        dropout=cfg.model.dropout,
        tanh_clip=cfg.model.tanh_clip,
        device=cfg.device,
        num_workers=cfg.data.num_workers,
        skip_sigmoid_routing=cfg.skip_sigmoid_routing,
        stage=cfg.stage,
        wandb=cfg.wandb,
    )
    if cfg.skip_completed:
        commands = [command for command in commands if not _command_is_complete(command)]
    action = "Running" if cfg.execute else "Dry run"
    summary = f"{action} {len(commands)} command(s)."
    print(summary)
    logger.info(summary)
    for index, command in enumerate(commands, start=1):
        command_text = f"[{index}/{len(commands)}] {shlex.join(command)}"
        print(f"\n{command_text}", flush=True)
        logger.info(command_text)
        if cfg.execute:
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError:
                logger.exception("Command failed: %s", shlex.join(command))
                raise
    return commands


def resolve_problems(
    value: tuple[ProblemName, ...] | None,
    stage: MatrixStage,
) -> tuple[ProblemName, ...]:
    return STAGES[stage] if value is None else value


def build_commands(
    *,
    problems: Sequence[ProblemName],
    encoders: Sequence[EncoderKind] | None,
    decoders: Sequence[DecoderKind],
    modes: Sequence[TrainingMode],
    seeds: Sequence[int],
    data_root: Path,
    scale: ScaleConfig,
    output_root: str,
    parameter_budget: str,
    use_parameter_budget: bool,
    parameter_budget_strict: bool,
    parameter_budget_max_delta_pct: float,
    eval_batch_size: int | None,
    graph_batch_size: int | None,
    graph_eval_batch_size: int | None,
    learning_rate: float,
    d_model: int | None,
    d_ff: int | None,
    num_layers: int,
    num_heads: int,
    transformer_pointer_layers: int,
    dropout: float = 0.0,
    tanh_clip: float = 10.0,
    device: str,
    num_workers: int,
    skip_sigmoid_routing: bool,
    stage: MatrixStage,
    wandb: WandbConfig,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for seed in seeds:
        for mode in modes:
            profile = scale.for_mode(mode)
            for problem in problems:
                for encoder in encoders_for_problem(problem, encoders):
                    for decoder in decoders:
                        if (
                            skip_sigmoid_routing
                            and decoder == "sigmoid_subset"
                            and problem in {"tsp", "cvrp"}
                        ):
                            continue
                        paths = problem_split_paths(
                            problem,
                            train_instances=profile.train.instances,
                            baseline_instances=(
                                profile.rollout_baseline.instances
                                if profile.rollout_baseline is not None
                                else None
                            ),
                            val_instances=profile.validation.instances,
                            test_instances=profile.test.instances,
                            dataset_size=profile.dataset_size,
                            labeled=mode == "supervised",
                            data_root=data_root,
                        )
                        output_dir = (
                            f"{output_root}/{scale.name}/seed_{seed}/{mode}/"
                            f"{problem}/{encoder}/{decoder}"
                        )
                        train_batch_size, eval_batch = resolve_batch_sizes(
                            problem,
                            batch_size=profile.batch_size,
                            eval_batch_size=eval_batch_size or profile.batch_size,
                            graph_batch_size=graph_batch_size,
                            graph_eval_batch_size=graph_eval_batch_size,
                        )
                        steps_per_epoch = resolve_steps_per_epoch(
                            instances=profile.train.instances,
                            batch_size=train_batch_size,
                            epochs=profile.epochs,
                            policy=profile.train_data_policy,
                        )
                        wandb_group = wandb.group or (
                            f"matrix/{scale.name}/{stage}/seed_{seed}"
                        )
                        wandb_entity = (
                            "null" if wandb.entity is None else wandb.entity
                        )
                        wandb_name = matrix_run_name(
                            problem=problem,
                            decoder=decoder,
                            mode=mode,
                        )
                        wandb_name_override = json.dumps(wandb_name)
                        wandb_tags = json.dumps(
                            matrix_run_tags(
                                wandb,
                                problem=problem,
                                decoder=decoder,
                                mode=mode,
                                seed=seed,
                            ),
                            separators=(",", ":"),
                        )
                        command = [
                            "uv",
                            "run",
                            "python",
                            "-m",
                            "src.experiments.run",
                            f"scale={scale.name}",
                            f"problem={problem}",
                            f"encoder={encoder}",
                            f"decoder={decoder}",
                            f"mode={mode}",
                            f"data.root={data_root}",
                            f"data.train_path={paths['train']}",
                            f"data.val_path={paths['val']}",
                            f"data.test_path={paths['test']}",
                            f"seed={seed}",
                            f"trainer.epochs={profile.epochs}",
                            f"trainer.steps_per_epoch={steps_per_epoch}",
                            f"trainer.train_data_policy={profile.train_data_policy}",
                            "trainer.expected_train_instances="
                            f"{profile.train.instances}",
                            f"data.batch_size={train_batch_size}",
                            f"data.eval_batch_size={eval_batch}",
                            f"trainer.learning_rate={learning_rate}",
                            f"model.num_layers={num_layers}",
                            f"model.num_heads={num_heads}",
                            "model.transformer_pointer_layers="
                            f"{transformer_pointer_layers}",
                            f"model.dropout={dropout}",
                            f"model.tanh_clip={tanh_clip}",
                            f"device={device}",
                            f"data.num_workers={num_workers}",
                            f"paths.output_dir={output_dir}",
                            f"parameter_budget.enabled={str(use_parameter_budget).lower()}",
                            f"parameter_budget.path={parameter_budget}",
                            "parameter_budget.strict="
                            f"{str(parameter_budget_strict).lower()}",
                            "parameter_budget.max_delta_pct="
                            f"{parameter_budget_max_delta_pct}",
                            f"wandb.enabled={str(wandb.enabled).lower()}",
                            f"wandb.project={wandb.project}",
                            f"wandb.entity={wandb_entity}",
                            f"wandb.name={wandb_name_override}",
                            f"wandb.group={wandb_group}",
                            f"wandb.tags={wandb_tags}",
                            f"wandb.mode={wandb.mode}",
                            f"wandb.train_eval_batches={wandb.train_eval_batches}",
                        ]
                        if "baseline" in paths:
                            command.append(f"data.baseline_path={paths['baseline']}")
                        if use_parameter_budget:
                            command.extend(["model.d_model=null", "model.d_ff=null"])
                        else:
                            command.extend(
                                [
                                    f"model.d_model={d_model}",
                                    f"model.d_ff={d_ff}",
                                ]
                            )
                        commands.append(command)
    return commands


def matrix_run_name(
    *,
    problem: ProblemName,
    decoder: DecoderKind,
    mode: TrainingMode,
) -> str:
    """Create the concise W&B display name for a matrix child run."""
    return f"{problem}_{decoder}_{mode}"


def matrix_run_tags(
    wandb: WandbConfig,
    *,
    problem: ProblemName,
    decoder: DecoderKind,
    mode: TrainingMode,
    seed: int,
) -> list[str]:
    """Combine user tags with matrix identity omitted from the display name."""
    candidates = [
        *wandb.tags,
        problem,
        decoder,
        mode,
        f"seed-{seed}",
    ]
    if wandb.name is not None:
        candidates.append(wandb.name)
    return list(dict.fromkeys(candidates))


def resolve_steps_per_epoch(
    *,
    instances: int,
    batch_size: int,
    epochs: int,
    policy: str,
) -> int:
    if policy == "repeat":
        return (instances + batch_size - 1) // batch_size
    presentations_per_step = batch_size * epochs
    if instances % presentations_per_step:
        raise ValueError(
            "A consume_once graph-batch override must divide the training stream "
            "exactly across all epochs: "
            f"instances={instances}, batch_size={batch_size}, epochs={epochs}"
        )
    return instances // presentations_per_step


def _command_is_complete(command: list[str]) -> bool:
    prefix = "paths.output_dir="
    output_arg = next((arg for arg in command if arg.startswith(prefix)), None)
    if output_arg is None:
        return False
    result_path = Path(output_arg.removeprefix(prefix)) / "result.json"
    if not result_path.is_file():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("global_steps", 0)) and isinstance(payload.get("test"), dict)


def resolve_batch_sizes(
    problem: ProblemName,
    *,
    batch_size: int,
    eval_batch_size: int,
    graph_batch_size: int | None,
    graph_eval_batch_size: int | None,
) -> tuple[int, int]:
    if problem not in GRAPH_PROBLEMS:
        return batch_size, eval_batch_size
    train_batch = graph_batch_size if graph_batch_size is not None else batch_size
    if graph_eval_batch_size is not None:
        eval_batch = graph_eval_batch_size
    elif graph_batch_size is not None:
        eval_batch = graph_batch_size
    else:
        eval_batch = eval_batch_size
    return train_batch, eval_batch


if __name__ == "__main__":
    main()
