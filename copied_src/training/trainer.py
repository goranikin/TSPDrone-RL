import json
import logging
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from src.models.model import NCOModel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import BaselineKind, NonNegativeInt, PositiveFloat, PositiveInt
from src.constants import TrainingDataPolicy, TrainingMode
from src.training.baselines import ExponentialRewardBaseline, RolloutRewardBaseline
from src.training.metrics import (
    EvaluationMetrics,
    aggregate_metrics,
    batch_metrics,
    wandb_metrics,
    wandb_rl_step_metrics,
    wandb_supervised_step_metrics,
)
from src.training.wandb_support import log as wandb_log
from src.training.wandb_support import update_summary as wandb_update_summary
from src.utils import move_to_device, timer


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: TrainingMode
    epochs: PositiveInt = 1
    steps_per_epoch: PositiveInt
    train_data_policy: TrainingDataPolicy
    expected_train_instances: PositiveInt
    learning_rate: PositiveFloat = 1e-4
    max_grad_norm: PositiveFloat = 1.0
    baseline: BaselineKind = "rollout"
    baseline_alpha: float = Field(default=0.05, ge=0, le=1)
    baseline_warmup_epochs: NonNegativeInt = 1
    exp_baseline_beta: float = Field(default=0.8, ge=0, le=1)
    log_every: PositiveInt = 25
    progress_bar: bool = True
    output_dir: str = "outputs"
    save_checkpoints: bool = True
    wandb_log: bool = False
    wandb_train_eval_batches: PositiveInt = 10


class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[dict[str, Any]] = Field(default_factory=list)
    training_time_sec: float = 0.0
    best_validation_objective: float | None = None
    best_validation_feasibility_rate: float | None = None
    best_validation_score: float | None = None
    best_checkpoint_path: str | None = None
    train_presentations: int = 0


class Trainer:
    def __init__(
        self,
        *,
        model: NCOModel,
        train_loader: DataLoader,
        baseline_loader: DataLoader | None,
        val_loader: DataLoader | None,
        config: TrainingConfig,
        device: torch.device,
        logger: logging.Logger | None = None,
        train_eval_loader: DataLoader | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.baseline_loader = baseline_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.train_eval_loader = train_eval_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.exp_baseline = ExponentialRewardBaseline(config.exp_baseline_beta)
        self.rollout_baseline = RolloutRewardBaseline(
            device=device,
            alpha=config.baseline_alpha,
        )
        self.global_step = 0
        self.train_presentations = 0
        self._train_stream = None

    def fit(self) -> TrainingResult:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        result = TrainingResult()
        with timer(self.device) as elapsed:
            for epoch in range(self.config.epochs):
                if self.config.mode == "supervised":
                    train_loss = self._train_supervised_epoch(epoch)
                    row: dict[str, Any] = {"epoch": epoch + 1, "train_loss": train_loss}
                else:
                    train_loss = self._train_rl_epoch(epoch)
                    row = {"epoch": epoch + 1, "train_policy_loss": train_loss}
                if self.val_loader is not None:
                    metrics = self.evaluate(
                        self.val_loader,
                        description="validation",
                        epoch=epoch + 1,
                    )
                    row.update(metrics.to_dict("val"))
                    validation_score = self._validation_score(metrics)
                    if self._is_better_validation_score(
                        validation_score,
                        result.best_validation_score,
                    ):
                        result.best_validation_objective = metrics.objective
                        result.best_validation_feasibility_rate = (
                            metrics.feasibility_rate
                        )
                        result.best_validation_score = validation_score
                        if self.config.save_checkpoints:
                            result.best_checkpoint_path = self.save_checkpoint(
                                "best.pt", epoch + 1
                            )
                    if self.config.mode == "rl" and self.config.baseline == "rollout":
                        if self.baseline_loader is None:
                            raise RuntimeError(
                                "RL rollout baseline requires a separate baseline loader"
                            )
                        updated = self.rollout_baseline.maybe_update(
                            self.model,
                            self.baseline_loader,
                            warmup_done=epoch + 1 >= self.config.baseline_warmup_epochs,
                        )
                        row["rollout_updated"] = updated
                result.history.append(row)
                self._write_history(result)
                if self.logger is not None:
                    self.logger.info(
                        "epoch=%d metrics=%s",
                        epoch + 1,
                        json.dumps(row, sort_keys=True),
                    )
                if self.config.wandb_log:
                    self._log_epoch_metrics(row)
                    self._log_train_eval_metrics(epoch + 1)
                if self.config.save_checkpoints:
                    self.save_checkpoint("last.pt", epoch + 1)
        if (
            self.config.train_data_policy == "consume_once"
            and self.train_presentations != self.config.expected_train_instances
        ):
            raise RuntimeError(
                "The consume-once stream produced "
                f"{self.train_presentations} instances; expected "
                f"{self.config.expected_train_instances}"
            )
        result.train_presentations = self.train_presentations
        result.training_time_sec = elapsed["elapsed"]
        self._write_history(result)
        if self.logger is not None:
            self.logger.info(
                "training_complete steps=%d training_time_sec=%.6f",
                self.global_step,
                result.training_time_sec,
            )
        if self.config.wandb_log:
            wandb_log(
                {
                    "train/training_time_sec": result.training_time_sec,
                    "train/best_validation_objective": result.best_validation_objective,
                    "train/best_validation_feasibility_rate": (
                        result.best_validation_feasibility_rate
                    ),
                    "train/best_validation_score": result.best_validation_score,
                    "train/presentations": result.train_presentations,
                },
                step=self.global_step,
            )
        return result

    def _train_supervised_epoch(self, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []
        window_losses: list[float] = []
        batches = self._epoch_batches(self.train_loader)
        iterator = tqdm(
            batches,
            total=self.config.steps_per_epoch,
            disable=not self.config.progress_bar,
            desc=f"supervised {epoch + 1}/{self.config.epochs}",
        )
        for step, batch in enumerate(iterator, start=1):
            batch = move_to_device(batch, self.device)
            self.train_presentations += self._batch_size(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model.supervised_loss(batch)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.global_step += 1
            batch_loss = float(loss.detach().item())
            losses.append(batch_loss)
            window_losses.append(batch_loss)
            if step % self.config.log_every == 0:
                window_loss = sum(window_losses) / len(window_losses)
                iterator.set_postfix(loss=window_loss)
                if self.logger is not None:
                    self.logger.info(
                        "epoch=%d step=%d global_step=%d train_loss=%.8f",
                        epoch + 1,
                        step,
                        self.global_step,
                        window_loss,
                    )
                if self.config.wandb_log:
                    step_metrics = self._greedy_batch_metrics(batch)
                    wandb_log(
                        wandb_supervised_step_metrics(
                            batch_loss=batch_loss,
                            window_loss=window_loss,
                            metrics=step_metrics,
                            epoch=epoch + 1,
                        ),
                        step=self.global_step,
                    )
                window_losses.clear()
        return sum(losses) / max(len(losses), 1)

    def _train_rl_epoch(self, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []
        window_losses: list[float] = []
        if (
            self.config.baseline == "rollout"
            and self.rollout_baseline.baseline_model is None
        ):
            self.rollout_baseline.init_from(self.model)
        batches = self._epoch_batches(self.train_loader)
        iterator = tqdm(
            batches,
            total=self.config.steps_per_epoch,
            disable=not self.config.progress_bar,
            desc=f"rl {epoch + 1}/{self.config.epochs}",
        )
        for step, batch in enumerate(iterator, start=1):
            batch = move_to_device(batch, self.device)
            self.train_presentations += self._batch_size(batch)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch, decode_type="sampling")
            if output.reward is None or output.log_probs is None:
                raise RuntimeError("RL requires reward and log_probs")
            if not bool(output.feasible.all()):
                infeasible = int((~output.feasible).sum().item())
                raise RuntimeError(
                    "RL decoder produced infeasible sampled solutions; objective-only "
                    f"rewards are unsafe for {infeasible} batch item(s)"
                )
            baseline = self._baseline_value(output.reward, batch, epoch)
            advantage = output.reward - baseline
            loss = -(advantage.detach() * output.log_probs).mean()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.global_step += 1
            batch_loss = float(loss.detach().item())
            losses.append(batch_loss)
            window_losses.append(batch_loss)
            if step % self.config.log_every == 0:
                window_loss = sum(window_losses) / len(window_losses)
                iterator.set_postfix(loss=window_loss)
                if self.logger is not None:
                    self.logger.info(
                        "epoch=%d step=%d global_step=%d train_policy_loss=%.8f",
                        epoch + 1,
                        step,
                        self.global_step,
                        window_loss,
                    )
                if self.config.wandb_log:
                    step_metrics = batch_metrics(self.model.problem, batch, output, 0.0)
                    wandb_log(
                        wandb_rl_step_metrics(
                            batch_policy_loss=batch_loss,
                            window_policy_loss=window_loss,
                            metrics=step_metrics,
                            reward=output.reward,
                            advantage=advantage,
                            baseline=baseline,
                            epoch=epoch + 1,
                        ),
                        step=self.global_step,
                    )
                window_losses.clear()
        return sum(losses) / max(len(losses), 1)

    def _baseline_value(
        self,
        reward: torch.Tensor,
        batch: dict[str, Any],
        epoch: int,
    ) -> torch.Tensor:
        if self.config.baseline == "exponential":
            return self.exp_baseline.evaluate(reward)
        if epoch < self.config.baseline_warmup_epochs:
            return self.exp_baseline.evaluate(reward)
        return self.rollout_baseline.evaluate_batch(batch)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        *,
        description: str,
        epoch: int | None = None,
    ) -> EvaluationMetrics:
        self.model.eval()
        items = []
        total = len(loader)
        label = (
            description
            if epoch is None
            else f"{description} {epoch}/{self.config.epochs}"
        )
        iterator = tqdm(
            loader,
            total=total,
            disable=not self.config.progress_bar,
            desc=label,
        )
        self._update_evaluation_progress(description, completed=0, total=total)
        for batch_index, batch in enumerate(iterator, start=1):
            batch = move_to_device(batch, self.device)
            with timer(self.device) as elapsed:
                output = self.model(batch, decode_type="greedy")
            items.append(
                batch_metrics(self.model.problem, batch, output, elapsed["elapsed"])
            )
            self._update_evaluation_progress(
                description,
                completed=batch_index,
                total=total,
            )
        return aggregate_metrics(items)

    @torch.no_grad()
    def _evaluate_subset(
        self, loader: DataLoader, *, max_batches: int
    ) -> EvaluationMetrics:
        self.model.eval()
        items = []
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = move_to_device(batch, self.device)
            with timer(self.device) as elapsed:
                output = self.model(batch, decode_type="greedy")
            items.append(
                batch_metrics(self.model.problem, batch, output, elapsed["elapsed"])
            )
        return aggregate_metrics(items)

    def _epoch_batches(self, loader: DataLoader):
        if self.config.train_data_policy == "consume_once":
            if self._train_stream is None:
                self._train_stream = iter(loader)
            for _ in range(self.config.steps_per_epoch):
                try:
                    yield next(self._train_stream)
                except StopIteration as exc:
                    raise RuntimeError(
                        "The consume-once training stream ended before the configured "
                        "training budget was exhausted"
                    ) from exc
            return

        iterator = iter(loader)
        for _ in range(self.config.steps_per_epoch):
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(loader)
                yield next(iterator)

    def _validation_score(self, metrics: EvaluationMetrics) -> float:
        feasibility_penalty = (1.0 - metrics.feasibility_rate) * 1_000_000.0
        if metrics.reference_gap is not None:
            return feasibility_penalty + metrics.reference_gap
        objective_score = (
            metrics.objective
            if self.model.problem.objective_sense == "min"
            else -metrics.objective
        )
        return feasibility_penalty + objective_score

    @staticmethod
    def _batch_size(batch: dict[str, Any]) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.size(0))
        raise RuntimeError("Cannot determine batch size from a tensor-free batch")

    @staticmethod
    def _is_better_validation_score(
        candidate: float,
        incumbent: float | None,
    ) -> bool:
        return incumbent is None or candidate < incumbent

    def _update_evaluation_progress(
        self,
        description: str,
        *,
        completed: int,
        total: int,
    ) -> None:
        if not self.config.wandb_log:
            return
        fraction = completed / max(total, 1)
        wandb_update_summary(
            {
                "progress/phase": description,
                f"progress/{description}/completed_batches": completed,
                f"progress/{description}/total_batches": total,
                f"progress/{description}/fraction": fraction,
            }
        )

    def save_checkpoint(self, filename: str, epoch: int) -> str:
        path = Path(self.config.output_dir) / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_config": self.config.model_dump(mode="json"),
                "problem": self.model.problem_name,
                "encoder": self.model.encoder_kind,
                "decoder": self.model.decoder_kind,
            },
            path,
        )
        return str(path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _log_epoch_metrics(self, row: dict[str, Any]) -> None:
        metrics: dict[str, Any] = {"epoch": row["epoch"]}
        mode_prefix = "sl" if self.config.mode == "supervised" else "rl"
        for key, value in row.items():
            if key == "epoch":
                continue
            if key == "train_loss":
                metrics[f"train/{mode_prefix}/loss_epoch"] = value
            elif key == "train_policy_loss":
                metrics["train/rl/policy_loss_epoch"] = value
            elif key == "rollout_updated":
                metrics["train/rl/rollout_updated"] = float(value)
            elif key.startswith("val/"):
                metrics[key] = value
            else:
                metrics[key] = value
        wandb_log(metrics, step=self.global_step)

    def _log_train_eval_metrics(self, epoch: int) -> None:
        if self.train_eval_loader is None:
            return
        was_training = self.model.training
        with torch.random.fork_rng(devices=self._fork_rng_devices()):
            metrics = self._evaluate_subset(
                self.train_eval_loader,
                max_batches=self.config.wandb_train_eval_batches,
            )
        if was_training:
            self.model.train()
        mode_prefix = "sl" if self.config.mode == "supervised" else "rl"
        wandb_log(
            {
                "epoch": epoch,
                **wandb_metrics(metrics, f"train/{mode_prefix}/eval"),
            },
            step=self.global_step,
        )

    @torch.no_grad()
    def _greedy_batch_metrics(self, batch: dict[str, Any]):
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.random.fork_rng(devices=self._fork_rng_devices()):
                output = self.model(batch, decode_type="greedy")
                return batch_metrics(self.model.problem, batch, output, 0.0)
        finally:
            self.model.train(was_training)

    def _fork_rng_devices(self) -> list[int]:
        if self.device.type != "cuda":
            return []
        return [
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        ]

    def _write_history(self, result: TrainingResult) -> None:
        payload = {
            "training_time_sec": result.training_time_sec,
            "best_validation_objective": result.best_validation_objective,
            "best_validation_feasibility_rate": (
                result.best_validation_feasibility_rate
            ),
            "best_validation_score": result.best_validation_score,
            "best_checkpoint_path": result.best_checkpoint_path,
            "train_presentations": result.train_presentations,
            "history": result.history,
        }
        path = Path(self.config.output_dir) / "history.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
