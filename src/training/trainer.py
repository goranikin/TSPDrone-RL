"""Policy-gradient trainer for TSP-D with greedy rollout baseline."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import optim
from tqdm.auto import tqdm

from src.config import RunConfig
from src.models.policy import Policy
from src.paths import REPOSITORY_ROOT, checkpoint_dir, resolve_user_path
from src.problems.tspd import DataGenerator, Env
from src.training import wandb_support
from src.training.baselines import (
    ExponentialMakespanBaseline,
    RolloutMakespanBaseline,
)
from src.training.metrics import makespan_metrics, wandb_step_metrics

DecodeMode = Literal["train_sample", "greedy", "eval_sample"]


class Trainer:
    def __init__(
        self,
        *,
        policy: Policy,
        cfg: RunConfig,
        env: Env,
        data_gen: DataGenerator,
        device: torch.device,
        output_dir: str,
        logger: logging.Logger | None = None,
        wandb_log: bool = False,
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.env = env
        self.data_gen = data_gen
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        self.wandb_log = wandb_log
        self.hidden_dim = cfg.model.hidden_dim
        self.decode_len = cfg.model.decode_len
        self.ckpt_dir = checkpoint_dir(output_dir, cfg.physics.n_nodes)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        results = resolve_user_path(cfg.data.results_dir)
        if not results.is_absolute():
            results = REPOSITORY_ROOT / results
        self.results_dir = results
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.exp_baseline = ExponentialMakespanBaseline(cfg.trainer.exp_baseline_beta)
        self.rollout_baseline = RolloutMakespanBaseline(
            device=device,
            alpha=cfg.trainer.baseline_alpha,
        )

    @property
    def actor(self) -> Policy:
        """Alias used by rollout baseline deepcopy helpers."""
        return self.policy

    def train(self) -> dict[str, Any]:
        policy = self.policy
        cfg = self.cfg
        policy.train()
        optimizer = optim.Adam(policy.parameters(), lr=cfg.trainer.actor_lr)

        best_model = float("inf")
        history: list[dict[str, Any]] = []
        r_test: list[float] = []
        start = time.time()
        print(
            f"training started architecture={cfg.architecture} "
            f"(greedy rollout baseline)"
        )

        epoch_iter: Any = range(cfg.trainer.epochs)
        if cfg.trainer.progress_bar:
            epoch_iter = tqdm(epoch_iter, desc="train", total=cfg.trainer.epochs)

        for episode in epoch_iter:
            if (
                episode >= cfg.trainer.baseline_warmup_episodes
                and self.rollout_baseline.baseline_actor is None
            ):
                self.rollout_baseline.init_from(policy)

            data = self.data_gen.get_train_next()
            makespan, log_sum = self._rollout(
                policy,
                data,
                mode="train_sample",
                collect_log_probs=True,
            )
            assert log_sum is not None

            baseline = self._baseline_value(makespan, data, episode)
            actor_loss = torch.mean((makespan - baseline).detach() * log_sum)

            optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg.trainer.max_grad_norm
            )
            optimizer.step()

            train_makespan = float(makespan.mean().item())
            baseline_mean = float(baseline.mean().item())
            elapsed = time.time() - start
            row: dict[str, Any] = {
                "episode": episode + 1,
                "actor_loss": float(actor_loss.item()),
                "train_makespan": train_makespan,
                "baseline_makespan": baseline_mean,
                "elapsed_sec": elapsed,
            }

            if (episode + 1) % cfg.trainer.log_every == 0 or episode == 0:
                print(
                    f"episode={episode + 1} makespan={train_makespan:.4f} "
                    f"baseline={baseline_mean:.4f} "
                    f"actor_loss={actor_loss.item():.4f} e_t={elapsed:.1f}"
                )
                if self.wandb_log:
                    wandb_support.log(
                        wandb_step_metrics(
                            episode=episode + 1,
                            actor_loss=float(actor_loss.item()),
                            train_makespan=train_makespan,
                            baseline_makespan=baseline_mean,
                            elapsed_sec=elapsed,
                        ),
                        step=episode + 1,
                    )

            if episode % cfg.trainer.test_interval == 0:
                test_R = self.test()
                r_test.append(test_R)
                row["val_makespan"] = test_R
                np.savetxt(self.ckpt_dir / "test_rewards.txt", r_test)
                print(f"testing average rewards: {test_R}")
                if self.wandb_log:
                    wandb_support.log({"val/makespan": test_R}, step=episode + 1)
                if test_R < best_model:
                    best_model = test_R
                    if cfg.trainer.save_checkpoints:
                        self._save_checkpoint("best_model")
                    row["best_val_makespan"] = best_model

                updated = self.rollout_baseline.maybe_update(
                    policy,
                    self.data_gen.get_test_all(),
                    self._greedy_makespans,
                    warmup_done=episode + 1 >= cfg.trainer.baseline_warmup_episodes,
                )
                row["rollout_updated"] = updated
                if updated:
                    print("rollout baseline updated")

            if (
                cfg.trainer.save_checkpoints
                and episode % cfg.trainer.save_interval == 0
            ):
                self._save_checkpoint(str(episode // cfg.trainer.save_interval))

            history.append(row)
            if self.logger is not None and "val_makespan" in row:
                self.logger.info("episode=%d metrics=%s", episode + 1, json.dumps(row))

        result = {
            "history": history,
            "best_val_makespan": best_model if best_model < float("inf") else None,
            "training_time_sec": time.time() - start,
            "architecture": cfg.architecture,
        }
        self._write_history(result)
        if self.wandb_log:
            wandb_support.update_summary(
                {
                    "best_val_makespan": result["best_val_makespan"],
                    "training_time_sec": result["training_time_sec"],
                    "architecture": cfg.architecture,
                }
            )
            wandb_support.log(
                {
                    "train/training_time_sec": result["training_time_sec"],
                    "train/best_val_makespan": result["best_val_makespan"],
                }
            )
        return result

    def _baseline_value(
        self,
        makespan: torch.Tensor,
        data: np.ndarray,
        episode: int,
    ) -> torch.Tensor:
        if episode < self.cfg.trainer.baseline_warmup_episodes:
            return self.exp_baseline.evaluate(makespan)
        if self.rollout_baseline.baseline_actor is None:
            self.rollout_baseline.init_from(self.policy)
        return self.rollout_baseline.evaluate(data, self._greedy_makespans)

    @torch.no_grad()
    def _greedy_makespans(self, policy: Policy, data: np.ndarray) -> torch.Tensor:
        makespan, _ = self._rollout(
            policy, data, mode="greedy", collect_log_probs=False
        )
        return makespan

    def _rollout(
        self,
        policy: Policy,
        data: np.ndarray,
        *,
        mode: DecodeMode,
        collect_log_probs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        env = self.env
        device = self.device
        env.input_data = data
        state, avail_actions = env.reset()
        batch_size = env.batch_size

        was_training = policy.training
        was_sample_mode = policy.sample_mode
        if mode == "train_sample":
            policy.train()
            policy.set_sample_mode(False)
        elif mode == "eval_sample":
            policy.eval()
            policy.set_sample_mode(True)
        else:
            policy.eval()
            policy.set_sample_mode(False)

        coords = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(device)
        policy.embed(coords)
        policy.reset_episode(batch_size)
        prev_embed = policy.depot_embedding()
        ter = np.zeros(batch_size, dtype=np.float32)
        time_vec_truck = np.zeros([batch_size, 2])
        time_vec_drone = np.zeros([batch_size, 3])
        logs: list[torch.Tensor] = []

        context = torch.enable_grad() if collect_log_probs else torch.no_grad()
        with context:
            for _ in range(self.decode_len):
                terminated = torch.from_numpy(ter).to(device)
                for vehicle in (0, 1):
                    if vehicle == 0:
                        avail = torch.from_numpy(
                            avail_actions[:, :, 0]
                            .reshape([batch_size, env.n_nodes])
                            .astype(np.float32)
                        ).to(device)
                        dynamic = torch.from_numpy(
                            np.expand_dims(state[:, :, 0], 2).astype(np.float32)
                        ).to(device)
                        idx_truck, logp = policy(
                            prev_embed, dynamic, terminated, avail
                        )
                        free = np.where(
                            np.logical_and(
                                avail_actions[:, :, 1].sum(axis=1) > 1, env.sortie == 0
                            )
                        )[0]
                        avail_actions[free, idx_truck[free].cpu(), 1] = 0
                        idx = idx_truck
                    else:
                        avail = torch.from_numpy(
                            avail_actions[:, :, 1]
                            .reshape([batch_size, env.n_nodes])
                            .astype(np.float32)
                        ).to(device)
                        dynamic = torch.from_numpy(
                            np.expand_dims(state[:, :, 1], 2).astype(np.float32)
                        ).to(device)
                        idx_drone, logp = policy(
                            prev_embed, dynamic, terminated, avail
                        )
                        idx = idx_drone

                    prev_embed = policy.node_embedding(idx).detach()
                    if collect_log_probs:
                        logs.append(logp.unsqueeze(1))

                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(
                    idx_truck.cpu().numpy(),
                    idx_drone.cpu().numpy(),
                    time_vec_truck,
                    time_vec_drone,
                    ter,
                )

        makespan = torch.from_numpy(env.current_time.astype(np.float32)).to(device)
        log_sum = torch.cat(logs, dim=1).sum(dim=1) if collect_log_probs else None
        policy.train(was_training)
        policy.set_sample_mode(was_sample_mode)
        return makespan, log_sum

    def test(self) -> float:
        data = self.data_gen.get_test_all()
        makespan = self._greedy_makespans(self.policy, data)
        values = makespan.detach().cpu().numpy()
        print("finished greedy eval")
        np.savetxt(
            self.results_dir
            / (
                f"test_results-{self.cfg.scale.test_size}-len-"
                f"{self.cfg.physics.n_nodes}.txt"
            ),
            values,
        )
        self.policy.train()
        return makespan_metrics(values).makespan

    def sampling_batch(
        self, sample_size: int | None = None
    ) -> tuple[list[float], list[float]]:
        sample_size = sample_size or self.cfg.n_samples
        data = self.data_gen.get_test_all()
        best_rewards: list[float] = []
        times: list[float] = []
        initial_t = time.time()

        for index in range(data.shape[0]):
            repeated = np.repeat(
                np.expand_dims(data[index], axis=0), sample_size, axis=0
            )
            makespan, _ = self._rollout(
                self.policy,
                repeated,
                mode="eval_sample",
                collect_log_probs=False,
            )
            best_rewards.append(float(makespan.min().item()))
            times.append(time.time() - initial_t)

        np.savetxt(
            self.results_dir / f"best_rewards_list_{sample_size}_samples.txt",
            best_rewards,
        )
        self.policy.train()
        return best_rewards, times

    def _save_checkpoint(self, prefix: str) -> None:
        torch.save(
            self.policy.state_dict(),
            self.ckpt_dir / f"{prefix}_actor_truck_params.pkl",
        )

    def _write_history(self, result: dict[str, Any]) -> None:
        path = Path(self.output_dir) / "history.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
