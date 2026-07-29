"""A2C trainer for TSP-D (truck + drone cooperative routing)."""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from tqdm.auto import tqdm

from src.config import RunConfig
from src.models.actor import Actor
from src.models.critic import Critic
from src.paths import REPOSITORY_ROOT, checkpoint_dir, resolve_user_path
from src.problems.tspd import DataGenerator, Env
from src.training.metrics import makespan_metrics, wandb_step_metrics
from src.training import wandb_support


class Trainer:
    def __init__(
        self,
        *,
        actor: Actor,
        critic: Critic,
        cfg: RunConfig,
        env: Env,
        data_gen: DataGenerator,
        device: torch.device,
        output_dir: str,
        logger: logging.Logger | None = None,
        wandb_log: bool = False,
    ) -> None:
        self.actor = actor
        self.critic = critic
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

    def train(self) -> dict[str, Any]:
        actor = self.actor
        critic = self.critic
        env = self.env
        cfg = self.cfg
        device = self.device

        actor.train()
        critic.train()

        actor_optim = optim.Adam(actor.parameters(), lr=cfg.trainer.actor_lr)
        critic_optim = optim.Adam(critic.parameters(), lr=cfg.trainer.critic_lr)

        best_model = float("inf")
        history: list[dict[str, Any]] = []
        r_test: list[float] = []
        start = time.time()
        print("training started")

        epoch_iter = range(cfg.trainer.epochs)
        if cfg.trainer.progress_bar:
            epoch_iter = tqdm(epoch_iter, desc="train", total=cfg.trainer.epochs)

        for episode in epoch_iter:
            data = self.data_gen.get_train_next()
            env.input_data = data
            state, avail_actions = env.reset()

            coords = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(device)
            static_hidden = actor.emd_stat(coords).permute(0, 2, 1)

            static = (
                torch.from_numpy(env.input_data[:, :, :2].astype(np.float32))
                .permute(0, 2, 1)
                .to(device)
            )
            w = torch.from_numpy(
                env.input_data[:, :, 2]
                .reshape(env.batch_size, env.n_nodes, 1)
                .astype(np.float32)
            ).to(device)

            hx = torch.zeros(1, env.batch_size, self.hidden_dim, device=device)
            cx = torch.zeros(1, env.batch_size, self.hidden_dim, device=device)
            last_hh = (hx, cx)

            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, env.n_nodes - 1].unsqueeze(2)
            time_vec_truck = np.zeros([env.batch_size, 2])
            time_vec_drone = np.zeros([env.batch_size, 3])

            logs: list[torch.Tensor] = []
            time_step = 0

            while time_step < self.decode_len:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(device)
                for j in range(2):
                    if j == 0:
                        avail_actions_truck = torch.from_numpy(
                            avail_actions[:, :, 0]
                            .reshape([env.batch_size, env.n_nodes])
                            .astype(np.float32)
                        ).to(device)
                        dynamic_truck = torch.from_numpy(
                            np.expand_dims(state[:, :, 0], 2).astype(np.float32)
                        ).to(device)
                        idx_truck, _prob, logp, last_hh = actor.forward(
                            static_hidden,
                            dynamic_truck,
                            decoder_input,
                            last_hh,
                            terminated,
                            avail_actions_truck,
                        )
                        b_s = np.where(
                            np.logical_and(
                                avail_actions[:, :, 1].sum(axis=1) > 1, env.sortie == 0
                            )
                        )[0]
                        avail_actions[b_s, idx_truck[b_s].cpu(), 1] = 0
                        avail_actions_drone = torch.from_numpy(
                            avail_actions[:, :, 1]
                            .reshape([env.batch_size, env.n_nodes])
                            .astype(np.float32)
                        ).to(device)
                        idx = idx_truck
                    else:
                        dynamic_drone = torch.from_numpy(
                            np.expand_dims(state[:, :, 1], 2).astype(np.float32)
                        ).to(device)
                        idx_drone, _prob, logp, last_hh = actor.forward(
                            static_hidden,
                            dynamic_drone,
                            decoder_input,
                            last_hh,
                            terminated,
                            avail_actions_drone,
                        )
                        idx = idx_drone

                    decoder_input = torch.gather(
                        static_hidden,
                        2,
                        idx.view(-1, 1, 1).expand(
                            env.batch_size, self.hidden_dim, 1
                        ),
                    ).detach()
                    logs.append(logp.unsqueeze(1))

                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(
                    idx_truck.cpu().numpy(),
                    idx_drone.cpu().numpy(),
                    time_vec_truck,
                    time_vec_drone,
                    ter,
                )
                time_step += 1

            log_tensor = torch.cat(logs, dim=1)
            critic_est = critic(static, w).view(-1)
            R = torch.from_numpy(env.current_time.astype(np.float32)).to(device)
            advantage = R - critic_est
            actor_loss = torch.mean(advantage.detach() * log_tensor.sum(dim=1))
            critic_loss = torch.mean(advantage**2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.trainer.max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.trainer.max_grad_norm)
            critic_optim.step()

            train_makespan = float(R.mean().item())
            elapsed = time.time() - start
            row: dict[str, Any] = {
                "episode": episode + 1,
                "actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "train_makespan": train_makespan,
                "elapsed_sec": elapsed,
            }

            if (episode + 1) % cfg.trainer.log_every == 0 or episode == 0:
                print(
                    f"episode={episode + 1} makespan={train_makespan:.4f} "
                    f"actor_loss={actor_loss.item():.4f} "
                    f"critic_loss={critic_loss.item():.4f} e_t={elapsed:.1f}"
                )

            if self.wandb_log and (
                (episode + 1) % cfg.trainer.log_every == 0 or episode == 0
            ):
                wandb_support.log(
                    wandb_step_metrics(
                        episode=episode + 1,
                        actor_loss=float(actor_loss.item()),
                        critic_loss=float(critic_loss.item()),
                        train_makespan=train_makespan,
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
                    wandb_support.log(
                        {"val/makespan": test_R},
                        step=episode + 1,
                    )
                if test_R < best_model:
                    best_model = test_R
                    if cfg.trainer.save_checkpoints:
                        self._save_checkpoint("best_model")
                    row["best_val_makespan"] = best_model

            if (
                cfg.trainer.save_checkpoints
                and episode % cfg.trainer.save_interval == 0
            ):
                num = str(episode // cfg.trainer.save_interval)
                self._save_checkpoint(num)

            history.append(row)
            if self.logger is not None and "val_makespan" in row:
                self.logger.info("episode=%d metrics=%s", episode + 1, json.dumps(row))

        result = {
            "history": history,
            "best_val_makespan": best_model if best_model < float("inf") else None,
            "training_time_sec": time.time() - start,
        }
        self._write_history(result)
        if self.wandb_log:
            wandb_support.update_summary(
                {
                    "best_val_makespan": result["best_val_makespan"],
                    "training_time_sec": result["training_time_sec"],
                }
            )
            wandb_support.log(
                {
                    "train/training_time_sec": result["training_time_sec"],
                    "train/best_val_makespan": result["best_val_makespan"],
                }
            )
        return result

    def test(self) -> float:
        actor = self.actor
        env = self.env
        device = self.device
        actor.eval()

        data = self.data_gen.get_test_all()
        env.input_data = data
        state, avail_actions = env.reset()

        time_vec_truck = np.zeros([env.batch_size, 2])
        time_vec_drone = np.zeros([env.batch_size, 3])

        with torch.no_grad():
            coords = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(device)
            static_hidden = actor.emd_stat(coords).permute(0, 2, 1)

            hx = torch.zeros(1, env.batch_size, self.hidden_dim, device=device)
            cx = torch.zeros(1, env.batch_size, self.hidden_dim, device=device)
            last_hh = (hx, cx)

            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, env.n_nodes - 1].unsqueeze(2)
            time_step = 0
            while time_step < self.decode_len:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(device)
                for j in range(2):
                    if j == 0:
                        avail_actions_truck = torch.from_numpy(
                            avail_actions[:, :, 0]
                            .reshape([env.batch_size, env.n_nodes])
                            .astype(np.float32)
                        ).to(device)
                        dynamic_truck = torch.from_numpy(
                            np.expand_dims(state[:, :, 0], 2).astype(np.float32)
                        ).to(device)
                        idx_truck, _prob, _logp, last_hh = actor.forward(
                            static_hidden,
                            dynamic_truck,
                            decoder_input,
                            last_hh,
                            terminated,
                            avail_actions_truck,
                        )
                        b_s = np.where(
                            np.logical_and(
                                avail_actions[:, :, 1].sum(axis=1) > 1, env.sortie == 0
                            )
                        )[0]
                        avail_actions[b_s, idx_truck[b_s].cpu(), 1] = 0
                        avail_actions_drone = torch.from_numpy(
                            avail_actions[:, :, 1]
                            .reshape([env.batch_size, env.n_nodes])
                            .astype(np.float32)
                        ).to(device)
                        idx = idx_truck
                    else:
                        dynamic_drone = torch.from_numpy(
                            np.expand_dims(state[:, :, 1], 2).astype(np.float32)
                        ).to(device)
                        idx_drone, _prob, _logp, last_hh = actor.forward(
                            static_hidden,
                            dynamic_drone,
                            decoder_input,
                            last_hh,
                            terminated,
                            avail_actions_drone,
                        )
                        idx = idx_drone

                    decoder_input = torch.gather(
                        static_hidden,
                        2,
                        idx.view(-1, 1, 1).expand(
                            env.batch_size, self.hidden_dim, 1
                        ),
                    ).detach()

                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(
                    idx_truck.cpu().numpy(),
                    idx_drone.cpu().numpy(),
                    time_vec_truck,
                    time_vec_drone,
                    ter,
                )
                time_step += 1

        R = copy.copy(env.current_time)
        print("finished: ", sum(ter))

        fname = self.results_dir / (
            f"test_results-{self.cfg.scale.test_size}-len-"
            f"{self.cfg.physics.n_nodes}.txt"
        )
        np.savetxt(fname, R)

        metrics = makespan_metrics(R)
        actor.train()
        return metrics.makespan

    def sampling_batch(
        self, sample_size: int | None = None
    ) -> tuple[list[float], list[float]]:
        sample_size = sample_size or self.cfg.n_samples
        actor = self.actor
        env = self.env
        device = self.device

        actor.eval()
        actor.set_sample_mode(True)
        times: list[float] = []
        initial_t = time.time()
        data = self.data_gen.get_test_all()
        data_list = [np.expand_dims(data[i, ...], axis=0) for i in range(data.shape[0])]
        best_rewards_list: list[float] = []

        for d in data_list:
            data_rep = np.repeat(d, sample_size, axis=0)
            env.input_data = data_rep
            state, avail_actions = env.reset()

            time_vec_truck = np.zeros([sample_size, 2])
            time_vec_drone = np.zeros([sample_size, 3])
            with torch.no_grad():
                coords = torch.from_numpy(data_rep[:, :, :2].astype(np.float32)).to(
                    device
                )
                static_hidden = actor.emd_stat(coords).permute(0, 2, 1)

                hx = torch.zeros(1, sample_size, self.hidden_dim, device=device)
                cx = torch.zeros(1, sample_size, self.hidden_dim, device=device)
                last_hh = (hx, cx)

                ter = np.zeros(sample_size).astype(np.float32)
                decoder_input = static_hidden[:, :, env.n_nodes - 1].unsqueeze(2)
                time_step = 0
                while time_step < self.decode_len:
                    terminated = torch.from_numpy(ter).to(device)
                    for j in range(2):
                        if j == 0:
                            avail_actions_truck = torch.from_numpy(
                                avail_actions[:, :, 0]
                                .reshape([sample_size, env.n_nodes])
                                .astype(np.float32)
                            ).to(device)
                            dynamic_truck = torch.from_numpy(
                                np.expand_dims(state[:, :, 0], 2).astype(np.float32)
                            ).to(device)
                            idx_truck, _prob, _logp, last_hh = actor.forward(
                                static_hidden,
                                dynamic_truck,
                                decoder_input,
                                last_hh,
                                terminated,
                                avail_actions_truck,
                            )
                            b_s = np.where(
                                np.logical_and(
                                    avail_actions[:, :, 1].sum(axis=1) > 1,
                                    env.sortie == 0,
                                )
                            )[0]
                            avail_actions[b_s, idx_truck[b_s].cpu(), 1] = 0
                            avail_actions_drone = torch.from_numpy(
                                avail_actions[:, :, 1]
                                .reshape([sample_size, env.n_nodes])
                                .astype(np.float32)
                            ).to(device)
                            idx = idx_truck
                        else:
                            dynamic_drone = torch.from_numpy(
                                np.expand_dims(state[:, :, 1], 2).astype(np.float32)
                            ).to(device)
                            idx_drone, _prob, _logp, last_hh = actor.forward(
                                static_hidden,
                                dynamic_drone,
                                decoder_input,
                                last_hh,
                                terminated,
                                avail_actions_drone,
                            )
                            idx = idx_drone

                        decoder_input = torch.gather(
                            static_hidden,
                            2,
                            idx.view(-1, 1, 1).expand(sample_size, self.hidden_dim, 1),
                        ).detach()

                    state, avail_actions, ter, time_vec_truck, time_vec_drone = (
                        env.step(
                            idx_truck.cpu().numpy(),
                            idx_drone.cpu().numpy(),
                            time_vec_truck,
                            time_vec_drone,
                            ter,
                        )
                    )
                    time_step += 1

            R = copy.copy(env.current_time)
            best_rewards = float(R.min(axis=0))
            t = time.time() - initial_t
            times.append(t)
            best_rewards_list.append(best_rewards)

        np.savetxt(
            self.results_dir / f"best_rewards_list_{sample_size}_samples.txt",
            best_rewards_list,
        )
        actor.set_sample_mode(False)
        actor.train()
        return best_rewards_list, times

    def _save_checkpoint(self, prefix: str) -> None:
        torch.save(
            self.actor.state_dict(),
            self.ckpt_dir / f"{prefix}_actor_truck_params.pkl",
        )
        torch.save(
            self.critic.state_dict(),
            self.ckpt_dir / f"{prefix}_critic_params.pkl",
        )

    def _write_history(self, result: dict[str, Any]) -> None:
        path = Path(self.output_dir) / "history.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
