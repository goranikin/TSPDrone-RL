"""TSP-D problem: on-the-fly data generation and truck-drone environment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from src.config import RunConfig
from src.paths import REPOSITORY_ROOT, resolve_user_path


def _resolve_repo_path(path: str) -> str:
    candidate = resolve_user_path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(REPOSITORY_ROOT / candidate)


def _cfg_dict(cfg: RunConfig) -> dict[str, Any]:
    """Flat dict used by Env/DataGenerator (mirrors legacy options)."""
    return {
        "random_seed": cfg.seed,
        "test_size": cfg.scale.test_size,
        "n_nodes": cfg.physics.n_nodes,
        "data_dir": _resolve_repo_path(cfg.data.data_dir),
        "batch_size": cfg.trainer.batch_size,
        "v_t": cfg.physics.v_t,
        "v_d": cfg.physics.v_d,
        "R": cfg.physics.R,
        "max_w": cfg.physics.max_w,
        "decode_len": cfg.model.decode_len,
        "hidden_dim": cfg.model.hidden_dim,
    }


def create_test_dataset(args: dict[str, Any]) -> np.ndarray:
    n_problems = args["test_size"]
    n_nodes = args["n_nodes"]
    data_dir = args["data_dir"]
    task_name = f"DroneTruck-size-{n_problems}-len-{n_nodes}.txt"
    fname = Path(data_dir) / task_name

    if fname.exists():
        print(f"Loading dataset for {task_name}...")
        data = np.loadtxt(fname, delimiter=" ")
        return data.reshape(-1, n_nodes, 3)

    print(f"Creating dataset for {task_name}...")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    input_pnt = np.random.uniform(1, 100, size=(n_problems, n_nodes - 1, 2))
    input_pnt = np.concatenate(
        [input_pnt, np.random.uniform(0, 1, size=(n_problems, 1, 2))], axis=1
    )
    demand = np.ones([n_problems, n_nodes - 1, 1])
    network = np.concatenate([demand, np.zeros([n_problems, 1, 1])], 1)
    input_data = np.concatenate([input_pnt, network], 2)
    np.savetxt(fname, input_data.reshape(-1, n_nodes * 3))
    return input_data


class DataGenerator:
    def __init__(self, cfg: RunConfig | dict[str, Any]):
        self.args = cfg if isinstance(cfg, dict) else _cfg_dict(cfg)
        self.rnd = np.random.RandomState(seed=self.args["random_seed"])
        self.test_data = create_test_dataset(self.args)

    def get_train_next(self) -> np.ndarray:
        args = self.args
        batch_size = args["batch_size"]
        n_nodes = args["n_nodes"]
        input_pnt = np.random.uniform(1, 100, size=(batch_size, n_nodes - 1, 2))
        input_pnt = np.concatenate(
            [input_pnt, np.random.uniform(0, 1, size=(batch_size, 1, 2))], axis=1
        )
        demand = np.ones([batch_size, n_nodes - 1, 1])
        network = np.concatenate([demand, np.zeros([batch_size, 1, 1])], 1)
        return np.concatenate([input_pnt, network], 2)

    def get_test_all(self) -> np.ndarray:
        return self.test_data


class Env:
    """Active TSP-D environment (no customer revisits)."""

    def __init__(self, cfg: RunConfig | dict[str, Any], data: np.ndarray):
        self.args = cfg if isinstance(cfg, dict) else _cfg_dict(cfg)
        self.rnd = np.random.RandomState(seed=self.args["random_seed"])
        self.input_data = data
        self.n_nodes = self.args["n_nodes"]
        self.v_t = self.args["v_t"]
        self.v_d = self.args["v_d"]
        self.batch_size = self.args["batch_size"]
        print("Using Not revisiting nodes")

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.batch_size = self.input_data[:, :, :2].shape[0]
        self.input_pnt = self.input_data[:, :, :2]
        self.dist_mat = np.zeros([self.batch_size, self.n_nodes, self.n_nodes])
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                self.dist_mat[:, i, j] = (
                    (self.input_pnt[:, i, 0] - self.input_pnt[:, j, 0]) ** 2
                    + (self.input_pnt[:, i, 1] - self.input_pnt[:, j, 1]) ** 2
                ) ** 0.5
                self.dist_mat[:, j, i] = self.dist_mat[:, i, j]

        self.drone_mat = self.dist_mat / self.v_d
        avail_actions = np.ones([self.batch_size, self.n_nodes, 2], dtype=np.float32)
        avail_actions[:, self.n_nodes - 1, :] = np.zeros([self.batch_size, 2])
        self.state = np.ones([self.batch_size, self.n_nodes])
        self.state[:, self.n_nodes - 1] = np.zeros([self.batch_size])
        self.sortie = np.zeros(self.batch_size)
        self.returned = np.ones(self.batch_size)
        self.current_time = np.zeros(self.batch_size)
        self.truck_loc = np.ones([self.batch_size], dtype=np.int32) * (self.n_nodes - 1)
        self.drone_loc = np.ones([self.batch_size], dtype=np.int32) * (self.n_nodes - 1)

        dynamic = np.zeros([self.batch_size, self.n_nodes, 2], dtype=np.float32)
        dynamic[:, :, 0] = self.dist_mat[:, self.n_nodes - 1]
        dynamic[:, :, 1] = self.drone_mat[:, self.n_nodes - 1]
        return dynamic, avail_actions

    def step(
        self,
        idx_truck: np.ndarray,
        idx_drone: np.ndarray,
        time_vec_truck: np.ndarray,
        time_vec_drone: np.ndarray,
        terminated: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        old_sortie = copy.copy(self.sortie)

        t_truck = self.dist_mat[
            np.arange(self.batch_size, dtype=np.int64), self.truck_loc, idx_truck
        ]
        t_drone = self.drone_mat[
            np.arange(self.batch_size, dtype=np.int64), self.drone_loc, idx_drone
        ]
        A = (
            t_truck
            + np.equal(t_truck, np.zeros(self.batch_size)).astype(int)
            * np.ones(self.batch_size)
            * 10000
        )
        B = (
            t_drone
            + np.equal(t_drone, np.zeros(self.batch_size)).astype(int)
            * np.ones(self.batch_size)
            * 10000
        )
        C = (
            time_vec_truck[:, 1]
            + np.equal(time_vec_truck[:, 1], np.zeros(self.batch_size)).astype(int)
            * np.ones(self.batch_size)
            * 10000
        )
        D = (
            time_vec_drone[:, 1]
            + np.equal(time_vec_drone[:, 1], np.zeros(self.batch_size)).astype(int)
            * np.ones(self.batch_size)
            * 10000
        )
        time_step = np.minimum.reduce([A, B, C, D])

        b_s = np.where(terminated == 1)[0]
        time_step[b_s] = np.zeros(len(b_s))

        self.time_step = time_step
        self.current_time += time_step

        time_vec_truck[:, 1] += np.logical_and(
            np.equal(time_vec_truck[:, 1], np.zeros(self.batch_size)),
            np.greater(t_truck, np.zeros(self.batch_size)),
        ).astype(int) * (t_truck - time_step) - np.greater(
            time_vec_truck[:, 1], np.zeros(self.batch_size)
        ) * (time_step)

        time_vec_drone[:, 1] += np.logical_and(
            np.equal(time_vec_drone[:, 1], np.zeros(self.batch_size)),
            np.greater(t_drone, np.zeros(self.batch_size)),
        ).astype(int) * (t_drone - time_step) - np.greater(
            time_vec_drone[:, 1], np.zeros(self.batch_size)
        ) * (time_step)

        self.truck_loc += np.equal(time_vec_truck[:, 1], np.zeros(self.batch_size)) * (
            idx_truck - self.truck_loc
        )
        self.drone_loc += np.equal(time_vec_drone[:, 1], np.zeros(self.batch_size)) * (
            idx_drone - self.drone_loc
        )
        time_vec_truck[:, 0] = (
            np.logical_and(
                np.less(time_step, t_truck),
                np.greater(time_vec_truck[:, 1], np.zeros(self.batch_size)),
            )
            * idx_truck
        )
        time_vec_drone[:, 0] = (
            np.logical_and(
                np.less(time_step, t_drone),
                np.greater(time_vec_drone[:, 1], np.zeros(self.batch_size)),
            )
            * idx_drone
        )

        b_s = np.where(np.equal(time_vec_truck[:, 1], np.zeros(self.batch_size)))[0]
        self.state[b_s, idx_truck[b_s]] = np.zeros(len(b_s))
        idx_satis = np.where(
            np.less(
                self.sortie - np.equal(time_vec_drone[:, 1], 0),
                np.zeros(self.batch_size),
            )
        )[0]
        self.state[idx_satis, idx_drone[idx_satis]] -= (
            np.equal(time_vec_drone[idx_satis, 1], np.zeros(len(idx_satis)))
            * self.state[idx_satis, idx_drone[idx_satis]]
        )
        self.sortie[idx_satis] = np.ones(len(idx_satis))
        a = np.equal(
            (self.truck_loc == self.drone_loc).astype(int)
            + (time_vec_drone[:, 1] == 0).astype(int)
            + (time_vec_truck[:, 1] == 0).astype(int),
            3,
        )
        idx_stais = np.where(np.expand_dims(a, 1))[0]
        self.sortie[idx_stais] = np.zeros(len(idx_stais))
        self.returned = np.ones(self.batch_size) - np.equal(
            (old_sortie == 1).astype(int)
            + (self.sortie == 1).astype(int)
            + (time_vec_drone[:, 1] == 0).astype(int),
            3,
        )
        self.returned[idx_stais] = np.ones(len(idx_stais))

        avail_actions = np.zeros([self.batch_size, self.n_nodes, 2], dtype=np.float32)

        b_s = np.where(np.expand_dims(time_vec_truck[:, 1], 1) > 0)[0]
        idx_fixed = time_vec_truck[b_s, np.zeros(len(b_s), dtype=np.int64)]
        avail_actions[b_s, idx_fixed.astype(int), 0] = np.ones(len(b_s))
        b_s_d = np.where(np.expand_dims(time_vec_drone[:, 1], 1) > 0)[0]
        idx_fixed_d = time_vec_drone[b_s_d, np.zeros(len(b_s_d), dtype=np.int64)]
        avail_actions[b_s_d, idx_fixed_d.astype(int), 1] = np.ones(len(b_s_d))

        a = np.equal(
            np.greater_equal(time_vec_truck[:, 1], 0).astype(int)
            + np.equal(time_vec_drone[:, 1], 0).astype(int),
            2,
        )
        b_s = np.where(np.expand_dims(a, 1))[0]
        avail_actions[b_s, :, 1] = np.greater(self.state[b_s, :], 0)

        a = np.equal(
            np.equal(self.returned, 0).astype(int)
            + np.equal(time_vec_drone[:, 1], 0).astype(int),
            2,
        )
        b_s = np.where(np.expand_dims(a, 1))[0]
        avail_actions[b_s, :, 1] = 0
        avail_actions[b_s, self.drone_loc[b_s], 1] = 1

        b_s = np.where(np.expand_dims(time_vec_truck[:, 1], 1) == 0)[0]
        avail_actions[b_s, :, 0] += np.greater(self.state[b_s, :], 0)

        a = np.equal(
            np.equal(self.sortie, 0).astype(int)
            + np.greater(time_vec_drone[:, 1], 0).astype(int)
            + np.equal(time_vec_truck[:, 1], 0).astype(int),
            3,
        )
        b_s_s = np.where(np.expand_dims(a, 1))[0]
        idx_fixed_d = time_vec_drone[b_s_s, np.zeros(len(b_s_s), dtype=np.int64)]
        avail_actions[b_s_s, idx_fixed_d.astype(int), 0] = 0

        a = np.equal(
            np.equal(self.truck_loc, time_vec_drone[:, 0]).astype(int)
            + np.greater(time_vec_drone[:, 1], 0).astype(int)
            + np.equal(time_vec_truck[:, 1], 0).astype(int),
            3,
        )
        b_s = np.where(np.expand_dims(a, 1))[0]
        avail_actions[b_s, self.truck_loc[b_s], 0] = 1

        a = np.equal(
            np.equal(self.returned, 0).astype(int)
            + np.equal(time_vec_drone[:, 1], 0).astype(int)
            + np.equal(time_vec_truck[:, 1], 0).astype(int),
            3,
        )
        b_s = np.where(np.expand_dims(a, 1))[0]
        avail_actions[b_s, self.drone_loc[b_s], 0] = 1

        a = np.equal(
            np.equal(self.state.sum(axis=1), 1).astype(int)
            + np.equal(
                (avail_actions[:, :, 0] == avail_actions[:, :, 1]).sum(axis=1),
                self.n_nodes,
            ).astype(int)
            + np.equal(self.drone_loc, self.truck_loc).astype(int),
            3,
        )
        b_s = np.where(a)[0]
        avail_actions[b_s, :, 0] = np.zeros(self.n_nodes)

        a = np.equal(
            np.equal(self.state.sum(axis=1), 1).astype(int)
            + np.equal(time_vec_drone[:, 1], 0).astype(int)
            + np.greater(time_vec_truck[:, 1], 0).astype(int)
            + np.equal(self.returned, 1).astype(int),
            4,
        )
        b_s = np.where(a)[0]
        avail_actions[b_s, :, 1] = np.zeros(self.n_nodes)
        avail_actions[:, self.n_nodes - 1, 0] += np.equal(
            avail_actions[:, :, 0].sum(axis=1), 0
        )
        avail_actions[:, self.n_nodes - 1, 1] += np.equal(
            avail_actions[:, :, 1].sum(axis=1), 0
        )

        dynamic = np.zeros([self.batch_size, self.n_nodes, 2], dtype=np.float32)
        dynamic[:, :, 0] = self.dist_mat[np.arange(self.batch_size), self.truck_loc]
        dynamic[:, :, 1] = self.drone_mat[np.arange(self.batch_size), self.drone_loc]

        terminated = np.logical_and(
            np.equal(self.truck_loc, self.n_nodes - 1),
            np.equal(self.drone_loc, self.n_nodes - 1),
        ).astype(int)
        return dynamic, avail_actions, terminated, time_vec_truck, time_vec_drone
