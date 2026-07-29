"""Nearest-neighbor (smallest-distance) constructive heuristic for TSP-D.

Mirrors the RL decode loop: at each step choose a truck destination, then a
drone destination, among nodes allowed by ``Env`` availability masks.  Among
feasible nodes, pick the one with smallest Euclidean distance (truck) or
drone travel time (drone = distance / ``v_d``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.problems.tspd import Env


@dataclass(frozen=True)
class NearestNeighborResult:
    makespans: np.ndarray
    mean: float
    std: float
    min: float
    max: float
    n_nodes: int
    n_instances: int
    decode_len: int
    unfinished: int
    elapsed_sec: float
    v_d: float


def _env_cfg(*, n_nodes: int, v_d: float, batch_size: int) -> Any:
    return SimpleNamespace(
        physics=SimpleNamespace(n_nodes=n_nodes, v_d=v_d),
        trainer=SimpleNamespace(batch_size=batch_size),
    )


def _nearest_feasible(
    costs: np.ndarray,
    avail: np.ndarray,
) -> np.ndarray:
    """``costs`` / ``avail`` are ``[B, N]``; return argmin cost among available."""
    masked = np.where(avail > 0, costs, np.inf)
    # If a row is all-masked (should not happen after Env depot fallback), stay put.
    all_masked = ~np.isfinite(masked).any(axis=1)
    if all_masked.any():
        masked = masked.copy()
        masked[all_masked, 0] = 0.0
    return masked.argmin(axis=1).astype(np.int64)


def solve_nearest_neighbor(
    data: np.ndarray,
    *,
    n_nodes: int | None = None,
    v_d: float = 2.0,
    decode_len: int | None = None,
) -> tuple[np.ndarray, int]:
    """Solve a batch of TSP-D instances with the nearest-neighbor heuristic.

    Parameters
    ----------
    data:
        Array ``[B, N, 3]`` (x, y, demand) as used by ``Env``.
    n_nodes:
        Problem size; defaults to ``data.shape[1]``.
    v_d:
        Drone speed relative to truck (truck speed = 1).
    decode_len:
        Max synchronized truck/drone decision pairs. Defaults to
        ``max(round(N * 3), 30)`` so larger instances finish.

    Returns
    -------
    makespans:
        Per-instance completion times ``[B]``.
    unfinished:
        Count of instances that never returned both vehicles to the depot.
    """
    if data.ndim != 3:
        raise ValueError(f"expected data shape [B, N, 3], got {data.shape}")
    n_nodes = int(n_nodes if n_nodes is not None else data.shape[1])
    if data.shape[1] != n_nodes:
        raise ValueError(f"n_nodes={n_nodes} but data has N={data.shape[1]}")
    if decode_len is None:
        decode_len = max(round(n_nodes * 3), 30)

    batch_size = data.shape[0]
    env = Env(_env_cfg(n_nodes=n_nodes, v_d=v_d, batch_size=batch_size), data)
    env.input_data = data
    _, avail_actions = env.reset()

    ter = np.zeros(batch_size, dtype=np.float32)
    time_vec_truck = np.zeros([batch_size, 2])
    time_vec_drone = np.zeros([batch_size, 3])

    for _ in range(decode_len):
        if ter.all():
            break

        truck_costs = env.dist_mat[np.arange(batch_size), env.truck_loc]
        idx_truck = _nearest_feasible(truck_costs, avail_actions[:, :, 0])

        # Same conflict rule as the RL trainer: if the drone is free to start a
        # sortie, do not let it pick the truck's newly chosen customer.
        free = np.where(
            np.logical_and(
                avail_actions[:, :, 1].sum(axis=1) > 1,
                env.sortie == 0,
            )
        )[0]
        if free.size:
            avail_actions = avail_actions.copy()
            avail_actions[free, idx_truck[free], 1] = 0

        drone_costs = env.drone_mat[np.arange(batch_size), env.drone_loc]
        idx_drone = _nearest_feasible(drone_costs, avail_actions[:, :, 1])

        _, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(
            idx_truck,
            idx_drone,
            time_vec_truck,
            time_vec_drone,
            ter,
        )

    makespans = env.current_time.astype(np.float64)
    unfinished = int((1 - ter).sum())
    return makespans, unfinished
