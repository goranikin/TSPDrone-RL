"""TSP-D constructive heuristics (non-learning baselines)."""

from src.heuristics.nearest_neighbor import (
    NearestNeighborResult,
    solve_nearest_neighbor,
)

__all__ = [
    "NearestNeighborResult",
    "solve_nearest_neighbor",
]
