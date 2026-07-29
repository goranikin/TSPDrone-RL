import random
import time
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


@contextmanager
def timer(device: torch.device | None = None) -> Iterator[dict[str, float]]:
    payload = {"elapsed": 0.0}
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        yield payload
    finally:
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        payload["elapsed"] = time.perf_counter() - start
