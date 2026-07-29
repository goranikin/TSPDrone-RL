import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve a device string. CUDA is required; CPU/MPS fallbacks are disabled."""
    if device in {"auto", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required; torch.cuda.is_available() is False. "
                "Install a PyTorch build matching your NVIDIA driver, or upgrade "
                "the driver. CPU/MPS fallback is disabled."
            )
        return torch.device("cuda")
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError(
            f"CUDA is required; refusing non-CUDA device={resolved!s} "
            f"(from device={device!r})."
        )
    return resolved
