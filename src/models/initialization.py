import math

import torch
from torch import nn


def kool_uniform_(parameter: torch.Tensor, input_dim: int) -> None:
    """Initialize a parameter with Kool et al.'s input-width convention."""
    bound = 1.0 / math.sqrt(input_dim)
    nn.init.uniform_(parameter, -bound, bound)


def initialize_kool_linear(linear: nn.Linear) -> None:
    kool_uniform_(linear.weight, linear.in_features)
    if linear.bias is not None:
        kool_uniform_(linear.bias, linear.in_features)


def initialize_pointer_network(module: nn.Module) -> None:
    """Use the uniform initialization reported for Pointer Networks."""
    for parameter in module.parameters():
        nn.init.uniform_(parameter, -0.08, 0.08)
