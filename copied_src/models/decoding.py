"""Tensor operations shared by decoder models and pointer layers."""

import torch
import torch.nn.functional as F

from src.constants import DecodeType


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_probabilities = F.log_softmax(
        logits.masked_fill(mask, float("-inf")),
        dim=-1,
    )
    return torch.nan_to_num(log_probabilities, nan=0.0, neginf=0.0, posinf=0.0)


def select_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    decode_type: DecodeType,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_logits = logits.masked_fill(mask, float("-inf"))
    if decode_type == "greedy":
        action = masked_logits.argmax(dim=-1)
    else:
        probabilities = torch.softmax(masked_logits, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        action = torch.multinomial(probabilities, 1).squeeze(1)
    log_probability = masked_log_softmax(logits, mask).gather(
        1,
        action.unsqueeze(1),
    )
    return action, log_probability.squeeze(1)


def actions_to_selected_mask(
    actions: torch.Tensor,
    node_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert padded node/stop action sequences into an unordered node mask."""
    if actions.ndim != 2:
        raise ValueError("actions must have shape [batch, steps]")
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    actions = actions.to(device=device, dtype=torch.long)
    valid = (actions >= 0) & (actions < node_count)
    safe_actions = actions.clamp(min=0, max=node_count - 1)
    counts = torch.zeros(
        actions.size(0),
        node_count,
        dtype=torch.long,
        device=device,
    )
    counts.scatter_add_(1, safe_actions, valid.long())
    return counts > 0


def select_set_supervision_action(
    logits: torch.Tensor,
    invalid_mask: torch.Tensor,
    selected_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the best remaining target and return their combined log mass."""
    if logits.ndim != 2 or invalid_mask.shape != logits.shape:
        raise ValueError("logits and invalid_mask must have shape [batch, actions]")
    batch_size, action_count = logits.shape
    if selected_mask.ndim != 2 or selected_mask.size(0) != batch_size:
        raise ValueError("selected_mask must have shape [batch, nodes]")
    node_count = selected_mask.size(1)
    if action_count != node_count + 1:
        raise ValueError(
            "Set-valued supervision requires one stop action after the node actions"
        )
    if target_mask.shape != selected_mask.shape:
        raise ValueError(
            "target_mask must have shape "
            f"{tuple(selected_mask.shape)}; got {tuple(target_mask.shape)}"
        )

    targets = target_mask.to(device=logits.device, dtype=torch.bool)
    selected = selected_mask.to(device=logits.device, dtype=torch.bool)
    invalid = invalid_mask.to(device=logits.device, dtype=torch.bool)
    remaining = targets & ~selected
    invalid_targets = remaining & invalid[:, :node_count]
    if bool(invalid_targets.any()):
        raise ValueError(
            "The target set contains a node made infeasible by an earlier target action"
        )

    acceptable = torch.zeros_like(invalid)
    acceptable[:, :node_count] = remaining
    acceptable[:, node_count] = ~remaining.any(dim=1)
    if bool((acceptable & invalid).any()):
        raise ValueError("The stop action is invalid after completing the target set")

    log_probabilities = masked_log_softmax(logits, invalid)
    acceptable_log_probabilities = log_probabilities.masked_fill(
        ~acceptable,
        float("-inf"),
    )
    log_target_mass = torch.logsumexp(acceptable_log_probabilities, dim=1)
    action = logits.masked_fill(~acceptable, float("-inf")).argmax(dim=1)
    return action, log_target_mass


def action_embeddings(
    node_embeddings: torch.Tensor,
    action: torch.Tensor,
    placeholder: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, node_count, d_model = node_embeddings.shape
    safe_action = action.clamp(min=0, max=node_count - 1)
    selected = node_embeddings.gather(
        1,
        safe_action.view(batch_size, 1, 1).expand(-1, 1, d_model),
    ).squeeze(1)
    missing = action < 0
    if not bool(missing.any()):
        return selected
    if placeholder is None:
        raise ValueError(
            "A learned start placeholder is required before the first action"
        )
    return torch.where(
        missing.unsqueeze(1), placeholder.expand(batch_size, -1), selected
    )


def append_stop_embedding(
    embeddings: torch.Tensor,
    stop_embedding: torch.Tensor | None,
    action_count: int,
) -> torch.Tensor:
    if action_count == embeddings.size(1):
        return embeddings
    if action_count != embeddings.size(1) + 1 or stop_embedding is None:
        raise ValueError(
            "The action mask must match the node count or include one configured stop action"
        )
    return torch.cat(
        [
            embeddings,
            stop_embedding.view(1, 1, -1).expand(embeddings.size(0), 1, -1),
        ],
        dim=1,
    )
