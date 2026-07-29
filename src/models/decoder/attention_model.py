"""Kool Attention Model decoder adapted to one TSP-D step."""

import math
from typing import Any

import torch
from torch import nn

from src.models.decoder.base import StepDecoder
from src.models.initialization import initialize_kool_linear, kool_uniform_
from src.models.types import EncoderOutput


class AttentionModelDecoder(StepDecoder):
    def __init__(
        self,
        d_model: int,
        *,
        use_dynamics: bool,
        num_heads: int = 8,
        tanh_clip: float = 10.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_dynamics = use_dynamics
        self.tanh_clip = tanh_clip
        context_extra = d_model if use_dynamics else 0

        self.fixed_context_proj = nn.Linear(d_model, d_model, bias=False)
        self.step_context_proj = nn.Linear(
            d_model + context_extra,
            d_model,
            bias=False,
        )
        self.node_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.glimpse_out_proj = nn.Linear(d_model, d_model, bias=False)
        self.final_query_proj = nn.Linear(d_model, d_model, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        initialize_kool_linear(self.node_proj)
        initialize_kool_linear(self.final_query_proj)
        context_input_dim = (
            self.fixed_context_proj.in_features + self.step_context_proj.in_features
        )
        kool_uniform_(self.fixed_context_proj.weight, context_input_dim)
        kool_uniform_(self.step_context_proj.weight, context_input_dim)
        kool_uniform_(self.glimpse_out_proj.weight, self.head_dim)

    def reset(self, encoder_output: EncoderOutput, batch_size: int) -> Any:
        del batch_size
        projected = self.node_proj(encoder_output.node_embeddings)
        glimpse_keys, glimpse_values, logit_keys = projected.chunk(3, dim=-1)
        fixed_context = self.fixed_context_proj(encoder_output.graph_embedding)
        return {
            "fixed_context": fixed_context,
            "glimpse_keys": glimpse_keys,
            "glimpse_values": glimpse_values,
            "logit_keys": logit_keys,
        }

    def step(
        self,
        encoder_output: EncoderOutput,
        *,
        prev_embed: torch.Tensor,
        dynamic_hidden: torch.Tensor | None,
        state: Any,
        avail_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        del encoder_output
        parts = [prev_embed]
        if self.use_dynamics:
            if dynamic_hidden is None:
                raise RuntimeError("Dynamic context required when use_dynamics=True")
            # dynamic_hidden is [B, H, N] from DynamicEncoder; pool over nodes
            dyn_ctx = dynamic_hidden.mean(dim=2)
            parts.append(dyn_ctx)
        query = state["fixed_context"] + self.step_context_proj(
            torch.cat(parts, dim=-1)
        )
        node_mask = avail_actions == 0
        glimpse = self._multi_head_glimpse(
            query,
            state["glimpse_keys"],
            state["glimpse_values"],
            node_mask,
        )
        final_query = self.final_query_proj(glimpse)
        compatibility = torch.einsum("bd,bad->ba", final_query, state["logit_keys"])
        compatibility = compatibility / math.sqrt(self.d_model)
        logits = self.tanh_clip * torch.tanh(compatibility)
        return logits, state

    def _multi_head_glimpse(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, _ = keys.shape
        query_heads = query.reshape(batch_size, self.num_heads, 1, self.head_dim)
        key_heads = keys.reshape(
            batch_size, node_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        value_heads = values.reshape(
            batch_size, node_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        compatibility = torch.matmul(query_heads, key_heads.transpose(-2, -1))
        compatibility = compatibility / math.sqrt(self.head_dim)
        head_mask = mask[:, None, None, :]
        compatibility = compatibility.masked_fill(head_mask, float("-inf"))
        attention = torch.softmax(compatibility, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0).masked_fill(head_mask, 0.0)
        heads = torch.matmul(attention, value_heads)
        concatenated = heads.squeeze(2).reshape(batch_size, self.d_model)
        return self.glimpse_out_proj(concatenated)
