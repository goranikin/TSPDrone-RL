import math

import torch
from torch import nn

from src.constants import DecodeType
from src.models.decoder.base import AutoregressiveDecoder
from src.models.decoding import action_embeddings, append_stop_embedding
from src.models.initialization import initialize_kool_linear, kool_uniform_
from src.types import EncoderOutput, ProblemDecodeState, SolutionOutput


class AttentionModelDecoder(AutoregressiveDecoder):
    """Attention Model decoder from Kool et al., equations (4)--(8)."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        *,
        num_heads: int = 8,
        include_first_node: bool = True,
        has_extra_stop: bool = False,
        tanh_clip: float = 10.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.context_dim = context_dim
        self.include_first_node = include_first_node
        self.tanh_clip = tanh_clip

        step_embedding_count = 2 if include_first_node else 1
        self.fixed_context_proj = nn.Linear(d_model, d_model, bias=False)
        self.step_context_proj = nn.Linear(
            step_embedding_count * d_model + context_dim,
            d_model,
            bias=False,
        )
        self.node_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.glimpse_out_proj = nn.Linear(d_model, d_model, bias=False)
        self.final_query_proj = nn.Linear(d_model, d_model, bias=False)
        self.previous_placeholder = (
            nn.Parameter(torch.empty(d_model)) if include_first_node else None
        )
        self.first_placeholder = (
            nn.Parameter(torch.empty(d_model)) if include_first_node else None
        )
        self.stop_logit_key = (
            nn.Parameter(torch.empty(d_model)) if has_extra_stop else None
        )

        self._fixed_context: torch.Tensor | None = None
        self._glimpse_keys: torch.Tensor | None = None
        self._glimpse_values: torch.Tensor | None = None
        self._logit_keys: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (
            self.node_proj,
            self.final_query_proj,
        ):
            initialize_kool_linear(projection)
        context_input_dim = (
            self.fixed_context_proj.in_features + self.step_context_proj.in_features
        )
        kool_uniform_(self.fixed_context_proj.weight, context_input_dim)
        kool_uniform_(self.step_context_proj.weight, context_input_dim)
        kool_uniform_(self.glimpse_out_proj.weight, self.head_dim)
        for parameter in (
            self.previous_placeholder,
            self.first_placeholder,
            self.stop_logit_key,
        ):
            if parameter is not None:
                kool_uniform_(parameter, self.d_model)

    def decode(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        decode_type: DecodeType,
    ) -> SolutionOutput:
        projected_nodes = self.node_proj(encoder_output.node_embeddings)
        self._glimpse_keys, self._glimpse_values, self._logit_keys = (
            projected_nodes.chunk(3, dim=-1)
        )
        self._fixed_context = self.fixed_context_proj(encoder_output.graph_embedding)
        try:
            return super().decode(encoder_output, problem_state, decode_type)
        finally:
            self._fixed_context = None
            self._glimpse_keys = None
            self._glimpse_values = None
            self._logit_keys = None

    def step_logits(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self._fixed_context is None
            or self._glimpse_keys is None
            or self._glimpse_values is None
            or self._logit_keys is None
        ):
            raise RuntimeError("Attention decoder projections are not initialized")
        node_embeddings = encoder_output.node_embeddings
        state = problem_state.state
        step_parts = [
            action_embeddings(
                node_embeddings,
                state.prev_action,
                self.previous_placeholder,
            )
        ]
        if self.include_first_node:
            step_parts.append(
                action_embeddings(
                    node_embeddings,
                    state.first_action,
                    self.first_placeholder,
                )
            )
        scalar_context = problem_state.problem.attention_context_features(state).to(
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        if scalar_context.size(1) != self.context_dim:
            raise ValueError(
                "Attention context width does not match the decoder configuration: "
                f"expected {self.context_dim}, got {scalar_context.size(1)}"
            )
        step_parts.append(scalar_context)
        query = self._fixed_context + self.step_context_proj(
            torch.cat(step_parts, dim=-1)
        )

        node_mask = mask[:, : node_embeddings.size(1)]
        glimpse = self._multi_head_glimpse(
            query,
            self._glimpse_keys,
            self._glimpse_values,
            node_mask,
        )
        final_query = self.final_query_proj(glimpse)
        logit_keys = append_stop_embedding(
            self._logit_keys,
            self.stop_logit_key,
            mask.size(1),
        )
        compatibility = torch.einsum("bd,bad->ba", final_query, logit_keys)
        compatibility = compatibility / math.sqrt(self.d_model)
        return self.tanh_clip * torch.tanh(compatibility)

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
            batch_size,
            node_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        value_heads = values.reshape(
            batch_size,
            node_count,
            self.num_heads,
            self.head_dim,
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
