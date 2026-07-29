import math

import torch
from torch import nn

from src.constants import DecodeType
from src.models.decoder.base import AutoregressiveDecoder
from src.models.decoding import action_embeddings, append_stop_embedding
from src.models.initialization import initialize_kool_linear, kool_uniform_
from src.types import EncoderOutput, ProblemDecodeState, SolutionOutput


class AttentionModelDecoderWithoutGlimpse(AutoregressiveDecoder):
    """Attention Model context and pointer with encoder cross-attention removed."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        *,
        include_first_node: bool = True,
        has_extra_stop: bool = False,
        tanh_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
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
        self.node_key_proj = nn.Linear(d_model, d_model, bias=False)
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
        self._logit_keys: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_kool_linear(self.node_key_proj)
        initialize_kool_linear(self.final_query_proj)
        context_input_dim = (
            self.fixed_context_proj.in_features + self.step_context_proj.in_features
        )
        kool_uniform_(self.fixed_context_proj.weight, context_input_dim)
        kool_uniform_(self.step_context_proj.weight, context_input_dim)
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
        self._logit_keys = self.node_key_proj(encoder_output.node_embeddings)
        self._fixed_context = self.fixed_context_proj(encoder_output.graph_embedding)
        try:
            return super().decode(encoder_output, problem_state, decode_type)
        finally:
            self._fixed_context = None
            self._logit_keys = None

    def step_logits(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._fixed_context is None or self._logit_keys is None:
            raise RuntimeError("No-glimpse decoder projections are not initialized")
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

        # Unlike AttentionModelDecoder, this query is not updated by a multi-head
        # glimpse over encoder nodes before pointer compatibility is computed.
        final_query = self.final_query_proj(query)
        logit_keys = append_stop_embedding(
            self._logit_keys,
            self.stop_logit_key,
            mask.size(1),
        )
        compatibility = torch.einsum("bd,bad->ba", final_query, logit_keys)
        compatibility = compatibility / math.sqrt(self.d_model)
        return self.tanh_clip * torch.tanh(compatibility)
