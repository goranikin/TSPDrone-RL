import math

import torch
from torch import nn

from src.constants import DecodeType
from src.models.decoder.base import AutoregressiveDecoder
from src.models.decoding import action_embeddings, append_stop_embedding
from src.models.layers.causal_transformer_layer import CausalTransformerLayerCell
from src.types import EncoderOutput, ProblemDecodeState, SolutionOutput


class TransformerPointerDecoder(AutoregressiveDecoder):
    """Incremental causal Transformer followed by scaled dot-product pointers."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        *,
        num_heads: int = 8,
        d_ff: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        tanh_clip: float = 10.0,
        use_start_placeholder: bool,
        has_extra_stop: bool,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.tanh_clip = tanh_clip
        self.input_proj = nn.Linear(d_model + context_dim, d_model)
        self.initial_proj = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList(
            [
                CausalTransformerLayerCell(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.start_placeholder = (
            nn.Parameter(torch.empty(d_model)) if use_start_placeholder else None
        )
        self.stop_key = nn.Parameter(torch.empty(d_model)) if has_extra_stop else None
        self._layer_histories: list[torch.Tensor | None] | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.input_proj, self.initial_proj, self.key_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if self.start_placeholder is not None:
            nn.init.normal_(self.start_placeholder, std=0.02)
        if self.stop_key is not None:
            nn.init.normal_(self.stop_key, std=0.02)

    def decode(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        decode_type: DecodeType,
    ) -> SolutionOutput:
        initial_token = torch.tanh(self.initial_proj(encoder_output.graph_embedding))
        self._layer_histories = [None] * len(self.layers)
        try:
            self._advance_history(initial_token)
            return super().decode(encoder_output, problem_state, decode_type)
        finally:
            self._layer_histories = None

    def step_logits(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._layer_histories is None:
            raise RuntimeError("Transformer decoder history is not initialized")
        node_embeddings = encoder_output.node_embeddings
        state = problem_state.state
        previous = action_embeddings(
            node_embeddings,
            state.prev_action,
            self.start_placeholder,
        )
        context = problem_state.problem.context_features(state).to(
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        decoder_input = self.input_proj(torch.cat([previous, context], dim=-1))
        decoder_state = self._advance_history(decoder_input)
        keys = append_stop_embedding(
            self.key_proj(node_embeddings),
            self.stop_key,
            mask.size(1),
        )
        logits = torch.einsum("bd,bad->ba", decoder_state, keys)
        logits = logits / math.sqrt(self.d_model)
        return self.tanh_clip * torch.tanh(logits)

    def _advance_history(self, token: torch.Tensor) -> torch.Tensor:
        if self._layer_histories is None:
            raise RuntimeError("Transformer decoder history is not initialized")
        current = token.unsqueeze(1)
        for index, layer in enumerate(self.layers):
            current, layer_history = layer(current, self._layer_histories[index])
            self._layer_histories[index] = layer_history
        return self.output_norm(current.squeeze(1))
