import torch
from src.models.decoder.base import AutoregressiveDecoder
from src.models.decoding import action_embeddings
from src.models.initialization import initialize_pointer_network
from src.models.layers.additive_pointer_attention import AdditivePointerAttention
from src.types import EncoderOutput, ProblemDecodeState, SolutionOutput
from torch import nn

from src.constants import DecodeType


class LSTMPointerDecoder(AutoregressiveDecoder):
    """Single-layer LSTM and additive pointer scores from Vinyals et al."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        *,
        use_start_placeholder: bool,
        has_extra_stop: bool,
    ) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(d_model + context_dim, d_model)
        self.pointer = AdditivePointerAttention(
            encoder_dim=d_model,
            decoder_dim=d_model,
            attention_dim=d_model,
            has_extra_stop=has_extra_stop,
        )
        self.start_placeholder = (
            nn.Parameter(torch.empty(d_model)) if use_start_placeholder else None
        )
        self._hidden: torch.Tensor | None = None
        self._cell_state: torch.Tensor | None = None
        initialize_pointer_network(self)

    def decode(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        decode_type: DecodeType,
    ) -> SolutionOutput:
        self._hidden = encoder_output.graph_embedding
        self._cell_state = encoder_output.graph_embedding
        try:
            return super().decode(encoder_output, problem_state, decode_type)
        finally:
            self._hidden = None
            self._cell_state = None

    def step_logits(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._hidden is None or self._cell_state is None:
            raise RuntimeError("LSTM decoder state is not initialized")
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
        decoder_input = torch.cat([previous, context], dim=-1)
        self._hidden, self._cell_state = self.cell(
            decoder_input,
            (self._hidden, self._cell_state),
        )
        return self.pointer(node_embeddings, self._hidden, mask.size(1))
