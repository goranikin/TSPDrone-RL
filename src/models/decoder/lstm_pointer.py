"""Vinyals LSTM pointer decoder adapted to one TSP-D step."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.decoder.base import StepDecoder
from src.models.initialization import initialize_pointer_network
from src.models.layers.additive_pointer_attention import AdditivePointerAttention
from src.models.types import EncoderOutput


class LstmPointerDecoder(StepDecoder):
    def __init__(
        self,
        d_model: int,
        *,
        use_dynamics: bool,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_dynamics = use_dynamics
        cell_input = d_model + (d_model if use_dynamics else 0)
        self.cell = nn.LSTMCell(cell_input, d_model)
        self.pointer = AdditivePointerAttention(
            encoder_dim=d_model,
            decoder_dim=d_model,
            attention_dim=d_model,
        )
        initialize_pointer_network(self)

    def reset(self, encoder_output: EncoderOutput, batch_size: int) -> Any:
        del batch_size
        graph = encoder_output.graph_embedding
        return {"hidden": graph, "cell": graph.clone()}

    def step(
        self,
        encoder_output: EncoderOutput,
        *,
        prev_embed: torch.Tensor,
        dynamic_hidden: torch.Tensor | None,
        state: Any,
        avail_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        del avail_actions
        parts = [prev_embed]
        if self.use_dynamics:
            if dynamic_hidden is None:
                raise RuntimeError("Dynamic context required when use_dynamics=True")
            parts.append(dynamic_hidden.mean(dim=2))
        decoder_input = torch.cat(parts, dim=-1)
        hidden, cell = self.cell(decoder_input, (state["hidden"], state["cell"]))
        logits = self.pointer(encoder_output.node_embeddings, hidden)
        return logits, {"hidden": hidden, "cell": cell}
