"""Paper TSP-D decoder: LSTM + additive pointer (optional dynamics)."""

from typing import Any

import torch
from torch import nn

from src.models.decoder.base import StepDecoder
from src.models.layers.pointer import PointerAttention
from src.models.types import EncoderOutput


class TSPDLstmDecoder(StepDecoder):
    def __init__(
        self,
        hidden_size: int,
        *,
        use_dynamics: bool,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_tanh: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_dynamics = use_dynamics
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            bias=False,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.encoder_attn = PointerAttention(
            hidden_size,
            use_dynamics=use_dynamics,
            use_tanh=use_tanh,
        )
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def reset(self, encoder_output: EncoderOutput, batch_size: int) -> Any:
        device = encoder_output.node_embeddings.device
        dtype = encoder_output.node_embeddings.dtype
        hx = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype
        )
        cx = torch.zeros_like(hx)
        return (hx, cx)

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
        # prev_embed: [B, H] → [B, H, 1] for LSTM
        decoder_input = prev_embed.unsqueeze(2)
        # node_embeddings [B, N, H] → static layout [B, H, N]
        static_hidden = encoder_output.node_embeddings.permute(0, 2, 1)
        rnn_out, last_hh = self.lstm(decoder_input.transpose(2, 1), state)
        rnn_out = self.drop_rnn(rnn_out.squeeze(1))
        if self.num_layers == 1:
            last_hh = (self.drop_hh(last_hh[0]), self.drop_hh(last_hh[1]))
        query = last_hh[0].squeeze(0)
        dyn = dynamic_hidden if self.use_dynamics else None
        logits = self.encoder_attn(static_hidden, query, dyn)
        return logits, last_hh
