"""Policy factory: shared Kool encoder + selectable TSP-D decoder."""

from typing import Any

import torch
from torch import nn

from src.constants import DecoderKind, DynamicsMode
from src.models.decoder.attention_model import AttentionModelDecoder
from src.models.decoder.base import StepDecoder
from src.models.decoder.lstm_pointer import LstmPointerDecoder
from src.models.decoder.tspd_lstm import TSPDLstmDecoder
from src.models.encoder.attention import AttentionEncoder
from src.models.layers.pointer import ConvEncoder
from src.models.types import EncoderOutput


class Policy(nn.Module):
    """Static encoder + optional dynamic encoder + step decoder."""

    def __init__(
        self,
        *,
        encoder: AttentionEncoder,
        decoder: StepDecoder,
        dynamic_encoder: ConvEncoder | None,
        mask_logits: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dynamic_encoder = dynamic_encoder
        self.mask_logits = mask_logits
        self.mask_value = 100_000.0
        self.sample_mode = False
        self._decoder_state: Any = None
        self._encoder_output: EncoderOutput | None = None

    @property
    def use_dynamics(self) -> bool:
        return self.dynamic_encoder is not None

    def embed(self, coords: torch.Tensor) -> EncoderOutput:
        """Encode static coordinates ``[B, N, 2]``."""
        self._encoder_output = self.encoder(coords)
        return self._encoder_output

    def reset_episode(self, batch_size: int) -> None:
        if self._encoder_output is None:
            raise RuntimeError("Call embed() before reset_episode()")
        self._decoder_state = self.decoder.reset(self._encoder_output, batch_size)

    def encode_dynamic(self, dynamic: torch.Tensor) -> torch.Tensor | None:
        """``dynamic`` is ``[B, N, 1]`` travel-time features."""
        if self.dynamic_encoder is None:
            return None
        return self.dynamic_encoder(dynamic.permute(0, 2, 1))

    def forward(
        self,
        prev_embed: torch.Tensor,
        dynamic: torch.Tensor,
        terminated: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._encoder_output is None or self._decoder_state is None:
            raise RuntimeError("Call embed() and reset_episode() before forward()")
        dynamic_hidden = self.encode_dynamic(dynamic)
        logits, self._decoder_state = self.decoder.step(
            self._encoder_output,
            prev_embed=prev_embed,
            dynamic_hidden=dynamic_hidden,
            state=self._decoder_state,
            avail_actions=avail_actions,
        )
        if self.mask_logits:
            logits = logits.masked_fill(avail_actions == 0, -self.mask_value)

        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)
        if self.training or self.sample_mode:
            distribution = torch.distributions.Categorical(probs)
            action = distribution.sample()
            logp = distribution.log_prob(action)
        else:
            prob, action = torch.max(probs, 1)
            logp = prob.log()
        return action, logp * (1.0 - terminated)

    def node_embedding(self, index: torch.Tensor) -> torch.Tensor:
        """Gather node embeddings ``[B, H]`` for selected indices."""
        if self._encoder_output is None:
            raise RuntimeError("Call embed() before node_embedding()")
        nodes = self._encoder_output.node_embeddings
        batch_size, _, hidden = nodes.shape
        return nodes.gather(
            1, index.view(batch_size, 1, 1).expand(batch_size, 1, hidden)
        ).squeeze(1)

    def depot_embedding(self) -> torch.Tensor:
        if self._encoder_output is None:
            raise RuntimeError("Call embed() before depot_embedding()")
        return self._encoder_output.node_embeddings[:, -1, :]

    def set_sample_mode(self, value: bool) -> None:
        self.sample_mode = value


def build_decoder(
    decoder: DecoderKind,
    *,
    hidden_dim: int,
    use_dynamics: bool,
    num_layers: int,
    dropout: float,
    use_tanh: bool,
    n_heads: int,
    tanh_clip: float,
) -> StepDecoder:
    if decoder == "tspd_lstm":
        return TSPDLstmDecoder(
            hidden_dim,
            use_dynamics=use_dynamics,
            num_layers=num_layers,
            dropout=dropout,
            use_tanh=use_tanh,
        )
    if decoder == "attention_model":
        return AttentionModelDecoder(
            hidden_dim,
            use_dynamics=use_dynamics,
            num_heads=n_heads,
            tanh_clip=tanh_clip,
        )
    if decoder == "lstm_pointer":
        return LstmPointerDecoder(hidden_dim, use_dynamics=use_dynamics)
    raise ValueError(f"Unknown decoder: {decoder}")


def build_policy(
    *,
    decoder: DecoderKind,
    dynamics: DynamicsMode,
    hidden_dim: int,
    n_heads: int,
    n_encode_layers: int,
    d_ff: int,
    dropout: float,
    num_layers: int,
    use_tanh: bool,
    tanh_clip: float,
    mask_logits: bool,
) -> Policy:
    use_dynamics = dynamics == "on"
    encoder = AttentionEncoder(
        input_dim=2,
        d_model=hidden_dim,
        num_layers=n_encode_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout if dropout > 0 else 0.0,
    )
    step_decoder = build_decoder(
        decoder,
        hidden_dim=hidden_dim,
        use_dynamics=use_dynamics,
        num_layers=num_layers,
        dropout=dropout,
        use_tanh=use_tanh,
        n_heads=n_heads,
        tanh_clip=tanh_clip,
    )
    dynamic_encoder = ConvEncoder(1, hidden_dim) if use_dynamics else None
    return Policy(
        encoder=encoder,
        decoder=step_decoder,
        dynamic_encoder=dynamic_encoder,
        mask_logits=mask_logits,
    )
