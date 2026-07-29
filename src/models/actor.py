import torch
from torch import nn

from src.models.encoder.graph_attention import AttentionEncoder
from src.models.layers.attention import ConvEncoder, PointerAttention


class Decoder(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
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

        self.encoder_attn = PointerAttention(hidden_size)
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(
        self,
        static_hidden: torch.Tensor,
        dynamic_hidden: torch.Tensor,
        decoder_input: torch.Tensor,
        last_hh: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        rnn_out, last_hh = self.lstm(decoder_input.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            hx = self.drop_hh(last_hh[0])
            cx = self.drop_hh(last_hh[1])
            last_hh = (hx, cx)
        hy = last_hh[0].squeeze(0)

        _, logits = self.encoder_attn(static_hidden, dynamic_hidden, hy)
        return logits, last_hh


class Actor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        mask_logits: bool = True,
        n_heads: int = 8,
        n_encode_layers: int = 3,
    ):
        super().__init__()

        self.mask_logits = mask_logits
        self.attention_encoder = AttentionEncoder(
            embedding_dim=hidden_size,
            hidden_dim=hidden_size,
            n_encode_layers=n_encode_layers,
            n_heads=n_heads,
        )
        self.dynamic_d_ex = ConvEncoder(1, hidden_size)
        self.decoder = Decoder(hidden_size, num_layers, dropout)
        self.logsoft = nn.LogSoftmax(dim=-1)
        self.Bignumber = 100000
        self.sample_mode = False

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def emd_stat(self, static: torch.Tensor) -> torch.Tensor:
        return self.attention_encoder(static)

    def forward(
        self,
        static_hidden: torch.Tensor,
        dynamic: torch.Tensor,
        decoder_input: torch.Tensor,
        last_hh: tuple[torch.Tensor, torch.Tensor],
        terminated: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        dynamic_hidden = self.dynamic_d_ex(dynamic.permute(0, 2, 1))

        logits, last_hh = self.decoder(
            static_hidden, dynamic_hidden, decoder_input, last_hh
        )
        if self.mask_logits:
            logits = logits.clone()
            logits[avail_actions == 0] = -self.Bignumber

        logprobs = self.logsoft(logits)
        probs = torch.exp(logprobs)

        if self.training or self.sample_mode:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            logp = m.log_prob(action)
        else:
            prob, action = torch.max(probs, 1)
            logp = prob.log()

        logp = logp * (1.0 - terminated)
        return action, probs, logp, last_hh

    def set_sample_mode(self, value: bool) -> None:
        self.sample_mode = value
