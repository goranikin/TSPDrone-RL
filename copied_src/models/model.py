"""Top-level neural combinatorial optimization model."""

from typing import Any

import torch
import torch.nn.functional as F
from src.models.decoder.attention_model import AttentionModelDecoder
from src.models.decoder.attention_model_without_glimpse import (
    AttentionModelDecoderWithoutGlimpse,
)
from src.models.decoder.gru_pointer import GRUPointerDecoder
from src.models.decoder.lstm_pointer import LSTMPointerDecoder
from src.models.decoder.sigmoid_subset import SigmoidSubsetDecoder
from src.models.decoder.transformer_pointer import TransformerPointerDecoder
from src.models.decoding import actions_to_selected_mask
from src.models.encoder.attention import AttentionEncoder
from src.models.encoder.selection import encoder_supports_problem
from src.problems.base import Problem
from src.problems.registry import make_problem
from src.types import (
    EncoderOutput,
    ProblemDecodeState,
    SolutionOutput,
    SupervisedTarget,
)
from torch import nn

from src.constants import DecoderKind, DecodeType, EncoderKind, ProblemName
from src.models.encoder.graph_attention import GraphAttentionEncoder


class NCOModel(nn.Module):
    def __init__(
        self,
        *,
        problem: ProblemName,
        encoder_kind: EncoderKind,
        decoder_kind: DecoderKind,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        transformer_pointer_layers: int = 1,
        dropout: float = 0.0,
        tanh_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.problem_name = problem
        self.problem: Problem = make_problem(problem)
        self.encoder_kind = encoder_kind
        self.decoder_kind = decoder_kind
        encoder_type = (
            GraphAttentionEncoder
            if encoder_kind == "graph_attention"
            else AttentionEncoder
        )
        if not encoder_supports_problem(problem, encoder_kind):
            raise ValueError(
                f"encoder={encoder_kind} does not support problem={problem}"
            )
        self.encoder = encoder_type(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            depot_input_dim=self.problem.depot_input_dim,
        )

        if decoder_kind == "attention_model":
            self.decoder = AttentionModelDecoder(
                d_model=d_model,
                context_dim=self.problem.attention_context_dim,
                num_heads=num_heads,
                include_first_node=self.problem.attention_uses_first_node,
                has_extra_stop=self.problem.attention_has_extra_stop,
                tanh_clip=tanh_clip,
            )
        elif decoder_kind == "attention_model_without_glimpse":
            self.decoder = AttentionModelDecoderWithoutGlimpse(
                d_model=d_model,
                context_dim=self.problem.attention_context_dim,
                include_first_node=self.problem.attention_uses_first_node,
                has_extra_stop=self.problem.attention_has_extra_stop,
                tanh_clip=tanh_clip,
            )
        elif decoder_kind == "lstm_pointer":
            self.decoder = LSTMPointerDecoder(
                d_model=d_model,
                context_dim=self.problem.context_dim,
                use_start_placeholder=self.problem.attention_uses_first_node,
                has_extra_stop=self.problem.attention_has_extra_stop,
            )
        elif decoder_kind == "gru_pointer":
            self.decoder = GRUPointerDecoder(
                d_model=d_model,
                context_dim=self.problem.context_dim,
                use_start_placeholder=self.problem.attention_uses_first_node,
                has_extra_stop=self.problem.attention_has_extra_stop,
            )
        elif decoder_kind == "transformer_pointer":
            self.decoder = TransformerPointerDecoder(
                d_model=d_model,
                context_dim=self.problem.context_dim,
                num_heads=num_heads,
                d_ff=d_ff,
                num_layers=transformer_pointer_layers,
                dropout=dropout,
                tanh_clip=tanh_clip,
                use_start_placeholder=self.problem.attention_uses_first_node,
                has_extra_stop=self.problem.attention_has_extra_stop,
            )
        elif decoder_kind == "sigmoid_subset":
            self.decoder = SigmoidSubsetDecoder(d_model=d_model)
        else:
            raise ValueError(f"Unsupported decoder_kind: {decoder_kind}")

    def encode(self, batch: dict[str, Any]) -> EncoderOutput:
        node_features, adjacency, edge_features = self.problem.build_features(batch)
        node_mask = _node_mask(batch, node_features.size(1), node_features.device)
        return self.encoder(
            node_features,
            adjacency=adjacency,
            edge_features=edge_features,
            node_mask=node_mask,
        )

    def forward(
        self,
        batch: dict[str, Any],
        *,
        decode_type: DecodeType = "greedy",
        target_actions: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> SolutionOutput:
        encoder_output = self.encode(batch)
        state = self.problem.make_state(batch)
        decode_target_actions = target_actions
        decode_target_mask = None
        if self.problem.supervision_kind == "set":
            if target_mask is not None:
                decode_target_mask = target_mask
            elif target_actions is not None:
                decode_target_mask = actions_to_selected_mask(
                    target_actions,
                    state.selected_mask.size(1),
                    state.device,
                )
            if decode_target_mask is not None:
                decode_target_actions = None
        decode_state = ProblemDecodeState(
            problem=self.problem,
            batch=batch,
            state=state,
            target_actions=decode_target_actions,
            target_mask=decode_target_mask,
        )
        return self.decoder.decode(encoder_output, decode_state, decode_type)

    def supervised_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        target = self.problem.get_supervised_target(batch)
        if self.decoder_kind == "sigmoid_subset":
            return self._sigmoid_supervised_loss(batch, target)
        set_supervision = self.problem.supervision_kind == "set"
        if set_supervision and target.actions is None and target.selected_mask is None:
            raise ValueError(f"{self.problem_name} is missing a target set")
        if not set_supervision and target.actions is None:
            raise ValueError(f"{self.problem_name} is missing target_actions")
        output = self.forward(
            batch,
            decode_type="greedy",
            target_actions=target.actions,
            target_mask=target.selected_mask if set_supervision else None,
        )
        if output.log_probs is None:
            raise RuntimeError("Autoregressive decoder did not return log_probs")
        if set_supervision:
            if output.selected_mask is None:
                raise RuntimeError("Set decoder did not return a selected mask")
            valid_steps = output.selected_mask.sum(dim=1) + 1
        else:
            if target.actions is None:
                raise RuntimeError("Sequence supervision target is missing")
            valid_steps = (target.actions >= 0).sum(dim=1).clamp_min(1)
        valid_steps = valid_steps.to(output.log_probs.dtype)
        return -(output.log_probs / valid_steps).mean()

    def _sigmoid_supervised_loss(
        self,
        batch: dict[str, Any],
        target: SupervisedTarget,
    ) -> torch.Tensor:
        encoder_output = self.encode(batch)
        if not isinstance(self.decoder, SigmoidSubsetDecoder):
            raise TypeError("Expected SigmoidSubsetDecoder")
        logits = self.decoder.logits(encoder_output)
        target_mask = target.selected_mask
        if target_mask is None and target.actions is not None:
            target_mask = actions_to_selected_mask(
                target.actions,
                logits.size(1),
                logits.device,
            )
        if target_mask is None:
            raise ValueError(f"{self.problem_name} is missing target_mask")
        target_mask = target_mask.to(device=logits.device, dtype=logits.dtype)
        if target_mask.shape != logits.shape:
            if target_mask.size(1) == logits.size(1) - 1:
                pad = torch.zeros(
                    target_mask.size(0),
                    1,
                    dtype=target_mask.dtype,
                    device=target_mask.device,
                )
                target_mask = torch.cat([pad, target_mask], dim=1)
            else:
                raise ValueError(
                    f"target_mask shape {tuple(target_mask.shape)} does not match "
                    f"logits shape {tuple(logits.shape)}"
                )
        return F.binary_cross_entropy_with_logits(logits, target_mask)


def _node_mask(
    batch: dict[str, Any],
    node_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    num_nodes = batch.get("num_nodes")
    if not isinstance(num_nodes, torch.Tensor):
        return None
    positions = torch.arange(node_count, device=device).unsqueeze(0)
    return positions < num_nodes.to(device=device).unsqueeze(1)
