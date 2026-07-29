"""Select encoders that match each problem's graph structure."""

from collections.abc import Sequence

from src.constants import (
    GRAPH_PROBLEMS,
    EncoderKind,
    EncoderSelection,
    ProblemName,
)


def default_encoder_for_problem(problem: ProblemName) -> EncoderKind:
    if problem in GRAPH_PROBLEMS:
        return "graph_attention"
    return "attention"


def encoder_supports_problem(problem: ProblemName, encoder: EncoderKind) -> bool:
    return encoder != "graph_attention" or problem in GRAPH_PROBLEMS


def resolve_encoder_for_problem(
    problem: ProblemName,
    encoder: EncoderSelection,
) -> EncoderKind:
    resolved = default_encoder_for_problem(problem) if encoder == "auto" else encoder
    if not encoder_supports_problem(problem, resolved):
        raise ValueError(
            f"encoder={resolved} requires a sparse graph problem; got problem={problem}"
        )
    return resolved


def encoders_for_problem(
    problem: ProblemName,
    encoders: Sequence[EncoderKind] | None,
) -> tuple[EncoderKind, ...]:
    if encoders is None:
        return (default_encoder_for_problem(problem),)
    supported = tuple(
        encoder for encoder in encoders if encoder_supports_problem(problem, encoder)
    )
    if not supported:
        raise ValueError(f"No configured encoder supports problem={problem}")
    return supported
