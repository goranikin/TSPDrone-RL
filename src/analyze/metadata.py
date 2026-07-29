from dataclasses import dataclass
from typing import Literal

from src.constants import DECODER_KINDS, DYNAMICS_MODES, PROBLEM_NAMES

type SolutionScope = Literal["full_topology", "partial_selection"]
type DecoderFamily = Literal["attention", "recurrent", "hybrid"]


@dataclass(frozen=True)
class ProblemMetadata:
    objective_sense: Literal["min", "max"]
    solution_scope: SolutionScope
    family: Literal["full_routing", "hybrid_routing"]


@dataclass(frozen=True)
class DecoderMetadata:
    family: DecoderFamily
    autoregressive: bool
    hypothesis_group: Literal["recurrent", "nonrecurrent"] | None


PROBLEMS: dict[str, ProblemMetadata] = {
    "tspd": ProblemMetadata("min", "full_topology", "hybrid_routing"),
}

DECODERS: dict[str, DecoderMetadata] = {
    "tspd_lstm": DecoderMetadata("hybrid", True, "recurrent"),
    "attention_model": DecoderMetadata("attention", True, "nonrecurrent"),
    "lstm_pointer": DecoderMetadata("recurrent", True, "recurrent"),
}

EXPECTED_PROBLEMS: tuple[str, ...] = tuple(PROBLEM_NAMES)
EXPECTED_DECODERS: tuple[str, ...] = tuple(DECODER_KINDS)
EXPECTED_DYNAMICS: tuple[str, ...] = tuple(DYNAMICS_MODES)
EXPECTED_MODES: tuple[str, ...] = ("rl",)
DEFAULT_EXPECTED_SEEDS: tuple[int, ...] = (5,)
DEFAULT_ENCODER = "attention"

HISTORY_BASE_KEYS: frozenset[str] = frozenset(
    {
        "_step",
        "_timestamp",
        "_runtime",
        "epoch",
        "train/epoch",
        "train/episode",
    }
)
HISTORY_PREFIXES: tuple[str, ...] = ("train/", "val/", "test/")


def objective_sign(problem: str) -> int:
    """Return +1 when a larger raw objective is better, otherwise -1."""
    return 1 if PROBLEMS[problem].objective_sense == "max" else -1


def hypothesis_decoders() -> tuple[str, ...]:
    return tuple(
        decoder
        for decoder in EXPECTED_DECODERS
        if DECODERS[decoder].hypothesis_group is not None
    )
