from dataclasses import dataclass
from typing import Literal

from src.constants import DECODER_KINDS, PROBLEM_NAMES

type SolutionScope = Literal["full_topology", "partial_selection"]
type DecoderFamily = Literal[
    "attention",
    "recurrent",
    "transformer",
    "independent_subset",
]


@dataclass(frozen=True)
class ProblemMetadata:
    objective_sense: Literal["min", "max"]
    solution_scope: SolutionScope
    family: Literal[
        "full_routing", "hybrid_routing", "non_graph_subset", "graph_subset"
    ]


@dataclass(frozen=True)
class DecoderMetadata:
    family: DecoderFamily
    autoregressive: bool
    hypothesis_group: Literal["recurrent", "nonrecurrent"] | None


PROBLEMS: dict[str, ProblemMetadata] = {
    "tsp": ProblemMetadata("min", "full_topology", "full_routing"),
    "cvrp": ProblemMetadata("min", "full_topology", "full_routing"),
    "orienteering": ProblemMetadata("max", "partial_selection", "hybrid_routing"),
    "knapsack": ProblemMetadata("max", "partial_selection", "non_graph_subset"),
    "mis": ProblemMetadata("max", "partial_selection", "graph_subset"),
    "max_clique": ProblemMetadata("max", "partial_selection", "graph_subset"),
    "vertex_cover": ProblemMetadata("min", "partial_selection", "graph_subset"),
}

DECODERS: dict[str, DecoderMetadata] = {
    "attention_model": DecoderMetadata("attention", True, "nonrecurrent"),
    "attention_model_without_glimpse": DecoderMetadata(
        "attention", True, "nonrecurrent"
    ),
    "lstm_pointer": DecoderMetadata("recurrent", True, "recurrent"),
    "gru_pointer": DecoderMetadata("recurrent", True, "recurrent"),
    "transformer_pointer": DecoderMetadata("transformer", True, "nonrecurrent"),
    # This is still included in coverage, sanity, and descriptive comparisons. It is
    # excluded from the recurrent-vs-autoregressive hypothesis because it represents
    # a different independent subset policy family.
    "sigmoid_subset": DecoderMetadata("independent_subset", False, None),
}

EXPECTED_PROBLEMS: tuple[str, ...] = tuple(PROBLEM_NAMES)
EXPECTED_DECODERS: tuple[str, ...] = tuple(DECODER_KINDS)
EXPECTED_MODES: tuple[str, ...] = ("supervised", "rl")
DEFAULT_EXPECTED_SEEDS: tuple[int, ...] = (1234, 4321, 9999)

HISTORY_BASE_KEYS: frozenset[str] = frozenset(
    {"_step", "_timestamp", "_runtime", "epoch", "train/epoch"}
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
