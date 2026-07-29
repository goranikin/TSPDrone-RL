from typing import Literal

type ProblemName = Literal[
    "tsp",
    "cvrp",
    "orienteering",
    "knapsack",
    "mis",
    "max_clique",
    "vertex_cover",
]
type EncoderKind = Literal["attention", "graph_attention"]
type EncoderSelection = EncoderKind | Literal["auto"]
type DecoderKind = Literal[
    "attention_model",
    "attention_model_without_glimpse",
    "lstm_pointer",
    "gru_pointer",
    "transformer_pointer",
    "sigmoid_subset",
]
type DecodeType = Literal["greedy", "sampling"]
type TrainingMode = Literal["supervised", "rl"]
type ObjectiveSense = Literal["min", "max"]
type SupervisionKind = Literal["sequence", "set"]
type DataSplit = Literal["train", "baseline", "val", "test"]
type DatasetSize = Literal[2000, 80000, 512000, 1280000]
type MatrixStage = Literal["all", "routing", "subset", "hybrid"]
type TrainingDataPolicy = Literal["repeat", "consume_once"]

PROBLEM_NAMES: tuple[ProblemName, ...] = (
    "tsp",
    "cvrp",
    "orienteering",
    "knapsack",
    "mis",
    "max_clique",
    "vertex_cover",
)
ENCODER_KINDS: tuple[EncoderKind, ...] = ("attention", "graph_attention")
DECODER_KINDS: tuple[DecoderKind, ...] = (
    "attention_model",
    "attention_model_without_glimpse",
    "lstm_pointer",
    "gru_pointer",
    "transformer_pointer",
    "sigmoid_subset",
)
GRAPH_PROBLEMS: frozenset[ProblemName] = frozenset(
    {"mis", "max_clique", "vertex_cover"}
)
