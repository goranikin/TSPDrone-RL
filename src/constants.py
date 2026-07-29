from typing import Literal

type ProblemName = Literal["tspd"]
type ArchitectureKind = Literal["lstm_attention"]
type TrainingMode = Literal["rl"]
type RunAction = Literal["train", "test", "sampling"]
type ObjectiveSense = Literal["min"]

PROBLEM_NAMES: tuple[ProblemName, ...] = ("tspd",)
ARCHITECTURE_KINDS: tuple[ArchitectureKind, ...] = ("lstm_attention",)
