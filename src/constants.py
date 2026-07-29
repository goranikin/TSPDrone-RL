from typing import Literal

type ProblemName = Literal["tspd"]
type ArchitectureKind = Literal["lstm_attention"]
type TrainingMode = Literal["rl"]
type RunAction = Literal["train", "test", "sampling"]
