from typing import Literal

type ProblemName = Literal["tspd"]
type DecoderKind = Literal["tspd_lstm", "attention_model", "lstm_pointer"]
type DynamicsMode = Literal["on", "off"]
type TrainingMode = Literal["rl"]
type RunAction = Literal["train", "test", "sampling"]

DECODER_KINDS: tuple[DecoderKind, ...] = (
    "tspd_lstm",
    "attention_model",
    "lstm_pointer",
)
DYNAMICS_MODES: tuple[DynamicsMode, ...] = ("on", "off")


def architecture_name(decoder: DecoderKind, dynamics: DynamicsMode) -> str:
    return f"{decoder}_{dynamics}"
