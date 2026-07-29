"""Match total trainable params across decoder architectures.

Canonical reference is ``tspd_lstm_on`` at the configured base ``hidden_dim`` /
``d_ff``. Other architectures search nearby ``hidden_dim`` (divisible by
``n_heads``) and ``d_ff`` to minimize |params - target|.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.constants import DecoderKind, DynamicsMode, architecture_name
from src.models.policy import build_policy


@dataclass(frozen=True)
class MatchedDimensions:
    decoder: DecoderKind
    dynamics: DynamicsMode
    hidden_dim: int
    d_ff: int
    matched_params: int
    target_params: int
    base_hidden_dim: int
    base_d_ff: int
    delta: int
    delta_pct: float
    source: str

    @property
    def architecture(self) -> str:
        return architecture_name(self.decoder, self.dynamics)


def count_policy_params(
    *,
    decoder: DecoderKind,
    dynamics: DynamicsMode,
    hidden_dim: int,
    d_ff: int,
    n_heads: int,
    n_encode_layers: int,
    dropout: float,
    num_layers: int,
    use_tanh: bool,
    tanh_clip: float,
    mask_logits: bool,
) -> int:
    policy = build_policy(
        decoder=decoder,
        dynamics=dynamics,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_encode_layers=n_encode_layers,
        d_ff=d_ff,
        dropout=dropout,
        num_layers=num_layers,
        use_tanh=use_tanh,
        tanh_clip=tanh_clip,
        mask_logits=mask_logits,
    )
    return sum(parameter.numel() for parameter in policy.parameters())


def find_closest_dimensions(
    *,
    decoder: DecoderKind,
    dynamics: DynamicsMode,
    target_params: int,
    n_heads: int,
    n_encode_layers: int,
    dropout: float,
    num_layers: int,
    use_tanh: bool,
    tanh_clip: float,
    mask_logits: bool,
    base_hidden_dim: int,
    base_d_ff: int,
    min_hidden_dim: int = 128,
    max_hidden_dim: int = 384,
    hidden_dim_step: int = 8,
) -> tuple[int, int, int]:
    """Return ``(hidden_dim, d_ff, matched_params)`` closest to ``target_params``."""
    if base_hidden_dim % n_heads:
        raise ValueError("base_hidden_dim must be divisible by n_heads")
    if hidden_dim_step % n_heads and base_hidden_dim % n_heads == 0:
        # Prefer steps that preserve divisibility; fall back to n_heads.
        hidden_dim_step = n_heads

    def evaluate(hidden_dim: int, d_ff: int) -> int:
        return count_policy_params(
            decoder=decoder,
            dynamics=dynamics,
            hidden_dim=hidden_dim,
            d_ff=d_ff,
            n_heads=n_heads,
            n_encode_layers=n_encode_layers,
            dropout=dropout,
            num_layers=num_layers,
            use_tanh=use_tanh,
            tanh_clip=tanh_clip,
            mask_logits=mask_logits,
        )

    # Canonical cell keeps the base width exactly.
    if decoder == "tspd_lstm" and dynamics == "on":
        params = evaluate(base_hidden_dim, base_d_ff)
        return base_hidden_dim, base_d_ff, params

    best: tuple[tuple[int, int, int], int, int, int] | None = None
    # Stage 1: locate a good hidden_dim with d_ff ≈ 2H.
    for hidden_dim in range(min_hidden_dim, max_hidden_dim + 1, hidden_dim_step):
        if hidden_dim % n_heads:
            continue
        params = evaluate(hidden_dim, 2 * hidden_dim)
        key = (abs(params - target_params), abs(hidden_dim - base_hidden_dim))
        if best is None or key < best[0]:
            best = (key, hidden_dim, 2 * hidden_dim, params)

    assert best is not None
    _, center_h, _, _ = best

    # Stage 2: refine d_ff around neighboring hidden dims.
    for hidden_dim in range(
        max(min_hidden_dim, center_h - 2 * hidden_dim_step),
        min(max_hidden_dim, center_h + 2 * hidden_dim_step) + 1,
        hidden_dim_step,
    ):
        if hidden_dim % n_heads:
            continue
        for d_ff in range(max(n_heads, hidden_dim // 2), 4 * hidden_dim + 1, 8):
            params = evaluate(hidden_dim, d_ff)
            key = (
                abs(params - target_params),
                abs(hidden_dim - base_hidden_dim),
                abs(d_ff - base_d_ff),
            )
            if key < best[0]:
                best = (key, hidden_dim, d_ff, params)

    _, hidden_dim, d_ff, params = best
    return hidden_dim, d_ff, params


def resolve_matched_dimensions(
    *,
    decoder: DecoderKind,
    dynamics: DynamicsMode,
    n_heads: int,
    n_encode_layers: int,
    dropout: float,
    num_layers: int,
    use_tanh: bool,
    tanh_clip: float,
    mask_logits: bool,
    base_hidden_dim: int,
    base_d_ff: int,
    match_target: str = "tspd_lstm_on",
    max_delta_pct: float = 1.0,
    strict: bool = True,
    min_hidden_dim: int = 128,
    max_hidden_dim: int = 384,
    hidden_dim_step: int = 8,
) -> MatchedDimensions:
    if match_target != "tspd_lstm_on":
        raise ValueError(
            f"Unsupported parameter_budget.match_target={match_target!r}; "
            "only 'tspd_lstm_on' is implemented"
        )

    target_params = count_policy_params(
        decoder="tspd_lstm",
        dynamics="on",
        hidden_dim=base_hidden_dim,
        d_ff=base_d_ff,
        n_heads=n_heads,
        n_encode_layers=n_encode_layers,
        dropout=dropout,
        num_layers=num_layers,
        use_tanh=use_tanh,
        tanh_clip=tanh_clip,
        mask_logits=mask_logits,
    )
    hidden_dim, d_ff, matched_params = find_closest_dimensions(
        decoder=decoder,
        dynamics=dynamics,
        target_params=target_params,
        n_heads=n_heads,
        n_encode_layers=n_encode_layers,
        dropout=dropout,
        num_layers=num_layers,
        use_tanh=use_tanh,
        tanh_clip=tanh_clip,
        mask_logits=mask_logits,
        base_hidden_dim=base_hidden_dim,
        base_d_ff=base_d_ff,
        min_hidden_dim=min_hidden_dim,
        max_hidden_dim=max_hidden_dim,
        hidden_dim_step=hidden_dim_step,
    )
    delta = matched_params - target_params
    delta_pct = 100.0 * delta / max(target_params, 1)
    if strict and abs(delta_pct) > max_delta_pct:
        raise RuntimeError(
            f"Parameter budget match for {architecture_name(decoder, dynamics)} "
            f"missed target by {delta_pct:.3f}% "
            f"(matched={matched_params:,}, target={target_params:,}, "
            f"max_delta_pct={max_delta_pct})"
        )
    source = (
        "canonical"
        if decoder == "tspd_lstm" and dynamics == "on"
        else "search"
    )
    return MatchedDimensions(
        decoder=decoder,
        dynamics=dynamics,
        hidden_dim=hidden_dim,
        d_ff=d_ff,
        matched_params=matched_params,
        target_params=target_params,
        base_hidden_dim=base_hidden_dim,
        base_d_ff=base_d_ff,
        delta=delta,
        delta_pct=delta_pct,
        source=source,
    )
