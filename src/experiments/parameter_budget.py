"""Print matched parameter budgets for the decoder × dynamics matrix."""

from src.constants import DECODER_KINDS, DYNAMICS_MODES, architecture_name
from src.models.parameter_budget import resolve_matched_dimensions


def main() -> None:
    base_hidden_dim = 256
    base_d_ff = 512
    print(
        f"target = tspd_lstm_on @ hidden_dim={base_hidden_dim} d_ff={base_d_ff}\n"
    )
    print(
        f"{'architecture':28} {'H':>4} {'d_ff':>5} "
        f"{'params':>10} {'delta':>10} {'pct':>8}"
    )
    for decoder in DECODER_KINDS:
        for dynamics in DYNAMICS_MODES:
            matched = resolve_matched_dimensions(
                decoder=decoder,
                dynamics=dynamics,
                n_heads=8,
                n_encode_layers=3,
                dropout=0.1,
                num_layers=1,
                use_tanh=False,
                tanh_clip=10.0,
                mask_logits=True,
                base_hidden_dim=base_hidden_dim,
                base_d_ff=base_d_ff,
            )
            print(
                f"{architecture_name(decoder, dynamics):28} "
                f"{matched.hidden_dim:4d} {matched.d_ff:5d} "
                f"{matched.matched_params:10,} "
                f"{matched.delta:+10,} "
                f"{matched.delta_pct:+7.2f}%"
            )


if __name__ == "__main__":
    main()
