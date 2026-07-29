# TSPDrone-RL Architecture

This project solves the **Traveling Salesman Problem with Drone (TSPD)** using deep reinforcement learning. A truck and a drone cooperate to serve customers; the objective is to minimize the **makespan**.

Paper: [A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone](https://arxiv.org/abs/2112.12545)

---

## Repository layout

```
TSPDrone-RL/
├── configs/train.yaml
├── src/
│   ├── config.py / constants.py / paths.py / utils.py / logs.py
│   ├── models/
│   │   ├── policy.py                 # shared encoder + decoder factory
│   │   ├── encoder/attention.py      # Kool AttentionEncoder (from compare-architectures)
│   │   ├── decoder/
│   │   │   ├── tspd_lstm.py          # paper LSTM + additive pointer
│   │   │   ├── attention_model.py    # Kool AM glimpse + pointer
│   │   │   └── lstm_pointer.py       # Vinyals LSTMCell + additive pointer
│   │   ├── layers/                   # MHA, BN, pointer layers
│   │   └── initialization.py
│   ├── problems/tspd.py
│   ├── training/                     # trainer, rollout baseline, W&B
│   └── experiments/run.py
└── copied_src/                       # reference only
```

---

## Decoder × dynamics matrix

Detailed write-up of the three decoders and the dynamics hook: [DECODERS.md](DECODERS.md).

Shared static encoder: ported Kool `AttentionEncoder` (`input_dim=2`).

| Architecture name | `decoder=` | `dynamics=` |
| --- | --- | --- |
| `tspd_lstm_on` | `tspd_lstm` | `on` |
| `tspd_lstm_off` | `tspd_lstm` | `off` |
| `attention_model_on` | `attention_model` | `on` |
| `attention_model_off` | `attention_model` | `off` |
| `lstm_pointer_on` | `lstm_pointer` | `on` |
| `lstm_pointer_off` | `lstm_pointer` | `off` |

- **on**: travel-time features pass through a `Linear(1→H)` and enter the decoder (paper fusion / AM context / LSTMCell concat).
- **off**: dynamic branch disabled.

```bash
uv run python -m src.experiments.run \
  wandb.enabled=false action=train \
  decoder=attention_model dynamics=on \
  physics.n_nodes=11 scale=small
```

Baseline: frozen greedy rollout of the same policy (`dynamics` and `decoder` match the train run).

**Checkpoints:** the encoder port invalidates legacy `trained_models/` weights. Auto-load is only attempted for `tspd_lstm_on` and is skipped if keys do not match.

---

## Training loop

1. Encode coordinates once per episode.
2. For each decode step: truck action then drone action (masked); `Env.step` advances time.
3. Advantage = sampled makespan − greedy-rollout baseline; REINFORCE update on Σ log π.
