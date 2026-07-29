# TSPDrone-RL Architecture

This project solves the **Traveling Salesman Problem with Drone (TSPD)** using deep reinforcement learning (Advantage Actor–Critic / A2C). A truck and a drone cooperate to serve customers; the objective is to minimize the **makespan** (time until both vehicles finish and return).

Paper: [A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone](https://arxiv.org/abs/2112.12545)

---

## Repository layout

```
TSPDrone-RL/
├── main.py                     # Thin redirect → src.experiments.run
├── configs/
│   ├── train.yaml              # Hydra defaults
│   └── scale/{small,full}.yaml # Episode / batch budgets
├── src/
│   ├── config.py               # pydantic RunConfig
│   ├── constants.py            # problem / architecture literals
│   ├── paths.py                # repo + local_db roots
│   ├── utils.py                # seed / device helpers
│   ├── logs.py                 # file logger
│   ├── models/
│   │   ├── actor.py            # policy network
│   │   ├── encoder/            # graph attention static encoder
│   │   └── layers/             # attention / conv building blocks
│   ├── problems/
│   │   └── tspd.py             # DataGenerator + Env (no revisits)
│   ├── training/
│   │   ├── trainer.py          # REINFORCE train / test / sampling
│   │   ├── baselines.py        # greedy rollout + EMA baselines
│   │   ├── metrics.py
│   │   └── wandb_support.py
│   └── experiments/
│       └── run.py              # Hydra entrypoint
├── data/                       # Fixed test instances
├── trained_models/             # Legacy checkpoints (n11/, …)
├── copied_src/                 # Reference architecture only (not imported)
└── images/
```

Namespace packages under `src/` intentionally have no `__init__.py`.

---

## High-level data flow

```
Hydra (configs/train.yaml)
      │
      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  DataGenerator  │────▶│     Env      │◀────│    Trainer      │
│  (on-the-fly)   │     │ reset/step   │     │ train/test/samp │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                      │
                          ┌───────────────────────────┼───────────────────────────┐
                          ▼                           ▼                           ▼
                       Actor                   Rollout baseline              Optimizer
                   (policy π)               (frozen greedy copy)               Adam
```

1. **`src.experiments.run`** parses Hydra → `RunConfig`, seeds RNGs, builds data/env/actor, optionally loads `trained_models/n{N}/`, then dispatches `action=train|test|sampling`.
2. **`Trainer`** rolls out episodes: actor picks truck then drone destinations; `Env.step` advances simulation time.
3. Objective is minimizing makespan (`env.current_time`). A frozen **greedy rollout** of a baseline actor copy estimates `b(x)`; advantage = `R − b` drives the policy update (EMA warmup for the first episodes).

---

## Problem formulation (TSPD)

| Concept | Representation |
|--------|----------------|
| Nodes | Customers + depot (last index = depot) |
| Instance | Shape `[batch, n_nodes, 3]` → `(x, y, demand)` |
| Truck | Slower (`v_t`), can serve any remaining customer |
| Drone | Faster (`v_d`), launches/returns relative to truck (sortie logic) |
| Action (per step) | Pair `(idx_truck, idx_drone)` — next target for each vehicle |
| Termination | All customers served and both back at depot (or `decode_len` reached) |
| Objective | Minimize makespan `current_time` |

---

## Module responsibilities

### `src/experiments/run.py`

- Seeds NumPy / Python / PyTorch.
- Instantiates `DataGenerator`, `Env`, `Actor`, `Critic`.
- Loads best checkpoints if present under `trained_models/n{N}/` or the run output dir.
- Inits W&B when enabled.
- Dispatches train / greedy test / batch sampling.

### `src/config.py` + `configs/`

Validated pydantic config. Notable groups:

- **Physics:** `n_nodes`, `R`, `v_t`, `v_d`, `max_w`
- **Training:** `trainer.batch_size`, `trainer.epochs`, actor/critic LRs, `max_grad_norm`, `test_interval`
- **Model:** `hidden_dim` (default 256), `decode_len` (floored to `round(n_nodes * 1.8)`)
- **W&B:** `wandb.enabled`, required `wandb.name` when enabled

### `src/problems/tspd.py`

**`DataGenerator`** — training batches sampled on the fly; test loads or creates `data/DroneTruck-size-{test_size}-len-{n_nodes}.txt`.

**`Env`** — `reset` / `step` with distance matrices, sortie/return flags, and availability masks (same logic as the former `utils/env_no_comb.py`).

### `src/training/trainer.py`

Sampled truck→drone→step rollout, then greedy rollout baseline on the same batch:
`loss = mean((R − b).detach() * sum(log π))`. Periodic paired t-test may refresh the
frozen baseline actor (lower makespan = better). EMA warmup for the first
`baseline_warmup_episodes`.

### `src/models/`

```
Static (x,y) ──▶ GraphAttentionEncoder ──▶ static_hidden
Dynamic (time features) ──▶ Conv1d Encoder ──▶ dynamic_hidden
                              │
Decoder LSTM + Attention ─────┴──▶ logits over nodes ──▶ sample / greedy
```

Architecture kind is currently only `lstm_attention` (hook for future comparisons).

---

## Runtime modes

```bash
# Train (local, no W&B)
uv run python -m src.experiments.run \
  wandb.enabled=false \
  action=train \
  physics.n_nodes=11 \
  scale=small

# Greedy evaluation (loads trained_models/n11 when present)
uv run python -m src.experiments.run \
  wandb.enabled=false \
  action=test \
  physics.n_nodes=11

# Batch sampling
uv run python -m src.experiments.run \
  wandb.enabled=false \
  action=sampling \
  n_samples=5 \
  physics.n_nodes=11
```

---

## Artifacts

| Path | Contents |
|------|----------|
| `~/local_db/tspdrone-rl/outputs/training/.../n{N}/best_model_*_params.pkl` | Best actor/critic weights |
| `trained_models/n{N}/` | Legacy checkpoints (still auto-loaded) |
| `results/` | Per-instance completion times / sampling bests |
| `~/local_db/tspdrone-rl/log/run.log` | File logger |
