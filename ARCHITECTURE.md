# TSPDrone-RL Architecture

This project solves the **Traveling Salesman Problem with Drone (TSPD)** using deep reinforcement learning (Advantage Actor–Critic / A2C). A truck and a drone cooperate to serve customers; the objective is to minimize the **makespan** (time until both vehicles finish and return).

Paper: [A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone](https://arxiv.org/abs/2112.12545)

---

## Repository layout

```
TSPDrone-RL/
├── main.py                 # Entry point: wire args → data → env → models → agent
├── pyproject.toml          # Project metadata & dependencies (uv / Python ≥ 3.13)
├── model/                  # Neural networks (policy & value)
│   ├── nnets.py            # Actor, Critic, Attention decoder (primary models)
│   ├── AttentionModel.py   # Graph-attention static encoder wrapper
│   └── graph_encoder.py    # Multi-head attention / GraphAttentionEncoder
├── utils/                  # Training loop, env, CLI, helpers
│   ├── agent.py            # A2CAgent: train / test / sampling
│   ├── env_no_comb.py      # Active Env + DataGenerator (no node revisits)
│   ├── env.py              # Alternate Env (legacy / with revisit variant)
│   ├── options.py          # argparse CLI → config dict
│   └── utils.py            # Logging helpers
├── data/                   # Fixed test instances (n = 11, 15, 20, 50, 100)
├── trained_models/         # Checkpoints per size: n11/, n15/, …
├── results/                # Inference outputs (greedy / sampling)
├── logs/                   # Training print logs
└── images/                 # Example solution figures
```

**Active path:** `main.py` imports `Env` and `DataGenerator` from `utils/env_no_comb.py` (customers are not revisited once served).

---

## High-level data flow

```
CLI (options.py)
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ DataGenerator│────▶│     Env      │◀────│   A2CAgent      │
│  (batches)   │     │ reset/step   │     │ train/test/samp │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          ▼                        ▼                        ▼
                       Actor                    Critic                 Optimizer
                   (policy π)                 (value V)              Adam × 2
```

1. **`main.py`** parses args, seeds RNGs, builds `DataGenerator` + `Env`, constructs `Actor` / `Critic`, optionally loads weights from `trained_models/n{N}/`, then runs train, greedy test, or batch sampling.
2. **`A2CAgent`** rolls out episodes: at each decode step the actor picks a **truck** destination, then a **drone** destination; `Env.step` advances simulation time.
3. Reward is the negative of completion time conceptually (the code minimizes `env.current_time`). The critic estimates expected cost; advantage = `R − V(s)` drives the actor update.

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

Test files under `data/` use the row format:

```text
x1 y1 d1  x2 y2 d2  …  xn yn dn
```

Demands are `1.0` for customers and `0.0` for the depot.

---

## Module responsibilities

### `main.py` — orchestration

- Seeds NumPy / Python / PyTorch.
- Instantiates data, env, actor, critic.
- Loads best checkpoints if present under `save_path/n{n_nodes}/`.
- Dispatches:
  - `--train=True` → `agent.train()`
  - else if `--sampling=True` → `agent.sampling_batch(n_samples)`
  - else → `agent.test()` (greedy decode)

### `utils/options.py` — configuration

All hyperparameters are CLI flags (booleans via `str2bool`). Notable groups:

- **Instance / physics:** `n_nodes`, `R` (drone range), `v_t`, `v_d`, `max_w`
- **Training:** `batch_size`, `n_train`, `actor_net_lr`, `critic_net_lr`, `max_grad_norm`, `decode_len`
- **Eval:** `train`, `sampling`, `n_samples`, `test_size`
- **Model size:** `hidden_dim` (default 256)

`decode_len` is floored to at least `round(n_nodes * 1.8)`.

### `utils/env_no_comb.py` — environment & data

**`DataGenerator`**

- Training: samples random points on the fly (`get_train_next`).
- Testing: loads or creates `data/DroneTruck-size-{test_size}-len-{n_nodes}.txt`.

**`Env`**

| Method | Role |
|--------|------|
| `reset()` | Build distance / drone-time matrices; init locations at depot; return dynamic features + availability masks |
| `step(idx_truck, idx_drone, …)` | Advance the earliest pending event; update locations, demand/`state`, sortie/return flags, masks |

Dynamic features encode travel times from each vehicle’s current location. Availability masks prevent illegal moves (already served nodes, unfinished moves, drone/truck coupling constraints).

### `utils/agent.py` — A2C loop

**Training episode (simplified):**

1. Encode static coordinates once: `static_hidden = actor.emd_stat(coords)`.
2. For `t = 1 … decode_len`:
   - Actor selects truck node (masked logits).
   - Optionally forbid that node for the drone if both are free.
   - Actor selects drone node.
   - `env.step(...)`.
3. Critic predicts cost from static layout + weights.
4. Losses:
   - Actor: `mean(advantage.detach() * sum(log π))`
   - Critic: `mean(advantage²)`
5. Periodic `test()` and checkpoint saves under `trained_models/`.

**Modes:**

| Mode | Behavior |
|------|----------|
| `train()` | Sample actions, update actor & critic |
| `test()` | Greedy `argmax`, write `results/test_results-*.txt` |
| `sampling_batch()` | Repeat each instance `n_samples` times with stochastic decode; keep best reward |

### `model/nnets.py` — Actor & Critic

```
Static (x,y) ──▶ GraphAttentionEncoder ──▶ static_hidden
Dynamic (time features) ──▶ Conv1d Encoder ──▶ dynamic_hidden
                              │
Decoder LSTM + Attention ─────┴──▶ logits over nodes ──▶ sample / greedy
```

- **`Actor`:** attention encoder for static graph + LSTM decoder with pointer-style attention; masks unavailable nodes; supports train / greedy / sample modes.
- **`Critic`:** attention over the static embedding conditioned on demand/weight features → scalar cost estimate.

### `model/graph_encoder.py` & `AttentionModel.py`

Implement the Kool et al. style **multi-head graph attention encoder** used for static node embeddings (from *Attention, Learn to Route*).

### `utils/utils.py`

`printOut` mirrors logs to file + stdout; `get_time()` for timestamped naming.

---

## Runtime modes (quick reference)

```bash
# Train
python main.py --train=True --n_nodes=11

# Greedy evaluation (load weights for that n_nodes if present)
python main.py --train=False --sampling=False --n_nodes=11

# Batch sampling
python main.py --train=False --sampling=True --n_samples=5 --n_nodes=11
```

Defaults live in `utils/options.py` (`train=False`, `sampling=True` at the time of writing—pass flags explicitly when debugging).

---

## Artifacts

| Path | Contents |
|------|----------|
| `trained_models/n{N}/best_model_actor_truck_params.pkl` | Best actor weights |
| `trained_models/n{N}/best_model_critic_params.pkl` | Best critic weights |
| `results/` | Per-instance completion times / sampling bests |
| `logs/results.txt` | Agent stdout mirror |

---

## Mental model for reading the code

1. Start at **`main.py`** to see what is constructed.
2. Read **`Env.reset` / `Env.step`** to understand legal actions and time advancement.
3. Read **`Actor.forward`** for how masks and sampling produce node indices.
4. Read **`A2CAgent.train`** for the full truck→drone→step→loss cycle.

That path covers almost all of the learning system; `env.py` is an alternate environment variant and is not used by the current entry point.
