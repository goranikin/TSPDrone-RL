# Remote training commands

Commands for the TSPDrone-RL training pipelines on the remote box.

## Hardware

| Item | Value |
| --- | --- |
| GPUs | 2× NVIDIA GeForce RTX 4090 |
| VRAM | 24564 MiB each |
| Driver / CUDA | 570.211.01 / 12.8 |

Effective batch under DDP = `trainer.batch_size × num_processes`.

Recommended per-rank batch (n=11, bf16; measured ~5 GiB at 2048 on a 4090):

| Mode | `trainer.batch_size` | Effective batch |
| --- | ---: | ---: |
| 1 GPU | 8192 | 8192 |
| 2 GPU DDP | 8192 | 16384 |

If OOM, halve. If VRAM is idle and step/sec still rises, try doubling.

---

## One-time setup

```bash
cd ~/path/to/TSPDrone-RL   # repo root on the remote host

# Torch 2.13 Triton kernels need a host C compiler (gcc/clang).
# Without this, backward crashes with: Failed to find C compiler.
sudo apt-get update && sudo apt-get install -y build-essential

uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: 2.13.0+cu126 12.6 True   (cu130 needs a newer driver than 12.8)

nvidia-smi                 # confirm both 4090s are free
```

W&B runs need a name when logging is on:

```bash
export WANDB_API_KEY=...   # if not already logged in
```

---

## Smoke test (before long runs)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.experiments.run \
  wandb.enabled=false \
  action=train \
  decoder=attention_model dynamics=on \
  physics.n_nodes=11 scale=small \
  trainer.epochs=50 \
  trainer.batch_size=8192 \
  trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=smoke_am_on
```

---

## Single-GPU full train

One architecture on GPU 0 (GPU 1 free for another job):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.experiments.run \
  scale=full \
  action=train \
  decoder=attention_model dynamics=on \
  physics.n_nodes=11 \
  trainer.batch_size=8192 \
  trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=am_on_n11_bs8192 \
  wandb.group=n11_full_matrix
```

Swap `decoder` / `dynamics` as needed:

| `decoder=` | `dynamics=` | Suggested `wandb.name` |
| --- | --- | --- |
| `tspd_lstm` | `on` | `tspd_lstm_on_n11` |
| `tspd_lstm` | `off` | `tspd_lstm_off_n11` |
| `tspd_transformer` | `on` | `tspd_transformer_on_n11` |
| `tspd_transformer` | `off` | `tspd_transformer_off_n11` |
| `attention_model` | `on` | `am_on_n11` |
| `attention_model` | `off` | `am_off_n11` |
| `lstm_pointer` | `on` | `lstm_ptr_on_n11` |
| `lstm_pointer` | `off` | `lstm_ptr_off_n11` |

---

## Dual-GPU DDP (one job across both 4090s)

Use when you want **one** architecture trained faster / with larger effective batch:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  -m src.experiments.run \
  scale=full \
  action=train \
  decoder=attention_model dynamics=on \
  physics.n_nodes=11 \
  trainer.batch_size=8192 \
  trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=am_on_n11_ddp2_bs8192 \
  wandb.group=n11_full_matrix
```

---

## Parallel jobs (fastest way to finish the 6-run matrix)

Run **two different architectures at once** (one per GPU). Prefer this over DDP when sweeping the decoder × dynamics grid.

**Shell A — GPU 0**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.experiments.run \
  scale=full action=train \
  decoder=attention_model dynamics=on \
  physics.n_nodes=11 \
  trainer.batch_size=8192 trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=am_on_n11 wandb.group=n11_full_matrix
```

**Shell B — GPU 1**

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m src.experiments.run \
  scale=full action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=11 \
  trainer.batch_size=8192 trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n11 wandb.group=n11_full_matrix
```

Suggested schedule (3 rounds):

1. `attention_model_on` + `tspd_lstm_on`
2. `attention_model_off` + `tspd_lstm_off`
3. `lstm_pointer_on` + `lstm_pointer_off`

Detach with `tmux` / `screen` so SSH drops do not kill jobs:

```bash
tmux new -s tspd0
# paste GPU-0 command, Ctrl-b d to detach

tmux new -s tspd1
# paste GPU-1 command, Ctrl-b d to detach

tmux ls
tmux attach -t tspd0
```

---

## Larger node counts

Keep bf16; lower batch if OOM.

```bash
# n=20 example (single GPU)
CUDA_VISIBLE_DEVICES=0 uv run python -m src.experiments.run \
  scale=full action=train \
  decoder=attention_model dynamics=on \
  physics.n_nodes=20 \
  trainer.batch_size=512 \
  trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=am_on_n20_bs512
```

Rough starting batches:

| `physics.n_nodes` | 1 GPU batch | 2 GPU DDP per-rank |
| ---: | ---: | ---: |
| 11 | 8192 | 8192 |
| 20 | 512 | 256 |
| 50 | 256 | 128 |
| 100 | 128 | 64 |

---

## Eval / sampling (after training)

```bash
# Greedy test
CUDA_VISIBLE_DEVICES=0 uv run python -m src.experiments.run \
  action=test \
  decoder=attention_model dynamics=on \
  physics.n_nodes=11 scale=full \
  data.load_checkpoint=true \
  wandb.enabled=false \
  wandb.name=eval_am_on_n11

# Sampling
CUDA_VISIBLE_DEVICES=0 uv run python -m src.experiments.run \
  action=sampling \
  decoder=attention_model dynamics=on \
  physics.n_nodes=11 scale=full \
  n_samples=5 \
  data.load_checkpoint=true \
  wandb.enabled=false \
  wandb.name=sample_am_on_n11
```

Point `data.checkpoint_dir=` at the run’s `n{N}` checkpoint folder if auto-resolve misses it.

---

## Analyze W&B (on the remote or locally)

```bash
uv run python -m src.analyze run \
  --expected-scales full \
  --expected-modes rl \
  --expected-dynamics on off \
  --expected-seeds 5
```

Fetch only / analyze only:

```bash
uv run python -m src.analyze fetch
uv run python -m src.analyze analyze --expected-scales full
```

---

## Alpha sweep (stability re-run)

Re-run `tspd_lstm` + `dynamics=on` across `alpha ∈ {1.0, 1.2, 1.5}` with **logit clipping** and **fp32** (no AMP) after NaN crashes under bf16.

Keep both flags in sync: `--mixed_precision no` (accelerate) and `trainer.mixed_precision=no` (Hydra).

**n=11, batch=4096**

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision no \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=11 physics.alpha=1.0 \
  model.use_tanh=true \
  trainer.batch_size=4096 trainer.mixed_precision=no \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n11_a1.0_tanh_fp32 wandb.group=tspd_lstm_on_alpha_small

CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision no \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=11 physics.alpha=1.2 \
  model.use_tanh=true \
  trainer.batch_size=4096 trainer.mixed_precision=no \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n11_a1.2_tanh_fp32 wandb.group=tspd_lstm_on_alpha_small

CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision no \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=11 physics.alpha=1.5 \
  model.use_tanh=true \
  trainer.batch_size=4096 trainer.mixed_precision=no \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n11_a1.5_tanh_fp32 wandb.group=tspd_lstm_on_alpha_small
```

**n=20, batch=512**

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision no \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=20 physics.alpha=1.0 \
  model.use_tanh=true \
  trainer.batch_size=512 trainer.mixed_precision=no \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n20_a1.0_tanh_fp32 wandb.group=tspd_lstm_on_alpha_small

CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision no \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=20 physics.alpha=1.2 \
  model.use_tanh=true \
  trainer.batch_size=512 trainer.mixed_precision=no \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n20_a1.2_tanh_fp32 wandb.group=tspd_lstm_on_alpha_small

CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision no \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=20 physics.alpha=1.5 \
  model.use_tanh=true \
  trainer.batch_size=512 trainer.mixed_precision=no \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n20_a1.5_tanh_fp32 wandb.group=tspd_lstm_on_alpha_small
```

**n=50, batch=256** (includes the previously crashing `a1.2` setting)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision no \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=50 physics.alpha=1.0 \
  model.use_tanh=true \
  trainer.batch_size=256 trainer.mixed_precision=no \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n50_a1.0_tanh_fp32 wandb.group=tspd_lstm_on_alpha_small

CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision no \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=50 physics.alpha=1.2 \
  model.use_tanh=true \
  trainer.batch_size=256 trainer.mixed_precision=no \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n50_a1.2_tanh_fp32 wandb.group=tspd_lstm_on_alpha_small

CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision no \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=50 physics.alpha=1.5 \
  model.use_tanh=true \
  trainer.batch_size=256 trainer.mixed_precision=no \
  data.load_checkpoint=false \
  wandb.name=tspd_lstm_on_n50_a1.5_tanh_fp32 wandb.group=tspd_lstm_on_alpha_small
```

---

## Useful overrides

| Override | Meaning |
| --- | --- |
| `scale=small` / `scale=full` | 1k vs 1e6 updates (epochs) |
| `trainer.epochs=100000` | Cap updates without changing scale file |
| `trainer.batch_size=...` | Per-process batch |
| `trainer.mixed_precision=bf16` | Default; use `no` to debug AMP / NaN issues |
| `model.use_tanh=true` | Clip pointer logits with \(C\tanh\) (helps NaN stability) |
| `trainer.test_interval=200` | Val frequency (episodes) |
| `data.load_checkpoint=false` | Fresh weights (recommended for new encoder/decoder matrix) |
| `wandb.enabled=false` | Local-only run |
| `physics.n_nodes=11` | Problem size |
| `physics.alpha=2.0` | Drone/truck speed ratio (`v_d = alpha * v_t`) |
| `physics.v_t=1.0` | Truck speed (default 1) |
| `parameter_budget.enabled=true` | Match total params to `tspd_lstm_on` (default) |
| `parameter_budget.enabled=false` | Use raw `model.hidden_dim` / `model.d_ff` for every arch |

Print the matched width table:

```bash
uv run python -m src.experiments.parameter_budget
```

Outputs default under `~/local_db/tspdrone-rl/outputs/training/`.

---

## `tspd_transformer` vs `tspd_lstm` (α = 2.0)

Causal Transformer decoder with the **same** additive pointer / dynamics hook as `tspd_lstm`. Default `physics.alpha=2.0` (`v_d = 2.0`). Use `dynamics=on` for the paper-matched setting.

Parameter budget stays on (matches total params to `tspd_lstm_on`). Suggested batches from the table above; halve on OOM.

### Train (`scale=small`, 2-GPU DDP)

Effective batch = `trainer.batch_size × 2`. The Transformer keeps a KV cache over the
decode history, so it needs **smaller per-rank batches** than `tspd_lstm` (OOM at
8192). Halve again on OOM.

**n=11** (eff. batch 4096)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision bf16 \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_transformer dynamics=on \
  physics.n_nodes=11 physics.alpha=2.0 \
  trainer.batch_size=2048 trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=tspd_transformer_on_n11_a2.0 wandb.group=tspd_transformer_vs_lstm_small
```

**n=20** (eff. batch 256)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision bf16 \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_transformer dynamics=on \
  physics.n_nodes=20 physics.alpha=2.0 \
  trainer.batch_size=128 trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=tspd_transformer_on_n20_a2.0 wandb.group=tspd_transformer_vs_lstm_small
```

**n=50** (eff. batch 128)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes 2 --mixed_precision bf16 \
  -m src.experiments.run \
  scale=small action=train \
  decoder=tspd_transformer dynamics=on \
  physics.n_nodes=50 physics.alpha=2.0 \
  trainer.batch_size=64 trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=tspd_transformer_on_n50_a2.0 wandb.group=tspd_transformer_vs_lstm_small
```

Optional matched LSTM controls (same α / scale): swap `decoder=tspd_lstm` and
`wandb.name=tspd_lstm_on_n{N}_a2.0` (LSTM can keep the larger dual-GPU batches).

### Greedy test

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.experiments.run \
  action=test scale=small \
  decoder=tspd_transformer dynamics=on \
  physics.n_nodes=11 physics.alpha=2.0 \
  data.load_checkpoint=true \
  wandb.enabled=false \
  wandb.name=tspd_transformer_on_n11_a2.0
```

Repeat with `physics.n_nodes=20` / `50` and matching `wandb.name`.

For longer training, use `scale=full` with the same overrides.

---

## Monitor

```bash
watch -n 1 nvidia-smi
# or
nvtop
```

Confirm each parallel job is on a different GPU and memory is rising under load.

---

## Table 2 OR baselines (TSP-ep-all / DPS)

These rows are **not** the Python RL stack. In-repo Julia package:

`julia/TSPDroneBaselines/` (adapted from TSPDrone.jl; does not import that package).

Only external solver dep: **Concorde.jl**. Setup for macOS and Ubuntu 24.04: see `julia/TSPDroneBaselines/README.md`.

```bash
# one-time
julia --project=julia/TSPDroneBaselines -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.build("Concorde")'

# smoke (5 instances, N=20)
julia --project=julia/TSPDroneBaselines \
  julia/TSPDroneBaselines/scripts/reproduce_table2.jl \
  --n 20 --methods TSP-ep-all,DPS/10 --limit 5

# full Table 2 OR rows (TSP-ep-all @ N=100 is very slow)
julia --project=julia/TSPDroneBaselines \
  julia/TSPDroneBaselines/scripts/reproduce_table2.jl \
  --n 20,50,100 --methods TSP-ep-all,DPS/10,DPS/25
```
