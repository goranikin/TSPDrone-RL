# Remote training commands

Commands for the TSPDrone-RL training pipelines on the remote box.

## Hardware

| Item | Value |
| --- | --- |
| GPUs | 2× NVIDIA GeForce RTX 4090 |
| VRAM | 24564 MiB each |
| Driver / CUDA | 570.211.01 / 12.8 |

Effective batch under DDP = `trainer.batch_size × num_processes`.

Recommended per-rank batch (n=11, leave headroom for sample + greedy baseline):

| Mode | `trainer.batch_size` | Effective batch |
| --- | ---: | ---: |
| 1 GPU | 1024 | 1024 |
| 2 GPU DDP | 512 | 1024 |

If OOM, halve. If VRAM is idle and step/sec still rises, try doubling.

---

## One-time setup

```bash
cd ~/path/to/TSPDrone-RL   # repo root on the remote host
uv sync
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
  trainer.batch_size=1024 \
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
  trainer.batch_size=1024 \
  trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=am_on_n11_bs1024 \
  wandb.group=n11_full_matrix
```

Swap `decoder` / `dynamics` as needed:

| `decoder=` | `dynamics=` | Suggested `wandb.name` |
| --- | --- | --- |
| `tspd_lstm` | `on` | `tspd_lstm_on_n11` |
| `tspd_lstm` | `off` | `tspd_lstm_off_n11` |
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
  trainer.batch_size=512 \
  trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=am_on_n11_ddp2_bs512 \
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
  trainer.batch_size=1024 trainer.mixed_precision=bf16 \
  data.load_checkpoint=false \
  wandb.name=am_on_n11 wandb.group=n11_full_matrix
```

**Shell B — GPU 1**

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m src.experiments.run \
  scale=full action=train \
  decoder=tspd_lstm dynamics=on \
  physics.n_nodes=11 \
  trainer.batch_size=1024 trainer.mixed_precision=bf16 \
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
| 11 | 1024 | 512 |
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

## Useful overrides

| Override | Meaning |
| --- | --- |
| `scale=small` / `scale=full` | 1k vs 1e6 updates (epochs) |
| `trainer.epochs=100000` | Cap updates without changing scale file |
| `trainer.batch_size=...` | Per-process batch |
| `trainer.mixed_precision=bf16` | Default; use `no` to debug AMP issues |
| `trainer.test_interval=200` | Val frequency (episodes) |
| `data.load_checkpoint=false` | Fresh weights (recommended for new encoder/decoder matrix) |
| `wandb.enabled=false` | Local-only run |
| `physics.n_nodes=11` | Problem size |

Outputs default under `~/local_db/tspdrone-rl/outputs/training/`.

---

## Monitor

```bash
watch -n 1 nvidia-smi
# or
nvtop
```

Confirm each parallel job is on a different GPU and memory is rising under load.
