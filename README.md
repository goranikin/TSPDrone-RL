# A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone

This repository contains code for deep reinforcement learning to solve the Traveling Salesman Problem with Drone (TSPD). For details, please see our paper [A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone](https://arxiv.org/abs/2112.12545). If this code is useful for your work, please cite our paper:

```
@article{bogyrbayeva2022deep,
      title={A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone}, 
      author={Aigerim Bogyrbayeva and Taehyun Yoon and Hanbum Ko and Sungbin Lim and Hyokun Yun and Changhyun Kwon},
      year={2022},
      journal={Transportation Research Part C: Emerging Technologies},
      volume={To Appear}
}
``` 

For the optimization heuristic algorithms used in the paper, please see [TSPDrone.jl](https://github.com/chkwon/TSPDrone.jl).


## Dependencies

* Python>=3.13
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)
* Hydra / OmegaConf / pydantic / tqdm / wandb

```bash
uv sync
```


## Usage

### Generating data

Training data is generated on the fly with the batch size and node numbers specified in `configs/train.yaml` (and `configs/scale/`). If test data is not given in the data folder, the test data will be generated randomly as well.

### Training

```bash
uv run python -m src.experiments.run \
  wandb.enabled=false \
  action=train \
  physics.n_nodes=11 \
  scale=small
```

With W&B:

```bash
uv run python -m src.experiments.run \
  action=train \
  wandb.name=tspd-n11-small-v1 \
  physics.n_nodes=11
```

Checkpoints are written under `~/local_db/tspdrone-rl/outputs/training/`. Legacy weights in `/trained_models` are still loaded automatically when present.

Pre-trained weight files for random data as described in the paper are located in the `/trained_models` directory for some sizes, `n = 11, 15, 20, 50, 100`.


### Evaluation

```bash
uv run python -m src.experiments.run \
  wandb.enabled=false \
  action=test \
  physics.n_nodes=11
```

By default, the greedy decoding will run.

### Sampling

```bash
uv run python -m src.experiments.run \
  wandb.enabled=false \
  action=sampling \
  n_samples=5 \
  physics.n_nodes=11
```

The results of both greedy and batch sampling decoding will be stored in the `results` folder. 




## Test Instances

The `/data` directory includes random test instances used in the paper for `n=11, 15, 20, 50, 100`.
Each file includes 100 instances. 

Each row represents an instance, in the form of 
```
x_1 y_1 d_1 x_2 y_2 d_2 ... x_n y_n d_n
```
where `x_i y_i d_i` represents the x-y coordinate of customer `i` and demand. All demands are set to be 1.0 for customers.
The last components `x_n y_n d_n` represents the depot and `d_n` is set to 0.0 for the depot.



## Example TSPD solution
A sample solution of TSPD for 11 nodes is depicted below:
![](/images/optimal-n11-6-2.svg)



## Acknowledgements
This repository heavily benefited from the following repositories:
- https://github.com/wouterkool/attention-learn-to-route
- https://github.com/OptMLGroup/VRP-RL

---
the src structure.

# Compare Architectures: training pipeline

This repository contains the model, problem, data-loading, training,
experiment-launch, and post-hoc W&B analysis code used to compare neural
combinatorial optimization architectures. Dataset generation remains in a
separate repository.

## Project structure

```text
src/
  config.py                 # validated runtime configuration
  constants.py              # names and literal value domains
  data.py                   # JSONL datasets and data loaders
  types.py                  # validated tensor containers
  models/
    model.py                # complete encoder-decoder NCO model
    decoding.py             # shared decoding tensor operations
    encoder/                # complete encoder models
    decoder/                # complete decoder models and rollout control
    layers/                 # reusable neural-network layers
    initialization.py       # architecture-specific initialization rules
  problems/                 # problem state, masks, objectives, and feasibility
  training/                 # training, baselines, metrics, and W&B support
  experiments/              # executable training and matrix entrypoints
  analyze/                  # W&B fetching, processing, comparisons, plots, reports
```

The project uses namespace packages, so these directories intentionally do not
contain `__init__.py` files.

## Data layout

The default database root is `~/local_db/compare-architectures`.

```text
compare-architectures/
  labeled/
    2000/<problem>/*.jsonl
    80000/<problem>/*.jsonl
  non-labeled/
    512000/<problem>/*.jsonl
    1280000/<problem>/*.jsonl
  outputs/                    # generated training and analysis artifacts
```

The path resolver selects the branch automatically:

- `mode=supervised` uses `labeled/`.
- `mode=rl` uses `non-labeled/`.
- Small supervised training uses `labeled/2000` (`1600/200/200`).
- Full supervised training uses `labeled/80000` (`64000/8000/8000`).
- Small RL consumes 512,000 training instances once from `non-labeled/512000`.
- Full RL consumes 1,280,000 training instances once from
  `non-labeled/1280000`.

RL directories also contain an independent rollout-baseline split. For example:

```text
non-labeled/512000/tsp/
  tsp50_train_512000.jsonl
  tsp50_baseline_200.jsonl
  tsp50_val_200.jsonl
  tsp50_test_200.jsonl
```

The training profiles are:

| Scale | Mode | Train | Batch | Steps/epoch | Epochs | Updates | Presentations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | supervised | 1,600 | 64 | 25 | 16 | 400 | 25,600 |
| small | RL | 512,000 | 512 | 25 | 40 | 1,000 | 512,000 |
| full | supervised | 64,000 | 512 | 125 | 10 | 1,250 | 640,000 |
| full | RL | 1,280,000 | 512 | 25 | 100 | 2,500 | 1,280,000 |

Supervised profiles reshuffle and repeat their labeled dataset. RL profiles use
a streaming JSONL loader and consume every training row once across logical
epochs. Validation and test remain fixed; rollout-baseline comparison uses its
own split.

Labeled `2000` (and `80000`) datasets include all seven problems with solution
labels for supervised training. The separate generator repository must create
the RL train/baseline/val/test files under `non-labeled/` before executing the
full matrix. Matrix runs generate both supervised and RL commands for all seven
problem categories.

## Setup

```bash
uv sync
```

The default comparison matches each model's total trainable parameter count to
the canonical Attention Model for that problem. It keeps `d_model=128`, three
encoder layers, and eight heads, then selects an architecture-specific integer
`d_ff`. A 0.1% maximum relative difference is enforced; a run fails before
training if its resolved architecture exceeds that tolerance.

Generate the reproducible parameter budget explicitly with:

```bash
uv run python -m src.experiments.parameter_comparison
```

This writes
`~/local_db/compare-architectures/outputs/parameter_budget.json`. Missing rows
can also be computed at runtime with the same settings.

This is a whole-model budget comparison, not a decoder-only parameter match.
Because `d_ff` belongs to the shared encoder and to the Transformer decoder, it
changes encoder capacity for every model and both components for the
Transformer. Use a fixed-width run as a separate architecture-control ablation:

```bash
uv run python -m src.experiments.run \
  parameter_budget.enabled=false \
  model.d_model=128 \
  model.d_ff=512 \
  wandb.name=fixed-width-ablation-v1
```

See
[Architecture mathematics](docs/Architecture%20Mathematics.md) for the full
input-to-output equations and current parameter totals.

Encoder selection defaults to `encoder=auto`:

- TSP, CVRP, orienteering, and knapsack use the dense `AttentionEncoder`.
- MIS, maximum clique, and minimum vertex cover use `GraphAttentionEncoder`,
  whose attention is restricted to adjacency edges plus self-loops.

An explicit `encoder=attention` remains available for graph-ablation runs.
`model.num_layers=3` is the encoder depth from Kool et al. The separate
`model.transformer_pointer_layers=1` setting applies only to the experimental
`TransformerPointerDecoder`; Kool's decoder has no stacked Transformer layers.
The decoder identifier `attention_model_without_glimpse` is the controlled AM
ablation: it keeps the same graph/step context and final pointer, but sends the
context query directly to the pointer projection without multi-head attention
over encoder nodes.

## Run training

Small supervised TSP run using default database paths:

```bash
uv run python -m src.experiments.run \
  problem=tsp \
  decoder=attention_model \
  mode=supervised \
  wandb.name=small-tsp-am-sl-v1
```

The corresponding no-glimpse ablation is:

```bash
uv run python -m src.experiments.run \
  problem=tsp \
  decoder=attention_model_without_glimpse \
  mode=supervised \
  wandb.name=small-tsp-am-no-glimpse-sl-v1
```

Full RL CVRP run:

```bash
uv run python -m src.experiments.run \
  scale=full \
  problem=cvrp \
  decoder=lstm_pointer \
  mode=rl \
  wandb.name=full-cvrp-lstm-rl-v1
```

Small supervised MIS run using the automatically selected graph encoder:

```bash
uv run python -m src.experiments.run \
  problem=mis \
  decoder=attention_model \
  mode=supervised \
  wandb.name=small-mis-am-sl-v1
```

Supervised pointer decoders use exact action sequences for TSP, CVRP, and
orienteering. Knapsack, MIS, maximum clique, and minimum vertex cover use a
set-valued autoregressive loss: each step receives credit for probability mass
on any remaining labeled node, and the model chooses the highest-scoring one to
advance the target state. Therefore, sorted solver output does not impose a
node-index decoding order. The sigmoid decoder continues to use node-membership
binary cross-entropy. Routing labels are canonicalized before training: TSP
cycles share one rotation and direction, CVRP routes share one route order and
orientation, and orienteering paths share one direction. Invalid or masked
teacher actions raise an error instead of silently contributing zero loss.

Use explicit files by disabling default paths:

```bash
uv run python -m src.experiments.run \
  data.use_default_paths=false \
  data.train_path=/path/to/train.jsonl \
  data.baseline_path=/path/to/baseline.jsonl \
  data.val_path=/path/to/val.jsonl \
  data.test_path=/path/to/test.jsonl \
  wandb.name=explicit-data-run-v1
```

W&B logging is enabled by default. New runs are sent to the
`goranikin-my-project/compare-architectures` W&B project. Use Hydra overrides
such as `wandb.entity=...` or `wandb.project=...` only when a run intentionally
belongs elsewhere. Every logged training run requires a non-empty
`wandb.name=<pipeline-name>` override; use a short purpose/version label such as
`matched-parameters-sl-v1`. For a matrix, this value becomes a short experiment
tag. Each child display name is `{problem}_{decoder}_{mode}`, while
the problem, decoder, mode, `seed-<seed>`, and experiment label are tags. Set
`wandb.enabled=false` for a local run that should not be logged; local runs do
not require a name.
Training, validation, and test use terminal `tqdm` progress
bars. W&B receives validation/test metrics and live batch progress under
`progress/validation/*` and `progress/test/*`. Test evaluation restores the
validation-selected `best.pt` checkpoint first. Solver labels are logged as
`reference_objective` and `reference_gap`, not as "optimal": dataset provenance
records the selected solver and exact/non-exact counts for every split. Training
snapshots use an evaluation-mode forward pass and a separate non-shuffled
loader, so monitoring does not update BatchNorm state or consume the shuffled
training iterator.

The experiment entrypoints append local logs under the database root:

```text
~/local_db/compare-architectures/log/
  run.log
  matrix.log
  parameter_comparison.log
```

## Architecture matrix

By default, the matrix emits one compatible encoder per problem: dense
attention for non-graph problems and graph attention for sparse graph problems.
Set `encoders=[attention]` explicitly to run the topology-blind graph ablation.
All six registered decoders, including `attention_model_without_glimpse`, are
included unless `decoders=[...]` overrides the axis.

Dry-run the small matrix:

```bash
uv run python -m src.experiments.matrix
```

Execute it:

```bash
uv run python -m src.experiments.matrix \
  wandb.name=small-complete-v1 \
  execute=true
```

Completed leaf runs with a valid `result.json` are skipped by default. Set
`skip_completed=false` only when intentionally rerunning into those directories.

Executed matrices use W&B by default. The matrix propagates its W&B entity,
project, tags, mode, name label, and evaluation settings to each child run.
Unless overridden, runs are logged to
`goranikin-my-project/compare-architectures` and grouped by scale, selected
stage, and seed. For example, `wandb.name=small-sl-v1` produces child names such
as `tsp_lstm_pointer_supervised`, tagged with `tsp`, `lstm_pointer`,
`supervised`, `seed-1234`, and `small-sl-v1`. Pass `wandb.enabled=false` to
keep an executed matrix local without supplying a name.

Use the full profiles:

```bash
uv run python -m src.experiments.matrix scale=full
```

## Analyze W&B runs

Fetch the complete scalar histories and generate the small-matrix report:

```bash
uv run python -m src.analyze run
```

The defaults read `goranikin-my-project/compare-architectures`. W&B snapshots,
generated tables, and figures are kept outside the repository under the
dedicated analysis database root. Only human-readable Markdown reports belong
in the repository's root `docs/` directory:

```text
~/local_db/compare-architectures/outputs/wandb_analysis/
  raw/                      # reproducible W&B metadata/history snapshot
  results/
    analysis.json           # machine-readable headline results
    *.csv                   # coverage, sanity, comparisons, hypothesis tests
    history.csv.gz          # normalized exported history
    figures/*.png           # coverage, curves, comparisons, hypothesis
docs/
  W&B Architecture Comparison Analysis.md
```

Use a batch-specific child directory, such as
`wandb_analysis/small-sv-same-param-v1/{raw,results}`, when retaining multiple
tag-scoped analyses. Do not save raw W&B exports, generated CSV/JSON files, or
figures under the repository's `outputs/` directory.

The migrated snapshots remain separate as
[Previous Experiment Analysis](docs/Previous%20Experiment%20Analysis.md) and
[Second Experiment Analysis](docs/Second%20Experiment%20Analysis.md); the latter
contains only W&B runs whose names have the `second` prefix.

Fetching and analysis can be run independently so the same immutable snapshot
can be reanalyzed without another network request:

```bash
uv run python -m src.analyze fetch
uv run python -m src.analyze analyze
```

For a completed full-scale matrix, declare both expected scales so coverage is
checked against both grids:

```bash
uv run python -m src.analyze run --expected-scales small full
```

For a tag-scoped batch containing only one training mode, restrict the expected
coverage grid accordingly:

```bash
uv run python -m src.analyze run \
  --filters-json '{"tags":{"$in":["matched-parameters-sl-v1"]}}' \
  --expected-modes supervised
```

To remove a decoder from the analytical population—not merely hide it in a
plot—exclude it when analyzing the preserved raw snapshot:

```bash
uv run python -m src.analyze analyze \
  --input-dir ~/local_db/compare-architectures/outputs/wandb_analysis/small-rl-v1/raw \
  --output-dir ~/local_db/compare-architectures/outputs/wandb_analysis/small-rl-v1/results \
  --expected-scales small \
  --expected-modes rl \
  --exclude-decoders sigmoid_subset \
  --exclude-problems orienteering \
  --report-path 'docs/Small RL V1 Analysis.md'
```

The `--exclude-decoders` and `--exclude-problems` options scope coverage,
sanity checks, result tables, statistical comparisons, the report, and every
figure while leaving the raw W&B export unchanged.

The analysis compares decoders within matched problem/mode/scale/seed/encoder
cells. Cross-problem summaries use standardized quality and ranks rather than
averaging incompatible raw objectives. See
[W&B analysis methodology](copied_src/analyze/README.md) for sanity criteria,
hypothesis definitions, output ownership, and interpretation limits.

## Validation

```bash
uv run ruff check .
uv run ty check
uv run python -m unittest discover -s tests
```
