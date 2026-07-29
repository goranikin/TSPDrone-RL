# W&B analysis methodology (TSPDrone-RL)

Migrated from `copied_src/analyze` and adapted to this project's decoder × dynamics matrix.

## Commands

```bash
uv run python -m src.analyze fetch
uv run python -m src.analyze analyze
uv run python -m src.analyze run
```

Defaults:

- Entity/project: `goranikin-my-project/tspdrone-rl`
- Raw snapshot: `~/local_db/tspdrone-rl/outputs/wandb_analysis/raw`
- Results: `~/local_db/tspdrone-rl/outputs/wandb_analysis/results`
- Report: `docs/W&B TSP-D Decoder Analysis.md`

## Matrix identity

Coverage cells are:

`problem × encoder × decoder × dynamics × mode × seed × scale`

Expected defaults:

| Axis | Values |
| --- | --- |
| problem | `tspd` |
| encoder | `attention` |
| decoder | `tspd_lstm`, `attention_model`, `lstm_pointer` |
| dynamics | `on`, `off` |
| mode | `rl` |
| seed | `5` |
| scale | CLI `--expected-scales` (`small` / `full`) |

## Metric mapping

TSP-D W&B keys are normalized onto the shared analysis schema:

| Logged key | Analysis alias |
| --- | --- |
| `train/episode` | `analysis_epoch` |
| `train/actor_loss` | `train/rl/policy_loss(_epoch)` |
| `train/makespan` | negated → `train/rl/reward` |
| `val/makespan` | `val/objective` |
| `train/best_val_makespan` | summary best validation objective |

Makespan is minimized; `quality_value = -objective`. Feasibility defaults to 1.0 because the env masks illegal actions.

## Hypothesis

- Recurrent: `tspd_lstm`, `lstm_pointer`
- Nonrecurrent: `attention_model`

Comparisons are matched within the same `dynamics` setting.

## Module ownership

| Module | Responsibility |
| --- | --- |
| `fetch.py` | Export W&B metadata/history |
| `loader.py` | Load a local export |
| `processing.py` | Normalize configs/history, dedupe runs, select metrics |
| `sanity.py` | Coverage + execution/learning health |
| `comparisons.py` | Decoder and hypothesis comparisons |
| `visualization.py` | Figures |
| `results.py` / `report.py` | CSV/JSON/Markdown artifacts |
| `pipeline.py` / `cli.py` | Orchestration and CLI |
