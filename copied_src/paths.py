from pathlib import Path

from src.constants import DatasetSize, DataSplit, ProblemName

LOCAL_DB_ROOT = Path.home() / "local_db" / "compare-architectures"
LOCAL_OUTPUT_ROOT = LOCAL_DB_ROOT / "outputs"
WANDB_ANALYSIS_ROOT = LOCAL_OUTPUT_ROOT / "wandb_analysis"
PARAMETER_BUDGET_PATH = LOCAL_OUTPUT_ROOT / "parameter_budget.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"

_DATASET_PREFIX: dict[ProblemName, str] = {
    "tsp": "tsp50",
    "cvrp": "cvrp50",
    "orienteering": "orienteering50",
    "knapsack": "knapsack100",
    "mis": "mis100_p015",
    "max_clique": "max_clique100_p050",
    "vertex_cover": "vertex_cover100_p015",
}


def resolve_user_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def resolve_data_root(value: str | Path | None = None) -> Path:
    return LOCAL_DB_ROOT if value is None else resolve_user_path(value)


def experiment_log_path(
    filename: str,
    *,
    data_root: str | Path | None = None,
) -> Path:
    return resolve_data_root(data_root) / "log" / filename


def problem_dataset_path(
    problem: ProblemName,
    *,
    split: DataSplit,
    instances: int,
    dataset_size: DatasetSize,
    labeled: bool,
    data_root: str | Path | None = None,
) -> Path:
    label_directory = "labeled" if labeled else "non-labeled"
    filename = f"{_DATASET_PREFIX[problem]}_{split}_{instances}.jsonl"
    return (
        resolve_data_root(data_root)
        / label_directory
        / str(dataset_size)
        / problem
        / filename
    )


def problem_split_paths(
    problem: ProblemName,
    *,
    train_instances: int,
    baseline_instances: int | None,
    val_instances: int,
    test_instances: int,
    dataset_size: DatasetSize,
    labeled: bool,
    data_root: str | Path | None = None,
) -> dict[DataSplit, str]:
    split_counts: list[tuple[DataSplit, int]] = [
        ("train", train_instances),
        ("val", val_instances),
        ("test", test_instances),
    ]
    if baseline_instances is not None:
        split_counts.insert(1, ("baseline", baseline_instances))
    return {
        split: str(
            problem_dataset_path(
                problem,
                split=split,
                instances=instances,
                dataset_size=dataset_size,
                labeled=labeled,
                data_root=data_root,
            )
        )
        for split, instances in split_counts
    }
