from pathlib import Path

LOCAL_DB_ROOT = Path.home() / "local_db" / "tspdrone-rl"
LOCAL_OUTPUT_ROOT = LOCAL_DB_ROOT / "outputs"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data"
DEFAULT_TRAINED_MODELS_DIR = REPOSITORY_ROOT / "trained_models"
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results"


def resolve_user_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def experiment_log_path(filename: str, *, data_root: str | Path) -> Path:
    return resolve_user_path(data_root) / "log" / filename


def checkpoint_dir(output_dir: str | Path, n_nodes: int) -> Path:
    return resolve_user_path(output_dir) / f"n{n_nodes}"
