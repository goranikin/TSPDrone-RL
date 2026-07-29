from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
from typing import Any

from src.analyze.metadata import HISTORY_BASE_KEYS, HISTORY_PREFIXES

SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _summary_dict(run: Any) -> dict[str, Any]:
    summary = run.summary
    if hasattr(summary, "_json_dict"):
        return dict(summary._json_dict)
    return dict(summary)


def _safe_run_directory(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)


def _keep_history_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _json_safe(value)
        for key, value in row.items()
        if key in HISTORY_BASE_KEYS or key.startswith(HISTORY_PREFIXES)
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not row:
                continue
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
            handle.write("\n")
            count += 1
    temporary.replace(path)
    return count


def fetch_project(
    *,
    entity: str,
    project: str,
    output_dir: Path,
    filters: dict[str, Any] | None = None,
    refresh: bool = False,
    page_size: int = 1000,
) -> Path:
    """Export project metadata and complete scalar histories through W&B Public API."""
    import wandb

    output_dir = output_dir.resolve()
    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    public_runs = api.runs(
        f"{entity}/{project}",
        filters=filters or None,
        order="+created_at",
    )

    manifest_runs: list[dict[str, Any]] = []
    for index, run in enumerate(public_runs, start=1):
        run_id = str(run.id)
        run_dir_name = _safe_run_directory(run_id)
        run_dir = runs_root / run_dir_name
        metadata_path = run_dir / "metadata.json"
        history_path = run_dir / "history.jsonl"
        cached_state = None
        if metadata_path.exists():
            try:
                with metadata_path.open(encoding="utf-8") as handle:
                    cached_state = json.load(handle).get("state")
            except (json.JSONDecodeError, OSError, AttributeError):
                cached_state = None

        metadata = {
            "run_id": run_id,
            "name": run.name,
            "display_name": getattr(run, "display_name", run.name),
            "state": run.state,
            "entity": entity,
            "project": project,
            "url": run.url,
            "group": getattr(run, "group", None),
            "tags": list(run.tags or []),
            "created_at": getattr(run, "created_at", None),
            "config": _json_safe(dict(run.config)),
            "summary": _json_safe(_summary_dict(run)),
        }
        _write_json(metadata_path, metadata)

        should_refresh_history = (
            refresh
            or not history_path.exists()
            or str(run.state).lower() != "finished"
            or str(cached_state).lower() != "finished"
        )
        if should_refresh_history:
            rows = (
                _keep_history_row(row)
                for row in run.scan_history(page_size=page_size, use_cache=True)
            )
            history_rows = _write_jsonl(history_path, rows)
        else:
            with history_path.open("r", encoding="utf-8") as handle:
                history_rows = sum(1 for line in handle if line.strip())

        manifest_runs.append(
            {
                "run_id": run_id,
                "name": run.name,
                "state": run.state,
                "directory": f"runs/{run_dir_name}",
                "history_rows": history_rows,
            }
        )
        print(f"[{index}] exported {run_id} ({run.state}, {history_rows} rows)")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": datetime.now(UTC).isoformat(),
        "entity": entity,
        "project": project,
        "filters": filters or {},
        "run_count": len(manifest_runs),
        "runs": manifest_runs,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path
