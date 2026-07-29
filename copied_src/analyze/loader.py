import json
from pathlib import Path
from typing import Any

from src.analyze.fetch import SCHEMA_VERSION
from src.analyze.records import ExportBundle


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def load_export(input_dir: Path) -> ExportBundle:
    root = input_dir.resolve()
    manifest = _read_json(root / "manifest.json")
    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported W&B export schema {schema_version}; expected {SCHEMA_VERSION}"
        )

    run_metadata: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest.get("runs", []):
        run_id = str(entry["run_id"])
        run_dir = root / str(entry["directory"])
        metadata = _read_json(run_dir / "metadata.json")
        if str(metadata.get("run_id")) != run_id:
            raise ValueError(f"Run ID mismatch in {run_dir}")
        run_metadata.append(metadata)
        histories[run_id] = _read_jsonl(run_dir / "history.jsonl")

    return ExportBundle(
        root=root,
        manifest=manifest,
        runs=run_metadata,
        histories=histories,
    )
