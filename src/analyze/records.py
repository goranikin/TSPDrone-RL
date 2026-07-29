from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExportBundle:
    root: Path
    manifest: dict[str, Any]
    runs: list[dict[str, Any]]
    histories: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class ProcessedData:
    runs: pd.DataFrame
    selected_runs: pd.DataFrame
    history: pd.DataFrame
    final_metrics: pd.DataFrame
    duplicate_runs: pd.DataFrame = field(default_factory=pd.DataFrame)
