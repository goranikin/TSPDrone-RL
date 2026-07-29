from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_value(record) for record in frame.to_dict(orient="records")]


def build_analysis_summary(
    *,
    source: dict[str, Any],
    coverage: pd.DataFrame,
    sanity: pd.DataFrame,
    by_problem: pd.DataFrame,
    across_problems: pd.DataFrame,
    hypothesis: pd.DataFrame,
    figures: list[Path],
    included_decoders: tuple[str, ...],
    excluded_decoders: tuple[str, ...],
    included_problems: tuple[str, ...],
    excluded_problems: tuple[str, ...],
) -> dict[str, Any]:
    coverage_counts = (
        coverage["coverage_status"].value_counts().sort_index().to_dict()
        if not coverage.empty
        else {}
    )
    sanity_counts = (
        sanity["sanity_status"].value_counts().sort_index().to_dict()
        if not sanity.empty
        else {}
    )
    best_by_problem = pd.DataFrame()
    if not by_problem.empty:
        best_by_problem = (
            by_problem.sort_values(
                [
                    "scale",
                    "mode",
                    "comparison_regime",
                    "comparison_condition",
                    "problem",
                    "within_problem_rank_score_mean",
                ],
                ascending=[True, True, True, True, True, False],
            )
            .groupby(
                [
                    "scale",
                    "mode",
                    "comparison_regime",
                    "comparison_condition",
                    "problem",
                ],
                dropna=False,
            )
            .head(1)
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "entity": source.get("entity"),
            "project": source.get("project"),
            "fetched_at": source.get("fetched_at"),
            "exported_run_count": source.get("run_count", 0),
        },
        "analysis_scope": {
            "included_decoders": list(included_decoders),
            "excluded_decoders": list(excluded_decoders),
            "included_problems": list(included_problems),
            "excluded_problems": list(excluded_problems),
            "analyzed_run_count": int(sanity.shape[0]),
        },
        "coverage_counts": _json_value(coverage_counts),
        "sanity_counts": _json_value(sanity_counts),
        "best_decoder_by_problem": _records(
            best_by_problem[
                [
                    "scale",
                    "mode",
                    "comparison_regime",
                    "comparison_condition",
                    "problem",
                    "decoder",
                    "within_problem_rank_score_mean",
                ]
            ]
            if not best_by_problem.empty
            else best_by_problem
        ),
        "decoder_across_problems": _records(across_problems),
        "hypothesis_tests": _records(hypothesis),
        "figures": [str(path.name) for path in figures],
    }


def write_result_tables(
    output_dir: Path,
    *,
    runs: pd.DataFrame,
    selected_runs: pd.DataFrame,
    duplicate_runs: pd.DataFrame,
    history: pd.DataFrame,
    coverage: pd.DataFrame,
    sanity: pd.DataFrame,
    final_metrics: pd.DataFrame,
    by_problem: pd.DataFrame,
    across_problems: pd.DataFrame,
    pairwise: pd.DataFrame,
    problem_contrasts: pd.DataFrame,
    hypothesis: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "runs": runs,
        "selected_runs": selected_runs,
        "duplicate_runs": duplicate_runs,
        "coverage": coverage,
        "run_sanity": sanity,
        "final_metrics": final_metrics,
        "decoder_by_problem": by_problem,
        "decoder_across_problems": across_problems,
        "pairwise_decoder_comparisons": pairwise,
        "hypothesis_problem_contrasts": problem_contrasts,
        "hypothesis_tests": hypothesis,
    }
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    history_path = output_dir / "history.csv.gz"
    history.to_csv(history_path, index=False, compression="gzip")
    paths["history"] = history_path
    return paths


def write_summary(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
