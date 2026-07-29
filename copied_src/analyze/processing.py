import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.analyze.metadata import DECODERS, PROBLEMS, objective_sign
from src.analyze.records import ExportBundle, ProcessedData

IDENTITY_COLUMNS: tuple[str, ...] = (
    "problem",
    "encoder",
    "decoder",
    "mode",
    "seed",
    "scale",
)
VARIANT_IDENTITY_COLUMNS: tuple[str, ...] = (*IDENTITY_COLUMNS, "config_fingerprint")
COMPARISON_CONTEXT_COLUMNS: tuple[str, ...] = (
    "comparison_regime",
    "comparison_condition",
)


def nested_value(payload: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read either a nested W&B config field or a literal dotted field."""
    if path in payload:
        return payload[path]
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def stable_gap_percent(problem: str, objective: Any, target: Any) -> float:
    """Compute a stable gap from aggregate objectives, not mean per-item ratios."""
    try:
        objective_value = float(objective)
        target_value = float(target)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(objective_value) or not np.isfinite(target_value):
        return float("nan")
    denominator = abs(target_value)
    if denominator <= np.finfo(float).eps:
        return float("nan")
    regret = -objective_sign(problem) * (objective_value - target_value)
    return 100.0 * regret / denominator


def _comparison_context(config: Mapping[str, Any]) -> tuple[str, str]:
    budget_enabled = nested_value(config, "parameter_budget.enabled")
    regime = (
        "matched_total_params"
        if budget_enabled is True
        else "fixed_encoder"
        if budget_enabled is False
        else "legacy_unknown"
    )
    model_fields = {
        key: nested_value(config, f"model.{key}")
        for key in (
            "num_layers",
            "num_heads",
            "transformer_pointer_layers",
            "dropout",
            "tanh_clip",
        )
    }
    if budget_enabled is not True:
        model_fields.update(
            {
                "d_model": nested_value(config, "model.d_model"),
                "d_ff": nested_value(config, "model.d_ff"),
            }
        )
    condition_payload = {
        "regime": regime,
        "data": {
            key: nested_value(config, f"data.{key}")
            for key in (
                "scale",
                "dataset_size",
                "train_instances",
                "validation_instances",
                "test_instances",
                "planned_updates",
                "planned_presentations",
                "train_data_policy",
                "batch_size",
                "eval_batch_size",
                "shuffle",
            )
        },
        "model": model_fields,
        "trainer": nested_value(config, "trainer", {}),
        "parameter_budget": (
            {
                "target_params_override": nested_value(
                    config, "parameter_budget.target_params_override"
                ),
                "search": nested_value(config, "parameter_budget.search"),
            }
            if budget_enabled is True
            else {"enabled": budget_enabled}
        ),
        "code_fingerprint": nested_value(config, "provenance.code_fingerprint"),
    }
    canonical = json.dumps(
        condition_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    condition = hashlib.sha256(canonical.encode()).hexdigest()
    return regime, condition


def _run_record(metadata: Mapping[str, Any]) -> dict[str, Any]:
    config = metadata.get("config", {})
    summary = metadata.get("summary", {})
    if not isinstance(config, Mapping):
        config = {}
    if not isinstance(summary, Mapping):
        summary = {}

    problem = nested_value(config, "run.problem")
    decoder = nested_value(config, "run.decoder")
    comparison_regime, comparison_condition = _comparison_context(config)
    config_fingerprint = nested_value(config, "provenance.config_fingerprint")
    if config_fingerprint is None:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
        config_fingerprint = f"legacy-{hashlib.sha256(canonical.encode()).hexdigest()}"
    record = {
        "run_id": str(metadata.get("run_id", "")),
        "name": metadata.get("name"),
        "state": str(metadata.get("state", "unknown")).lower(),
        "url": metadata.get("url"),
        "created_at": metadata.get("created_at"),
        "group": metadata.get("group"),
        "tags": ",".join(str(tag) for tag in metadata.get("tags", [])),
        "problem": problem,
        "encoder": nested_value(config, "run.encoder"),
        "decoder": decoder,
        "mode": nested_value(config, "run.mode"),
        "seed": nested_value(config, "run.seed"),
        "scale": nested_value(config, "data.scale"),
        "comparison_regime": comparison_regime,
        "comparison_condition": comparison_condition,
        "config_fingerprint": config_fingerprint,
        "code_fingerprint": nested_value(config, "provenance.code_fingerprint"),
        "git_commit": nested_value(config, "provenance.git_commit"),
        "git_dirty": nested_value(config, "provenance.git_dirty"),
        "reference_kind": nested_value(
            config, "data.provenance.validation.reference_kind"
        ),
        "expected_epochs": nested_value(config, "trainer.epochs"),
        "steps_per_epoch": nested_value(config, "trainer.steps_per_epoch"),
        "planned_updates": nested_value(config, "data.planned_updates"),
        "train_instances": nested_value(config, "data.train_instances"),
        "validation_instances": nested_value(config, "data.validation_instances"),
        "test_instances": nested_value(config, "data.test_instances"),
        "trainable_params": nested_value(config, "model.trainable_params"),
        "training_time_sec": summary.get("train/training_time_sec"),
        "summary_best_validation_objective": summary.get(
            "train/best_validation_objective"
        ),
        "summary_best_validation_feasibility_rate": summary.get(
            "train/best_validation_feasibility_rate"
        ),
        "summary_best_validation_score": summary.get("train/best_validation_score"),
    }
    record["objective_sense"] = (
        PROBLEMS[str(problem)].objective_sense if problem in PROBLEMS else None
    )
    record["solution_scope"] = (
        PROBLEMS[str(problem)].solution_scope if problem in PROBLEMS else None
    )
    record["problem_family"] = (
        PROBLEMS[str(problem)].family if problem in PROBLEMS else None
    )
    record["decoder_family"] = (
        DECODERS[str(decoder)].family if decoder in DECODERS else None
    )
    record["identity_complete"] = all(
        record.get(column) is not None for column in IDENTITY_COLUMNS
    )
    return record


def build_run_table(bundle: ExportBundle) -> pd.DataFrame:
    records = [_run_record(metadata) for metadata in bundle.runs]
    if not records:
        return pd.DataFrame(columns=("run_id", *IDENTITY_COLUMNS))
    runs = pd.DataFrame.from_records(records)
    runs["created_at"] = pd.to_datetime(runs["created_at"], errors="coerce", utc=True)
    for column in (
        "seed",
        "expected_epochs",
        "steps_per_epoch",
        "planned_updates",
        "train_instances",
        "validation_instances",
        "test_instances",
        "trainable_params",
        "training_time_sec",
    ):
        runs[column] = pd.to_numeric(runs[column], errors="coerce")
    return runs


def select_canonical_runs(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select reruns only when a cell has one unambiguous configuration."""
    if runs.empty:
        empty = runs.copy()
        empty["selected"] = pd.Series(dtype=bool)
        return empty, empty, empty

    result = runs.copy()
    if "config_fingerprint" not in result:
        result["config_fingerprint"] = "legacy-unspecified"
    result["config_fingerprint"] = result["config_fingerprint"].fillna(
        "legacy-unspecified"
    )
    result["selected"] = False
    complete = result[result["identity_complete"]].copy()
    incomplete = result[~result["identity_complete"]].copy()

    complete["_finished_rank"] = complete["state"].eq("finished").astype(int)
    complete = complete.sort_values(
        [*VARIANT_IDENTITY_COLUMNS, "_finished_rank", "created_at", "run_id"],
        ascending=[True] * len(VARIANT_IDENTITY_COLUMNS) + [False, False, False],
        na_position="last",
    )
    complete["duplicate_count"] = complete.groupby(
        list(VARIANT_IDENTITY_COLUMNS), dropna=False
    )["run_id"].transform("size")
    complete["configuration_variant_count"] = complete.groupby(
        list(IDENTITY_COLUMNS), dropna=False
    )["config_fingerprint"].transform("nunique")
    candidate_index = (
        complete.groupby(list(VARIANT_IDENTITY_COLUMNS), dropna=False, sort=False)
        .head(1)
        .index
    )
    variant_counts = complete.set_index("run_id")["configuration_variant_count"]
    eligible_candidates = complete.loc[candidate_index]
    selected_index = eligible_candidates.loc[
        eligible_candidates["configuration_variant_count"].eq(1)
    ].index
    result.loc[selected_index, "selected"] = True
    duplicate_ids = complete.loc[complete["duplicate_count"] > 1, "run_id"]
    duplicates = result[result["run_id"].isin(duplicate_ids)].copy()
    duplicate_counts = complete.set_index("run_id")["duplicate_count"]
    result["duplicate_count"] = result["run_id"].map(duplicate_counts).fillna(1)
    result["configuration_variant_count"] = (
        result["run_id"].map(variant_counts).fillna(0).astype(int)
    )
    result["selection_status"] = "not_selected"
    result.loc[selected_index, "selection_status"] = "selected"
    result.loc[
        result["configuration_variant_count"].gt(1), "selection_status"
    ] = "configuration_variant"
    result.loc[incomplete.index, "duplicate_count"] = 1
    selected = result[result["selected"]].copy()
    return result, selected, duplicates


def _history_records(
    bundle: ExportBundle, selected_runs: pd.DataFrame
) -> Iterable[dict[str, Any]]:
    selected_ids = set(selected_runs["run_id"].astype(str))
    identities = selected_runs.set_index("run_id").to_dict(orient="index")
    for run_id, rows in bundle.histories.items():
        if run_id not in selected_ids:
            continue
        identity = identities[run_id]
        for row in rows:
            yield {
                "run_id": run_id,
                **{
                    column: identity.get(column)
                    for column in (*IDENTITY_COLUMNS, *COMPARISON_CONTEXT_COLUMNS)
                },
                **row,
            }


def build_history_table(
    bundle: ExportBundle, selected_runs: pd.DataFrame
) -> pd.DataFrame:
    history = pd.DataFrame.from_records(_history_records(bundle, selected_runs))
    if history.empty:
        return pd.DataFrame(
            columns=(
                "run_id",
                *IDENTITY_COLUMNS,
                *COMPARISON_CONTEXT_COLUMNS,
                "analysis_epoch",
            )
        )
    if "_step" not in history:
        history["_step"] = np.arange(len(history), dtype=int)
    for column in history.columns:
        if column in {
            "run_id",
            *IDENTITY_COLUMNS,
            *COMPARISON_CONTEXT_COLUMNS,
        }:
            continue
        history[column] = pd.to_numeric(history[column], errors="coerce")
    if "epoch" in history:
        analysis_epoch = history["epoch"]
    else:
        analysis_epoch = pd.Series(np.nan, index=history.index)
    if "train/epoch" in history:
        analysis_epoch = analysis_epoch.fillna(history["train/epoch"])
    history["analysis_epoch"] = analysis_epoch

    for prefix in (
        "train/sl",
        "train/sl/eval",
        "train/rl",
        "train/rl/eval",
        "val",
        "test",
    ):
        objective_column = f"{prefix}/objective"
        reference_column = f"{prefix}/reference_objective"
        legacy_target_column = f"{prefix}/target_objective"
        if reference_column not in history and legacy_target_column in history:
            history[reference_column] = history[legacy_target_column]
        legacy_gap_column = f"{prefix}/optimal_gap"
        reference_gap_column = f"{prefix}/reference_gap"
        if reference_gap_column not in history and legacy_gap_column in history:
            history[reference_gap_column] = history[legacy_gap_column]
        legacy_gap_pct_column = f"{prefix}/optimal_gap_pct"
        reference_gap_pct_column = f"{prefix}/reference_gap_pct"
        if (
            reference_gap_pct_column not in history
            and legacy_gap_pct_column in history
        ):
            history[reference_gap_pct_column] = history[legacy_gap_pct_column]
        if objective_column not in history or reference_column not in history:
            continue
        history[f"{prefix}/aggregate_gap_pct"] = history.apply(
            lambda row: stable_gap_percent(
                str(row["problem"]), row[objective_column], row[reference_column]
            ),
            axis=1,
        )
    return history.sort_values(["run_id", "_step"], na_position="last")


def _last_metric_row(run_history: pd.DataFrame, prefix: str) -> pd.Series | None:
    objective_column = f"{prefix}/objective"
    if objective_column not in run_history:
        return None
    available = run_history[run_history[objective_column].notna()]
    if available.empty:
        return None
    return available.sort_values("_step", na_position="first").iloc[-1]


def _best_validation_row(run_history: pd.DataFrame, problem: str) -> pd.Series | None:
    if "val/objective" not in run_history:
        return None
    available = run_history[run_history["val/objective"].notna()].copy()
    if available.empty:
        return None
    if "val/feasibility_rate" in available:
        best_feasibility = available["val/feasibility_rate"].max()
        available = available[available["val/feasibility_rate"].eq(best_feasibility)]
    gap_column = "val/aggregate_gap_pct"
    if gap_column in available and available[gap_column].notna().any():
        return available.loc[available[gap_column].idxmin()]
    quality = objective_sign(problem) * available["val/objective"]
    return available.loc[quality.idxmax()]


def _final_metric_record(run: pd.Series, run_history: pd.DataFrame) -> dict[str, Any]:
    problem = str(run["problem"])
    chosen = _last_metric_row(run_history, "test")
    split = "test"
    if chosen is None:
        chosen = _best_validation_row(run_history, problem)
        split = "val_best"

    record = {column: run.get(column) for column in run.index}
    record["evaluation_split"] = split if chosen is not None else None
    if chosen is None:
        record.update(
            {
                "objective": np.nan,
                "reference_objective": np.nan,
                "feasibility_rate": np.nan,
                "aggregate_gap_pct": np.nan,
                "quality_kind": None,
                "quality_value": np.nan,
                "evaluation_step": np.nan,
                "evaluation_epoch": np.nan,
            }
        )
        return record

    prefix = "test" if split == "test" else "val"
    objective = chosen.get(f"{prefix}/objective", np.nan)
    reference = chosen.get(f"{prefix}/reference_objective", np.nan)
    gap = chosen.get(f"{prefix}/aggregate_gap_pct", np.nan)
    feasibility = chosen.get(f"{prefix}/feasibility_rate", np.nan)
    has_gap = pd.notna(gap)
    record.update(
        {
            "objective": objective,
            "reference_objective": reference,
            "feasibility_rate": feasibility,
            "aggregate_gap_pct": gap,
            "quality_kind": "negative_aggregate_gap_pct"
            if has_gap
            else "directional_objective",
            "quality_value": -float(gap)
            if has_gap
            else objective_sign(problem) * float(objective),
            "evaluation_step": chosen.get("_step", np.nan),
            "evaluation_epoch": chosen.get("analysis_epoch", np.nan),
        }
    )
    return record


def add_within_problem_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    result["within_problem_z"] = np.nan
    result["within_problem_rank"] = np.nan
    result["within_problem_rank_score"] = np.nan
    cell = [
        "problem",
        "mode",
        "scale",
        "seed",
        "encoder",
        *COMPARISON_CONTEXT_COLUMNS,
    ]
    valid = result["quality_value"].notna()
    for _, index in result[valid].groupby(cell, dropna=False).groups.items():
        values = result.loc[index, "quality_value"].astype(float)
        standard_deviation = values.std(ddof=0)
        z_scores = (
            (values - values.mean()) / standard_deviation
            if standard_deviation > 0
            else pd.Series(0.0, index=values.index)
        )
        ranks = values.rank(method="average", ascending=False)
        denominator = max(len(values) - 1, 1)
        result.loc[index, "within_problem_z"] = z_scores
        result.loc[index, "within_problem_rank"] = ranks
        result.loc[index, "within_problem_rank_score"] = (
            1.0 - (ranks - 1.0) / denominator
        )
    return result


def build_final_metrics(
    selected_runs: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, run in selected_runs.iterrows():
        run_history = history[history["run_id"] == run["run_id"]]
        records.append(_final_metric_record(run, run_history))
    if not records:
        return pd.DataFrame()
    return add_within_problem_scores(pd.DataFrame.from_records(records))


def process_export(bundle: ExportBundle) -> ProcessedData:
    runs = build_run_table(bundle)
    runs, selected_runs, duplicate_runs = select_canonical_runs(runs)
    history = build_history_table(bundle, selected_runs)
    final_metrics = build_final_metrics(selected_runs, history)
    return ProcessedData(
        runs=runs,
        selected_runs=selected_runs,
        history=history,
        final_metrics=final_metrics,
        duplicate_runs=duplicate_runs,
    )
