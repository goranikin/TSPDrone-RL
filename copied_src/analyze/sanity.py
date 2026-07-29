from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from src.analyze.metadata import (
    EXPECTED_DECODERS,
    EXPECTED_MODES,
    EXPECTED_PROBLEMS,
    PROBLEMS,
    objective_sign,
)
from src.analyze.processing import IDENTITY_COLUMNS


def expected_encoder(problem: str) -> str:
    return (
        "graph_attention" if PROBLEMS[problem].family == "graph_subset" else "attention"
    )


def build_coverage_table(
    runs: pd.DataFrame,
    *,
    expected_scales: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    expected_modes: tuple[str, ...] = EXPECTED_MODES,
    expected_decoders: tuple[str, ...] = EXPECTED_DECODERS,
    expected_problems: tuple[str, ...] = EXPECTED_PROBLEMS,
) -> pd.DataFrame:
    expected_rows = [
        {
            "problem": problem,
            "encoder": expected_encoder(problem),
            "decoder": decoder,
            "mode": mode,
            "seed": seed,
            "scale": scale,
            "expected": True,
        }
        for scale, mode, problem, decoder, seed in product(
            expected_scales,
            expected_modes,
            expected_problems,
            expected_decoders,
            expected_seeds,
        )
    ]
    expected = pd.DataFrame.from_records(expected_rows)
    if "identity_complete" in runs:
        complete = runs[runs["identity_complete"]].copy()
    else:
        complete = runs.iloc[0:0].copy()
    if complete.empty:
        counts = pd.DataFrame(
            columns=(
                *IDENTITY_COLUMNS,
                "run_count",
                "finished_count",
                "configuration_variant_count",
                "rerun_count",
            )
        )
    else:
        counts = (
            complete.groupby(list(IDENTITY_COLUMNS), dropna=False)
            .agg(
                run_count=("run_id", "size"),
                finished_count=(
                    "state",
                    lambda values: int(values.eq("finished").sum()),
                ),
                configuration_variant_count=("config_fingerprint", "nunique"),
                rerun_count=("run_id", "size"),
            )
            .reset_index()
        )
    coverage = expected.merge(counts, on=list(IDENTITY_COLUMNS), how="outer")
    coverage["expected"] = coverage["expected"].fillna(False).astype(bool)
    coverage["run_count"] = coverage["run_count"].fillna(0).astype(int)
    coverage["finished_count"] = coverage["finished_count"].fillna(0).astype(int)
    coverage["configuration_variant_count"] = (
        coverage["configuration_variant_count"].fillna(0).astype(int)
    )
    coverage["coverage_status"] = np.select(
        [
            ~coverage["expected"],
            coverage["run_count"].eq(0),
            coverage["configuration_variant_count"].gt(1),
            (coverage["run_count"] > coverage["configuration_variant_count"]),
            coverage["finished_count"].eq(0),
        ],
        ["unexpected", "missing", "configuration_variant", "duplicate", "not_finished"],
        default="complete",
    )
    return coverage.sort_values(list(IDENTITY_COLUMNS)).reset_index(drop=True)


def _finite_values(history: pd.DataFrame, column: str) -> pd.Series:
    if column not in history:
        return pd.Series(dtype=float)
    values = pd.to_numeric(history[column], errors="coerce").dropna()
    return values[np.isfinite(values)]


def _early_late(values: pd.Series) -> tuple[float, float, float]:
    values = values.dropna().astype(float)
    if values.empty:
        return float("nan"), float("nan"), float("nan")
    window = max(1, min(3, len(values) // 3))
    early = float(values.iloc[:window].median())
    late = float(values.iloc[-window:].median())
    return early, late, late - early


def _validation_quality(history: pd.DataFrame, problem: str) -> pd.Series:
    if "val/objective" not in history:
        return pd.Series(dtype=float)
    rows = history[history["val/objective"].notna()].copy()
    if rows.empty:
        return pd.Series(dtype=float)
    if "val/aggregate_gap_pct" in rows and rows["val/aggregate_gap_pct"].notna().any():
        return -rows["val/aggregate_gap_pct"].dropna().astype(float)
    return objective_sign(problem) * rows["val/objective"].astype(float)


def _minimum_feasibility(history: pd.DataFrame) -> float:
    columns = [
        column
        for column in history.columns
        if column.endswith("/feasibility_rate")
        and (
            column.startswith("val/")
            or column.startswith("test/")
            or "/eval/" in column
        )
    ]
    values = pd.concat(
        [_finite_values(history, column) for column in columns], ignore_index=True
    )
    return float(values.min()) if not values.empty else float("nan")


def _final_heldout_feasibility(history: pd.DataFrame) -> float:
    for column in ("test/feasibility_rate", "val/feasibility_rate"):
        values = _finite_values(history, column)
        if not values.empty:
            return float(values.iloc[-1])
    return float("nan")


def _training_signal(history: pd.DataFrame, mode: str) -> tuple[str, pd.Series]:
    candidates = (
        ("train/sl/loss_epoch", "train/sl/loss")
        if mode == "supervised"
        else ("train/rl/policy_loss_epoch", "train/rl/policy_loss")
    )
    for column in candidates:
        values = _finite_values(history, column)
        if not values.empty:
            return column, values
    return candidates[0], pd.Series(dtype=float)


def _sanity_record(
    run: pd.Series,
    history: pd.DataFrame,
    *,
    feasibility_threshold: float,
) -> dict[str, Any]:
    mode = str(run["mode"])
    problem = str(run["problem"])
    training_column, training_values = _training_signal(history, mode)
    train_early, train_late, train_delta = _early_late(training_values)
    validation_values = _validation_quality(history, problem)
    val_early, val_late, val_delta = _early_late(validation_values)
    reward_values = _finite_values(history, "train/rl/reward")
    reward_early, reward_late, reward_delta = _early_late(reward_values)

    observed_epoch = (
        float(history["analysis_epoch"].max())
        if not history.empty and history["analysis_epoch"].notna().any()
        else float("nan")
    )
    expected_epoch = float(run["expected_epochs"])
    epoch_complete = (
        np.isfinite(expected_epoch)
        and np.isfinite(observed_epoch)
        and (observed_epoch >= expected_epoch)
    )
    minimum_feasibility = _minimum_feasibility(history)
    final_feasibility = _final_heldout_feasibility(history)
    feasibility_ok = np.isfinite(final_feasibility) and (
        final_feasibility >= feasibility_threshold
    )
    validation_present = not validation_values.empty
    training_finite = not training_values.empty

    problems: list[str] = []
    warnings: list[str] = []
    incomplete: list[str] = []
    if run["state"] != "finished":
        incomplete.append(f"state={run['state']}")
    if history.empty:
        incomplete.append("history_missing")
    if not epoch_complete:
        incomplete.append("expected_epochs_not_reached")
    if not incomplete:
        if not training_finite:
            problems.append(f"{training_column}_missing_or_nonfinite")
        if not validation_present:
            problems.append("validation_quality_missing_or_nonfinite")
        if not np.isfinite(final_feasibility):
            problems.append("feasibility_missing_or_nonfinite")
        elif not feasibility_ok:
            problems.append("final_heldout_infeasible_predictions")
        elif (
            np.isfinite(minimum_feasibility)
            and minimum_feasibility < feasibility_threshold
        ):
            warnings.append("intermediate_infeasible_predictions")

        if mode == "supervised" and np.isfinite(train_delta) and train_delta >= 0:
            warnings.append("supervised_loss_not_lower_at_end")
        if np.isfinite(val_delta) and val_delta <= 0:
            warnings.append("validation_quality_not_better_at_end")
        if mode == "rl" and reward_values.empty:
            warnings.append("rl_reward_missing")
        elif mode == "rl" and np.isfinite(reward_delta) and reward_delta <= 0:
            warnings.append("rl_reward_not_better_at_end")

    if incomplete:
        status = "incomplete"
    elif problems:
        status = "fail"
    elif warnings:
        status = "warning"
    else:
        status = "pass"

    record = {column: run.get(column) for column in run.index}
    record.update(
        {
            "sanity_status": status,
            "fail_reasons": ";".join(problems),
            "incomplete_reasons": ";".join(incomplete),
            "warnings": ";".join(warnings),
            "history_rows": len(history),
            "observed_epoch": observed_epoch,
            "epoch_complete": epoch_complete,
            "training_signal": training_column,
            "training_early": train_early,
            "training_late": train_late,
            "training_delta": train_delta,
            "validation_quality_early": val_early,
            "validation_quality_late": val_late,
            "validation_quality_delta": val_delta,
            "reward_early": reward_early,
            "reward_late": reward_late,
            "reward_delta": reward_delta,
            "minimum_feasibility_rate": minimum_feasibility,
            "final_heldout_feasibility_rate": final_feasibility,
            "feasibility_ok": feasibility_ok,
        }
    )
    return record


def build_sanity_table(
    selected_runs: pd.DataFrame,
    history: pd.DataFrame,
    *,
    feasibility_threshold: float = 1.0 - 1e-6,
) -> pd.DataFrame:
    records = []
    for _, run in selected_runs.iterrows():
        run_history = history[history["run_id"] == run["run_id"]]
        records.append(
            _sanity_record(
                run,
                run_history,
                feasibility_threshold=feasibility_threshold,
            )
        )
    return pd.DataFrame.from_records(records)
