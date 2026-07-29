from itertools import combinations
from math import comb
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from src.analyze.metadata import DECODERS, PROBLEMS, hypothesis_decoders
from src.analyze.processing import COMPARISON_CONTEXT_COLUMNS


def decoder_by_problem(final_metrics: pd.DataFrame) -> pd.DataFrame:
    valid = final_metrics[final_metrics["quality_value"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    group = [
        "scale",
        "mode",
        *COMPARISON_CONTEXT_COLUMNS,
        "problem",
        "decoder",
    ]
    return (
        valid.groupby(group, dropna=False)
        .agg(
            runs=("run_id", "size"),
            seeds=("seed", "nunique"),
            quality_mean=("quality_value", "mean"),
            quality_std=("quality_value", "std"),
            objective_mean=("objective", "mean"),
            objective_std=("objective", "std"),
            aggregate_gap_pct_mean=("aggregate_gap_pct", "mean"),
            aggregate_gap_pct_std=("aggregate_gap_pct", "std"),
            feasibility_rate_mean=("feasibility_rate", "mean"),
            within_problem_z_mean=("within_problem_z", "mean"),
            within_problem_rank_mean=("within_problem_rank", "mean"),
            within_problem_rank_score_mean=("within_problem_rank_score", "mean"),
        )
        .reset_index()
        .sort_values(group)
    )


def decoder_across_problems(by_problem: pd.DataFrame) -> pd.DataFrame:
    if by_problem.empty:
        return pd.DataFrame()
    # Each problem receives equal weight after seed-level aggregation. This prevents
    # problems with extra runs from dominating the architecture-level conclusion.
    group = ["scale", "mode", *COMPARISON_CONTEXT_COLUMNS, "decoder"]
    return (
        by_problem.groupby(group, dropna=False)
        .agg(
            problems=("problem", "nunique"),
            mean_problem_z=("within_problem_z_mean", "mean"),
            std_problem_z=("within_problem_z_mean", "std"),
            mean_problem_rank=("within_problem_rank_mean", "mean"),
            mean_problem_rank_score=("within_problem_rank_score_mean", "mean"),
            mean_feasibility_rate=("feasibility_rate_mean", "mean"),
        )
        .reset_index()
        .sort_values(group)
    )


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    running_maximum = 0.0
    total = len(valid)
    for rank, (index, p_value) in enumerate(valid.items()):
        candidate = min(1.0, float(p_value) * (total - rank))
        running_maximum = max(running_maximum, candidate)
        adjusted.loc[index] = running_maximum
    return adjusted


def pairwise_decoder_comparisons(final_metrics: pd.DataFrame) -> pd.DataFrame:
    valid = final_metrics[final_metrics["quality_value"].notna()].copy()
    records: list[dict[str, Any]] = []
    group_columns = ["scale", "mode", *COMPARISON_CONTEXT_COLUMNS, "problem"]
    match_columns = ["seed", "encoder"]
    for group_values, frame in valid.groupby(group_columns, dropna=False):
        scale, mode, comparison_regime, comparison_condition, problem = group_values
        for decoder_a, decoder_b in combinations(sorted(frame["decoder"].unique()), 2):
            left = frame[frame["decoder"] == decoder_a][
                [*match_columns, "quality_value"]
            ].rename(columns={"quality_value": "quality_a"})
            right = frame[frame["decoder"] == decoder_b][
                [*match_columns, "quality_value"]
            ].rename(columns={"quality_value": "quality_b"})
            paired = left.merge(right, on=match_columns, how="inner")
            if paired.empty:
                continue
            difference = paired["quality_a"] - paired["quality_b"]
            p_value = (
                float(stats.ttest_1samp(difference, 0.0).pvalue)
                if len(difference) >= 2 and difference.std(ddof=1) > 0
                else np.nan
            )
            records.append(
                {
                    "scale": scale,
                    "mode": mode,
                    "comparison_regime": comparison_regime,
                    "comparison_condition": comparison_condition,
                    "problem": problem,
                    "decoder_a": decoder_a,
                    "decoder_b": decoder_b,
                    "paired_seeds": len(difference),
                    "quality_difference_mean": difference.mean(),
                    "quality_difference_std": difference.std(ddof=1),
                    "decoder_a_wins": int((difference > 0).sum()),
                    "ties": int(np.isclose(difference, 0).sum()),
                    "decoder_b_wins": int((difference < 0).sum()),
                    "paired_t_p_value": p_value,
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result["holm_p_value_within_problem"] = result.groupby(group_columns, dropna=False)[
        "paired_t_p_value"
    ].transform(_holm_adjust)
    return result.sort_values([*group_columns, "decoder_a", "decoder_b"]).reset_index(
        drop=True
    )


def _standardize_hypothesis_cells(final_metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = final_metrics[
        final_metrics["decoder"].isin(hypothesis_decoders())
        & final_metrics["quality_value"].notna()
    ].copy()
    if candidates.empty:
        return candidates
    cell = [
        "problem",
        "mode",
        "scale",
        "seed",
        "encoder",
        *COMPARISON_CONTEXT_COLUMNS,
    ]
    required = len(hypothesis_decoders())
    complete = candidates.groupby(cell, dropna=False)["decoder"].transform("nunique")
    candidates = candidates[complete.eq(required)].copy()
    if candidates.empty:
        return candidates

    candidates["hypothesis_z"] = candidates.groupby(cell, dropna=False)[
        "quality_value"
    ].transform(
        lambda values: (
            (values - values.mean()) / values.std(ddof=0)
            if values.std(ddof=0) > 0
            else 0.0
        )
    )
    candidates["hypothesis_group"] = candidates["decoder"].map(
        lambda decoder: DECODERS[str(decoder)].hypothesis_group
    )
    return candidates


def hypothesis_problem_contrasts(final_metrics: pd.DataFrame) -> pd.DataFrame:
    standardized = _standardize_hypothesis_cells(final_metrics)
    if standardized.empty:
        return pd.DataFrame()
    cell = [
        "problem",
        "mode",
        "scale",
        "seed",
        "encoder",
        *COMPARISON_CONTEXT_COLUMNS,
    ]
    family_scores = (
        standardized.groupby([*cell, "hypothesis_group"], dropna=False)["hypothesis_z"]
        .mean()
        .unstack("hypothesis_group")
        .reset_index()
    )
    family_scores["recurrent_advantage"] = (
        family_scores["recurrent"] - family_scores["nonrecurrent"]
    )
    family_scores["solution_scope"] = family_scores["problem"].map(
        lambda problem: PROBLEMS[str(problem)].solution_scope
    )
    family_scores["problem_family"] = family_scores["problem"].map(
        lambda problem: PROBLEMS[str(problem)].family
    )
    return (
        family_scores.groupby(
            [
                "scale",
                "mode",
                *COMPARISON_CONTEXT_COLUMNS,
                "problem",
                "solution_scope",
                "problem_family",
            ],
            dropna=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            recurrent_advantage_mean=("recurrent_advantage", "mean"),
            recurrent_advantage_std=("recurrent_advantage", "std"),
        )
        .reset_index()
        .sort_values(
            [
                "scale",
                "mode",
                *COMPARISON_CONTEXT_COLUMNS,
                "solution_scope",
                "problem",
            ]
        )
    )


def _exact_label_permutation_p_value(
    values: np.ndarray, labels: np.ndarray, observed: float
) -> tuple[float, int]:
    partial_count = int(labels.sum())
    total = len(values)
    if partial_count == 0 or partial_count == total:
        return float("nan"), 0
    null_effects = []
    for partial_indices in combinations(range(total), partial_count):
        mask = np.zeros(total, dtype=bool)
        mask[list(partial_indices)] = True
        null_effects.append(values[mask].mean() - values[~mask].mean())
    null = np.asarray(null_effects)
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return p_value, comb(total, partial_count)


def _bootstrap_interaction_interval(
    partial: np.ndarray,
    full: np.ndarray,
    *,
    random_seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    if len(partial) < 2 or len(full) < 2:
        return float("nan"), float("nan")
    generator = np.random.default_rng(random_seed)
    partial_samples = generator.choice(
        partial, size=(samples, len(partial)), replace=True
    )
    full_samples = generator.choice(full, size=(samples, len(full)), replace=True)
    effects = partial_samples.mean(axis=1) - full_samples.mean(axis=1)
    lower, upper = np.quantile(effects, [0.025, 0.975])
    return float(lower), float(upper)


def hypothesis_tests(
    problem_contrasts: pd.DataFrame,
    *,
    random_seed: int = 20260716,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if problem_contrasts.empty:
        return pd.DataFrame()
    test_group = ["scale", "mode", *COMPARISON_CONTEXT_COLUMNS]
    for group_key, frame in problem_contrasts.groupby(test_group, dropna=False):
        scale, mode, comparison_regime, comparison_condition = cast(
            tuple[Any, Any, Any, Any], group_key
        )
        partial = frame.loc[
            frame["solution_scope"] == "partial_selection",
            "recurrent_advantage_mean",
        ].dropna()
        full = frame.loc[
            frame["solution_scope"] == "full_topology",
            "recurrent_advantage_mean",
        ].dropna()
        if partial.empty or full.empty:
            interaction = np.nan
            p_value = np.nan
            permutations = 0
            lower, upper = np.nan, np.nan
        else:
            interaction = float(partial.mean() - full.mean())
            values = frame["recurrent_advantage_mean"].to_numpy(dtype=float)
            labels = frame["solution_scope"].eq("partial_selection").to_numpy()
            p_value, permutations = _exact_label_permutation_p_value(
                values, labels, interaction
            )
            lower, upper = _bootstrap_interaction_interval(
                partial.to_numpy(),
                full.to_numpy(),
                random_seed=random_seed,
            )
        records.append(
            {
                "scale": scale,
                "mode": mode,
                "comparison_regime": comparison_regime,
                "comparison_condition": comparison_condition,
                "partial_problems": len(partial),
                "full_topology_problems": len(full),
                "partial_recurrent_advantage_mean": partial.mean(),
                "full_recurrent_advantage_mean": full.mean(),
                "interaction_effect": interaction,
                "bootstrap_ci_95_low": lower,
                "bootstrap_ci_95_high": upper,
                "exact_permutation_p_value": p_value,
                "exact_permutations": permutations,
                "direction_supports_hypothesis": bool(interaction > 0)
                if np.isfinite(interaction)
                else False,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(test_group)
