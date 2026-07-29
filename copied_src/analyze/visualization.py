from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from src.analyze.metadata import EXPECTED_DECODERS, EXPECTED_PROBLEMS, objective_sign

DECODER_LABELS = {
    "attention_model": "Attention",
    "attention_model_without_glimpse": "Attention No Glimpse",
    "lstm_pointer": "LSTM",
    "gru_pointer": "GRU",
    "transformer_pointer": "Transformer",
    "sigmoid_subset": "Sigmoid",
}

PROBLEM_LABELS = {
    "tsp": "TSP",
    "cvrp": "CVRP",
    "orienteering": "Orienteering",
    "knapsack": "Knapsack",
    "mis": "MIS",
    "max_clique": "Maximum clique",
    "vertex_cover": "Vertex cover",
}


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _axes_array(axes: Any) -> np.ndarray:
    return np.asarray(axes, dtype=object).reshape(-1)


def _two_key_panels(
    frame: pd.DataFrame,
) -> list[tuple[tuple[Any, Any], pd.DataFrame]]:
    panel_frame = frame.copy()
    scale_column = "scale"
    context_columns = ("comparison_regime", "comparison_condition")
    has_multiple_contexts = (
        all(column in panel_frame for column in context_columns)
        and len(panel_frame[list(context_columns)].drop_duplicates()) > 1
    )
    if has_multiple_contexts:
        scale_column = "_comparison_panel"
        panel_frame[scale_column] = (
            panel_frame["scale"].astype(str)
            + "-"
            + panel_frame["comparison_regime"].astype(str)
            + "-"
            + panel_frame["comparison_condition"].astype(str).str[:8]
        )
    return [
        (cast(tuple[Any, Any], key), group)
        for key, group in panel_frame.groupby([scale_column, "mode"], dropna=False)
    ]


def _annotate_heatmap(axis: plt.Axes, values: np.ndarray, labels: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(
                    column,
                    row,
                    labels[row, column],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )


def plot_coverage(
    coverage: pd.DataFrame,
    path: Path,
    *,
    decoders: tuple[str, ...] = EXPECTED_DECODERS,
    problems: tuple[str, ...] = EXPECTED_PROBLEMS,
) -> Path | None:
    expected = coverage[coverage["expected"]].copy()
    if expected.empty:
        return None
    panels = _two_key_panels(expected)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(6.0, 5.2 * len(panels)), 5.2),
        squeeze=False,
    )
    for axis, ((scale, mode), frame) in zip(_axes_array(axes), panels, strict=True):
        summary = (
            frame.assign(valid=frame["coverage_status"].eq("complete"))
            .groupby(["problem", "decoder"], observed=False)
            .agg(valid=("valid", "sum"), expected=("expected", "size"))
            .reset_index()
        )
        summary["fraction"] = summary["valid"] / summary["expected"]
        pivot = summary.pivot(
            index="problem", columns="decoder", values="fraction"
        ).reindex(index=problems, columns=decoders)
        counts = (
            summary.assign(
                label=summary["valid"].astype(str)
                + "/"
                + summary["expected"].astype(str)
            )
            .pivot(index="problem", columns="decoder", values="label")
            .reindex(index=problems, columns=decoders)
        )
        image = axis.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="RdYlGn")
        _annotate_heatmap(
            axis, pivot.to_numpy(dtype=float), counts.fillna("-").to_numpy()
        )
        axis.set_title(f"{scale} / {mode}")
        axis.set_xticks(range(len(decoders)))
        axis.set_xticklabels(
            [DECODER_LABELS[item] for item in decoders],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(len(problems)))
        axis.set_yticklabels(problems)
    figure.colorbar(
        image, ax=_axes_array(axes).tolist(), label="complete seed fraction"
    )
    figure.suptitle("Expected experiment coverage")
    return _save(figure, path)


def plot_sanity(
    sanity: pd.DataFrame,
    path: Path,
    *,
    decoders: tuple[str, ...] = EXPECTED_DECODERS,
    problems: tuple[str, ...] = EXPECTED_PROBLEMS,
) -> Path | None:
    if sanity.empty:
        return None
    status_score = {"fail": 0, "incomplete": 1, "warning": 2, "pass": 3}
    panels = _two_key_panels(sanity)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(6.0, 5.2 * len(panels)), 5.2),
        squeeze=False,
    )
    cmap = ListedColormap(["#d73027", "#9e9e9e", "#fee08b", "#1a9850"])
    for axis, ((scale, mode), frame) in zip(_axes_array(axes), panels, strict=True):
        scored = frame.assign(score=frame["sanity_status"].map(status_score))
        worst = (
            scored.groupby(["problem", "decoder"], observed=False)["score"]
            .min()
            .unstack("decoder")
            .reindex(index=problems, columns=decoders)
        )
        labels = worst.map(
            lambda value: (
                "-" if pd.isna(value) else ("FAIL", "INC", "WARN", "PASS")[int(value)]
            )
        )
        axis.imshow(worst.to_numpy(dtype=float), vmin=-0.5, vmax=3.5, cmap=cmap)
        _annotate_heatmap(axis, worst.to_numpy(dtype=float), labels.to_numpy())
        axis.set_title(f"{scale} / {mode}")
        axis.set_xticks(range(len(decoders)))
        axis.set_xticklabels(
            [DECODER_LABELS[item] for item in decoders],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(len(problems)))
        axis.set_yticklabels(problems)
    figure.suptitle("Worst run-level sanity status across seeds")
    return _save(figure, path)


def plot_decoder_by_problem(
    by_problem: pd.DataFrame,
    path: Path,
    *,
    decoders: tuple[str, ...] = EXPECTED_DECODERS,
    problems: tuple[str, ...] = EXPECTED_PROBLEMS,
) -> Path | None:
    if by_problem.empty:
        return None
    panels = _two_key_panels(by_problem)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(6.0, 5.2 * len(panels)), 5.2),
        squeeze=False,
    )
    for axis, ((scale, mode), frame) in zip(_axes_array(axes), panels, strict=True):
        pivot = frame.pivot(
            index="problem", columns="decoder", values="within_problem_rank_score_mean"
        ).reindex(index=problems, columns=decoders)
        values = pivot.to_numpy(dtype=float)
        labels = np.full(values.shape, "-", dtype=object)
        finite = np.isfinite(values)
        labels[finite] = np.char.mod("%.2f", values[finite])
        image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis")
        _annotate_heatmap(axis, values, labels)
        axis.set_title(f"{scale} / {mode}")
        axis.set_xticks(range(len(decoders)))
        axis.set_xticklabels(
            [DECODER_LABELS[item] for item in decoders],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(len(problems)))
        axis.set_yticklabels(problems)
    figure.colorbar(
        image, ax=_axes_array(axes).tolist(), label="within-problem rank score"
    )
    figure.suptitle("Decoder performance within each problem (1 = best)")
    return _save(figure, path)


def plot_decoder_across_problems(
    across: pd.DataFrame,
    path: Path,
    *,
    decoders: tuple[str, ...] = EXPECTED_DECODERS,
) -> Path | None:
    if across.empty:
        return None
    panels = _two_key_panels(across)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(6.0, 5.2 * len(panels)), 4.8),
        squeeze=False,
    )
    for axis, ((scale, mode), frame) in zip(_axes_array(axes), panels, strict=True):
        ordered = (
            frame.set_index("decoder")
            .reindex(decoders)
            .dropna(subset=["mean_problem_rank_score"])
        )
        positions = np.arange(len(ordered))
        axis.bar(
            positions,
            ordered["mean_problem_rank_score"],
            color=plt.cm.viridis(ordered["mean_problem_rank_score"]),
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [DECODER_LABELS[item] for item in ordered.index], rotation=35, ha="right"
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("mean within-problem rank score")
        axis.set_title(f"{scale} / {mode}")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Decoder performance across problem categories")
    return _save(figure, path)


def _validation_quality_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "val/objective" not in frame:
        return pd.DataFrame()
    rows = frame[frame["val/objective"].notna()].copy()
    if rows.empty:
        return rows
    rows["plot_value"] = rows.apply(
        lambda row: (
            -row["val/aggregate_gap_pct"]
            if "val/aggregate_gap_pct" in rows
            and pd.notna(row["val/aggregate_gap_pct"])
            else objective_sign(str(row["problem"])) * row["val/objective"]
        ),
        axis=1,
    )
    return rows


def _curve_summary(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    rows = frame[frame[value_column].notna() & frame["analysis_epoch"].notna()].copy()
    if rows.empty:
        return pd.DataFrame()
    per_run = (
        rows.groupby(["run_id", "problem", "decoder", "analysis_epoch"], dropna=False)[
            value_column
        ]
        .median()
        .reset_index()
    )
    return (
        per_run.groupby(["problem", "decoder", "analysis_epoch"], dropna=False)[
            value_column
        ]
        .agg(
            median="median",
            lower=lambda values: values.quantile(0.25),
            upper=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )


def plot_learning_curves(
    history: pd.DataFrame,
    figures_dir: Path,
    *,
    decoders: tuple[str, ...] = EXPECTED_DECODERS,
    problems: tuple[str, ...] = EXPECTED_PROBLEMS,
) -> list[Path]:
    written: list[Path] = []
    if history.empty:
        return written
    colors = {
        decoder: matplotlib.colormaps["tab10"](index)
        for index, decoder in enumerate(EXPECTED_DECODERS)
    }
    for (scale, mode), frame in _two_key_panels(history):
        if mode == "supervised":
            train_column = (
                "train/sl/loss_epoch"
                if "train/sl/loss_epoch" in frame
                and frame["train/sl/loss_epoch"].notna().any()
                else "train/sl/loss"
            )
            train_label = "Supervised loss\n(lower is better)"
        else:
            train_column = "train/rl/reward"
            train_label = "Training reward\n(higher is better)"
        if train_column not in frame:
            continue
        validation = _validation_quality_rows(frame)
        train_summary = _curve_summary(frame, train_column)
        validation_summary = _curve_summary(validation, "plot_value")
        plotted_problems = [
            problem for problem in problems if problem in set(frame["problem"])
        ]
        if not plotted_problems:
            continue
        problem_pairs_per_row = 2
        grid_rows = (
            len(plotted_problems) + problem_pairs_per_row - 1
        ) // problem_pairs_per_row
        figure, axes = plt.subplots(
            grid_rows,
            problem_pairs_per_row * 2,
            figsize=(18, max(3.4 * grid_rows, 7.0)),
            squeeze=False,
            layout="constrained",
        )
        used_axes: set[tuple[int, int]] = set()
        for problem_index, problem in enumerate(plotted_problems):
            grid_row = problem_index // problem_pairs_per_row
            first_column = (problem_index % problem_pairs_per_row) * 2
            if (
                len(plotted_problems) % problem_pairs_per_row
                and problem_index == len(plotted_problems) - 1
            ):
                first_column = 1
            for column_index, (summary, y_label) in enumerate(
                (
                    (train_summary, train_label),
                    (
                        validation_summary,
                        "Validation quality\n(higher is better)",
                    ),
                )
            ):
                axis_column = first_column + column_index
                used_axes.add((grid_row, axis_column))
                axis = axes[grid_row, axis_column]
                problem_rows = summary[summary["problem"] == problem]
                for decoder, decoder_rows in problem_rows.groupby(
                    "decoder", dropna=False
                ):
                    decoder_rows = decoder_rows.sort_values("analysis_epoch")
                    axis.plot(
                        decoder_rows["analysis_epoch"],
                        decoder_rows["median"],
                        label=DECODER_LABELS.get(str(decoder), str(decoder)),
                        color=colors.get(str(decoder)),
                        linewidth=1.8,
                    )
                    axis.fill_between(
                        decoder_rows["analysis_epoch"],
                        decoder_rows["lower"],
                        decoder_rows["upper"],
                        color=colors.get(str(decoder)),
                        alpha=0.12,
                    )
                axis.set_title(
                    f"{PROBLEM_LABELS.get(problem, problem)} — {y_label}",
                    fontsize=10,
                    pad=8,
                )
                axis.set_xlabel("Epoch")
                axis.margins(x=0.01)
                axis.grid(alpha=0.22)
                axis.set_axisbelow(True)

        for grid_row in range(grid_rows):
            for grid_column in range(problem_pairs_per_row * 2):
                if (grid_row, grid_column) not in used_axes:
                    axes[grid_row, grid_column].axis("off")

        plotted_decoders = set(train_summary["decoder"].dropna().astype(str)) | set(
            validation_summary["decoder"].dropna().astype(str)
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=colors[decoder],
                linewidth=2,
                label=DECODER_LABELS.get(decoder, decoder),
            )
            for decoder in decoders
            if decoder in plotted_decoders
        ]
        if legend_handles:
            figure.legend(
                handles=legend_handles,
                loc="outside upper center",
                ncols=len(legend_handles),
                frameon=False,
                fontsize=9,
                title=f"Learning curves: {scale} / {str(mode).upper()}",
                title_fontsize=13,
            )
        written.append(
            _save(figure, figures_dir / f"learning_curves_{scale}_{mode}.png")
        )
    return written


def plot_hypothesis(problem_contrasts: pd.DataFrame, path: Path) -> Path | None:
    if problem_contrasts.empty:
        return None
    panels = _two_key_panels(problem_contrasts)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(6.0, 5.4 * len(panels)), 5.0),
        squeeze=False,
    )
    colors = {"full_topology": "#4575b4", "partial_selection": "#d73027"}
    for axis, ((scale, mode), frame) in zip(_axes_array(axes), panels, strict=True):
        ordered = frame.sort_values(["solution_scope", "problem"])
        positions = np.arange(len(ordered))
        errors = ordered["recurrent_advantage_std"].fillna(0)
        axis.errorbar(
            positions,
            ordered["recurrent_advantage_mean"],
            yerr=errors,
            fmt="none",
            ecolor="#555555",
            capsize=3,
            alpha=0.7,
        )
        for scope, scope_rows in ordered.groupby("solution_scope", dropna=False):
            scope_positions = [
                ordered.index.get_loc(index) for index in scope_rows.index
            ]
            axis.scatter(
                scope_positions,
                scope_rows["recurrent_advantage_mean"],
                label=str(scope),
                color=colors.get(str(scope), "#777777"),
                s=45,
                zorder=3,
            )
        axis.axhline(0, color="black", linewidth=1, alpha=0.5)
        axis.set_xticks(positions)
        axis.set_xticklabels(ordered["problem"], rotation=35, ha="right")
        axis.set_ylabel("recurrent minus nonrecurrent standardized quality")
        axis.set_title(f"{scale} / {mode}")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=7)
    figure.suptitle("Predeclared recurrent-decoder hypothesis")
    return _save(figure, path)


def create_all_figures(
    *,
    coverage: pd.DataFrame,
    sanity: pd.DataFrame,
    history: pd.DataFrame,
    by_problem: pd.DataFrame,
    across_problems: pd.DataFrame,
    problem_contrasts: pd.DataFrame,
    figures_dir: Path,
    decoders: tuple[str, ...] = EXPECTED_DECODERS,
    problems: tuple[str, ...] = EXPECTED_PROBLEMS,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path | None] = [
        plot_coverage(
            coverage,
            figures_dir / "coverage.png",
            decoders=decoders,
            problems=problems,
        ),
        plot_sanity(
            sanity,
            figures_dir / "sanity_status.png",
            decoders=decoders,
            problems=problems,
        ),
        plot_decoder_by_problem(
            by_problem,
            figures_dir / "decoder_by_problem.png",
            decoders=decoders,
            problems=problems,
        ),
        plot_decoder_across_problems(
            across_problems,
            figures_dir / "decoder_across_problems.png",
            decoders=decoders,
        ),
        plot_hypothesis(problem_contrasts, figures_dir / "recurrent_hypothesis.png"),
    ]
    written = [path for path in paths if path is not None]
    written.extend(
        plot_learning_curves(
            history,
            figures_dir,
            decoders=decoders,
            problems=problems,
        )
    )
    return written
