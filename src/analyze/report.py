from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _artifact_link(label: str, relative_path: str, artifacts_dir: Path) -> str:
    target = (artifacts_dir / relative_path).resolve().as_posix()
    return f"[{label}]({target})"


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value).replace("|", "\\|")


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    limit: int | None = None,
) -> str:
    if frame.empty:
        return "No rows are available."
    available = [column for column in columns if column in frame]
    view = frame[available].head(limit) if limit is not None else frame[available]
    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join("---" for _ in available) + " |"
    rows = [
        "| " + " | ".join(_format_value(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _coverage_section(coverage: pd.DataFrame, artifacts_dir: Path) -> str:
    expected = coverage[coverage["expected"]]
    total = len(expected)
    complete = int(expected["coverage_status"].eq("complete").sum())
    missing = int(expected["coverage_status"].eq("missing").sum())
    duplicates = int(expected["coverage_status"].eq("duplicate").sum())
    variants = int(expected["coverage_status"].eq("configuration_variant").sum())
    not_finished = int(expected["coverage_status"].eq("not_finished").sum())
    return (
        f"The configured grid contains **{total} expected seed-level cells**. "
        f"{complete} are complete, {missing} are missing, {duplicates} have duplicate "
        f"reruns, {variants} mix different configurations, and {not_finished} have no "
        "finished run. Configuration variants are not selected for comparisons. See "
        f"{_artifact_link('coverage.csv', 'coverage.csv', artifacts_dir)} and "
        f"{_artifact_link('coverage.png', 'figures/coverage.png', artifacts_dir)}."
    )


def _sanity_section(sanity: pd.DataFrame, artifacts_dir: Path) -> str:
    if sanity.empty:
        return "No canonical runs were available for sanity checks."
    counts = sanity["sanity_status"].value_counts()
    formatted = ", ".join(
        f"{status}={int(counts.get(status, 0))}"
        for status in ("pass", "warning", "incomplete", "fail")
    )
    failures = sanity[sanity["sanity_status"].isin(["fail", "incomplete"])].copy()
    failure_table = (
        "No runs failed or were incomplete."
        if failures.empty
        else _markdown_table(
            failures,
            [
                "problem",
                "decoder",
                "mode",
                "seed",
                "scale",
                "sanity_status",
                "fail_reasons",
                "incomplete_reasons",
            ],
            limit=30,
        )
    )
    return (
        f"Run-level result: **{formatted}**. Supervised runs require finite loss, "
        "held-out quality, completed epochs, and feasible evaluation outputs. RL policy "
        "loss is checked for finiteness but not monotonic decrease; RL learning evidence "
        "comes from reward and held-out quality.\n\n"
        f"{failure_table}\n\n"
        "Full diagnostics are in "
        f"{_artifact_link('run_sanity.csv', 'run_sanity.csv', artifacts_dir)}, "
        "with visual status in "
        f"{_artifact_link('sanity_status.png', 'figures/sanity_status.png', artifacts_dir)}."
    )


def _provenance_section(selected_runs: pd.DataFrame) -> str:
    if selected_runs.empty:
        return "No selected runs were available for provenance checks."
    regimes = selected_runs["comparison_regime"].value_counts().to_dict()
    regime_text = ", ".join(
        f"{regime}={count}" for regime, count in sorted(regimes.items())
    )
    reference_known = (
        int(selected_runs["reference_kind"].notna().sum())
        if "reference_kind" in selected_runs
        else 0
    )
    message = (
        f"Comparison regimes: **{regime_text}**. Results from different regimes or "
        "comparison-condition fingerprints are never pooled. "
    )
    if reference_known < len(selected_runs):
        message += (
            f"Only {reference_known}/{len(selected_runs)} selected runs contain the "
            "new dataset reference provenance. Runs without it are legacy exports: "
            "their solver exactness cannot be recovered from W&B config, so reported "
            "gaps must be interpreted as comparisons to stored solver labels, not "
            "certified optimality gaps."
        )
    else:
        kinds = selected_runs["reference_kind"].value_counts().to_dict()
        kind_text = ", ".join(
            f"{kind}={count}" for kind, count in sorted(kinds.items())
        )
        message += f"Dataset reference provenance is present for every run: {kind_text}."
    return message


def _by_problem_section(by_problem: pd.DataFrame, artifacts_dir: Path) -> str:
    if by_problem.empty:
        return "No final held-out metrics were available."
    ranked = by_problem.sort_values(
        [
            "scale",
            "mode",
            "comparison_regime",
            "problem",
            "within_problem_rank_score_mean",
        ],
        ascending=[True, True, True, True, False],
    )
    table = _markdown_table(
        ranked,
        [
            "scale",
            "mode",
            "comparison_regime",
            "problem",
            "decoder",
            "seeds",
            "within_problem_rank_score_mean",
            "aggregate_gap_pct_mean",
            "feasibility_rate_mean",
        ],
        limit=60,
    )
    return (
        "The rank score is computed only among decoders in the same problem, mode, "
        "scale, seed, and encoder cell (1 is best; 0 is worst). Aggregate gap percent "
        "uses the ratio of mean objective regret to the absolute mean solver "
        "reference, avoiding "
        "unstable per-instance percentages when a reference is close to zero. Runs with "
        "`fail` or `incomplete` sanity status are retained in `final_metrics.csv` but "
        "excluded from these performance comparisons.\n\n"
        "Fixed-width and total-parameter-matched runs are reported separately. The "
        "comparison-condition fingerprint also prevents "
        "different training or model controls from being pooled.\n\n"
        f"{table}\n\n"
        f"See {_artifact_link('decoder_by_problem.csv', 'decoder_by_problem.csv', artifacts_dir)}, "
        f"{_artifact_link('pairwise_decoder_comparisons.csv', 'pairwise_decoder_comparisons.csv', artifacts_dir)}, and "
        f"{_artifact_link('decoder_by_problem.png', 'figures/decoder_by_problem.png', artifacts_dir)}."
    )


def _across_problem_section(across: pd.DataFrame, artifacts_dir: Path) -> str:
    if across.empty:
        return "No cross-problem comparison was available."
    ranked = across.sort_values(
        ["scale", "mode", "comparison_regime", "mean_problem_rank_score"],
        ascending=[True, True, True, False],
    )
    return (
        "Raw objectives are not averaged across categories. Each problem is first "
        "normalized through within-problem rank/standardized scores and then receives "
        "equal weight.\n\n"
        + _markdown_table(
            ranked,
            [
                "scale",
                "mode",
                "comparison_regime",
                "decoder",
                "problems",
                "mean_problem_rank_score",
                "mean_problem_z",
                "mean_feasibility_rate",
            ],
        )
        + "\n\nSee "
        + _artifact_link(
            "decoder_across_problems.csv", "decoder_across_problems.csv", artifacts_dir
        )
        + " and "
        + _artifact_link(
            "decoder_across_problems.png",
            "figures/decoder_across_problems.png",
            artifacts_dir,
        )
        + "."
    )


def _hypothesis_section(
    hypothesis: pd.DataFrame,
    artifacts_dir: Path,
    included_decoders: tuple[str, ...],
) -> str:
    if hypothesis.empty:
        return (
            "The hypothesis could not be evaluated because no cells contained all five "
            "autoregressive comparison decoders."
        )
    sigmoid_scope = (
        "`sigmoid_subset` remains in coverage and descriptive results but is "
        "excluded here because it is an independent subset policy, not an "
        "autoregressive decoder. "
        if "sigmoid_subset" in included_decoders
        else "`sigmoid_subset` was excluded from this entire analysis. It would also "
        "be outside this hypothesis because it is not an autoregressive decoder. "
    )
    return (
        "The predeclared comparison is **recurrent** (`lstm_pointer`, `gru_pointer`) "
        "versus **nonrecurrent autoregressive** (`attention_model`, "
        "`attention_model_without_glimpse`, `transformer_pointer`). "
        + sigmoid_scope
        + "A positive interaction means the recurrent family is "
        "relatively stronger on partial-selection problems than on full-topology routing "
        "problems.\n\n"
        + _markdown_table(
            hypothesis,
            [
                "scale",
                "mode",
                "comparison_regime",
                "partial_problems",
                "full_topology_problems",
                "interaction_effect",
                "bootstrap_ci_95_low",
                "bootstrap_ci_95_high",
                "exact_permutation_p_value",
                "direction_supports_hypothesis",
            ],
        )
        + "\n\nThis is a problem-level association, not a causal architecture result: solution "
        "scope is confounded with graph structure, encoder choice, feasibility masks, and "
        "problem family. With only two full-topology categories, the exact permutation "
        "test is necessarily coarse. Treat the effect size and per-problem consistency as "
        "the primary evidence. See "
        + _artifact_link(
            "hypothesis_problem_contrasts.csv",
            "hypothesis_problem_contrasts.csv",
            artifacts_dir,
        )
        + ", "
        + _artifact_link("hypothesis_tests.csv", "hypothesis_tests.csv", artifacts_dir)
        + ", and "
        + _artifact_link(
            "recurrent_hypothesis.png",
            "figures/recurrent_hypothesis.png",
            artifacts_dir,
        )
        + "."
    )


def write_report(
    path: Path,
    *,
    artifacts_dir: Path,
    source: dict[str, Any],
    selected_runs: pd.DataFrame,
    coverage: pd.DataFrame,
    sanity: pd.DataFrame,
    by_problem: pd.DataFrame,
    across_problems: pd.DataFrame,
    hypothesis: pd.DataFrame,
    included_decoders: tuple[str, ...],
    excluded_decoders: tuple[str, ...],
    included_problems: tuple[str, ...],
    excluded_problems: tuple[str, ...],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    included_text = ", ".join(f"`{decoder}`" for decoder in included_decoders)
    excluded_text = (
        ", ".join(f"`{decoder}`" for decoder in excluded_decoders)
        if excluded_decoders
        else "none"
    )
    included_problem_text = ", ".join(f"`{problem}`" for problem in included_problems)
    excluded_problem_text = (
        ", ".join(f"`{problem}`" for problem in excluded_problems)
        if excluded_problems
        else "none"
    )
    content = f"""# W&B architecture comparison analysis

- Source: `{source.get("entity")}/{source.get("project")}`
- Fetched: `{source.get("fetched_at")}`
- Raw exported runs: `{source.get("run_count", 0)}`
- Analyzed canonical runs: `{len(selected_runs)}`
- Included decoders: {included_text}
- Excluded decoders: {excluded_text}
- Included problems: {included_problem_text}
- Excluded problems: {excluded_problem_text}

## Experimental sanity check

{_coverage_section(coverage, artifacts_dir)}

{_sanity_section(sanity, artifacts_dir)}

{_provenance_section(selected_runs)}

Learning curves are saved separately by scale and training mode under `figures/`.

## Decoder performance within each problem

{_by_problem_section(by_problem, artifacts_dir)}

## Decoder performance across problem categories

{_across_problem_section(across_problems, artifacts_dir)}

## Recurrent decoder hypothesis

{_hypothesis_section(hypothesis, artifacts_dir, included_decoders)}

## Interpretation rules

- `quality_value` is always oriented so larger is better. When reference objectives exist,
  it is negative stable aggregate gap percent; otherwise it is the direction-corrected
  objective.
- Test metrics are preferred. If a run has no test metric, its best validation row is
  used and marked `val_best` in `final_metrics.csv`.
- A W&B `finished` state and completed epochs establish completion, not learning. Loss,
  reward, held-out quality, feasibility, and final comparisons provide separate evidence.
- Statistical comparisons pair identical problem/mode/scale/seed/encoder cells. With
  three seeds, p-values have low power and should accompany effect sizes and seed wins.
"""
    path.write_text(content, encoding="utf-8")
    return path
