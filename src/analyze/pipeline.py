from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.analyze.comparisons import (
    decoder_across_problems,
    decoder_by_problem,
    hypothesis_problem_contrasts,
    hypothesis_tests,
    pairwise_decoder_comparisons,
)
from src.analyze.loader import load_export
from src.analyze.metadata import (
    DEFAULT_EXPECTED_SEEDS,
    EXPECTED_DECODERS,
    EXPECTED_DYNAMICS,
    EXPECTED_MODES,
    EXPECTED_PROBLEMS,
)
from src.analyze.processing import add_within_problem_scores, process_export
from src.analyze.records import ProcessedData
from src.analyze.report import write_report
from src.analyze.results import (
    build_analysis_summary,
    write_result_tables,
    write_summary,
)
from src.analyze.sanity import build_coverage_table, build_sanity_table
from src.analyze.visualization import create_all_figures
from src.paths import DOCS_ROOT, WANDB_ANALYSIS_ROOT, resolve_user_path


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path = WANDB_ANALYSIS_ROOT / "raw"
    output_dir: Path = WANDB_ANALYSIS_ROOT / "results"
    report_path: Path = DOCS_ROOT / "W&B TSP-D Decoder Analysis.md"
    expected_scales: tuple[str, ...] = ("small",)
    expected_seeds: tuple[int, ...] = DEFAULT_EXPECTED_SEEDS
    expected_modes: tuple[str, ...] = EXPECTED_MODES
    expected_dynamics: tuple[str, ...] = EXPECTED_DYNAMICS
    excluded_decoders: tuple[str, ...] = ()
    excluded_problems: tuple[str, ...] = ()
    feasibility_threshold: float = 1.0 - 1e-6
    random_seed: int = 20260716


def _apply_scope_exclusions(
    processed: ProcessedData,
    excluded_decoders: tuple[str, ...],
    excluded_problems: tuple[str, ...],
) -> ProcessedData:
    decoder_exclusions = set(excluded_decoders)
    problem_exclusions = set(excluded_problems)

    def filtered(frame: Any) -> Any:
        result = frame.copy()
        if "decoder" in result:
            result = result[~result["decoder"].isin(decoder_exclusions)]
        if "problem" in result:
            result = result[~result["problem"].isin(problem_exclusions)]
        return result.copy()

    return ProcessedData(
        runs=filtered(processed.runs),
        selected_runs=filtered(processed.selected_runs),
        history=filtered(processed.history),
        final_metrics=filtered(processed.final_metrics),
        duplicate_runs=filtered(processed.duplicate_runs),
    )


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    unknown_decoders = set(config.excluded_decoders) - set(EXPECTED_DECODERS)
    if unknown_decoders:
        raise ValueError(f"Unknown excluded decoders: {sorted(unknown_decoders)}")
    included_decoders = tuple(
        decoder
        for decoder in EXPECTED_DECODERS
        if decoder not in set(config.excluded_decoders)
    )
    if not included_decoders:
        raise ValueError("At least one decoder must remain in the analysis")
    unknown_problems = set(config.excluded_problems) - set(EXPECTED_PROBLEMS)
    if unknown_problems:
        raise ValueError(f"Unknown excluded problems: {sorted(unknown_problems)}")
    included_problems = tuple(
        problem
        for problem in EXPECTED_PROBLEMS
        if problem not in set(config.excluded_problems)
    )
    if not included_problems:
        raise ValueError("At least one problem must remain in the analysis")
    unknown_dynamics = set(config.expected_dynamics) - set(EXPECTED_DYNAMICS)
    if unknown_dynamics:
        raise ValueError(f"Unknown expected dynamics: {sorted(unknown_dynamics)}")

    bundle = load_export(resolve_user_path(config.input_dir))
    processed = _apply_scope_exclusions(
        process_export(bundle),
        config.excluded_decoders,
        config.excluded_problems,
    )
    coverage = build_coverage_table(
        processed.runs,
        expected_scales=config.expected_scales,
        expected_seeds=config.expected_seeds,
        expected_modes=config.expected_modes,
        expected_decoders=included_decoders,
        expected_problems=included_problems,
        expected_dynamics=config.expected_dynamics,
    )
    sanity = build_sanity_table(
        processed.selected_runs,
        processed.history,
        feasibility_threshold=config.feasibility_threshold,
    )
    final_metrics = processed.final_metrics.merge(
        sanity[["run_id", "sanity_status"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    )
    final_metrics["comparison_eligible"] = final_metrics["sanity_status"].isin(
        ["pass", "warning"]
    )
    comparison_metrics = add_within_problem_scores(
        final_metrics[final_metrics["comparison_eligible"]].copy()
    )
    by_problem = decoder_by_problem(comparison_metrics)
    across_problems = decoder_across_problems(by_problem)
    pairwise = pairwise_decoder_comparisons(comparison_metrics)
    problem_contrasts = hypothesis_problem_contrasts(comparison_metrics)
    hypothesis = hypothesis_tests(
        problem_contrasts,
        random_seed=config.random_seed,
    )

    output_dir = resolve_user_path(config.output_dir).resolve()
    figures = create_all_figures(
        coverage=coverage,
        sanity=sanity,
        history=processed.history,
        by_problem=by_problem,
        across_problems=across_problems,
        problem_contrasts=problem_contrasts,
        figures_dir=output_dir / "figures",
        decoders=included_decoders,
        problems=included_problems,
    )
    table_paths = write_result_tables(
        output_dir,
        runs=processed.runs,
        selected_runs=processed.selected_runs,
        duplicate_runs=processed.duplicate_runs,
        history=processed.history,
        coverage=coverage,
        sanity=sanity,
        final_metrics=final_metrics,
        by_problem=by_problem,
        across_problems=across_problems,
        pairwise=pairwise,
        problem_contrasts=problem_contrasts,
        hypothesis=hypothesis,
    )
    summary = build_analysis_summary(
        source=bundle.manifest,
        coverage=coverage,
        sanity=sanity,
        by_problem=by_problem,
        across_problems=across_problems,
        hypothesis=hypothesis,
        figures=figures,
        included_decoders=included_decoders,
        excluded_decoders=config.excluded_decoders,
        included_problems=included_problems,
        excluded_problems=config.excluded_problems,
    )
    summary_path = write_summary(output_dir / "analysis.json", summary)
    report_path = write_report(
        resolve_user_path(config.report_path).resolve(),
        artifacts_dir=output_dir,
        source=bundle.manifest,
        selected_runs=processed.selected_runs,
        coverage=coverage,
        sanity=sanity,
        by_problem=by_problem,
        across_problems=across_problems,
        hypothesis=hypothesis,
        included_decoders=included_decoders,
        excluded_decoders=config.excluded_decoders,
        included_problems=included_problems,
        excluded_problems=config.excluded_problems,
    )
    return {
        "output_dir": output_dir,
        "tables": table_paths,
        "figures": figures,
        "summary": summary_path,
        "report": report_path,
    }
