import argparse
import json
from pathlib import Path
from typing import Any

from src.analyze.fetch import fetch_project
from src.analyze.metadata import (
    DEFAULT_EXPECTED_SEEDS,
    EXPECTED_DECODERS,
    EXPECTED_MODES,
    EXPECTED_PROBLEMS,
)
from src.analyze.pipeline import AnalysisConfig, run_analysis
from src.paths import DOCS_ROOT, WANDB_ANALYSIS_ROOT, resolve_user_path

DEFAULT_ENTITY = "goranikin-my-project"
DEFAULT_PROJECT = "compare-architectures"
DEFAULT_ROOT = WANDB_ANALYSIS_ROOT
DEFAULT_REPORT = DOCS_ROOT / "W&B Architecture Comparison Analysis.md"


def _filters(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if value.startswith("@"):
        payload = Path(value[1:]).read_text(encoding="utf-8")
    else:
        payload = value
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise argparse.ArgumentTypeError("W&B filters must decode to a JSON object")
    return decoded


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--filters-json",
        help="W&B API filters as JSON, or @path/to/filters.json",
    )
    parser.add_argument("--refresh", action="store_true")


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-scales",
        nargs="+",
        default=["small"],
        help="Scales whose complete matrix should be present",
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_EXPECTED_SEEDS),
    )
    parser.add_argument(
        "--expected-modes",
        nargs="+",
        choices=EXPECTED_MODES,
        default=list(EXPECTED_MODES),
        help="Training modes whose complete matrix should be present",
    )
    parser.add_argument(
        "--exclude-decoders",
        nargs="+",
        choices=EXPECTED_DECODERS,
        default=[],
        help="Decoders to exclude from all analysis tables, statistics, and figures",
    )
    parser.add_argument(
        "--exclude-problems",
        nargs="+",
        choices=EXPECTED_PROBLEMS,
        default=[],
        help="Problems to exclude from all analysis tables, statistics, and figures",
    )
    parser.add_argument("--feasibility-threshold", type=float, default=1.0 - 1e-6)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and analyze compare-architectures W&B runs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Export W&B metadata and history"
    )
    _add_source_arguments(fetch_parser)
    fetch_parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "raw")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a local W&B export")
    analyze_parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT / "raw")
    analyze_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ROOT / "results"
    )
    _add_analysis_arguments(analyze_parser)

    run_parser = subparsers.add_parser("run", help="Fetch and analyze in one command")
    _add_source_arguments(run_parser)
    run_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_ROOT / "raw")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "results")
    _add_analysis_arguments(run_parser)
    return parser


def _analysis_config(arguments: argparse.Namespace, input_dir: Path) -> AnalysisConfig:
    return AnalysisConfig(
        input_dir=resolve_user_path(input_dir),
        output_dir=resolve_user_path(arguments.output_dir),
        report_path=resolve_user_path(arguments.report_path),
        expected_scales=tuple(arguments.expected_scales),
        expected_seeds=tuple(arguments.expected_seeds),
        expected_modes=tuple(arguments.expected_modes),
        excluded_decoders=tuple(arguments.exclude_decoders),
        excluded_problems=tuple(arguments.exclude_problems),
        feasibility_threshold=arguments.feasibility_threshold,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "fetch":
        manifest = fetch_project(
            entity=arguments.entity,
            project=arguments.project,
            output_dir=resolve_user_path(arguments.output_dir),
            filters=_filters(arguments.filters_json),
            refresh=arguments.refresh,
        )
        print(f"W&B export written to {manifest}")
        return 0

    if arguments.command == "analyze":
        result = run_analysis(_analysis_config(arguments, arguments.input_dir))
        print(f"Analysis report written to {result['report']}")
        return 0

    fetch_project(
        entity=arguments.entity,
        project=arguments.project,
        output_dir=resolve_user_path(arguments.raw_dir),
        filters=_filters(arguments.filters_json),
        refresh=arguments.refresh,
    )
    result = run_analysis(_analysis_config(arguments, arguments.raw_dir))
    print(f"Analysis report written to {result['report']}")
    return 0
