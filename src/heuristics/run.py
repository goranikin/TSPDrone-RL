"""CLI: run the nearest-neighbor TSP-D heuristic and write organized results.

Examples
--------
::

    uv run python -m src.heuristics \\
      --n-nodes 11 --test-size 100

    uv run python -m src.heuristics --all-sizes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.heuristics.nearest_neighbor import (
    NearestNeighborResult,
    solve_nearest_neighbor,
)
from src.paths import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, REPOSITORY_ROOT, resolve_user_path
from src.problems.tspd import create_test_dataset

DEFAULT_SIZES = (11, 15, 20, 50, 100)


def _resolve_data_dir(path: str | Path) -> Path:
    candidate = resolve_user_path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def run_nearest_neighbor(
    *,
    n_nodes: int,
    test_size: int = 100,
    v_d: float = 2.0,
    decode_len: int | None = None,
    data_dir: Path | None = None,
) -> NearestNeighborResult:
    data_dir = data_dir or DEFAULT_DATA_DIR
    data = create_test_dataset(
        data_dir=_resolve_data_dir(data_dir),
        test_size=test_size,
        n_nodes=n_nodes,
    )
    if data.shape[0] > test_size:
        data = data[:test_size]
    start = time.perf_counter()
    makespans, unfinished = solve_nearest_neighbor(
        data,
        n_nodes=n_nodes,
        v_d=v_d,
        decode_len=decode_len,
    )
    elapsed = time.perf_counter() - start
    return NearestNeighborResult(
        makespans=makespans,
        mean=float(makespans.mean()),
        std=float(makespans.std()),
        min=float(makespans.min()),
        max=float(makespans.max()),
        n_nodes=n_nodes,
        n_instances=int(makespans.size),
        decode_len=decode_len if decode_len is not None else max(round(n_nodes * 3), 30),
        unfinished=unfinished,
        elapsed_sec=elapsed,
        v_d=v_d,
    )


def result_dir(
    *,
    output_root: Path,
    n_nodes: int,
    test_size: int,
) -> Path:
    return output_root / "heuristics" / f"nearest_neighbor-n{n_nodes}-size-{test_size}"


def write_result(result: NearestNeighborResult, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    makespan_path = out_dir / "makespans.txt"
    np.savetxt(makespan_path, result.makespans, fmt="%.6f")

    summary: dict[str, Any] = {
        "heuristic": "nearest_neighbor",
        "description": (
            "Smallest-distance constructive TSP-D heuristic: at each step pick "
            "the feasible truck node with min Euclidean distance, then the "
            "feasible drone node with min travel time (distance / v_d)."
        ),
        "n_nodes": result.n_nodes,
        "n_instances": result.n_instances,
        "v_d": result.v_d,
        "v_t": 1.0,
        "decode_len": result.decode_len,
        "unfinished": result.unfinished,
        "elapsed_sec": result.elapsed_sec,
        "makespan_mean": result.mean,
        "makespan_std": result.std,
        "makespan_min": result.min,
        "makespan_max": result.max,
        "makespans_file": makespan_path.name,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def write_aggregate_table(
    summaries: list[dict[str, Any]],
    output_root: Path,
) -> Path:
    table_dir = output_root / "heuristics"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "nearest_neighbor_summary.json"
    payload = {
        "heuristic": "nearest_neighbor",
        "runs": summaries,
        "table": [
            {
                "n_nodes": row["n_nodes"],
                "n_instances": row["n_instances"],
                "makespan_mean": row["makespan_mean"],
                "makespan_std": row["makespan_std"],
                "unfinished": row["unfinished"],
                "elapsed_sec": row["elapsed_sec"],
            }
            for row in summaries
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    md_path = table_dir / "nearest_neighbor_summary.md"
    lines = [
        "# Nearest-neighbor (smallest-distance) TSP-D heuristic",
        "",
        "Constructive baseline: truck then drone each pick the **nearest "
        "feasible** node under the same availability rules as the RL env.",
        "",
        "| n_nodes | instances | mean makespan | std | unfinished | time (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['n_nodes']} | {row['n_instances']} | "
            f"{row['makespan_mean']:.4f} | {row['makespan_std']:.4f} | "
            f"{row['unfinished']} | {row['elapsed_sec']:.3f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run nearest-neighbor TSP-D heuristic and save results.",
    )
    parser.add_argument("--n-nodes", type=int, default=11, help="Problem size N.")
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of test instances (uses/creates DroneTruck-size-*.txt).",
    )
    parser.add_argument(
        "--all-sizes",
        action="store_true",
        help=f"Run on all bundled sizes {DEFAULT_SIZES}.",
    )
    parser.add_argument("--v-d", type=float, default=2.0, help="Drone speed.")
    parser.add_argument(
        "--decode-len",
        type=int,
        default=None,
        help="Max decision pairs (default: max(3N, 30)).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with DroneTruck-size-*.txt files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root for results/heuristics/...",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    output_root = resolve_user_path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root

    sizes = list(DEFAULT_SIZES) if args.all_sizes else [args.n_nodes]
    summaries: list[dict[str, Any]] = []

    for n_nodes in sizes:
        print(f"nearest_neighbor: n_nodes={n_nodes} test_size={args.test_size} ...")
        result = run_nearest_neighbor(
            n_nodes=n_nodes,
            test_size=args.test_size,
            v_d=args.v_d,
            decode_len=args.decode_len,
            data_dir=args.data_dir,
        )
        out = result_dir(
            output_root=output_root,
            n_nodes=n_nodes,
            test_size=args.test_size,
        )
        summary = write_result(result, out)
        summaries.append(summary)
        print(
            f"  mean={result.mean:.4f} std={result.std:.4f} "
            f"unfinished={result.unfinished} "
            f"elapsed={result.elapsed_sec:.3f}s → {out}"
        )

    aggregate_path = write_aggregate_table(summaries, output_root)
    print(f"aggregate summary → {aggregate_path}")
    return {"runs": summaries, "aggregate": str(aggregate_path)}


if __name__ == "__main__":
    main()
