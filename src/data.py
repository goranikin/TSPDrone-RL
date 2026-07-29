import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from src.constants import ProblemName
from src.paths import resolve_user_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_user_path(path)
    records: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            records.append(record)
    return records


class _ProblemRecordParser:
    def __init__(
        self,
        path: str | Path,
        problem: ProblemName,
        target_algorithm: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.path = str(resolve_user_path(path))
        self.problem = problem
        self.target_algorithm = target_algorithm
        self.dtype = dtype

    def parse(self, raw: dict[str, Any], index: int) -> dict[str, Any]:
        record = self._normalise_record(raw)
        if _canonical_problem(record.get("problem")) != self.problem:
            raise ValueError(
                f"Record {index} in {self.path} has "
                f"problem={record.get('problem')!r}; expected {self.problem!r}"
            )
        if self.problem == "tsp":
            return self._tsp(record)
        if self.problem == "cvrp":
            return self._cvrp(record)
        if self.problem == "orienteering":
            return self._orienteering(record)
        if self.problem == "knapsack":
            return self._knapsack(record)
        if self.problem in ("mis", "max_clique", "vertex_cover"):
            return self._graph_subset(record)
        raise ValueError(f"Unsupported problem: {self.problem}")

    @staticmethod
    def _normalise_record(raw: dict[str, Any]) -> dict[str, Any]:
        if "instance" not in raw:
            return dict(raw)
        instance = raw["instance"]
        if not isinstance(instance, dict):
            raise ValueError("Nested JSONL row field 'instance' must be an object")
        record = dict(instance)
        for key in ("problem", "index", "seed"):
            if key in raw and key not in record:
                record[key] = raw[key]
        if "solutions" not in record and "label" in raw:
            label = raw["label"]
            if not isinstance(label, dict):
                raise ValueError("Nested JSONL row field 'label' must be an object")
            solver = raw.get("solver") or label.get("algorithm") or "solver"
            record["solutions"] = {str(solver): label}
        return record

    def _base_item(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": torch.tensor(int(record.get("index", 0)), dtype=torch.long),
            "seed": torch.tensor(int(record.get("seed", 0)), dtype=torch.long),
        }

    def _solution(self, record: dict[str, Any]) -> dict[str, Any] | None:
        solutions = record.get("solutions")
        if not isinstance(solutions, dict):
            return None
        if not solutions:
            return None
        if self.target_algorithm is not None:
            if self.target_algorithm not in solutions:
                available = ", ".join(str(key) for key in solutions) or "none"
                raise ValueError(
                    f"{self.path} record does not contain requested target_algorithm="
                    f"{self.target_algorithm!r}. Available: {available}"
                )
            solution = solutions[self.target_algorithm]
            return _as_string_dict(solution)
        if len(solutions) == 1:
            solution = next(iter(solutions.values()))
            return _as_string_dict(solution)
        available = ", ".join(str(key) for key in solutions) or "none"
        raise ValueError(
            f"{self.path} record has multiple solutions; set target_algorithm. "
            f"Available: {available}"
        )

    def _tsp(self, record: dict[str, Any]) -> dict[str, Any]:
        loc = torch.tensor(record["coordinates"], dtype=self.dtype)
        item = self._base_item(record)
        item.update({"loc": loc})
        solution = self._solution(record)
        if solution is not None and "tour" in solution:
            tour = _canonical_cycle(solution["tour"], record["coordinates"])
            target = torch.tensor(tour, dtype=torch.long)
            item["target_actions"] = target
            item["target_tour"] = target
            item["target_mask"] = torch.ones(loc.size(0), dtype=self.dtype)
            if "cost" in solution:
                item["target_value"] = torch.tensor(solution["cost"], dtype=self.dtype)
        return item

    def _cvrp(self, record: dict[str, Any]) -> dict[str, Any]:
        depot = torch.tensor(record["depot"], dtype=self.dtype)
        coordinates = torch.tensor(record["coordinates"], dtype=self.dtype)
        loc = torch.cat([depot.view(1, -1), coordinates], dim=0)
        demands = torch.tensor(record["demands"], dtype=self.dtype)
        capacity = torch.tensor(
            record.get("vehicle_capacity", record.get("capacity")),
            dtype=self.dtype,
        )
        node_demands = torch.cat([torch.zeros(1, dtype=self.dtype), demands])
        item = self._base_item(record)
        item.update(
            {
                "depot": depot,
                "coordinates": coordinates,
                "loc": loc,
                "demands": demands,
                "node_demands": node_demands,
                "capacity": capacity,
            }
        )
        solution = self._solution(record)
        if solution is not None:
            actions = _cvrp_solution_actions(solution, record["coordinates"])
            if actions:
                item["target_actions"] = torch.tensor(actions, dtype=torch.long)
            mask = torch.zeros(loc.size(0), dtype=self.dtype)
            mask[1:] = 1.0
            item["target_mask"] = mask
            if "cost" in solution:
                item["target_value"] = torch.tensor(solution["cost"], dtype=self.dtype)
        return item

    def _orienteering(self, record: dict[str, Any]) -> dict[str, Any]:
        depot = torch.tensor(record["depot"], dtype=self.dtype)
        coordinates = torch.tensor(record["coordinates"], dtype=self.dtype)
        loc = torch.cat([depot.view(1, -1), coordinates], dim=0)
        prizes = torch.tensor(record["prizes"], dtype=self.dtype)
        node_prizes = torch.cat([torch.zeros(1, dtype=self.dtype), prizes])
        item = self._base_item(record)
        item.update(
            {
                "depot": depot,
                "coordinates": coordinates,
                "loc": loc,
                "prizes": prizes,
                "node_prizes": node_prizes,
                "travel_budget": torch.tensor(
                    record["travel_budget"],
                    dtype=self.dtype,
                ),
            }
        )
        solution = self._solution(record)
        if solution is not None:
            canonical_tour = _canonical_path(
                solution.get("tour", []),
                record["coordinates"],
            )
            tour = [int(node) + 1 for node in canonical_tour]
            item["target_actions"] = torch.tensor([*tour, 0], dtype=torch.long)
            target_mask = torch.zeros(loc.size(0), dtype=self.dtype)
            if tour:
                target_mask[torch.tensor(tour, dtype=torch.long)] = 1.0
            item["target_mask"] = target_mask
            if "value" in solution:
                item["target_value"] = torch.tensor(
                    solution["value"],
                    dtype=self.dtype,
                )
            if "length" in solution:
                item["target_length"] = torch.tensor(
                    solution["length"],
                    dtype=self.dtype,
                )
        return item

    def _knapsack(self, record: dict[str, Any]) -> dict[str, Any]:
        weights = torch.tensor(record["weights"], dtype=self.dtype)
        values = torch.tensor(record["values"], dtype=self.dtype)
        capacity = torch.tensor(record["capacity"], dtype=self.dtype)
        item = self._base_item(record)
        item.update({"weights": weights, "values": values, "capacity": capacity})
        solution = self._solution(record)
        if solution is not None:
            selected = [int(item_id) for item_id in solution.get("items", [])]
            target_mask = torch.zeros(weights.size(0), dtype=self.dtype)
            if selected:
                target_mask[torch.tensor(selected, dtype=torch.long)] = 1.0
            item["target_mask"] = target_mask
            item["target_actions"] = torch.tensor(
                [*selected, weights.size(0)],
                dtype=torch.long,
            )
            if "value" in solution:
                item["target_value"] = torch.tensor(
                    solution["value"],
                    dtype=self.dtype,
                )
            if "weight" in solution:
                item["target_weight"] = torch.tensor(
                    solution["weight"],
                    dtype=self.dtype,
                )
        return item

    def _graph_subset(self, record: dict[str, Any]) -> dict[str, Any]:
        num_nodes = int(record["num_nodes"])
        adjacency = _edges_to_adjacency(num_nodes, record.get("edges", []))
        item = self._base_item(record)
        item.update(
            {
                "adjacency": torch.tensor(adjacency, dtype=self.dtype),
                "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
            }
        )
        solution = self._solution(record)
        if solution is not None:
            selected = [int(node) for node in solution.get("nodes", [])]
            target_mask = torch.zeros(num_nodes, dtype=self.dtype)
            if selected:
                target_mask[torch.tensor(selected, dtype=torch.long)] = 1.0
            item["target_mask"] = target_mask
            item["target_actions"] = torch.tensor(
                [*selected, num_nodes],
                dtype=torch.long,
            )
            if "size" in solution:
                item["target_value"] = torch.tensor(
                    solution["size"],
                    dtype=self.dtype,
                )
        return item


class ProblemDataset(Dataset):
    """Load a reusable JSONL dataset into memory."""

    def __init__(
        self,
        path: str | Path,
        problem: ProblemName,
        target_algorithm: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.parser = _ProblemRecordParser(path, problem, target_algorithm, dtype)
        self.records = read_jsonl(path)
        for index, raw in enumerate(self.records):
            record = self.parser._normalise_record(raw)
            if _canonical_problem(record.get("problem")) != problem:
                raise ValueError(
                    f"Record {index} in {path} has problem={record.get('problem')!r}; "
                    f"expected {problem!r}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.parser.parse(self.records[index], index)

    def provenance(self) -> dict[str, Any]:
        return _dataset_provenance(
            path=Path(self.parser.path),
            parser=self.parser,
            records=self.records,
        )


class StreamingProblemDataset(IterableDataset):
    """Read a large JSONL training stream once without retaining it in memory."""

    def __init__(
        self,
        path: str | Path,
        problem: ProblemName,
        target_algorithm: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.path = resolve_user_path(path)
        self.parser = _ProblemRecordParser(path, problem, target_algorithm, dtype)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if (line_number - 1) % worker_count != worker_id:
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {self.path}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL line {line_number} must contain an object")
                yield self.parser.parse(record, line_number - 1)

    def provenance(self) -> dict[str, Any]:
        return {
            **_file_provenance(self.path),
            "problem": self.parser.problem,
            "target_algorithm_requested": self.parser.target_algorithm,
            "streaming": True,
            "reference_kind": "unlabeled_stream",
        }


def build_dataloader(
    path: str | Path,
    problem: ProblemName,
    *,
    batch_size: int,
    target_algorithm: str | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    stream: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if stream and shuffle:
        raise ValueError("streaming datasets cannot be shuffled by the DataLoader")
    dataset = (
        StreamingProblemDataset(path, problem, target_algorithm=target_algorithm)
        if stream
        else ProblemDataset(path, problem, target_algorithm=target_algorithm)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not stream else False,
        num_workers=num_workers,
        collate_fn=collate_problem_batch,
        generator=generator,
    )


def collate_problem_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    keys = set().union(*(item.keys() for item in items))
    num_nodes = _stack_num_nodes(items)
    max_graph_nodes = _max_graph_nodes(items)
    batch: dict[str, Any] = {}
    for key in sorted(keys):
        if key == "num_nodes":
            continue
        values = [item.get(key) for item in items]
        present = [value for value in values if value is not None]
        if not present:
            continue
        first = present[0]
        if not isinstance(first, torch.Tensor):
            batch[key] = values
            continue
        if all(isinstance(value, torch.Tensor) for value in values):
            tensors = [value for value in values if isinstance(value, torch.Tensor)]
            if _same_shape(tensors):
                batch[key] = torch.stack(tensors)
            elif _is_square_adjacency(tensors):
                batch[key] = _pad_square_2d(tensors, pad_value=0.0)
            elif tensors[0].ndim == 1 and tensors[0].dtype == torch.long:
                padded = _pad_1d_long(tensors, pad_value=-1)
                if (
                    key == "target_actions"
                    and num_nodes is not None
                    and max_graph_nodes is not None
                ):
                    padded = _remap_graph_stop_actions(
                        padded,
                        num_nodes,
                        max_graph_nodes,
                    )
                batch[key] = padded
            elif tensors[0].ndim == 1:
                batch[key] = _pad_1d_float(tensors, pad_value=0.0)
            else:
                raise ValueError(f"Cannot collate variable-shape tensor key {key!r}")
        else:
            batch[key] = values
    if num_nodes is not None:
        batch["num_nodes"] = num_nodes
    return batch


def _same_shape(tensors: Iterable[torch.Tensor]) -> bool:
    tensors = list(tensors)
    shape = tensors[0].shape
    return all(tensor.shape == shape for tensor in tensors)


def _pad_1d_long(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(int(tensor.numel()) for tensor in tensors)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
    for row, tensor in enumerate(tensors):
        padded[row, : tensor.numel()] = tensor.long()
    return padded


def _pad_1d_float(tensors: list[torch.Tensor], pad_value: float) -> torch.Tensor:
    max_len = max(int(tensor.numel()) for tensor in tensors)
    padded = torch.full(
        (len(tensors), max_len),
        pad_value,
        dtype=tensors[0].dtype,
    )
    for row, tensor in enumerate(tensors):
        padded[row, : tensor.numel()] = tensor
    return padded


def _pad_square_2d(tensors: list[torch.Tensor], pad_value: float) -> torch.Tensor:
    max_size = max(int(tensor.shape[0]) for tensor in tensors)
    padded = torch.full(
        (len(tensors), max_size, max_size),
        pad_value,
        dtype=tensors[0].dtype,
    )
    for row, tensor in enumerate(tensors):
        size = int(tensor.shape[0])
        padded[row, :size, :size] = tensor
    return padded


def _is_square_adjacency(tensors: list[torch.Tensor]) -> bool:
    if not tensors:
        return False
    first = tensors[0]
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        return False
    return all(
        tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1] for tensor in tensors
    )


def _stack_num_nodes(items: list[dict[str, Any]]) -> torch.Tensor | None:
    values = [item.get("num_nodes") for item in items]
    if not any(value is not None for value in values):
        return None
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if len(tensors) != len(values):
        raise ValueError("Batch items with num_nodes must all provide it")
    return torch.stack([value.long().reshape(()) for value in tensors])


def _max_graph_nodes(items: list[dict[str, Any]]) -> int | None:
    sizes = [
        int(item["adjacency"].shape[0])
        for item in items
        if isinstance(item.get("adjacency"), torch.Tensor)
    ]
    if not sizes:
        return None
    return max(sizes)


def _remap_graph_stop_actions(
    actions: torch.Tensor,
    num_nodes: torch.Tensor,
    stop_index: int,
) -> torch.Tensor:
    remapped = actions.clone()
    for row in range(actions.size(0)):
        original_stop = int(num_nodes[row].item())
        remapped[row] = torch.where(
            remapped[row] == original_stop,
            torch.tensor(stop_index, dtype=torch.long),
            remapped[row],
        )
    return remapped


def _canonical_problem(raw: Any) -> str:
    if raw == "maximum_clique":
        return "max_clique"
    if raw == "minimum_vertex_cover":
        return "vertex_cover"
    return str(raw)


def _as_string_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _edges_to_adjacency(num_nodes: int, edges: list[list[int]]) -> list[list[float]]:
    adjacency = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        adjacency[u][v] = 1.0
        adjacency[v][u] = 1.0
    return adjacency


def _coordinate_key(node: int, coordinates: list[list[float]]) -> tuple[float, ...]:
    return (*map(float, coordinates[node]), float(node))


def _path_key(
    path: list[int],
    coordinates: list[list[float]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(_coordinate_key(node, coordinates) for node in path)


def _canonical_path(
    raw_path: Iterable[int],
    coordinates: list[list[float]],
) -> list[int]:
    path = [int(node) for node in raw_path]
    reverse = list(reversed(path))
    if _path_key(reverse, coordinates) < _path_key(path, coordinates):
        return reverse
    return path


def _canonical_cycle(
    raw_tour: Iterable[int],
    coordinates: list[list[float]],
) -> list[int]:
    tour = [int(node) for node in raw_tour]
    if not tour:
        return []
    start = min(tour, key=lambda node: _coordinate_key(node, coordinates))

    def rotate_to_start(path: list[int]) -> list[int]:
        offset = path.index(start)
        return [*path[offset:], *path[:offset]]

    forward = rotate_to_start(tour)
    reverse = rotate_to_start(list(reversed(tour)))
    return min((forward, reverse), key=lambda path: _path_key(path, coordinates))


def _cvrp_solution_actions(
    solution: dict[str, Any],
    coordinates: list[list[float]],
) -> list[int]:
    routes = solution.get("routes")
    if not routes:
        tour = solution.get("tour") or []
        canonical = _canonical_path(tour, coordinates)
        return [int(customer) + 1 for customer in canonical]
    canonical_routes = [
        _canonical_path(route, coordinates)
        for route in routes
        if route
    ]
    canonical_routes.sort(key=lambda route: _path_key(route, coordinates))
    actions: list[int] = []
    for route_index, route in enumerate(canonical_routes):
        if route_index > 0:
            actions.append(0)
        actions.extend(int(customer) + 1 for customer in route)
    return actions


def _file_provenance(path: Path) -> dict[str, Any]:
    resolved = resolve_user_path(path)
    stat = resolved.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    with resolved.open("rb") as handle:
        first = handle.readline()
        digest.update(first)
        if stat.st_size > 65_536:
            handle.seek(max(stat.st_size - 65_536, 0))
        tail = handle.read()
        digest.update(tail)
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sample_sha256": digest.hexdigest(),
    }


def _dataset_provenance(
    *,
    path: Path,
    parser: _ProblemRecordParser,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    algorithms: Counter[str] = Counter()
    exactness: Counter[str] = Counter()
    bound_count = 0
    zero_reference_count = 0
    labeled_count = 0
    for raw in records:
        record = parser._normalise_record(raw)
        solutions = record.get("solutions")
        if not isinstance(solutions, dict) or not solutions:
            continue
        if parser.target_algorithm is not None:
            if parser.target_algorithm not in solutions:
                available = ", ".join(str(key) for key in solutions) or "none"
                raise ValueError(
                    f"{parser.path} record does not contain requested "
                    f"target_algorithm={parser.target_algorithm!r}. "
                    f"Available: {available}"
                )
            algorithm = parser.target_algorithm
        elif len(solutions) == 1:
            algorithm = str(next(iter(solutions)))
        else:
            available = ", ".join(str(key) for key in solutions) or "none"
            raise ValueError(
                f"{parser.path} record has multiple solutions; set target_algorithm. "
                f"Available: {available}"
            )
        solution = _as_string_dict(solutions[algorithm])
        if solution is None:
            continue
        labeled_count += 1
        algorithms[str(solution.get("algorithm", algorithm))] += 1
        is_exact = solution.get("is_exact")
        exactness[
            "exact" if is_exact is True else "non_exact" if is_exact is False else "unknown"
        ] += 1
        metadata = solution.get("metadata")
        if isinstance(metadata, dict) and metadata.get("objective_bound") is not None:
            bound_count += 1
        reference = next(
            (
                solution[key]
                for key in ("cost", "value", "size")
                if solution.get(key) is not None
            ),
            None,
        )
        if reference is not None and abs(float(reference)) <= 1e-12:
            zero_reference_count += 1

    exact_count = exactness["exact"]
    non_exact_count = exactness["non_exact"]
    if labeled_count == 0:
        reference_kind = "unlabeled"
    elif exact_count == labeled_count:
        reference_kind = "exact_optimum"
    elif non_exact_count == labeled_count:
        reference_kind = "solver_incumbent"
    else:
        reference_kind = "mixed_or_unknown"
    return {
        **_file_provenance(path),
        "problem": parser.problem,
        "record_count": len(records),
        "labeled_count": labeled_count,
        "target_algorithm_requested": parser.target_algorithm,
        "selected_algorithms": dict(sorted(algorithms.items())),
        "exactness_counts": {
            key: exactness[key] for key in ("exact", "non_exact", "unknown")
        },
        "bound_count": bound_count,
        "zero_reference_count": zero_reference_count,
        "reference_kind": reference_kind,
        "streaming": False,
    }
