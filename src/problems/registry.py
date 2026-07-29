from src.constants import ProblemName
from src.problems.base import Problem
from src.problems.graph import GraphSubsetProblem, MaxCliqueProblem, VertexCoverProblem
from src.problems.routing import CVRPProblem, TSPProblem
from src.problems.subset import KnapsackProblem, OrienteeringProblem


def make_problem(name: ProblemName) -> Problem:
    if name == "tsp":
        return TSPProblem()
    if name == "cvrp":
        return CVRPProblem()
    if name == "orienteering":
        return OrienteeringProblem()
    if name == "knapsack":
        return KnapsackProblem()
    if name == "mis":
        return GraphSubsetProblem()
    if name == "max_clique":
        return MaxCliqueProblem()
    if name == "vertex_cover":
        return VertexCoverProblem()
    raise ValueError(f"Unsupported problem: {name}")


def get_problem(name: ProblemName) -> Problem:
    return make_problem(name)
