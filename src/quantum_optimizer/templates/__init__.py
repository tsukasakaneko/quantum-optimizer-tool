"""Domain QUBO templates (TSP, knapsack, scheduling, ...)."""

from .tsp import TspSolution, decode_tsp_tour, tsp_to_qubo

__all__ = ["TspSolution", "decode_tsp_tour", "tsp_to_qubo"]
