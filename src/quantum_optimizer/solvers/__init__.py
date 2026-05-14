"""QUBO solvers: classical baselines and (later) AWS Braket wrappers."""

from .classical import SolverResult, brute_force, simulated_annealing

__all__ = ["SolverResult", "brute_force", "simulated_annealing"]
