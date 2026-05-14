"""quantum-optimizer-tool: bridge natural-language optimization to AWS Braket."""

from .qubo import QuboProblem
from .solvers import SolverResult, brute_force, simulated_annealing
from .templates import TspSolution, decode_tsp_tour, tsp_to_qubo

__version__ = "0.1.0"

__all__ = [
    "QuboProblem",
    "SolverResult",
    "TspSolution",
    "__version__",
    "brute_force",
    "decode_tsp_tour",
    "simulated_annealing",
    "tsp_to_qubo",
]
