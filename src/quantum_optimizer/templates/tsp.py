"""Traveling Salesman Problem QUBO formulation (Lucas 2014, §6.4).

Variable layout
---------------
For ``n`` cities the QUBO has ``n*n`` variables. ``x[i, t]`` equals 1 if city
``i`` is visited at position ``t`` in the (closed) tour. The flat bitstring
follows city-major order: ``flat_index(i, t) = i * n + t``.

Hamiltonian
-----------
::

    H = sum_{t, i != j} D[i, j] x[i, t] x[j, (t+1) mod n]
      + A * sum_i (1 - sum_t x[i, t])^2          # each city visited exactly once
      + A * sum_t (1 - sum_i x[i, t])^2          # each position has exactly one city

With binary x, ``(1 - sum_k x_k)^2`` expands to
``1 - sum_k x_k + 2 sum_{k<k'} x_k x_{k'}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..qubo import QuboProblem

__all__ = ["TspSolution", "decode_tsp_tour", "tsp_to_qubo"]


@dataclass
class TspSolution:
    """Decoded TSP tour together with feasibility info."""

    tour: list[int]
    is_valid: bool
    total_distance: float


def tsp_to_qubo(
    distance_matrix: np.ndarray,
    penalty: float | None = None,
) -> QuboProblem:
    """Encode a symmetric TSP instance as a :class:`QuboProblem`.

    Parameters
    ----------
    distance_matrix
        ``(n, n)`` non-negative distance matrix. The diagonal is treated as
        zero (self-loops contribute nothing).
    penalty
        Constraint penalty ``A``. When ``None`` (default), uses
        ``max(distance_matrix) * n + 1`` which dominates any feasible tour cost.
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distance_matrix must be square 2D, got shape {D.shape}")
    n = D.shape[0]
    if n < 2:
        raise ValueError("TSP requires at least 2 cities")
    if np.any(D < 0):
        raise ValueError("distance_matrix must be non-negative")

    A = float(D.max()) * n + 1.0 if penalty is None else float(penalty)

    num_vars = n * n
    Q = np.zeros((num_vars, num_vars), dtype=np.float64)

    def idx(i: int, t: int) -> int:
        return i * n + t

    # Each city visited exactly once: A * (1 - sum_t x[i,t])^2 per city i.
    for i in range(n):
        for t in range(n):
            Q[idx(i, t), idx(i, t)] -= A
        for t in range(n):
            for tp in range(t + 1, n):
                Q[idx(i, t), idx(i, tp)] += 2.0 * A

    # Each position has exactly one city: A * (1 - sum_i x[i,t])^2 per position t.
    for t in range(n):
        for i in range(n):
            Q[idx(i, t), idx(i, t)] -= A
        for i in range(n):
            for ip in range(i + 1, n):
                Q[idx(i, t), idx(ip, t)] += 2.0 * A

    # Distance objective: closed tour x[i,t] -> x[j,(t+1) mod n].
    for t in range(n):
        tp = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = idx(i, t), idx(j, tp)
                if a <= b:
                    Q[a, b] += D[i, j]
                else:
                    Q[b, a] += D[i, j]

    offset = 2.0 * A * n
    return QuboProblem(Q, offset=offset)


def decode_tsp_tour(
    bitstring: np.ndarray,
    n_cities: int,
    distance_matrix: np.ndarray,
) -> TspSolution:
    """Decode a TSP bitstring into a tour.

    The tour is rotation-normalized so that the smallest city index appears
    first; this stabilizes equality checks under the ``n``-fold rotational
    symmetry of closed tours.
    """
    bits = np.asarray(bitstring).reshape(n_cities, n_cities)
    tour_raw = [int(np.argmax(bits[:, t])) for t in range(n_cities)]

    col_sums = bits.sum(axis=0)
    row_sums = bits.sum(axis=1)
    is_valid = bool(
        sorted(tour_raw) == list(range(n_cities))
        and np.all(col_sums == 1)
        and np.all(row_sums == 1)
    )

    min_city = min(tour_raw)
    pivot = tour_raw.index(min_city)
    tour = tour_raw[pivot:] + tour_raw[:pivot]

    D = np.asarray(distance_matrix, dtype=np.float64)
    total = sum(float(D[tour[k], tour[(k + 1) % n_cities]]) for k in range(n_cities))

    return TspSolution(tour=tour, is_valid=is_valid, total_distance=total)
