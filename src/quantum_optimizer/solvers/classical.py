"""Classical baseline solvers for QUBO problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..qubo import QuboProblem

__all__ = ["SolverResult", "brute_force", "simulated_annealing"]


@dataclass
class SolverResult:
    """Outcome of a QUBO solve."""

    bitstring: np.ndarray
    energy: float
    num_evals: int
    meta: dict[str, Any] = field(default_factory=dict)


def brute_force(qubo: QuboProblem, *, max_vars: int = 16) -> SolverResult:
    """Exhaustively evaluate every bitstring. Returns the global minimum.

    Raises ``ValueError`` when ``qubo.num_variables`` exceeds ``max_vars`` to
    prevent accidental ``2**n`` blowups.
    """
    n = qubo.num_variables
    if n > max_vars:
        raise ValueError(
            f"brute_force requires n <= max_vars (got n={n}, max_vars={max_vars})"
        )
    if n == 0:
        return SolverResult(
            bitstring=np.zeros(0, dtype=np.uint8),
            energy=qubo.offset,
            num_evals=1,
        )
    total = 1 << n
    indices = np.arange(total, dtype=np.uint64)
    bits = np.zeros((total, n), dtype=np.uint8)
    for i in range(n):
        bits[:, i] = (indices >> np.uint64(i)) & np.uint64(1)
    energies = qubo.energies(bits)
    best = int(np.argmin(energies))
    return SolverResult(
        bitstring=bits[best].copy(),
        energy=float(energies[best]),
        num_evals=total,
    )


def simulated_annealing(
    qubo: QuboProblem,
    *,
    num_reads: int = 10,
    num_sweeps: int = 1000,
    t_start: float | None = None,
    t_end: float = 0.01,
    seed: int | None = None,
) -> SolverResult:
    """Run Metropolis simulated annealing with ``num_reads`` independent chains.

    Each chain performs ``num_sweeps`` sweeps; one sweep proposes ``n`` random
    single-bit flips. The temperature decays geometrically from ``t_start`` to
    ``t_end`` across the sweeps. ``t_start=None`` picks a value scaled by the
    largest ``|Q|`` entry.
    """
    n = qubo.num_variables
    if n == 0:
        return SolverResult(
            bitstring=np.zeros(0, dtype=np.uint8),
            energy=qubo.offset,
            num_evals=0,
        )
    if num_reads < 1 or num_sweeps < 1:
        raise ValueError("num_reads and num_sweeps must be >= 1")

    Q = qubo.Q
    M = Q + Q.T - np.diag(np.diag(Q))  # symmetric form, M[i,i] = Q[i,i]
    diag = np.diag(Q).astype(np.float64)

    if t_start is None:
        scale = float(np.abs(Q).max())
        t_start = max(scale * 0.5, t_end * 10.0, 1e-3)
    t_start_f = float(t_start)
    t_end_f = float(t_end)
    if t_start_f <= 0 or t_end_f <= 0 or t_end_f >= t_start_f:
        raise ValueError("require 0 < t_end < t_start")

    rng = np.random.default_rng(seed)
    decay = (t_end_f / t_start_f) ** (1.0 / max(num_sweeps - 1, 1))

    best_x = None
    best_E = np.inf
    histories: list[list[float]] = []

    for _ in range(num_reads):
        x = rng.integers(0, 2, size=n).astype(np.uint8)
        xf = x.astype(np.float64)
        field_vec = M @ xf  # field_vec[i] = (M @ x)[i]
        E = float(xf @ Q @ xf + qubo.offset)
        history = [E]
        T = t_start_f

        for _sweep in range(num_sweeps):
            flips = rng.integers(0, n, size=n)
            randoms = rng.random(size=n)
            for k in range(n):
                i = int(flips[k])
                v = float(x[i])
                qii = float(diag[i])
                # ΔE = (1 - 2v) * (Q[i,i] + neighbor_sum), where
                # neighbor_sum = field_vec[i] - Q[i,i] * v.
                dE = (1.0 - 2.0 * v) * (qii + (field_vec[i] - qii * v))
                if dE <= 0.0 or randoms[k] < np.exp(-dE / T):
                    x[i] = 1 - x[i]
                    delta = 1.0 - 2.0 * v
                    field_vec += M[:, i] * delta
                    E += dE
            history.append(E)
            T *= decay

        histories.append(history)
        if best_E > E:
            best_E = E
            best_x = x.copy()

    assert best_x is not None  # num_reads >= 1
    return SolverResult(
        bitstring=best_x,
        energy=best_E,
        num_evals=num_reads * num_sweeps * n,
        meta={
            "final_energies": [h[-1] for h in histories],
            "histories": histories,
            "t_start": t_start_f,
            "t_end": t_end_f,
        },
    )
