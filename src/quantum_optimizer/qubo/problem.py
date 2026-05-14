"""QUBO problem representation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["QuboProblem"]


class QuboProblem:
    """A Quadratic Unconstrained Binary Optimization problem.

    Energy is ``x @ Q @ x + offset`` where ``x`` is a binary vector of length
    ``num_variables`` and ``Q`` is stored as an upper-triangular ``(n, n)``
    ``float64`` matrix: ``Q[i, i]`` are linear coefficients and ``Q[i, j]`` for
    ``i < j`` are quadratic coefficients.
    """

    __slots__ = ("Q", "offset")

    def __init__(self, Q: np.ndarray, offset: float = 0.0) -> None:
        arr = np.asarray(Q, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Q must be a square 2D array, got shape {arr.shape}")
        # Normalize to upper-triangular: fold strict-lower into strict-upper.
        upper = np.triu(arr).copy()
        upper += np.tril(arr, k=-1).T
        self.Q: np.ndarray = np.triu(upper)
        self.offset: float = float(offset)

    @property
    def num_variables(self) -> int:
        return int(self.Q.shape[0])

    def energy(self, x: np.ndarray | Sequence[int]) -> float:
        """Return ``x @ Q @ x + offset`` for binary vector ``x``."""
        xv = np.asarray(x, dtype=np.float64)
        if xv.shape != (self.num_variables,):
            raise ValueError(
                f"x must have shape ({self.num_variables},), got {xv.shape}"
            )
        return float(xv @ self.Q @ xv + self.offset)

    def energies(self, X: np.ndarray) -> np.ndarray:
        """Vectorized energy for a batch of bitstrings ``X`` of shape ``(K, n)``."""
        Xv = np.asarray(X, dtype=np.float64)
        if Xv.ndim != 2 or Xv.shape[1] != self.num_variables:
            raise ValueError(
                f"X must have shape (K, {self.num_variables}), got {Xv.shape}"
            )
        # einsum keeps it readable and avoids allocating an intermediate (K, n).
        return np.einsum("ki,ij,kj->k", Xv, self.Q, Xv) + self.offset

    @classmethod
    def from_dict(
        cls,
        q: dict[tuple[int, int], float],
        num_variables: int,
        offset: float = 0.0,
    ) -> QuboProblem:
        """Build from a sparse coefficient dict, summing symmetric entries."""
        Q = np.zeros((num_variables, num_variables), dtype=np.float64)
        for (i, j), v in q.items():
            if not (0 <= i < num_variables and 0 <= j < num_variables):
                raise ValueError(f"index ({i},{j}) out of range for n={num_variables}")
            a, b = (i, j) if i <= j else (j, i)
            Q[a, b] += float(v)
        return cls(Q, offset=offset)

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Convert to Ising form: returns ``(h, J, const)`` such that
        ``energy(x) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + const``
        with the substitution ``x_i = (1 - z_i) / 2``, ``z_i in {-1, +1}``.
        ``J`` is returned as an upper-triangular matrix.
        """
        n = self.num_variables
        Q = self.Q
        # x_i x_j = (1 - z_i)(1 - z_j) / 4 = (1 - z_i - z_j + z_i z_j) / 4
        # x_i      = (1 - z_i) / 2
        h = np.zeros(n, dtype=np.float64)
        J = np.zeros((n, n), dtype=np.float64)
        const = self.offset
        for i in range(n):
            const += Q[i, i] * 0.5
            h[i] += -Q[i, i] * 0.5
        for i in range(n):
            for j in range(i + 1, n):
                qij = Q[i, j]
                if qij == 0.0:
                    continue
                const += qij * 0.25
                h[i] += -qij * 0.25
                h[j] += -qij * 0.25
                J[i, j] += qij * 0.25
        return h, J, float(const)

    def __repr__(self) -> str:
        return (
            f"QuboProblem(num_variables={self.num_variables}, offset={self.offset:.6g})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, QuboProblem):
            return NotImplemented
        return (
            self.Q.shape == other.Q.shape
            and np.array_equal(self.Q, other.Q)
            and self.offset == other.offset
        )
