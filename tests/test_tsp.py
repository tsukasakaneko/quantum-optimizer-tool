"""Tests for the TSP -> QUBO encoder/decoder."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_optimizer import (
    QuboProblem,
    brute_force,
    decode_tsp_tour,
    tsp_to_qubo,
)


def _equilateral(n: int, edge: float = 2.0) -> np.ndarray:
    D = np.full((n, n), edge, dtype=float)
    np.fill_diagonal(D, 0.0)
    return D


def _square_4cities(side: float = 1.0) -> np.ndarray:
    coords = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=float)
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)


def test_tsp_qubo_dimensions() -> None:
    qubo = tsp_to_qubo(_equilateral(3))
    assert isinstance(qubo, QuboProblem)
    assert qubo.num_variables == 9


def test_brute_force_recovers_equilateral_triangle_tour() -> None:
    D = _equilateral(3, edge=2.0)
    qubo = tsp_to_qubo(D)
    res = brute_force(qubo)
    sol = decode_tsp_tour(res.bitstring, 3, D)
    assert sol.is_valid
    assert sol.total_distance == pytest.approx(3 * 2.0)
    # Rotation-normalized: smallest city index first.
    assert sol.tour[0] == 0
    assert sorted(sol.tour) == [0, 1, 2]


def test_brute_force_recovers_square_4city_tour() -> None:
    side = 1.0
    D = _square_4cities(side)
    qubo = tsp_to_qubo(D)
    res = brute_force(qubo)
    sol = decode_tsp_tour(res.bitstring, 4, D)
    assert sol.is_valid
    assert sol.total_distance == pytest.approx(4 * side)
    # Two valid optimal tours under rotation normalization (city 0 first).
    assert sol.tour in ([0, 1, 2, 3], [0, 3, 2, 1])


def test_zero_penalty_breaks_constraints() -> None:
    # With A=0 the encoder loses its feasibility pressure: the trivial
    # all-zeros bitstring beats every valid tour.
    D = _square_4cities()
    qubo = tsp_to_qubo(D, penalty=0.0)
    res = brute_force(qubo)
    sol = decode_tsp_tour(res.bitstring, 4, D)
    assert not sol.is_valid


def test_decode_handles_rotated_solutions() -> None:
    # Manually construct a valid tour bitstring [2, 3, 0, 1] and verify it
    # gets normalized to start at city 0.
    n = 4
    tour = [2, 3, 0, 1]
    bits = np.zeros((n, n), dtype=np.uint8)
    for t, city in enumerate(tour):
        bits[city, t] = 1
    D = _square_4cities()
    sol = decode_tsp_tour(bits.ravel(), n, D)
    assert sol.is_valid
    assert sol.tour[0] == 0
    assert sol.tour == [0, 1, 2, 3]


def test_tsp_qubo_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        tsp_to_qubo(np.zeros((3, 4)))


def test_tsp_qubo_rejects_negative_distances() -> None:
    with pytest.raises(ValueError):
        tsp_to_qubo(np.array([[0.0, -1.0], [-1.0, 0.0]]))


def test_tsp_qubo_rejects_single_city() -> None:
    with pytest.raises(ValueError):
        tsp_to_qubo(np.zeros((1, 1)))
