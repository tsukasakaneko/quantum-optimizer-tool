"""Tests for classical QUBO solvers."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_optimizer import (
    QuboProblem,
    brute_force,
    decode_tsp_tour,
    simulated_annealing,
    tsp_to_qubo,
)


def test_brute_force_finds_known_minimum() -> None:
    # Energy = -x0 - x1 + 3 * x0 * x1; optimum at (1, 0) or (0, 1) with E = -1.
    Q = np.array([[-1.0, 3.0], [0.0, -1.0]])
    qubo = QuboProblem(Q)
    res = brute_force(qubo)
    assert res.energy == pytest.approx(-1.0)
    assert tuple(res.bitstring) in ((1, 0), (0, 1))
    assert res.num_evals == 4


def test_brute_force_respects_max_vars() -> None:
    qubo = QuboProblem(np.zeros((5, 5)))
    with pytest.raises(ValueError):
        brute_force(qubo, max_vars=4)


def test_simulated_annealing_matches_brute_force_on_small_qubo() -> None:
    rng = np.random.default_rng(42)
    Q = np.triu(rng.normal(size=(6, 6)))
    qubo = QuboProblem(Q, offset=0.7)
    bf = brute_force(qubo)
    sa = simulated_annealing(qubo, num_reads=10, num_sweeps=500, seed=0)
    assert sa.energy == pytest.approx(bf.energy)


def test_simulated_annealing_solves_4city_tsp() -> None:
    side = 1.0
    coords = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=float)
    D = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    qubo = tsp_to_qubo(D)
    res = simulated_annealing(qubo, num_reads=20, num_sweeps=500, seed=0)
    sol = decode_tsp_tour(res.bitstring, 4, D)
    assert sol.is_valid
    assert sol.total_distance == pytest.approx(4 * side)


def test_simulated_annealing_meta_fields() -> None:
    qubo = QuboProblem(np.array([[-1.0, 1.0], [0.0, -1.0]]))
    res = simulated_annealing(qubo, num_reads=3, num_sweeps=20, seed=1)
    assert res.num_evals == 3 * 20 * 2
    assert len(res.meta["final_energies"]) == 3
    assert len(res.meta["histories"]) == 3
    assert len(res.meta["histories"][0]) == 21  # initial + per-sweep
    assert res.meta["t_start"] > res.meta["t_end"] > 0


def test_simulated_annealing_validates_inputs() -> None:
    qubo = QuboProblem(np.eye(2))
    with pytest.raises(ValueError):
        simulated_annealing(qubo, num_reads=0)
    with pytest.raises(ValueError):
        simulated_annealing(qubo, t_start=0.1, t_end=1.0)


def test_simulated_annealing_is_deterministic_with_seed() -> None:
    rng = np.random.default_rng(7)
    Q = np.triu(rng.normal(size=(5, 5)))
    qubo = QuboProblem(Q)
    a = simulated_annealing(qubo, num_reads=4, num_sweeps=200, seed=99)
    b = simulated_annealing(qubo, num_reads=4, num_sweeps=200, seed=99)
    assert a.energy == b.energy
    np.testing.assert_array_equal(a.bitstring, b.bitstring)
