"""Tests for QuboProblem."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_optimizer import QuboProblem


def test_energy_known_qubo() -> None:
    # Energy = x0 + x1 - 2 * x0 * x1
    Q = np.array([[1.0, -2.0], [0.0, 1.0]])
    qubo = QuboProblem(Q)
    assert qubo.num_variables == 2
    assert qubo.energy([0, 0]) == 0.0
    assert qubo.energy([1, 0]) == 1.0
    assert qubo.energy([0, 1]) == 1.0
    assert qubo.energy([1, 1]) == 0.0


def test_offset_added_to_energy() -> None:
    qubo = QuboProblem(np.array([[1.0]]), offset=2.5)
    assert qubo.energy([0]) == 2.5
    assert qubo.energy([1]) == 3.5


def test_symmetric_input_is_normalized() -> None:
    sym = np.array([[1.0, 0.5], [0.5, 2.0]])
    qubo = QuboProblem(sym)
    # Upper-triangular form should have Q[0,1] = 1.0 (= 0.5 + 0.5)
    np.testing.assert_allclose(qubo.Q, [[1.0, 1.0], [0.0, 2.0]])
    # Energy must match the dense symmetric computation
    for x in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xv = np.array(x, dtype=float)
        assert qubo.energy(x) == pytest.approx(xv @ sym @ xv)


def test_from_dict_sums_symmetric_entries() -> None:
    qubo = QuboProblem.from_dict({(0, 1): 1.0, (1, 0): 2.0, (0, 0): 0.5}, 2)
    np.testing.assert_allclose(qubo.Q, [[0.5, 3.0], [0.0, 0.0]])


def test_from_dict_index_validation() -> None:
    with pytest.raises(ValueError):
        QuboProblem.from_dict({(2, 0): 1.0}, 2)


def test_energies_batch_matches_loop() -> None:
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(4, 4))
    qubo = QuboProblem(Q, offset=0.3)
    bits = (rng.integers(0, 2, size=(7, 4))).astype(np.uint8)
    batched = qubo.energies(bits)
    one_by_one = np.array([qubo.energy(b) for b in bits])
    np.testing.assert_allclose(batched, one_by_one)


def test_energy_shape_validation() -> None:
    qubo = QuboProblem(np.eye(3))
    with pytest.raises(ValueError):
        qubo.energy([1, 0])


def test_to_ising_matches_qubo_under_substitution() -> None:
    rng = np.random.default_rng(1)
    Q = np.triu(rng.normal(size=(3, 3)))
    qubo = QuboProblem(Q, offset=1.25)
    h, J, const = qubo.to_ising()
    # For each binary x, compare QUBO energy to Ising energy with z = 1 - 2x.
    for k in range(8):
        x = np.array([(k >> b) & 1 for b in range(3)], dtype=float)
        z = 1.0 - 2.0 * x
        ising = float(h @ z + z @ J @ z + const)
        np.testing.assert_allclose(ising, qubo.energy(x))


def test_repr_and_equality() -> None:
    q1 = QuboProblem(np.eye(2), offset=1.0)
    q2 = QuboProblem(np.eye(2), offset=1.0)
    q3 = QuboProblem(np.eye(2), offset=2.0)
    assert q1 == q2
    assert q1 != q3
    assert "num_variables=2" in repr(q1)


def test_non_square_raises() -> None:
    with pytest.raises(ValueError):
        QuboProblem(np.zeros((2, 3)))
