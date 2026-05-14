# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-14

### Added

- `QuboProblem`: upper-triangular dense QUBO container with `energy`,
  vectorized `energies`, `from_dict`, and `to_ising` helpers.
- `tsp_to_qubo` / `decode_tsp_tour` / `TspSolution`: Traveling Salesman
  encoder and rotation-normalized decoder based on Lucas (2014) §6.4.
- Classical solvers: `brute_force` (≤16 vars, vectorized exhaustive search)
  and `simulated_annealing` (pure numpy Metropolis with incremental ΔE,
  geometric cooling, deterministic seeding, multi-read).
- `notebooks/02_tsp_qubo.ipynb`: end-to-end TSP demo with brute force,
  SA energy traces, tour visualization, and penalty sensitivity sweep.
- Unit tests for QUBO algebra, TSP encoding/decoding, and both solvers.

## [0.0.1] - 2026-05-14

### Added

- Initial repository skeleton (Phase 1).
