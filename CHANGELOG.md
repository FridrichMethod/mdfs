# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Performance

- **Dense (N, N) nonbonded path, now the default.** Forces come from `jax.grad`
  of the energy; with a pair list the gradient must scatter-add O(N^2) pair
  contributions onto atoms, which is very slow on GPU. The dense formulation
  builds pairwise terms by broadcasting and reduces along an axis (no scatter),
  giving 3.4x–~2,600x faster dynamics (e.g. 2,060 atoms: ~0.1 -> 265 ns/day) and
  making the GPU the fastest backend. Pass an explicit pair list / neighbor list
  for the O(N) pair-list path on larger systems. See `benchmarks/`.

### Fixed

- **Force-field loading now works.** `params.py` was rewritten to read
  fully-resolved parameters from `ForceField.createSystem` via OpenMM's public
  API instead of reverse-engineering private internals (which were incompatible
  with OpenMM 8.x). This unblocks every downstream step.
- **Physical units.** The Langevin thermostat now uses the correct Boltzmann
  constant and electrostatics use the correct Coulomb constant
  (`mdfs.constants`); the previous reduced-unit defaults (`kB = 1`, `k_e = 1`)
  produced unphysical temperatures and forces.
- **Per-atom masses** are taken from the force field instead of a single scalar.
- **NaN-safe angle/dihedral gradients** via `atan2` with softened norms.
- **`run()` drives NVE correctly** (NVE vs Langevin dispatch fixed).
- **Packaging:** subpackages now ship correctly (`ffxml`, `utils` packages).

### Added

- Improper torsions and exception-based 1-4 interactions (matching OpenMM).
- Energy minimization (`steepest_descent`, BFGS `minimize_energy`).
- Trajectory and energy reporters (`TrajectoryRecorder`, `EnergyLogger`).
- Maxwell-Boltzmann velocity sampling and kinetic-energy/temperature helpers.
- Physics test suite validated against OpenMM, plus a poly_A end-to-end regression.
- House-style tooling: Makefile, strict ruff/mypy/bandit, pre-commit, CI, docs.

### Removed

- AlphaFold-derived `common/` modules (license-incompatible, unused) and the
  superseded bond-graph `topology.py`.
- `uv` as a runtime dependency (it is a dev/installer tool).
