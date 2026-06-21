# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-06-21

First tagged release: a runnable, validated, differentiable MD engine. `mdfs`
simulates from a PDB end to end (add hydrogens -> assign Amber ff19SB -> minimize
-> NVE/NVT, vacuum or periodic) with forces from `jax.grad`, validated against
OpenMM to machine precision (minus CMAP). Small/medium-system scope; see the
README "Limitations".

### Performance

- **LINCS bond constraints** (`mdfs.constraints`: `setup_hbond_constraints`,
  `apply_position_constraint`/`apply_velocity_constraint`) with RATTLE
  velocity-Verlet and constrained BAOAB. Constraining X-H bonds gives a robust
  2 fs timestep, and 4 fs combined with HMR (~2,200 ns/day on poly_A). Non-iterative
  4th-order matrix expansion + one length correction (JIT-friendly); temperature
  uses `constrained_dof` = 3N - K. See `examples/nvt_constraints.py`.
- **Hydrogen mass repartitioning** (`repartition_hydrogen_masses`) moves mass onto
  hydrogens so dt = 2 fs is stable without constraints, ~4x more ns/day (poly_A:
  ~400 -> ~1,400 ns/day). Mass-conserving; see `examples/nvt_hmr.py`.
- **Dense (N, N) nonbonded path, now the default.** Forces come from `jax.grad`
  of the energy; with a pair list the gradient must scatter-add O(N^2) pair
  contributions onto atoms, which is very slow on GPU. The dense formulation
  builds pairwise terms by broadcasting and reduces along an axis (no scatter),
  giving 3.4x–~2,600x faster dynamics (e.g. 2,060 atoms: ~0.1 -> 265 ns/day) and
  making the GPU the fastest backend. Pass an explicit pair list / neighbor list
  for the O(N) pair-list path on larger systems. See `benchmarks/`.

### Fixed

- **HMR + LINCS no longer silently selects zero constraints.** Hydrogen detection
  keyed off the integration masses, so HMR-inflated H (3.024 amu) exceeded the
  threshold and no bonds were constrained. `setup_hbond_constraints` now takes
  `selection_masses` (pass the pre-HMR masses); `select_hbond_constraints` warns
  when it finds no hydrogen bonds.
- **Dihedral force is finite at exact collinearity** (unnormalized Blondel-Karplus
  with an `atan2` degeneracy guard; previously `atan2(0,0)` produced NaN forces).
- **Neighbor-list rebuild trigger uses the minimum image**, so a periodic box-face
  crossing no longer forces a spurious O(N^2) rebuild every step.
- **Constraint-aware reporting/sampling.** `EnergyLogger` accepts `constraints` to
  use `3N-K` dof; `maxwell_boltzmann_velocities` accepts `constraints`/`positions`
  to project the initial velocities; LJ cutoff is energy-shifted by default.
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
