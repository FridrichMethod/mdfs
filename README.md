# mdfs

**Molecular dynamics from scratch** -- a small, JAX-based, *differentiable* MD
engine. Energy terms are written explicitly and forces come from `jax.grad`.
OpenMM assigns force-field parameters (Amber ff19SB) and adds hydrogens; MDTraj
handles trajectory output.

[![CI](https://github.com/FridrichMethod/mdfs/actions/workflows/ci.yml/badge.svg)](https://github.com/FridrichMethod/mdfs/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)

> **Author:** [Zhaoyang Li](mailto:zhaoyangli@stanford.edu)

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Layout](#layout)
- [Units](#units)
- [Limitations](#limitations)
- [See also](#see-also)

## Installation

`mdfs` uses a `uv`-managed virtual environment (`.venv`, Python 3.12). The GPU
path is JAX (`jax[cuda12]`); OpenMM runs on CPU.

```bash
# One-shot (creates .venv, installs the package + dev tools + CUDA JAX):
make venv

# Or manually:
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -e ".[dev,mypy]" "jax[cuda12]"
```

Verify the GPU is visible:

```bash
.venv/bin/python -c "import jax; print(jax.devices())"
```

For a CPU-only setup, drop `"jax[cuda12]"`.

## Quickstart

Run a short vacuum simulation of the bundled polyalanine and write a trajectory
and energy log:

```python
import jax
import mdfs

# PDB -> add hydrogens -> resolve ff19SB parameters via OpenMM
sp, openmm_top = mdfs.system_params_from_pdb("assets/poly_A.pdb")

bonded = mdfs.to_bonded_set(sp)
nonbonded = mdfs.to_nonbonded_set(sp)  # dense (N, N) path (fast forces; default)
energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nonbonded)

# Relax the structure before dynamics
R0 = mdfs.minimize_energy(energy_fn, jax.numpy.asarray(sp.positions)).positions

# NVT (Langevin) dynamics at 300 K
key = jax.random.PRNGKey(0)
V0 = mdfs.maxwell_boltzmann_velocities(key, sp.masses, 300.0, sp.n_atoms)
state, step = mdfs.simulate_langevin(
    R0, V0, box=None, bonded=bonded, nonbonded=nonbonded,
    dt=0.0005, mass=sp.masses, gamma=2.0, temperature=300.0,
)

traj = mdfs.TrajectoryRecorder(mdfs.mdtraj_topology_from_openmm(openmm_top))
log = mdfs.EnergyLogger(energy_fn, sp.masses)
mdfs.run(step, state, 2000, key=jax.random.PRNGKey(1),
         report_interval=50, callback=mdfs.combine_callbacks(traj, log))

traj.save("poly_A.xtc")
log.save_csv("poly_A_energy.csv")
```

A complete walkthrough is in `notebooks/md_simulation.ipynb`.

## Examples and benchmarks

[`examples/`](examples) has runnable scripts covering each configuration — NVE/NVT,
vacuum/periodic, minimization (see [`examples/README.md`](examples/README.md)).
[`benchmarks/`](benchmarks) measures throughput and size scaling
(see [`benchmarks/README.md`](benchmarks/README.md)).

## Layout

```
src/mdfs/      package (see CLAUDE.md for a module-by-module map)
assets/        poly_A.pdb test system
notebooks/     end-to-end demo
tests/         mirrors src/ (+ regressions/ for the poly_A e2e)
```

## Units

OpenMM/Amber units throughout: length **nm**, time **ps**, mass **amu**, energy
**kJ/mol**, charge **e**. See `src/mdfs/constants.py`.

## Limitations

- **No CMAP.** ff19SB's backbone CMAP correction is intentionally not implemented;
  all other bonded (incl. impropers) and nonbonded terms are. mdfs energies/forces
  match an OpenMM `NoCutoff` system to machine precision *minus the CMAP term*.
- **Electrostatics:** plain Coulomb in vacuum (no cutoff); a damped-shifted-force
  (DSF) option is provided for cutoff/periodic use. No Ewald/PME.
- **Timestep.** Unconstrained, use a small timestep (`dt = 0.0005 ps`). For larger
  steps: hydrogen mass repartitioning (`mdfs.repartition_hydrogen_masses`) runs at
  `dt = 0.002 ps`, or LINCS H-bond constraints (`mdfs.setup_hbond_constraints`,
  RATTLE/constrained-BAOAB) give a robust `dt = 0.002 ps` and, with HMR,
  `dt = 0.004 ps` (~8× faster). See `examples/nvt_hmr.py` and
  `examples/nvt_constraints.py`.
- **Small/medium-system scope.** Forces come from `jax.grad` of the energy. The
  default **dense (N, N)** nonbonded path makes that gradient a fast GPU reduction
  (e.g. ~420 ns/day at 100 atoms, ~265 at 2,000, ~75 at 5,000) but uses O(N^2)
  memory — practical up to a few thousand atoms; see [`benchmarks/`](benchmarks).
  For larger or solvated systems pass an O(N) neighbor list
  (`mdfs.partition.neighbor_list`) or use OpenMM/GROMACS.

## See also

- [JAX, M.D.](https://github.com/google/jax-md) -- differentiable, hardware-accelerated MD
- [OpenMM](https://github.com/openmm/openmm) -- high-performance MD toolkit
- [MDTraj](https://github.com/mdtraj/mdtraj) -- MD trajectory analysis
