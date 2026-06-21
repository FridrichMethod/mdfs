# CLAUDE.md

Project-specific guidance for working in this repository.

## What this repo is

`mdfs` ("molecular dynamics from scratch") is a small, JAX-based **differentiable**
MD engine. Energy terms are written explicitly in `src/mdfs/energy.py`; forces are
obtained by `jax.grad`. OpenMM is used only to assign force-field parameters
(via `ForceField.createSystem`, read through its public API) and to add hydrogens;
MDTraj is used for trajectory output. The reference test system is a 10-residue
polyalanine, `assets/poly_A.pdb`.

## Environment

- Use the local **`.venv`** (Python 3.12), created and managed with `uv`:
  ```bash
  make venv          # uv venv .venv + editable install + jax[cuda12]
  # or manually:
  uv venv .venv --python 3.12
  uv pip install -p .venv/bin/python -e ".[dev,mypy]" "jax[cuda12]"
  ```
- The GPU path is **JAX** (`jax[cuda12]`); verify with
  `.venv/bin/python -c "import jax; print(jax.devices())"`. OpenMM runs on CPU
  (it only parses force fields / adds hydrogens), so no `openmm[cuda12]` is needed.
- Use `uv pip`, not bare `pip`. Always run tools through `.venv` (the Makefile does).

## Units (repo-wide convention)

OpenMM/Amber units everywhere: length **nm**, time **ps**, mass **amu**, energy
**kJ/mol**, charge **e**. Constants live in `src/mdfs/constants.py`
(`BOLTZMANN_KJ_PER_MOL_K`, `ONE_4PI_EPS0`). Never reintroduce reduced units
(`kB = 1`, `k_e = 1`).

## Layout

```
src/mdfs/
  constants.py    physical constants (kB, Coulomb)
  types.py        shared aliases (StrPath, DisplacementFn, ShiftFn)
  paths.py        bundled FFXML path (MDFS_FFXML override)
  params.py       OpenMM createSystem -> SystemParams -> Bonded/Nonbonded sets
  energy.py       bonded + nonbonded energy terms (differentiable)
  space.py        free / periodic (MIC) displacement & shift
  partition.py    all_pairs (static) + Verlet neighbor_list
  integrators.py  velocity-Verlet (NVE), BAOAB Langevin (NVT), KE/temperature
  minimize.py     steepest descent + BFGS minimizers
  simulate.py     high-level drivers (simulate_nve/langevin) + run loop
  io.py           TrajectoryRecorder, EnergyLogger reporters
  utils/logging.py  configure_logging
  ffxml/          bundled Amber ff19SB (package data)
tests/            mirrors src/ (+ regressions/ for the poly_A e2e)
```

## Conventions

- Absolute imports only; `pathlib` over `os.path`; `logging.getLogger(__name__)`
  over `print` (CLI/notebooks call `configure_logging` once).
- Type hints + Google-style docstrings on public functions; prefer immutable
  dataclasses / NamedTuples for aggregates.
- Energy functions stay pure (`E(R) -> scalar`) so `jax.grad`/`jit` apply cleanly.
- New parameters come from OpenMM's resolved `System`, never by re-parsing FFXML
  internals.
- **Nonbonded forces use the dense (N, N) path by default** (`NonbondedSet.pairs is None`): broadcasting + axis-reduction so the autodiff gradient reduces rather
  than scatters (fast on GPU, O(N^2) memory). Only switch to the pair-list path
  (pass `pairs=`) for large/dilute systems; never reintroduce a scatter-based
  default. See `benchmarks/`.
- For a larger timestep: `repartition_hydrogen_masses` (HMR, dt = 2 fs) and/or
  LINCS bond constraints (`setup_hbond_constraints` + `constraints=` on the
  integrators; dt = 2 fs, or 4 fs with HMR). Constraints live in the integrator
  (RATTLE / constrained BAOAB), not the energy, so autodiff forces are unaffected;
  remember to use the reduced bonded set and `constrained_dof` for temperature.
  `lax.scan` over steps was measured and *not* adopted (no speedup on the dense
  path); don't re-add it.

## Correctness bar

The core is validated against OpenMM per force group on poly_A (bond/angle/torsion
to ~1e-15, nonbonded ~1e-8, forces ~1e-5) -- minus CMAP, which mdfs intentionally
omits (see README "Limitations"). When changing energy/params, keep
`tests/test_params_vs_openmm.py` green.

## Commit / push gate

Before committing: `make format && make lint && make mypy && make test`
(and `make bandit`). Use conventional-commit messages. Never use `--no-verify`.
Slow/e2e tests run via `make slow-test`.
