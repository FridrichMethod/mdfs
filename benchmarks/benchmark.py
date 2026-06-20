"""Throughput and size-scaling benchmark for mdfs.

Covers the implemented MD matrix: NVE (velocity-Verlet) and NVT (Langevin BAOAB),
in vacuum (free space, plain nonbonded) and periodic (orthorhombic MIC, DSF
electrostatics + LJ cutoff). Larger systems are built by replicating the bundled
poly_A peptide into spatially-separated copies.

Reports steps/s, ns/day, and ms/step for each configuration. Device (CPU/GPU) is
whatever JAX selects; force it with e.g. ``JAX_PLATFORMS=cpu``.

Usage:
    python benchmarks/benchmark.py [--x64] [--replicas 1 5 10 20]
"""

from __future__ import annotations

import argparse
import time

import jax

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--x64", action="store_true", help="use float64 (default: float32)")
parser.add_argument("--replicas", type=int, nargs="+", default=[1, 5, 10, 20])
parser.add_argument("--steps", type=int, default=2000, help="timed steps for the base size")
args = parser.parse_args()

jax.config.update("jax_enable_x64", args.x64)

import jax.numpy as jnp
import numpy as np

import mdfs
from mdfs.energy import DSFParams
from mdfs.params import SystemParams
from mdfs.paths import REPO_ROOT

DT_PS = 0.0005  # 0.5 fs timestep


def replicate(sp: SystemParams, m: int, spacing: float = 4.0) -> SystemParams:
    """Return ``m`` spatially-offset, non-interacting copies of ``sp`` as one system."""
    n = sp.n_atoms

    def offset_rows(a: np.ndarray) -> np.ndarray:
        return np.concatenate([a + k * n for k in range(m)])

    def tile(a: np.ndarray) -> np.ndarray:
        return np.tile(a, m)

    positions = np.concatenate([sp.positions + np.array([spacing * k, 0.0, 0.0]) for k in range(m)])
    return SystemParams(
        positions=positions,
        masses=tile(sp.masses),
        box_lengths=None,
        bonds=offset_rows(sp.bonds),
        bond_k=tile(sp.bond_k),
        bond_r0=tile(sp.bond_r0),
        angles=offset_rows(sp.angles),
        angle_k=tile(sp.angle_k),
        angle_theta0=tile(sp.angle_theta0),
        torsions=offset_rows(sp.torsions),
        torsion_periodicity=tile(sp.torsion_periodicity),
        torsion_phase=tile(sp.torsion_phase),
        torsion_k=tile(sp.torsion_k),
        charges=tile(sp.charges),
        sigma=tile(sp.sigma),
        epsilon=tile(sp.epsilon),
        exclude_pairs=offset_rows(sp.exclude_pairs),
        exc_qq=tile(sp.exc_qq),
        exc_sigma=tile(sp.exc_sigma),
        exc_eps=tile(sp.exc_eps),
        atom_names=sp.atom_names * m,
        res_names=sp.res_names * m,
    )


def throughput(step, state, n_steps, key=None) -> float:
    """Steps per second for ``step`` over ``n_steps`` (one warmup step compiles it)."""
    warm = step(state) if key is None else step(state, key)[0]
    warm.R.block_until_ready()
    t0 = time.perf_counter()
    st, kk = state, key
    for _ in range(n_steps):
        if kk is None:
            st = step(st)
        else:
            st, kk = step(st, kk)
    st.R.block_until_ready()
    return n_steps / (time.perf_counter() - t0)


def fmt(label: str, n_atoms: int, n_pairs: int, sps: float) -> str:
    ns_day = sps * DT_PS * 86400.0 / 1000.0
    return (
        f"{label:<16} {n_atoms:>7} {n_pairs:>12,} {sps:>10,.0f} {ns_day:>9,.1f} {1e3 / sps:>9.2f}"
    )


def main() -> None:
    print(
        f"device={jax.devices()[0].platform.upper()}  precision={'float64' if args.x64 else 'float32'}  dt={DT_PS * 1000} fs\n"
    )
    print(f"{'config':<16} {'atoms':>7} {'pairs':>12} {'steps/s':>10} {'ns/day':>9} {'ms/step':>9}")
    print("-" * 70)

    base = mdfs.system_params_from_pdb(REPO_ROOT / "assets" / "poly_A.pdb")[0]
    for m in args.replicas:
        sp = replicate(base, m) if m > 1 else base
        n = sp.n_atoms
        n_steps = max(200, args.steps // m)
        bonded = mdfs.to_bonded_set(sp)
        nb = mdfs.to_nonbonded_set(sp, mdfs.all_pairs(n))
        n_pairs = n * (n - 1) // 2
        R0 = jnp.asarray(sp.positions)
        V0 = mdfs.maxwell_boltzmann_velocities(
            jax.random.PRNGKey(0), jnp.asarray(sp.masses), 300.0, n
        )
        mass = jnp.asarray(sp.masses)

        nve_state, nve_step = mdfs.simulate_nve(R0, V0, None, bonded, nb, dt=DT_PS, mass=mass)
        print(fmt(f"NVE vacuum x{m}", n, n_pairs, throughput(nve_step, nve_state, n_steps)))

        nvt_state, nvt_step = mdfs.simulate_langevin(
            R0, V0, None, bonded, nb, dt=DT_PS, mass=mass, gamma=10.0, temperature=300.0
        )
        sps = throughput(nvt_step, nvt_state, n_steps, key=jax.random.PRNGKey(1))
        print(fmt(f"NVT vacuum x{m}", n, n_pairs, sps))

    # Periodic (PBC + DSF + LJ cutoff) at the base size, to exercise that path.
    sp = base
    n = sp.n_atoms
    extent = np.ptp(sp.positions, axis=0).max()
    box = jnp.array([extent + 4.0] * 3)  # ensure r_cut <= min(box)/2
    r_cut = float(extent + 4.0) / 2.0 - 0.1
    nb_pbc = mdfs.to_nonbonded_set(
        sp, mdfs.all_pairs(n), r_cut_lj=r_cut, dsf=DSFParams(alpha=2.0, r_cut=r_cut)
    )
    bonded = mdfs.to_bonded_set(sp)
    R0 = jnp.asarray(sp.positions)
    mass = jnp.asarray(sp.masses)
    V0 = mdfs.maxwell_boltzmann_velocities(jax.random.PRNGKey(0), mass, 300.0, n)
    state, step = mdfs.simulate_langevin(
        R0, V0, box, bonded, nb_pbc, dt=DT_PS, mass=mass, gamma=10.0, temperature=300.0
    )
    sps = throughput(step, state, 2000, key=jax.random.PRNGKey(2))
    print(fmt("NVT periodic x1", n, n * (n - 1) // 2, sps))


if __name__ == "__main__":
    main()
