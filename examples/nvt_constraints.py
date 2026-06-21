"""NVT MD of poly_A with LINCS H-bond constraints, dt = 2 fs.

Constraining the X-H bonds removes the fastest vibrations so a 2 fs timestep is
stable (the standard constrained-MD timestep; ~4x faster than unconstrained 0.5 fs).
Combine with hydrogen mass repartitioning for dt = 4 fs (better suited to NVT).
Demonstrates: LINCS constraints, constrained BAOAB, reduced dof for temperature.

Run:
    python examples/nvt_constraints.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import mdfs
from mdfs.paths import REPO_ROOT


def main() -> None:
    sp, _ = mdfs.system_params_from_pdb(REPO_ROOT / "assets" / "poly_A.pdb")
    nonbonded = mdfs.to_nonbonded_set(sp)
    bonded = mdfs.to_bonded_set(sp)
    mass = jnp.asarray(sp.masses)

    # Build H-bond constraints and drop those bonds from the harmonic term.
    cset, bonded = mdfs.setup_hbond_constraints(sp.bonds, sp.bond_r0, sp.masses, bonded, sp.n_atoms)
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nonbonded)

    R0 = mdfs.minimize_energy(energy_fn, jnp.asarray(sp.positions), max_iter=500).positions
    R0 = mdfs.apply_position_constraint(R0, R0, cset)  # satisfy constraints initially
    V0 = mdfs.apply_velocity_constraint(
        R0, mdfs.maxwell_boltzmann_velocities(jax.random.PRNGKey(0), mass, 300.0, sp.n_atoms), cset
    )

    state, step = mdfs.simulate_langevin(
        R0,
        V0,
        None,
        bonded,
        nonbonded,
        dt=0.002,
        mass=mass,
        gamma=10.0,
        temperature=300.0,
        constraints=cset,
    )
    dof = mdfs.constrained_dof(cset)
    energy_log = mdfs.EnergyLogger(energy_fn, mass, n_dof=dof, log_to_logger=False)
    state = mdfs.run(
        step,
        state,
        n_steps=2000,
        key=jax.random.PRNGKey(1),
        report_interval=50,
        callback=energy_log,
    )

    temps = np.array([r["temperature_K"] for r in energy_log.records])
    pairs = np.asarray(cset.pairs)
    rf = np.asarray(state.R)
    bond_dev = np.max(
        np.abs(np.linalg.norm(rf[pairs[:, 0]] - rf[pairs[:, 1]], axis=1) - np.asarray(cset.lengths))
    )
    print(f"NVT + LINCS: {sp.n_atoms} atoms, {pairs.shape[0]} H-bond constraints, dt = 2 fs")
    print(f"  mean temperature (last half) = {temps[len(temps) // 2 :].mean():.1f} K (dof = {dof})")
    print(f"  max |bond - target| = {bond_dev * 10:.1e} A (constraints held)")


if __name__ == "__main__":
    main()
