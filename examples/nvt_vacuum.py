"""NVT (constant-temperature) MD of poly_A in vacuum.

Demonstrates: Langevin BAOAB thermostat, free space (no PBC), plain LJ + Coulomb,
energy minimization, and the trajectory/energy reporters. Writes an XTC trajectory
and an energy CSV (both gitignored).

Run:
    python examples/nvt_vacuum.py
"""

from __future__ import annotations

import logging

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import mdfs

logger = logging.getLogger("examples.nvt_vacuum")


def main() -> None:
    mdfs.configure_logging(logging.WARNING)
    sp, openmm_top = mdfs.system_params_from_pdb(mdfs.paths.REPO_ROOT / "assets" / "poly_A.pdb")
    bonded = mdfs.to_bonded_set(sp)
    nonbonded = mdfs.to_nonbonded_set(sp, mdfs.all_pairs(sp.n_atoms))
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nonbonded)

    mass = jnp.asarray(sp.masses)
    R0 = mdfs.minimize_energy(energy_fn, jnp.asarray(sp.positions), max_iter=500).positions
    V0 = mdfs.maxwell_boltzmann_velocities(jax.random.PRNGKey(0), mass, 300.0, sp.n_atoms)

    state, step = mdfs.simulate_langevin(
        R0, V0, None, bonded, nonbonded, dt=0.0005, mass=mass, gamma=10.0, temperature=300.0
    )

    traj = mdfs.TrajectoryRecorder(mdfs.mdtraj_topology_from_openmm(openmm_top))
    energy_log = mdfs.EnergyLogger(energy_fn, mass, log_to_logger=False)
    state = mdfs.run(
        step,
        state,
        n_steps=6000,
        key=jax.random.PRNGKey(1),
        report_interval=50,
        callback=mdfs.combine_callbacks(traj, energy_log),
    )

    temps = np.array([r["temperature_K"] for r in energy_log.records])
    print(f"NVT: {sp.n_atoms} atoms, 6000 steps (3 ps)")
    print(f"  mean temperature (last half) = {temps[len(temps) // 2 :].mean():.1f} K (target 300)")
    traj.save("poly_A_nvt.xtc")
    energy_log.save_csv("poly_A_nvt_energy.csv")
    print("  wrote poly_A_nvt.xtc and poly_A_nvt_energy.csv")


if __name__ == "__main__":
    main()
