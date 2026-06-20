"""NVT MD of poly_A with hydrogen mass repartitioning (HMR), dt = 2 fs.

HMR moves mass onto hydrogens so the fastest motions slow down, allowing a ~4x
larger timestep (2 fs vs 0.5 fs) without bond constraints -- a ~4x throughput win
for the same simulated time. Demonstrates: Langevin NVT, dense nonbonded, HMR.

Run:
    python examples/nvt_hmr.py
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
    bonded = mdfs.to_bonded_set(sp)
    nonbonded = mdfs.to_nonbonded_set(sp)
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nonbonded)

    # Repartition hydrogen masses so dt = 2 fs is stable.
    mass = jnp.asarray(mdfs.repartition_hydrogen_masses(sp.masses, sp.bonds))
    R0 = mdfs.minimize_energy(energy_fn, jnp.asarray(sp.positions), max_iter=500).positions
    V0 = mdfs.maxwell_boltzmann_velocities(jax.random.PRNGKey(0), mass, 300.0, sp.n_atoms)

    state, step = mdfs.simulate_langevin(
        R0, V0, None, bonded, nonbonded, dt=0.002, mass=mass, gamma=10.0, temperature=300.0
    )
    energy_log = mdfs.EnergyLogger(energy_fn, mass, log_to_logger=False)
    state = mdfs.run(
        step,
        state,
        n_steps=2000,
        key=jax.random.PRNGKey(1),
        report_interval=50,
        callback=energy_log,
    )
    temps = np.array([r["temperature_K"] for r in energy_log.records])
    print(f"NVT + HMR: {sp.n_atoms} atoms, dt = 2 fs, 2000 steps (4 ps)")
    print(f"  mean temperature (last half) = {temps[len(temps) // 2 :].mean():.1f} K (target 300)")


if __name__ == "__main__":
    main()
