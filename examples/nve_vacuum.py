"""NVE (constant-energy) MD of poly_A in vacuum.

Demonstrates: velocity-Verlet integrator, free space (no PBC), plain LJ + Coulomb
over all atom pairs. Reports total-energy conservation (the NVE correctness check).

Run:
    python examples/nve_vacuum.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # accurate energy conservation

import jax.numpy as jnp
import numpy as np

import mdfs
from mdfs.paths import REPO_ROOT


def main() -> None:
    sp, _ = mdfs.system_params_from_pdb(REPO_ROOT / "assets" / "poly_A.pdb")
    bonded = mdfs.to_bonded_set(sp)
    nonbonded = mdfs.to_nonbonded_set(sp, mdfs.all_pairs(sp.n_atoms))
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nonbonded)

    mass = jnp.asarray(sp.masses)
    R0 = mdfs.minimize_energy(energy_fn, jnp.asarray(sp.positions), max_iter=300).positions
    V0 = mdfs.maxwell_boltzmann_velocities(jax.random.PRNGKey(0), mass, 300.0, sp.n_atoms)

    state, step = mdfs.simulate_nve(R0, V0, None, bonded, nonbonded, dt=0.0005, mass=mass)

    total = []
    state = mdfs.run(
        step,
        state,
        n_steps=4000,
        report_interval=50,
        callback=lambda _i, st: total.append(
            float(energy_fn(st.R)) + float(mdfs.kinetic_energy(st.V, mass))
        ),
    )
    total = np.array(total)
    drift = (total.max() - total.min()) / abs(total.mean())
    print(f"NVE: {sp.n_atoms} atoms, 4000 steps (2 ps)")
    print(f"  total energy mean = {total.mean():.2f} kJ/mol, relative drift = {drift:.2e}")


if __name__ == "__main__":
    main()
