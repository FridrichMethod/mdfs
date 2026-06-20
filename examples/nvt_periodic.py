"""NVT MD of poly_A in a periodic box (PBC).

Demonstrates the periodic path: orthorhombic minimum-image convention (MIC),
damped-shifted-force (DSF) electrostatics, and an LJ cutoff. The peptide is placed
in a cubic box large enough that the cutoff satisfies r_cut <= min(box)/2.

Run:
    python examples/nvt_periodic.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import mdfs
from mdfs.energy import DSFParams
from mdfs.paths import REPO_ROOT


def main() -> None:
    sp, _ = mdfs.system_params_from_pdb(REPO_ROOT / "assets" / "poly_A.pdb")

    # Cubic box around the molecule; cutoff must satisfy r_cut <= min(box)/2.
    extent = float(np.ptp(sp.positions, axis=0).max())
    side = extent + 4.0  # nm
    box = jnp.array([side, side, side])
    r_cut = side / 2.0 - 0.1
    # center the molecule in the box
    R0 = jnp.asarray(sp.positions - sp.positions.mean(axis=0) + side / 2.0)

    bonded = mdfs.to_bonded_set(sp)
    nonbonded = mdfs.to_nonbonded_set(  # dense (N, N) path, periodic + DSF cutoff
        sp,
        r_cut_lj=r_cut,
        dsf=DSFParams(alpha=2.0, r_cut=r_cut),
    )

    mass = jnp.asarray(sp.masses)
    V0 = mdfs.maxwell_boltzmann_velocities(jax.random.PRNGKey(0), mass, 300.0, sp.n_atoms)
    state, step = mdfs.simulate_langevin(
        R0, V0, box, bonded, nonbonded, dt=0.0005, mass=mass, gamma=10.0, temperature=300.0
    )

    energy_log = mdfs.EnergyLogger(
        mdfs.make_energy_fn(box, bonded, nonbonded)[0], mass, log_to_logger=False
    )
    state = mdfs.run(
        step,
        state,
        n_steps=4000,
        key=jax.random.PRNGKey(1),
        report_interval=50,
        callback=energy_log,
    )
    temps = np.array([r["temperature_K"] for r in energy_log.records])
    print(f"NVT periodic: {sp.n_atoms} atoms, box={side:.1f} nm, r_cut={r_cut:.1f} nm")
    print(f"  mean temperature (last half) = {temps[len(temps) // 2 :].mean():.1f} K (target 300)")


if __name__ == "__main__":
    main()
