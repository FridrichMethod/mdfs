"""End-to-end regression: minimize then thermostat poly_A and check it stays sane.

Marked ``slow`` (runs real dynamics); excluded from the fast suite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mdfs


@pytest.mark.slow
def test_poly_a_minimize_and_langevin(poly_a_params):
    sp = poly_a_params
    bonded = mdfs.to_bonded_set(sp)
    nb = mdfs.to_nonbonded_set(sp, mdfs.all_pairs(sp.n_atoms))
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nb)

    # Relax.
    R0 = mdfs.minimize_energy(energy_fn, jnp.asarray(sp.positions), max_iter=300).positions
    assert jnp.all(jnp.isfinite(R0))

    # Thermostatted dynamics at 300 K.
    target_t = 300.0
    key = jax.random.PRNGKey(0)
    V0 = mdfs.maxwell_boltzmann_velocities(key, sp.masses, target_t, sp.n_atoms)
    state, step = mdfs.simulate_langevin(
        R0, V0, None, bonded, nb, dt=0.0005, mass=sp.masses, gamma=10.0, temperature=target_t
    )
    logger = mdfs.EnergyLogger(energy_fn, sp.masses, log_to_logger=False)
    state = mdfs.run(
        step, state, 4000, key=jax.random.PRNGKey(1), report_interval=50, callback=logger
    )

    temps = np.array([r["temperature_K"] for r in logger.records])
    energies = np.array([r["total_kj_mol"] for r in logger.records])
    assert np.all(np.isfinite(energies))

    # Temperature equilibrates near the target (last half of the trajectory).
    mean_t = temps[len(temps) // 2 :].mean()
    assert target_t * 0.8 < mean_t < target_t * 1.2

    # Structural integrity: bonds have not blown up.
    r_final = np.linalg.norm(
        np.array(state.R)[np.array(sp.bonds[:, 1])] - np.array(state.R)[np.array(sp.bonds[:, 0])],
        axis=1,
    )
    rms_bond_dev = float(np.sqrt(np.mean((r_final - np.array(sp.bond_r0)) ** 2)))
    assert rms_bond_dev < 0.02  # nm
