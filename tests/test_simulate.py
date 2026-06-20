"""Tests for the high-level simulation drivers and run loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import mdfs


def _setup(poly_a_params):
    sp = poly_a_params
    bonded = mdfs.to_bonded_set(sp)
    nb = mdfs.to_nonbonded_set(sp, mdfs.all_pairs(sp.n_atoms))
    R0 = jnp.asarray(sp.positions)
    V0 = jnp.zeros((sp.n_atoms, 3))
    return sp, bonded, nb, R0, V0


def test_make_energy_fn_free_vs_periodic(poly_a_params):
    _sp, bonded, nb, R0, _ = _setup(poly_a_params)
    e_free, _, box_free = mdfs.make_energy_fn(None, bonded, nb)
    assert jnp.isfinite(e_free(R0))
    assert jnp.all(box_free == 0)
    box = jnp.array([5.0, 5.0, 5.0])
    _, _, sim_box = mdfs.make_energy_fn(box, bonded, nb)
    assert jnp.allclose(sim_box, box)


def test_simulate_nve_runs(poly_a_params):
    sp, bonded, nb, R0, V0 = _setup(poly_a_params)
    state, step = mdfs.simulate_nve(R0, V0, None, bonded, nb, dt=0.0005, mass=sp.masses)
    state = mdfs.run(step, state, 5)
    assert jnp.all(jnp.isfinite(state.R))
    assert float(state.t) > 0


def test_simulate_langevin_runs_with_key(poly_a_params):
    sp, bonded, nb, R0, V0 = _setup(poly_a_params)
    state, step = mdfs.simulate_langevin(
        R0, V0, None, bonded, nb, dt=0.0005, mass=sp.masses, gamma=2.0, temperature=300.0
    )
    state = mdfs.run(step, state, 5, key=jax.random.PRNGKey(0))
    assert jnp.all(jnp.isfinite(state.R))


def test_run_callback_invoked(poly_a_params):
    sp, bonded, nb, R0, V0 = _setup(poly_a_params)
    state, step = mdfs.simulate_nve(R0, V0, None, bonded, nb, dt=0.0005, mass=sp.masses)
    seen = []
    mdfs.run(step, state, 6, report_interval=2, callback=lambda s, st: seen.append(s))
    assert seen == [2, 4, 6]
