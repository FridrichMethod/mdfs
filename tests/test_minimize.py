"""Tests for energy minimization."""

from __future__ import annotations

import jax.numpy as jnp

import mdfs
from mdfs.minimize import minimize_energy, steepest_descent


def _quadratic(center):
    def energy_fn(R):
        return jnp.sum((R - center) ** 2)

    return energy_fn


def test_steepest_descent_decreases_quadratic():
    center = jnp.array([[1.0, -2.0, 0.5]])
    energy_fn = _quadratic(center)
    R0 = jnp.zeros((1, 3))
    res = steepest_descent(energy_fn, R0, n_steps=500, learning_rate=0.05, max_step=0.5)
    assert res.energy < res.initial_energy
    assert res.energy < 1e-3


def test_bfgs_finds_quadratic_minimum():
    center = jnp.array([[1.0, -2.0, 0.5]])
    res = minimize_energy(_quadratic(center), jnp.zeros((1, 3)), max_iter=200)
    assert res.energy < 1e-8


def test_minimize_relaxes_poly_a(poly_a_params):
    sp = poly_a_params
    bonded = mdfs.to_bonded_set(sp)
    nb = mdfs.to_nonbonded_set(sp)  # dense default
    energy_fn, _, _ = mdfs.make_energy_fn(None, bonded, nb)
    res = minimize_energy(energy_fn, jnp.asarray(sp.positions), max_iter=200)
    assert res.energy < res.initial_energy
    assert jnp.all(jnp.isfinite(res.positions))
