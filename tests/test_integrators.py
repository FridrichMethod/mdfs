"""Tests for integrators: NVE conservation and Langevin (BAOAB) thermostatting."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mdfs.constants import BOLTZMANN_KJ_PER_MOL_K
from mdfs.integrators import (
    LangevinParams,
    State,
    kinetic_energy,
    langevin_baoab,
    maxwell_boltzmann_velocities,
    temperature,
    velocity_verlet,
)
from mdfs.space import free


def _harmonic(k: float):
    def energy_fn(R):
        return 0.5 * k * jnp.sum(R * R)

    return energy_fn


def test_kinetic_energy_and_temperature():
    V = jnp.ones((10, 3))
    mass = 2.0
    ke = float(kinetic_energy(V, mass))
    assert ke == pytest.approx(0.5 * mass * 30.0, rel=1e-9)
    t = float(temperature(V, mass))
    assert t == pytest.approx(2.0 * ke / (3 * 10 * BOLTZMANN_KJ_PER_MOL_K), rel=1e-9)


def test_maxwell_boltzmann_variance():
    key = jax.random.PRNGKey(0)
    n, mass, temp = 5000, 12.0, 300.0
    V = maxwell_boltzmann_velocities(key, mass, temp, n)
    expected_var = BOLTZMANN_KJ_PER_MOL_K * temp / mass
    assert float(jnp.var(V)) == pytest.approx(expected_var, rel=0.05)


def test_nve_energy_conservation():
    k, mass, dt = 50.0, 10.0, 0.005
    energy_fn = _harmonic(k)
    _, shift = free()
    key = jax.random.PRNGKey(1)
    R0 = jax.random.normal(key, (20, 3)) * 0.1
    V0 = maxwell_boltzmann_velocities(key, mass, 300.0, 20)
    step = velocity_verlet(energy_fn, shift, dt, mass)
    state = State(R0, V0, jnp.zeros(3), 0.0)
    energies = []
    for _ in range(2000):
        state = step(state)
        energies.append(float(energy_fn(state.R)) + float(kinetic_energy(state.V, mass)))
    energies = np.array(energies)
    rel_drift = (energies.max() - energies.min()) / abs(energies.mean())
    assert rel_drift < 1e-3


def test_baoab_equipartition():
    k, mass, dt, temp = 100.0, 12.0, 0.005, 300.0
    energy_fn = _harmonic(k)
    _, shift = free()
    key = jax.random.PRNGKey(2)
    n = 400
    R0 = jax.random.normal(key, (n, 3)) * 0.1
    V0 = maxwell_boltzmann_velocities(key, mass, temp, n)
    step = langevin_baoab(energy_fn, shift, dt, mass, LangevinParams(gamma=5.0, temperature=temp))
    state = State(R0, V0, jnp.zeros(3), 0.0)
    k2 = jax.random.PRNGKey(3)
    temps = []
    for i in range(4000):
        state, k2 = step(state, k2)
        if i >= 2000 and i % 5 == 0:
            temps.append(float(temperature(state.V, mass)))
    assert np.mean(temps) == pytest.approx(temp, rel=0.05)
