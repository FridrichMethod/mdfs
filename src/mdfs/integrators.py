"""Time integrators (velocity-Verlet for NVE, BAOAB Langevin for NVT).

Forces are obtained by automatic differentiation of a pure energy function
``energy_fn(R) -> scalar`` (periodic boundary conditions, if any, are baked into
that closure and into ``shift_fn``). This keeps integrators independent of how
the energy/neighbor list is constructed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import grad, jit

from mdfs.constants import BOLTZMANN_KJ_PER_MOL_K
from mdfs.types import EnergyFn, ShiftFn


class State(NamedTuple):
    """Dynamical state of the system."""

    R: jax.Array  # (N, 3) positions, nm
    V: jax.Array  # (N, 3) velocities, nm/ps
    box: jax.Array  # (3,) orthorhombic side lengths, nm (zeros for vacuum)
    t: float | jax.Array  # time, ps (a traced scalar inside jit)


def _as_col(mass: jax.Array | float, like: jax.Array) -> jax.Array:
    """Broadcast a scalar or ``(N,)`` mass to a ``(N, 1)`` column matching ``like``."""
    if jnp.ndim(mass) == 0:
        return jnp.full((like.shape[0], 1), mass)
    mass = jnp.asarray(mass)
    return mass[:, None] if mass.ndim == 1 else mass


def kinetic_energy(V: jax.Array, mass: jax.Array | float) -> jax.Array:
    """Kinetic energy ``0.5 * sum(m * v**2)`` in kJ/mol."""
    m = _as_col(mass, V)
    return 0.5 * jnp.sum(m * V * V)


def temperature(
    V: jax.Array,
    mass: jax.Array | float,
    kB: float = BOLTZMANN_KJ_PER_MOL_K,
    n_dof: int | None = None,
) -> jax.Array:
    """Instantaneous temperature ``2 * KE / (n_dof * kB)`` in kelvin.

    ``n_dof`` defaults to ``3 * N`` (no constraints, no COM removal).
    """
    if n_dof is None:
        n_dof = 3 * V.shape[0]
    return 2.0 * kinetic_energy(V, mass) / (n_dof * kB)


def velocity_verlet(
    energy_fn: EnergyFn,
    shift_fn: ShiftFn,
    dt: float,
    mass: jax.Array | float,
) -> Callable[[State], State]:
    """Velocity-Verlet (NVE) stepper. Returns ``step(state) -> state``."""
    force_fn = grad(energy_fn)

    @jit
    def step(state: State) -> State:
        R, V, box, t = state
        m = _as_col(mass, R)
        F = -force_fn(R)
        v_half = V + 0.5 * dt * F / m
        r_new = shift_fn(R, v_half * dt)
        f_new = -force_fn(r_new)
        v_new = v_half + 0.5 * dt * f_new / m
        return State(r_new, v_new, box, t + dt)

    return step


@dataclass(frozen=True)
class LangevinParams:
    """Langevin thermostat parameters."""

    gamma: float  # friction, 1/ps
    temperature: float  # target temperature, K
    kB: float = BOLTZMANN_KJ_PER_MOL_K


def langevin_baoab(
    energy_fn: EnergyFn,
    shift_fn: ShiftFn,
    dt: float,
    mass: jax.Array | float,
    params: LangevinParams,
) -> Callable[[State, jax.Array], tuple[State, jax.Array]]:
    """BAOAB Langevin (NVT) stepper. Returns ``step(state, key) -> (state, key)``.

    The O step is the exact Ornstein-Uhlenbeck velocity update
    ``V <- c V + sqrt((1 - c^2) kT / m) xi`` with ``c = exp(-gamma dt)``.
    """
    force_fn = grad(energy_fn)
    gamma, temp, kB = params.gamma, params.temperature, params.kB
    c = jnp.exp(-gamma * dt)

    @jit
    def step(state: State, key: jax.Array) -> tuple[State, jax.Array]:
        R, V, box, t = state
        m = _as_col(mass, R)
        sigma = jnp.sqrt((1.0 - c * c) * kB * temp / m)

        F = -force_fn(R)
        V = V + 0.5 * dt * F / m  # B
        R = shift_fn(R, 0.5 * dt * V)  # A
        key, sub = jax.random.split(key)  # O
        V = c * V + sigma * jax.random.normal(sub, V.shape)
        R = shift_fn(R, 0.5 * dt * V)  # A
        F = -force_fn(R)
        V = V + 0.5 * dt * F / m  # B
        return State(R, V, box, t + dt), key

    return step


def maxwell_boltzmann_velocities(
    key: jax.Array,
    mass: jax.Array | float,
    temperature: float,
    n_atoms: int,
    kB: float = BOLTZMANN_KJ_PER_MOL_K,
) -> jax.Array:
    """Sample ``(N, 3)`` velocities from the Maxwell-Boltzmann distribution at ``temperature``."""
    m = jnp.broadcast_to(_as_col(mass, jnp.zeros((n_atoms, 1))), (n_atoms, 1))
    return jax.random.normal(key, (n_atoms, 3)) * jnp.sqrt(kB * temperature / m)
