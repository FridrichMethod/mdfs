from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import grad, jit


class State(NamedTuple):
    R: jax.Array  # (N,3) positions in box coordinates
    V: jax.Array  # (N,3) velocities
    box: jax.Array  # (3,) side lengths (orthorhombic); can be ignored by energy_fn if unused
    t: float  # scalar time


def _as_col(mass: jax.Array | float, like: jax.Array) -> jax.Array:
    """Broadcast mass (scalar or (N,) or (N,1)) to shape (N,1) matching 'like'."""
    if jnp.ndim(mass) == 0:
        return jnp.full((like.shape[0], 1), mass)
    mass = jnp.asarray(mass)
    if mass.ndim == 1:
        mass = mass[:, None]
    return mass


def velocity_verlet(
    energy_fn: Callable[[jax.Array, jax.Array], jax.Array],
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
    shift_fn: Callable[[jax.Array, jax.Array], jax.Array],
    dt: float,
    mass: jax.Array | float,
):
    """
    Build a Velocity-Verlet stepper:
      - Forces from automatic differentiation: F = -∂E/∂R
      - Uses shift_fn for PBC-safe drifting and minimum-image displacements.
    Returns step(state) -> new_state, suitable for lax.scan.
    """
    # F(R, box) = -grad_R E(R, box)
    force_fn = jit(grad(energy_fn, argnums=0))

    @jit
    def step(state: State) -> State:
        R, V, box, t = state
        # broadcast mass to actual system size lazily
        m_eff = _as_col(mass, like=R)
        F = -force_fn(R, box)
        V_half = V + 0.5 * dt * F / m_eff
        R_new = shift_fn(R, V_half * dt)  # drift with PBC wrapping
        F_new = -force_fn(R_new, box)
        V_new = V_half + 0.5 * dt * F_new / m_eff
        return State(R_new, V_new, box, t + dt)

    return step


@dataclass(frozen=True)
class LangevinParams:
    gamma: float  # friction (1/time)
    temperature: float
    kB: float = 1.0  # Boltzmann constant (reduced units)


def langevin_baoab(
    energy_fn: Callable[[jax.Array, jax.Array], jax.Array],
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
    shift_fn: Callable[[jax.Array, jax.Array], jax.Array],
    dt: float,
    mass: jax.Array | float,
    params: LangevinParams,
):
    """
    Build a BAOAB Langevin stepper:
      B: half-kick (deterministic force)
      A: half-drift (positions)
      O: Ornstein–Uhlenbeck (stochastic velocity update)
      A: half-drift
      B: half-kick
    Returns step(state, key) -> (new_state, new_key).
    """
    force_fn = jit(grad(lambda R, box: energy_fn(R, box), argnums=0))
    gamma, T, kB = params.gamma, params.temperature, params.kB

    @jit
    def step(state: State, key: jax.Array) -> tuple[State, jax.Array]:
        R, V, box, t = state
        m_eff = _as_col(mass, like=R)

        # Precompute noise scale for OU step
        # Exact OU update over dt:
        #   c = exp(-gamma*dt)
        #   V <- c * V + sqrt((1 - c^2) * kT / m) * Normal(0,1)
        c = jnp.exp(-gamma * dt)
        sigma = jnp.sqrt((1.0 - c * c) * (kB * T)) / jnp.sqrt(m_eff)

        # --- B: half kick
        F = -force_fn(R, box)
        V = V + 0.5 * dt * F / m_eff

        # --- A: half drift
        R = shift_fn(R, V * (0.5 * dt))

        # --- O: Ornstein–Uhlenbeck
        key, sub = jax.random.split(key)
        xi = jax.random.normal(sub, shape=V.shape)
        V = c * V + sigma * xi

        # --- A: half drift
        R = shift_fn(R, V * (0.5 * dt))

        # --- B: half kick
        F = -force_fn(R, box)
        V = V + 0.5 * dt * F / m_eff

        return State(R, V, box, t + dt), key

    return step


# --- add below your existing LangevinParams and integrators ---


def velocity_verlet_pairs(
    energy_pairs_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
    shift_fn: Callable[[jax.Array, jax.Array], jax.Array],
    dt: float,
    mass: jax.Array | float,
):
    """Velocity-Verlet where energy depends on (R, box, pairs)."""
    force_fn = jit(grad(lambda R, box, pairs: energy_pairs_fn(R, box, pairs), argnums=0))

    @jit
    def step(state: State, pairs: jax.Array) -> State:
        R, V, box, t = state
        m_eff = _as_col(mass, like=R)
        F = -force_fn(R, box, pairs)
        V_half = V + 0.5 * dt * F / m_eff
        R_new = shift_fn(R, V_half * dt)
        F_new = -force_fn(R_new, box, pairs)
        V_new = V_half + 0.5 * dt * F_new / m_eff
        return State(R_new, V_new, box, t + dt)

    return step


def langevin_baoab_pairs(
    energy_pairs_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
    shift_fn: Callable[[jax.Array, jax.Array], jax.Array],
    dt: float,
    mass: jax.Array | float,
    params: LangevinParams,
):
    """BAOAB Langevin where energy depends on (R, box, pairs)."""
    force_fn = jit(grad(lambda R, box, pairs: energy_pairs_fn(R, box, pairs), argnums=0))
    gamma, T, kB = params.gamma, params.temperature, params.kB

    @jit
    def step(state: State, key: jax.Array, pairs: jax.Array) -> tuple[State, jax.Array]:
        R, V, box, t = state
        m_eff = _as_col(mass, like=R)

        c = jnp.exp(-gamma * dt)
        sigma = jnp.sqrt((1.0 - c * c) * (kB * T)) / jnp.sqrt(m_eff)

        # B
        F = -force_fn(R, box, pairs)
        V = V + 0.5 * dt * F / m_eff
        # A
        R = shift_fn(R, V * (0.5 * dt))
        # O
        key, sub = jax.random.split(key)
        xi = jax.random.normal(sub, shape=V.shape)
        V = c * V + sigma * xi
        # A
        R = shift_fn(R, V * (0.5 * dt))
        # B
        F = -force_fn(R, box, pairs)
        V = V + 0.5 * dt * F / m_eff

        return State(R, V, box, t + dt), key

    return step
