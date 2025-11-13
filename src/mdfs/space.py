from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

DisplacementFn = Callable[[jax.Array, jax.Array], jax.Array]
ShiftFn = Callable[[jax.Array, jax.Array], jax.Array]


def free() -> tuple[DisplacementFn, ShiftFn]:
    """Unbounded Euclidean space (no PBC)."""

    def displacement_fn(Ra: jax.Array, Rb: jax.Array) -> jax.Array:
        return Rb - Ra

    def shift_fn(R: jax.Array, dR: jax.Array) -> jax.Array:
        return R + dR

    return displacement_fn, shift_fn


def periodic(box: jax.Array) -> tuple[DisplacementFn, ShiftFn]:
    """
    Orthorhombic periodic space with the Minimum-Image Convention (MIC).
    `box` is shape (D,) with side lengths (e.g., jnp.array([Lx, Ly, Lz])) in nm.

    - displacement_fn returns the nearest-image displacement.
    - shift_fn drifts positions and wraps them back into [0, L) per dimension.

    Note: For MIC to be valid, short-range cutoffs should satisfy r_cut <= min(box)/2.
    """
    box = jnp.asarray(box)
    inv_box = 1.0 / box

    def displacement_fn(Ra: jax.Array, Rb: jax.Array) -> jax.Array:
        # nearest-image displacement for orthorhombic boxes
        dR = Rb - Ra
        return dR - box * jnp.round(dR * inv_box)

    def shift_fn(R: jax.Array, dR: jax.Array) -> jax.Array:
        R_new = R + dR
        # wrap into [0, L) with floor-based modular arithmetic
        return R_new - box * jnp.floor(R_new * inv_box)

    return displacement_fn, shift_fn


def wrap(R: jax.Array, box: jax.Array) -> jax.Array:
    """
    Convenience helper to wrap arbitrary coordinates into [0, L) under orthorhombic PBC.
    """
    box = jnp.asarray(box)
    inv_box = 1.0 / box
    return R - box * jnp.floor(R * inv_box)
