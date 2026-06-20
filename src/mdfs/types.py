"""Shared type aliases for ``mdfs``."""

from __future__ import annotations

import os
from collections.abc import Callable

import jax

# Accept anything path-like at API boundaries; emit concrete ``Path`` internally.
StrPath = str | os.PathLike[str]

# A pairwise displacement function ``(Ra, Rb) -> Rb - Ra`` (with optional PBC).
DisplacementFn = Callable[[jax.Array, jax.Array], jax.Array]

# A position-update function ``(R, dR) -> R + dR`` (with optional PBC wrapping).
ShiftFn = Callable[[jax.Array, jax.Array], jax.Array]

# A pure potential-energy function ``E(R) -> scalar``.
EnergyFn = Callable[[jax.Array], jax.Array]
