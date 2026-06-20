"""Hydrogen mass repartitioning (HMR).

HMR moves mass from heavy atoms onto the hydrogens bonded to them. This slows the
fastest motions (X-H stretches) so a larger integration timestep is stable
(~2 fs instead of ~0.5 fs without constraints), giving a few-fold speedup. Total
mass is conserved, so thermodynamic/equilibrium properties are unaffected; only
the (unphysical) high-frequency H dynamics change.
"""

from __future__ import annotations

import numpy as np


def repartition_hydrogen_masses(
    masses: np.ndarray,
    bonds: np.ndarray,
    hydrogen_mass: float = 3.024,
    heavy_threshold: float = 1.5,
) -> np.ndarray:
    """Return masses with hydrogens repartitioned to ``hydrogen_mass`` (amu).

    For every bond connecting a hydrogen (mass < ``heavy_threshold``) to a heavy
    atom, the hydrogen is set to ``hydrogen_mass`` and the difference is subtracted
    from the bonded heavy atom (total mass is conserved). ``hydrogen_mass=3.024``
    typically permits a ~2 fs timestep without bond constraints.

    Args:
        masses: ``(N,)`` per-atom masses (amu).
        bonds: ``(Nb, 2)`` bonded atom-index pairs.
        hydrogen_mass: target mass for each hydrogen (amu).
        heavy_threshold: atoms with mass below this are treated as hydrogens.

    Returns:
        A new ``(N,)`` array of repartitioned masses.

    Note:
        Each hydrogen is assumed to be bonded to exactly one heavy atom (true for
        standard biomolecular force fields). A bridging hydrogen (bonded to two
        heavy atoms) is rejected, since repartitioning it would not conserve mass.

    Raises:
        ValueError: if ``hydrogen_mass`` is non-positive, ``bonds`` is not shaped
            ``(Nb, 2)``, a hydrogen bridges two heavy atoms, or the result contains
            a non-positive / non-finite mass (``hydrogen_mass`` too large).
    """
    if hydrogen_mass <= 0.0:
        raise ValueError(f"hydrogen_mass must be positive, got {hydrogen_mass}")
    m = np.asarray(masses, dtype=np.float64).copy()
    bonds = np.asarray(bonds)
    if bonds.size == 0:
        return m
    if bonds.ndim != 2 or bonds.shape[1] != 2:
        raise ValueError(f"bonds must have shape (Nb, 2), got {bonds.shape}")
    is_h = m < heavy_threshold
    a, b = bonds[:, 0], bonds[:, 1]
    xh = is_h[a] ^ is_h[b]  # bonds connecting exactly one hydrogen to a heavy atom
    if not np.any(xh):
        return m
    pair = bonds[xh]
    first_is_h = is_h[pair[:, 0]]
    h_idx = np.where(first_is_h, pair[:, 0], pair[:, 1])
    heavy_idx = np.where(first_is_h, pair[:, 1], pair[:, 0])
    if np.unique(h_idx).size != h_idx.size:
        raise ValueError(
            "A hydrogen is bonded to more than one heavy atom (bridging hydrogen); "
            "only single-heavy-atom hydrogens are supported."
        )
    delta = hydrogen_mass - m[h_idx]  # mass to move onto each hydrogen
    m[h_idx] = hydrogen_mass
    np.add.at(m, heavy_idx, -delta)  # heavy atoms may carry several hydrogens
    if not np.all(np.isfinite(m) & (m > 0.0)):
        raise ValueError(
            f"Repartitioning produced a non-positive or non-finite mass "
            f"(hydrogen_mass={hydrogen_mass} too large, or non-finite input masses)."
        )
    return m
