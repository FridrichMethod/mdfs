"""Bond-length constraints via LINCS (Hess et al., J. Comput. Chem. 1997).

Constraining the fast X-H bond stretches lets the integrator take a larger
timestep (~2 fs; ~4 fs combined with hydrogen mass repartitioning). LINCS resets
the bonds after an unconstrained move by projecting onto the constraint manifold;
it is non-iterative (a fixed-order matrix power expansion plus one length
correction), so it is JIT-friendly.

This is a dense ``K x K`` implementation, appropriate for the small/medium systems
mdfs targets. The constraint solve is part of the integrator and is not
differentiated, so it does not affect the autodiff forces.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mdfs.constants import EPS as _EPS
from mdfs.energy import BondedSet

logger = logging.getLogger(__name__)


class ConstraintSet(NamedTuple):
    """Precomputed LINCS data for a fixed set of bond constraints.

    All arrays are JAX arrays so the set can be closed over inside ``jit``.
    """

    pairs: jax.Array  # (K, 2) constrained atom-index pairs
    lengths: jax.Array  # (K,) target bond lengths (nm)
    w_a: jax.Array  # (K,) inverse mass of atom a_k
    w_b: jax.Array  # (K,) inverse mass of atom b_k
    sdiag: jax.Array  # (K,) S_k = 1/sqrt(w_a + w_b)
    coupling: jax.Array  # (K, K) S_k S_n * sign * w_shared (0 on diagonal/uncoupled)
    order: int = 4  # power-expansion order
    n_atoms: int = 0  # total atoms in the system


def select_hbond_constraints(
    bonds: np.ndarray,
    bond_r0: np.ndarray,
    masses: np.ndarray,
    heavy_threshold: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pairs, lengths)`` for every bond involving a hydrogen.

    A bond is selected if either atom is a hydrogen (mass < ``heavy_threshold``).
    Lengths are the force-field equilibrium bond lengths ``bond_r0``.

    Important: hydrogen mass repartitioning inflates H masses above the default
    threshold, so pass the **pre-HMR** masses here (``setup_hbond_constraints``
    exposes ``selection_masses`` for exactly this).
    """
    is_h = masses < heavy_threshold
    mask = is_h[bonds[:, 0]] | is_h[bonds[:, 1]]
    if bonds.size and not mask.any():
        logger.warning(
            "select_hbond_constraints found 0 hydrogen bonds among %d bonds "
            "(min mass %.3f >= threshold %.3f). If hydrogen mass repartitioning was "
            "applied, pass the pre-HMR masses for selection.",
            len(bonds),
            float(np.min(masses)),
            heavy_threshold,
        )
    return bonds[mask].astype(np.int64), bond_r0[mask].astype(np.float64)


def make_constraint_set(
    pairs: np.ndarray,
    lengths: np.ndarray,
    masses: np.ndarray,
    n_atoms: int,
    order: int = 4,
) -> ConstraintSet:
    """Precompute the LINCS coupling matrix for a fixed set of constraints.

    ``masses`` must be the masses used for integration (e.g. after HMR).
    """
    pairs = np.asarray(pairs, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.float64)
    w = 1.0 / np.asarray(masses, dtype=np.float64)
    a, b = pairs[:, 0], pairs[:, 1]
    w_a, w_b = w[a], w[b]
    sdiag = 1.0 / np.sqrt(w_a + w_b)

    # Coupling: two bond constraints couple iff they share exactly one atom.
    # coupling[k, n] = S_k S_n * sign_kn * w_shared, with sign from bond orientation
    # (+1 if the shared atom is the same end in both, -1 otherwise). The (u_k . u_n)
    # factor is applied at runtime.
    k = len(pairs)
    coupling = np.zeros((k, k), dtype=np.float64)
    atom_to_cons: dict[int, list[int]] = {}
    for idx, (ai, bi) in enumerate(pairs):
        atom_to_cons.setdefault(int(ai), []).append(idx)
        atom_to_cons.setdefault(int(bi), []).append(idx)
    for shared, cons in atom_to_cons.items():
        for ki in cons:
            for ni in cons:
                if ki == ni:
                    continue
                sign_k = 1.0 if pairs[ki, 0] == shared else -1.0
                sign_n = 1.0 if pairs[ni, 0] == shared else -1.0
                coupling[ki, ni] = sdiag[ki] * sdiag[ni] * sign_k * sign_n * w[shared]

    return ConstraintSet(
        pairs=jnp.asarray(pairs),
        lengths=jnp.asarray(lengths),
        w_a=jnp.asarray(w_a),
        w_b=jnp.asarray(w_b),
        sdiag=jnp.asarray(sdiag),
        coupling=jnp.asarray(coupling),
        order=order,
        n_atoms=n_atoms,
    )


def setup_hbond_constraints(
    bonds: np.ndarray,
    bond_r0: np.ndarray,
    masses: np.ndarray,
    bonded: BondedSet,
    n_atoms: int,
    order: int = 4,
    selection_masses: np.ndarray | None = None,
) -> tuple[ConstraintSet, BondedSet]:
    """Build an H-bond constraint set and the matching reduced bonded set.

    Returns an H-bond :class:`ConstraintSet` plus a :class:`~mdfs.energy.BondedSet`
    with those bonds removed from the harmonic term (the constraint replaces them).

    ``masses`` are the **integration** masses (used for the LINCS coupling; apply HMR
    first if used). ``selection_masses`` (default: ``masses``) are used only to detect
    hydrogens -- with HMR you must pass the **pre-HMR** masses here, otherwise the
    inflated H masses exceed the detection threshold and no bonds are selected.
    """
    sel = masses if selection_masses is None else selection_masses
    pairs, lengths = select_hbond_constraints(bonds, bond_r0, sel)
    cset = make_constraint_set(pairs, lengths, masses, n_atoms, order=order)
    return cset, remove_constrained_bonds(bonded, pairs)


def remove_constrained_bonds(bonded: BondedSet, pairs: np.ndarray) -> BondedSet:
    """Drop constrained bonds from the harmonic bond term (the constraint replaces it)."""
    bonds = np.asarray(bonded.bonds)
    constrained = {frozenset(map(int, p)) for p in np.asarray(pairs)}
    keep = np.array([frozenset((int(i), int(j))) not in constrained for i, j in bonds], dtype=bool)
    return bonded._replace(bonds=bonded.bonds[keep], k_r=bonded.k_r[keep], r0=bonded.r0[keep])


def constrained_dof(cset: ConstraintSet) -> int:
    """Degrees of freedom ``3N - K`` for temperature with this constraint set.

    Note: this does not subtract the 3 center-of-mass translational DOF; subtract
    an extra 3 if COM motion is removed. (Without constraints, dof is just ``3N``,
    the default used by :func:`mdfs.temperature`.)
    """
    return 3 * cset.n_atoms - cset.pairs.shape[0]


def _unit_bonds(positions: jax.Array, cset: ConstraintSet, eps: float = _EPS) -> jax.Array:
    """Reference unit bond vectors ``u_k`` (K, 3) at ``positions``."""
    d = positions[cset.pairs[:, 0]] - positions[cset.pairs[:, 1]]
    return d / jnp.sqrt(jnp.sum(d * d, axis=1, keepdims=True) + eps * eps)


def _coupling_matrix(u: jax.Array, cset: ConstraintSet) -> jax.Array:
    """LINCS expansion matrix ``A = -coupling * (u_k . u_n)`` (zero diagonal)."""
    return -cset.coupling * (u @ u.T)


def _solve(a_mat: jax.Array, rhs: jax.Array, order: int) -> jax.Array:
    """Apply ``(I - A)^-1 ~ sum_p A^p`` to ``rhs`` (truncated at ``order``)."""
    sol = rhs
    acc = rhs
    for _ in range(order):
        acc = a_mat @ acc
        sol = sol + acc
    return sol


def _apply_multipliers(
    positions: jax.Array, u: jax.Array, lam: jax.Array, cset: ConstraintSet
) -> jax.Array:
    """Displace atoms by ``-/+ w * lam * u`` along each constraint."""
    a, b = cset.pairs[:, 0], cset.pairs[:, 1]
    disp_a = -(cset.w_a * lam)[:, None] * u
    disp_b = (cset.w_b * lam)[:, None] * u
    return positions.at[a].add(disp_a).at[b].add(disp_b)


def apply_position_constraint(
    r_ref: jax.Array, r_unc: jax.Array, cset: ConstraintSet, eps: float = _EPS
) -> jax.Array:
    """Project unconstrained positions ``r_unc`` onto the constraint manifold.

    Directions are taken at the reference (start-of-step) positions ``r_ref``.
    Runs the LINCS length expansion plus one length-correction pass.
    """
    u = _unit_bonds(r_ref, cset, eps=eps)
    a_mat = _coupling_matrix(u, cset)
    a, b = cset.pairs[:, 0], cset.pairs[:, 1]

    # Step 1: linear projection toward the target lengths.
    proj = jnp.sum(u * (r_unc[a] - r_unc[b]), axis=1) - cset.lengths
    lam = cset.sdiag * _solve(a_mat, cset.sdiag * proj, cset.order)
    r = _apply_multipliers(r_unc, u, lam, cset)

    # Step 2: length correction p_k = sqrt(2 d^2 - l^2) for the rotational error.
    bond = r[a] - r[b]
    l2 = jnp.sum(bond * bond, axis=1)
    # Floor the radicand (a squared length) just above zero; the solve is not
    # differentiated, so a bare eps rather than eps**2 is fine here.
    p = jnp.sqrt(jnp.maximum(2.0 * cset.lengths**2 - l2, eps))
    proj2 = jnp.sum(u * bond, axis=1) - p
    lam2 = cset.sdiag * _solve(a_mat, cset.sdiag * proj2, cset.order)
    return _apply_multipliers(r, u, lam2, cset)


def apply_velocity_constraint(
    r_ref: jax.Array, velocities: jax.Array, cset: ConstraintSet, eps: float = _EPS
) -> jax.Array:
    """Project out velocity components along the constraints (RATTLE velocity step)."""
    u = _unit_bonds(r_ref, cset, eps=eps)
    a_mat = _coupling_matrix(u, cset)
    a, b = cset.pairs[:, 0], cset.pairs[:, 1]
    proj = jnp.sum(u * (velocities[a] - velocities[b]), axis=1)
    mu = cset.sdiag * _solve(a_mat, cset.sdiag * proj, cset.order)
    return _apply_multipliers(velocities, u, mu, cset)
