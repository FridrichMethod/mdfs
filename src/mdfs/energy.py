"""Differentiable potential-energy terms.

Energies are pure functions of positions; forces are obtained elsewhere via
``jax.grad``. Conventions match OpenMM (see :mod:`mdfs.constants`):

- bond:     ``0.5 * k * (r - r0)**2``
- angle:    ``0.5 * k * (theta - theta0)**2``
- torsion:  ``k * (1 + cos(n * phi - phase))``  (propers and impropers alike)
- LJ:       ``4 * eps * ((sig/r)**12 - (sig/r)**6)`` with Lorentz-Berthelot mixing
- Coulomb:  ``k_e * q_i * q_j / r`` (plain) or damped-shifted-force (DSF) under a cutoff

Angle/dihedral terms use ``atan2`` with softened norms so gradients (forces)
stay finite for near-degenerate geometries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from mdfs.constants import ONE_4PI_EPS0
from mdfs.types import DisplacementFn

# Softening floor for vector norms (nm); keeps division and atan2 gradients finite.
_EPS: float = 1e-12


def _safe_norm(
    x: jax.Array, axis: int = -1, keepdims: bool = False, eps: float = _EPS
) -> jax.Array:
    """Euclidean norm with a small floor so the gradient is finite at ``x == 0``."""
    return jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=keepdims) + eps * eps)


# ---------------------------------------------------------------------------
# Bonded terms
# ---------------------------------------------------------------------------


def bond_energy(R: jax.Array, bonds: jax.Array, k_r: jax.Array, r0: jax.Array) -> jax.Array:
    """Harmonic bond energy ``0.5 * k_r * (r - r0)**2`` summed over bonds."""
    i, j = bonds[:, 0], bonds[:, 1]
    r = _safe_norm(R[j] - R[i], axis=1)
    return 0.5 * jnp.sum(k_r * (r - r0) ** 2)


def angle_energy(
    R: jax.Array, angles: jax.Array, k_theta: jax.Array, theta0: jax.Array
) -> jax.Array:
    """Harmonic angle energy ``0.5 * k * (theta - theta0)**2`` summed over angles.

    ``theta`` is computed as ``atan2(|v1 x v2|, v1 . v2)``, which is numerically
    stable and (via the softened cross-product norm) keeps the force finite even
    when the three atoms are nearly collinear.
    """
    i, j, k = angles[:, 0], angles[:, 1], angles[:, 2]
    v1 = R[i] - R[j]
    v2 = R[k] - R[j]
    cross = jnp.cross(v1, v2)
    y = _safe_norm(cross, axis=1)
    x = jnp.sum(v1 * v2, axis=1)
    theta = jnp.arctan2(y, x)
    return 0.5 * jnp.sum(k_theta * (theta - theta0) ** 2)


def dihedral_phi(R: jax.Array, dihedrals: jax.Array) -> jax.Array:
    """Signed dihedral angle phi for each ``(i, j, k, l)`` (Blondel-Karplus form).

    The plane normals are normalized with a softened norm, so the gradient (and
    hence the torsion force) stays finite for near-collinear geometries where a
    normal vector nearly vanishes.
    """
    i, j, k, m = dihedrals[:, 0], dihedrals[:, 1], dihedrals[:, 2], dihedrals[:, 3]
    b1 = R[j] - R[i]
    b2 = R[k] - R[j]
    b3 = R[m] - R[k]
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    n1 = n1 / _safe_norm(n1, axis=1, keepdims=True)
    n2 = n2 / _safe_norm(n2, axis=1, keepdims=True)
    b2_hat = b2 / _safe_norm(b2, axis=1, keepdims=True)
    m1 = jnp.cross(n1, b2_hat)
    x = jnp.sum(n1 * n2, axis=1)
    y = jnp.sum(m1 * n2, axis=1)
    return jnp.arctan2(y, x)


def torsion_energy(
    R: jax.Array,
    dihedrals: jax.Array,
    periodicity: jax.Array,
    k: jax.Array,
    phase: jax.Array,
) -> jax.Array:
    """Periodic torsion energy ``sum k * (1 + cos(n * phi - phase))`` over entries.

    Each row is a single Fourier term; multi-term torsions appear as multiple
    rows sharing the same atom quadruplet (matching OpenMM's representation).
    """
    phi = dihedral_phi(R, dihedrals)
    return jnp.sum(k * (1.0 + jnp.cos(periodicity * phi - phase)))


# ---------------------------------------------------------------------------
# Nonbonded primitives
# ---------------------------------------------------------------------------


class LJMixParams(NamedTuple):
    """Per-type Lennard-Jones parameters (one entry per particle when types = arange(N))."""

    eps_type: jax.Array  # (Nt,) epsilon, kJ/mol
    sig_type: jax.Array  # (Nt,) sigma, nm


def lj_mixed_eps_sigma(
    types_i: jax.Array, types_j: jax.Array, params: LJMixParams
) -> tuple[jax.Array, jax.Array]:
    """Lorentz-Berthelot mixing: ``sig_ij = (sig_i + sig_j)/2``, ``eps_ij = sqrt(eps_i eps_j)``."""
    eps_ij = jnp.sqrt(params.eps_type[types_i] * params.eps_type[types_j])
    sig_ij = 0.5 * (params.sig_type[types_i] + params.sig_type[types_j])
    return eps_ij, sig_ij


def lj_12_6(eps_ij: jax.Array, sig_ij: jax.Array, r: jax.Array) -> jax.Array:
    """Lennard-Jones 12-6 potential ``4 eps [(sig/r)^12 - (sig/r)^6]``.

    ``r`` is floored at ``0.1 * sigma`` so ``(sig/r)**12`` cannot overflow to inf
    (which would poison forces as ``inf - inf = nan``) for catastrophic clashes in
    float32. This only caps the repulsion at separations well below any physical
    contact, keeping minimization from a clashy start finite.
    """
    r_safe = jnp.maximum(r, 0.1 * sig_ij + _EPS)
    sr6 = (sig_ij / r_safe) ** 6
    return 4.0 * eps_ij * (sr6 * sr6 - sr6)


def coulomb_plain(qiqj: jax.Array, r: jax.Array, k_e: float) -> jax.Array:
    """Unscreened Coulomb energy ``k_e * q_i q_j / r``."""
    return k_e * qiqj / (r + _EPS)


class DSFParams(NamedTuple):
    """Damped-shifted-force Coulomb parameters (for periodic/cutoff electrostatics).

    Note: the position-independent DSF self-energy term ``-k_e (alpha/sqrt(pi)) sum_i q_i^2``
    is omitted; it does not affect forces or dynamics but offsets absolute energies.
    """

    alpha: float  # damping (1/nm)
    r_cut: float  # cutoff (nm)
    k_e: float = ONE_4PI_EPS0


def coulomb_dsf_pair(
    qiqj: jax.Array, r: jax.Array, alpha: float, r_cut: float, k_e: float
) -> jax.Array:
    """Damped Shifted-Force (Fennell-Gezelter) Coulomb for a pair."""
    inv_r = 1.0 / (r + _EPS)
    erfc_ar = jax.scipy.special.erfc(alpha * r)
    erfc_arc = jax.scipy.special.erfc(alpha * r_cut)
    term_c = (
        erfc_arc / (r_cut**2)
        + (2.0 * alpha / jnp.sqrt(jnp.pi)) * jnp.exp(-(alpha**2) * (r_cut**2)) / r_cut
    )
    e = qiqj * (erfc_ar * inv_r - erfc_arc / r_cut + term_c * (r - r_cut))
    return k_e * e


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------


class BondedSet(NamedTuple):
    """Per-instance bonded parameters (bonds, angles, periodic torsions)."""

    bonds: jax.Array
    k_r: jax.Array
    r0: jax.Array
    angles: jax.Array
    k_theta: jax.Array
    theta0: jax.Array
    torsions: jax.Array
    periodicity: jax.Array
    torsion_k: jax.Array
    phase: jax.Array


class NonbondedSet(NamedTuple):
    """Nonbonded parameters and configuration.

    ``r_cut_lj=None`` and ``dsf=None`` select the plain, no-cutoff vacuum form
    (LJ + unscreened Coulomb over all non-excluded pairs), which reproduces an
    OpenMM ``NoCutoff`` system exactly (minus CMAP). Provide ``r_cut_lj`` and a
    ``DSFParams`` for periodic/cutoff electrostatics.
    """

    pairs: jax.Array  # (Np, 2) main-loop neighbor pairs, i < j
    types: jax.Array  # (N,) index into lj_params (= arange(N) for per-particle LJ)
    q: jax.Array  # (N,) charges (e)
    lj_params: LJMixParams
    exclude_mask: jax.Array  # (N, N) bool, True to exclude from the main loop
    exc_pairs: jax.Array  # (Ne, 2) exception pairs (1-2/1-3/1-4)
    exc_qq: jax.Array  # (Ne,) chargeProd (e^2)
    exc_sigma: jax.Array  # (Ne,) sigma (nm)
    exc_eps: jax.Array  # (Ne,) epsilon (kJ/mol)
    k_e: float = ONE_4PI_EPS0
    r_cut_lj: float | None = None
    dsf: DSFParams | None = None
    shift_lj: bool = False


def exception_energy(
    R: jax.Array,
    nb: NonbondedSet,
    displacement_fn: DisplacementFn,
) -> jax.Array:
    """Energy of nonbonded exception (1-4) pairs using their own parameters.

    Pure 1-2/1-3 exclusions carry ``qq = eps = 0`` and contribute nothing, so the
    full exception list can be passed safely.
    """
    if nb.exc_pairs.shape[0] == 0:
        return jnp.asarray(0.0)
    i, j = nb.exc_pairs[:, 0], nb.exc_pairs[:, 1]
    r = _safe_norm(displacement_fn(R[i], R[j]), axis=1)
    e_lj = lj_12_6(nb.exc_eps, nb.exc_sigma, r)
    e_coul = coulomb_plain(nb.exc_qq, r, nb.k_e)
    return jnp.sum(e_lj + e_coul)


def nonbonded_energy(
    R: jax.Array,
    nb: NonbondedSet,
    displacement_fn: DisplacementFn,
) -> jax.Array:
    """LJ + Coulomb over the main pair list, with exclusions and the exception term."""
    i, j = nb.pairs[:, 0], nb.pairs[:, 1]
    r = _safe_norm(displacement_fn(R[i], R[j]), axis=1)
    keep = ~nb.exclude_mask[i, j]

    eps_ij, sig_ij = lj_mixed_eps_sigma(nb.types[i], nb.types[j], nb.lj_params)
    e_lj = lj_12_6(eps_ij, sig_ij, r)
    if nb.r_cut_lj is not None:
        if nb.shift_lj:
            e_lj = e_lj - lj_12_6(eps_ij, sig_ij, jnp.full_like(r, nb.r_cut_lj))
        e_lj = jnp.where(r < nb.r_cut_lj, e_lj, 0.0)

    qiqj = nb.q[i] * nb.q[j]
    if nb.dsf is None:
        e_coul = coulomb_plain(qiqj, r, nb.k_e)
    else:
        e_coul = coulomb_dsf_pair(qiqj, r, nb.dsf.alpha, nb.dsf.r_cut, nb.dsf.k_e)
        e_coul = jnp.where(r < nb.dsf.r_cut, e_coul, 0.0)

    e_main = jnp.sum(jnp.where(keep, e_lj + e_coul, 0.0))
    return e_main + exception_energy(R, nb, displacement_fn)


# ---------------------------------------------------------------------------
# Total energy
# ---------------------------------------------------------------------------


def bonded_energy(R: jax.Array, bonded: BondedSet) -> jax.Array:
    """Sum of bond, angle, and torsion energies."""
    return (
        bond_energy(R, bonded.bonds, bonded.k_r, bonded.r0)
        + angle_energy(R, bonded.angles, bonded.k_theta, bonded.theta0)
        + torsion_energy(R, bonded.torsions, bonded.periodicity, bonded.torsion_k, bonded.phase)
    )


def total_energy_fn(
    displacement_fn: DisplacementFn,
    bonded: BondedSet,
    nonbonded: NonbondedSet,
) -> Callable[[jax.Array], jax.Array]:
    """Return a pure ``E(R)`` summing bonded and nonbonded energy (grad/jit-friendly)."""

    def energy(R: jax.Array) -> jax.Array:
        return bonded_energy(R, bonded) + nonbonded_energy(R, nonbonded, displacement_fn)

    return energy
