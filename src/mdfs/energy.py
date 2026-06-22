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

from mdfs.constants import EPS as _EPS
from mdfs.constants import ONE_4PI_EPS0
from mdfs.types import DisplacementFn

# Soft-core floor: LJ separations are clamped to this fraction of sigma so the
# r**-12 term cannot overflow for catastrophic clashes (see ``lj_12_6``).
_LJ_FLOOR_FRAC: float = 0.1


def _safe_norm(
    x: jax.Array, axis: int = -1, keepdims: bool = False, eps: float = _EPS
) -> jax.Array:
    """Euclidean norm with a small floor so the gradient is finite at ``x == 0``."""
    return jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=keepdims) + eps * eps)


# ---------------------------------------------------------------------------
# Bonded terms
# ---------------------------------------------------------------------------


def bond_energy(
    R: jax.Array, bonds: jax.Array, k_r: jax.Array, r0: jax.Array, eps: float = _EPS
) -> jax.Array:
    """Harmonic bond energy ``0.5 * k_r * (r - r0)**2`` summed over bonds."""
    i, j = bonds[:, 0], bonds[:, 1]
    r = _safe_norm(R[j] - R[i], axis=1, eps=eps)
    return 0.5 * jnp.sum(k_r * (r - r0) ** 2)


def angle_energy(
    R: jax.Array, angles: jax.Array, k_theta: jax.Array, theta0: jax.Array, eps: float = _EPS
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
    y = _safe_norm(cross, axis=1, eps=eps)
    x = jnp.sum(v1 * v2, axis=1)
    theta = jnp.arctan2(y, x)
    return 0.5 * jnp.sum(k_theta * (theta - theta0) ** 2)


def dihedral_phi(R: jax.Array, dihedrals: jax.Array, eps: float = _EPS) -> jax.Array:
    """Signed dihedral angle phi for each ``(i, j, k, l)`` (Blondel-Karplus form).

    Uses unnormalized plane normals (``x`` and ``y`` carry the same scale, so the
    ratio gives phi) with a guard so that at exact collinearity -- where a normal
    vanishes and ``x = y = 0`` -- the gradient stays finite rather than NaN
    (``atan2(0, 0)`` has an undefined gradient that would poison all forces).
    """
    i, j, k, m = dihedrals[:, 0], dihedrals[:, 1], dihedrals[:, 2], dihedrals[:, 3]
    b1 = R[j] - R[i]
    b2 = R[k] - R[j]
    b3 = R[m] - R[k]
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    b2_hat = b2 / _safe_norm(b2, axis=1, keepdims=True, eps=eps)
    m1 = jnp.cross(n1, b2_hat)
    x = jnp.sum(n1 * n2, axis=1)
    y = jnp.sum(m1 * n2, axis=1)
    x = x + jnp.where(x * x + y * y < eps * eps, eps, 0.0)  # avoid atan2(0, 0)
    return jnp.arctan2(y, x)


def torsion_energy(
    R: jax.Array,
    dihedrals: jax.Array,
    periodicity: jax.Array,
    k: jax.Array,
    phase: jax.Array,
    eps: float = _EPS,
) -> jax.Array:
    """Periodic torsion energy ``sum k * (1 + cos(n * phi - phase))`` over entries.

    Each row is a single Fourier term; multi-term torsions appear as multiple
    rows sharing the same atom quadruplet (matching OpenMM's representation).
    """
    phi = dihedral_phi(R, dihedrals, eps=eps)
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


def lj_12_6(eps_ij: jax.Array, sig_ij: jax.Array, r: jax.Array, eps: float = _EPS) -> jax.Array:
    """Lennard-Jones 12-6 potential ``4 eps_ij [(sig/r)^12 - (sig/r)^6]``.

    ``r`` is floored at ``0.1 * sigma`` so ``(sig/r)**12`` cannot overflow to inf
    (which would poison forces as ``inf - inf = nan``) for catastrophic clashes in
    float32. This only caps the repulsion at separations well below any physical
    contact, keeping minimization from a clashy start finite. ``eps`` is the
    numerical floor (distinct from the LJ well depth ``eps_ij``).
    """
    r_safe = jnp.maximum(r, _LJ_FLOOR_FRAC * sig_ij + eps)
    sr6 = (sig_ij / r_safe) ** 6
    return 4.0 * eps_ij * (sr6 * sr6 - sr6)


def coulomb_plain(
    qiqj: jax.Array, r: jax.Array, k_e: float = ONE_4PI_EPS0, eps: float = _EPS
) -> jax.Array:
    """Unscreened Coulomb energy ``k_e * q_i q_j / r``."""
    return k_e * qiqj / (r + eps)


class DSFParams(NamedTuple):
    """Damped-shifted-force Coulomb parameters (for periodic/cutoff electrostatics).

    Note: the position-independent DSF self-energy term ``-k_e (alpha/sqrt(pi)) sum_i q_i^2``
    is omitted; it does not affect forces or dynamics but offsets absolute energies.
    """

    alpha: float  # damping (1/nm)
    r_cut: float  # cutoff (nm)
    k_e: float = ONE_4PI_EPS0


def coulomb_dsf_pair(
    qiqj: jax.Array,
    r: jax.Array,
    alpha: float,
    r_cut: float,
    k_e: float = ONE_4PI_EPS0,
    eps: float = _EPS,
) -> jax.Array:
    """Damped Shifted-Force (Fennell-Gezelter) Coulomb for a pair."""
    inv_r = 1.0 / (r + eps)
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

    ``pairs=None`` (the default) uses the **dense** ``(N, N)`` formulation: forces
    (``jax.grad``) reduce over an axis instead of scattering onto atoms, which is
    far faster on GPU but uses O(N^2) memory -- ideal for up to a few thousand
    atoms. Provide an explicit ``(Np, 2)`` pair list (e.g. from
    :func:`mdfs.partition.neighbor_list`) to use the O(N) pair-list path for
    larger systems.

    ``shift_lj`` (default True) shifts the LJ energy to zero at ``r_cut_lj`` for
    energy continuity. Note this removes only the energy jump, not the force
    discontinuity at the cutoff; for strict NVE prefer the no-cutoff vacuum form or
    a thermostat. (Ignored when ``r_cut_lj`` is None.)
    """

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
    shift_lj: bool = True
    pairs: jax.Array | None = None  # (Np, 2); None -> dense (N, N) path


def exception_energy(
    R: jax.Array,
    nb: NonbondedSet,
    displacement_fn: DisplacementFn,
    eps: float = _EPS,
) -> jax.Array:
    """Energy of nonbonded exception (1-4) pairs using their own parameters.

    Pure 1-2/1-3 exclusions carry ``qq = eps = 0`` and contribute nothing, so the
    full exception list can be passed safely. This is an O(N) sparse term.
    """
    if nb.exc_pairs.shape[0] == 0:
        return jnp.asarray(0.0)
    i, j = nb.exc_pairs[:, 0], nb.exc_pairs[:, 1]
    r = _safe_norm(displacement_fn(R[i], R[j]), axis=1, eps=eps)
    e_lj = lj_12_6(nb.exc_eps, nb.exc_sigma, r, eps=eps)
    e_coul = coulomb_plain(nb.exc_qq, r, nb.k_e, eps=eps)
    return jnp.sum(e_lj + e_coul)


def _pair_lj_coulomb(
    r: jax.Array,
    eps_ij: jax.Array,
    sig_ij: jax.Array,
    qiqj: jax.Array,
    nb: NonbondedSet,
    eps: float = _EPS,
) -> jax.Array:
    """LJ + Coulomb energy for a batch of pairs at distances ``r`` (cutoffs applied)."""
    e_lj = lj_12_6(eps_ij, sig_ij, r, eps=eps)
    if nb.r_cut_lj is not None:
        if nb.shift_lj:
            e_lj = e_lj - lj_12_6(eps_ij, sig_ij, jnp.full_like(r, nb.r_cut_lj), eps=eps)
        e_lj = jnp.where(r < nb.r_cut_lj, e_lj, 0.0)
    if nb.dsf is None:
        e_coul = coulomb_plain(qiqj, r, nb.k_e, eps=eps)
    else:
        e_coul = coulomb_dsf_pair(qiqj, r, nb.dsf.alpha, nb.dsf.r_cut, nb.dsf.k_e, eps=eps)
        e_coul = jnp.where(r < nb.dsf.r_cut, e_coul, 0.0)
    return e_lj + e_coul


def _nonbonded_dense(
    R: jax.Array, nb: NonbondedSet, displacement_fn: DisplacementFn, eps: float = _EPS
) -> jax.Array:
    """Dense ``(N, N)`` LJ + Coulomb via broadcasting (grad reduces, never scatters)."""
    n = R.shape[0]
    d_r = displacement_fn(R[:, None, :], R[None, :, :])  # (N, N, 3)
    # Push the (zero-distance) diagonal to a finite distance so 1/r and (sig/r)^12
    # have finite values *and* gradients; it is masked out below regardless.
    r = jnp.sqrt(jnp.sum(d_r * d_r, axis=-1) + jnp.eye(n) + eps * eps)

    lj_eps = nb.lj_params.eps_type[nb.types]
    lj_sig = nb.lj_params.sig_type[nb.types]
    eps_ij = jnp.sqrt(lj_eps[:, None] * lj_eps[None, :])
    sig_ij = 0.5 * (lj_sig[:, None] + lj_sig[None, :])
    qiqj = nb.q[:, None] * nb.q[None, :]

    e = _pair_lj_coulomb(r, eps_ij, sig_ij, qiqj, nb, eps=eps)
    keep = (~nb.exclude_mask) & (~jnp.eye(n, dtype=bool))
    e_main = 0.5 * jnp.sum(jnp.where(keep, e, 0.0))  # each unordered pair counted twice
    return e_main + exception_energy(R, nb, displacement_fn, eps=eps)


def _nonbonded_pairs(
    R: jax.Array, nb: NonbondedSet, displacement_fn: DisplacementFn, eps: float = _EPS
) -> jax.Array:
    """Pair-list ``(Np, 2)`` LJ + Coulomb (O(N) with a neighbor list; grad scatters)."""
    assert nb.pairs is not None
    i, j = nb.pairs[:, 0], nb.pairs[:, 1]
    r = _safe_norm(displacement_fn(R[i], R[j]), axis=1, eps=eps)
    keep = ~nb.exclude_mask[i, j]
    eps_ij, sig_ij = lj_mixed_eps_sigma(nb.types[i], nb.types[j], nb.lj_params)
    e = _pair_lj_coulomb(r, eps_ij, sig_ij, nb.q[i] * nb.q[j], nb, eps=eps)
    e_main = jnp.sum(jnp.where(keep, e, 0.0))
    return e_main + exception_energy(R, nb, displacement_fn, eps=eps)


def nonbonded_energy(
    R: jax.Array,
    nb: NonbondedSet,
    displacement_fn: DisplacementFn,
    eps: float = _EPS,
) -> jax.Array:
    """LJ + Coulomb with exclusions and the exception term.

    Uses the dense ``(N, N)`` path when ``nb.pairs is None`` (default) and the
    pair-list path otherwise. Both produce identical energies and forces.
    """
    if nb.pairs is None:
        return _nonbonded_dense(R, nb, displacement_fn, eps=eps)
    return _nonbonded_pairs(R, nb, displacement_fn, eps=eps)


# ---------------------------------------------------------------------------
# Total energy
# ---------------------------------------------------------------------------


def bonded_energy(R: jax.Array, bonded: BondedSet, eps: float = _EPS) -> jax.Array:
    """Sum of bond, angle, and torsion energies."""
    return (
        bond_energy(R, bonded.bonds, bonded.k_r, bonded.r0, eps=eps)
        + angle_energy(R, bonded.angles, bonded.k_theta, bonded.theta0, eps=eps)
        + torsion_energy(
            R, bonded.torsions, bonded.periodicity, bonded.torsion_k, bonded.phase, eps=eps
        )
    )


def total_energy_fn(
    displacement_fn: DisplacementFn,
    bonded: BondedSet,
    nonbonded: NonbondedSet,
    eps: float = _EPS,
) -> Callable[[jax.Array], jax.Array]:
    """Return a pure ``E(R)`` summing bonded and nonbonded energy (grad/jit-friendly).

    ``eps`` (the numerical softening floor) is threaded uniformly into every term.
    """

    def energy(R: jax.Array) -> jax.Array:
        return bonded_energy(R, bonded, eps=eps) + nonbonded_energy(
            R, nonbonded, displacement_fn, eps=eps
        )

    return energy
