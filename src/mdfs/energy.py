from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp


def bond_energy(
    R: jax.Array,
    bonds: jax.Array,  # (Nb, 2) atom indices
    k_r: jax.Array,  # (Nb,) force constants
    r0: jax.Array,  # (Nb,) equilibrium lengths
) -> jax.Array:
    """Harmonic bond energy: 0.5 * k_r * (r - r0)^2"""
    i, j = bonds[:, 0], bonds[:, 1]
    rij = R[j] - R[i]
    r = jnp.linalg.norm(rij, axis=1)
    return 0.5 * jnp.sum(k_r * (r - r0) ** 2)


def angle_energy(
    R: jax.Array,
    angles: jax.Array,  # (Na, 3)
    k_theta: jax.Array,  # (Na,)
    theta0: jax.Array,  # (Na,) in radians
) -> jax.Array:
    """Harmonic angle: 0.5 * k_theta * (theta - theta0)^2"""
    i, j, k = angles[:, 0], angles[:, 1], angles[:, 2]
    v1 = R[i] - R[j]
    v2 = R[k] - R[j]
    v1 = v1 / jnp.linalg.norm(v1, axis=1, keepdims=True)
    v2 = v2 / jnp.linalg.norm(v2, axis=1, keepdims=True)
    cos_th = jnp.clip(jnp.sum(v1 * v2, axis=1), -1.0, 1.0)
    th = jnp.arccos(cos_th)
    return 0.5 * jnp.sum(k_theta * (th - theta0) ** 2)


def _dihedral_phi(R: jax.Array, dihs: jax.Array) -> jax.Array:
    """Return dihedral angles φ for each (i,j,k,l)."""
    i, j, k, l = [dihs[:, t] for t in range(4)]
    b1 = R[j] - R[i]
    b2 = R[k] - R[j]
    b3 = R[l] - R[k]
    # normals
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    # normalize
    n1 = n1 / jnp.linalg.norm(n1, axis=1, keepdims=True)
    n2 = n2 / jnp.linalg.norm(n2, axis=1, keepdims=True)
    # m1 = n1 x (b2/|b2|)
    b2_hat = b2 / jnp.linalg.norm(b2, axis=1, keepdims=True)
    m1 = jnp.cross(n1, b2_hat)
    x = jnp.sum(n1 * n2, axis=1)
    y = jnp.sum(m1 * n2, axis=1)
    return jnp.arctan2(y, x)


def torsion_energy_fourier(
    R: jax.Array,
    dihs: jax.Array,  # (Nd, 4) proper torsions
    n: jax.Array,  # (Nd, T_max) integer multiplicities n_m
    k_n: jax.Array,  # (Nd, T_max) Fourier coefficients (amplitudes)
    delta: jax.Array,  # (Nd, T_max) phase offsets (radians)
    active_mask: jax.Array | None = None,  # (Nd, T_max) booleans for used terms
) -> jax.Array:
    """Proper torsion as OPLS/AMBER-style Fourier series."""
    phi = _dihedral_phi(R, dihs)  # (Nd,)
    phi_terms = phi[:, None]
    contrib = k_n * (1.0 + jnp.cos(n * phi_terms - delta))
    if active_mask is not None:
        contrib = jnp.where(active_mask, contrib, 0.0)
    return jnp.sum(jnp.sum(contrib, axis=1))


class LJMixParams(NamedTuple):
    eps_type: jax.Array  # (Nt,) epsilon per atom type
    sig_type: jax.Array  # (Nt,) sigma per atom type


def lj_mixed_eps_sigma(
    types_i: jax.Array, types_j: jax.Array, params: LJMixParams
) -> tuple[jax.Array, jax.Array]:
    """
    Lorentz–Berthelot:
      sigma_ij = (sigma_i + sigma_j)/2,
      epsilon_ij = sqrt(eps_i * eps_j).
    Widely used; e.g., OpenMM NonbondedForce default.  # noqa
    """
    eps_i = params.eps_type[types_i]
    eps_j = params.eps_type[types_j]
    sig_i = params.sig_type[types_i]
    sig_j = params.sig_type[types_j]
    eps_ij = jnp.sqrt(eps_i * eps_j)
    sig_ij = 0.5 * (sig_i + sig_j)
    return eps_ij, sig_ij


def lj_12_6(eps_ij: jax.Array, sig_ij: jax.Array, r: jax.Array) -> jax.Array:
    """Lennard–Jones 12-6: 4ε[(σ/r)^12 - (σ/r)^6]."""
    inv_r = 1.0 / (r + 1e-12)
    sr6 = (sig_ij * inv_r) ** 6
    return 4.0 * eps_ij * (sr6 * sr6 - sr6)


def lennard_jones_energy(
    R: jax.Array,
    pairs: jax.Array,  # (Np, 2) i<j neighbor pairs
    types: jax.Array,  # (N,) atom types (ints)
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
    lj_params: LJMixParams,
    r_cut: float,
    e_shift: bool = True,  # shift energy to 0 at cutoff
) -> jax.Array:
    """Compute LJ energy over neighbor pairs with LB mixing and cutoff."""
    i, j = pairs[:, 0], pairs[:, 1]
    dR = displacement_fn(R[i], R[j])
    r = jnp.linalg.norm(dR, axis=1)
    eps_ij, sig_ij = lj_mixed_eps_sigma(types[i], types[j], lj_params)
    # mask within cutoff
    mask = r < r_cut
    e = lj_12_6(eps_ij, sig_ij, r)
    if e_shift:
        # Shift by value at cutoff for continuity of energy (not force).
        e_c = lj_12_6(eps_ij, sig_ij, jnp.full_like(r, r_cut))
        e = e - e_c
    return jnp.sum(jnp.where(mask, e, 0.0))


class DSFParams(NamedTuple):
    alpha: float  # damping parameter (1/length)
    r_cut: float  # Coulomb cutoff
    k_e: float = 1.0  # Coulomb prefactor; set to 1 in reduced units (or 1/4πϵ0)


def coulomb_dsf_pair(
    qiqj: jax.Array, r: jax.Array, alpha: float, r_cut: float, k_e: float
) -> jax.Array:
    """Damped Shifted-Force (DSF) Coulomb."""
    # Avoid division by zero for r==0 in same spirit as LJ
    inv_r = 1.0 / (r + 1e-12)
    erfc_ar = jax.scipy.special.erfc(alpha * r)
    # constants at cutoff
    erfc_arc = jax.scipy.special.erfc(alpha * r_cut)
    term_c = (
        erfc_arc / (r_cut**2)
        + (2.0 * alpha / jnp.sqrt(jnp.pi)) * jnp.exp(-(alpha**2) * (r_cut**2)) / r_cut
    )
    e = qiqj * (erfc_ar * inv_r - erfc_arc / r_cut + term_c * (r - r_cut))
    return k_e * e


def coulomb_dsf_energy(
    R: jax.Array,
    pairs: jax.Array,  # (Np, 2) i<j
    q: jax.Array,  # (N,) partial charges
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
    dsf: DSFParams,
) -> jax.Array:
    """Compute DSF Coulomb energy over neighbor pairs (r<r_cut)."""
    i, j = pairs[:, 0], pairs[:, 1]
    dR = displacement_fn(R[i], R[j])
    r = jnp.linalg.norm(dR, axis=1)
    mask = r < dsf.r_cut
    qiqj = q[i] * q[j]
    e = coulomb_dsf_pair(qiqj, r, dsf.alpha, dsf.r_cut, dsf.k_e)
    return jnp.sum(jnp.where(mask, e, 0.0))


# ---------------------------------------------------------------------------
# 1–4 scaling & exclusions helper
# ---------------------------------------------------------------------------


def apply_one_four_and_exclusions(
    pairs: jax.Array,  # (Np,2)
    scale14_vdw: jax.Array,  # (N,N) or (Np,) scaling for LJ on 1-4 pairs; others 1.0
    scale14_elec: jax.Array,  # (N,N) or (Np,) scaling for Coul on 1-4 pairs; others 1.0
    exclude_mask: jax.Array,  # (N,N) or (Np,) booleans for 1-2 and 1-3 exclusions
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Return per-pair scaling for LJ and Coulomb, and an exclusion mask.
    If provided as (N,N), we gather to (Np,) here.
    """
    if scale14_vdw.ndim == 2:
        s_vdw = scale14_vdw[pairs[:, 0], pairs[:, 1]]
        s_e = scale14_elec[pairs[:, 0], pairs[:, 1]]
        ex = exclude_mask[pairs[:, 0], pairs[:, 1]]
    else:
        s_vdw, s_e, ex = scale14_vdw, scale14_elec, exclude_mask
    return s_vdw, s_e, ex


# ---------------------------------------------------------------------------
# Combined nonbonded with scaling & exclusions.
# ---------------------------------------------------------------------------


def nonbonded_energy(
    R: jax.Array,
    pairs: jax.Array,  # (Np,2) i<j
    types: jax.Array,  # (N,)
    q: jax.Array,  # (N,)
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
    lj_params: LJMixParams,
    r_cut_lj: float,
    dsf: DSFParams,
    scale14_vdw: jax.Array | None = None,  # (N,N) or (Np,)
    scale14_elec: jax.Array | None = None,  # (N,N) or (Np,)
    exclude_mask: jax.Array | None = None,  # (N,N) or (Np,) True to exclude (1-2,1-3)
    shift_lj: bool = True,
) -> jax.Array:
    """
    Compute LJ + Coulomb(DSF) over neighbor pairs with:
      - Lorentz–Berthelot mixing for LJ,
      - optional 1–4 scaling (vdW & electrostatics),
      - exclusions for 1–2 and 1–3 pairs,
      - distinct LJ and Coulomb cutoffs (DSF uses dsf.r_cut).
    """
    i, j = pairs[:, 0], pairs[:, 1]
    dR = displacement_fn(R[i], R[j])
    r = jnp.linalg.norm(dR, axis=1)
    # default scaling/exclusions
    if scale14_vdw is None:
        scale14_vdw = jnp.ones_like(r)
    if scale14_elec is None:
        scale14_elec = jnp.ones_like(r)
    if exclude_mask is None:
        exclude_mask = jnp.zeros_like(r, dtype=bool)

    s_vdw, s_elec, ex = apply_one_four_and_exclusions(
        pairs, scale14_vdw, scale14_elec, exclude_mask
    )

    # LJ
    eps_ij, sig_ij = lj_mixed_eps_sigma(types[i], types[j], lj_params)
    lj_mask = (r < r_cut_lj) & (~ex)
    e_lj = lj_12_6(eps_ij, sig_ij, r)
    if shift_lj:
        e_c = lj_12_6(eps_ij, sig_ij, jnp.full_like(r, r_cut_lj))
        e_lj = e_lj - e_c
    e_lj = jnp.where(lj_mask, s_vdw * e_lj, 0.0)

    # Coulomb (DSF)
    coul_mask = (r < dsf.r_cut) & (~ex)
    qiqj = q[i] * q[j]
    e_coul = coulomb_dsf_pair(qiqj, r, dsf.alpha, dsf.r_cut, dsf.k_e)
    e_coul = jnp.where(coul_mask, s_elec * e_coul, 0.0)

    return jnp.sum(e_lj + e_coul)


# ---------------------------------------------------------------------------
# Full energy builder (sum of bonded + nonbonded).
# ---------------------------------------------------------------------------


class BondedSet(NamedTuple):
    bonds: jax.Array
    k_r: jax.Array
    r0: jax.Array
    angles: jax.Array
    k_theta: jax.Array
    theta0: jax.Array
    dihs: jax.Array
    n: jax.Array
    k_n: jax.Array
    delta: jax.Array
    active_mask: jax.Array | None = None


class NonbondedSet(NamedTuple):
    pairs: jax.Array
    types: jax.Array
    q: jax.Array
    lj_params: LJMixParams
    r_cut_lj: float
    dsf: DSFParams
    scale14_vdw: jax.Array | None = None
    scale14_elec: jax.Array | None = None
    exclude_mask: jax.Array | None = None
    shift_lj: bool = True


def total_energy_fn(
    displacement_fn: Callable[[jax.Array, jax.Array], jax.Array],
    bonded: BondedSet,
    nonbonded: NonbondedSet,
) -> Callable[[jax.Array], jax.Array]:
    """
    Return E(R) that sums bonded (bond, angle, torsion) and nonbonded (LJ + DSF Coulomb).
    Mirrors jax-md's pattern of returning a pure energy function usable with grad/JIT.  # noqa
    """

    def E(R: jax.Array) -> jax.Array:
        Eb = bond_energy(R, bonded.bonds, bonded.k_r, bonded.r0)
        Ea = angle_energy(R, bonded.angles, bonded.k_theta, bonded.theta0)
        Ed = torsion_energy_fourier(
            R, bonded.dihs, bonded.n, bonded.k_n, bonded.delta, bonded.active_mask
        )
        Enb = nonbonded_energy(
            R,
            nonbonded.pairs,
            nonbonded.types,
            nonbonded.q,
            displacement_fn,
            nonbonded.lj_params,
            nonbonded.r_cut_lj,
            nonbonded.dsf,
            nonbonded.scale14_vdw,
            nonbonded.scale14_elec,
            nonbonded.exclude_mask,
            nonbonded.shift_lj,
        )
        return Eb + Ea + Ed + Enb

    return E
