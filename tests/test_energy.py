"""Unit tests for the differentiable energy terms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mdfs
from mdfs import energy
from mdfs.constants import ONE_4PI_EPS0


def test_bond_energy_analytic():
    R = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.12]])
    bonds = jnp.array([[0, 1]])
    k = jnp.array([1000.0])
    r0 = jnp.array([0.1])
    # 0.5 * 1000 * (0.12 - 0.10)^2 = 0.2
    assert float(energy.bond_energy(R, bonds, k, r0)) == pytest.approx(0.2, rel=1e-6)


def test_angle_energy_right_angle():
    R = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    angles = jnp.array([[0, 1, 2]])
    k = jnp.array([10.0])
    theta0 = jnp.array([np.pi / 2])  # exactly at equilibrium -> 0
    assert float(energy.angle_energy(R, angles, k, theta0)) == pytest.approx(0.0, abs=1e-9)
    # displaced equilibrium contributes 0.5 k (pi/2)^2 when theta0 = 0
    e = float(energy.angle_energy(R, angles, k, jnp.array([0.0])))
    assert e == pytest.approx(0.5 * 10.0 * (np.pi / 2) ** 2, rel=1e-6)


# Planar trans arrangement (atom 3 on the opposite side of the j-k axis) -> phi = pi.
_TRANS = [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]


def test_dihedral_phi_known():
    R = jnp.array(_TRANS)
    phi = float(energy.dihedral_phi(R, jnp.array([[0, 1, 2, 3]]))[0])
    assert abs(abs(phi) - np.pi) < 1e-6


def test_torsion_energy_analytic():
    R = jnp.array(_TRANS)
    dih = jnp.array([[0, 1, 2, 3]])
    # phi = pi; k(1+cos(1*pi - 0)) = k(1 + (-1)) = 0
    e = float(energy.torsion_energy(R, dih, jnp.array([1.0]), jnp.array([5.0]), jnp.array([0.0])))
    assert e == pytest.approx(0.0, abs=1e-6)
    # n=2: k(1+cos(2pi)) = 2k
    e2 = float(energy.torsion_energy(R, dih, jnp.array([2.0]), jnp.array([5.0]), jnp.array([0.0])))
    assert e2 == pytest.approx(10.0, rel=1e-6)


def test_lj_minimum_at_2_pow_1_6_sigma():
    sigma, eps = 0.3, 1.5
    rmin = 2.0 ** (1.0 / 6.0) * sigma
    val = float(energy.lj_12_6(jnp.array(eps), jnp.array(sigma), jnp.array(rmin)))
    assert val == pytest.approx(-eps, rel=1e-6)


def test_coulomb_plain():
    q = 0.5
    r = 0.4
    val = float(energy.coulomb_plain(jnp.array(q), jnp.array(r), ONE_4PI_EPS0))
    assert val == pytest.approx(ONE_4PI_EPS0 * q / r, rel=1e-6)


def test_bond_force_equals_negative_grad():
    R = jnp.array([[0.0, 0.0, 0.0], [0.13, 0.02, -0.01]])
    bonds = jnp.array([[0, 1]])
    k, r0 = jnp.array([500.0]), jnp.array([0.1])

    def e(x):
        return energy.bond_energy(x, bonds, k, r0)

    g = np.array(jax.grad(e)(R))
    # central finite difference
    eps = 1e-5
    fd = np.zeros_like(g)
    Rn = np.array(R)
    for a in range(2):
        for d in range(3):
            p = Rn.copy()
            p[a, d] += eps
            m = Rn.copy()
            m[a, d] -= eps
            fd[a, d] = (float(e(jnp.array(p))) - float(e(jnp.array(m)))) / (2 * eps)
    assert np.max(np.abs(g - fd)) < 1e-4


def test_angle_gradient_finite_near_collinear():
    # Nearly collinear: tiny perturbation off a straight line.
    R = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 1e-7, 0.0]])
    angles = jnp.array([[0, 1, 2]])
    k, theta0 = jnp.array([100.0]), jnp.array([np.pi])

    def e(x):
        return energy.angle_energy(x, angles, k, theta0)

    g = np.array(jax.grad(e)(R))
    assert np.all(np.isfinite(g))


def test_dihedral_gradient_finite():
    R = jnp.array([[1.0, 1.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, -0.1]])
    dih = jnp.array([[0, 1, 2, 3]])

    def e(x):
        return energy.torsion_energy(x, dih, jnp.array([2.0]), jnp.array([3.0]), jnp.array([0.0]))

    g = np.array(jax.grad(e)(R))
    assert np.all(np.isfinite(g))


def test_exception_energy_matches_manual():
    R = jnp.array([[0.0, 0.0, 0.0], [0.35, 0.0, 0.0]])
    disp, _ = mdfs.free()
    nb = energy.NonbondedSet(
        pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        types=jnp.arange(2),
        q=jnp.array([0.0, 0.0]),
        lj_params=energy.LJMixParams(jnp.zeros(2), jnp.ones(2)),
        exclude_mask=jnp.zeros((2, 2), dtype=bool),
        exc_pairs=jnp.array([[0, 1]]),
        exc_qq=jnp.array([0.2]),
        exc_sigma=jnp.array([0.3]),
        exc_eps=jnp.array([1.0]),
    )
    r = 0.35
    expected = (
        float(energy.lj_12_6(jnp.array(1.0), jnp.array(0.3), jnp.array(r))) + ONE_4PI_EPS0 * 0.2 / r
    )
    assert float(energy.exception_energy(R, nb, disp)) == pytest.approx(expected, rel=1e-6)


def test_dsf_value_and_force_zero_at_cutoff():
    rc, alpha, qiqj = 1.0, 2.0, 0.5
    e_at_rc = float(
        energy.coulomb_dsf_pair(jnp.array(qiqj), jnp.array(rc), alpha, rc, ONE_4PI_EPS0)
    )
    assert abs(e_at_rc) < 1e-9
    dedr = float(
        jax.grad(lambda r: energy.coulomb_dsf_pair(jnp.array(qiqj), r, alpha, rc, ONE_4PI_EPS0))(
            jnp.array(rc)
        )
    )
    assert abs(dedr) < 1e-6  # shifted-force: derivative also vanishes at the cutoff


def _two_atom_cutoff_nb(r_cut_lj):
    return energy.NonbondedSet(
        pairs=jnp.array([[0, 1]]),
        types=jnp.arange(2),
        q=jnp.array([0.5, -0.5]),
        lj_params=energy.LJMixParams(jnp.array([1.0, 1.0]), jnp.array([0.3, 0.3])),
        exclude_mask=jnp.zeros((2, 2), dtype=bool),
        exc_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exc_qq=jnp.zeros(0),
        exc_sigma=jnp.zeros(0),
        exc_eps=jnp.zeros(0),
        r_cut_lj=r_cut_lj,
        dsf=energy.DSFParams(alpha=1.0, r_cut=r_cut_lj),
    )


def test_nonbonded_cutoff_excludes_far_pairs():
    disp, _ = mdfs.free()
    nb = _two_atom_cutoff_nb(0.5)
    far = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # beyond cutoff -> 0
    assert abs(float(energy.nonbonded_energy(far, nb, disp))) < 1e-9
    near = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.35]])  # within cutoff -> nonzero
    assert abs(float(energy.nonbonded_energy(near, nb, disp))) > 1e-6
